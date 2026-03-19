"""
rag_engine.py
─────────────
Lightweight RAG pipeline — NO heavy LLM, NO API key needed.

Strategy:
  1. Load & chunk the disease-symptom dataset (CSV or plain text)
  2. Embed chunks with HuggingFace Sentence-Transformers (~90MB, one-time download)
  3. Build / persist a FAISS index
  4. At query time: embed query -> FAISS search -> extract & format answer from top chunks
     (smart keyword extraction replaces a generative LLM -- instant, offline, accurate)
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# -- Paths -------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
INDEX_PATH = BASE_DIR / "models" / "faiss.index"
META_PATH  = BASE_DIR / "models" / "metadata.pkl"

# -- Config ------------------------------------------------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # ~90MB only, no API key
TOP_K       = 5
CHUNK_SIZE  = 300

# -- Intent -> CSV column mapping --------------------------------------------
INTENT_MAP = {
    "symptom":     ["symptoms", "symptom_list", "symptom"],
    "treatment":   ["treatment", "treatments", "therapy"],
    "cause":       ["cause", "causes", "reason"],
    "prevention":  ["prevention", "prevent", "precautions"],
    "medication":  ["medication", "medications", "medicine", "drug"],
    "diet":        ["diet", "food", "nutrition"],
    "description": ["description", "about", "overview"],
}

INTENT_KEYWORDS = {
    "symptom":     ["symptom", "symptoms", "sign", "signs", "feel", "indicate"],
    "treatment":   ["treat", "treatment", "cure", "therapy", "manage", "remedy"],
    "cause":       ["cause", "causes", "reason", "why", "origin", "caused"],
    "prevention":  ["prevent", "prevention", "avoid", "precaution", "reduce risk"],
    "medication":  ["medicine", "medication", "drug", "tablet", "pill", "prescription"],
    "diet":        ["diet", "food", "eat", "nutrition", "meal", "consume"],
    "description": ["what is", "explain", "describe", "about", "overview", "definition"],
}


class MedicalRAG:
    """Lightweight Retrieval-Augmented Generation -- no LLM required."""

    def __init__(self):
        print("[RAG] Loading embedding model (all-MiniLM-L6-v2 ~90MB) ...")
        self.embedder  = SentenceTransformer(EMBED_MODEL)
        self.index     = None
        self.documents : List[str] = []
        self.metadatas : List[dict] = []
        self.raw_rows  : List[dict] = []
        self._load_or_build_index()
        print("[RAG] Ready! (no LLM needed -- answers extracted directly from dataset)")

    # -- Index management ----------------------------------------------------

    def _load_or_build_index(self):
        if INDEX_PATH.exists() and META_PATH.exists():
            print("[RAG] Loading existing FAISS index ...")
            self.index = faiss.read_index(str(INDEX_PATH))
            with open(META_PATH, "rb") as f:
                data = pickle.load(f)
            self.documents = data["documents"]
            self.metadatas = data["metadatas"]
            self.raw_rows  = data.get("raw_rows", [])
        else:
            print("[RAG] Building FAISS index from data ...")
            self._build_index()

    def _build_index(self):
        docs, metas, rows = self._load_data()
        if not docs:
            raise RuntimeError("No documents found in data/. Add a CSV or .txt file.")

        self.documents = docs
        self.metadatas = metas
        self.raw_rows  = rows

        print(f"[RAG] Embedding {len(docs)} chunks ...")
        embeddings = self.embedder.encode(
            docs, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True
        )

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_PATH))
        with open(META_PATH, "wb") as f:
            pickle.dump({"documents": docs, "metadatas": metas, "raw_rows": rows}, f)
        print("[RAG] Index saved!")

    # -- Data loading --------------------------------------------------------

    def _load_data(self) -> Tuple[List[str], List[dict], List[dict]]:
        docs, metas, rows = [], [], []

        for csv_path in DATA_DIR.glob("*.csv"):
            print(f"[RAG] Loading CSV: {csv_path.name}")
            df = pd.read_csv(csv_path)
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            for _, row in df.iterrows():
                text = self._row_to_text(row)
                if text.strip():
                    docs.append(text)
                    metas.append({"source": csv_path.name, "type": "csv"})
                    rows.append(row.to_dict())

        for txt_path in DATA_DIR.glob("*.txt"):
            print(f"[RAG] Loading text: {txt_path.name}")
            content = txt_path.read_text(errors="ignore")
            for chunk in self._chunk_text(content):
                docs.append(chunk)
                metas.append({"source": txt_path.name, "type": "text"})
                rows.append({"_text": chunk})

        return docs, metas, rows

    @staticmethod
    def _row_to_text(row: pd.Series) -> str:
        priority = [
            "disease", "condition", "disorder",
            "symptoms", "symptom_list", "symptom",
            "description", "treatment", "cause", "prevention",
            "precautions", "diet", "medication", "workout",
        ]
        parts, used = [], set()
        for key in priority:
            for col in row.index:
                if key in col and col not in used and pd.notna(row[col]):
                    parts.append(f"{col.replace('_',' ').title()}: {str(row[col]).strip()}")
                    used.add(col)
        for col in row.index:
            if col not in used and pd.notna(row[col]):
                parts.append(f"{col.replace('_',' ').title()}: {str(row[col]).strip()}")
        return " | ".join(parts)

    @staticmethod
    def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks, start = [], 0
        while start < len(words):
            chunks.append(" ".join(words[start:min(start + size, len(words))]))
            start += size - overlap
        return [c for c in chunks if len(c) > 30]

    # -- Intent detection ----------------------------------------------------

    @staticmethod
    def _detect_intent(question: str) -> str:
        q = question.lower()
        for intent, keywords in INTENT_KEYWORDS.items():
            if any(kw in q for kw in keywords):
                return intent
        return "description"

    # -- Smart answer builder ------------------------------------------------

    def _build_answer(self, question: str, retrieved: List[dict]) -> str:
        if not retrieved:
            return "No relevant information found in the dataset for your question."

        intent  = self._detect_intent(question)
        best    = retrieved[0]
        row     = self.raw_rows[best["idx"]] if best["idx"] < len(self.raw_rows) else {}

        # Find column matching intent
        target_cols = INTENT_MAP.get(intent, [])
        answer_val  = None
        for target in target_cols:
            for col, val in row.items():
                if target in col and val and str(val).strip() not in ("nan", ""):
                    answer_val = str(val).strip()
                    break
            if answer_val:
                break

        # Get disease name
        disease_name = ""
        for col, val in row.items():
            if any(k in col for k in ["disease", "condition", "disorder"]):
                disease_name = str(val).strip()
                break

        # Compose answer
        if answer_val and disease_name:
            label  = intent.replace("_", " ").title()
            answer = f"Based on the dataset, here is the information about **{disease_name}**:\n\n"
            answer += f"**{label}:** {answer_val}"
            if intent != "description":
                for col, val in row.items():
                    if "description" in col and val and str(val).strip() not in ("nan", ""):
                        answer += f"\n\n**Overview:** {str(val).strip()}"
                        break

        elif disease_name:
            answer = f"Information found for **{disease_name}**:\n\n"
            for col, val in row.items():
                if val and str(val).strip() not in ("nan", "") and not any(k in col for k in ["disease","condition"]):
                    answer += f"**{col.replace('_',' ').title()}:** {str(val).strip()}\n\n"
        else:
            answer = best["text"][:600]
            if len(best["text"]) > 600:
                answer += "..."

        return answer.strip()

    # -- Query ---------------------------------------------------------------

    def query(self, question: str, top_k: int = TOP_K) -> dict:
        if self.index is None or self.index.ntotal == 0:
            return {"answer": "Index is empty. Please add data files to the data/ folder.", "sources": []}

        q_vec = self.embedder.encode([question], normalize_embeddings=True, convert_to_numpy=True)
        scores, indices = self.index.search(q_vec, top_k)

        retrieved = [
            {
                "idx":   int(indices[0][j]),
                "text":  self.documents[indices[0][j]],
                "score": float(scores[0][j]),
                "meta":  self.metadatas[indices[0][j]],
            }
            for j in range(len(indices[0]))
            if indices[0][j] != -1
        ]

        answer = self._build_answer(question, retrieved)

        return {
            "answer": answer,
            "sources": [
                {
                    "text":   r["text"][:300] + ("..." if len(r["text"]) > 300 else ""),
                    "score":  round(r["score"] * 100, 1),
                    "source": r["meta"]["source"],
                }
                for r in retrieved
            ],
        }

    def rebuild_index(self):
        if INDEX_PATH.exists(): INDEX_PATH.unlink()
        if META_PATH.exists():  META_PATH.unlink()
        self._build_index()

    @property
    def stats(self) -> dict:
        return {
            "total_chunks": len(self.documents),
            "embed_model":  EMBED_MODEL,
            "llm_model":    "None (retrieval-based answering, no LLM)",
            "index_type":   "FAISS IndexFlatIP (cosine)",
        }
