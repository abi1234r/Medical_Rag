"""
routes.py — Flask Blueprint
Endpoints:
  GET  /          → chat UI
  POST /api/chat  → RAG query
  GET  /api/stats → index statistics
  POST /api/upload → upload new dataset
  POST /api/rebuild → force re-index
"""

from __future__ import annotations

import os
from pathlib import Path
from flask import Blueprint, request, jsonify, render_template, current_app

bp = Blueprint("main", __name__)

# Lazy-load the RAG engine so Flask starts instantly
_rag = None

def get_rag():
    global _rag
    if _rag is None:
        from app.rag_engine import MedicalRAG
        _rag = MedicalRAG()
    return _rag


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    top_k    = int(data.get("top_k", 5))

    if not question:
        return jsonify({"error": "question is required"}), 400

    # Medical safety disclaimer
    disclaimer = (
        "⚠️ This information is for educational purposes only and does not "
        "constitute medical advice. Always consult a qualified healthcare professional."
    )

    try:
        result = get_rag().query(question, top_k=top_k)
        result["disclaimer"] = disclaimer
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@bp.route("/api/stats")
def stats():
    try:
        return jsonify(get_rag().stats)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@bp.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No file selected"}), 400

    allowed = {".csv", ".txt"}
    ext = Path(f.filename).suffix.lower()
    if ext not in allowed:
        return jsonify({"error": f"Only {allowed} files are accepted"}), 400

    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    save_path = data_dir / f.filename
    f.save(save_path)

    # Rebuild index with new file
    try:
        get_rag().rebuild_index()
        return jsonify({"message": f"{f.filename} uploaded and indexed successfully.",
                        "chunks": get_rag().stats["total_chunks"]})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@bp.route("/api/rebuild", methods=["POST"])
def rebuild():
    try:
        get_rag().rebuild_index()
        return jsonify({"message": "Index rebuilt.", "stats": get_rag().stats})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
