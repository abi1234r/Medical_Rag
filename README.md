# ⚕️ MedRAG — Medical AI Chatbot

An AI-powered healthcare assistant using **Retrieval-Augmented Generation (RAG)** to provide
context-aware medical answers from specialized datasets.

## 🏗️ Architecture

```
User Query
    │
    ▼
[Sentence-Transformer]  ← all-MiniLM-L6-v2
    │  (384-dim embedding)
    ▼
[FAISS IndexFlatIP]     ← cosine similarity search
    │  (top-K chunks)
    ▼
[Prompt Builder]        ← context + question
    │
    ▼
[flan-t5-base LLM]      ← HuggingFace (local, no API key)
    │
    ▼
[Flask API]  →  [Chat UI]
```

## 📁 Project Structure

```
medical-rag/
├── app.py                  # Flask entry point
├── requirements.txt
├── data/
│   └── disease_symptoms.csv   # Sample dataset (add your own CSVs here)
├── models/                 # Auto-generated FAISS index (after first run)
│   ├── faiss.index
│   └── metadata.pkl
├── app/
│   ├── __init__.py
│   ├── rag_engine.py       # Core RAG pipeline
│   └── routes.py           # Flask API routes
├── templates/
│   └── index.html          # Chat UI
└── static/
    ├── css/style.css
    └── js/app.js
```

## 🚀 Setup & Run

### 1. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** First run downloads ~500MB of model weights (MiniLM + flan-t5). This is cached locally.

### 3. Add your dataset
Place your CSV file(s) in the `data/` folder. Recommended columns:

| Column | Description |
|--------|-------------|
| `Disease` | Disease or condition name |
| `Symptoms` | Comma-separated symptom list |
| `Description` | Detailed description |
| `Treatment` | Treatment methods |
| `Cause` | Root causes |
| `Prevention` | Preventive measures |

The engine auto-detects and prioritises these column names. Any CSV structure works.

### 4. Start the server
```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## 🔌 API Reference

### `POST /api/chat`
Ask a question against the RAG pipeline.

**Request:**
```json
{
  "question": "What are the symptoms of diabetes?",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "Symptoms of diabetes include increased thirst, frequent urination...",
  "sources": [
    {
      "text": "Disease: Diabetes Type 2 | Symptoms: Increased thirst...",
      "score": 87.3,
      "source": "disease_symptoms.csv"
    }
  ],
  "disclaimer": "⚠️ This information is for educational purposes only..."
}
```

### `GET /api/stats`
Returns index statistics (chunk count, model names).

### `POST /api/upload`
Upload a new CSV or TXT dataset file (multipart form).

### `POST /api/rebuild`
Force a full re-index of all files in `data/`.

---

## ⚙️ Configuration

Edit the constants at the top of `app/rag_engine.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `LLM_MODEL` | `google/flan-t5-base` | Local LLM for generation |
| `TOP_K` | `5` | Default number of retrieved chunks |
| `MAX_NEW_TOKENS` | `256` | Max tokens in LLM output |
| `CHUNK_SIZE` | `300` | Words per chunk (for plain text) |

### Use a larger / better LLM
Change `LLM_MODEL` to any HuggingFace seq2seq or causal LM:
```python
LLM_MODEL = "google/flan-t5-large"    # More capable, ~3GB
LLM_MODEL = "mistralai/Mistral-7B-v0.1"  # Much stronger, needs GPU
```

---

## 🛡️ Disclaimer

This system is for **educational and research purposes only**.
It does **not** constitute medical advice. Always consult a qualified
healthcare professional for personal medical decisions.

---

## 📦 Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | `google/flan-t5-base` (HuggingFace) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Search | FAISS (IndexFlatIP — cosine) |
| Backend | Flask 3.x |
| Frontend | Vanilla JS + CSS |
| Data | Pandas CSV parser |
