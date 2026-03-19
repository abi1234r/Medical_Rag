MedRAG is a **Retrieval-Augmented Generation (RAG)** pipeline built for medical Q&A. It takes structured disease-symptom CSV datasets, converts them into dense vector embeddings using HuggingFace Sentence Transformers, indexes them with FAISS for lightning-fast semantic search, and serves answers through a clean Flask web interface.

Unlike LLM-based chatbots, MedRAG answers are **grounded entirely in your dataset** — no hallucinations, no API keys, no internet connection required after setup.
