# 📘 Short Report: France RAG QA Assistant

## 1. 🧱 System Design

| Component | Description |
|----------|-------------|
| Chunking | Word-overlap based chunking (100 words with 20 overlap) for context expansion |
| Embedding | MiniLM-L6-v2 embeddings using SentenceTransformers |
| Vector Store | FAISS (cosine similarity with L2 normalization) |
| Retriever | Top-k search with optional metadata filtering |
| Generator | Together AI LLM using custom constrained prompts |
| UI | Streamlit app with sliders, model choice, feedback, and visualization |

## 2. 🔧 Justification

- **MiniLM** is chosen over `m2-BERT` due to smaller size + faster speed for fast indexing with good accuracy.
- **FAISS FlatIP + normalization** provides cosine similarity and fast runtime.
- **Prompt design** ensures no hallucination: model can say "I don't know" and avoids fabrications.

## 3. 💬 Sample Input/Output

**Question:** "What are the mountain ranges in France?"  
**Chunks Retrieved:** France - Alps, France - Massif Central  
**Answer:** "The major mountain ranges in France are the Alps, Pyrenees, and Massif Central."

---

## 4. 🧹 Preprocessing Scripts

- `scraper.py`: downloads all 8 official Britannica subsections.
- `embed.py`: chunks, cleans, and embeds the corpus.
- `user_additions.pkl`: stores user-added data permanently.

---

## 5. ✅ Evaluation Preparation

- Prompt constraints ensure faithfulness.
- UI allows top-k tuning (precision/recall).
- Hybrid filtering using metadata categories.
- Feedback section encourages rating.

---

© 2025 | RAG Final Project | UI - Iran
