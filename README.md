[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/M-Amin-Kiani/RichKids_ChatRoom)

# 🇫🇷 RAG Assistant

An end-to-end Retrieval-Augmented Generation (RAG) system for answering questions about France’s geography using web-scraped content and LLMs.

## 🚀 Features

- 🧠 Retrieval with FAISS over Britannica Land section
- 🔗 Generation via TogetherAI LLMs (Mistral, DeepSeek)
- 🎛 Customizable temperature, top-k, max tokens
- 📄 View and analyze retrieved chunks with similarity scores
- 📊 Visual bar chart for chunk comparison
- 💬 User feedback buttons & prompt inspection
- 🎨 Modern Streamlit UI with responsive design

## 🔧 Technologies
- **FastAPI** for backend API (`/retrieve` and `/generate`)
- **FAISS** for fast vector similarity search
- **Sentence-Transformers** for embedding generation
- **Streamlit** as a responsive, interactive UI
- **Together AI** for LLM-based answer generation

## 🧪 Endpoints

### `/retrieve`
Returns top-k relevant chunks using cosine similarity over embeddings.

### `/generate`
Uses Together AI to generate final answer based on top-k chunks.

---

## 📝 Prompt Design
```text
"You are a factual assistant. Use ONLY the given context. If uncertain, say you don't know."
```

## 📦 Embedding Model
- `sentence-transformers/all-MiniLM-L6-v2` (compact, accurate, fast for FAISS)

---

## 📥 Adding Data
Users can add new URLs or custom text. These are chunked, embedded, and persisted in `user_additions.pkl`.

---

## 🧱 Architecture

```bash
User Question
     │
     ▼
Retrieval (FAISS)  ←── Pre-embedded Britannica Chunks
     │
     ▼
Prompt Construction
     │
     ▼
TogetherAI LLM Call
     │
     ▼
Final Answer + UI Visualization
```

## 🧠 API Used

- TogetherAI — https://www.together.ai
- Model: `mistralai/Mistral-7B-Instruct-v0.1`

## 📚 Dataset

- Source: Britannica (Land section of France)
- Scraped via BeautifulSoup and saved as `embedding_store.pkl`

## ▶️ How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run FastAPI server

```bash
uvicorn main:app --reload
```

Or via Colab-compatible threading setup if needed.

### 3. Run UI with Streamlit

```bash
streamlit run app.py
```

## 📄 Report

See `France_RAG_Report.docx` for full architecture and results.

## 🤝 Contributors

- Amin Kiani — AI Engineer
- Radmehr AghaKhani — AI Engineer
  
## 📜 License

UI License
© 2025 - RAG Final Project
