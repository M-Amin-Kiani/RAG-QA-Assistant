
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

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🖥️ Run the App

```bash
streamlit run app.py
```

## 🧠 API Used

- TogetherAI — https://www.together.ai
- Model: `mistralai/Mistral-7B-Instruct-v0.1`

## 📚 Dataset

- Source: Britannica (Land section of France)
- Scraped via BeautifulSoup and saved as `embedding_store.pkl`

## 📄 Report

See `France_RAG_Report.docx` for full architecture and results.

## 🤝 Contributors

- Amin Kiani — AI Engineer
- Radmehr AghaKhani — AI Engineer
  
## 📜 License

MIT License
