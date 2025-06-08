import streamlit as st
import requests
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import plotly.express as px

# ----------------- بارگذاری داده‌ها -----------------
with open("embedding_store.pkl", "rb") as f:
    store = pickle.load(f)

chunks = store["chunks"]
metadata = store["metadata"]
embeddings = np.array(store["embeddings"]).astype("float32")
model = SentenceTransformer("all-MiniLM-L6-v2")

faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# ----------------- توابع اصلی -----------------
def retrieve_top_k(query, k=3):
    query_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_emb)
    scores, indices = index.search(query_emb, k)
    return [
        {
            "chunk": chunks[i],
            "metadata": metadata[i],
            "score": round(float(scores[0][j]), 4)
        }
        for j, i in enumerate(indices[0])
    ]

TOGETHER_API_KEY = "a645ea5d637a4b47630140c2eb3579f8d2b3761155813a3a2761fb669098c831"  #  توکن خودت

def generate_answer(query, retrieved, model_name, temperature, top_p, max_tokens):
    context = "\n\n".join([c["chunk"] for c in retrieved])

    prompt = f"""You are a helpful assistant answering based on provided context.

Context:
{context}

Question:
{query}

Answer:"""

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    }

    res = requests.post("https://api.together.xyz/v1/completions", headers=headers, json=payload)

    if res.status_code != 200:
        return f"❌ API Error {res.status_code}: {res.text}", prompt

    answer = res.json()["choices"][0]["text"].strip()
    return answer, prompt

# ----------------- Streamlit UI -----------------
st.set_page_config("📘 France RAG Assistant", layout="wide")

st.markdown("<h1 style='color:#0f62fe;'>🇫🇷 RAK QA Assistant</h1>", unsafe_allow_html=True)
st.write("Ask questions about France's **Land, Mountains, Rivers, Climate** etc. Powered by RAG and LLM.")

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("🧠 What do you want to know?", placeholder="E.g. What are the main mountains in France?")
with col2:
    top_k = st.slider("🔍 Top-K Chunks", 1, 10, 3)

# Parameters section
with st.expander("⚙️ Model & Generation Settings"):
    model_name = st.selectbox("LLM Model", ["mistralai/Mistral-7B-Instruct-v0.1", "deepseek-ai/DeepSeek-Coder-6.7B-Instruct"])
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7)
    top_p = st.slider("Top-p", 0.0, 1.0, 0.9)
    max_tokens = st.slider("Max Tokens", 64, 1024, 256)

if st.button("🚀 Generate Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("🔄 Retrieving context and generating response..."):
            retrieved = retrieve_top_k(query, top_k)
            answer, prompt_used = generate_answer(query, retrieved, model_name, temperature, top_p, max_tokens)

        st.success("✅ Final Answer:")
        st.markdown(f"**{answer}**")
        st.code(answer, language="markdown")
        st.caption(f"📝 Word count: {len(answer.split())}")
        st.download_button("📥 Download Answer", data=answer, file_name="answer.txt")

        st.markdown("---")
        st.subheader("📄 Retrieved Chunks and Similarity")

        for i, c in enumerate(retrieved):
            st.markdown(f"**Chunk {i+1} — Score: `{c['score']}`**")
            st.code(c["chunk"][:500] + "...")
            st.markdown(f"🔗 [{c['metadata']['title']}]({c['metadata']['url']})")

        st.markdown("### 📊 Chunk Similarity Chart")
        fig = px.bar(
            x=[c["score"] for c in retrieved],
            y=[f"Chunk {i+1}" for i in range(len(retrieved))],
            orientation='h',
            labels={'x': 'Similarity Score', 'y': 'Chunk'},
            color=[c["score"] for c in retrieved],
            color_continuous_scale='blues'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        with st.expander("🧾 View Prompt Sent to LLM"):
            st.code(prompt_used)

        st.markdown("💬 Was this answer useful?")
        cols = st.columns([1, 1])
        with cols[0]:
            if st.button("👍 Yes"):
                st.success("Thanks for your feedback!")
        with cols[1]:
            if st.button("👎 No"):
                st.info("We'll try to improve.")
