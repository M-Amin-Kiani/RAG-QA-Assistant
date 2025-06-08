import streamlit as st
import requests
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import plotly.express as px
import os
# ----------------- بارگذاری داده‌ها -----------------
with open("embedding_store.pkl", "rb") as f:
    store = pickle.load(f)

chunks = store["chunks"]
metadata = store["metadata"]
embeddings = np.array(store["embeddings"]).astype("float32")

# اگر user_additions وجود دارد، اضافه کن
if os.path.exists("user_additions.pkl"):
    with open("user_additions.pkl", "rb") as f:
        user_store = pickle.load(f)
    chunks += user_store["chunks"]
    metadata += user_store["metadata"]
    user_embeddings = np.array(user_store["embeddings"]).astype("float32")
    embeddings = np.concatenate([embeddings, user_embeddings], axis=0)


model = SentenceTransformer("all-MiniLM-L6-v2")
# model = SentenceTransformer("togethercomputer/m2-bert-80M-8k-retrieval")

#Vector Retrieval
faiss.normalize_L2(embeddings) 
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# ----------------- توابع اصلی -----------------
# def retrieve_top_k(query, k=3):
#     query_emb = model.encode([query]).astype("float32")
#     faiss.normalize_L2(query_emb)
#     scores, indices = index.search(query_emb, k)
#     return [
#         {
#             "chunk": chunks[i],
#             "metadata": metadata[i],
#             "score": round(float(scores[0][j]), 4)
#         }
#         for j, i in enumerate(indices[0])
#     ]

def retrieve_top_k(query, k=3, filter_by=None):
    """
    query: سوال کاربر
    k: تعداد نتایج برتر
    filter_by: دیکشنری فیلتر مثل {'category': 'France-Geography'}
    """
    # فیلتر کردن داده‌ها بر اساس متادیتا
    filtered_chunks = []
    filtered_metadata = []
    filtered_embeddings = []

    for i, meta in enumerate(metadata):
        if filter_by is None or all(meta.get(key) == val for key, val in filter_by.items()):
            filtered_chunks.append(chunks[i])
            filtered_metadata.append(meta)
            filtered_embeddings.append(embeddings[i])

    if not filtered_chunks:
        return []

    index_temp = faiss.IndexFlatIP(embeddings.shape[1])
    vectors = np.array(filtered_embeddings).astype("float32")
    faiss.normalize_L2(vectors)
    index_temp.add(vectors)

    # Embedding سوال کاربر و جستجو
    query_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_emb)
    scores, indices = index_temp.search(query_emb, k)

    return [
        {
            "chunk": filtered_chunks[i],
            "metadata": filtered_metadata[i],
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
    model_name = st.selectbox("LLM Model", ["mistralai/Mistral-7B-Instruct-v0.1",
                                            "deepseek-ai/DeepSeek-Coder-6.7B-Instruct",
                                            "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                                            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"])
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7)
    top_p = st.slider("Top-p", 0.0, 1.0, 0.9)
    max_tokens = st.slider("Max Tokens", 64, 1024, 256)

with st.expander("📂 Metadata Filters"):
    category_options = sorted(set(
        m["category"] for m in metadata if m.get("category") not in [None, ""]
    ))
    source_options = sorted(set(
        m["source"] for m in metadata if m.get("source") not in [None, ""]
    ))

    category_options = ["All"] + category_options
    source_options = ["All"] + source_options

    selected_category = st.selectbox("Filter by Category", category_options)
    selected_source = st.selectbox("Filter by Source", source_options)

    filters = {}
    if selected_category != "All":
        filters["category"] = selected_category
    if selected_source != "All":
        filters["source"] = selected_source


retrieved = retrieve_top_k(query, top_k, filter_by=filters)

if st.button("🚀 Generate Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("🔄 Retrieving context and generating response..."):
            retrieved = retrieve_top_k(query, top_k)
            answer, prompt_used = generate_answer(query, retrieved, model_name, temperature, top_p, max_tokens)

        st.success("✅ Final Answer:")
        # st.markdown(f"**{answer}**")
        st.code(f"**{answer}**", language="markdown")
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


with st.expander("ℹ️ Retrieval Strategy Explanation"):
    st.markdown("""
    **Retrieval Approach Used:**
    - ✅ Cosine Similarity over Sentence Embeddings
    - ✅ Optional Metadata Filtering (category, source, etc.)
    - ✅ Top-K configurable

    **Why this works well:**
    - Cosine similarity finds chunks that are semantically close to your query.
    - Metadata filtering improves *precision* by narrowing context to relevant sections.
    - Top-K tuning allows control over *recall* (bigger K → more results).

    This balances precision and recall effectively by combining dense vector search with symbolic filters.
    """)


with st.expander("➕ Add New Data (URL or Text)"):
    mode = st.radio("Input type:", ["🧭 URL", "📝 Raw Text"])
    new_title = st.text_input("Title")
    new_data = st.text_area("Paste your URL or text here")

    if st.button("📥 Add to Dataset"):
        if not new_title or not new_data:
            st.warning("Both title and data are required.")
        else:
            if mode == "🧭 URL":
                import requests
                from bs4 import BeautifulSoup
                try:
                    res = requests.get(new_data)
                    soup = BeautifulSoup(res.text, "html.parser")
                    paragraphs = soup.find_all("p")
                    new_content = "\n".join(p.get_text(strip=True) for p in paragraphs)
                except Exception as e:
                    st.error(f"Failed to fetch URL: {e}")
                    new_content = ""
            else:
                new_content = new_data

            if new_content.strip():
                # Chunk + Embed
                from sentence_transformers import SentenceTransformer
                import faiss
                import numpy as np
                import nltk
                nltk.download('punkt')
                from nltk.tokenize import sent_tokenize

                def chunk_text(text, size=100, overlap=20):
                    words = text.split()
                    chunks = []
                    for i in range(0, len(words), size - overlap):
                        chunk = " ".join(words[i:i+size])
                        if len(chunk.split()) > 10:
                            chunks.append(chunk)
                    return chunks

                new_chunks = chunk_text(new_content)
                new_embeddings = model.encode(new_chunks).astype("float32")
                faiss.normalize_L2(new_embeddings)
                index.add(new_embeddings)

                chunks.extend(new_chunks)
                metadata.extend([{"title": new_title, "url": new_data}] * len(new_chunks))

                user_path = "user_additions.pkl"

                if os.path.exists(user_path):
                    with open(user_path, "rb") as f:
                        existing = pickle.load(f)
                    user_chunks = existing["chunks"] + new_chunks
                    user_metadata = existing["metadata"] + [{"title": new_title, "url": new_data}] * len(new_chunks)
                    user_embeddings = np.concatenate([existing["embeddings"], new_embeddings], axis=0)
                else:
                    user_chunks = new_chunks
                    user_metadata = [{"title": new_title, "url": new_data}] * len(new_chunks)
                    user_embeddings = new_embeddings

                with open(user_path, "wb") as f:
                    pickle.dump({
                        "chunks": user_chunks,
                        "embeddings": user_embeddings,
                        "metadata": user_metadata
                    }, f)

                # # ذخیره به embedding_store.pkl
                # import pickle
                # with open("embedding_store.pkl", "wb") as f:
                #     pickle.dump({
                #         "chunks": chunks,
                #         "embeddings": index.reconstruct_n(0, index.ntotal),  # بازسازی همه بردارها
                #         "metadata": metadata
                #     }, f)

                st.success(f"{len(new_chunks)} chunks added successfully ✅")

# with st.expander("📂 View User-Added Data"):
#     import os
#     if os.path.exists("user_additions.pkl"):
#         with open("user_additions.pkl", "rb") as f:
#             udata = pickle.load(f)

#         st.markdown(f"**Total Chunks:** {len(udata['chunks'])}")
#         for i, (chunk, meta) in enumerate(zip(udata["chunks"], udata["metadata"])):
#             with st.expander(f"🧩 Chunk {i+1} — {meta['title']}"):
#                 st.code(chunk[:500] + "...")
#                 st.markdown(f"🔗 [Source]({meta['url']})")
#     else:
#         st.info("No user-added data yet.")


with st.expander("📂 View & Manage User-Added Data"):
    import os
    user_path = "user_additions.pkl"

    if os.path.exists(user_path):
        with open(user_path, "rb") as f:
            udata = pickle.load(f)

        st.markdown(f"**🧠 Total Chunks:** {len(udata['chunks'])}")

        for i, (chunk, meta) in enumerate(zip(udata["chunks"], udata["metadata"])):
            st.markdown(f"---\n### 🧩 Chunk {i+1} — {meta['title']}")
            st.code(chunk[:500] + "...")
            st.markdown(f"🔗 [Source]({meta['url']})")

            if st.button(f"❌ Delete Chunk {i+1}", key=f"del_{i}"):
                udata["chunks"].pop(i)
                udata["metadata"].pop(i)
                udata["embeddings"] = np.delete(udata["embeddings"], i, axis=0)

                with open(user_path, "wb") as f:
                    pickle.dump(udata, f)

                st.experimental_rerun()
    else:
        st.info("No user-added data yet.")

