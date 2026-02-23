import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader

# ================= LOAD ENV =================
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("❌ GOOGLE_API_KEY not found in .env file")
    st.stop()

genai.configure(api_key=API_KEY)

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI File Chat (Gemini RAG)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= DARK THEME CSS =================
st.markdown("""
<style>
body {
    background-color: #0f1117;
}
.chat-user {
    background-color: #2563eb;
    color: white;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
}
.chat-ai {
    background-color: #1f2933;
    color: white;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.title("🤖 AI File Chat (Gemini RAG)")
st.caption("Upload a PDF file and ask questions only from that document")

# ================= SESSION STATE =================
if "documents" not in st.session_state:
    st.session_state.documents = []

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ================= LOAD EMBEDDING MODEL =================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embedding_model()

# ================= FILE UPLOAD =================
uploaded_file = st.file_uploader("📂 Upload PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing document..."):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name

        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"

        # Split into chunks
        chunk_size = 800
        overlap = 100
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap

        # Create embeddings
        embeddings = embed_model.encode(chunks)

        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        st.session_state.index = index
        st.session_state.chunks = chunks
        st.session_state.chat_history = []

        st.success("✅ Document processed successfully!")

# ================= QUESTION INPUT =================
query = st.text_input("💬 Ask a question from the document")

if st.button("Ask"):
    if st.session_state.index is None:
        st.warning("⚠️ Please upload a document first.")
    elif query.strip() == "":
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            # Embed the query
            query_embedding = embed_model.encode([query])

            # Search similar chunks
            k = 4
            distances, indices = st.session_state.index.search(query_embedding, k)

            context = ""
            for idx in indices[0]:
                context += st.session_state.chunks[idx] + "\n\n"

            # Gemini model
            model = genai.GenerativeModel("gemini-2.5-flash")

            prompt = f"""
You are an AI assistant.
Answer ONLY using the context below.
If the answer is not found in the context, reply exactly:
"Not found in the document."

Context:
{context}

Question:
{query}
"""

            response = model.generate_content(prompt)

            st.session_state.chat_history.append(("You", query))
            st.session_state.chat_history.append(("AI", response.text))

# ================= CHAT DISPLAY =================
st.markdown("### 💬 Chat History")

for role, msg in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"<div class='chat-user'><b>You:</b> {msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-ai'><b>AI:</b> {msg}</div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("---")
st.caption("Powered by Gemini + Streamlit | Secure RAG AI Tool")
