"""
PDF Question Answering Streamlit App with Groq LLM + FAISS + Sentence Transformers

Features:
- Upload PDFs and ask questions interactively.
- Embeds PDF contents with strong SentenceTransformer model.
- Retrieves and reranks relevant chunks using FAISS index.
- Asks Groq LLM via API with smart prompt formatting.
- Handles large context gracefully to avoid API errors.
- Full lazy logging enabled for transparency and debugging.

Author: [Your Name]
"""

import os
import logging
import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
import requests
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ----------------------------
# Setup Logging
# ----------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

torch.classes.__path__ = []  # Neutralize torch path inspection warning
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

# ----------------------------
# Streamlit Page Config
# ----------------------------

st.set_page_config(page_title="PDF Q&A with ChatBot", layout="wide")

# ----------------------------
# Constants
# ----------------------------
DEFAULT_MODEL = "llama3-8b-8192"
DEFAULT_TOP_K = 7
MAX_CONTEXT_CHARS = 12000  # Safe limit for Groq API context
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # can upgrade later
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# ----------------------------
# Caching the Embedding Model
# ----------------------------
@st.cache_resource
def load_embedding_model():
    """Load SentenceTransformer model."""
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model

embedding_model = load_embedding_model()

# ----------------------------
# Helper Functions
# ----------------------------
def extract_text_from_pdf(file):
    """Extract raw text from PDF file using PyMuPDF."""
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in pdf_document:
        text += page.get_text()
    pdf_document.close()
    return text.strip()

def split_text_into_chunks(text, chunk_size=400, overlap=50):
    """
    Split text into overlapping chunks.

    Args:
        text (str): The full extracted text.
        chunk_size (int): Number of words per chunk.
        overlap (int): Number of words to overlap.

    Returns:
        List[str]: List of text chunks.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def embed_chunks(chunks):
    """Embed text chunks using loaded embedding model."""
    logger.info("Embedding %d chunks...", len(chunks))
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    return np.array(embeddings)

def build_faiss_index(embeddings):
    """Build a FAISS index from chunk embeddings."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    logger.info("Built FAISS index with dimension %d", dim)
    return index

def truncate_context(context, max_chars=MAX_CONTEXT_CHARS):
    """Truncate context text safely within character limit."""
    if len(context) > max_chars:
        logger.warning("Context too long (%d chars), truncating...", len(context))
        context = context[:max_chars]
    return context

def query_groq(system_prompt, user_prompt, groq_api_key, model=DEFAULT_MODEL):
    """Send prompt to Groq API and return response."""
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    try:
        response = requests.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        logger.info("Groq API responded successfully.")
        return response.json()['choices'][0]['message']['content'].strip()
    except requests.exceptions.RequestException as e:
        logger.error("Groq API Request error: %s", str(e))
        return f"Network Error: {str(e)}"
    except Exception as e:
        logger.error("Unexpected error while calling Groq: %s", str(e))
        return f"Unexpected Error: {str(e)}"

def retrieve_context(question, index, chunks, k=DEFAULT_TOP_K):
    """Retrieve top-k relevant chunks for a query."""
    logger.info("Searching top %d chunks for the query.", k)
    query_vec = embedding_model.encode([question])[0]
    distances, indices = index.search(np.array([query_vec]).astype('float32'), k)

    retrieved = []
    for idx in indices[0]:
        retrieved.append(chunks[idx])

    context = "\n\n".join(retrieved)
    return truncate_context(context)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📝 Ask Questions to Your PDFs")

# Sidebar - API key input
# groq_api_key = st.sidebar.text_input("🔑 Groq API Key", type="password")
groq_api_key = st.secrets["groq"]["api_key"]
uploaded_files = st.file_uploader("📄 Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and groq_api_key:
    metadata = []
    all_chunks = []

    with st.spinner("Extracting and processing PDFs..."):
        for uploaded_file in uploaded_files:
            text = extract_text_from_pdf(uploaded_file)
            chunks = split_text_into_chunks(text)
            all_chunks.extend(chunks)
            for chunk in chunks:
                metadata.append({'source': uploaded_file.name, 'text': chunk})

    with st.spinner("Embedding chunks and building FAISS index..."):
        embeddings = embed_chunks(all_chunks)
        index = build_faiss_index(np.array(embeddings))

    st.success(f"✅ Indexed {len(all_chunks)} chunks from {len(uploaded_files)} PDFs.")

    question = st.text_input("❓ Ask a question about the PDFs")

    if st.button("🚀 Get Answer"):
        if not question.strip():
            st.error("Please enter a valid question.")
        else:
            with st.spinner("Generating answer..."):
                context = retrieve_context(question, index, all_chunks)
                system_prompt = f"You are an expert assistant. Answer based only on the following PDF content:\n\n{context}\n\nIf unsure, say 'I don't know based on provided data.'"
                answer = query_groq(system_prompt, question, groq_api_key)

            st.success("🎯 Answer Ready!")
            st.markdown(f"**Answer:**\n\n{answer}")

else:
    st.warning("⬅️ Please upload PDF(s)")

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; padding: 10px;'>"
    "Built with ❤️ using Streamlit, FAISS, Sentence Transformers, PyMuPDF, and Groq LLM.<br>"
    "© 2025"
    "</div>",
    unsafe_allow_html=True
)

