"""
PDF Question Answering App using Streamlit, FAISS, Sentence Transformers, PyMuPDF, and Groq API.

Features:
- Upload one or more PDFs.
- Chunk the extracted text and embed with a transformer model.
- Build a FAISS index for semantic search.
- Retrieve relevant chunks and query the Groq LLM.
- Automatic context truncation to prevent API request errors.
- Logging added for better debugging.
"""

import os
import logging
import fitz  # PyMuPDF
import faiss
import numpy as np
import requests
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer

# ----------------------- Configuration -----------------------

torch.classes.__path__ = []
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

st.set_page_config(page_title="PDF Q&A with Groq", layout="wide")

GROQ_API_KEY = "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"
GROQ_MODEL = "llama3-8b-8192"
TOP_K = 3
MAX_CONTEXT_CHARS = 3000

# ----------------------- Logging Setup -----------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ----------------------- Model Loading -----------------------

@st.cache_resource
def load_embedding_model():
    """Load and cache the sentence transformer embedding model."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Embedding model loaded successfully.")
    return model

embedding_model = load_embedding_model()

# ----------------------- Functions -----------------------

def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in pdf_document:
        text += page.get_text() + "\n\n\n"
    pdf_document.close()
    logger.info("Extracted %d characters from PDF.", len(text))
    return text.strip()

def create_chunks(text, chunk_size=150, overlap=30):
    """Split text into overlapping chunks."""
    words = text.split("\n\n\n")
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    logger.info("Created %d chunks.", len(chunks))
    return chunks

def embed_chunks(chunks):
    """Embed text chunks using the sentence transformer model."""
    embeddings = embedding_model.encode(chunks)
    logger.info("Generated embeddings with shape %s.", embeddings.shape)
    return embeddings

def build_faiss_index(embeddings):
    """Build a FAISS index from the embeddings."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    logger.info("Built FAISS index with %d vectors.", index.ntotal)
    return index

def rag_pipeline(query, index, metadata, k=TOP_K):
    """Retrieve relevant chunks and query Groq API for answer."""
    query_vec = embedding_model.encode([query])[0]
    distances, indices = index.search(np.array([query_vec]).astype('float32'), k)

    context = "Reference materials:\n"
    for idx in indices[0]:
        context += f"\n▸ From {metadata[idx]['source']}:\n{metadata[idx]['text']}\n"

    if len(context) > MAX_CONTEXT_CHARS:
        logger.warning("Context too large (%d chars). Truncating to %d chars.", len(context), MAX_CONTEXT_CHARS)
        context = context[:MAX_CONTEXT_CHARS]

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": context},
            {"role": "user", "content": query}
        ]
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=20
        )
        response.raise_for_status()
        answer = response.json()['choices'][0]['message']['content'].strip()
        logger.info("Received response from Groq API.")
        return answer
    except requests.exceptions.RequestException as e:
        logger.error("Network or Request Error: %s", str(e))
        return f"Network or Request Error: {str(e)}"
    except (KeyError, IndexError) as e:
        logger.error("Groq Response Parsing Error: %s", str(e))
        return f"Groq Response Parsing Error: {str(e)}"
    except Exception as e:
        logger.error("Unexpected Error: %s", str(e))
        return f"Unexpected Error: {str(e)}"

# ----------------------- Streamlit App -----------------------

st.title("Ask Questions to Your PDF using Groq LLM")

uploaded_files = st.file_uploader("Upload your PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    metadata = []
    all_chunks = []

    with st.spinner("Extracting and processing text from PDFs..."):
        for uploaded_file in uploaded_files:
            text = extract_text_from_pdf(uploaded_file)
            chunks = create_chunks(text)
            all_chunks.extend(chunks)
            for chunk in chunks:
                metadata.append({'source': uploaded_file.name, 'text': chunk})

    with st.spinner("Embedding chunks and building FAISS index..."):
        chunk_embeddings = embed_chunks(all_chunks)
        index = build_faiss_index(np.array(chunk_embeddings))

    st.success(f"Indexed {len(metadata)} text chunks from {len(uploaded_files)} PDFs.")

    question = st.text_input("Ask a question about the PDFs")

    if st.button("Get Answer"):
        if not question.strip():
            st.error("Please enter a valid question.")
        else:
            with st.spinner("Retrieving answer..."):
                answer = rag_pipeline(question, index, metadata)
            st.success("Answer generated!")
            st.markdown(f"**Answer:**\n\n{answer}")

st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit, FAISS, Sentence Transformers, PyMuPDF, and Groq.")

st.sidebar.markdown("""
### Tips for Using This Application:
- Upload one or more PDF files to get started.
- After processing, ask specific questions related to the content.
- Make sure your Groq API Key and Model are correctly set.
""")