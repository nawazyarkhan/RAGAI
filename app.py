### app.py
import streamlit as st
from loader import load_or_build_vectorstore
from rag_chain import get_rag_chain

st.set_page_config(page_title="Financial RAG Chat", layout="centered")
st.title("📊 Financial RAG Chat with Ollama")

pdf_folder = st.text_input("Path to folder containing financial PDFs:", "./pdfs")

if "chain" not in st.session_state:
    with st.spinner("Loading documents and initializing model..."):
        vectordb = load_or_build_vectorstore(pdf_folder)
        st.session_state.chain = get_rag_chain(vectordb)

query = st.text_input("Ask a question about the financial documents:", "What is Tesla's debt situation?")
if st.button("Ask") and query:
    with st.spinner("Thinking..."):
        result = st.session_state.chain.run(query)
        st.markdown("### 💡 Answer")
        st.write(result)