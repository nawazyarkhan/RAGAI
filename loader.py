### loader.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from config import PERSIST_DIRECTORY, EMBED_MODEL
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_pdfs_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, filename))
            documents.extend(loader.load())
    return documents

def build_vectorstore_from_docs(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(PERSIST_DIRECTORY)
    return db

def load_or_build_vectorstore(folder_path):
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    index_path = os.path.join(PERSIST_DIRECTORY, "index.faiss")
    if os.path.exists(PERSIST_DIRECTORY) and os.path.exists(index_path):
        return FAISS.load_local(PERSIST_DIRECTORY, embeddings, allow_dangerous_deserialization=True)
    docs = load_pdfs_from_folder(folder_path)
    return build_vectorstore_from_docs(docs)