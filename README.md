#H1 Local RAG / LLM code for financial report analysis
#H3 How to use it
** 1. Install dependencies **
# pip install langchain faiss-cpu streamlit pypdf langchain-community pypdf

** 2. start ollma with LLM & embedding **
#ollama run llama3
#ollama pull nomic-embed-text

** 3. Put hte financial pdfs under ./pdfs folder

** 4. Run the streamlit app
# streamlit run app.py

### 📁 Folder Structure
# .
# ├── app.py
# ├── config.py
# ├── loader.py
# ├── rag_chain.py
# ├── pdfs/                 <-- Place your financial PDFs here
# └── vectorstore/          <-- Auto-generated FAISS vector DB
