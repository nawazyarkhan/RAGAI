### rag_chain.py
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from config import OLLAMA_MODEL


def get_rag_chain(vectorstore):
    llm = Ollama(model=OLLAMA_MODEL)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain