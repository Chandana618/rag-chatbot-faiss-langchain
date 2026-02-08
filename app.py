import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from PyPDF2 import PdfReader

from langchain.schema import Document

# -----------------------------
# Load documents
# -----------------------------
def load_documents():
    loader = PyPDFDirectoryLoader("documents")
    return loader.load()

# -----------------------------
# Split documents
# -----------------------------
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)

# -----------------------------
# Embeddings
# -----------------------------
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

# -----------------------------
# Vector store
# -----------------------------
def create_vectorstore(chunks, embeddings):
    return FAISS.from_documents(chunks, embeddings)

# -----------------------------
# Retriever
# -----------------------------
def create_retriever(vectorstore):
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# -----------------------------
# LLM (Ollama)
# -----------------------------
def load_llm():
    return Ollama(model="mistral", temperature=0)

# -----------------------------
# RAG chain
# -----------------------------
def ask_question(llm, retriever, query):
    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}
"""

    return llm.invoke(prompt)


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📄 Free RAG Chatbot")

pdf = st.file_uploader("Upload a PDF", type="pdf")

# Session state for vectorstore
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    
if pdf and st.session_state.vectorstore is None:
    st.info("Processing PDF...")

    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    docs = [Document(page_content=text)]

    chunks = split_documents(docs)
    embeddings = load_embeddings()
    st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)

    st.success("PDF processed successfully!")
    

query = st.text_input("Ask a question from your documents")

if query:
    docs = load_documents()
    chunks = split_documents(docs)
    embeddings = load_embeddings()
    vectorstore = create_vectorstore(chunks, embeddings)
    retriever = create_retriever(vectorstore)
    llm = load_llm()
    response = ask_question(llm, retriever, query)
    st.subheader("Answer")
    st.write(response)


    
