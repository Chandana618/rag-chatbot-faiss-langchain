import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from pypdf import PdfReader

from langchain_core.documents import Document
load_dotenv()
# -----------------------------
# Load documents
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


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
# def create_vectorstore(chunks, embeddings):
#     st.success("Vector store created")
#     return FAISS.from_documents(chunks, embeddings)

# -----------------------------
# Retriever
# -----------------------------
# def create_retriever(vectorstore):
    
#     return vectorstore.as_retriever(search_kwargs={"k": 3})

# -----------------------------
# LLM (Ollama)
# -----------------------------
def load_llm():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)

# -----------------------------
# RAG chain
# -----------------------------
def ask_question(llm, vectorstore, query,chat_history):
    results = vectorstore.similarity_search_with_score(query,k=5)
    
    if not results:
        return "I couldn’t find relevant information in the document."
    scores = [score for _, score in results]
    best_score = min(scores)
    if best_score > 1.2:
        return "The document does not clearly contain an answer to this question."

    docs = [doc for doc, _ in results]
    context = "\n\n".join(doc.page_content for doc in docs)

    if best_score < 0.6:
        tone = "Answer confidently."
    else:
        tone = "Answer cautiously and mention possible uncertainty."
    
    # Use last 3 turns only (realistic)
    history_text = ""
    for q, a in chat_history[-3:]:
        history_text += f"User: {q}\nAssistant: {a}\n"

    prompt = f"""
Answer the question using ONLY the context below.
You are a helpful assistant.
{tone}

Context:
{context}

Question:
{query}
"""

    return llm.invoke(prompt).content


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

if query and st.session_state.vectorstore:
    retriever = st.session_state.vectorstore.as_retriever(
        search_kwargs={"k": 6}
    )

    llm = load_llm()
    answer = ask_question(llm, st.session_state.vectorstore, query,st.session_state.chat_history)
    st.session_state.chat_history.append((query, answer))
    st.subheader("Answer")
    st.write(answer)

elif query:
    st.warning("Please upload a PDF first.")
