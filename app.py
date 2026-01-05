import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage


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
@st.cache_resource(show_spinner=False)
def create_vectorstore(chunks):
    embeddings = load_embeddings()
    return FAISS.from_documents(chunks, embeddings)


# -----------------------------
# Retriever
# -----------------------------
def create_retriever(vectorstore):
    return vectorstore.as_retriever(search_kwargs={"k": 2})


# -----------------------------
# LLM (Groq)
# -----------------------------
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )


# -----------------------------
# RAG Question Answering
# -----------------------------
def ask_question(llm, retriever, query):
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found in the documents."

    # HARD limit context to avoid Groq BadRequest
    context = "\n\n".join(doc.page_content[:600] for doc in docs)

    messages = [
        HumanMessage(
            content=f"""
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}
"""
        )
    ]

    response = llm.invoke(messages)
    return response.content


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Free RAG Chatbot", layout="centered")

st.title("📄 Free RAG Chatbot")
st.write("Ask questions based on your uploaded PDF documents.")

query = st.text_input("Ask a question from your documents")

if query:
    with st.spinner("Processing your question..."):
        docs = load_documents()
        chunks = split_documents(docs)
        vectorstore = create_vectorstore(chunks)
        retriever = create_retriever(vectorstore)
        llm = load_llm()

        response = ask_question(llm, retriever, query)

    st.subheader("Answer")
    st.write(response)

