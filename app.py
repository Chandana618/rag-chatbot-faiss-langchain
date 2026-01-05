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
    return ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0
    )


# -----------------------------
# RAG chain
# -----------------------------
def ask_question(llm, retriever, query):
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found in the documents."
        
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}
"""

try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return "⚠️ The model is temporarily unavailable. Please try again later."


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📄 Free RAG Chatbot")

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


    



