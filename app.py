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
def ask_question(llm, vectorstore, query):
    results = vectorstore.similarity_search_with_score(query,k=3)
    
    if not results:
        return "I couldn’t find relevant information in the document."
    scores = [score for _, score in results]
    best_score = min(scores)
    if best_score > 1.0:
        return "The document does not clearly contain an answer to this question."

    docs = [doc for doc, _ in results]
    
    sources = []
    context = ""
    for doc in docs:
      context += doc.page_content + "\n\n"
      if "page" in doc.metadata:
         sources.append(doc.metadata["page"])
 
    
    
    
   # if best_score < 0.6:
    #     tone = "Answer confidently."
    # else:
    #     tone = "Answer cautiously and mention possible uncertainty."
    
    # # Use last 3 turns only (realistic)
    # history_text = ""
    # for q, a in chat_history[-3:]:
    #     history_text += f"User: {q}\nAssistant: {a}\n"

    prompt = f"""
You are an academic assistant.

Using ONLY the provided context:
- Give a clear definition.
- Explain key points.
- Keep the answer concise (5-7 sentences) and structured way.
- Do not repeat ideas.
- Do NOT mention uncertainty unless the context explicitly says so.

Context:
{context}

Question:
{query}
"""

    answer=llm.invoke(prompt).content
    return answer, list(set(sources))


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

    docs = []

    for i, page in enumerate(reader.pages):
       text = page.extract_text()
       if text:
            docs.append(
               Document(
                page_content=text,
                metadata={"page": i + 1}
               )
            )
    
   
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
    answer,sources = ask_question(llm, st.session_state.vectorstore, query)
    # st.session_state.chat_history.append((query, answer))
    st.subheader("Answer")
    st.write(answer)
    clean_sources = sorted(set(sources))
    formatted = ", ".join([f"Page {p}" for p in clean_sources])
    st.write("📌 Sources:", formatted)
    

elif query:
    st.warning("Please upload a PDF first.")
