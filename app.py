import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from pypdf import PdfReader

from langchain_core.documents import Document
load_dotenv()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
FAISS_PATH = "faiss_index"
# -----------------------------
# Split documents
# -----------------------------
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)


def retrivepages_with_metadata(pdfs):
    all_docs=[]
    for pdf in pdfs:
      reader = PdfReader(pdf)
      docs=[]
      
      for i, page in enumerate(reader.pages):
         text = page.extract_text()
         if text:
             docs.append(
               Document(
                page_content=text,
                metadata={"page": i + 1,
                          "source": filename}

               )
            )
      all_docs.extend(docs)
    return all_docs
# -----------------------------
# Embeddings
# -----------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )


def load_vectorstore(embeddings):
    if os.path.exists(FAISS_PATH):
        return FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    return None


def load_llm():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.5,streaming=True)

# -----------------------------
# RAG chain
# -----------------------------
def ask_question(llm, vectorstore, query):
    results = vectorstore.similarity_search(query,k=4)
    
    if not results:
        return "I couldn’t find relevant information in the document.",[]   
    sources = []
    context = ""
    for doc in results:
      src=doc.metadata.get('source','unknown')
      page=doc.metadata.get('page','?')
      context += f"[Source: {src}, Page: {page}]\n"
      context += doc.page_content + "\n\n"
      if "page" in doc.metadata:
         sources.append(f"{src} - Page {page}")
 
    prompt = f"""
       You are an academic assistant. Using ONLY the provided context:
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
st.sidebar.title("📄 Free RAG Chatbot")

pdfs = st.sidebar.file_uploader("Upload a PDFs", type="pdf",accept_multiple_files=True)
process=st.sidebar.button("🚀 Process PDFs")

# Session state for vectorstore
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if process:
    if not pdfs:
        st.sidebar.warning("please upload atleast one pdf")
    else:
         st.session_state.vectorstore = None
         st.sidebar.info("Processing PDF...")
        

    
    

    text=get_raw_text(pdfs) 
    chunks = split_documents(text)
    embeddings = load_embeddings()
    st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
    # ✅ SAVE (Persistence)
    st.session_state.vectorstore.save_local(FAISS_PATH)
    st.sidebar.success("PDF processed successfully!")
    
    st.sidebar.markdown("### 📊 Stats")
    st.sidebar.write(f"Documents: {len(all_docs)}")
    st.sidebar.write(f"Chunks: {len(chunks)}")


st.title("💬 RAG Chatbot")
query = st.chat_input("Ask a question")

if query and st.session_state.vectorstore:

    llm = load_llm()
    answer,sources = ask_question(llm, st.session_state.vectorstore, query)
    if "Not found in document" in answer:
        sources = []
    
    st.session_state.chat_history.append({
        "question": query,
        "answer": answer,
        "sources": sources
    })
    
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])

        with st.chat_message("assistant"):
           st.write(chat["answer"])
           if chat["sources"]:
               formatted = ", ".join([f"Page {p}" for p in sorted(chat["sources"])])
               st.write(f"📌 Sources: {formatted}")
elif query:
    st.warning("Please upload a PDF first.")
