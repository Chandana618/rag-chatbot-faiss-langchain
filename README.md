# 📄 RAG-based Document Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that enables users to ask questions from their own PDF documents using semantic search and a Large Language Model (LLM). The application retrieves relevant document content and generates accurate, context-based answers through an interactive Streamlit interface.

---

## 🚀 Features

- Query custom PDF documents
- Text chunking with overlap for better context understanding
- Semantic search using vector embeddings
- Fast similarity search with FAISS
- LLM-based answer generation (Groq API)
- Interactive Streamlit web interface
- No OpenAI API key required

---

## 🧠 How It Works

1. PDF documents are loaded from the `documents/` folder  
2. Text is split into overlapping chunks  
3. Chunks are converted into vector embeddings  
4. FAISS performs similarity search on embeddings  
5. Relevant context is passed to the LLM  
6. The LLM generates answers strictly based on retrieved context  

---

## 🛠 Tech Stack

- Python  
- LangChain  
- FAISS  
- Hugging Face Sentence Transformers  
- Groq LLM  
- Streamlit  

---

## 📂 Project Structure

```text
rag-chatbot-faiss-langchain/
│
├── app.py
├── requirements.txt
├── documents/
├── README.md
└── .gitignore
