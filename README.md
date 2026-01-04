# 📄 RAG Chatbot using LangChain & FAISS (Free)

This project is a **Retrieval-Augmented Generation (RAG) chatbot** that allows users to query custom documents using semantic search and a local LLM.

## 🚀 Features
- Document ingestion (PDF)
- Text chunking & embeddings
- FAISS vector search
- Local LLM using Ollama (no OpenAI key needed)
- Streamlit-based UI

## 🛠 Tech Stack
- Python
- LangChain
- FAISS
- Sentence Transformers
- Ollama
- Streamlit

## 📂 Project Structure
```text
rag-chatbot-faiss-langchain/
│
├── documents/
│   └── sample.pdf        # Add your PDF files here
│
├── app.py                # Main Streamlit application
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
└── .gitignore            # Ignored files (venv, cache, etc.)
