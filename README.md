📄 RAG-based Document Chatbot.

A Retrieval-Augmented Generation (RAG) chatbot that allows users to ask questions from their own PDF documents using semantic search and a large language model. The system retrieves relevant document chunks using vector similarity and generates accurate, context-grounded answers through an interactive Streamlit interface.

🚀 Features

📑 Upload and query custom PDF documents

✂️ Intelligent text chunking for better context retrieval

🔍 Semantic search using vector embeddings

⚡ Fast similarity search with FAISS

🤖 LLM-powered answer generation (Groq API)

🖥️ Interactive web UI built with Streamlit

🔑 No OpenAI API key required

🧠 How It Works (RAG Pipeline)

Document Ingestion
PDF files are loaded from the documents/ directory.

Text Splitting
Documents are split into overlapping chunks to preserve context.

Vector Embeddings
Each chunk is converted into embeddings using a Hugging Face sentence transformer.

Vector Store (FAISS)
Embeddings are stored in FAISS for efficient similarity search.

Retrieval
The most relevant document chunks are retrieved for a given user query.

Generation
A Large Language Model generates answers strictly based on the retrieved context.

🛠 Tech Stack

Programming Language: Python

Framework: LangChain

Vector Database: FAISS

Embeddings: Hugging Face Sentence Transformers

LLM Provider: Groq

Frontend: Streamlit

📂 Project Structure
rag-chatbot-faiss-langchain/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── documents/             # Folder containing PDF files
├── README.md              # Project documentation
└── .gitignore
