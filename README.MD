Persistent Multi-PDF RAG Chatbot

An AI-powered Multi-PDF RAG Chatbot built using Python, Streamlit, LangChain, Google Gemini and ChromaDB.

This application allows users to upload multiple PDF documents and ask questions based on their content. It retrieves relevant information from uploaded documents and generates context-based answers with source file names and page references.

Features:

Multiple PDF Upload: Upload and manage multiple PDF documents.

Persistent Storage: Uploaded PDFs are saved locally and remain available after application restarts.

RAG-Based Question Answering: Ask questions and get answers based on uploaded PDF content.

Vector Database: Uses ChromaDB to store document embeddings.

Document Selection: Ask questions from all uploaded documents or select a specific PDF.

Source References: Displays the source PDF name and page number with retrieved information.

Avoid Re-indexing: Tracks PDF file size and modification time to avoid unnecessary re-indexing of unchanged documents.

Chat History: Maintains conversation history during the application session.

Technologies Used :

Python

Streamlit

LangChain

Google Gemini

ChromaDB

PyPDF

Google Generative AI Embeddings

How It Works-

Upload one or more PDF documents.

The application extracts text from each PDF.

Extracted text is divided into smaller chunks using RecursiveCharacterTextSplitter.

Google Generative AI Embeddings converts chunks into vector embeddings.

ChromaDB stores the embeddings in a persistent vector database.

When a user asks a question, the system retrieves the 4 most relevant chunks.

Retrieved information is passed to Google Gemini.

Gemini generates an answer using only the retrieved document context.

The application displays the answer along with source file names and page references.