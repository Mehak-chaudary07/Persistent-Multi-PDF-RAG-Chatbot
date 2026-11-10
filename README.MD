# Persistent Multi-PDF RAG Chatbot

An AI-powered Multi-PDF RAG Chatbot built using Python, Streamlit, LangChain, Google Gemini, and ChromaDB.

This application allows users to upload multiple PDF documents and ask questions based on their content. It retrieves relevant information from uploaded documents and generates context-based answers with source file names and page references.

## Features

- **Multiple PDF Upload:** Upload and manage multiple PDF documents.
- **Persistent Storage:** Uploaded PDFs are saved locally and remain available after application restarts.
- **RAG-Based Question Answering:** Ask questions and get answers based on uploaded PDF content.
- **Vector Database:** Uses ChromaDB to store document embeddings.
- **Document Selection:** Ask questions from all uploaded documents or select a specific PDF.
- **Source References:** Displays the source PDF name and page number with retrieved information.
- **Avoid Re-indexing:** Tracks PDF file size and modification time to avoid unnecessary re-indexing of unchanged documents.
- **Chat History:** Maintains conversation history during the application session.

## Technologies Used

- Python
- Streamlit
- LangChain
- Google Gemini
- ChromaDB
- PyPDF
- Google Generative AI Embeddings

## How It Works

1. Upload one or more PDF documents.
2. The application extracts text from each PDF.
3. Extracted text is divided into smaller chunks using `RecursiveCharacterTextSplitter`.
4. Google Generative AI Embeddings converts chunks into vector embeddings.
5. ChromaDB stores the embeddings in a persistent vector database.
6. When a user asks a question, the system retrieves the 4 most relevant chunks.
7. Retrieved information is passed to Google Gemini.
8. Gemini generates an answer using only the retrieved document context.
9. The application displays the answer along with source file names and page references.

## Installation and Setup

### Prerequisites

- Python installed
- Google Gemini API Key

### 1. Clone the Repository

```bash
git clone https://github.com/Mehak-chaudary07/Persistent-Multi-PDF-RAG-Chatbot.git
```

### 2. Navigate to Project Folder

```bash
cd Persistent-Multi-PDF-RAG-Chatbot
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Setup API Key

- Create a `.env` file in the root project folder.
- Copy the contents of `.env.example` into `.env`.
- Get your API key from [Google AI Studio](https://aistudio.google.com/apikey
