import uuid  #Universally Unique Identifier.(it basically used for creating unique ids)
import streamlit as st #Import Streamlit to build the interactive web application.
from pathlib import Path #it handles the path of files and folders(Path to create and manage file and directory paths.)
import warnings  #warnings to control warning messages displayed by Python.
import json #JavaScript Object Notation.JSON to store and retrieve PDF indexing information.(structured data)
from dotenv import load_dotenv #load_dotenv to load environment variables from the .env file.
load_dotenv()
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage #classes to structure conversation between user and ai model
from langchain_core.documents import Document #document is a structured object it have two imp attribute page_content and metadata
from langchain_google_genai import (ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings)  #they both are models
from langchain_chroma import Chroma  #used for vector database 
from pypdf import PdfReader   #PdfReader to read PDF files and extract their text.pdfreader is class from pypdf library
from langchain_text_splitters import RecursiveCharacterTextSplitter #the recursive text splitter to divide documents into smaller chunks.

PDF_FOLDER = Path("uploaded_pdfs") #uploaded_pdfs is a folder basically(Define the folder path where uploaded PDF files will be stored.)
DB_FOLDER = Path("chroma_db")
INDEX_FILE = Path("indexed_files.json") #the JSON file path used to track indexed PDF information.

PDF_FOLDER.mkdir(exist_ok=True) #mkdir make directory means creating a folder 
DB_FOLDER.mkdir(parents=True,exist_ok=True)
warnings.filterwarnings("ignore")


# ---------------- MODELS ----------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview") #this is a embedding model

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,) # this is a chat model

#this function is used to  Save uploaded PDF files into the local PDF folder.
def save_uploaded_pdfs(uploaded_files):  #uploaded_files is a parameter(here uploaded files values from down streamlit code uploaded_files=st.sidebar.....)
    for file in uploaded_files:
        file_path = PDF_FOLDER / file.name
        file_data = file.getvalue() #it returns the binary content
        if not file_path.exists() or file_path.read_bytes() != file_data:
            with open(file_path, "wb") as f:
                f.write(file_data)
    #wb means write binary format 
# uploaded_files = st.sidebar.file_uploader(
#     "Upload PDF files",
#     type=["pdf"],
#     accept_multiple_files=True)

# if uploaded_files:
#     save_uploaded_pdfs(uploaded_files)


# ---------------- LOAD PDF ----------------

def load_pdf(pdf_path):  #Read a PDF file, extract text from each page, and return the extracted pages as Document objects.
    reader = PdfReader(str(pdf_path)) #Create a PDF reader object to access the contents of the PDF file.
    pages = [] #Create an empty list to store the extracted PDF pages.
    for page_number, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(
            Document(
                page_content=text,
                metadata={
                    "source": pdf_path.name,
                    "page": page_number 
                }))
    return pages  #page content is a actual information and meta data is additional information


# ---------------- LOAD MULTIPLE PDFs ----------------

def load_multiple_pdfs(pdf_paths):
    all_pages = []
    for pdf_path in pdf_paths: #this pdf paths from the (pdf_paths = list(PDF_FOLDER.glob("*.pdf")))
        pages = load_pdf(pdf_path)
        all_pages.extend(pages)
    return all_pages  #used to add individual elemeny in another list 
  


# ---------------- SPLIT PAGES ----------------

def split_pages(pages):  #here pages value from pages = load_pdf(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(pages)
    return chunks


# ---------------- CREATE VECTOR STORE ----------------

def create_vector_store(chunks): #here chucks value from chunks = split_pages(pages)
    vector_store = Chroma(
        collection_name="pdf_collection",
        embedding_function=embeddings,
        persist_directory=str(DB_FOLDER)   #persist directory basiccaly uuid is used for create new collection on every refresh
    )
    vector_store.add_documents(chunks)
    return vector_store

def load_vector_store(): #Load or connect to the existing persistent Chroma vector store.
    vector_store = Chroma(
        collection_name="pdf_collection",
        embedding_function=embeddings,
        persist_directory=str(DB_FOLDER)
    )
    return vector_store #existng collection access
  
# ---------------- RETRIEVE CHUNKS ----------------

def retrieve_chunks(question, selected_file): #question from st.chat_input and selected_file from st.sidebar.selectbox()
    search_kwargs = {"k": 4}
    if selected_file != "All documents":
        search_kwargs["filter"] = {
            "source": selected_file
        }
    retriever = st.session_state.vector_store.as_retriever(
        search_kwargs=search_kwargs
    )
    return retriever.invoke(question)


# ---------------- SYSTEM PROMPT ----------------

SYSTEM_PROMPT = """
You are a helpful assistant that answers questions about uploaded PDF documents.

Rules:
1. Use ONLY the provided context.
2. Do not use outside knowledge.
3. Add source references like [1], [2] after facts.
4. Mention the correct file name and page number.
5. If the answer is not available in the context, say:
   "I could not find this in the documents."
"""


# ---------------- FORMAT CONTEXT ----------------
## Formats retrieved document chunks with source, page number, and content
def format_context(docs):  #docs = retrieve_chunks(question, selected_file)
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 0) + 1
        parts.append(
            f"[{i}]\n"
            f"File: {source}\n"
            f"Page: {page}\n"
            f"Content: {doc.page_content}"
        )
    return "\n\n".join(parts) #retrieve chunks converted into text format


# ---------------- BUILD SOURCES ----------------
#purpose of this function is creating a list of source information from retrived chunks then we can show on ui 
def build_sources(docs):
    sources = []
    for i, doc in enumerate(docs, start=1):
        sources.append({
            "id": i, #chunk reference number
            "file": doc.metadata.get("source", "Unknown"), #pdf file name
            "page": doc.metadata.get("page", 0) + 1, #pdf page number
            "text": doc.page_content #retrieved chunk content
        })

    return sources


# ---------------- SHOW SOURCES ----------------
#retrieved pdf sources displayed on streamlit ui
def show_sources(sources): 
    with st.expander("View Sources"):
        for source in sources:
            st.markdown(
                f"**[{source['id']}] {source['file']} "
                f"- Page {source['page']}**"
            )
            st.caption(source["text"])


# ---------------- ASK MODEL ----------------

def ask_model(question, docs, history):
    context = format_context(docs)
    messages = [
        SystemMessage(
            content=SYSTEM_PROMPT + "\n\nContext:\n" + context
        )
    ]
    for message in history:
        if message["role"] == "user":
            messages.append(
                HumanMessage(content=message["content"])
            )
        elif message["role"] == "assistant":
            messages.append(
                AIMessage(content=message["content"])
            )

    messages.append(
        HumanMessage(content=question)
    )
    response = model.invoke(messages)
    if isinstance(response.content, str):
        return response.content
    if isinstance(response.content, list):
        return "\n".join(
            block["text"]
            for block in response.content
            if isinstance(block, dict) and "text" in block
        )
    return str(response.content)


# ---------------- STREAMLIT UI ----------------

st.set_page_config(
    page_title="Multiple PDF RAG",
    page_icon="📚",
    layout="wide"
)
st.title("📚 Multiple PDF RAG Chatbot")

# Initialize chat history

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- SIDEBAR ----------------

st.sidebar.header("Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# Save newly uploaded PDFs

if uploaded_files:
    save_uploaded_pdfs(uploaded_files)


# Get all saved PDF files

pdf_paths = list(PDF_FOLDER.glob("*.pdf"))


# Stop only if no PDF exists

if not pdf_paths:
    st.info("Please upload one or more PDF files from the sidebar.")
    st.stop()

# ---------------- FILE SELECTION ----------------

file_names = [pdf.name for pdf in pdf_paths]

selected_file = st.sidebar.selectbox(
    "Select document",
    ["All documents"] + file_names
)


# ---------------- INDEXING ----------------

# Load existing vector database
vector_store = load_vector_store()

# Load indexed files record
if INDEX_FILE.exists():
    with open(INDEX_FILE, "r") as f:
        try:
            indexed_files = json.load(f)
        except json.JSONDecodeError:
            indexed_files = {}
else:
    indexed_files = {}

# Check which PDFs are new or modified

with st.spinner("Checking PDF files..."):
    for pdf_path in pdf_paths:
        file_info = {
            "size": pdf_path.stat().st_size,
            "modified": pdf_path.stat().st_mtime_ns
        }
        saved_info = indexed_files.get(pdf_path.name, {})
        if saved_info.get("size") != file_info["size"] or \
           saved_info.get("modified") != file_info["modified"]:
            # Remove old chunks if this PDF was indexed earlier
            vector_store.delete(
                where={"source": pdf_path.name}
            )
            # Read PDF
            pages = load_pdf(pdf_path)
            # Split into chunks
            chunks = split_pages(pages)
            # Add new chunks to database
            if chunks:
                vector_store.add_documents(chunks)
            # Update record
            indexed_files[pdf_path.name] = {
                "size": file_info["size"],
                "modified": file_info["modified"],
                "pages": len(pages),
                "chunks": len(chunks)
            }

    # Save indexing record
    with open(INDEX_FILE, "w") as f:
        json.dump(indexed_files, f, indent=4)


# Store vector database in session
st.session_state.vector_store = vector_store

#calculate total pages from pdfs
st.session_state.num_pages = sum(
    indexed_files[p.name]["pages"] for p in pdf_paths
)

#calculate total chunks from pdfs
st.session_state.num_chunks = sum(
    indexed_files[p.name]["chunks"] for p in pdf_paths
)


# ---------------- DOCUMENT INFORMATION ----------------
st.sidebar.success(
    f"{st.session_state.num_pages} pages, "
    f"{st.session_state.num_chunks} chunks"
)


# ---------------- CLEAR CHAT ----------------

if st.sidebar.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()


# ---------------- DISPLAY CHAT HISTORY ----------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])


# ---------------- ASK QUESTION ----------------

question = st.chat_input("Ask a question about your PDFs...")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    docs = retrieve_chunks(question, selected_file)
    if not docs:
        answer = "I could not find this in the selected documents."
        sources = []
    else:
        with st.spinner("Generating answer..."):
            answer = ask_model(
                question,
                docs,
                st.session_state.messages
            )
            sources = build_sources(docs)
    with st.chat_message("assistant"):
        st.markdown(answer)
        if sources:
            show_sources(sources)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })