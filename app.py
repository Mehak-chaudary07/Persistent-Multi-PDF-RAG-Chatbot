import uuid  # Universally Unique Identifier
import streamlit as st
from pathlib import Path
import warnings
import json

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_chroma import Chroma
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_FOLDER = Path("uploaded_pdfs")
DB_FOLDER = Path("chroma_db")
INDEX_FILE = Path("indexed_files.json")


GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

PDF_FOLDER.mkdir(exist_ok=True)
DB_FOLDER.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore")


# ---------------- MODELS ----------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    google_api_key=GOOGLE_API_KEY
)

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.2,
    max_retries=2,
    google_api_key=GOOGLE_API_KEY
)


# ---------------- SAVE UPLOADED PDFs ----------------

def save_uploaded_pdfs(uploaded_files):
    for file in uploaded_files:
        file_path = PDF_FOLDER / file.name
        file_data = file.getvalue()

        if not file_path.exists() or file_path.read_bytes() != file_data:
            with open(file_path, "wb") as f:
                f.write(file_data)


# ---------------- LOAD PDF ----------------

def load_pdf(pdf_path):
    reader = PdfReader(str(pdf_path))
    pages = []

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text() or ""

        pages.append(
            Document(
                page_content=text,
                metadata={
                    "source": pdf_path.name,
                    "page": page_number
                }
            )
        )

    return pages


# ---------------- LOAD MULTIPLE PDFs ----------------

def load_multiple_pdfs(pdf_paths):
    all_pages = []

    for pdf_path in pdf_paths:
        pages = load_pdf(pdf_path)
        all_pages.extend(pages)

    return all_pages


# ---------------- SPLIT PAGES ----------------

def split_pages(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(pages)
    return chunks


# ---------------- CREATE VECTOR STORE ----------------

def create_vector_store(chunks):
    vector_store = Chroma(
        collection_name="pdf_collection",
        embedding_function=embeddings,
        persist_directory=str(DB_FOLDER)
    )

    vector_store.add_documents(chunks)
    return vector_store


def load_vector_store():
    vector_store = Chroma(
        collection_name="pdf_collection",
        embedding_function=embeddings,
        persist_directory=str(DB_FOLDER)
    )

    return vector_store


# ---------------- RETRIEVE CHUNKS ----------------

def retrieve_chunks(question, selected_file):
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

def format_context(docs):
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

    return "\n\n".join(parts)


# ---------------- BUILD SOURCES ----------------

def build_sources(docs):
    sources = []

    for i, doc in enumerate(docs, start=1):
        sources.append({
            "id": i,
            "file": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", 0) + 1,
            "text": doc.page_content
        })

    return sources


# ---------------- SHOW SOURCES ----------------

def show_sources(sources):
    with st.expander("📚 View Sources"):
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


# =====================================================
#                    STREAMLIT UI
# =====================================================

st.set_page_config(
    page_title="Multiple PDF RAG",
    page_icon="📚",
    layout="wide"
)


# ---------------- CUSTOM BLACK + BLUE THEME ----------------

st.markdown("""
<style>

/* Main application background */

.stApp {
    background-color: #080D14;
    color: #E6EDF6;
}

/* Main content area */

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background-color: #0D1520;
    border-right: 1px solid #1B3042;
}

[data-testid="stSidebar"] > div:first-child {
    background-color: #0D1520;
}

/* Sidebar text */

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #D5E2EF;
}

/* Main headings */

h1, h2, h3 {
    color: #EAF6FF !important;
    font-weight: 700 !important;
}

/* Main title */

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #EAF6FF;
    margin-bottom: 5px;
}

.main-subtitle {
    color: #8DA5BA;
    font-size: 15px;
    margin-bottom: 25px;
}

/* Header card */

.hero-card {
    background: linear-gradient(
        120deg,
        #102638,
        #10202F,
        #111A27
    );
    border: 1px solid #1D4B66;
    border-radius: 16px;
    padding: 25px;
    margin-bottom: 25px;
}

.hero-card h2 {
    color: #E8F7FF;
    margin-bottom: 8px;
    font-size: 25px;
}

.hero-card p {
    color: #A8C3D7;
    font-size: 14px;
    margin-bottom: 0;
}

/* Sidebar headings */

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #48C8E8 !important;
}

/* Buttons */

.stButton > button {
    background: linear-gradient(
        120deg,
        #087E9B,
        #126DAB
    );
    color: white;
    border: 1px solid #249CC0;
    border-radius: 9px;
    font-weight: 600;
    padding: 9px 18px;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(
        120deg,
        #0799B8,
        #1688CB
    );
    border: 1px solid #54D8F2;
    color: white;
}

/* File uploader */

[data-testid="stFileUploader"] {
    background-color: #111D2A;
    border: 1px dashed #27617D;
    border-radius: 12px;
    padding: 12px;
}

/* Select box */

[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: #142333;
    border: 1px solid #28536D;
    color: #E6F5FF;
    border-radius: 9px;
}

/* Metrics */

[data-testid="stMetric"] {
    background-color: #111D2A;
    border: 1px solid #1D4058;
    border-radius: 12px;
    padding: 15px;
}

[data-testid="stMetricLabel"] {
    color: #93B3C9 !important;
}

[data-testid="stMetricValue"] {
    color: #55D5F2 !important;
}

/* Chat messages */

[data-testid="stChatMessage"] {
    background-color: #111C29;
    border: 1px solid #1D3549;
    border-radius: 14px;
    padding: 15px;
    margin-bottom: 12px;
}

/* Chat input */

[data-testid="stChatInput"] {
    background-color: #101B28;
    border: 1px solid #28536D;
    border-radius: 12px;
}

[data-testid="stChatInput"] textarea {
    color: #EAF6FF !important;
}

/* Expander / Sources */

[data-testid="stExpander"] {
    background-color: #111D2A;
    border: 1px solid #24516B;
    border-radius: 10px;
    margin-top: 10px;
}

[data-testid="stExpander"] summary {
    color: #59D5F0;
}

/* Divider */

hr {
    border-color: #20394C;
}

/* Caption */

.stCaption {
    color: #8DA8BD !important;
}

/* Info messages */

[data-testid="stAlert"] {
    background-color: #102638;
    border: 1px solid #24617C;
    color: #D9F3FF;
    border-radius: 10px;
}

/* Scrollbar */

::-webkit-scrollbar {
    width: 7px;
}

::-webkit-scrollbar-track {
    background: #080D14;
}

::-webkit-scrollbar-thumb {
    background: #24516B;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #1688AA;
}

</style>
""", unsafe_allow_html=True)


# ---------------- INITIALIZE CHAT HISTORY ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- SIDEBAR ----------------

st.sidebar.markdown("""
<div style="padding: 10px 0 20px 0;">
    <h2 style="color:#50D5F2; margin-bottom:0;">
        📚 PDF Assistant
    </h2>
    <p style="color:#8DA8BD; font-size:13px;">
        Your personal document chatbot
    </p>
</div>
""", unsafe_allow_html=True)

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
    st.markdown("""
    <div class="hero-card">
        <h2>Welcome to your PDF Assistant 👋</h2>
        <p>
            Upload one or more PDF documents from the sidebar
            to start asking questions and get answers from your files.
        </p>
    </div>
    """, unsafe_allow_html=True)

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


# Calculate total pages from PDFs

st.session_state.num_pages = sum(
    indexed_files[p.name]["pages"] for p in pdf_paths
)


# Calculate total chunks from PDFs

st.session_state.num_chunks = sum(
    indexed_files[p.name]["chunks"] for p in pdf_paths
)


# ---------------- DOCUMENT INFORMATION ----------------

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Document Statistics")

col1, col2 = st.sidebar.columns(2)

with col1:
    st.metric(
        "Pages",
        st.session_state.num_pages
    )

with col2:
    st.metric(
        "Chunks",
        st.session_state.num_chunks
    )

st.sidebar.caption(
    f"📁 {len(pdf_paths)} PDF document(s) available"
)


# ---------------- CLEAR CHAT ----------------

st.sidebar.markdown("---")

if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()


# ---------------- MAIN HEADER ----------------

st.markdown("""
<div class="hero-card">
    <h2>📚 Multiple PDF RAG Chatbot</h2>
    <p>
        Ask questions from your uploaded PDF documents.
        Get context-based answers with document references and page numbers.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- DISPLAY CHAT HISTORY ----------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):
            show_sources(message["sources"])


# ---------------- ASK QUESTION ----------------

question = st.chat_input(
    "Ask anything about your uploaded PDFs..."
)

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
