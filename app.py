import sys


from flask import Flask, request, jsonify, send_file, Response, stream_with_context
from flask_cors import CORS
import os
import shutil
import importlib
from typing import Any
from io import BytesIO
from functools import lru_cache


def _safe_reconfigure_text_stream(stream: Any) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding='utf-8', errors='replace', line_buffering=True)


_safe_reconfigure_text_stream(sys.stdout)
_safe_reconfigure_text_stream(sys.stderr)

# LangChain and AI imports
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Cross-encoder for re-ranking
from sentence_transformers import CrossEncoder

# Conversation memory
from collections import defaultdict
import json
from datetime import datetime
import uuid

# Import markdown-based PDF extractor
from pdf_markdown_extractor import MarkdownPDFDirectoryLoader, extract_pdfs_from_directory

# Multi-agent agentic RAG system
try:
    from agentic_router import AgenticRAGSystem
    HAS_AGENTIC_ROUTER = True
except ImportError:
    HAS_AGENTIC_ROUTER = False
    print("⚠ LangGraph not available. Agentic routing disabled. Running in standard RAG mode.")

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_HOST = os.getenv("TTM_ASK_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("TTM_ASK_PORT", "5000"))
APP_DEBUG = os.getenv("TTM_ASK_DEBUG", "0") == "1"
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "").strip()
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
UI_FILE = os.path.join(BASE_DIR, "ttm_ask.html")

cors_origins = os.getenv("TTM_ASK_CORS_ORIGINS", "*")
if cors_origins.strip() == "*":
    CORS(app)
else:
    CORS(app, resources={r"/*": {"origins": [origin.strip() for origin in cors_origins.split(",") if origin.strip()]}})

# Configuration
DOCS_DIR = os.path.join(BASE_DIR, "drive_docs")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
DOC_LIST_CACHE = os.path.join(DB_DIR, "doc_list.json")
INDEX_META_CACHE = os.path.join(DB_DIR, "index_meta.json")
PDF_EXTRACTION_MAX_WORKERS = int(os.getenv("PDF_EXTRACTION_MAX_WORKERS", "4"))  # Parallel PDF extraction threads
MAX_EMBEDDING_CHUNK_SIZE = int(os.getenv("MAX_EMBEDDING_CHUNK_SIZE", "2000"))  # nomic-embed-text supports up to 8192 tokens; 2000 chars ≈ 500 tokens, safely under the limit
RETRIEVAL_MMR_K = int(os.getenv("RETRIEVAL_MMR_K", "8"))
RETRIEVAL_MMR_FETCH_K = int(os.getenv("RETRIEVAL_MMR_FETCH_K", "20"))
RETRIEVAL_RERANK_TOP_K = int(os.getenv("RETRIEVAL_RERANK_TOP_K", "4"))
RETRIEVAL_EXTRA_REF_K = int(os.getenv("RETRIEVAL_EXTRA_REF_K", "4"))
RETRIEVAL_FALLBACK_K = int(os.getenv("RETRIEVAL_FALLBACK_K", "20"))
FAMILY_VECTOR_SEARCH_K = int(os.getenv("FAMILY_VECTOR_SEARCH_K", "20"))
DIRECT_TABLE_VECTOR_K = int(os.getenv("DIRECT_TABLE_VECTOR_K", "12"))

# ============================================================================
# CONVERSATION MEMORY STORAGE
# ============================================================================
# Store conversations in memory (format: {session_id: [{"role": "human"|"ai", "content": str, "timestamp": str}]})
conversation_memory = defaultdict(list)
MAX_CONVERSATION_HISTORY = 10  # Keep last N messages per conversation
MAX_CONVERSATIONS = 100  # Maximum number of active conversations to keep

# ============================================================================
# 1. EMBEDDINGS INITIALIZATION
# ============================================================================
# Initialize Embeddings (This converts text to searchable math)
# Using Gemini's embedding model for managed cloud inference.
print(f"Loading embedding model ({GEMINI_EMBEDDING_MODEL})...")
print("Note: Requires GOOGLE_API_KEY to be set in the environment.")


def model_response_to_text(response: Any) -> str:
    if isinstance(response, BaseMessage):
        return str(getattr(response, "content", ""))
    return str(response)

def initialize_embeddings():
    print("Loading Google Gemini embedding model...")
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

try:
    embeddings = initialize_embeddings()
except Exception as e:
    print(f"\n{e}")
    print("\nFalling back to lightweight model for now...")
    print("To use Gemini embeddings, set GOOGLE_API_KEY and restart.\n")
    # Fallback to lightweight model if Ollama setup fails
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("⚠ Using lightweight embedding model (all-MiniLM-L6-v2) - results may be less accurate")


# Helper function to recursively find all PDFs
def find_all_pdfs(directory):
    """Recursively find all PDF files in a directory tree."""
    from pathlib import Path
    pdf_files = []
    for pdf_path in Path(directory).rglob("*.pdf"):
        pdf_files.append(str(pdf_path))
    return pdf_files


def semantic_chunk_markdown_documents(docs, chunk_size=1000, chunk_overlap=200):
    """
    Semantically chunk markdown documents using MarkdownHeaderTextSplitter.
    
    This splitter respects markdown structure (headers) rather than blindly cutting at character counts.
    Tables, lists, and sections stay intact within semantic chunks.
    
    Args:
        docs: List of LangChain Document objects with markdown content
        chunk_size: Max size of chunks within each header section
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of semantically chunked documents
    """
    # Define header hierarchy for markdown structure recognition
    # PyMuPDF4LLM extracts PDFs with these levels of headers
    headers_to_split_on = [
        ("#", "Header 1"),      # Main sections
        ("##", "Header 2"),     # Subsections
        ("###", "Header 3"),    # Sub-subsections
    ]
    
    # Create markdown-aware header splitter
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        return_each_line=False
    )
    
    # Fallback for non-markdown content
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "| ", " ", ""]
    )
    
    all_splits = []
    
    for doc in docs:
        content = doc.page_content
        metadata = doc.metadata
        
        # First attempt: split by markdown headers
        try:
            # MarkdownHeaderTextSplitter returns chunks with header metadata
            header_splits = markdown_splitter.split_text(content)
            
            if header_splits:
                # Process each header-based chunk
                for split in header_splits:
                    # Preserve original metadata and add header context
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update(split.metadata)  # Add header info
                    chunk_metadata["chunking_method"] = "semantic_markdown"
                    
                    chunk_doc = Document(
                        page_content=split.page_content,
                        metadata=chunk_metadata
                    )
                    all_splits.append(chunk_doc)
                continue
        except Exception as e:
            print(f"  ⚠ Header splitting failed for chunk, falling back to recursive: {str(e)[:50]}")
        
        # Fallback: recursive character splitting for content without clear headers
        try:
            recursive_splits = recursive_splitter.split_documents([doc])
            for split in recursive_splits:
                split.metadata["chunking_method"] = "recursive_fallback"
            all_splits.extend(recursive_splits)
        except Exception as e:
            print(f"  ⚠ Recursive splitting failed, keeping original chunk: {str(e)[:50]}")
            doc.metadata["chunking_method"] = "original"
            all_splits.append(doc)
    
    return all_splits



def build_index_meta(directory):
    """Build a deterministic fingerprint of current PDF sources."""
    from pathlib import Path
    entries = []
    if not os.path.exists(directory):
        return {"files": entries}

    for pdf_path in sorted(Path(directory).rglob("*.pdf")):
        try:
            stat = pdf_path.stat()
            entries.append({
                "path": str(pdf_path).replace("\\", "/").lower(),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns)
            })
        except OSError:
            continue
    return {"files": entries}

import json
from threading import Lock
from langchain_community.document_loaders import PyPDFLoader


@lru_cache(maxsize=50)
def get_cached_pdf_pages(pdf_path):
    """Cache parsed PDFs in memory to avoid repeated disk reads during queries."""
    return PyPDFLoader(pdf_path).load()


# Track loaded documents - initialized as empty, populated in initialize_or_reload_index()
loaded_documents = []

current_index_meta = None
vectorstore = None
retriever = None
rag_chain = None
reranker = None
agentic_system = None  # Multi-agent RAG system (LangGraph-based)
index_lock = Lock()


def read_cached_index_meta():
    if not os.path.exists(INDEX_META_CACHE):
        return None
    try:
        with open(INDEX_META_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def initialize_reranker():
    """
    Initialize the cross-encoder re-ranker for two-stage retrieval.
    
    The cross-encoder model (BAAI/bge-reranker-base) is more accurate than dense
    vector search but slower. It's used as a secondary ranking stage after MMR
    vector retrieval to improve precision.
    
    Returns:
        CrossEncoder instance or None if initialization fails
    """
    try:
        print("Loading cross-encoder re-ranker model (BAAI/bge-reranker-base)...")
        reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
        print("✓ Cross-encoder re-ranker loaded successfully")
        return reranker
    except Exception as e:
        print(f"⚠ Failed to load cross-encoder: {str(e)}")
        print("  Continuing with vector search only (less accurate but still functional)")
        return None


def rerank_retrieved_docs(query, documents, reranker, top_k=5):
    """
    Re-rank retrieved documents using a cross-encoder model.
    
    Two-stage retrieval process:
    1. First stage (dense): MMR retrieves ~12 candidate chunks (fast, broad)
    2. Second stage (cross-encoder): Re-rank to get top K most relevant (slower, precise)
    
    Args:
        query: The user's query string
        documents: List of Document objects from MMR retrieval
        reranker: CrossEncoder model instance
        top_k: Number of top documents to return after re-ranking (default: 5)
        
    Returns:
        List of re-ranked documents (top_k), sorted by relevance score descending
    """
    if not reranker or not documents:
        return documents[:top_k] if documents else []
    
    try:
        # Prepare pairs for cross-encoder: (query, document_text)
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Score all pairs
        scores = reranker.predict(pairs)
        
        # Create (score, doc) tuples and sort by score descending
        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Return top_k documents with scores added to metadata
        reranked = []
        for score, doc in scored_docs[:top_k]:
            doc.metadata["reranker_score"] = float(score)
            reranked.append(doc)
        
        return reranked
    except Exception as e:
        print(f"⚠ Re-ranking error: {str(e)}")
        return documents[:top_k]


def create_retriever(active_vectorstore):
    if not active_vectorstore:
        return None
    return active_vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVAL_MMR_K, "fetch_k": RETRIEVAL_MMR_FETCH_K, "lambda_mult": 0.7}
    )




def create_rag_chain(active_retriever):
    if not active_retriever:
        return None
    return (
        {"context": active_retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def initialize_or_reload_index(force_rebuild: bool = False):
    global loaded_documents, current_index_meta, vectorstore, retriever, rag_chain, reranker, agentic_system

    with index_lock:
        current_index_meta = build_index_meta(DOCS_DIR)
        cached_index_meta = read_cached_index_meta()

        print("Checking startup mode...")

        cache_is_current = (
            not force_rebuild
            and os.path.exists(DB_DIR)
            and os.path.exists(DOC_LIST_CACHE)
            and os.path.exists(INDEX_META_CACHE)
            and cached_index_meta == current_index_meta
        )

        active_loaded_documents = []
        active_vectorstore = None

        if cache_is_current:
            print("Loading existing ChromaDB and document list (fast start)...")
            active_vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
            with open(DOC_LIST_CACHE, "r", encoding="utf-8") as f:
                active_loaded_documents = json.load(f)
            print(f"Ready — {len(active_loaded_documents)} documents available.")
        else:
            if os.path.exists(DB_DIR):
                print("Detected document changes. Rebuilding index from PDFs...")

            print(f"Checking for PDFs in {DOCS_DIR}...")
            pdf_files = find_all_pdfs(DOCS_DIR) if os.path.exists(DOCS_DIR) else []
            print(f"Found {len(pdf_files)} PDF files")

            if len(pdf_files) > 0:
                print("Indexing documents... this may take a moment.")
                try:
                    # The new loader provides block-level chunks with precise metadata
                    loader = MarkdownPDFDirectoryLoader(
                        DOCS_DIR,
                        recursive=True,
                        max_workers=PDF_EXTRACTION_MAX_WORKERS
                    )
                    docs = loader.load()
                except Exception as e:
                    print(f"  ✗ Top-level extraction failed: {str(e)}")
                    docs = []

                if docs:
                    # The new loader gives pre-chunked documents, so we can process them directly.
                    # First, populate the list of unique source documents for the UI.
                    seen_paths = set()
                    for doc in docs:
                        source_path = doc.metadata.get("source", "")
                        if source_path not in seen_paths:
                            seen_paths.add(source_path)
                            active_loaded_documents.append({"title": os.path.basename(source_path), "path": source_path})
                    
                    print(f"Loaded {len(docs)} chunks from {len(active_loaded_documents)} files. Index will be rebuilt.")

                    if os.path.exists(DB_DIR):
                        shutil.rmtree(DB_DIR, ignore_errors=True)
                    
                    print("Building ChromaDB from block-level chunks...")
                    
                    # Filter out any chunks that are too long for the embedding model.
                    final_chunks = []
                    overlength_count = 0
                    for doc in docs:
                        if len(doc.page_content) > MAX_EMBEDDING_CHUNK_SIZE:
                            overlength_count += 1
                            continue
                        final_chunks.append(doc)

                    if overlength_count > 0:
                        print(f"  ⚠ Skipped {overlength_count} overlength chunks.")
                    
                    print(f"Embedding {len(final_chunks)} chunks into ChromaDB...")
                    
                    # Create and populate the vectorstore in batches
                    active_vectorstore = Chroma(embedding_function=embeddings, persist_directory=DB_DIR)
                    batch_size = 500
                    total_batches = (len(final_chunks) // batch_size) + 1
                    
                    for i in range(0, len(final_chunks), batch_size):
                        batch = final_chunks[i : i + batch_size]
                        print(f"  Embedding batch {i//batch_size + 1} of {total_batches}...")
                        active_vectorstore.add_documents(batch)

                    os.makedirs(DB_DIR, exist_ok=True)
                    with open(DOC_LIST_CACHE, "w", encoding="utf-8") as f:
                        json.dump(active_loaded_documents, f, indent=2)
                    with open(INDEX_META_CACHE, "w", encoding="utf-8") as f:
                        json.dump(current_index_meta, f, indent=2)
                    print("Indexing complete!")
                else:
                    print("ERROR: No documents could be loaded from PDFs.")
            else:
                print("No PDFs found. Loading existing database if available...")
                if os.path.exists(DB_DIR):
                    active_vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
                else:
                    print("ERROR: No PDFs found and no existing database. Please check drive_docs folder.")

        loaded_documents = active_loaded_documents
        vectorstore = active_vectorstore
        retriever = create_retriever(active_vectorstore)
        rag_chain = create_rag_chain(retriever)
        
        reranker = initialize_reranker()
        
        if HAS_AGENTIC_ROUTER and active_vectorstore and reranker:
            try:
                from agentic_router import AgenticRAGSystem
                agentic_system = AgenticRAGSystem(llm=llm, vectorstore=active_vectorstore, reranker=reranker)
                print("✓ Multi-agent agentic RAG system initialized (LangGraph-based)")
            except Exception as e:
                agentic_system = None
                print(f"⚠ Failed to initialize agentic system: {e}")
        else:
            agentic_system = None
        
        return {
            "document_count": len(loaded_documents),
            "pdf_count": len(current_index_meta.get("files", [])) if current_index_meta else 0,
            "cache_current": cache_is_current,
            "ready": vectorstore is not None,
        }

# 4. Create the RAG Prompt and Chain
system_prompt = (
    "You are an expert Australian Traffic Management assistant. "
    "Use the following retrieved context chunks to answer the question accurately. "
    "The chunks are semantically organized by document structure (headers, sections) rather than arbitrary character boundaries. "
    "This means chunks with shared headers maintain semantic coherence — tables are kept intact, lists are complete, and sections are whole. "
    "Read all provided chunks carefully and use their structural relationships to inform your reasoning. "
    "Extract facts explicitly stated in the user's question. "
    "Do not assume missing operational inputs (such as speed, traffic volume, lane status, or clearance bucket). "
    "If critical inputs are missing for a definitive recommendation, say so explicitly and ask focused follow-up questions. "
    "When inputs are missing, provide conditional guidance as IF/THEN bullets or a compact table instead of choosing one option. "
    "The document content has been extracted with markdown formatting that preserves table structures and headers. "
    "Tables appear as markdown tables (with | separators) and maintain their original layout. "
    "Markdown headers (indicated by #, ##, ###) help you understand the document hierarchy and context. "
    "If a specific table or section is referenced in the question (e.g. 'Table 5.1'), look for that data "
    "in markdown table format or in the context, even if the label is not explicitly present. "
    "Only state that information is missing if it genuinely cannot be found anywhere in the provided context. "
    "Do not invent or guess information. Keep answers clear, professional, and well-structured. "
    "When the answer is comparative or structured, format it as a markdown table.\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# gemini-1.5-flash is incredibly fast and perfect for RAG tasks like yours
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# ============================================================================
# CONVERSATION MEMORY FUNCTIONS
# ============================================================================

def create_session_id():
    """Generate a unique session ID for a new conversation."""
    return str(uuid.uuid4())[:12]


def add_message_to_conversation(session_id: str, role: str, content: str):
    """
    Add a message to the conversation history.
    
    Args:
        session_id: Unique identifier for the conversation
        role: Either "human" or "ai"
        content: The message content
    """
    # Trim old messages if we exceed max history
    if len(conversation_memory[session_id]) >= MAX_CONVERSATION_HISTORY:
        conversation_memory[session_id].pop(0)
    
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    }
    conversation_memory[session_id].append(message)
    
    # Trim conversations if we have too many
    if len(conversation_memory) > MAX_CONVERSATIONS:
        # Remove oldest conversation (by first message timestamp)
        oldest_session = min(conversation_memory.keys(), 
                            key=lambda sid: conversation_memory[sid][0].get("timestamp", ""))
        del conversation_memory[oldest_session]


def get_conversation_history(session_id: str) -> list:
    """
    Get the full conversation history for a session.
    
    Args:
        session_id: Unique identifier for the conversation
        
    Returns:
        List of message dictionaries with role, content, timestamp
    """
    return conversation_memory.get(session_id, [])


def format_conversation_history_for_prompt(session_id: str) -> str:
    """
    Format conversation history into a string for the prompt context.
    
    Args:
        session_id: Unique identifier for the conversation
        
    Returns:
        Formatted string with previous messages (excludes the current question)
    """
    history = get_conversation_history(session_id)
    if not history:
        return ""
    
    # Build context from previous messages (skip the most recent if it's a human message)
    messages = history[:-1] if history and history[-1].get("role") == "human" else history
    if not messages:
        return ""
    
    formatted = "\n\nPrevious conversation context:\n"
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        formatted += f"{role}: {content}\n"
    
    return formatted


def clear_conversation(session_id: str):
    """
    Clear the conversation history for a session.
    
    Args:
        session_id: Unique identifier for the conversation
    """
    if session_id in conversation_memory:
        del conversation_memory[session_id]


def list_conversations() -> dict:
    """
    Get summary of all active conversations.
    
    Returns:
        Dictionary with session_id as key and metadata as value
    """
    result = {}
    for session_id, messages in conversation_memory.items():
        if messages:
            result[session_id] = {
                "message_count": len(messages),
                "created_at": messages[0].get("timestamp", ""),
                "last_message_at": messages[-1].get("timestamp", ""),
                "last_message_role": messages[-1].get("role", "unknown")
            }
    return result

import re as _re


def normalize_source_path(path: str):
    return os.path.normpath(path or "").replace("\\", "/").lower()


JURISDICTION_ORDER = [
    "Federal / National",
    "Queensland",
    "New South Wales",
    "Victoria",
    "Western Australia",
    "South Australia",
    "Tasmania",
    "Australian Capital Territory",
    "Northern Territory",
    "Other"
]

FAMILY_DISPLAY_NAMES = {
    "agttm": "AGTTM",
    "qgttm": "QGTTM",
}


def classify_document_jurisdiction(path: str, title: str = ""):
    text = f"{title} {path}".lower()

    if any(token in text for token in [
        "austroads", "austroad", "safeworkaustralia", "safe work australia",
        "infrastructure.gov.au", "ntc.gov.au", "federal", "national", "federal_national"
    ]):
        return "Federal / National"
    if any(token in text for token in ["qgttm", "queensland", "tmr.qld.gov.au", "mutcd", " qld ", "/qld/", "_online_discovered/queensland/"]):
        return "Queensland"
    if any(token in text for token in ["nsw", "transport.nsw.gov.au", "tfnsw", "new south wales", "new_south_wales"]):
        return "New South Wales"
    if any(token in text for token in ["victoria", "vic.gov.au", "vicroads", "_online_discovered/victoria/"]):
        return "Victoria"
    if any(token in text for token in ["wa.gov.au", "mainroads.wa.gov.au", "western australia", "western_australia"]):
        return "Western Australia"
    if any(token in text for token in ["sa.gov.au", "south australia", "south_australia", "dpti", "dit.sa.gov.au"]):
        return "South Australia"
    if any(token in text for token in ["tasmania", "transport.tas.gov.au", "_online_discovered/tasmania/"]):
        return "Tasmania"
    if any(token in text for token in ["canberra", "cityservices.act.gov.au", "australian capital territory", "australian_capital_territory"]):
        return "Australian Capital Territory"
    if any(token in text for token in ["northern territory", "northern_territory", "nt.gov.au"]):
        return "Northern Territory"
    return "Other"


def jurisdiction_sort_key(name: str):
    try:
        return JURISDICTION_ORDER.index(name)
    except ValueError:
        return len(JURISDICTION_ORDER)


def source_matches_family(source_path: str, family: str):
    src = (source_path or "").lower()
    fam = (family or "").lower()
    if fam == "agttm":
        return "agttm" in src or "guide_to_temporary_traffic_management" in src
    if fam == "qgttm":
        return "qgttm" in src or "queensland guide to temporary traffic management" in src
    return False


def extract_named_references(text: str):
    if not text:
        return []

    patterns = [
        r"\btable\s+\d+(?:\.\d+)*[a-z]?\b",
        r"\bfigure\s+\d+(?:\.\d+)*[a-z]?\b",
        r"\bsection\s+\d+(?:\.\d+)*[a-z]?\b",
        r"\bpart\s+\d+[a-z]?\b",
    ]
    refs = []
    seen = set()
    for pattern in patterns:
        for match in _re.finditer(pattern, text, _re.IGNORECASE):
            ref = match.group(0).strip()
            normalized = ref.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            refs.append(ref.title())
            if len(refs) >= 8:
                return refs
    return refs


@lru_cache(maxsize=256)
def ocr_page_reference_scan(source_path: str, zero_based_page: int):
    """OCR a PDF page and extract figure/table/section refs when text extraction is poor.

    This is optional and only works when both PyMuPDF and Tesseract/pytesseract are available.
    """
    if not source_path or zero_based_page is None or zero_based_page < 0:
        return []

    try:
        import fitz
    except ImportError:
        return []

    try:
        pytesseract = importlib.import_module("pytesseract")
        image_module = importlib.import_module("PIL.Image")
    except ImportError:
        return []

    tesseract_cmd = shutil.which("tesseract")
    if not tesseract_cmd:
        return []

    try:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        pdf_doc = fitz.open(source_path)
        if zero_based_page >= pdf_doc.page_count:
            pdf_doc.close()
            return []
        page = pdf_doc.load_page(zero_based_page)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
        image = image_module.open(BytesIO(pix.tobytes("png")))
        ocr_text = pytesseract.image_to_string(image, config="--psm 6")
        pdf_doc.close()
    except Exception:
        return []

    return extract_named_references(ocr_text)


def extract_document_references(doc: Document):
    refs = extract_named_references(doc.page_content or "")
    if refs:
        return refs

    source_path = doc.metadata.get("source", "")
    raw_page = doc.metadata.get("page")
    zero_based_page = raw_page if isinstance(raw_page, int) else None
    if zero_based_page is None and isinstance(raw_page, str) and raw_page.strip().isdigit():
        zero_based_page = int(raw_page.strip())
    if zero_based_page is None:
        return []

    return ocr_page_reference_scan(source_path, zero_based_page)


def annotate_docs(docs, family: str | None = None, force_refs: list[str] | None = None):
    label = FAMILY_DISPLAY_NAMES.get((family or "").lower()) if family else None
    for doc in docs:
        if label:
            doc.metadata["family"] = label
        refs = force_refs if force_refs is not None else extract_document_references(doc)
        if refs:
            doc.metadata["refs"] = refs[:6]
    return docs


def collect_reference_summary(docs):
    refs = []
    seen = set()
    for doc in docs:
        for ref in doc.metadata.get("refs", []) or extract_document_references(doc):
            key = ref.lower()
            if key in seen:
                continue
            seen.add(key)
            refs.append(ref)
            if len(refs) >= 6:
                return refs
    return refs


def format_reference_list(refs: list[str]):
    return ", ".join(refs) if refs else "none found"


def parse_question_facts(question: str):
    """Extract structured signals from natural-language field questions."""
    q = question or ""
    q_lower = q.lower()

    def _first_float(patterns):
        for pattern in patterns:
            m = _re.search(pattern, q_lower, _re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    continue
        return None

    def _first_int(patterns):
        for pattern in patterns:
            m = _re.search(pattern, q_lower, _re.IGNORECASE)
            if m:
                raw = m.group(1).replace(",", "").strip()
                if raw.isdigit():
                    return int(raw)
        return None

    speed_kmh = _first_int([
        r"(?:speed|posted speed|limit)\s*(?:is|of|=)?\s*(\d{2,3})\s*(?:km\/?h|kph)",
        r"(\d{2,3})\s*(?:km\/?h|kph)"
    ])

    traffic_vpd = _first_int([
        r"(?:traffic\s*volume|volume|vpd|vehicles\s*per\s*day)\s*(?:is|of|=)?\s*([\d,]+)",
        r"([\d,]+)\s*(?:vpd|vehicles\s*per\s*day|veh\/?day)"
    ])

    road_width_m = _first_float([
        r"(\d+(?:\.\d+)?)\s*(?:m|meter|metre)s?\s*(?:wide\s*)?road",
        r"road\s*(?:width|is)\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(?:m|meter|metre)s?"
    ])

    excavation_depth_m = _first_float([
        r"(?:depth|deep|excavation\s*depth)\s*(?:is|of|=)?\s*(\d+(?:\.\d+)?)\s*(?:m|meter|metre)s?",
        r"(\d+(?:\.\d+)?)\s*(?:m|meter|metre)s?\s*deep"
    ])

    clearance_to_kerb_m = _first_float([
        r"(?:clearance|offset|distance)\s*(?:to|from)?\s*(?:the\s*)?(?:kerb|curb|excavation)\s*(?:is|of|=)?\s*(\d+(?:\.\d+)?)\s*(?:m|meter|metre)s?",
        r"(\d+(?:\.\d+)?)\s*(?:m|meter|metre)s?\s*from\s*(?:the\s*)?(?:kerb|curb)"
    ])

    asks_delineation = any(token in q_lower for token in [
        "delineation", "delineator", "cone", "cones", "bollard", "barrier", "taper", "what do i need"
    ])
    asks_excavation = any(token in q_lower for token in [
        "excavation", "trench", "pit", "hole"
    ])

    return {
        "speed_kmh": speed_kmh,
        "traffic_vpd": traffic_vpd,
        "road_width_m": road_width_m,
        "excavation_depth_m": excavation_depth_m,
        "clearance_to_kerb_m": clearance_to_kerb_m,
        "asks_delineation": asks_delineation,
        "asks_excavation": asks_excavation,
    }


def missing_critical_inputs(facts: dict, selected_document: str | None = None):
    """Return critical missing fields for deterministic safety recommendations."""
    missing = []
    if facts.get("asks_delineation") and facts.get("asks_excavation"):
        if facts.get("speed_kmh") is None:
            missing.append("posted/operating speed (km/h)")
        if facts.get("traffic_vpd") is None:
            missing.append("traffic volume (VPD or lane flow)")
        if not selected_document:
            missing.append("jurisdiction/manual to apply (e.g. AGTTM vs QGTTM)")
    return missing


def question_analysis(question: str, selected_document: str | None = None):
    facts = parse_question_facts(question)
    missing = missing_critical_inputs(facts, selected_document=selected_document)
    follow_up_questions = build_follow_up_questions(facts, missing, selected_document=selected_document)
    return {
        "facts": facts,
        "missing_critical_inputs": missing,
        "is_ambiguous": bool(missing),
        "follow_up_questions": follow_up_questions,
    }


def build_follow_up_questions(facts: dict, missing: list[str], selected_document: str | None = None):
    """Return concise user-facing follow-up questions to complete missing inputs."""
    prompts = []

    if "posted/operating speed (km/h)" in missing:
        prompts.append("What is the posted or operating speed at the worksite (km/h)?")

    if "traffic volume (VPD or lane flow)" in missing:
        prompts.append("What is the traffic volume (VPD), or per-lane flow during works?")

    if "jurisdiction/manual to apply (e.g. AGTTM vs QGTTM)" in missing:
        prompts.append("Which manual should I apply for this site: AGTTM or QGTTM (or specific state guide)?")

    # Helpful extras: if excavation is mentioned but details are not explicit, ask once.
    if facts.get("asks_excavation") and facts.get("excavation_depth_m") is None:
        prompts.append("What is the excavation depth (m)?")

    if facts.get("asks_excavation") and facts.get("clearance_to_kerb_m") is None:
        prompts.append("What is the minimum clearance from kerb/traffic lane to excavation edge (m)?")

    if facts.get("asks_excavation") and facts.get("road_width_m") is None:
        prompts.append("What is the road width at the work area (m)?")

    # Keep suggestions compact and actionable.
    return prompts[:5]


def build_clarification_first_answer(question: str, selected_document: str | None = None, strict_mode: bool = False):
    """Produce a safer answer style for ambiguous field questions with missing inputs."""
    source_docs = hybrid_retrieve(question, selected_document=selected_document)
    context_text = format_docs(source_docs)
    analysis = question_analysis(question, selected_document=selected_document)
    facts = analysis["facts"]
    missing = analysis["missing_critical_inputs"]

    if not missing:
        return None

    # In non-strict mode, only intercept when this looks like an excavation delineation question.
    if not strict_mode and not (facts.get("asks_delineation") and facts.get("asks_excavation")):
        return None

    facts_lines = []
    for k, v in facts.items():
        if v is None or isinstance(v, bool):
            continue
        facts_lines.append(f"- {k}: {v}")
    if not facts_lines:
        facts_lines.append("- No numeric fields confidently detected")

    facts_text = "\n".join(facts_lines)
    missing_lines = "\n".join(f"- {item}" for item in missing)

    clarification_prompt = (
        "You are an Australian traffic management specialist. "
        "Write a clarification-first answer that is useful but does not guess missing critical inputs.\n\n"
        f"User question: {question}\n\n"
        "Detected facts from the question:\n"
        f"{facts_text}\n\n"
        "Critical missing inputs:\n"
        f"{missing_lines}\n\n"
        "Use ONLY the context below.\n"
        "Output format:\n"
        "1) 'What I understood' bullet list.\n"
        "2) 'What I still need' bullet list (keep concise).\n"
        "3) 'Conditional guidance' table with clear IF/THEN rows using placeholders like [speed] and [volume] if needed.\n"
        "4) End with 2-4 direct follow-up questions for the site engineer.\n"
        "Do not pick a single option unless context supports it with provided inputs.\n\n"
        f"Context:\n{context_text}"
    )

    answer = llm.invoke(clarification_prompt)
    return model_response_to_text(answer), source_docs


def retrieve_family_docs_generic(question: str, family: str, k: int = 10):
    if not vectorstore:
        return []
    queries = [question]
    queries.extend(_re.findall(r'table\s*\d+\.\d+', question, _re.IGNORECASE))
    queries.extend(_re.findall(r'part\s*\d+', question, _re.IGNORECASE))

    pool = []
    seen = set()
    for query in queries:
        for doc in vectorstore.similarity_search(query, k=FAMILY_VECTOR_SEARCH_K):
            source_path = doc.metadata.get("source", "")
            if not source_matches_family(source_path, family):
                continue
            page = doc.metadata.get("page")
            key = (source_path, page, hash((doc.page_content or "")[:160]))
            if key in seen:
                continue
            seen.add(key)
            pool.append(doc)
            if len(pool) >= k:
                return annotate_docs(pool, family=family)
    return annotate_docs(pool, family=family)


def build_family_answer(question: str, family: str, strict_mode: bool = False, selected_document: str | None = None):
    label = FAMILY_DISPLAY_NAMES[family]
    direct_table = build_direct_table_answer(question, force_family=family, selected_document=selected_document)

    if direct_table:
        answer_text, source_docs = direct_table
        source_docs = annotate_docs(source_docs, family=family)
        refs = collect_reference_summary(source_docs)
    else:
        source_docs = retrieve_family_docs_generic(question, family, k=10)
        if not source_docs:
            return f"References used: none found\n\nNo relevant {label} context found for this question.", [], []

        refs = collect_reference_summary(source_docs)
        analysis = question_analysis(question, selected_document=selected_document)
        facts_lines = []
        for key, value in analysis.get("facts", {}).items():
            if isinstance(value, bool):
                facts_lines.append(f"- {key}: {'yes' if value else 'no'}")
            elif value is None:
                facts_lines.append(f"- {key}: unknown")
            else:
                facts_lines.append(f"- {key}: {value}")
        missing_text = "\n".join(f"- {item}" for item in analysis.get("missing_critical_inputs", [])) or "- none"
        refs_text = format_reference_list(refs) if refs else "no explicit table/figure/section label found in retrieved text"
        family_context = format_docs(source_docs)

        facts_lines_str = "\n".join(facts_lines)
        family_prompt = (
            f"You are answering using {label} context only.\n\n"
            f"Question: {question}\n\n"
            "Parsed facts:\n"
            f"{facts_lines_str}\n\n"
            "Critical missing inputs:\n"
            f"{missing_text}\n\n"
            f"Available references seen in the retrieved context: {refs_text}.\n\n"
            "Rules:\n"
            "- Answer only from the supplied context.\n"
            "- Start with 'References used: ...' and list the exact table/figure/section labels relied on.\n"
            "- If no explicit label is visible in the retrieved text, say so.\n"
            "- If the question is ambiguous, explain what this manual suggests conditionally without inventing missing inputs.\n"
            "- Keep the answer practical and concise.\n\n"
            f"Context:\n{family_context}"
        )
        response = llm.invoke(family_prompt)
        answer_text = model_response_to_text(response)

    if not answer_text.lower().startswith("references used:"):
        refs_text = format_reference_list(refs)
        answer_text = f"References used: {refs_text}\n\n{answer_text}"

    return answer_text, source_docs, refs


def build_regime_comparison_answer(question: str, strict_mode: bool = False, selected_document: str | None = None):
    ag_text, ag_docs, ag_refs = build_family_answer(question, "agttm", strict_mode=strict_mode, selected_document=selected_document)
    qg_text, qg_docs, qg_refs = build_family_answer(question, "qgttm", strict_mode=strict_mode, selected_document=selected_document)

    if not ag_docs and not qg_docs:
        return None

    reference_summary = (
        "| Regime | References used |\n"
        "|---|---|\n"
        f"| AGTTM | {format_reference_list(ag_refs)} |\n"
        f"| QGTTM | {format_reference_list(qg_refs)} |"
    )

    final_answer = (
        f"**Reference Summary**\n\n{reference_summary}\n\n"
        f"**AGTTM says this**\n\n{ag_text}\n\n"
        f"**QGTTM says this**\n\n{qg_text}"
    )

    combined_docs = []
    seen = set()
    for doc in ag_docs + qg_docs:
        source_path = doc.metadata.get("source", "")
        page = doc.metadata.get("page")
        key = (source_path, page, doc.metadata.get("family", ""))
        if key in seen:
            continue
        seen.add(key)
        combined_docs.append(doc)

    return final_answer, combined_docs

def hybrid_retrieve(question: str, selected_document: str | None = None):
    """MMR retrieval + keyword-boosted pass + cross-encoder re-ranking for better relevance."""
    if not retriever or not vectorstore:
        return []
    selected_norm = normalize_source_path(selected_document) if selected_document else ""

    # Stage 1: Fast vector retrieval with MMR (gets ~12 candidates)
    docs = retriever.invoke(question)
    
    # Stage 2: Cross-encoder re-ranking (scores and filters to top 5-6)
    if reranker:
        docs = rerank_retrieved_docs(question, docs, reranker, top_k=RETRIEVAL_RERANK_TOP_K)
    
    if selected_norm:
        docs = [d for d in docs if normalize_source_path(d.metadata.get("source", "")) == selected_norm]
    seen_ids = {id(d) for d in docs}

    # If the query explicitly mentions a table number or part number, do an
    # extra similarity search using just that reference as the query so the
    # relevant chunk is more likely to be included.
    table_refs = _re.findall(r'table\s*\d+\.\d+', question, _re.IGNORECASE)
    part_refs  = _re.findall(r'part\s*\d+', question, _re.IGNORECASE)
    extra_queries = table_refs + part_refs
    for ref in extra_queries:
        extra = vectorstore.similarity_search(ref, k=RETRIEVAL_EXTRA_REF_K)
        for d in extra:
            if selected_norm and normalize_source_path(d.metadata.get("source", "")) != selected_norm:
                continue
            if id(d) not in seen_ids:
                seen_ids.add(id(d))
                docs.append(d)

    # Fallback for document-scoped queries when MMR did not return any chunk.
    if selected_norm and not docs:
        fallback = vectorstore.similarity_search(question, k=RETRIEVAL_FALLBACK_K)
        for d in fallback:
            if normalize_source_path(d.metadata.get("source", "")) == selected_norm:
                docs.append(d)
            if len(docs) >= RETRIEVAL_MMR_K:
                break
    return docs


def source_matches_family_part(source_path: str, family: str, part_num: str):
    src = (source_path or "").lower()
    fam = (family or "").lower()
    part = str(part_num or "").strip()
    if not src or not fam or not part:
        return False

    if fam == "agttm" and "agttm" not in src:
        return False
    if fam == "qgttm" and "qgttm" not in src:
        return False

    part_patterns = [
        f"part_{part}",
        f"part-{part}",
        f"part {part}",
        f"agttm{int(part):02d}" if part.isdigit() else "",
        f"agttm{part}"
    ]
    return any(p and p in src for p in part_patterns)


def get_family_part_paths(family: str, part_num: str):
    if not loaded_documents:
        return []
    paths = [doc["path"] for doc in loaded_documents if isinstance(doc, dict) and "path" in doc]
    return [p for p in paths if source_matches_family_part(p, family, part_num)]


def retrieve_family_docs(question: str, family: str, part_num: str, k: int = 8):
    """Retrieve relevant chunks constrained to AGTTM/QGTTM and a specific part."""
    if not vectorstore:
        return []
    seed_queries = [question, f"Part {part_num}", f"Table Part {part_num}"]
    pool = []
    seen = set()

    for q in seed_queries:
        for d in vectorstore.similarity_search(q, k=FAMILY_VECTOR_SEARCH_K):
            src = d.metadata.get("source", "")
            if not source_matches_family_part(src, family, part_num):
                continue
            page = d.metadata.get("page")
            key = (src, page, hash((d.page_content or "")[:120]))
            if key in seen:
                continue
            seen.add(key)
            pool.append(d)
            if len(pool) >= k:
                return pool
    return pool


def extract_table_markdown_from_text(text: str, table_ref: str):
    """Extract a markdown table block for an explicit table reference from OCR/PDF text."""
    if not text:
        return None

    normalized = text.replace("\u2013", "-").replace("\u2014", "-")
    lines = [ln.strip() for ln in normalized.splitlines()]
    ref_pattern = _re.compile(rf"\btable\s*{_re.escape(table_ref)}\b", _re.IGNORECASE)
    row_pattern = _re.compile(r"^(<=?\s*\d+|≤\s*\d+|\d+\s*[-–]\s*\d+|>\s*\d+)\s+([0-9]+(?:\.[0-9]+)?)$")

    start_idx = None
    for i, ln in enumerate(lines):
        if ref_pattern.search(ln):
            start_idx = i
            break
    if start_idx is None:
        return None

    # Collect a bounded window after the table heading.
    window = lines[start_idx:start_idx + 45]

    title = window[0] if window else f"Table {table_ref}"
    headers = ["Column 1", "Column 2"]
    rows = []
    footnote = ""

    for ln in window[1:]:
        if not ln:
            continue
        lower_ln = ln.lower()
        if lower_ln.startswith("figure "):
            break
        if lower_ln.startswith("table ") and not ref_pattern.search(ln):
            break
        if lower_ln.startswith("*") and "clearance" in lower_ln:
            footnote = ln

        # Heuristic header pickup
        if "speed" in lower_ln and ("distance" in lower_ln or "recommended" in lower_ln):
            headers = ["Speed (km/h)", "Distance (m)*"]

        m = row_pattern.match(ln)
        if m:
            speed = m.group(1).replace("<=", "≤").replace(" - ", "-").strip()
            dist = m.group(2).strip()
            rows.append((speed, dist))

    if not rows:
        return None

    md = []
    md.append(f"**{title}**")
    md.append("")
    md.append(f"| {headers[0]} | {headers[1]} |")
    md.append("|---|---|")
    for speed, dist in rows:
        md.append(f"| {speed} | {dist} |")
    if footnote:
        md.append("")
        md.append(footnote)
    return "\n".join(md)


def build_direct_table_answer(question: str, force_family: str | None = None, force_part: str | None = None, selected_document: str | None = None):
    """Return a deterministic table answer when a query explicitly asks for Table X.Y."""
    if not vectorstore and not loaded_documents:
        return None
    q = question or ""
    q_lower = q.lower()

    table_match = _re.search(r"table\s*(\d+\.\d+)", q, _re.IGNORECASE)
    if not table_match:
        return None

    table_ref = table_match.group(1)
    part_match = _re.search(r"part\s*(\d+)", q, _re.IGNORECASE)
    part_num = force_part or (part_match.group(1) if part_match else None)

    targeted_query = f"Table {table_ref}"
    if part_num:
        targeted_query += f" Part {part_num}"
    if force_family == "agttm" or "agttm" in q_lower:
        targeted_query += " AGTTM"
    if force_family == "qgttm" or "qgttm" in q_lower:
        targeted_query += " QGTTM"

    # 1) Deterministic pass: scan likely PDF files page-by-page for Table X.Y.
    path_candidates = [doc["path"] for doc in loaded_documents if isinstance(doc, dict) and "path" in doc]

    selected_norm = normalize_source_path(selected_document) if selected_document else ""

    def _path_matches(path_lower: str, original_path: str):
        if selected_norm and normalize_source_path(original_path) != selected_norm:
            return False
        if force_family == "agttm" and "agttm" not in path_lower:
            return False
        if force_family == "qgttm" and "qgttm" not in path_lower:
            return False
        if "agttm" in q_lower and "agttm" not in path_lower:
            return False
        if "qgttm" in q_lower and "qgttm" not in path_lower:
            return False
        if part_num and (f"part_{part_num}" not in path_lower and f"part-{part_num}" not in path_lower and f"part {part_num}" not in path_lower):
            return False
        return True

    likely_paths = [p for p in path_candidates if _path_matches((p or "").lower(), p or "")]
    if not likely_paths:
        likely_paths = path_candidates

    for pdf_path in likely_paths:
        try:
            pages = get_cached_pdf_pages(pdf_path)
        except Exception:
            continue

        for i, page_doc in enumerate(pages):
            page_text = page_doc.page_content or ""
            # Include next page as overflow in case row extraction wraps.
            overflow = ""
            if i + 1 < len(pages):
                overflow = "\n" + (pages[i + 1].page_content or "")

            merged_text = page_text + overflow
            table_md = extract_table_markdown_from_text(merged_text, table_ref)
            if table_md:
                src_doc = Document(
                    page_content=merged_text,
                    metadata={"source": pdf_path, "page": i, "refs": [f"Table {table_ref}"], "family": FAMILY_DISPLAY_NAMES.get(force_family or "", "")}
                )
                return table_md, [src_doc]

    # 2) Fallback pass: try vector search candidates if deterministic scan fails.
    if not vectorstore:
        return None

    candidates = vectorstore.similarity_search(targeted_query, k=DIRECT_TABLE_VECTOR_K)

    # Optional filters based on explicit query hints.
    filtered = []
    for d in candidates:
        src = (d.metadata.get("source", "") or "").lower()
        if "agttm" in q_lower and "agttm" not in src:
            continue
        if "qgttm" in q_lower and "qgttm" not in src:
            continue
        if part_num and (f"part_{part_num}" not in src and f"part-{part_num}" not in src and f"part {part_num}" not in src):
            continue
        filtered.append(d)
    if filtered:
        candidates = filtered

    for d in candidates:
        table_md = extract_table_markdown_from_text(d.page_content, table_ref)
        if table_md:
            d.metadata["refs"] = [f"Table {table_ref}"]
            if force_family:
                d.metadata["family"] = FAMILY_DISPLAY_NAMES.get(force_family, force_family.upper())
            return table_md, [d]

    return None


def build_dual_part_answer(question: str, selected_part: str | None = None):
    """Compare AGTTM and QGTTM for an explicitly selected part."""
    q = question or ""
    part_num = str(selected_part).strip() if selected_part is not None else ""

    if not part_num:
        return None
    ag_docs = retrieve_family_docs(q, "agttm", part_num, k=8)
    qg_docs = retrieve_family_docs(q, "qgttm", part_num, k=8)

    if not ag_docs and not qg_docs:
        return None

    table_match = _re.search(r"table\s*(\d+\.\d+)", q, _re.IGNORECASE)
    ag_text = ""
    qg_text = ""
    selected_sources = []

    if table_match:
        ag_direct = build_direct_table_answer(q, force_family="agttm", force_part=part_num)
        qg_direct = build_direct_table_answer(q, force_family="qgttm", force_part=part_num)

        if ag_direct:
            ag_text = ag_direct[0]
            selected_sources.extend(ag_direct[1])
        if qg_direct:
            qg_text = qg_direct[0]
            selected_sources.extend(qg_direct[1])

    if not ag_text:
        if ag_docs:
            ag_ctx = format_docs(ag_docs)
            ag_prompt = (
                f"Question: {q}\n\n"
                "Using only the AGTTM context below, provide the direct answer in a concise way.\n\n"
                f"AGTTM Context:\n{ag_ctx}"
            )
            ag_resp = llm.invoke(ag_prompt)
            ag_text = model_response_to_text(ag_resp)
            selected_sources.extend(ag_docs)
        else:
            ag_text = "No relevant AGTTM Part context found for this question."

    if not qg_text:
        if qg_docs:
            qg_ctx = format_docs(qg_docs)
            qg_prompt = (
                f"Question: {q}\n\n"
                "Using only the QGTTM context below, provide the direct answer in a concise way.\n\n"
                f"QGTTM Context:\n{qg_ctx}"
            )
            qg_resp = llm.invoke(qg_prompt)
            qg_text = model_response_to_text(qg_resp)
            selected_sources.extend(qg_docs)
        else:
            qg_text = "No relevant QGTTM Part context found for this question."

    final_answer = (
        f"**AGTTM says this (Part {part_num})**\n\n{ag_text}\n\n"
        f"**QGTTM says this (Part {part_num})**\n\n{qg_text}"
    )

    # De-duplicate source docs by (source,page)
    unique = []
    seen = set()
    for d in selected_sources:
        src = d.metadata.get("source", "")
        pg = d.metadata.get("page")
        key = (src, pg)
        if key in seen:
            continue
        seen.add(key)
        unique.append(d)

    return final_answer, unique

# Create RAG chain using LCEL (only if retriever is available)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

initialize_or_reload_index()

def build_answer(question: str, selected_document: str | None = None, session_id: str | None = None):
    """Run the RAG chain but use hybrid_retrieve for richer context."""
    # Add conversation history context if session_id provided
    conversation_context = ""
    if session_id:
        conversation_context = format_conversation_history_for_prompt(session_id)
    
    source_docs = hybrid_retrieve(question, selected_document=selected_document)
    context_text = format_docs(source_docs)
    analysis = question_analysis(question, selected_document=selected_document)
    facts = analysis["facts"]
    missing = analysis["missing_critical_inputs"]

    facts_lines = []
    for k, v in facts.items():
        if isinstance(v, bool):
            facts_lines.append(f"- {k}: {'yes' if v else 'no'}")
        elif v is None:
            facts_lines.append(f"- {k}: unknown")
        else:
            facts_lines.append(f"- {k}: {v}")
    facts_text = "\n".join(facts_lines)

    input_with_facts = (
        f"Question:\n{question}\n\n"
        "Parsed question facts:\n"
        f"{facts_text}\n\n"
        "Critical missing inputs (if any):\n"
        + ("\n".join(f"- {item}" for item in missing) if missing else "- none")
        + conversation_context
    )

    filled_prompt = prompt.format_messages(context=context_text, input=input_with_facts)
    answer = llm.invoke(filled_prompt)
    return model_response_to_text(answer), source_docs


def should_run_dual_regime_answer(question: str, selected_part: str | None = None) -> bool:
    if selected_part:
        return True

    q_lower = (question or "").lower()
    if "agttm" in q_lower and "qgttm" in q_lower:
        return True

    compare_tokens = ["compare", "comparison", "difference", "different", "versus", " vs "]
    return any(token in q_lower for token in compare_tokens)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "host": APP_HOST,
        "port": APP_PORT,
        "drive_sync": {
            "status": "local_directory",
            "indexed_documents": len(loaded_documents),
            "indexed_pdfs": len(current_index_meta.get("files", [])) if current_index_meta else 0
        }
    })


@app.route('/', methods=['GET'])
def serve_ui():
    if not os.path.exists(UI_FILE):
        return jsonify({"error": "UI file not found"}), 404
    return send_file(UI_FILE, mimetype='text/html')


@app.route('/reload-index', methods=['POST'])
def reload_index():
    result = initialize_or_reload_index(force_rebuild=True)
    status = 200 if result.get("ready") else 503
    return jsonify(result), status

@app.route('/documents', methods=['GET'])
def list_documents():
    # Return deduplicated list of unique documents
    if not loaded_documents:
        return jsonify({"documents": []})
    
    # Deduplicate by source path
    unique_docs = {}
    for doc in loaded_documents:
        path = doc['path']
        if path not in unique_docs:
            unique_docs[path] = doc
    
    docs = []
    for doc in unique_docs.values():
        jurisdiction = classify_document_jurisdiction(doc['path'], doc['title'])
        docs.append({
            "title": doc['title'],
            "path": doc['path'],
            "jurisdiction": jurisdiction
        })

    docs.sort(key=lambda item: (jurisdiction_sort_key(item.get("jurisdiction", "Other")), item.get("title", "").lower()))
    return jsonify({"documents": docs})

@app.route('/ask', methods=['POST'])
def ask():
    if not retriever or not rag_chain:
        return jsonify({"error": "No documents indexed. Please add PDFs to the drive_docs folder and restart the app."}), 503
    
    data = request.json
    question = data.get('question', '')
    selected_part = data.get('selected_part')
    selected_document = data.get('selected_document')
    strict_mode = bool(data.get('strict_mode', False))
    session_id = data.get('session_id')  # Get session_id from request
    use_agentic_mode = data.get('use_agentic', False)  # NEW: Enable multi-agent routing
    
    # Generate new session_id if not provided
    if not session_id:
        session_id = create_session_id()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    print(f"User asked (session: {session_id}, mode: {'agentic' if use_agentic_mode else 'standard'}): {question}")

    # --- Streaming path: agentic mode with stream=true ---
    current_agentic_system = agentic_system
    if use_agentic_mode and current_agentic_system is not None and data.get('stream', False):
        import json as _json
        conversation_context = format_conversation_history_for_prompt(session_id)

        def _generate_sse():
            full_answer = []
            try:
                local_agentic_system = current_agentic_system
                if local_agentic_system is None:
                    raise RuntimeError("Agentic system is not available")
                for event in local_agentic_system.stream(
                    question=question,
                    selected_document=selected_document,
                    session_id=session_id,
                    conversation_context=conversation_context
                ):
                    if event['type'] == 'metadata':
                        yield f"data: {_json.dumps(event)}\n\n"
                    elif event['type'] == 'token':
                        full_answer.append(event['content'])
                        yield f"data: {_json.dumps(event)}\n\n"
                    elif event['type'] == 'done':
                        answer_text = ''.join(full_answer)
                        add_message_to_conversation(session_id, 'human', question)
                        add_message_to_conversation(session_id, 'ai', answer_text)
                        yield f"data: {_json.dumps({'type': 'done', 'session_id': session_id})}\n\n"
            except Exception as _e:
                yield f"data: {_json.dumps({'type': 'error', 'message': str(_e)})}\n\n"

        return Response(
            stream_with_context(_generate_sse()),
            mimetype='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
        )

    try:
        # NEW: Multi-agent agentic routing mode
        if use_agentic_mode and agentic_system:
            print("  → Using multi-agent agentic routing (Router → Researcher → Engineer)")
            conversation_context = format_conversation_history_for_prompt(session_id)
            agentic_result = agentic_system.invoke(
                question=question,
                selected_document=selected_document,
                session_id=session_id,
                conversation_context=conversation_context
            )
            
            # Extract data from agentic result
            answer = agentic_result.get("answer", "")
            source_docs = agentic_result.get("sources", [])
            routing_info = agentic_result.get("routing", {})
            
            # Convert source dicts to Document-like objects for compatibility
            source_docs_formatted = []
            for src in agentic_result.get("sources", []):
                doc_obj = Document(
                    page_content="",
                    metadata={
                        "source": src.get("path", ""),
                        "page": src.get("page", 0),
                        "title": src.get("title", "")
                    }
                )
                source_docs_formatted.append(doc_obj)
            source_docs = source_docs_formatted
            
            # Store in conversation memory
            add_message_to_conversation(session_id, "human", question)
            add_message_to_conversation(session_id, "ai", answer)
            
            # Format response with agentic metadata
            sources_by_path = {}
            for src in agentic_result.get("sources", []):
                path = src.get("path", "")
                if path not in sources_by_path:
                    sources_by_path[path] = src
            
            return jsonify({
                "answer": answer,
                "sources": list(sources_by_path.values()),
                "snippets": [],  # Agentic mode doesn't return snippets in same format
                "session_id": session_id,
                "agent_routing": {
                    "mode": "agentic",
                    "router_decision": routing_info.get("decision"),
                    "router_confidence": routing_info.get("confidence"),
                    "search_queries": agentic_result.get("search_queries", []),
                    "num_documents_retrieved": agentic_result.get("agent_details", {}).get("num_documents_retrieved", 0)
                },
                "question_analysis": {
                    "facts": {},
                    "missing_critical_inputs": [],
                    "is_ambiguous": False,
                    "follow_up_questions": [],
                    "strict_mode": strict_mode,
                }
            })
        
        # STANDARD: Original single-agent RAG pipeline
        analysis = question_analysis(question, selected_document=selected_document)
        comparison_answer = None
        if should_run_dual_regime_answer(question, selected_part=selected_part):
            comparison_question = question
            if selected_part and not _re.search(r'part\s*\d+', question, _re.IGNORECASE):
                comparison_question = f"{question}\n\nPart {selected_part}"

            comparison_answer = build_regime_comparison_answer(
                comparison_question,
                strict_mode=strict_mode,
                selected_document=selected_document,
            )
        if comparison_answer:
            answer, source_docs = comparison_answer
        else:
            clarification_answer = build_clarification_first_answer(
                question,
                selected_document=selected_document,
                strict_mode=strict_mode
            )
            if clarification_answer:
                answer, source_docs = clarification_answer
            else:
                answer, source_docs = build_answer(question, selected_document=selected_document, session_id=session_id)
        
        # This is the new part that adds detailed chunk info to the response
        sources_by_path = {}
        for doc in source_docs:
            metadata = doc.metadata
            source_path = metadata.get("source", "")
            if not source_path:
                continue

            if source_path not in sources_by_path:
                sources_by_path[source_path] = {
                    "title": os.path.basename(source_path),
                    "path": source_path,
                    "family": metadata.get("family"),
                    "chunks": []
                }
            
            page_num = metadata.get("page")
            
            sources_by_path[source_path]["chunks"].append({
                "page": page_num + 1 if isinstance(page_num, int) else None,
                "block_id": metadata.get("block_id"),
                "block_type": metadata.get("block_type"),
                "bbox": metadata.get("bbox"),
                "reranker_score": metadata.get("reranker_score"),
                "text_preview": doc.page_content[:150] + "..."
            })
            
            if metadata.get("family") and not sources_by_path[source_path].get("family"):
                sources_by_path[source_path]["family"] = metadata.get("family")

        # Sort chunks within each source by score
        for source in sources_by_path.values():
            source["chunks"].sort(key=lambda c: c.get("reranker_score", 0), reverse=True)

        # The 'snippets' list now contains full metadata for each chunk
        snippets = []
        for doc in source_docs:
            metadata = doc.metadata
            sp = metadata.get("source", "")
            rp = metadata.get("page")
            pn = rp + 1 if isinstance(rp, int) else None

            snippets.append({
                "title": os.path.basename(sp),
                "path": sp,
                "page": pn,
                "text": doc.page_content,
                "doc_id": metadata.get("doc_id"),
                "block_id": metadata.get("block_id"),
                "block_type": metadata.get("block_type"),
                "bbox": metadata.get("bbox"),
                "family": metadata.get("family"),
                "refs": list(metadata.get("refs", [])),
                "reranker_score": metadata.get("reranker_score"),
            })

        add_message_to_conversation(session_id, "human", question)
        add_message_to_conversation(session_id, "ai", answer)
        
        return jsonify({
            "answer": answer,
            "sources": list(sources_by_path.values()),
            "snippets": snippets,
            "session_id": session_id,
            "question_analysis": {
                "facts": analysis.get("facts", {}),
                "missing_critical_inputs": analysis.get("missing_critical_inputs", []),
                "is_ambiguous": analysis.get("is_ambiguous", False),
                "follow_up_questions": analysis.get("follow_up_questions", []),
                "strict_mode": strict_mode,
            }
        })
    except Exception as e:
        err_str = str(e)
        print(f"Error during AI generation: {e}")
        # Detect missing/invalid Google API key or Gemini auth issues
        err_lower = err_str.lower()
        if ("google_api_key" in err_lower) or ("api key" in err_lower) or ("permission denied" in err_lower) or ("unauthenticated" in err_lower):
            return jsonify({
                "error": "Google Gemini authentication failed. Check GOOGLE_API_KEY.",
                "error_code": "gemini_auth_error"
            }), 503
        return jsonify({"error": err_str}), 500

# ============================================================================
# CONVERSATION MEMORY ROUTES
# ============================================================================

@app.route('/conversations/start', methods=['POST'])
def start_conversation():
    """Start a new conversation session."""
    session_id = create_session_id()
    return jsonify({
        "session_id": session_id,
        "message": "New conversation started"
    }), 201


@app.route('/conversations', methods=['GET'])
def list_active_conversations():
    """Get list of all active conversations with metadata."""
    conversations = list_conversations()
    return jsonify({
        "active_conversations": conversations,
        "total_conversations": len(conversations)
    }), 200


@app.route('/conversations/<session_id>', methods=['GET'])
def get_conversation(session_id: str):
    """Get full conversation history for a session."""
    history = get_conversation_history(session_id)
    if not history:
        return jsonify({
            "error": "Conversation not found",
            "session_id": session_id
        }), 404
    
    return jsonify({
        "session_id": session_id,
        "messages": history,
        "message_count": len(history)
    }), 200


@app.route('/conversations/<session_id>', methods=['DELETE'])
def delete_conversation(session_id: str):
    """Clear conversation history for a session."""
    history = get_conversation_history(session_id)
    if not history:
        return jsonify({
            "error": "Conversation not found",
            "session_id": session_id
        }), 404
    
    clear_conversation(session_id)
    return jsonify({
        "message": "Conversation cleared",
        "session_id": session_id,
        "messages_deleted": len(history)
    }), 200

@app.route('/documents/open', methods=['GET'])
def open_document():
    """Serve a PDF file from drive_docs for in-browser viewing."""
    from pathlib import Path
    relative_path = request.args.get('path', '')
    if not relative_path:
        return jsonify({'error': 'Missing file path'}), 400

    base = Path(DOCS_DIR).resolve()
    # Accept both absolute paths (stored by PyPDFLoader) and relative paths
    target = Path(relative_path).resolve()
    if not target.exists():
        # Try treating it as relative to cwd
        target = (Path('.') / relative_path).resolve()

    if not target.exists() or not target.is_file():
        return jsonify({'error': 'File not found'}), 404

    # Security: ensure file is inside the drive_docs folder
    try:
        target.relative_to(base)
    except ValueError:
        return jsonify({'error': 'Access denied'}), 403

    from flask import send_file
    return send_file(str(target), mimetype='application/pdf')


@app.route('/documents/page-image', methods=['GET'])
def document_page_image():
    from pathlib import Path
    from flask import send_file

    relative_path = request.args.get('path', '')
    page_raw = request.args.get('page', '')
    if not relative_path or not page_raw:
        return jsonify({'error': 'Missing file path or page'}), 400

    try:
        page_number = max(1, int(page_raw))
    except ValueError:
        return jsonify({'error': 'Invalid page number'}), 400

    base = Path(DOCS_DIR).resolve()
    target = Path(relative_path).resolve()
    if not target.exists():
        target = (Path('.') / relative_path).resolve()

    if not target.exists() or not target.is_file():
        return jsonify({'error': 'File not found'}), 404

    try:
        target.relative_to(base)
    except ValueError:
        return jsonify({'error': 'Access denied'}), 403

    try:
        import fitz
    except ImportError:
        return jsonify({'error': 'PyMuPDF is not installed'}), 501

    pdf_doc = None
    try:
        pdf_doc = fitz.open(str(target))
        if page_number > pdf_doc.page_count:
            return jsonify({'error': 'Page out of range'}), 404
        page = pdf_doc.load_page(page_number - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        image_bytes = pix.tobytes('png')
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
    finally:
        if pdf_doc is not None:
            pdf_doc.close()

    return send_file(BytesIO(image_bytes), mimetype='image/png')


if __name__ == '__main__':
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"Created '{DOCS_DIR}' folder. Please place your PDFs inside and restart.")

    # use_reloader=False prevents Flask from starting a second process
    # that would re-run all the expensive PDF indexing on startup
    app.run(host=APP_HOST, port=APP_PORT, debug=APP_DEBUG, use_reloader=False)
