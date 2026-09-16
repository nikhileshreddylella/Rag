import os
import re
import tempfile
import json
import psycopg2
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langsmith import traceable
from langsmith.wrappers import wrap_openai

from rag_engine import DocumentParser, StructureChunker, GroqClient, GeminiClient, VectorStore, LocalEmbeddingClient
import evaluation

# Load environment variables (supports running from root or backend directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(BASE_DIR, ".env")
if not os.path.exists(dotenv_path):
    dotenv_path = os.path.join(os.path.dirname(BASE_DIR), ".env")
load_dotenv(dotenv_path=dotenv_path)

# Define Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    global GROUND_TRUTH_DATA
    init_db()
    GROUND_TRUTH_DATA = load_ground_truth_from_langsmith()
    
    # Eagerly load local ML models at startup to prevent first-query cold-start latency
    try:
        print("[STARTUP] Pre-loading local SentenceTransformer embedding model...")
        from rag_engine import LocalEmbeddingClient
        LocalEmbeddingClient().model
        
        print("[STARTUP] Pre-loading local CrossEncoder reranker model...")
        from rag_engine import RERANKER
        RERANKER.model

        print("[STARTUP] Pre-warming document cache...")
        col = VECTOR_STORE.get_collection(COLLECTION_NAME)
        if col.count() > 0:
            VECTOR_STORE._get_all_documents_cached(COLLECTION_NAME, col)
            print(f"[STARTUP] Warmed {col.count()} documents into memory cache.")

        print("[STARTUP] All local models and caches loaded and ready!")
    except Exception as e:
        print(f"[STARTUP WARNING] Failed to pre-load local models: {e}")
        
    yield

app = FastAPI(lifespan=lifespan)

# Mount static directory
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Initialize DB
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
VECTOR_STORE = VectorStore(persist_directory=DB_DIR)
COLLECTION_NAME = "document_rag_collection"

groq_api_key = os.getenv("GROQ_API_KEY", "")
evaluation_api_key = os.getenv("EVALUATION_API_KEY", groq_api_key)

RAG_MODEL = os.getenv("GROQ_RAG_MODEL", "openai/gpt-oss-20b")
EVALUATION_MODEL = os.getenv("GROQ_EVALUATION_MODEL", "openai/gpt-oss-20b")

# Load ground truth dynamically from LangSmith once at startup
GROUND_TRUTH_DATA: list = []

def load_ground_truth_from_langsmith() -> list[dict]:
    import os
    from langsmith import Client
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key or api_key == "your_langsmith_api_key_here":
        print("[GT ERROR] LANGSMITH_API_KEY is not set. Cannot fetch ground truths.")
        return []
    try:
        client = Client(api_key=api_key)
        examples = client.list_examples(dataset_name="qa-rag-golden-dataset")
        gt_entries = []
        for ex in examples:
            query = ex.inputs.get("query", "")
            gt_text = ex.outputs.get("ground_truth", "") or ex.outputs.get("answer", "")
            ans = ex.outputs.get("answer", "")
            ctxs = ex.inputs.get("contexts", [])
            gt_entries.append({
                "query": query,
                "ground_truth": gt_text,
                "answer": ans,
                "contexts": ctxs
            })
        print(f"[GT] Loaded {len(gt_entries)} ground truth entries dynamically from LangSmith dataset 'qa-rag-golden-dataset'")
        return gt_entries
    except Exception as e:
        print(f"[GT ERROR] Failed to load ground truth from LangSmith: {e}")
        return []

# Loaded dynamically inside FastAPI lifespan startup event

# RAG enhancements: cache and conversation memory
SEMANTIC_CACHE = []
CONVERSATION_MEMORIES = {}  # session_id -> list of turns

# Common follow-up triggers and pronouns that indicate dependence on previous conversation turns
FOLLOWUP_TRIGGERS = {
    "it", "its", "they", "them", "their", "theirs", "he", "him", "his", "she", "her",
    "this", "that", "these", "those", "why", "how", "what about", "what else",
    "explain more", "and", "then", "which one", "who was that", "the latter", "the former",
    "tell me more", "expand", "elaborate", "give an example", "where", "when did that"
}

def is_followup_query(query: str) -> bool:
    """Fast local check to see if query is a follow-up needing context."""
    words = set(re.findall(r"\w+", query.lower()))
    if words.intersection(FOLLOWUP_TRIGGERS):
        return True
    if len(words) <= 3:
        return True
    return False

def contextualize_query(query: str, history: list) -> str:
    """Rewrite follow-up queries using conversation history to make them standalone."""
    if not history or not is_followup_query(query):
        return query
        
    history_str = ""
    for msg in history[-4:]:  # last 4 turns
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"
        
    prompt = f"""Given the following conversation history and a follow-up query, rewrite the query into a standalone search query that contains all necessary context (resolving pronouns, references, and abbreviations). 
Do NOT answer the query, just rewrite it as a search query. If the query does not depend on the history, return it exactly as is.

Conversation History:
{history_str}
Follow-up Query: {query}

Standalone Search Query:"""

    try:
        llm = GroqClient(api_key=groq_api_key)
        rewritten = llm.generate_answer(prompt, model=RAG_MODEL)
        rewritten_clean = rewritten.strip().strip('"').strip("'")
        if rewritten_clean:
            print(f"[MEMORY] Contextualized query: '{query}' -> '{rewritten_clean}'")
            return rewritten_clean
    except Exception as e:
        print(f"[MEMORY WARNING] Query contextualization failed: {e}")
    return query

# Session token usage counter (resets on server restart)
SESSION_TOKEN_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "requests": 0,
    "provider": "groq",
    "model": "openai/gpt-oss-20b"
}

DB_OFFLINE = False

def get_db_connection():
    global DB_OFFLINE
    if DB_OFFLINE:
        return None
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            database=os.getenv("DB_NAME", "rag_logs"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres"),
            connect_timeout=1
        )
        return conn
    except Exception as e:
        DB_OFFLINE = True
        print(f"[DATABASE ERROR] Failed to connect to PostgreSQL: {e}. Disabling DB logging to prevent query latency.")
        return None

def init_db():
    db_name = os.getenv("DB_NAME", "rag_logs")
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "postgres")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")

    # 1. Connect to default 'postgres' database to check/create the target database
    try:
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            database="postgres",
            user=db_user,
            password=db_password,
            connect_timeout=1
        )
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (db_name,))
        exists = cur.fetchone()
        if not exists:
            print(f"[DB INIT] Creating database '{db_name}'...")
            cur.execute(f'CREATE DATABASE "{db_name}"')
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[DB INIT WARNING] Could not verify/create database: {e}")

    # 2. Connect to the target database and create the table
    try:
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password,
            connect_timeout=1
        )
        conn.autocommit = True
        cur = conn.cursor()
        create_table_query = """
        CREATE TABLE IF NOT EXISTS chat_logs (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(100),
            message_index INT,
            file_name VARCHAR(255),
            question TEXT,
            answer TEXT,
            faithfulness DOUBLE PRECISION,
            relevance DOUBLE PRECISION,
            context_precision DOUBLE PRECISION,
            context_relevance DOUBLE PRECISION,
            context_recall DOUBLE PRECISION,
            answer_correctness DOUBLE PRECISION,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cur.execute(create_table_query)

        # Database Schema Migrations: ADD columns if missing
        # 1. Check and add answer_correctness column if missing
        cur.execute("""
            SELECT 1 FROM information_schema.columns 
            WHERE table_name='chat_logs' AND column_name='answer_correctness';
        """)
        if not cur.fetchone():
            print("[DB MIGRATE] Adding column 'answer_correctness' to 'chat_logs' table...")
            cur.execute("ALTER TABLE chat_logs ADD COLUMN answer_correctness DOUBLE PRECISION;")

        # 2. Check and add context_relevance column if missing
        cur.execute("""
            SELECT 1 FROM information_schema.columns 
            WHERE table_name='chat_logs' AND column_name='context_relevance';
        """)
        if not cur.fetchone():
            print("[DB MIGRATE] Adding column 'context_relevance' to 'chat_logs' table...")
            cur.execute("ALTER TABLE chat_logs ADD COLUMN context_relevance DOUBLE PRECISION;")

        # Database Schema Migrations: DROP columns if they exist
        for col in ("context_entities_recall", "noise_sensitivity", "explanation", "inference", "inference_time_ms"):
            cur.execute(f"""
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='chat_logs' AND column_name='{col}';
            """)
            if cur.fetchone():
                print(f"[DB MIGRATE] Dropping column '{col}' from 'chat_logs' table...")
                cur.execute(f"ALTER TABLE chat_logs DROP COLUMN IF EXISTS {col};")

        cur.close()
        conn.close()
        print(f"[DB INIT] Database '{db_name}' table 'chat_logs' is ready.")
    except Exception as e:
        print(f"[DB INIT ERROR] Failed to initialize table: {e}")

import threading

def _async_log_qa_worker(session_id: str | None, message_index: int | None, file_name: str | None, question: str, answer: str):
    if not session_id or message_index is None:
        return
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            insert_query = """
            INSERT INTO chat_logs (session_id, message_index, file_name, question, answer)
            VALUES (%s, %s, %s, %s, %s);
            """
            cur.execute(insert_query, (session_id, message_index, file_name, question, answer))
            conn.commit()
            cur.close()
            conn.close()
            print(f"[DATABASE] Logged question and answer for session={session_id} index={message_index}")
        except Exception as db_err:
            print(f"[DATABASE ERROR] Failed to insert log record: {db_err}")

def log_qa_to_db(session_id: str | None, message_index: int | None, file_name: str | None, question: str, answer: str):
    threading.Thread(target=_async_log_qa_worker, args=(session_id, message_index, file_name, question, answer), daemon=True).start()

@app.get("/")
def read_root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

# Get count of document chunks in Database
@app.get("/db_count")
def db_count():
    try:
        count = VECTOR_STORE.get_collection(COLLECTION_NAME).count()
        source = None
        if count > 0:
            sample = VECTOR_STORE.get_collection(COLLECTION_NAME).get(limit=1)
            if sample and "metadatas" in sample and sample["metadatas"]:
                source = sample["metadatas"][0].get("source", None)
        return {"count": count, "source": source}
    except Exception as e:
        return {"count": 0, "source": None, "error": str(e)}

class SessionsBody(BaseModel):
    sessions: dict

SESSIONS_FILE = os.path.join(BASE_DIR, "sessions.json")

@app.get("/sessions")
def get_sessions():
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

@app.post("/sessions")
def save_sessions(body: SessionsBody):
    try:
        with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(body.sessions, f, indent=4, ensure_ascii=False)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Upload and index PDF/Word/Text/Image files
@app.post("/upload")
@traceable(name="upload_and_index", run_type="chain")
async def upload_file(file: UploadFile = File(...)):
    allowed_extensions = {".pdf", ".docx", ".doc", ".txt", ".png", ".jpg", ".jpeg", ".webp"}
    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(allowed_extensions))}"
        )
        
    if ext in {".png", ".jpg", ".jpeg", ".webp"} and not groq_api_key:
        raise HTTPException(
            status_code=400,
            detail="Groq API key is not configured in .env file, which is required for processing images."
        )
    
    import time
    start_time = time.time()
    try:
        # Read file contents and save to temporary file
        file_bytes = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file_path = tmp_file.name
            
        # Parse document pages
        t0 = time.time()
        pages = DocumentParser.parse(tmp_file_path, ext, groq_api_key=groq_api_key)
        
        # Chunk text based on document structure
        splitter = StructureChunker()
        chunks = splitter.split_documents(pages, file.filename)
        chunking_time = time.time() - t0
        
        # Clear previous document cache so only one file is indexed at a time
        try:
            VECTOR_STORE.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        # Embed
        t1 = time.time()
        client = LocalEmbeddingClient()
        embeddings = client.get_embeddings_batch([c["text"] for c in chunks])
        
        # Insert into Chroma
        VECTOR_STORE.add_documents(COLLECTION_NAME, chunks, embeddings)
        indexing_time = time.time() - t1
        
        # Cleanup temp file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
            
        total_time = time.time() - start_time
        return {
            "success": True, 
            "chunks": len(chunks),
            "chunking_time": round(chunking_time, 2),
            "indexing_time": round(indexing_time, 2),
            "total_time": round(total_time, 2)
        }
    except Exception as e:
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
        raise HTTPException(status_code=500, detail=str(e))

class QueryBody(BaseModel):
    query: str
    session_id: str | None = None
    message_index: int | None = None

# Execute search query and generate Groq answer
def get_expanded_context(collection, doc, window_size=1) -> str:
    """Fetch neighboring chunks from ChromaDB to prevent boundary text truncation."""
    metadata = doc.get("metadata", {})
    source = metadata.get("source")
    chunk_idx = metadata.get("chunk_index")
    if source is None or chunk_idx is None:
        return doc["text"]
        
    # Generate neighbor IDs (e.g. chunk_95, chunk_96, chunk_97)
    neighbor_ids = [f"{source}_chunk_{i}" for i in range(chunk_idx - window_size, chunk_idx + window_size + 1)]
    
    try:
        res = collection.get(ids=neighbor_ids)
        if res and res["documents"]:
            # Sort chunks in their correct reading order by chunk index
            doc_tuples = []
            for n_id, n_doc in zip(res["ids"], res["documents"]):
                try:
                    idx = int(n_id.split("_chunk_")[-1])
                    doc_tuples.append((idx, n_doc))
                except Exception:
                    pass
            doc_tuples.sort(key=lambda x: x[0])
            return "\n".join([text for _, text in doc_tuples])
    except Exception:
        pass
    return doc["text"]


def clean_query_string(query: str) -> str:
    # 1. Normalize curly apostrophes and quotes
    query = query.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    
    # 2. Replace hyphens and underscores with spaces to split conjoined terms
    query = query.replace("-", " ").replace("_", " ")
    
    # 3. Separate letters and digits (e.g., "THE3ME" -> "THE 3 ME")
    query = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', query)
    query = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', query)
    
    # 4. Case-insensitive replacements for known conjoined document terms
    replacements = {
        r'choicepoint': 'choice point',
        r'smartskills': 'smart skills',
        r'onesmartworld': 'one smart world',
        r'smartforlife': 'smart for life',
        r'the3me': 'the 3 me',
        r'3me': '3 me'
    }
    
    cleaned = query.lower()
    for pattern, repl in replacements.items():
        cleaned = re.sub(pattern, repl, cleaned)
        
    # Collapse multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


@app.post("/query")
@traceable(name="rag_query_pipeline", run_type="chain")
def query_rag(body: QueryBody):
    global SEMANTIC_CACHE, CONVERSATION_MEMORIES
    original_query = body.query
    query_str = clean_query_string(original_query)
    
    # Retrieve session history
    session_id = body.session_id or "default_session"
    if session_id not in CONVERSATION_MEMORIES:
        CONVERSATION_MEMORIES[session_id] = []
    session_history = CONVERSATION_MEMORIES[session_id]
    
    # Rewrite/Contextualize query based on history if it is a follow-up
    query_str = contextualize_query(query_str, session_history)
    
    if not groq_api_key or groq_api_key == "your_groq_api_key_here":
        return {"answer": "⚠️ Groq API key is not configured in .env file. Please check your setup.", "sources": [], "eval_contexts": []}

    # Retrieve currently active file name
    active_file = None
    try:
        count = VECTOR_STORE.get_collection(COLLECTION_NAME).count()
        if count > 0:
            sample = VECTOR_STORE.get_collection(COLLECTION_NAME).get(limit=1)
            if sample and "metadatas" in sample and sample["metadatas"]:
                active_file = sample["metadatas"][0].get("source", None)
    except Exception:
        pass

    # Greeting bypass check
    greetings = ["hi", "hello", "hey", "hai", "howdy", "greetings", "good morning", "good afternoon", "good evening", "how are you", "who are you"]
    clean_query = query_str.strip().lower().rstrip("?").rstrip("!").strip()
    if clean_query in greetings:
        try:
            prompt = f"The user says: '{query_str}'. Respond with a friendly, professional greeting as the Q&A RAG chatbot. Be brief."
            llm = GroqClient(api_key=groq_api_key)
            answer = llm.generate_answer(prompt, model=RAG_MODEL)
            log_qa_to_db(body.session_id, body.message_index, active_file, original_query, answer)
            return {"answer": answer, "sources": [], "eval_contexts": []}
        except Exception as e:
            fallback_answer = "Hello! How can I help you today?"
            log_qa_to_db(body.session_id, body.message_index, active_file, original_query, fallback_answer)
            return {"answer": fallback_answer, "sources": [], "eval_contexts": []}

    # 1. Semantic Cache check
    for cached in SEMANTIC_CACHE:
        q1 = set(re.findall(r"\w+", query_str.lower()))
        q2 = set(re.findall(r"\w+", cached["query"].lower()))
        intersection = q1.intersection(q2)
        union = q1.union(q2)
        jaccard = len(intersection) / len(union) if union else 0.0
        if jaccard >= 0.85:
            # Cache hit
            # Append conversation history for memory persistence
            session_history.append({"role": "user", "content": original_query})
            session_history.append({"role": "assistant", "content": cached["answer"]})
            if len(session_history) > 10:
                session_history.pop(0)
                session_history.pop(0)
            
            cache_answer = cached["answer"] + "\n\n*(Served from local semantic cache)*"
            log_qa_to_db(body.session_id, body.message_index, active_file, query_str, cache_answer)
            return {
                "answer": cache_answer,
                "sources": cached["sources"],
                "eval_contexts": cached.get("eval_contexts") or [],
                "observability": {
                    "cached": True
                }
            }

    try:
        import time
        t_start = time.time()
        
        embed_client = LocalEmbeddingClient()
        
        # 2. Embed query
        t_embed_start = time.time()
        query_vector = embed_client.get_embedding(query_str)
        t_embed = time.time() - t_embed_start
        
        # 3. Hybrid search (Dense + Sparse + Reranking)
        t_retrieve_start = time.time()
        retrieved_docs = VECTOR_STORE.query(COLLECTION_NAME, query_str, query_vector, top_k=10)
        
        # Check if the query is a specific page query (e.g., "page no 247")
        page_match = re.search(r'\b(?:page\s+number|page\s+no\.?|page|pg|p)\s*#?\s*(\d+)(?:st|nd|rd|th)?\b', query_str.lower())
        if not page_match:
            page_match = re.search(r'\b(\d+)(?:st|nd|rd|th)?\s*(?:page|pg|p)\b', query_str.lower())
            
        if page_match:
            target_page = int(page_match.group(1))
            # Filter context to ONLY contain chunks from the requested page to prevent cross-page confusion/hallucinations
            filtered_docs = [doc for doc in retrieved_docs if doc["metadata"].get("page_number") == target_page]
        else:
            # Filter and diversify retrieved docs by page number to increase source variety (broad queries)
            # while staying well within Groq's free-tier TPM limits
            seen_pages = {}
            diversified_docs = []
            for doc in retrieved_docs:
                if doc["similarity"] < 0.50:
                    continue
                pg = doc["metadata"].get("page_number")
                if pg:
                    # Limit to maximum 2 chunks per page to force diversity of sources across the document
                    if seen_pages.get(pg, 0) >= 2:
                        continue
                    seen_pages[pg] = seen_pages.get(pg, 0) + 1
                diversified_docs.append(doc)
            filtered_docs = diversified_docs[:10]
        t_retrieve = time.time() - t_retrieve_start
        
        if not filtered_docs:
            no_match_answer = "No matching context was found in the database. Please make sure you have uploaded and indexed your document."
            t_db_start = time.time()
            log_qa_to_db(body.session_id, body.message_index, active_file, original_query, no_match_answer)
            t_db = time.time() - t_db_start
            
            total_time = time.time() - t_start
            print(f"[RAG LATENCY] Query: '{original_query[:30]}...' | Embed: {t_embed:.2f}s | Retrieval: {t_retrieve:.2f}s | Context Expansion: 0.00s | LLM Gen: 0.00s | DB Log: {t_db:.2f}s | Total: {total_time:.2f}s")
            return {
                "answer": no_match_answer,
                "sources": [],
                "eval_contexts": [],
                "observability": {
                    "cached": False
                }
            }
            
        # 4. Inject Conversational History (Memory)
        history_context = ""
        if session_history:
            history_context = "Recent conversation context:\n"
            for msg in session_history[-4:]:  # last 4 turns
                role_label = "User" if msg["role"] == "user" else "Assistant"
                history_context += f"{role_label}: {msg['content']}\n"
            history_context += "\n"

        # 5. Build prompt
        t_expand_start = time.time()
        context_str = ""
        collection = VECTOR_STORE.get_collection(COLLECTION_NAME)
        
        # 5a. Gather all neighbor IDs to batch-fetch from ChromaDB in a single call
        window_size = 1
        all_neighbor_ids = []
        doc_neighbor_ids = {} # maps doc ID to list of neighbor IDs
        
        for doc in filtered_docs:
            metadata = doc.get("metadata", {})
            source = metadata.get("source")
            chunk_idx = metadata.get("chunk_index")
            if source is not None and chunk_idx is not None:
                neighbor_ids = [f"{source}_chunk_{i}" for i in range(chunk_idx - window_size, chunk_idx + window_size + 1)]
                all_neighbor_ids.extend(neighbor_ids)
                doc_neighbor_ids[doc["id"]] = neighbor_ids
            else:
                doc_neighbor_ids[doc["id"]] = []
                
        # Batch query ChromaDB
        db_docs_map = {}
        if all_neighbor_ids:
            try:
                res = collection.get(ids=list(set(all_neighbor_ids)))
                if res and res["documents"]:
                    for n_id, n_doc in zip(res["ids"], res["documents"]):
                        db_docs_map[n_id] = n_doc
            except Exception as ex:
                print(f"[RETRIEVER WARNING] Batch expansion failed: {ex}")
                
        # 5b. Build prompt contexts using the fetched chunks map
        for i, doc in enumerate(filtered_docs):
            doc_id = doc["id"]
            neighbor_ids = doc_neighbor_ids.get(doc_id, [])
            
            expanded_text = None
            if neighbor_ids:
                doc_tuples = []
                for n_id in neighbor_ids:
                    n_doc = db_docs_map.get(n_id)
                    if n_doc:
                        try:
                            idx = int(n_id.split("_chunk_")[-1])
                            doc_tuples.append((idx, n_doc))
                        except Exception:
                            pass
                if doc_tuples:
                    doc_tuples.sort(key=lambda x: x[0])
                    expanded_text = "\n".join([text for _, text in doc_tuples])
                    
            if not expanded_text:
                expanded_text = doc["text"]
                
            page_num = doc['metadata'].get('page_number', 'N/A')
            context_str += f"[{i+1}] (Source: {doc['metadata'].get('source', 'Unknown')}, Page: {page_num}):\n[Content of Page {page_num}]:\n{expanded_text}\n\n"
            
        t_expand = time.time() - t_expand_start
            
        prompt = f"""Use the following retrieved context chunks and conversation history to answer the user query.
{history_context}
Retrieved Contexts:
{context_str}

USER_QUERY:
{query_str}

AI Answer:
"""
        system_instruction = (
            "You are a retrieval-grounded assistant. You must answer using ONLY the "
            "information explicitly present in the provided context. Follow these rules strictly:\n\n"
            "1. Do NOT use any external knowledge, training data, or general facts, even "
            "if they seem related or helpful — only use what is written in the context.\n"
            "2. You MUST cite the relevant source number inside brackets (e.g. [1], [2]) at the end of claims. "
            "To avoid cluttering, do NOT repeat the same citation consecutively on every sentence or bullet point. If multiple consecutive sentences/bullets in a list refer to the exact same source, write the citation once at the end.\n"
            "3. If the context only PARTIALLY answers the question, answer only the part "
            "that is supported, and explicitly state: 'The provided material does not cover [specific missing part].'\n"
            "4. IMPORTANT: If the context does NOT answer the question at all, or is completely incomplete, "
            "you MUST start your response with 'REFUSAL:' followed by: 'I don't know. The document does not contain information to answer this question.'\n"
            "5. Never blend retrieved content with outside knowledge in the same answer, "
            "even if it would make the answer feel more complete.\n"
            "6. Every factual claim in your answer must be traceable to a specific sentence or section in the provided context. "
            "If you cannot point to where a claim came from, do not include it.\n"
            "7. Do not pad answers with generic advice, frameworks, or lists that are not explicitly present in the retrieved context, "
            "even if they are commonly associated with the topic.\n"
            "8. For any lists, rules, or bullet points in the context, you MUST preserve each item as its own separate, individual bullet point in your response. Do not merge separate bullet points together.\n"
            "9. Do NOT include any meta-commentary, apologies, or explanations about what is missing.\n"
            "10. Do NOT carry over, reuse, or repeat any facts or descriptions mentioned in the conversation history if they are not explicitly present in the new Retrieved Contexts. The new Retrieved Contexts always override any previous conversation history."
        )
        
        # 6. LLM Generation
        t_qa_start = time.time()
        llm = GroqClient(api_key=groq_api_key)
        answer = llm.generate_answer(prompt, model=RAG_MODEL, system_instruction=system_instruction)
        t_qa = time.time() - t_qa_start
        
        # Accumulate token usage
        if hasattr(llm, "last_usage") and llm.last_usage:
            SESSION_TOKEN_USAGE["prompt_tokens"] += llm.last_usage.get("prompt_tokens", 0)
            SESSION_TOKEN_USAGE["completion_tokens"] += llm.last_usage.get("completion_tokens", 0)
            SESSION_TOKEN_USAGE["total_tokens"] += llm.last_usage.get("total_tokens", 0)
            SESSION_TOKEN_USAGE["requests"] += 1
            SESSION_TOKEN_USAGE["model"] = llm.last_usage.get("model", SESSION_TOKEN_USAGE["model"])
        
        # 7. Handle out-of-context refusal
        if answer.strip().startswith("REFUSAL:"):
            clean_answer = answer.replace("REFUSAL:", "", 1).strip()
            t_db_start = time.time()
            log_qa_to_db(body.session_id, body.message_index, active_file, original_query, clean_answer)
            t_db = time.time() - t_db_start
            
            total_time = time.time() - t_start
            print(f"[RAG LATENCY] Query: '{original_query[:30]}...' | Embed: {t_embed:.2f}s | Retrieval: {t_retrieve:.2f}s | Context Expansion: {t_expand:.2f}s | LLM Gen: {t_qa:.2f}s | DB Log: {t_db:.2f}s | Total: {total_time:.2f}s")
            return {
                "answer": clean_answer,
                "sources": [],
                "eval_contexts": [],
                "observability": {
                    "cached": False
                }
            }
            
        # 8. Record in conversation memory
        session_history.append({"role": "user", "content": original_query})
        session_history.append({"role": "assistant", "content": answer})
        if len(session_history) > 10:
            session_history.pop(0)
            session_history.pop(0)

        # 9. Format sources for citation details UI
        sources = []
        for i, doc in enumerate(filtered_docs):
            source_name = doc['metadata'].get('source', 'Unknown')
            page_num = doc['metadata'].get('page_number', 'N/A')
            confidence = doc.get("confidence", "MEDIUM")
            similarity_pct = round(doc['similarity'] * 100, 1)
            
            sources.append({
                "index": i + 1,
                "source_name": source_name,
                "citation": f"[Source: {source_name}, page {page_num} — {similarity_pct}% Match ({confidence} CONFIDENCE)]",
                "text": doc['text'][:300],
                "page": page_num,
                "similarity": doc['similarity'],
                "confidence": confidence
            })
            
        # 10. Record in Semantic Cache
        eval_ctxs = [doc['text'] for doc in filtered_docs]
        SEMANTIC_CACHE.append({
            "query": original_query,
            "answer": answer,
            "sources": sources,
            "eval_contexts": eval_ctxs
        })
        if len(SEMANTIC_CACHE) > 20:
            SEMANTIC_CACHE.pop(0)
            
        # Log response in PG database
        t_db_start = time.time()
        log_qa_to_db(body.session_id, body.message_index, active_file, original_query, answer)
        t_db = time.time() - t_db_start
        
        total_time = time.time() - t_start
        print(f"[RAG LATENCY] Query: '{original_query[:30]}...' | Embed: {t_embed:.2f}s | Retrieval: {t_retrieve:.2f}s | Context Expansion: {t_expand:.2f}s | LLM Gen: {t_qa:.2f}s | DB Log: {t_db:.2f}s | Total: {total_time:.2f}s")

        return {
            "answer": answer,
            "sources": sources,
            "eval_contexts": eval_ctxs,  # FULL, untruncated contexts for evaluator
            "observability": {
                "cached": False
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Clear Chroma database collection, semantic cache, and memory
@app.post("/clear")
def clear_db():
    global SEMANTIC_CACHE, CONVERSATION_MEMORIES
    try:
        VECTOR_STORE.delete_collection(COLLECTION_NAME)
        SEMANTIC_CACHE.clear()
        CONVERSATION_MEMORIES.clear()
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Token usage stats endpoint
@app.get("/token_usage")
def token_usage():
    daily_limit = 500_000  # Groq free tier limit
    used = SESSION_TOKEN_USAGE["total_tokens"]
    remaining = max(0, daily_limit - used)
    pct_used = round((used / daily_limit) * 100, 1) if daily_limit > 0 else 0
    return {
        "provider": "groq",
        "model": SESSION_TOKEN_USAGE["model"],
        "prompt_tokens": SESSION_TOKEN_USAGE["prompt_tokens"],
        "completion_tokens": SESSION_TOKEN_USAGE["completion_tokens"],
        "total_used": used,
        "daily_limit": daily_limit,
        "remaining": remaining,
        "pct_used": pct_used,
        "requests": SESSION_TOKEN_USAGE["requests"]
    }

class EvaluationRequest(BaseModel):
    query: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None  # optional — only needed for reference-based metrics
    session_id: str | None = None
    message_index: int | None = None


@app.post("/evaluate")
@traceable(name="evaluate_rag_response", run_type="chain")
def evaluate_response(body: EvaluationRequest):
    # Handle case with empty contexts explicitly
    if not body.contexts:
        return {
            "faithfulness": None,
            "relevance": None,
            "context_precision": None,
            "context_relevance": None,
            "context_recall": None,
            "answer_correctness": None,
            "notes": [],
            "evaluation_model": EVALUATION_MODEL
        }

    has_gt = bool(body.ground_truth and body.ground_truth.strip())
    matched_from_file = False

    # Look up ground truth from the in-memory cache (loaded once at startup)
    if not has_gt and GROUND_TRUTH_DATA:
        query_key = body.query.strip().lower()
        matched_gt = None

        def _jaccard(a: str, b: str) -> float:
            wa = set(re.findall(r"\w+", a.lower()))
            wb = set(re.findall(r"\w+", b.lower()))
            if not wa or not wb:
                return 0.0
            return len(wa & wb) / len(wa | wb)

        best_score = 0.0
        for item in GROUND_TRUTH_DATA:
            if not isinstance(item, dict):
                continue
            gt_query = item.get("query", "").strip().lower()
            if gt_query == query_key:
                matched_gt = item.get("ground_truth") or item.get("answer")
                break
            score = _jaccard(query_key, gt_query)
            if score > best_score and score >= 0.5:
                best_score = score
                matched_gt = item.get("ground_truth") or item.get("answer")

        if matched_gt:
            body.ground_truth = matched_gt
            has_gt = True
            matched_from_file = True

    try:
        # Score query using custom multi-step evaluation script
        # Capping the contexts to top 3 elements reduces payload size and avoids TPM rate-limiting sleeps
        eval_contexts = body.contexts[:3]
        if has_gt:
            res = evaluation.evaluate_with_ground_truth(
                question=body.query,
                answer=body.answer,
                ground_truth=body.ground_truth,
                contexts=eval_contexts
            )
        else:
            res = evaluation.evaluate(
                question=body.query,
                answer=body.answer,
                contexts=eval_contexts
            )

        # Map returned score keys to matching database/UI properties
        scores = {
            "faithfulness": res.get("faithfulness"),
            "relevance": res.get("answer_relevancy"),
            "context_precision": res.get("context_precision"),
            "context_relevance": res.get("context_relevancy"),
            "context_recall": res.get("context_recall") if has_gt else None,
            "answer_correctness": res.get("answer_correctness") if has_gt else None,
        }

        notes = []

        # Update evaluation metrics in PostgreSQL database asynchronously
        if body.session_id and body.message_index is not None:
            def _async_update_eval_db(sess_id, msg_idx, score_dict):
                conn = get_db_connection()
                if conn:
                    try:
                        cur = conn.cursor()
                        update_query = """
                        UPDATE chat_logs
                        SET faithfulness = %s,
                            relevance = %s,
                            context_precision = %s,
                            context_relevance = %s,
                            context_recall = %s,
                            answer_correctness = %s
                        WHERE session_id = %s AND message_index = %s;
                        """
                        cur.execute(update_query, (
                            score_dict["faithfulness"],
                            score_dict["relevance"],
                            score_dict["context_precision"],
                            score_dict["context_relevance"],
                            score_dict["context_recall"],
                            score_dict["answer_correctness"],
                            sess_id,
                            msg_idx
                        ))
                        conn.commit()
                        cur.close()
                        conn.close()
                        print(f"[DATABASE] Updated evaluation metrics for session={sess_id} index={msg_idx}")
                    except Exception as db_err:
                        print(f"[DATABASE ERROR] Failed to update evaluation metrics: {db_err}")

            threading.Thread(target=_async_update_eval_db, args=(body.session_id, body.message_index, scores), daemon=True).start()

        return {
            **scores,
            "notes": notes,
            "evaluation_model": EVALUATION_MODEL
        }

    except Exception as e:
        err_msg = str(e)
        # Gracefully handle API rate limits and return HTTP 429
        if "rate_limit" in err_msg.lower() or "429" in err_msg or "too many requests" in err_msg.lower():
            raise HTTPException(status_code=429, detail="Groq Rate Limit Exceeded. Please wait a few seconds before trying again.")
        raise HTTPException(status_code=500, detail=f"LLM evaluator execution failed: {err_msg}")


if __name__ == "__main__":
    import uvicorn
    # Clean shutdown & binding port recovery
    uvicorn.run(app, host="0.0.0.0", port=8502)