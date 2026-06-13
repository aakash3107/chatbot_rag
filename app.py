"""
app.py — Main Flask Application for RAG Chatbot
Run: python app.py
"""

import os
import uuid
import json
import time
import logging
from pathlib import Path

from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# --- Flask App Setup ---
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management
CORS(app)

# --- Config from .env ---
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "./uploads")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./vectordb")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
TOP_K = int(os.getenv("TOP_K_RESULTS", 5))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

ALLOWED_EXTENSIONS = {"pdf"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main chat UI."""
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload_pdf():
    """
    Upload one or more PDFs → extract → chunk → embed → store in ChromaDB.
    This is the ingestion pipeline endpoint.
    """
    from utils.pdf_processor import extract_pdf
    from utils.chunker import create_chunks_from_pages
    from utils.vector_store import add_chunks_to_db

    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files")
    results = []

    for file in files:
        if not file or not allowed_file(file.filename):
            results.append({"filename": file.filename, "status": "error",
                            "message": "Not a valid PDF"})
            continue

        filename = file.filename
        pdf_id = str(uuid.uuid4())[:8]  # Short unique ID
        save_path = os.path.join(UPLOAD_FOLDER, f"{pdf_id}_{filename}")

        try:
            # 1. Save file
            file.save(save_path)
            file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
            logger.info(f"Saved: {filename} ({file_size_mb:.1f} MB)")

            # 2. Extract text (native + OCR fallback)
            pages_data = extract_pdf(save_path, pdf_id, filename)

            if not pages_data:
                results.append({
                    "filename": filename, "status": "error",
                    "message": "Could not extract text from PDF"
                })
                continue

            # 3. Chunk
            chunks = create_chunks_from_pages(
                pages_data,
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP
            )

            # 4. Embed & store in ChromaDB
            added = add_chunks_to_db(
                chunks,
                db_path=CHROMA_DB_PATH,
                model_name=EMBEDDING_MODEL
            )

            results.append({
                "filename": filename,
                "pdf_id": pdf_id,
                "status": "success",
                "pages_extracted": len(pages_data),
                "chunks_created": len(chunks),
                "chunks_added_to_db": added,
                "file_size_mb": round(file_size_mb, 2)
            })
            logger.info(f"Ingested {filename}: {len(pages_data)} pages → {len(chunks)} chunks")

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}", exc_info=True)
            results.append({
                "filename": filename, "status": "error", "message": str(e)
            })

    return jsonify({"results": results})


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    RAG query endpoint.
    Receives user question, runs full RAG pipeline, returns answer + sources.
    """
    from utils.rag_pipeline import query_rag

    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field"}), 400

    user_query = data["query"].strip()
    if not user_query:
        return jsonify({"error": "Empty query"}), 400

    # Retrieve conversation history from session
    if "history" not in session:
        session["history"] = []

    conversation_history = session["history"]

    try:
        result = query_rag(
            user_query=user_query,
            top_k=TOP_K,
            conversation_history=conversation_history,
            db_path=CHROMA_DB_PATH,
            model_name=EMBEDDING_MODEL
        )

        # Update conversation history for multi-turn
        conversation_history.append({"role": "user", "content": user_query})
        conversation_history.append({"role": "assistant", "content": result["answer"]})
        # Keep last 10 turns max
        session["history"] = conversation_history[-10:]

        return jsonify({
            "answer": result["answer"],
            "sources": result["sources"],
            "retrieved_chunks": result["retrieved_chunks"],
            "latency_ms": result["total_latency_ms"],
            "chunks_used": result["chunks_used"]
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return jsonify({"error": f"Internal error: {str(e)}"}), 500


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Return vector DB stats: total chunks, indexed files."""
    from utils.vector_store import get_db_stats
    try:
        stats = get_db_stats(CHROMA_DB_PATH)
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/clear_history", methods=["POST"])
def clear_history():
    """Clear conversation history for the current session."""
    session["history"] = []
    return jsonify({"message": "Conversation history cleared"})


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "embedding_model": EMBEDDING_MODEL,
        "vector_db": CHROMA_DB_PATH,
        "upload_folder": UPLOAD_FOLDER
    })


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("RAG Chatbot Starting...")
    logger.info(f"  Embedding Model : {EMBEDDING_MODEL}")
    logger.info(f"  Vector DB Path  : {CHROMA_DB_PATH}")
    logger.info(f"  Upload Folder   : {UPLOAD_FOLDER}")
    logger.info(f"  Chunk Size      : {CHUNK_SIZE} tokens")
    logger.info(f"  Chunk Overlap   : {CHUNK_OVERLAP} tokens")
    logger.info(f"  Top-K Retrieval : {TOP_K}")
    logger.info("=" * 60)
    logger.info("Open http://localhost:5000 in your browser")
    
    app.run(debug=True, host="0.0.0.0", port=5000)
