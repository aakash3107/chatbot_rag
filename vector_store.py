"""
vector_store.py
Manages ChromaDB vector database with sentence-transformers embeddings.
FREE & OPEN SOURCE: ChromaDB (vector DB) + all-MiniLM-L6-v2 (embedding model)
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import logging
import os

logger = logging.getLogger(__name__)

# Global singletons (loaded once, reused across requests)
_embedding_model = None
_chroma_client = None
_collection = None

COLLECTION_NAME = "rag_documents"


def get_embedding_model(model_name="all-MiniLM-L6-v2"):
    """Load embedding model once and cache it."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {model_name} (first time may take ~30s)")
        _embedding_model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded!")
    return _embedding_model


def get_chroma_collection(db_path="./vectordb"):
    """Get or create ChromaDB collection."""
    global _chroma_client, _collection
    
    if _collection is None:
        os.makedirs(db_path, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # HNSW index with cosine similarity
        )
        logger.info(f"ChromaDB collection '{COLLECTION_NAME}' ready. "
                    f"Documents: {_collection.count()}")
    
    return _collection


def embed_texts(texts, model_name="all-MiniLM-L6-v2"):
    """Generate embeddings for a list of texts."""
    model = get_embedding_model(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    return embeddings.tolist()


def add_chunks_to_db(chunks, db_path="./vectordb", model_name="all-MiniLM-L6-v2"):
    """
    Embed chunks and store in ChromaDB.
    chunks: list of dicts with 'text', 'chunk_id', 'pdf_id', 'filename', 'page_num'
    """
    collection = get_chroma_collection(db_path)
    
    # Check which chunk IDs already exist (avoid duplicates)
    existing_ids = set()
    try:
        existing = collection.get(ids=[c["chunk_id"] for c in chunks])
        existing_ids = set(existing["ids"])
    except Exception:
        pass
    
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
    
    if not new_chunks:
        logger.info("All chunks already in DB, skipping.")
        return 0
    
    logger.info(f"Embedding {len(new_chunks)} new chunks...")
    
    # Batch process to avoid memory issues
    BATCH_SIZE = 100
    added = 0
    
    for i in range(0, len(new_chunks), BATCH_SIZE):
        batch = new_chunks[i:i+BATCH_SIZE]
        
        texts = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metadatas = [{
            "pdf_id": c["pdf_id"],
            "filename": c["filename"],
            "page_num": c["page_num"],
            "chunk_index": c["chunk_index"]
        } for c in batch]
        
        embeddings = embed_texts(texts, model_name)
        
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        added += len(batch)
        logger.info(f"  Added batch {i//BATCH_SIZE + 1}: {added}/{len(new_chunks)} chunks")
    
    logger.info(f"Done. Total chunks in DB: {collection.count()}")
    return added


def search_similar_chunks(query, top_k=5, db_path="./vectordb", model_name="all-MiniLM-L6-v2"):
    """
    Search for most relevant chunks given a query.
    Returns list of dicts with text, metadata, and similarity score.
    """
    collection = get_chroma_collection(db_path)
    
    if collection.count() == 0:
        return []
    
    # Embed the query
    model = get_embedding_model(model_name)
    query_embedding = model.encode([query]).tolist()
    
    # Query ChromaDB (HNSW ANN search)
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )
    
    # Format results
    chunks = []
    if results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            similarity = 1 - results["distances"][0][i]  # cosine: 1 - distance
            chunks.append({
                "text": results["documents"][0][i],
                "filename": results["metadatas"][0][i]["filename"],
                "page_num": results["metadatas"][0][i]["page_num"],
                "pdf_id": results["metadatas"][0][i]["pdf_id"],
                "score": round(similarity, 4)
            })
    
    return chunks


def get_db_stats(db_path="./vectordb"):
    """Return stats about the vector DB."""
    collection = get_chroma_collection(db_path)
    count = collection.count()
    
    # Get unique filenames
    if count > 0:
        all_meta = collection.get(include=["metadatas"])
        filenames = list({m["filename"] for m in all_meta["metadatas"]})
    else:
        filenames = []
    
    return {
        "total_chunks": count,
        "indexed_files": filenames,
        "total_files": len(filenames)
    }


def delete_pdf_from_db(pdf_id, db_path="./vectordb"):
    """Remove all chunks for a given PDF from the DB."""
    collection = get_chroma_collection(db_path)
    collection.delete(where={"pdf_id": pdf_id})
    logger.info(f"Deleted all chunks for pdf_id={pdf_id}")
