"""
rag_pipeline.py
Core RAG pipeline: retrieve → rerank → generate answer with citations.
Uses Anthropic Claude for generation (fast, accurate, API-based).
"""

import anthropic
import os
import time
import logging
from utils.vector_store import search_similar_chunks

logger = logging.getLogger(__name__)

# Initialize Anthropic client
_anthropic_client = None


def get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key or api_key == "your_anthropic_api_key_here":
            raise ValueError("ANTHROPIC_API_KEY not set in .env file!")
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


def rerank_chunks(query, chunks):
    """
    Lightweight lexical reranker: boost chunks containing query keywords.
    (Cross-encoder reranking would be more powerful but slower.)
    """
    if not chunks:
        return chunks
    
    query_words = set(query.lower().split())
    # Remove common stop words
    stop_words = {"a", "an", "the", "is", "are", "was", "were", "in", "on",
                  "at", "to", "for", "of", "and", "or", "but", "with", "what",
                  "how", "why", "when", "where", "who", "which", "that", "this"}
    query_keywords = query_words - stop_words
    
    def keyword_boost(chunk):
        text_lower = chunk["text"].lower()
        keyword_hits = sum(1 for kw in query_keywords if kw in text_lower)
        # Combine semantic score (80%) + keyword boost (20%)
        return chunk["score"] * 0.8 + (keyword_hits / max(len(query_keywords), 1)) * 0.2
    
    reranked = sorted(chunks, key=keyword_boost, reverse=True)
    return reranked


def build_context_string(chunks):
    """Format retrieved chunks into a context block for the LLM."""
    context_parts = []
    
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['filename']}, Page {chunk['page_num']} "
            f"(relevance: {chunk['score']:.2f})]\n{chunk['text']}"
        )
    
    return "\n\n---\n\n".join(context_parts)


def generate_answer(query, chunks, conversation_history=None):
    """
    Generate an answer using Claude with retrieved context.
    Returns: {answer, sources, latency_ms, chunks_used}
    """
    start_time = time.time()
    
    if not chunks:
        return {
            "answer": "I couldn't find relevant information in the uploaded documents to answer your question. Please make sure you've uploaded PDFs and try rephrasing your query.",
            "sources": [],
            "latency_ms": 0,
            "chunks_used": 0
        }
    
    context = build_context_string(chunks)
    
    system_prompt = """You are a helpful RAG (Retrieval-Augmented Generation) assistant that answers questions based ONLY on the provided document context.

INSTRUCTIONS:
1. Answer the user's question using ONLY the information in the provided context.
2. Always cite your sources at the end: mention the exact PDF filename and page number.
3. If the context doesn't contain enough information, say so clearly — don't hallucinate.
4. Be concise but complete. Use bullet points for lists.
5. Format citations like: [Source: filename.pdf, Page X]

CONTEXT FROM DOCUMENTS:
""" + context

    # Build messages (with optional conversation history for multi-turn)
    messages = []
    if conversation_history:
        messages.extend(conversation_history[-6:])  # Last 3 turns max
    messages.append({"role": "user", "content": query})
    
    client = get_anthropic_client()
    
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system_prompt,
        messages=messages
    )
    
    answer_text = response.content[0].text
    latency_ms = int((time.time() - start_time) * 1000)
    
    # Build deduplicated sources list
    seen = set()
    sources = []
    for chunk in chunks:
        key = (chunk["filename"], chunk["page_num"])
        if key not in seen:
            seen.add(key)
            sources.append({
                "filename": chunk["filename"],
                "page_num": chunk["page_num"],
                "score": chunk["score"]
            })
    
    logger.info(f"Answer generated in {latency_ms}ms using {len(chunks)} chunks")
    
    return {
        "answer": answer_text,
        "sources": sources,
        "latency_ms": latency_ms,
        "chunks_used": len(chunks)
    }


def query_rag(user_query, top_k=5, conversation_history=None,
              db_path="./vectordb", model_name="all-MiniLM-L6-v2"):
    """
    Full RAG pipeline:
    1. Embed query
    2. Retrieve top-K chunks from ChromaDB
    3. Rerank with keyword boost
    4. Generate answer with Claude
    
    Returns full result dict including retrieved chunks for visualization.
    """
    overall_start = time.time()
    
    # Step 1: Retrieve
    logger.info(f"Retrieving top-{top_k} chunks for query: '{user_query[:60]}...'")
    raw_chunks = search_similar_chunks(
        query=user_query,
        top_k=top_k + 2,  # Fetch extra, will rerank down
        db_path=db_path,
        model_name=model_name
    )
    
    # Step 2: Rerank
    reranked_chunks = rerank_chunks(user_query, raw_chunks)[:top_k]
    
    # Step 3: Generate
    result = generate_answer(user_query, reranked_chunks, conversation_history)
    
    total_latency = int((time.time() - overall_start) * 1000)
    result["total_latency_ms"] = total_latency
    result["retrieved_chunks"] = reranked_chunks  # For visualization panel
    
    logger.info(f"Total RAG latency: {total_latency}ms")
    return result
