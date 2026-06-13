"""
chunker.py
Splits page text into overlapping chunks with full metadata.
"""

import re
import logging

logger = logging.getLogger(__name__)


def estimate_tokens(text):
    """Rough token estimate: ~4 chars per token (good enough for chunking)."""
    return len(text) // 4


def split_into_sentences(text):
    """Split text into sentences for clean chunk boundaries."""
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text, chunk_size_tokens=800, overlap_tokens=150):
    """
    Split text into chunks with token-based sizing and overlap.
    Returns list of (chunk_text, start_offset, end_offset).
    """
    sentences = split_into_sentences(text)
    
    if not sentences:
        return []
    
    chunks = []
    current_chunk_sentences = []
    current_tokens = 0
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_tokens = estimate_tokens(sentence)
        
        if current_tokens + sentence_tokens > chunk_size_tokens and current_chunk_sentences:
            # Save current chunk
            chunk_text = ' '.join(current_chunk_sentences)
            chunks.append(chunk_text)
            
            # Backtrack for overlap: keep last N tokens worth of sentences
            overlap_sentences = []
            overlap_count = 0
            for s in reversed(current_chunk_sentences):
                t = estimate_tokens(s)
                if overlap_count + t > overlap_tokens:
                    break
                overlap_sentences.insert(0, s)
                overlap_count += t
            
            # Start new chunk from overlap
            current_chunk_sentences = overlap_sentences
            current_tokens = overlap_count
        else:
            current_chunk_sentences.append(sentence)
            current_tokens += sentence_tokens
            i += 1
    
    # Don't forget the last chunk
    if current_chunk_sentences:
        chunks.append(' '.join(current_chunk_sentences))
    
    return chunks


def create_chunks_from_pages(pages_data, chunk_size=800, overlap=150):
    """
    Takes list of page dicts and returns list of chunk dicts with full metadata.
    
    Each chunk dict:
    {
        "text": str,
        "pdf_id": str,
        "filename": str,
        "page_num": int,
        "chunk_index": int,
        "chunk_id": str   (unique ID for ChromaDB)
    }
    """
    all_chunks = []
    chunk_global_index = 0
    
    for page_data in pages_data:
        page_chunks = chunk_text(
            page_data["text"],
            chunk_size_tokens=chunk_size,
            overlap_tokens=overlap
        )
        
        for local_idx, chunk_text_content in enumerate(page_chunks):
            if len(chunk_text_content.strip()) < 50:
                continue  # Skip tiny chunks
            
            chunk_id = f"{page_data['pdf_id']}_p{page_data['page_num']}_c{local_idx}"
            
            all_chunks.append({
                "text": chunk_text_content,
                "pdf_id": page_data["pdf_id"],
                "filename": page_data["filename"],
                "page_num": page_data["page_num"],
                "chunk_index": chunk_global_index,
                "chunk_id": chunk_id
            })
            chunk_global_index += 1
    
    logger.info(f"Created {len(all_chunks)} chunks from {len(pages_data)} pages")
    return all_chunks
