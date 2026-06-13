# 🔍 RAG Chatbot — Complete Setup Guide

## What This Project Does
A Retrieval-Augmented Generation (RAG) chatbot that:
1. Ingests large PDFs (text + OCR for scanned pages)
2. Chunks text into passages and embeds them using a FREE local model
3. Stores embeddings in ChromaDB (FREE, open-source vector DB)
4. On each question: retrieves relevant chunks → feeds to Claude → returns answer WITH citations

---

## 🗂️ Project Structure
```
rag_chatbot/
├── app.py                    ← Main Flask server
├── requirements.txt          ← Python dependencies
├── .env                      ← Your API keys and config
├── utils/
│   ├── pdf_processor.py      ← PDF text extraction + OCR
│   ├── chunker.py            ← Split text into overlapping chunks
│   ├── vector_store.py       ← ChromaDB + sentence-transformers
│   └── rag_pipeline.py       ← Retrieval + reranking + generation
├── templates/
│   └── index.html            ← Web UI
├── uploads/                  ← PDFs are saved here
└── vectordb/                 ← ChromaDB files stored here
```

---

## ⚙️ Step-by-Step Setup

### Step 1: Install Python (if not installed)
- Download Python 3.10 or 3.11 from https://python.org
- During install: ✅ check "Add Python to PATH"
- Verify: open Terminal/CMD and run: `python --version`

### Step 2: Install Tesseract OCR (for scanned PDFs)

**Windows:**
1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install (default path: `C:\Program Files\Tesseract-OCR\`)
3. Add to PATH or the app auto-detects it

**Mac:**
```bash
brew install tesseract
```

**Ubuntu/Linux:**
```bash
sudo apt-get install tesseract-ocr
```

### Step 3: Create Virtual Environment
Open Terminal in the `rag_chatbot/` folder and run:

```bash
# Create virtual environment
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# You should see (venv) at the start of your terminal prompt
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```
⚠️ This will take 5-10 minutes (downloads PyTorch + sentence-transformers).

### Step 5: Set Your API Key
Open `.env` file in any text editor and replace:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```
with your actual key from https://console.anthropic.com

### Step 6: Run the Application
```bash
python app.py
```
You should see:
```
RAG Chatbot Starting...
  Embedding Model : all-MiniLM-L6-v2
  Vector DB Path  : ./vectordb
  ...
Open http://localhost:5000 in your browser
```

### Step 7: Open the Web UI
Go to: **http://localhost:5000**

---

## 🚀 Using the Chatbot

1. **Upload PDFs**: Click the left panel or drag PDFs onto it
2. **Click "Ingest into Vector DB"**: Processes PDFs (extract → chunk → embed → store)
3. **Ask questions**: Type in the chat box and press Enter
4. **See results**: Answer appears with sources; right panel shows retrieved chunks

---

## 🏗️ Architecture Explained (for Interview)

```
User Query
    │
    ▼
[Embed Query] ──→ sentence-transformers (all-MiniLM-L6-v2, FREE, local)
    │
    ▼
[ANN Search] ──→ ChromaDB HNSW Index (cosine similarity, FREE, local)
    │
    ▼
[Top-K Chunks Retrieved] ──→ With filename + page metadata
    │
    ▼
[Rerank] ──→ Lexical keyword boost (semantic 80% + lexical 20%)
    │
    ▼
[LLM Generation] ──→ Claude Sonnet (Anthropic API)
    │                  System prompt includes chunks as context
    ▼
[Answer + Citations] ──→ PDF name + page number for each source
```

### Key Tech Choices:
| Component | Tool | Why |
|-----------|------|-----|
| PDF Extraction | PyMuPDF (fitz) | Fast, reliable, handles complex layouts |
| OCR | Tesseract | Free, runs locally, good accuracy |
| Embedding Model | all-MiniLM-L6-v2 | FREE, 384-dim, fast (22ms/query), good quality |
| Vector DB | ChromaDB | FREE, open-source, persistent, HNSW index |
| LLM | Claude Sonnet | Fast, accurate, good at following instructions |
| Web Framework | Flask | Simple, lightweight |

---

## 📊 Latency Breakdown (typical)
- Embed query: ~20ms
- ChromaDB ANN search: ~5ms
- Claude API response: ~1500-2500ms
- **Total: ~1.5-3 seconds** ✅ (within 2-5s target)

---

## 🐛 Troubleshooting

**"ModuleNotFoundError"**: Make sure your virtualenv is activated (`source venv/bin/activate`)

**"ANTHROPIC_API_KEY not set"**: Edit `.env` and add your real API key

**"Tesseract not found"**: Install Tesseract OCR (Step 2 above). For Windows, also add to PATH.

**Slow first query**: First run downloads embedding model (~90MB). Subsequent queries are fast.

**Port 5000 in use**: Change port in `app.py` last line: `app.run(port=5001)`
