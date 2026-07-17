# Sanjeevani AI – RAG Service

> **Enterprise-grade Retrieval-Augmented Generation microservice for scientific research papers.**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-1.12-red.svg)](https://qdrant.tech)
[![Gemini](https://img.shields.io/badge/Nvidia-green.svg)](https://nvidia.com)

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Tech Stack](#tech-stack)
3. [Project Structure](#project-structure)
4. [Quick Start (Docker)](#quick-start-docker)
5. [Quick Start (Local)](#quick-start-local)
6. [Configuration](#configuration)
7. [API Reference](#api-reference)
8. [RAG Pipeline](#rag-pipeline)
9. [Testing](#testing)
10. [Adding a New LLM / Embedding Provider](#extending)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI REST API                          │
│  /health  /documents/index  /retrieve  /chat                │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
┌───────▼──────┐             ┌────────▼────────┐
│  Indexing    │             │   Chat Service   │
│  Service     │             │                  │
│              │             │ Retrieve → Rerank│
│ Load → Chunk │             │ → Prompt → LLM   │
│ → Embed      │             │ → Citations      │
└──────┬───────┘             └────────┬─────────┘
       │                              │
       ▼                              ▼
┌─────────────────────────────────────────────┐
│              Qdrant Vector DB                │
│         Collection: research_documents       │
│     HNSW Index | Cosine Similarity           │
└─────────────────────────────────────────────┘
       │                              │
       ▼                              ▼
┌──────────────┐             ┌────────────────┐
│   OpenAI     │             │    OpenAI      │
│  Embeddings  │             │   Chat LLM     │
│  (3-small)   │             │   (GPT-4o)     │
└──────────────┘             └────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| Framework | FastAPI 0.115 |
| Vector DB | Qdrant 1.12 |
| Embeddings | OpenAI text-embedding-3-small (1536-dim) |
| LLM | OpenAI GPT-4o (configurable) |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| PDF Parsing | PyMuPDF (fitz) |
| DOCX Parsing | python-docx |
| Tokenization | tiktoken (cl100k_base) |
| Retry Logic | tenacity |
| Containerisation | Docker + docker-compose |
| Testing | pytest + pytest-asyncio |

---

## Project Structure

```
rag-service/
├── app/
│   ├── api/
│   │   ├── health.py          # GET /health
│   │   ├── documents.py       # POST /documents/index, batch-index, DELETE
│   │   ├── retrieval.py       # POST /retrieve
│   │   └── chat.py            # POST /chat
│   ├── config/
│   │   └── settings.py        # Pydantic-Settings (env vars)
│   ├── loaders/
│   │   ├── pdf_loader.py      # PyMuPDF + heading heuristics
│   │   ├── txt_loader.py      # Plain text + section detection
│   │   └── docx_loader.py     # python-docx + style-based headings
│   ├── chunking/
│   │   └── chunker.py         # Recursive char splitter (600 tok / 100 overlap)
│   ├── embeddings/
│   │   └── embedding_service.py  # OpenAI + Gemini stub
│   ├── vectordb/
│   │   ├── qdrant_client.py   # AsyncQdrantClient + HNSW collection
│   │   └── repository.py      # Upsert / search / delete / count
│   ├── retrieval/
│   │   ├── retriever.py       # Top-K + MMR semantic search
│   │   └── reranker.py        # Score normalisation + term-overlap boost
│   ├── prompts/
│   │   └── prompt_builder.py  # Context assembly (token-budgeted)
│   ├── llm/
│   │   └── openai_client.py   # Async chat completions + retry
│   ├── services/
│   │   ├── indexing_service.py  # Orchestrates indexing pipeline
│   │   └── chat_service.py      # Orchestrates RAG chat pipeline
│   ├── schemas/
│   │   ├── document.py        # Pydantic v2 document schemas
│   │   └── chat.py            # Pydantic v2 chat schemas
│   ├── utils/
│   │   └── tokenizer.py       # tiktoken count + truncate
│   └── main.py                # FastAPI app + lifespan
├── tests/
│   ├── conftest.py
│   ├── test_health.py
│   ├── test_chunker.py
│   ├── test_prompt_builder.py
│   ├── test_retriever.py
│   └── test_tokenizer.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── .env.example
```

---

## Quick Start (Docker)

### Prerequisites
- Docker Desktop 4.x+
- OpenAI API key

### 1. Clone and configure

```bash
git clone <repo-url>
cd rag-service
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### 2. Launch services

```bash
docker-compose up -d --build
```

### 3. Verify

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "Sanjeevani RAG Service",
  "version": "1.0.0"
}
```

---

## Quick Start (Local)

### Prerequisites
- Python 3.12
- Qdrant running locally (see below)

### 1. Start Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant:v1.12.4
```

### 2. Install dependencies

```bash
cd rag-service
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Set OPENAI_API_KEY in .env
```

### 4. Run the service

```bash
python -m uvicorn app.main:app --reload --port 8000
```

### 5. Open interactive docs

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Configuration

All configuration is done via environment variables (`.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | **required** | OpenAI API key |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `OPENAI_CHAT_MODEL` | `gpt-4o` | Chat completion model |
| `OPENAI_TEMPERATURE` | `0.2` | LLM sampling temperature |
| `OPENAI_MAX_TOKENS` | `4096` | Max completion tokens |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant HTTP port |
| `QDRANT_COLLECTION_NAME` | `research_documents` | Collection name |
| `CHUNK_SIZE` | `600` | Chunk size in tokens |
| `CHUNK_OVERLAP` | `100` | Chunk overlap in tokens |
| `RETRIEVAL_TOP_K` | `10` | Default chunks to retrieve |
| `RETRIEVAL_SIMILARITY_THRESHOLD` | `0.65` | Minimum similarity score |
| `MMR_LAMBDA` | `0.5` | MMR relevance/diversity balance |
| `RERANKER_TOP_N` | `5` | Chunks sent to LLM after reranking |
| `MAX_FILE_SIZE_MB` | `50` | Maximum upload file size |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## API Reference

### `GET /health`

Returns service health status and Qdrant connectivity.

```bash
curl http://localhost:8000/health
```

---

### `POST /documents/index`

Index a single research paper.

```bash
curl -X POST http://localhost:8000/documents/index \
  -F "file=@paper.pdf" \
  -F "title=Attention Is All You Need"
```

**Response:**
```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "title": "Attention Is All You Need",
  "total_chunks": 87,
  "pages": 15,
  "processing_time_ms": 4231.5,
  "status": "indexed"
}
```

---

### `POST /documents/batch-index`

Index multiple files at once.

```bash
curl -X POST http://localhost:8000/documents/batch-index \
  -F "files=@paper1.pdf" \
  -F "files=@paper2.docx" \
  -F "files=@notes.txt"
```

---

### `DELETE /documents/{document_id}`

Remove a document and all its chunks.

```bash
curl -X DELETE http://localhost:8000/documents/550e8400-e29b-41d4-a716-446655440000
```

---

### `POST /retrieve`

Retrieve relevant chunks without generating an LLM answer.

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the limitations of transformer models?",
    "top_k": 5,
    "similarity_threshold": 0.65,
    "use_mmr": true
  }'
```

---

### `POST /chat`

Full RAG chat – retrieves context and generates a grounded answer.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key contributions of the BERT architecture?",
    "top_k": 10,
    "similarity_threshold": 0.65,
    "use_mmr": true,
    "document_ids": ["optional-filter-by-doc-id"]
  }'
```

**Response:**
```json
{
  "question": "What are the key contributions of the BERT architecture?",
  "answer": "According to 'BERT: Pre-training of Deep Bidirectional Transformers' (page 2, section Introduction): BERT introduces bidirectional pre-training for language representation...",
  "citations": [
    {
      "document_id": "...",
      "title": "BERT Paper",
      "page": 2,
      "section": "Introduction",
      "chunk_text": "...",
      "similarity_score": 0.923
    }
  ],
  "total_chunks_retrieved": 8,
  "processing_time_ms": 2341.7,
  "llm_model": "gpt-4o",
  "embedding_model": "text-embedding-3-small"
}
```

---

## RAG Pipeline

### Indexing Pipeline

```
Document File (PDF/TXT/DOCX)
        ↓
   Load & Parse
   (PyMuPDF / python-docx / plain text)
        ↓
   Extract Text with Metadata
   (page numbers, headings, sections)
        ↓
   Clean Text
   (de-hyphenate, normalise whitespace)
        ↓
   Recursive Character Chunking
   (600 tokens / 100 overlap)
        ↓
   Generate Embeddings
   (OpenAI text-embedding-3-small, batch=100)
        ↓
   Upsert to Qdrant
   (HNSW, Cosine similarity, payload indexes)
```

### Chat Pipeline

```
User Question
        ↓
   Generate Query Embedding
   (OpenAI text-embedding-3-small)
        ↓
   Search Qdrant
   (Top-K or MMR, with optional document_id filter)
        ↓
   Re-rank Results
   (similarity + term-overlap scoring)
        ↓
   Build Prompt
   (SYSTEM + Context + Question + Instructions)
        ↓
   Send to LLM
   (OpenAI GPT-4o, max 4096 tokens)
        ↓
   Return Answer + Citations
   (answer text, citation list, timing stats)
```

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_chunker.py -v

# Run without coverage (faster)
pytest --no-cov
```

---

## Extending

### Adding a New Embedding Provider

1. Create a subclass of `BaseEmbeddingProvider` in `app/embeddings/embedding_service.py`
2. Implement `embed_texts()`, `embed_query()`, `model_name`, and `dimensions`
3. Add a new value to `EmbeddingProvider` enum in `app/config/settings.py`
4. Register the provider in `create_embedding_provider()`

### Adding a New LLM Provider

1. Create a subclass of `BaseLLMClient` in `app/llm/openai_client.py`
2. Implement `complete()` and `model_name`
3. Add a new value to `LLMProvider` enum in `app/config/settings.py`
4. Register it in `create_llm_client()`

---

## License

MIT © Sanjeevani AI Team
