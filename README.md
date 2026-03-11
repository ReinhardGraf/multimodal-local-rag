# Multimodal RAG System

100% local AI knowledge base with image recognition and citation support. No cloud, no external APIs — fully GDPR-compliant.

## What This Does

This system lets you ask natural-language questions about your internal documents (PDF, DOCX) and get answers with exact source citations and embedded images — all running locally.

### Document Ingestion

Documents are automatically processed through a pipeline:

1. **n8n** detects new/changed files and triggers ingestion
2. **Docling** analyzes documents — extracts text, images, and runs OCR
3. **Backend (FastAPI)** chunks text with HybridChunker and creates embeddings (multilingual-e5-large-instruct, 1024 dim) plus BM25 sparse vectors
4. **Qdrant** stores the vectors with metadata (source, page, image references)
5. **PostgreSQL** stores extracted images and file hashes for idempotent re-ingestion

### Question & Answer

1. User asks a question via **Open WebUI**
2. **n8n** receives the query and routes it to a **Qwen3-4B** query router
3. If RAG is needed: the **Backend** performs hybrid search (dense + sparse) against Qdrant, reranks results, resolves image placeholders, and assembles context
4. **Qwen3-4B Instruct (Q4)** generates an answer with source citations and embedded images
5. If RAG is not needed: the query goes directly to Ollama (no retrieval)

## Components

| Service | Description | Port |
|---------|-------------|------|
| **n8n** | Workflow automation — file detection & query routing | 5678 |
| **Backend** | FastAPI + Docling — chunking, embedding, hybrid search | 5008 |
| **Qdrant** | Vector database (dense + sparse vectors + metadata) | 6333 |
| **PostgreSQL** | Image storage, file hashes, chat history | 5432 (internal), 5436 (host) |
| **Ollama** | Local LLM inference (Qwen3-4B Instruct) | 11434 |
| **Open WebUI** | Chat interface | 3000 |

## Setup

### Prerequisites

- Docker and Docker Compose
- GPU with at least 16 GB VRAM recommended (e.g. NVIDIA RTX 4080)
- 32 GB+ system RAM

### 1. Clone and configure

```bash
git clone <repo-url> && cd multimodal-rag
cp .env.example .env
# Edit .env to adjust passwords, ports, or model choices
```

### 2. Choose your Docker Compose mode

#### Production (everything in Docker)

All services including Ollama and the Backend run inside Docker containers. This is the default.

```bash
docker compose up -d
```

Use `docker-compose.yml`. All containers communicate over the internal `rag-network` — no `host.docker.internal` needed.

#### Development (Backend + Ollama running locally)

Use this when you want to run the Backend and Ollama natively on your host (e.g. on macOS to access the GPU directly):

```bash
docker compose -f docker-compose.dev.yml up -d
```

This starts only n8n, PostgreSQL, Qdrant, and Open WebUI in Docker. You then run Ollama and the Backend yourself:

```bash
# Start Ollama natively
ollama serve

# Start the Backend
cd backend
pip install -e .
uvicorn src.main:app --host 0.0.0.0 --port 5008
```

In dev mode, update your `.env` to point containers at the host:

```
OLLAMA_BASE_URL_INTERNAL=http://host.docker.internal:11434
OLLAMA_BASE_URL_WEBUI=http://host.docker.internal:11434
```

### 3. Run the setup script

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Start Docker services
- Wait for all services to become healthy
- Pull the configured Ollama models
- Create the Qdrant collection
- Pre-download the HuggingFace tokenizer

### 4. Import the n8n workflow

1. Open n8n at `http://localhost:5678`
2. Go to **Workflows → Import**
3. Import `Multimodal RAG.json`
4. Configure credentials for Ollama, Qdrant, and PostgreSQL

### 5. Add documents

Place your PDF/DOCX files into the `documents/` folder and activate the ingestion workflow in n8n.

## Web UIs

| UI | URL | Purpose |
|----|-----|---------|
| **n8n** | http://localhost:5678 | Workflow editor and monitoring |
| **Open WebUI** | http://localhost:3000 | Chat interface for asking questions |
| **Qdrant Dashboard** | http://localhost:6333/dashboard | Inspect vector collections and data |

## Hardware Requirements

- **GPU**: 16 GB+ VRAM (NVIDIA RTX 4080 or similar)
- **RAM**: 32 GB+
- **OS**: Linux, macOS, or Windows
