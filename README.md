# Local RAG System with Citation Support

## Architecture Overview

This system consists of two n8n workflows that provide a complete local RAG (Retrieval-Augmented Generation) solution with **exact citations** for every piece of generated information.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INGESTION WORKFLOW                                 │
│                                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  Cron/   │───▶│  Hash    │───▶│  Delete  │───▶│  Extract │              │
│  │  Trigger │    │  Check   │    │  Old     │    │  w/Meta  │              │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│                                                         │                   │
│                                         ┌───────────────┼───────────────┐   │
│                                         ▼               ▼               ▼   │
│                                    ┌─────────┐    ┌─────────┐    ┌─────────┐│
│                                    │  PDF    │    │  Office │    │  Excel  ││
│                                    │  Pages  │    │  Paras  │    │  Rows   ││
│                                    └────┬────┘    └────┬────┘    └────┬────┘│
│                                         └───────────────┼───────────────┘   │
│                                                         ▼                   │
│                                    ┌──────────┐    ┌──────────┐            │
│                                    │  Smart   │───▶│  Qdrant  │            │
│                                    │  Chunker │    │  Insert  │            │
│                                    └──────────┘    └──────────┘            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           RETRIEVAL WORKFLOW                                 │
│                                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  Open    │───▶│  Query   │───▶│  Vector  │───▶│  Local   │              │
│  │  WebUI   │    │  Router  │    │  Search  │    │  Rerank  │              │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│                       │                                │                    │
│                       ▼                                ▼                    │
│                  ┌──────────┐                    ┌──────────┐              │
│                  │  Direct  │                    │  Context │              │
│                  │  Answer  │                    │  Assembly│              │
│                  └────┬─────┘                    └────┬─────┘              │
│                       │                               │                    │
│                       │         ┌──────────┐          │                    │
│                       └────────▶│  LLM     │◀─────────┘                    │
│                                 │  + Cite  │                               │
│                                 └────┬─────┘                               │
│                                      ▼                                     │
│                                 ┌──────────┐    ┌──────────┐              │
│                                 │  Format  │───▶│  Open    │              │
│                                 │  Response│    │  WebUI   │              │
│                                 └──────────┘    └──────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Smart Ingestion Workflow

### Key Features

1. **Metadata-Rich Extraction**
   - PDFs: Page-level extraction with `page_number` metadata
   - Office/Text: Paragraph-level with `paragraph` number
   - Spreadsheets: Row-level with `sheet_name` and `row` number

2. **Change Detection (Idempotency)**
   - MD5 hash calculation for every file
   - Postgres tracking table for processed files
   - Automatic deletion of old vectors before re-ingestion

3. **Smart Chunking**
   - Recursive character text splitter
   - Configurable chunk size (default: 512 tokens)
   - Preserves full citation metadata in every chunk

### Node Details

#### Calculate File Hash
```javascript
// Calculates MD5 hash for change detection
const crypto = require('crypto');
const fileBuffer = fs.readFileSync(filePath);
const hashSum = crypto.createHash('md5');
hashSum.update(fileBuffer);
const fileHash = hashSum.digest('hex');
```

#### PDF Page Extraction
```javascript
// Extracts text per page with page number metadata
const pdfData = await pdf(dataBuffer, {
  pagerender: function(pageData) {
    currentPage++;
    return pageData.getTextContent().then(textContent => {
      pages.push({
        pageNumber: currentPage,
        text: extractedText
      });
    });
  }
});
```

#### Vector Payload Structure
```json
{
  "pageContent": "The safety valve pressure is 50psi...",
  "metadata": {
    "source": "/data/documents/manual.pdf",
    "fileName": "manual.pdf",
    "page": 15,
    "contentType": "pdf",
    "citationString": "[Source: manual.pdf, Page: 15]",
    "fileHash": "a1b2c3d4e5f6..."
  }
}
```

---

## Part 2: Agentic Retrieval Workflow

### Key Features

1. **Query Router (7B Optimized)**
   - Heuristic-first approach minimizes LLM calls
   - Pattern matching for greetings/simple queries
   - JSON-mode LLM routing for ambiguous queries

2. **Local Reranking**
   - Hybrid scoring without external APIs
   - Vector similarity (50%) + Keyword overlap (25%) + Term frequency (15%) + Length (10%)

3. **Citation-Aware Generation**
   - Strict system prompt for source citation
   - Every fact must include `[Source: filename, Page: X]`

### Context Assembly Format

The Context Assembly node formats retrieved documents like this:

```
[ID: 1] [Source: manual.pdf, Page: 15]
Content: The safety valve pressure is 50psi. Always check the pressure gauge before operation.

---

[ID: 2] [Source: data.xlsx, Sheet: Q3 Report, Row: 4]
Content: **Row 4 from Q3 Report:**
- **Quarter**: Q3
- **Revenue**: $500k
- **Growth**: 15%

---

[ID: 3] [Source: procedures.docx, Paragraph: 12]
Content: Maintenance should be performed every 30 days according to the schedule outlined in section 4.2.
```

### System Prompt for Citation

```
You are a helpful assistant that answers questions using ONLY the provided context.

CRITICAL CITATION RULES:
1. You MUST cite the Source and Page/Row for EVERY fact you state
2. Format citations as [Source: filename, Page: X] at the END of each sentence
3. Do NOT reference "ID: 1" - use human-readable source information
4. If context doesn't contain relevant info, say "I don't have information about that"
5. Never make up information not in the context
```

### Example LLM Output

**User Query:** "What is the safety valve pressure?"

**LLM Response:**
> The safety valve pressure is 50psi [Source: manual.pdf, Page: 15]. You should always check the pressure gauge before operation to ensure it's within the acceptable range [Source: manual.pdf, Page: 15].

---

## Setup Instructions

### Prerequisites

1. **Docker Services Required:**
   - n8n (workflow automation)
   - Ollama (local LLM inference)
   - Qdrant (vector database)
   - PostgreSQL (state management)

2. **Ollama Models:**
   ```bash
   ollama pull llama3.1:8b-instruct-q4_K_M
   ollama pull mistral:7b-instruct-q4_K_M
   ollama pull nomic-embed-text:latest
   ```

### Database Setup

```sql
-- File hash tracking table
CREATE TABLE IF NOT EXISTS file_hashes (
  id SERIAL PRIMARY KEY,
  file_path TEXT UNIQUE NOT NULL,
  file_hash VARCHAR(32) NOT NULL,
  processed_at TIMESTAMP DEFAULT NOW(),
  chunk_count INTEGER DEFAULT 0
);

CREATE INDEX idx_file_path ON file_hashes(file_path);
CREATE INDEX idx_file_hash ON file_hashes(file_hash);

-- Chat memory table (for n8n Postgres Chat Memory)
CREATE TABLE IF NOT EXISTS n8n_chat_histories (
  id SERIAL PRIMARY KEY,
  session_id TEXT NOT NULL,
  message JSONB NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_session_id ON n8n_chat_histories(session_id);
```

### Qdrant Collection Setup

```bash
curl -X PUT 'http://localhost:6333/collections/documents' \
  -H 'Content-Type: application/json' \
  -d '{
    "vectors": {
      "size": 768,
      "distance": "Cosine"
    }
  }'
```

### Open WebUI Configuration

Configure Open WebUI to use your n8n webhook as the API endpoint:

1. Go to Settings → Connections
2. Add Custom Model:
   - Name: `local-rag`
   - Base URL: `http://n8n:5678/webhook/rag-chat`
   - Model ID: `local-rag-llama3.1`

---

## Performance Optimization

### For 7B Models

1. **Quantization:** Use `q4_K_M` quantization for best speed/quality balance
2. **Context Length:** Limit to 2048 tokens for faster inference
3. **Temperature:** Use 0.1-0.3 for factual responses
4. **Top-K Results:** Retrieve 10, rerank to 5 for context

### Memory Usage

| Component | Estimated RAM |
|-----------|--------------|
| Llama 3.1 8B (q4) | ~5GB |
| Mistral 7B (q4) | ~4GB |
| nomic-embed-text | ~500MB |
| Qdrant | ~1GB (10k docs) |

---

## Troubleshooting

### Common Issues

1. **Empty search results:**
   - Verify Qdrant collection exists and has data
   - Check embedding model is running
   - Verify collection name matches in both workflows

2. **Missing citations:**
   - Ensure metadata is preserved through chunking
   - Check Context Assembly node is formatting correctly
   - Verify system prompt is being applied

3. **Slow response times:**
   - Reduce top-K results from 10 to 5
   - Use smaller quantization (q4_0 vs q4_K_M)
   - Enable GPU acceleration in Ollama

### Debug Mode

Enable detailed logging in Code nodes:
```javascript
console.log('Debug:', JSON.stringify(items[0].json, null, 2));
```

---

## File Structure

```
multimodal-rag/
├── local_rag_ingestion.json    # Smart Ingestion Workflow
├── local_rag_retrieval.json    # Agentic Retrieval Workflow
├── local_rag_simple.json       # Original simple workflow
├── README.md                   # This documentation
└── docker-compose.yml          # Infrastructure setup
```
