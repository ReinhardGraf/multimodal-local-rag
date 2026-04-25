# Multimodal Local RAG — Architektur

> Technische Übersicht über den Aufbau des lokalen RAG-Systems.

---

## Grundprinzip

Das System ist eine **hybride Architektur**:

| Schicht | Wo läuft es | Warum |
|---|---|---|
| **LLM-Inferenz** | Nativ auf macOS | Volle Metal-GPU-Beschleunigung |
| **Datenbanken & Workflows** | Docker-Container | Reproduzierbarkeit, Isolation |
| **Backend (FastAPI)** | Nativ auf macOS | Einfaches Debugging, Hot-Reload |

Verbindung zwischen den Welten: `host.docker.internal` ermöglicht Container-zu-Host-Kommunikation.

---

## Komponenten-Übersicht

### Native Komponenten (auf macOS)

| Komponente | Port | Funktion |
|---|---|---|
| **Ollama** | 11434 | LLM-Inferenz mit Metal-GPU |
| **Backend (FastAPI)** | 5008 | RAG-Logik, Chunking, Embeddings, Search |

### Container-Komponenten (Docker)

| Container | Port | Funktion |
|---|---|---|
| **Postgres (pgvector)** | 5436 | n8n-Daten, Bilder, File-Hashes |
| **Qdrant** | 6333 / 6334 | Vector-Datenbank für Embeddings |
| **n8n** | 5678 | Workflow-Engine für Ingestion & Routing |
| **Open WebUI** | 3000 | Chat-Frontend |

---

## Geladene Modelle

| Modell | Typ | Größe | Funktion |
|---|---|---|---|
| `qwen3:4b-instruct-2507-q4_K_M` | LLM | ~4 GB | Antwort-Generierung, Query-Routing |
| `qllama/multilingual-e5-large-instruct` | Embedding | 605 MB | Semantische Vektorisierung (1024 dim) |

Beide Modelle bleiben dauerhaft im VRAM (`keep_alive: -1`).

---

## Datenfluss: Document Ingestion

```
PDF / DOCX
    ↓
n8n erkennt neue Datei
    ↓
Backend (FastAPI)
    ├─ Docling: Text + Bilder + OCR
    ├─ HybridChunker: semantische Chunks (~512 Tokens)
    └─ Embedding via Ollama (multilingual-e5)
    ↓
Speicherung
    ├─ Qdrant: Vektoren (dense 1024 + sparse BM25) + Metadaten
    └─ Postgres: Bilder, File-Hash (Idempotenz)
```

---

## Datenfluss: Question & Answer

```
User-Frage in Open WebUI
    ↓
n8n empfängt Query
    ↓
Qwen3-4B Query-Router entscheidet:
    ├─ RAG nötig? → Backend → Hybrid-Search in Qdrant
    │                 ├─ Dense-Vektoren (semantisch)
    │                 ├─ Sparse-Vektoren (Stichwort)
    │                 └─ CrossEncoder-Reranking
    └─ RAG nicht nötig? → direkt zu Ollama
    ↓
Qwen3-4B generiert Antwort
    ├─ mit Quellenangabe (Datei, Seite)
    └─ mit eingebetteten Bildern bei Bedarf
    ↓
Antwort in Open WebUI
```

---

## Backend-Endpoints (Port 5008)

### Schreiben (Ingestion)

| Endpoint | Funktion |
|---|---|
| `POST /v1/chunk/hierarchical/file` | PDF/DOCX in Chunks zerlegen |
| `POST /v1/vector-store/upsert` | Vektoren in Qdrant speichern |
| `POST /v1/tables/ingest` | Excel-Tabellen verarbeiten |

### Lesen (Suche)

| Endpoint | Funktion |
|---|---|
| `POST /v1/vector-store/search` | Hybrid-Search auf Dokumenten |
| `POST /v1/tables/query` | Tabellen-Daten befragen |
| `GET /v1/file-hashes/count` | Statistik über ingestierte Dateien |

### System

| Endpoint | Funktion |
|---|---|
| `GET /health` | System-Selbstdiagnose |
| `GET /v1/warmup` | Modelle vorladen |
| `POST /v1/reconciliation/file-hash` | Idempotenz-Prüfung |
| `POST /v1/tables/delete-by-source` | DSGVO-konformes Löschen |

---

## Qdrant: Vector-Konfiguration

Aktuelle Collection `documents`:

| Eigenschaft | Wert |
|---|---|
| Vektor-Modus | Hybrid (dense + sparse) |
| Dense-Dimension | 1024 |
| Distance-Metrik | Cosine |
| Sparse-Vektor | BM25-basiert |
| Status | GREEN (optimiert) |

---

## Netzwerk-Topologie

```
                 ┌─────────────────────────────┐
                 │       macOS Host             │
                 │                              │
                 │   Ollama (11434) ◄───┐      │
                 │   Backend (5008) ◄──┐│      │
                 └─────────────────────┼┼──────┘
                                       ││
                          host.docker.internal
                                       ││
                 ┌─────────────────────┼┼──────┐
                 │   Docker Network    ││       │
                 │   (rag-network)     ││       │
                 │                     ││       │
                 │   ┌─────────┐  ┌────▼▼───┐  │
                 │   │ n8n     │──┤open-WebUI│  │
                 │   └────┬────┘  └─────────┘  │
                 │        │                    │
                 │   ┌────▼────┐  ┌────────┐  │
                 │   │postgres │  │ qdrant │  │
                 │   └─────────┘  └────────┘  │
                 └────────────────────────────┘
```

---

## Persistente Daten (Docker Volumes)

| Volume | Inhalt |
|---|---|
| `n8n_data` | Workflows, Credentials |
| `postgres_data` | Datenbankinhalt |
| `qdrant_data` | Vektor-Index, Snapshots |
| `open_webui_data` | Chat-Verläufe, User-Settings |

Volumes überleben `docker compose down`. Nur `docker compose down -v` löscht sie.

---

## DSGVO-Eigenschaften

| Aspekt | Status |
|---|---|
| Externe API-Calls | Keine |
| Cloud-Abhängigkeiten | Keine |
| Datenspeicherung | Lokal auf Mac |
| Modelle | Lokal (kein Hugging Face Pull bei Inferenz) |
| Telemetrie | n8n-Diagnostics deaktiviert |
| Löschrecht | `delete-by-source`-Endpoint |

DSGVO Art. 25 ("Privacy by Design") als Architektur, nicht als Versprechen.

---

## Compose-Modi

| Datei | Zielsystem | Ollama läuft |
|---|---|---|
| `docker-compose.yml` | Linux + NVIDIA-GPU | Im Container |
| **`docker-compose.dev.yml`** | **macOS mit Metal-GPU** | **Nativ** |

Auf macOS immer `docker-compose.dev.yml` verwenden.

---

*Stand: 25.04.2026*
