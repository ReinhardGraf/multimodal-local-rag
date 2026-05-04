# Multimodal Local RAG — Start & Stop

> Schnelle Befehlsreferenz für den täglichen Betrieb des lokalen RAG-Stacks.

---

## Voraussetzungen (einmalig prüfen)

| Komponente | Befehl zur Prüfung | Erwartete Ausgabe |
|---|---|---|
| Docker Desktop | `docker info` | Server-Informationen |
| Ollama | `ollama --version` | `ollama version 0.x.x` |
| uv | `uv --version` | `uv 0.11.x` |
| Repo-Pfad | `ls ~/02_Zweites_Gehirn/Repos/multimodal-local-rag/rag-restart.sh` | Datei vorhanden |

---

## Standard-Befehle

```bash
cd ~/02_Zweites_Gehirn/Repos/multimodal-local-rag
```

### Hochfahren

```bash
./rag-restart.sh
```

Startet automatisch:

1. Natives Ollama (mit `keep_alive: -1`)
2. Router-, QA-, Vision- und Embedding-Modelle werden geprüft und vorgeladen
3. Docker-Stack: Postgres, Qdrant, n8n, Open WebUI
4. Backend-Start in neuem Terminal-Tab, wenn das Terminal das unterstützt

### Backend manuell starten

In neuem Terminal-Tab:

```bash
cd ~/02_Zweites_Gehirn/Repos/multimodal-local-rag/backend
uv run uvicorn src.main:app --reload --port 5008
```

### Status prüfen

```bash
./rag-restart.sh --status
```

Zeigt:

- geladene Ollama-Modelle
- Docker-Container
- Backend-Prozess
- Alle Endpoints

### Herunterfahren

```bash
./rag-restart.sh --stop
```

Stoppt:

1. Backend (uvicorn)
2. Docker-Container (alle 4)
3. Natives Ollama

### Mac komplett herunterfahren

```bash
./rag-restart.sh --stop
osascript -e 'quit app "Docker"'
sudo shutdown -h now
```

---

## Browser-Endpoints

| Service | URL | Funktion |
|---|---|---|
| Open WebUI | http://localhost:3000 | Chat-Frontend |
| n8n | http://localhost:5678 | Workflow-Editor |
| Qdrant Dashboard | http://localhost:6333/dashboard | Vector-DB-UI |
| Backend Swagger | http://localhost:5008/docs | API-Dokumentation |
| Ollama | http://localhost:11434 | (nur API, keine UI) |

---

## Optionen des Restart-Skripts

| Befehl | Zweck |
|---|---|
| `./rag-restart.sh` | Voller Neustart (Standard) |
| `./rag-restart.sh --status` | Nur Status anzeigen |
| `./rag-restart.sh --stop` | Sauber runterfahren |
| `./rag-restart.sh --no-backend` | Restart ohne Backend-Tab |
| `./rag-restart.sh --pull-models` | Fehlende Ollama-Modelle automatisch ziehen |
| `./rag-restart.sh --help` | Hilfe anzeigen |

---

## Cold-Start nach Reboot

Nach einem Mac-Neustart:

1. Mac einschalten, einloggen
2. Docker Desktop manuell starten (Spotlight: ⌘+Space → "Docker")
3. Warten, bis 🐳-Icon "Docker Desktop is running" zeigt (~30 Sekunden)
4. Terminal öffnen
5. ```bash
   cd ~/02_Zweites_Gehirn/Repos/multimodal-local-rag && ./rag-restart.sh
   ```

---

## Log-Dateien

| Datei | Inhalt |
|---|---|
| `~/rag-restart.log` | Skript-Aktivität (Zeitstempel pro Schritt) |
| `~/ollama.log` | Ollama-Server-Output |
| Backend-Tab | uvicorn-Logs live |

---

## Schnell-Verifikation nach Start

```bash
ollama ps                              # Konfigurierte Modelle geladen?
docker ps                              # 4 Container "Up"?
curl -s http://localhost:5008/health   # Backend antwortet?
curl -s http://localhost:3000/api/version  # Open WebUI antwortet?
```

Alle vier Befehle sollten Daten liefern, kein Fehler.

Wenn `./rag-restart.sh` beim Docker-Schritt abbricht, läuft meist Docker Desktop noch nicht.

---

*Stand: 25.04.2026*
