# Verbesserte Workflow-Version fuer Multimodales Lokales RAG

Stand: 2026-05-01

Diese Notiz dokumentiert die Aenderungen gegenueber der urspruenglichen Version des lokalen multimodalen RAG-Systems. Sie ergaenzt die bestehende Betriebsreferenz und beschreibt vor allem die neue n8n-Workflow-JSON sowie das angepasste Restart-Skript.

## Ausgangspunkt

Die urspruengliche Version war bereits lokal und hybrid aufgebaut:

- Ollama nativ auf macOS mit Metal-GPU
- FastAPI-Backend nativ auf Port `5008`
- PostgreSQL, Qdrant, n8n und Open WebUI in Docker
- Open WebUI spricht ueber eine Pipe Function mit einem n8n-Webhook
- n8n ruft Backend und Ollama ueber `host.docker.internal` auf

Die urspruenglichen Standardmodelle waren:

| Aufgabe | Urspruengliches Modell |
|---|---|
| Chat / QA | `qwen3:4b-instruct-2507-q4_K_M` |
| Query Router | `qwen3:4b-instruct-2507-q4_K_M` |
| Embedding | `qllama/multilingual-e5-large-instruct:latest` |
| Reranking | `BAAI/bge-reranker-v2-m3` bzw. Backend/Ollama-Reranker |

Die wichtigste Schwaeche lag bei der Bildsuche: Bilder wurden zwar gespeichert und ueber `[IMAGE:uuid]` in Antworten eingebettet, aber die Suche nach Bildern beruhte weitgehend auf Caption-Text oder Seitenkontext. Wenn ein Bild keine gute Beschriftung hatte, war es semantisch nur schwach auffindbar.

## Neue Artefakte

| Datei | Zweck |
|---|---|
| `Multimodal RAG Image Table improved.json` | Neue n8n-Workflow-Variante zum Importieren und Testen |
| `rag-restart.sh` | Ueberarbeitetes Start-/Stop-Skript passend zur neuen Workflow-Version |
| `rag-restart.sh.codex-bak` | Backup der vorherigen Skriptversion |

Die urspruengliche Workflow-JSON wurde nicht ueberschrieben. Die neue JSON ist als Testvariante gedacht.

## Aenderungen in der n8n-Workflow-JSON

### Staerkeres Antwortmodell

Die QA-Nodes verwenden nun:

```text
qwen3:14b
```

Der Query Router bleibt bewusst klein:

```text
qwen3:4b-instruct-2507-q4_K_M
```

Grund: Routing soll schnell bleiben; Antwortgenerierung, Quellenlogik und deutsche Zusammenfassung profitieren aber von einem groesseren Modell.

### Lokale Vision-Beschreibung fuer Bilder

Bei der Ingestion wird jedes extrahierte Bild zusaetzlich lokal mit einem Vision-Modell beschrieben:

```text
qwen2.5vl:latest
```

Die Bildbeschreibung wird direkt in den Embedding-Text geschrieben. Dadurch kann die spaetere Suche Bilder nicht nur ueber Captions oder Umgebungstext finden, sondern ueber sichtbare Inhalte wie:

- Objekte
- Personen
- Diagramme
- Screenshots
- Tabellen
- OCR/Text im Bild
- Achsen, Zahlen und Layout
- fachliche Begriffe

Der neue Embedding-Text fuer Bild-Chunks folgt sinngemaess dieser Struktur:

```text
[Image]
[Vision] Beschreibung des sichtbaren Bildinhalts
[Caption] vorhandene Caption/Children-Texte aus Docling
[Kontext] umgebender Seitenkontext
```

Das ist der wichtigste Schritt von "Bilder werden angezeigt" zu "Bilder werden semantisch suchbar".

### Einheitliche Logik fuer neue und geaenderte Dateien

Vorher war `Extract & Chunk (New)` bereits besser als `Extract & Chunk (Modified)`. Geaenderte Dateien konnten dadurch bei Bildern wieder schlechter verarbeitet werden.

Die neue Version gleicht beide Nodes an:

| Node | Neu |
|---|---|
| `Extract & Chunk (New)` | Vision-Beschreibung + Caption + Kontext |
| `Extract & Chunk (Modified)` | dieselbe Logik wie bei neuen Dateien |

Damit bleibt die Bildqualitaet auch nach Re-Ingest oder Datei-Aenderung stabil.

### Hoeheres Retrieval-Limit

Die Dokumentensuche wurde erweitert:

| Parameter | Vorher | Nachher |
|---|---:|---:|
| Dokument-Retrieval `limit` | 5 | 15 |
| Tabellen-Retrieval `limit` | 3 | 8 |
| Tabellen-Reranking | `false` | `true` |

Grund: Bei Bild- und Tabellenfragen ist ein zu enges erstes Retrieval oft der Engpass. Besser ist: mehr Kandidaten holen, dann reranken.

### Besserer Query Router fuer Bildfragen

Der Router erkennt nun Bild-, Screenshot-, Diagramm- und Chart-Fragen expliziter als `vector`-Route. Beispiele:

- "Zeig mir das Bild zu Anspruchsdenken"
- "Welche Grafik erklaert die Architektur?"
- "Gibt es einen Screenshot zur Konfiguration?"
- "Wo ist das Diagramm mit den Tabellen?"

Vorher konnten solche Fragen leichter als direkte allgemeine Frage oder normale Textfrage behandelt werden.

### Angepasster Antwortprompt

Der Antwortprompt wurde entschaerft:

- statt maximal 200 Woerter nun maximal 350 Woerter bei komplexeren Bild-/Tabellenfragen
- Bildmarker muessen direkt nach dem passenden Absatz stehen
- relevante `[IMAGE:uuid]`-Marker sollen nicht gesammelt am Ende erscheinen

## Aenderungen in `rag-restart.sh`

Das Restart-Skript ist nun auf die neue Workflow-Variante abgestimmt.

### Konfigurierte Modelle

| Aufgabe | Modell |
|---|---|
| Router | `qwen3:4b-instruct-2507-q4_K_M` |
| QA | `qwen3:14b` |
| Vision | `qwen2.5vl:latest` |
| Embedding | `qllama/multilingual-e5-large-instruct:latest` |
| Reranker | `qllama/bge-reranker-v2-m3:q4_k_m` |

Das Embedding-Modell wurde bewusst nicht automatisch auf `bge-m3` umgestellt. Ein Embedding-Wechsel ist kein reiner Modellwechsel, sondern erfordert Re-Indexing und muss zur Qdrant-Vektordimension passen.

### Neue Option: `--pull-models`

Das Skript prueft nun, ob die Pflichtmodelle vorhanden sind. Wenn sie fehlen, gibt es zwei Wege:

```bash
./rag-restart.sh
```

bricht mit Hinweis auf fehlende Modelle ab.

```bash
./rag-restart.sh --pull-models
```

zieht fehlende Ollama-Modelle automatisch nach.

### Warmup

Das Skript laedt nun mehrere Modelle mit `keep_alive=-1` vor:

- Router-Modell
- QA-Modell
- Vision-Modell
- Embedding-Modell

Dadurch sollen die Modelle dauerhaft warm bleiben und nicht zwischen einzelnen Anfragen entladen werden.

### Status-Ausgabe

`./rag-restart.sh --status` zeigt nun zusaetzlich die konfigurierten Modelle an:

```text
Router
QA
Vision
Embedding
Reranker
```

## Neuer empfohlener Start

Beim ersten Start nach der Umstellung:

```bash
cd /Users/reini/02_Zweites_Gehirn/Repos/multimodal-local-rag
./rag-restart.sh --pull-models
```

Danach im Normalbetrieb:

```bash
./rag-restart.sh
```

Status:

```bash
./rag-restart.sh --status
```

Stop:

```bash
./rag-restart.sh --stop
```

## Docker-Fehler vom 2026-05-01

Beim Start trat diese Meldung auf:

```text
unable to get image 'pgvector/pgvector:pg16':
failed to connect to the docker API at unix:///Users/reini/.docker/run/docker.sock;
check if the path is correct and if the daemon is running:
dial unix /Users/reini/.docker/run/docker.sock: connect: no such file or directory
```

Das ist kein Fehler der neuen n8n-Workflow-JSON und kein Modellproblem. Die Meldung bedeutet: Docker Desktop bzw. der Docker-Daemon laeuft nicht oder der Socket existiert gerade nicht.

Pruefung:

```bash
docker info
```

Wenn das fehlschlaegt:

1. Docker Desktop starten.
2. Warten, bis Docker wirklich "Running" ist.
3. Erneut ausfuehren:

```bash
cd /Users/reini/02_Zweites_Gehirn/Repos/multimodal-local-rag
./rag-restart.sh
```

Wenn Docker Desktop sichtbar laeuft, aber der Socket weiter fehlt:

```bash
ls -la ~/.docker/run/docker.sock
docker context ls
docker context use desktop-linux
```

Danach erneut `docker info` pruefen.

## Was sich fachlich verbessert

| Bereich | Vorher | Nachher |
|---|---|---|
| Textfragen | lokales Qwen3-4B | staerkeres Qwen3-14B fuer Antworten |
| Router | Qwen3-4B | bleibt schnell mit Qwen3-4B |
| Bildsuche | Caption/Seitenkontext | Vision-Beschreibung + Caption + Kontext |
| Geaenderte Dateien | schlechtere Bildlogik moeglich | gleiche Bildlogik wie neue Dateien |
| Tabellen | wenige Kandidaten, kein Rerank | mehr Kandidaten, Rerank aktiv |
| Modellstart | zwei Modelle vorgewarmt | Router, QA, Vision, Embedding vorgewarmt |
| Fehlende Modelle | spaeter Laufzeitfehler moeglich | fruehe Pruefung, optional Auto-Pull |

## Erwarteter Nutzen

Die groesste Verbesserung liegt bei Suchanfragen nach visuellen Inhalten. Bisher musste ein Bild ueber seinen Textkontext gefunden werden. Jetzt wird das Bild selbst beim Einlesen beschrieben. Dadurch werden auch folgende Fragen realistischer:

- "Zeig mir das Bild mit dem Cartoon zu Anspruchsdenken."
- "Wo ist die Grafik zur Architektur?"
- "Welche Abbildung zeigt den Ablauf?"
- "Gibt es eine Tabelle oder ein Diagramm zu diesem Thema?"

## Grenzen der neuen Version

Die neue Version ist noch kein vollstaendiges echtes multimodales Embedding-System. Die Bildsuche laeuft weiterhin ueber Textvektoren, nur ist dieser Text jetzt viel besser, weil er aus einer lokalen Vision-Beschreibung des Bildes stammt.

Echte Text-zu-Bild-Vektorsuche mit CLIP/Jina-CLIP waere ein weiterer Ausbauschritt. Dafuer braeuchte es:

- eigene Bildvektor-Collection oder Multi-Vector-Setup
- Bildembedding-Modell
- Query-Pfad fuer Text-zu-Bild-Suche
- Ergebnis-Merging mit normalem Dokument-RAG

## Naechste Schritte

- [ ] Neue Workflow-JSON in n8n importieren und als Testworkflow aktivieren
- [ ] `qwen3:14b` und `qwen2.5vl:latest` per `./rag-restart.sh --pull-models` installieren
- [ ] Docker Desktop starten und Docker-Fehler verifizieren
- [ ] Einen kleinen PDF-Testbestand neu ingestieren
- [ ] Bildfragen gegen alte und neue Workflow-Version vergleichen
- [ ] Wenn stabil: alte Workflow-Version archivieren
- [ ] Optional spaeter: echte CLIP/Jina-CLIP-Bildvektoren planen

## Log

- 2026-04-30: Neue n8n-Workflow-JSON mit Qwen3-14B, Qwen2.5-VL-Bildbeschreibung, hoeherem Retrieval-Limit und Tabellen-Reranking erstellt.
- 2026-04-30: `rag-restart.sh` auf neue Modelle und `--pull-models` erweitert.
- 2026-05-01: Docker-Startfehler als Docker-Daemon-/Socket-Problem dokumentiert.
