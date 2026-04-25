# Multimodal Local RAG — Troubleshooting

> Diagnose und Behebung typischer Probleme im laufenden Betrieb.

---

## Erste Diagnose: Status prüfen

```bash
cd ~/02_Zweites_Gehirn/Repos/multimodal-local-rag
./rag-restart.sh --status
```

Zeigt auf einen Blick:

- Ollama-Modelle (sollten beide auf "Forever" stehen)
- Docker-Container (alle 4 sollten "Up" sein)
- Backend-Prozess (sollte als uvicorn-PID sichtbar sein)

---

## Problem 1: Modelle entladen sich nach kurzer Zeit

**Symptom:** `ollama ps` zeigt z.B. "9 minutes from now" statt "Forever".

**Ursache:** `OLLAMA_KEEP_ALIVE=-1` wurde nicht gesetzt, oder das Modell wurde vor dem Setzen geladen.

**Lösung A — Schnellfix per API:**

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3:4b-instruct-2507-q4_K_M",
  "keep_alive": -1
}'

curl http://localhost:11434/api/embeddings -d '{
  "model": "qllama/multilingual-e5-large-instruct:latest",
  "prompt": "warmup",
  "keep_alive": -1
}'
```

**Lösung B — Persistent über Reboots:**

```bash
launchctl setenv OLLAMA_KEEP_ALIVE -1
```

Anschließend Ollama neu starten (über das Restart-Skript).

---

## Problem 2: Open WebUI ist plötzlich weg (Exit-Code 137)

**Symptom:** `docker ps -a | grep open-webui` zeigt "Exited (137)".

**Ursache:** SIGKILL durch Out-Of-Memory — Docker Desktop hat den Container abgeschossen wegen RAM-Knappheit.

**Lösung:**

```bash
docker start open-webui
```

**Vermeidung:** Beim Beenden von Ollama (z.B. `kill -9`) entsteht eine Speicher-Spitze. Lieber das Restart-Skript nutzen, das die Reihenfolge sauber orchestriert.

---

## Problem 3: "Orphan Containers" Warnung

**Symptom:** Beim Compose-Start erscheint:

```
WARN[0000] Found orphan containers ([rag-backend ollama]) for this project.
```

**Ursache:** Container aus früheren Compose-Versionen (Production-Modus) sind übrig geblieben.

**Lösung — einmaliges Aufräumen:**

```bash
docker compose -f docker-compose.yml down --remove-orphans
```

Beachten: Das beendet **auch** den aktuellen Stack. Danach mit `./rag-restart.sh` neu starten.

---

## Problem 4: Bash-Skript meldet "bad substitution"

**Symptom:** `${variable^^}: bad substitution`

**Ursache:** macOS liefert Bash 3.2 (von 2007). Modernere Bash-4-Syntax funktioniert nicht.

**Lösung im Skript:** `${var^^}` ersetzen durch:

```bash
$(echo "$var" | tr "[:lower:]" "[:upper:]")
```

**Alternative:** Moderne Bash via Homebrew installieren:

```bash
brew install bash
```

---

## Problem 5: `uv: command not found`

**Symptom:** Backend-Start scheitert mit `zsh: command not found: uv`.

**Ursache:** uv ist nicht installiert oder nicht im PATH.

**Lösung — Installation:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**PATH dauerhaft setzen:**

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

**Verifizieren:**

```bash
uv --version
```

---

## Problem 6: Ollama-Server lässt sich nicht beenden

**Symptom:** Nach `osascript -e 'quit app "Ollama"'` läuft der Server-Prozess noch.

**Ursache:** GUI-App und Server-Prozess (`ollama serve`) sind getrennt. Die GUI-Quit-Aktion beendet nur das Menüleisten-Icon.

**Lösung:**

```bash
ps aux | grep -i ollama | grep -v grep   # PID herausfinden
kill <PID>                                # Sanftes Beenden
kill -9 <PID>                             # Hartes Beenden, falls nötig
```

Verifikation:

```bash
lsof -i :11434                            # Sollte leer sein
```

Das Restart-Skript handhabt das automatisch.

---

## Problem 7: Backend startet, aber Endpoints geben 500-Fehler

**Symptom:** http://localhost:5008/docs zeigt zwar Swagger UI, aber Aufrufe schlagen fehl.

**Mögliche Ursachen:**

| Symptom in den Logs | Ursache | Lösung |
|---|---|---|
| `Connection refused` (Port 6333) | Qdrant noch nicht bereit | 30 Sek warten |
| `Connection refused` (Port 5436) | Postgres noch nicht bereit | 30 Sek warten |
| `Connection refused` (Port 11434) | Ollama nicht gestartet | `./rag-restart.sh --status` |
| `Module not found` | Dependency fehlt | `cd backend && uv sync` |

**Diagnose:** Backend-Tab anschauen (uvicorn schreibt detaillierte Stack-Traces).

---

## Problem 8: Open WebUI zeigt keine Modelle

**Symptom:** http://localhost:3000 öffnet sich, aber im Modell-Dropdown sind keine Einträge.

**Ursache:** Open WebUI erreicht das native Ollama nicht.

**Diagnose im Browser:**

```
http://localhost:3000/api/version  → muss antworten
```

**Test der Ollama-Verbindung aus Container:**

```bash
docker exec open-webui curl -s http://host.docker.internal:11434/api/tags
```

Sollte JSON mit Modell-Liste zurückgeben. Falls nicht, ist `host.docker.internal` nicht aufgelöst.

**Lösung:**

```bash
docker compose -f docker-compose.dev.yml restart open-webui
```

---

## Problem 9: Qdrant-Status nicht GREEN

**Symptom:** Im Dashboard zeigt eine Collection "yellow" oder "red".

| Status | Bedeutung | Aktion |
|---|---|---|
| 🟢 GREEN | Optimal indexiert | Nichts zu tun |
| 🟡 YELLOW | Optimierung läuft | 1-5 Min warten |
| 🔴 RED | Fehler im Index | Logs prüfen, ggf. Container neu starten |

**Logs einsehen:**

```bash
docker logs qdrant --tail 50
```

---

## Problem 10: Mac wird nach Restart-Skript langsam

**Symptom:** System reagiert träge, Lüfter aktiv.

**Ursache:** Beide LLM-Modelle sind im VRAM, Docker-Container laufen, Backend läuft — kann je nach Mac-RAM eng werden.

**Diagnose:** Activity Monitor öffnen, Speicher-Tab prüfen.

**Lösung — Modelle nur bei Bedarf laden:**

```bash
# Modell explizit entladen
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3:4b-instruct-2507-q4_K_M",
  "keep_alive": 0
}'
```

Bei Bedarf wieder mit `keep_alive: -1` laden.

---

## Notfall-Sequenz: Komplett-Reset

Wenn nichts mehr funktioniert:

```bash
# 1. Alles stoppen
./rag-restart.sh --stop

# 2. Alle Container und Networks aufräumen
docker compose -f docker-compose.yml down --remove-orphans
docker compose -f docker-compose.dev.yml down --remove-orphans

# 3. Verwaiste Prozesse killen
pkill -9 -f ollama
pkill -9 -f uvicorn

# 4. Docker Desktop neu starten
osascript -e 'quit app "Docker"'
sleep 5
open -a Docker

# 5. Warten bis Docker bereit ist (~30 Sek)

# 6. Frischer Start
./rag-restart.sh
```

⚠️ **Achtung:** `docker compose down -v` würde **Volumes löschen** — das vernichtet alle Daten in Postgres, Qdrant, n8n. Niemals ohne Backup!

---

## Backup-Empfehlung

Wichtige Daten zur regelmäßigen Sicherung:

| Quelle | Befehl |
|---|---|
| Qdrant | Snapshots im Dashboard erstellen → `qdrant_data/snapshots/` |
| Postgres | `docker exec postgres pg_dump -U n8n n8n_rag > backup.sql` |
| n8n-Workflows | UI → Workflows exportieren als JSON |
| Open WebUI Chats | UI → Settings → Export |

---

## Log-Inspektion

| Log | Befehl |
|---|---|
| Ollama | `tail -f ~/ollama.log` |
| Restart-Skript | `tail -f ~/rag-restart.log` |
| Docker-Container | `docker logs <containername> --tail 50 -f` |
| Backend (uvicorn) | Im Backend-Terminal-Tab live |

---

## Wann Hilfe holen?

Konkret bei diesen Symptomen Hilfe einholen oder Skript-Update prüfen:

- Skript bricht reproduzierbar an immer derselben Stelle ab
- Modelle laden, aber Antworten dauern >30 Sekunden
- Qdrant zeigt dauerhaft Status RED
- Speicher-Verbrauch steigt unkontrolliert (Memory Leak)

---

*Stand: 25.04.2026*
