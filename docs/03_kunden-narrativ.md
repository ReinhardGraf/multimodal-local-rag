# Multimodal Local RAG — Kunden-Narrativ

> Drei Sprach-Ebenen für Beratungsgespräche, je nach Gegenüber.

---

## Die Grundbotschaft in einem Satz

> *"Eine eigene, datenschutzkonforme Wissens-KI auf Ihrem eigenen Server — keine Cloud, keine Datenfreigabe an OpenAI oder Microsoft, volle Kontrolle."*

---

## Ebene 1 — Vorstand & Geschäftsführer

**Sprachstil:** Strategisch, business-orientiert, ohne Fachjargon.
**Länge:** 60-90 Sekunden.

> *"Stellen Sie sich vor, Ihr gesamtes Unternehmenswissen — Verträge, Protokolle, Handbücher — wird durchsuchbar wie Google, aber bleibt dabei zu hundert Prozent auf Ihrem eigenen Server. Keine Cloud, keine externe API, keine Datenfreigabe an OpenAI oder Microsoft. Mitarbeiter stellen Fragen in natürlicher Sprache und bekommen Antworten mit Quellenangabe — bezogen auf Ihre Dokumente. Das ist nicht ChatGPT. Das ist Ihre eigene Wissens-KI."*

**Schlüsselbegriffe:** Datenhoheit, Quellennachweis, eigener Server, DSGVO-konform.

---

## Ebene 2 — IT-Verantwortliche & Datenschutzbeauftragte

**Sprachstil:** Technisch präzise, mit Architektur-Bezug.
**Länge:** 2-3 Minuten.

> *"Das ist eine lokale Retrieval-Augmented-Generation-Architektur mit hybrider Suche. PDFs werden via Docling extrahiert, in semantische Chunks zerlegt, mit multilingualen Embeddings (E5-Large, 1024-dimensional) versehen und in Qdrant gespeichert. Die Suche kombiniert dichte Vektor-Suche mit BM25-Sparse-Vektoren — semantische Treffer plus exakte Stichwort-Matches. Reranking via CrossEncoder hebt die relevantesten Chunks nach oben. Die Antwort generiert ein lokales Qwen3-Modell auf Apple-Silicon mit Metal-GPU-Beschleunigung. Komplett ohne externe Calls. DSGVO Artikel 25 — Privacy by Design — nicht als Versprechen, sondern als Architektur."*

**Schlüsselbegriffe:** RAG, Hybrid-Search, Embeddings, Reranking, Privacy by Design.

---

## Ebene 3 — Praktische Anwender (Sachbearbeiter, Pfarrer, Verwalter)

**Sprachstil:** Konkret, alltagsnah, mit Beispielen.
**Länge:** 1-2 Minuten.

> *"Sie laden Ihre Dokumente einmal hoch. Das System merkt sich, welche Datei es schon gesehen hat — Sie können also auch beim hundertsten Mal die gleiche Datei reinwerfen, es passiert nichts Doppeltes. Dann fragen Sie über die Chat-Oberfläche, was Sie wissen wollen — etwa: 'Was sagt unser Mietvertrag zur Kündigungsfrist?' — und bekommen eine Antwort mit der Seitenzahl, auf der das steht. Auch Excel-Tabellen mit Zahlen können Sie befragen: 'Wie hoch waren die Kollekten-Erträge im letzten Quartal?'"*

**Schlüsselbegriffe:** einfach, mit Quellenangabe, kein doppelter Aufwand.

---

## Bilder & Metaphern

### Die Bibliothekarin

> *"Klassische Datenbanken sind wie Karteikarten: präzise, aber stur. Mein System ist wie eine Bibliothekarin, die nicht nur weiß, wo ein Buch steht, sondern auch sagt: 'Ah, dazu hat unser Pfarrer letztes Jahr eine Predigt geschrieben, und es gibt da einen Artikel im Sonntagsblatt, der das Thema von einer anderen Seite beleuchtet.' Sie versteht den Sinn, nicht nur den Wortlaut."*

### Die zwei Suchmodi

> *"Mein System sucht zweimal gleichzeitig: einmal mit Bedeutungs-Verständnis und einmal mit Wortgleichheit. So findet es 'Mietvertrag' auch dann, wenn Sie nach 'Pachtvereinbarung' fragen — aber es findet auch 'Müller GmbH', wenn Sie nach genau diesem Namen suchen. Das Beste aus zwei Welten."*

### Suchmaschine vs. Verstehmaschine

> *"Google ist eine Suchmaschine. Mein System ist eine Verstehmaschine."*

---

## Die fünf stärksten Verkaufsargumente

| # | Argument | Erklärung beim Kunden |
|---|---|---|
| **1** | **Echte Datenhoheit** | "Alles bleibt auf Ihrem Server. Punkt." |
| **2** | **Quellennachweis** | "Jede Antwort sagt Ihnen, woher sie kommt." |
| **3** | **Idempotenz** | "Sie können doppelt hochladen — es passiert nichts Doppeltes." |
| **4** | **Multimodal** | "Texte, Tabellen, Bilder — alles durchsuchbar." |
| **5** | **DSGVO-Architektur** | "Datenschutz ist nicht eingebaut, sondern eingebaut von Anfang an." |

---

## Demo-Choreographie

### Vor dem Termin

- [ ] `./rag-restart.sh` ausführen (~30 Sek)
- [ ] Browser-Tabs vorbereiten:
    - http://localhost:3000 (Open WebUI)
    - http://localhost:5008/docs (Backend Swagger)
    - http://localhost:6333/dashboard (Qdrant)
- [ ] Beispiel-Frage parat haben

### Während der Demo

**Schritt 1 — Open WebUI zeigen** (60 Sek)
> *"Das ist die Oberfläche, die Ihre Mitarbeiter sehen würden. Schaut aus wie ChatGPT — ist aber etwas grundlegend anderes."*

**Schritt 2 — Eine Frage stellen** (60 Sek)
> *"Ich frage jetzt: 'Was sagt Ditz über Verantwortung im KI-Einsatz?'"*

→ Antwort mit Quellenangabe abwarten

**Schritt 3 — Qdrant zeigen** (90 Sek)
> *"Hier sehen Sie die Datenbank. 611 Wissens-Punkte aus drei Büchern. Status grün — voll funktionsfähig. Hybrid-Search aktiv."*

**Schritt 4 — Backend Swagger** (60 Sek)
> *"Und hier sehen Sie die technische Schnittstelle. Zehn Endpoints, alle dokumentiert. Das ist Ihre API für eigene Integrationen."*

**Schritt 5 — Der Knaller** (30 Sek)
> *"Und jetzt das Wichtigste: Ich ziehe das Netzwerkkabel."*

→ WLAN ausschalten, gleiche Frage nochmal stellen
→ Antwort kommt trotzdem

> *"Genau. Komplett offline. Ihre Daten verlassen das Haus nicht."*

---

## Häufige Einwände & Antworten

### "Aber ChatGPT ist doch viel besser?"

> *"Ja — und ChatGPT ist auch viel teurer, viel weniger datenschutzkonform und kennt Ihre Dokumente nicht. Wir reden hier nicht über das beste Sprachmodell der Welt, sondern über das passende für Ihren Anwendungsfall: Ihre Dokumente, Ihre Daten, Ihre Kontrolle."*

### "Was passiert, wenn die KI sich irrt?"

> *"Genau deshalb sehen Sie immer die Quelle. Bei jeder Antwort steht: aus welchem Dokument, von welcher Seite. Sie überprüfen also nicht eine KI-Aussage, sondern Sie nutzen die KI als schnellen Wegweiser zur Originalquelle."*

### "Ist das nicht aufwändig zu betreiben?"

> *"Es läuft auf einem MacBook. Mit einem einzigen Befehl gestartet. Die Wartung übernimmt ein Update-Skript einmal im Quartal. Vergleichen Sie das mit der Komplexität einer Cloud-Lösung mit Vertragsmanagement, Auftragsverarbeitungsvertrag, Datenschutz-Folgenabschätzung..."*

### "Wie sicher ist das wirklich?"

> *"Das System hat keine Verbindung nach außen. Keine API-Keys, keine Cloud-Endpoints, keine Telemetrie. Sie können den Stecker ziehen und es läuft weiter. Das ist Sicherheit by Design."*

---

## Was NICHT versprochen wird

Wichtig für die eigene Glaubwürdigkeit — bewusste Beschränkungen offen kommunizieren:

- ❌ "Es wird so klug sein wie GPT-4." → *Nein. Es wird so klug sein wie Qwen3-4B, das ist eine andere Liga, aber für strukturierte Wissensabfragen ausreichend.*
- ❌ "Es funktioniert ohne Vorbereitung." → *Nein. Dokumente müssen einmal ingestiert werden, das dauert je nach Umfang Minuten bis Stunden.*
- ❌ "Es lernt selbständig dazu." → *Nein. Neue Erkenntnisse müssen aktiv eingespeist werden.*
- ❌ "Es ersetzt menschliches Urteilen." → *Nein, und das ist auch nicht das Ziel.*

---

## Anschlussfähigkeit zum 3-Regler-Modell

Das System bedient alle drei Regler:

| Regler | Beitrag der lokalen RAG |
|---|---|
| **Zeit** | Schnelle Antworten, kein Cloud-Latency |
| **Sozial** | Vertrauensbasis: Daten bleiben im Haus |
| **Sache** | Faktentreue durch Quellenangabe |

---

*Stand: 25.04.2026*
