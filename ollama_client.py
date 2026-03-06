import requests
import sys

def ask_ollama(prompt, model="qwen3:4b-instruct-2507-q4_K_M"):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_ctx": 8192,
        },
        "keep_alive": "10m",
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response.json()['response']
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        # user_input = input("Enter your prompt: ")
        user_input = """Du bist sind ein hilfreicher Assistent, der Fragen AUSSCHLIESSLICH anhand des bereitgestellten Kontexts beantwortet. Du versuchst dem User so gut wie möglich mit den bereitgestelltem Kontext zu helfen und auf sein Problem einzugehen. Du musst Deine Quellen genau angeben.

WICHTIGE REGELN FÜR DIE QUELLENANGABE:
1. Du musst für JEDE von Dir genannte Tatsache die Quelle und die Seite/Zeile angeben.
2. Formatiere Quellenangaben wie folgt: [Quelle: ganzer/dokumenten/pfad/doc.txt, Seite: X] oder [Quelle: ganzer/dokumenten/pfad/doc.txt, Zeile: X] am ENDE jedes Satzes.
3. Verweise NICHT auf „ID: 1” oder Ähnliches – verwende die für Menschen lesbaren Quellenangaben.
4. Wenn der Kontext keine relevanten Informationen enthält, sag: „Ich habe dazu keine Informationen in den verfügbaren Dokumenten.”
5. Erfinde oder folgere niemals Informationen, die nicht im Kontext vorhanden sind.
6. Antworte mit weniger als 200 Wörtern und fasse gegebenenfalls zusammen.


CONTEXT: [ID: 0] [Quelle: /data/documents/BDA-MOMENTO100.pdf, Seite: 33]
Content: Die angezeigte Sprache ist nicht korrekt., Mögliche Lösung = Passen Sie die Spracheinstellungen im Setup-Menü Ihrer Maschine an. Falls das Problem weiter besteht, kontaktieren Sie den Betreiber der Maschine.. Die Maschine schaltet zu schnell in den Ruhe-/ Standby-Modus., Mögliche Lösung = Passen Sie die Einstellungen zum Energiesparen im Setup-Menü Ihrer Maschine an. Falls das Problem weiter besteht, kontaktieren Sie den Betreiber der Maschine.. DieTemperatur von Momento 100 Getränken ist zu heiß/zu kalt., Mögliche Lösung = Passen Sie dieTemperatureinstellungen im Setup-Menü Ihrer Maschine an. Falls das Problem weiter besteht, kontaktieren Sie den Betreiber der Maschine.. Die Rezeptmenge von Momento 100 Getränken ist zu groß/zu klein., Mögliche Lösung = Passen Sie die Einstellungen zurTassengröße/ Rezeptmenge im Setup-Menü Ihrer Maschine an. Falls das Problem weiter besteht, kontaktieren Sie den Betreiber der Maschine.. Momento 100 scheint zu funktionieren, jedoch kommt kein Wasser/Kaff.ligaee aus dem Kaff.ligaeeauslauf., Mögliche Lösung = Stellen Sie sicher, dass derWasserbehälter gefüllt ist. Stellen Sie sicher, dass der Wasserfilter keine Lu/f_t.ligablasen enthält. Füllen Sie zu diesem Zweck den Wasserbehälter mit frischemTrinkwasser, halten Sie den Filter umgedreht insWasser und stellen Sie sicher, dass alle Lu/f_t.liga daraus entweicht. Setzen Sie den Filter ein und testen Sie die Maschine erneut.. DerTassenwärmer funktioniert nicht., Mögliche Lösung = Schalten Sie denTassenwärmer ein. Lesen Sie zum Ein- und Ausschalten desTassenwärmers den Abschni/t_t.liga 'Energiesparmodi' auf Seite 21.

---

[ID: 1] [Quelle: /data/documents/BDA-MOMENTO100.pdf, Seite: 32]
Content: Warnung, Problem = Warnung: Fehlercode #, bi/t_t.ligae kontaktieren Sie den Betreiber der Maschine.. Warnung, Mögliche Lösung = Schalten Sie die Maschine aus/ein. Falls der Warnhinweis nicht erlischt, kontaktieren Sie den Betreiber der Maschine.. Warnung, Problem = Warnung: Momento 100 befindet sich im Abkühlvorgang, bi/t_t.ligae warten.. Warnung, Mögliche Lösung = Warten Sie, bis die Maschine abgekühlt ist. Falls derWarnhinweis nicht erlischt, kontaktieren Sie den Betreiber der Maschine.. Warnung, Problem = Warnung: Brüheinheit ist blockiert: bi/t_t.ligae (X) drücken. Falls das Problem weiter besteht, kontaktieren Sie den Betreiber der Maschine.. Warnung, Mögliche Lösung = Drücken Sie (X) amunteren Displayrand. Falls das Problem weiter besteht, überprüfen Sie, ob etwas in der Brüheinheit feststeckt. Ziehen Sie in diesem Fall den Netzstecker und versuchen Sie, das Objekt mit einem Werkzeug nach unten zu drücken (z. B. Löff.ligael, Rührstäbchen). Falls das Problem weiter besteht, kontaktieren Sie den Betreiber der Maschine.. Warnung, Problem = Warnung: Problem mit dem direkten Wasseranschluss: Bi/t_t.ligae kontaktieren Sie den Betreiber der Maschine.. Warnung, Mögliche Lösung = Schalten Sie die Maschine aus/ein. Falls der Warnhinweis nicht erlischt, kontaktieren Sie den Betreiber der Maschine.. Wartungs- hinweis, Problem = Warnung: Maschinenwartung erforderlich. Bi/t_t.ligae kontaktieren Sie den Betreiber der Maschine.. Wartungs- hinweis, Mögliche Lösung = Bi/t_t.ligae kontaktieren Sie den Betreiber der Maschine, umdieWartung zu veranlassen.. Wartungs- hinweis, Problem = Warnung: Bi/t_t.ligae reinigen Sie den Wasserbehälter.. Wartungs- hinweis, Mögliche Lösung = Bi/t_t.ligae reinigen Sie denWasserbehälter gemäß der Bedienungsanleitung.

---

[ID: 2] [Quelle: /data/documents/BDA-MOMENTO100.pdf, Seite: 34]
Content: Momento 100

---

[ID: 3] [Quelle: /data/documents/BDA-MOMENTO100.pdf, Seite: 20]
Content: [Image] Deckel des Wasserbehälters abnehmen. Wasserbehälter ausleeren und mit Trinkwasser ausspülen. Mit Einwegtüchern oder Papierhandtüchern trocknen. Deckel wieder aufsetzen. Wasserbehälter wieder in die Maschine einsetzen.

1
⚠️ THIS CHUNK HAS AN ASSOCIATED IMAGE. Image marker: [IMAGE:6b1cf85e-8c25-49ec-b07f-5aa838ea2284]. Add that to the response

---

[ID: 4] [Quelle: /data/documents/BDA-MOMENTO100.pdf, Seite: 34]
Content: Version: Bedienungsanleitung Momento 100 Originalbedienungsanleitung


---------------------

Handhabung von Bildern: 

IMAGE PLACEMENT RULES:
- Some context chunks have associated images marked with [IMAGE:uuid].
- When you use information from a chunk that has an image, you MUST place the exact image marker tag on its OWN line at the most contextually relevant position in your response.
- Place the image marker IMMEDIATELY AFTER the sentence or paragraph that describes or references what the image shows.
- Do NOT place all images at the end. Each image marker must appear inline, close to the related text.
- Do NOT modify the image marker format. Use it exactly as provided: [IMAGE:uuid]
- If a chunk has an image but you don't use that chunk's content, do NOT include its image marker.

Available image markers:
- [IMAGE:6b1cf85e-8c25-49ec-b07f-5aa838ea2284] (from [Quelle: /data/documents/BDA-MOMENTO100.pdf, Seite: 20])

---------------------

Benutzer-Query: Wie wechsel ich das Wasser im Momento 100?

Chatverlauf: []"""
    
    print(ask_ollama(user_input))
