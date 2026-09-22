# Installing by hand

The one-line installer in the [README](README.md) does all of this for you. Follow
these steps if you would rather see each one, or if the installer failed and you want
to know where.

## What you need

| | | |
|---|---|---|
| **Python 3.10+** | runs everything | python.org/downloads — tick *Add python.exe to PATH* |
| **A free API key** | writes the answers | [Groq](https://console.groq.com/keys) or [Gemini](https://aistudio.google.com/apikey) |
| **~700 MB of disk** | packages and two local models | downloaded once |
| **~400 MB of memory** | while running | measured at 307 MB with both models loaded |

Nothing else. No Ollama, no Docker, no database to install, no model to download by
hand.

## 1. Get the code

```powershell
git clone https://github.com/akashverma-lang/RAG.git LocalRAG
cd LocalRAG
```

Or download the ZIP from GitHub and unpack it.

## 2. Create the environment

A virtual environment keeps these packages away from everything else on the machine,
and deleting the folder removes them completely.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```bash
# macOS / Linux
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

This takes a few minutes. It pulls FastAPI, the ONNX runtime the embedder uses, and
the document readers.

## 3. Start it

```powershell
.venv\Scripts\python.exe launcher.py
```

Or double-click `run.bat` on Windows. Your browser opens by itself.

## 4. The setup screen

On the first run the app asks for four things and writes them for you:

- **Documents folder** — everything inside it, including subfolders.
- **Index folder** — where the search index lives. The default is fine.
- **Groq API key** and **Gemini API key** — either is enough; both let you switch.

They are saved to your own profile, not into the program folder:

| | |
|---|---|
| Windows | `%APPDATA%\LocalRAG\settings.env` |
| macOS | `~/Library/Application Support/LocalRAG/settings.env` |
| Linux | `~/.config/local-rag/settings.env` |

The app restarts itself once, and that is setup finished.

## 5. Index, then ask

**Manage → Index files**. The first run reads every document and downloads the
embedding model (~90 MB). Afterwards only changed files are re-read.

Then ask something.

## Checking an installation

```powershell
.venv\Scripts\python.exe rag.py doctor
```

It reports each package, the folders, the OCR engine, which providers answer, and how
much is indexed.

## When something is wrong

| What you see | Usually | Fix |
|---|---|---|
| `python` is not recognised | Python not on PATH | reinstall Python with *Add python.exe to PATH* ticked |
| Installation fails on `onnxruntime` | 32-bit Python | install 64-bit Python 3.10+ |
| `No matching distribution found for rapidocr-onnxruntime` | an old copy of this project, pinned above what your Python supports | re-run the installer; current versions drop OCR and carry on rather than failing |
| It installed but says *no engine available* for OCR | the OCR engine would not install on this Python | everything except reading text out of pictures still works; `pip install rapidocr-onnxruntime` to retry |
| Red dot, *no model configured* | no API key yet | add one on the setup screen (the gear icon) |
| Red dot, *rate limit* | free tier quota | wait a minute, or switch provider in the dropdown |
| *That documents folder does not exist* | typo, or a drive not mounted | browse to it rather than typing |
| Indexing finds 0 files | folder has no readable documents | check `INCLUDE_EXT` covers the types you have |
| Nothing found for an obvious question | not indexed yet | Manage → Index files, and check the file count |
| Port 8000 already in use | something else on it | the launcher moves to the next free port by itself |

## Running it offline

Every part of search is local, but the first run downloads packages and the embedding
model, and the provider that writes answers is reached over the network.

To work fully offline, point `OPENAI_BASE_URL` at a model served on the same machine —
LM Studio and similar expose the same protocol on `localhost`. Run the app once
connected first, so the embedding and reranking models are cached.

## Removing it

Delete the folder you installed into, and the settings folder listed above. Nothing
is written anywhere else and nothing is registered with the system.
