# RAG CHATBOT

RAG CHATBOT is a Streamlit assistant for asking questions about PDF documents. It builds a local FAISS index, retrieves the most relevant passages, and asks an OpenRouter-hosted model to answer with page-level citations.

## Features

- Multi-format document uploads (PDF, DOCX, TXT, CSV, XLSX, PPTX, MD, JSON) with a 50 MB limit
- Local Sentence Transformers embeddings and FAISS retrieval
- Persistent, content-addressed indexes for faster reuse
- Relevance filtering and page-level source excerpts
- Follow-up-question rewriting for conversational retrieval
- Streaming answers through OpenRouter
- Per-answer translation into 16 languages with instant live streaming
- Automatic read-aloud playback with instant language switching
- Real-time voice questions with volume normalization and automatic submission

## Setup

Use Python 3.10 on Windows PowerShell:

```powershell
cd "C:\Users\chinn\Downloads\RAG_CHATBOT\RAG_CHATBOT\llm-chatbot-rag-main"
py -3.10 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create `.env/.env` with an OpenRouter API key:

```dotenv
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct
```

Start the app:

```powershell
python -m streamlit run app.py
```

Then open <http://localhost:8501>.

## Voice notes

- The browser will ask for microphone permission the first time voice input is used.
- Microphone access works on localhost or a secure HTTPS deployment.
- Audio recording uses Streamlit's native chat microphone, and transcription uses the Google recognizer through `SpeechRecognition`.
- Read-aloud uses the browser's Web Speech API, so available voices depend on the operating system and browser.
- Retrieved document excerpts and translation requests are sent to the configured OpenRouter model; do not upload sensitive documents unless that usage is acceptable.

## Local data

Generated indexes are saved under `indexes/` and reused when the same PDFs and chunk settings are selected. Uploaded PDF files are processed in a temporary directory and removed immediately after indexing. Indexes, secrets, virtual environments, and generated caches are excluded by `.gitignore`.
