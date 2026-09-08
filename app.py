import audioop
import base64
import html
import io
import json
import logging
import wave
from datetime import datetime, timezone
from pathlib import Path

import speech_recognition as sr
import streamlit as st
import streamlit.components.v1 as components

from src.model import ChatModel
from src.rag_util import (
    Encoder,
    FaissDb,
    collection_fingerprint,
    index_path,
    load_and_split_documents,
    load_manifest,
    save_manifest,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "logo.png"


def get_logo_base64() -> str:
    if LOGO_PATH.exists():
        return base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return ""


LANGUAGES = {
    "English": "en-US",
    "Hindi": "hi-IN",
    "Telugu": "te-IN",
    "Tamil": "ta-IN",
    "Kannada": "kn-IN",
    "Malayalam": "ml-IN",
    "Marathi": "mr-IN",
    "Bengali": "bn-IN",
    "Gujarati": "gu-IN",
    "Urdu": "ur-PK",
    "Spanish": "es-ES",
    "French": "fr-FR",
    "German": "de-DE",
    "Arabic": "ar-SA",
    "Chinese (Simplified)": "zh-CN",
    "Japanese": "ja-JP",
}

VOICE_FALLBACKS = {
    "en-IN": ("en-IN", "en-US", "en-GB", "te-IN", "hi-IN"),
    "en-US": ("en-US", "en-IN", "en-GB", "te-IN", "hi-IN"),
    "en-GB": ("en-GB", "en-IN", "en-US", "te-IN", "hi-IN"),
    "hi-IN": ("hi-IN", "en-IN", "en-US"),
    "te-IN": ("te-IN", "en-IN", "en-US"),
}

st.set_page_config(
    page_title="RAG CHATBOT · Smart AI Assistant",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { max-width: 1080px; padding-top: 2.2rem; padding-bottom: 5rem; }
        [data-testid="stSidebar"] { border-right: 1px solid #dbe3f0; }
        [data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
        [data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            margin-bottom: 0.75rem;
            padding: 0.3rem 0.45rem;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.035);
        }
        .hero { padding: 0.3rem 0 1.4rem; }
        .eyebrow {
            color: #4f46e5;
            font-size: 0.75rem;
            font-weight: 750;
            letter-spacing: 0.13em;
            text-transform: uppercase;
        }
        .hero h1 {
            color: #0f172a;
            font-size: clamp(2rem, 4vw, 3.15rem);
            letter-spacing: -0.045em;
            line-height: 1.03;
            margin: 0.45rem 0 0.75rem;
        }
        .hero p { color: #64748b; font-size: 1.03rem; margin: 0; max-width: 680px; }
        .brand { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.25rem; }
        .brand-logo {
            border-radius: 8px;
            display: block;
            flex-shrink: 0;
            height: 40px;
            object-fit: contain;
            width: 40px;
        }
        .brand-mark {
            align-items: center;
            background: linear-gradient(135deg, #4f46e5, #7c3aed);
            border-radius: 11px;
            color: white;
            display: inline-flex;
            font-size: 1.25rem;
            height: 38px;
            justify-content: center;
            width: 38px;
        }
        .brand-copy { color: #0f172a; font-size: 1.05rem; font-weight: 750; }
        .brand-sub { color: #64748b; font-size: 0.73rem; font-weight: 500; }
        .feature-card {
            align-items: center;
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            display: flex;
            gap: 0.65rem;
            min-height: 52px;
            padding: 0.75rem 0.95rem;
            margin-bottom: 0.5rem;
        }
        .feature-card span {
            color: #334155;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            font-size: 1.35rem;
            line-height: 1;
        }
        .feature-card strong {
            color: #1e293b;
            font-size: 0.94rem;
            font-weight: 650;
            line-height: 1.3;
            margin: 0;
        }
        .feature-caption {
            color: #64748b;
            font-size: 0.88rem;
            line-height: 1.5;
            margin-top: 1.75rem !important;
            padding-top: 0.5rem;
            margin-bottom: 0.5rem;
            display: block;
        }
        .source-label { color: #475569; font-size: 0.84rem; }
        .stButton > button { border-radius: 10px; font-weight: 650; }
        [data-testid="stChatInput"] { border-radius: 14px; }
        [data-testid="stChatInputMicButton"] {
            border-radius: 10px !important;
            color: #4f46e5 !important;
            background: #eef2ff !important;
            border: 1px solid #c7d2fe !important;
            height: 34px !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 4px !important;
            transition: all 0.2s ease !important;
        }
        [data-testid="stChatInputMicButton"]:hover {
            background: #e0e7ff !important;
            color: #3730a3 !important;
            border-color: #a5b4fc !important;
        }
        @media (min-width: 640px) {
            [data-testid="stChatInputMicButton"] {
                width: auto !important;
                padding: 0 10px !important;
            }
            [data-testid="stChatInputMicButton"]::after {
                content: "Speak";
                font-size: 0.82rem;
                font-weight: 650;
                margin-left: 3px;
            }
        }
        footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_state() -> None:
    defaults = {
        "db": None,
        "messages": [],
        "manifest": {},
        "collection_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_llm() -> ChatModel:
    return ChatModel()


@st.cache_resource(show_spinner=False)
def load_encoder():
    return Encoder().embedding_function


def voice_language_candidates(language_code: str) -> list[str]:
    candidates = [language_code, *VOICE_FALLBACKS.get(language_code, ())]
    deduped = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped or ["en-IN"]


def extract_transcript(recognition_response) -> tuple[str, list[str]]:
    if isinstance(recognition_response, str):
        transcript = recognition_response.strip()
        return transcript, [transcript] if transcript else []

    if not isinstance(recognition_response, dict):
        return "", []

    alternatives = recognition_response.get("alternative") or []
    transcripts = []
    for alternative in alternatives:
        transcript = str(alternative.get("transcript", "")).strip()
        if transcript and transcript not in transcripts:
            transcripts.append(transcript)

    return (transcripts[0] if transcripts else ""), transcripts[:3]


def preprocess_audio(audio_bytes: bytes) -> bytes:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            frames = wf.readframes(nframes)

        if not frames or sampwidth not in (1, 2, 4):
            return audio_bytes

        # If stereo, convert to mono
        if nchannels == 2:
            frames = audioop.tomono(frames, sampwidth, 0.5, 0.5)
            nchannels = 1

        # Check peak amplitude and amplify if too quiet
        peak = audioop.max(frames, sampwidth)
        max_possible = 2 ** (8 * sampwidth - 1) - 1
        target_peak = int(max_possible * 0.75)
        if 0 < peak < (target_peak / 2):
            factor = min(float(target_peak) / float(peak), 8.0)
            frames = audioop.mul(frames, sampwidth, factor)

        out_buf = io.BytesIO()
        with wave.open(out_buf, "wb") as out_wf:
            out_wf.setnchannels(nchannels)
            out_wf.setsampwidth(sampwidth)
            out_wf.setframerate(framerate)
            out_wf.writeframes(frames)
        return out_buf.getvalue()
    except Exception as e:
        logger.warning(f"Audio preprocessing warning: {e}")
        return audio_bytes


def transcribe_voice(audio_file, language_code: str) -> tuple[str, str, list[str]]:
    audio_file.seek(0)
    raw_bytes = audio_file.getvalue()
    if len(raw_bytes) < 1024:
        raise sr.UnknownValueError()

    audio_bytes = preprocess_audio(raw_bytes)

    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = False
    recognizer.energy_threshold = 150

    with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
        audio_data = recognizer.record(source)

    last_request_error = None
    for candidate_language in voice_language_candidates(language_code):
        try:
            response = recognizer.recognize_google(
                audio_data,
                language=candidate_language,
                show_all=True,
            )
            transcript, alternatives = extract_transcript(response)
            if transcript:
                return transcript, candidate_language, alternatives
        except sr.UnknownValueError:
            continue
        except sr.RequestError as error:
            last_request_error = error

    if last_request_error:
        raise last_request_error
    raise sr.UnknownValueError()


def format_context(results) -> str:
    blocks = []
    for position, result in enumerate(results, start=1):
        blocks.append(
            f"[{position}] File: {result.source} | Page: {result.page}\n"
            f"{result.text}"
        )
    return "\n\n".join(blocks)


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"View {len(sources)} supporting source(s)"):
        for position, source in enumerate(sources, start=1):
            source_name = html.escape(str(source["source"]))
            st.markdown(
                f"**[{position}] {source_name} · page {source['page']}**  "
                f"\n<span class='source-label'>Relevance {source['score']:.0%}</span>",
                unsafe_allow_html=True,
            )
            excerpt = " ".join(source["text"].split())
            st.caption(excerpt[:500] + ("…" if len(excerpt) > 500 else ""))
            if position < len(sources):
                st.divider()


@st.cache_data(show_spinner=False)
def generate_speech_audio_b64(text: str, language_name: str) -> str:
    try:
        import base64
        import io
        import re
        from gtts import gTTS

        clean = re.sub(r"\[\d+\]", "", text)
        clean = re.sub(r"[*#_`~>]", "", clean)
        clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        if not clean:
            return ""

        gtts_codes = {
            "English": "en",
            "Hindi": "hi",
            "Telugu": "te",
            "Tamil": "ta",
            "Kannada": "kn",
            "Malayalam": "ml",
            "Marathi": "mr",
            "Bengali": "bn",
            "Gujarati": "gu",
            "Urdu": "ur",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Arabic": "ar",
            "Chinese (Simplified)": "zh-CN",
            "Japanese": "ja",
        }
        lang_code = gtts_codes.get(language_name, "en")
        tts = gTTS(text=clean, lang=lang_code, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        return base64.b64encode(fp.getvalue()).decode("ascii")
    except Exception as error:
        logger.warning("gTTS speech generation failed for %s: %s", language_name, error)
        return ""


def render_speech_controls(
    text: str,
    language_code: str,
    language_name: str,
    control_id: str,
    audio_b64: str = "",
    autoplay: bool = False,
) -> None:
    safe_text = json.dumps(text, ensure_ascii=False).replace("</", "<\\/")
    safe_language = json.dumps(language_code)
    safe_lang_name = json.dumps(language_name)
    safe_control_id = json.dumps(control_id)
    safe_audio_b64 = json.dumps(audio_b64 or "")
    safe_autoplay = json.dumps(autoplay)
    speech_markup = f"""
        <!DOCTYPE html>
        <html>
        <head><meta charset="utf-8"></head>
        <body>
        <div class="speech-controls">
            <button id="play-{control_id}" onclick="speakText()">Read</button>
            <button class="stop" onclick="stopSpeaking()">Stop</button>
        </div>
        <script>
            const speechText = {safe_text};
            const speechLanguage = {safe_language};
            const speechLangName = {safe_lang_name};
            const controlId = {safe_control_id};
            const audioB64 = {safe_audio_b64};
            const shouldAutoplay = {safe_autoplay};
            const playBtn = document.getElementById("play-" + controlId);
            let currentAudio = null;

            function stopSpeaking() {{
                if (currentAudio) {{
                    try {{
                        currentAudio.pause();
                        currentAudio.currentTime = 0;
                        currentAudio.src = "";
                    }} catch (e) {{}}
                    currentAudio = null;
                }}
                try {{
                    window.speechSynthesis.cancel();
                }} catch (e) {{}}
                try {{
                    if (window.top && window.top.speechSynthesis) {{
                        window.top.speechSynthesis.cancel();
                    }}
                }} catch (e) {{}}
                if (playBtn) playBtn.innerHTML = "Read";
            }}

            function stopAllGlobalAudio() {{
                try {{
                    if (window.top && typeof window.top.__rag_stop_speech === "function") {{
                        window.top.__rag_stop_speech();
                    }}
                }} catch (e) {{}}
                try {{
                    window.speechSynthesis.cancel();
                }} catch (e) {{}}
                try {{
                    if (window.top && window.top.speechSynthesis) {{
                        window.top.speechSynthesis.cancel();
                    }}
                }} catch (e) {{}}
                try {{
                    if (window.parent && window.parent.postMessage) {{
                        window.parent.postMessage({{ type: "RAG_STOP_SPEECH", id: controlId }}, "*");
                    }}
                }} catch (e) {{}}
            }}

            try {{
                if (window.top) {{
                    window.top.__rag_stop_speech = function() {{
                        stopSpeaking();
                    }};
                }}
            }} catch (e) {{}}

            window.addEventListener("message", function(ev) {{
                if (ev && ev.data && ev.data.type === "RAG_STOP_SPEECH" && ev.data.id !== controlId) {{
                    stopSpeaking();
                }}
            }});

            window.addEventListener("beforeunload", stopSpeaking);
            window.addEventListener("unload", stopSpeaking);
            window.addEventListener("pagehide", stopSpeaking);

            function speakWithBrowser() {{
                try {{
                    let synth = (window.top && window.top.speechSynthesis) || window.speechSynthesis;
                    if (!synth) {{
                        if (playBtn) playBtn.innerHTML = "Read";
                        return;
                    }}
                    let clean = speechText
                        .replace(/\\[\\d+\\]/g, '')
                        .replace(/[*#_`~>]/g, '')
                        .replace(/\\[([^\\]]+)\\]\\([^)]+\\)/g, '$1')
                        .replace(/\\s+/g, ' ')
                        .trim();
                    if (!clean) {{
                        if (playBtn) playBtn.innerHTML = "Read";
                        return;
                    }}
                    const utterance = new SpeechSynthesisUtterance(clean);
                    utterance.lang = speechLanguage || "en-US";
                    utterance.rate = 0.95;
                    utterance.onend = function() {{
                        if (playBtn) playBtn.innerHTML = "Read";
                    }};
                    utterance.onerror = function() {{
                        if (playBtn) playBtn.innerHTML = "Read";
                    }};
                    synth.speak(utterance);
                }} catch (e) {{
                    if (playBtn) playBtn.innerHTML = "Read";
                }}
            }}

            function speakText() {{
                stopAllGlobalAudio();
                stopSpeaking();

                if (playBtn) playBtn.innerHTML = "Reading...";

                if (audioB64 && audioB64.length > 50) {{
                    try {{
                        let AudioConstructor = Audio;
                        try {{
                            if (window.top && window.top.Audio) {{
                                AudioConstructor = window.top.Audio;
                            }}
                        }} catch (e) {{}}
                        currentAudio = new AudioConstructor("data:audio/mp3;base64," + audioB64);
                        currentAudio.onended = function() {{
                            if (playBtn) playBtn.innerHTML = "Read";
                        }};
                        currentAudio.onerror = function() {{
                            speakWithBrowser();
                        }};
                        currentAudio.play().catch(function() {{
                            speakWithBrowser();
                        }});
                        return;
                    }} catch (err) {{
                        speakWithBrowser();
                        return;
                    }}
                }}
                speakWithBrowser();
            }}

            if (shouldAutoplay) {{
                stopAllGlobalAudio();
                setTimeout(function() {{
                    speakText();
                }}, 200);
            }}
        </script>
        <style>
            body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
            .speech-controls {{ display: flex; align-items: center; gap: 8px; padding-top: 3px; }}
            button {{
                background: #eef2ff; border: 1px solid #c7d2fe; border-radius: 9px;
                color: #4338ca; cursor: pointer; font-size: 13px; font-weight: 650;
                padding: 7px 12px; transition: all 0.15s ease;
            }}
            button:hover {{ background: #e0e7ff; }}
            button.stop {{ background: white; border-color: #e2e8f0; color: #64748b; }}
            button.stop:hover {{ background: #f1f5f9; color: #334155; }}
        </style>
        </body>
        </html>
        """
    components.html(speech_markup, height=45)


def render_answer_tools(message: dict, message_index: int, model: ChatModel) -> None:
    st.caption("Translate or listen")
    current_active = message.get("active_translation")
    lang_keys = list(LANGUAGES)
    default_idx = lang_keys.index(current_active) if current_active in lang_keys else 0

    language_column, button_column = st.columns([3, 1])
    target_language = language_column.selectbox(
        "Translation language",
        lang_keys,
        index=default_idx,
        key=f"translation_language_{message_index}",
        label_visibility="collapsed",
    )

    translations = message.setdefault("translations", {})
    translate_clicked = button_column.button(
        "Translate",
        key=f"translate_{message_index}",
        use_container_width=True,
    )

    if translate_clicked:
        if target_language == "English":
            message["active_translation"] = "English"
            translations["English"] = message["content"]
        else:
            if target_language not in translations:
                placeholder = st.empty()
                streamed_chunks = []
                with st.spinner(f"Translating to {target_language}…"):
                    try:
                        if hasattr(model, "stream_translate"):
                            for chunk in model.stream_translate(
                                message["content"],
                                target_language,
                            ):
                                streamed_chunks.append(chunk)
                                placeholder.markdown(
                                    f"**🌐 {target_language} translation:**\n\n"
                                    + "".join(streamed_chunks)
                                    + " ▌"
                                )
                            placeholder.empty()
                            translated = "".join(streamed_chunks).strip()
                        else:
                            translated = model.translate(
                                message["content"],
                                target_language,
                            ).strip()
                        if translated:
                            translations[target_language] = translated
                            message["active_translation"] = target_language
                    except Exception as err:
                        logger.exception(f"Failed to translate to {target_language}")
                        st.error(f"Translation failed: {err}")
                    finally:
                        placeholder.empty()
            else:
                message["active_translation"] = target_language
    elif target_language in translations and message.get("active_translation") is not None:
        message["active_translation"] = target_language

    active_language = message.get("active_translation")
    if active_language and active_language == target_language and active_language in translations:
        active_text = translations[active_language]
        if active_language != "English":
            st.markdown(f"**🌐 {active_language} translation:**")
            st.markdown(active_text)

        clean_lang_id = active_language.lower().replace(" ", "_")
        audio_b64 = generate_speech_audio_b64(active_text, active_language)
        render_speech_controls(
            active_text,
            LANGUAGES.get(active_language, "en-US"),
            active_language,
            f"answer-{message_index}-{clean_lang_id}",
            audio_b64=audio_b64,
            autoplay=False,
        )


def build_or_load_collection(
    uploaded_files,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[FaissDb, dict, bool]:
    payloads = []
    for uploaded_file in uploaded_files:
        safe_name = Path(uploaded_file.name).name or "document.txt"
        payloads.append((safe_name, uploaded_file.getvalue()))

    collection_id = collection_fingerprint(payloads, chunk_size, chunk_overlap)
    directory = index_path(collection_id)
    encoder = load_encoder()

    if (directory / "index.faiss").exists() and (directory / "index.pkl").exists():
        db = FaissDb.load(directory, encoder)
        manifest = load_manifest(directory)
        return db, manifest, True

    with tempfile.TemporaryDirectory() as temp_dir:
        paths = []
        original_names = {}
        for position, (name, content) in enumerate(payloads):
            temporary_name = f"{position:03d}_{name}"
            path = Path(temp_dir) / temporary_name
            path.write_bytes(content)
            paths.append(str(path))
            original_names[temporary_name] = name

        docs = load_and_split_documents(
            paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for doc in docs:
            temporary_name = Path(doc.metadata.get("source", "")).name
            doc.metadata["source"] = original_names.get(temporary_name, temporary_name)

        if not docs:
            raise ValueError("No readable text was found in the uploaded documents.")
        db = FaissDb.from_documents(docs, encoder)

    manifest = {
        "collection_id": collection_id,
        "documents": [name for name, _ in payloads],
        "document_count": len(payloads),
        "chunk_count": db.chunk_count,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    db.save(directory)
    save_manifest(directory, manifest)
    return db, manifest, False


initialize_state()
llm = load_llm()

with st.sidebar:
    logo_b64 = get_logo_base64()
    logo_html = (
        f'<img class="brand-logo" src="data:image/png;base64,{logo_b64}" alt="Logo" />'
        if logo_b64
        else '<div class="brand-mark">◈</div>'
    )
    st.markdown(
        f"""
        <div class="brand">
            {logo_html}
            <div><div class="brand-copy">RAG CHATBOT</div><div class="brand-sub">SMART AI ASSISTANT</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not llm.is_configured:
        st.error("Add `OPENROUTER_API_KEY` to `.env/.env`, then restart the app.")

    st.subheader("Documents")
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=None,
        accept_multiple_files=True,
        help="Upload files in any format: PDF, DOCX, TXT, CSV, XLSX, PPTX, MD, JSON, etc. Max 50 MB total.",
    )

    with st.expander("Index settings"):
        chunk_size = st.slider("Chunk size", 300, 1000, 600, 50)
        chunk_overlap = st.slider("Chunk overlap", 50, 200, 100, 10)

    build_clicked = st.button(
        "Build knowledge base",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    )

    if build_clicked:
        total_bytes = sum(len(file.getvalue()) for file in uploaded_files)
        if total_bytes > 50 * 1024 * 1024:
            st.error("This collection is larger than 50 MB. Remove a file and try again.")
        else:
            try:
                with st.spinner("Indexing documents…"):
                    db, manifest, loaded_from_cache = build_or_load_collection(
                        uploaded_files,
                        chunk_size,
                        chunk_overlap,
                    )
                st.session_state.db = db
                st.session_state.manifest = manifest
                st.session_state.collection_id = manifest.get("collection_id")
                status = "Loaded saved index" if loaded_from_cache else "Knowledge base ready"
                st.success(status)
            except Exception as error:
                logger.exception("Failed to build the document collection")
                st.error(f"Could not process these documents: {error}")

    if st.session_state.db is not None:
        manifest = st.session_state.manifest
        st.success(
            f"Active · {manifest.get('document_count', 0)} document(s) · "
            f"{manifest.get('chunk_count', st.session_state.db.chunk_count)} chunks"
        )
        document_names = manifest.get("documents", [])
        if document_names:
            st.caption("\n".join(f"• {name}" for name in document_names))

    st.divider()
    st.subheader("Answer settings")
    rag_controls_disabled = st.session_state.db is None
    top_k = st.slider(
        "Sources per answer",
        2,
        8,
        4,
        disabled=rag_controls_disabled,
        help="Available after a PDF knowledge base is active.",
    )
    score_threshold = st.slider(
        "Minimum relevance",
        0.0,
        0.8,
        0.0,
        0.05,
        disabled=rag_controls_disabled,
        help="Raise this to exclude loosely related passages.",
    )
    max_tokens = st.slider("Maximum response tokens", 200, 2000, 700, 100)
    temperature = st.slider("Creativity", 0.0, 1.0, 0.2, 0.1)

    st.divider()
    clear_column, unload_column = st.columns(2)
    if clear_column.button(
        "Clear chat",
        use_container_width=True,
        disabled=not st.session_state.messages,
    ):
        st.session_state.messages = []
        st.rerun()
    if unload_column.button(
        "Unload documents",
        use_container_width=True,
        disabled=st.session_state.db is None,
    ):
        st.session_state.db = None
        st.session_state.manifest = {}
        st.session_state.collection_id = None
        st.rerun()

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Retrieval-augmented assistant</div>
        <h1>Answers grounded in your documents.</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    if st.session_state.db is None:
        st.info("Start by uploading documents in the sidebar and building a knowledge base.")
    else:
        st.success("Your documents are ready. Ask a question below.")

    feature_columns = st.columns(3)
    features = [
        ("⌕", "Focused retrieval"),
        ("⌁", "Page citations"),
        ("↻", "Reusable indexes"),
    ]
    for column, (icon, title) in zip(feature_columns, features):
        with column:
            st.markdown(
                f"<div class='feature-card'><span>{icon}</span><strong>{title}</strong></div>",
                unsafe_allow_html=True,
            )
    st.markdown(
        "<div class='feature-caption' style='margin-top: 1.75rem; padding-top: 0.5rem; color: #64748b; font-size: 0.88rem; line-height: 1.5; display: block;'>"
        "Use the Speak button or type in the bar below. Your spoken question follows the same document retrieval flow."
        "</div>",
        unsafe_allow_html=True,
    )

for message_index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))
            render_answer_tools(message, message_index, llm)

prompt = None

chat_prompt = st.chat_input(
    "Ask about your documents…" if st.session_state.db else "Ask a question…",
    disabled=not llm.is_configured,
    accept_audio=True,
)

if chat_prompt:
    prompt_text = ""
    if hasattr(chat_prompt, "text") and chat_prompt.text:
        prompt_text = chat_prompt.text.strip()
    elif isinstance(chat_prompt, str):
        prompt_text = chat_prompt.strip()

    if hasattr(chat_prompt, "audio") and chat_prompt.audio:
        with st.spinner("Transcribing your question…"):
            try:
                selected_lang = "en-IN"
                recognized, recognized_lang, alternatives = transcribe_voice(
                    chat_prompt.audio,
                    selected_lang,
                )
                if recognized:
                    prompt = (
                        f"{prompt_text} {recognized}".strip()
                        if prompt_text
                        else recognized.strip()
                    )
            except sr.UnknownValueError:
                st.warning("I could not hear a clear sentence. Please speak clearly into the microphone and try again.")
            except sr.RequestError as e:
                st.error(f"Speech recognition service unavailable: {e}")
            except Exception as e:
                logger.exception("Voice transcription failed")
                st.error(f"Voice transcription failed: {e}")

    if prompt is None and prompt_text:
        prompt = prompt_text

if prompt:
    prior_history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        sources = []
        context = None

        if st.session_state.db is not None:
            with st.spinner("Searching your documents…"):
                retrieval_query = llm.rewrite_query(prompt, prior_history)
                results = st.session_state.db.search(
                    retrieval_query,
                    k=top_k,
                    score_threshold=score_threshold,
                )
            sources = [result.to_dict() for result in results]
            context = format_context(results) if results else None

        if st.session_state.db is not None and not context:
            answer = "I couldn't find enough relevant information in the uploaded documents to answer that."
            st.markdown(answer)
        else:
            response_placeholder = st.empty()
            response_parts = []
            for token in llm.stream_answer(
                prompt,
                context=context,
                history=prior_history,
                max_tokens=max_tokens,
                temperature=temperature,
            ):
                response_parts.append(token)
                response_placeholder.markdown("".join(response_parts) + " ▌")
            answer = "".join(response_parts).strip()
            response_placeholder.markdown(answer)

        assistant_message = {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "translations": {},
            "active_translation": None,
        }
        st.session_state.messages.append(assistant_message)
        render_sources(sources)
        render_answer_tools(
            assistant_message,
            len(st.session_state.messages) - 1,
            llm,
        )
