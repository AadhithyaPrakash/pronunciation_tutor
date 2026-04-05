"""
Pronunciation Checker – Main Page
-----------------------------------
Run with:
    streamlit run app/streamlit_app.py

Flow:
    0. Login / Register (pages/0_Login.py)
    1. Record audio
    2. Confirm / edit transcript
    3. Word-by-word phoneme analysis (this page)
    4. "View Full Report" → pages/2_Overall_Report.py
    5. "Profile" → pages/3_Profile.py
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import streamlit as st

try:
    from infrastructure.logging_config import configure_logging
    configure_logging()
except Exception:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

from app.analyzer import PronunciationAnalyzer, PronunciationReport, WordReport
from app.ui import configure_page
from services import asr_service, phoneme_recognition_service, tts_audio_service
from infrastructure import database

configure_page(
    title="Pronunciation Checker",
    icon="🗣️",
    layout="wide",
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@st.cache_resource(show_spinner=False)
def _initialize_runtime_dependencies() -> dict:
    """Initialize DB and speech models once per Streamlit server process."""
    database.init_db()
    asr_backend = asr_service.warmup_model()
    phoneme_model = phoneme_recognition_service.warmup_model()
    return {
        "asr_backend": asr_backend,
        "phoneme_model": phoneme_model,
    }


if _env_flag("PRELOAD_MODELS_ON_STARTUP", True):
    with st.spinner("Initializing database and speech models..."):
        try:
            _initialize_runtime_dependencies()
        except Exception as exc:
            logger.exception("Startup initialization failed")
            st.error(
                "Initialization failed before recorder startup. "
                "Please check model/network setup and restart.\n\n"
                f"Details: {exc}"
            )
            st.stop()
else:
    database.init_db()

# ── Auth guard ────────────────────────────────────────────────────────────────
if not st.session_state.get("user_id"):
    st.switch_page("pages/0_Login.py")

st.markdown("""
<style>
.phoneme-tag {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 5px;
    margin: 2px;
    font-family: monospace;
    font-size: 0.85rem;
    font-weight: 600;
}
.ph-correct  { background: #d5f5e3; color: #1a7a45; }
.ph-error    { background: #fde8e8; color: #c0392b; }
.score-pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: 800;
    font-size: 1.1rem;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────

def _reset():
    st.session_state.analyzer        = PronunciationAnalyzer()
    st.session_state.raw_transcript   = ""
    st.session_state.corrected        = ""
    st.session_state.report           = None
    st.session_state.audio_path       = None
    st.session_state.stage            = "record"


if "stage" not in st.session_state:
    _reset()


def _save_audio(uploaded) -> Path:
    data = uploaded.read()
    tmp  = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(data)
    tmp.close()
    return Path(tmp.name)


def _read_audio_bytes(audio_path: str | Path) -> bytes:
    return Path(audio_path).read_bytes()


analyzer: PronunciationAnalyzer = st.session_state.analyzer

# ── Header with user info ──────────────────────────────────────────────────────
header_col, nav_col = st.columns([4, 1])
with header_col:
    st.title("🗣️ Pronunciation Checker")
    st.caption("Record → get instant phoneme-level feedback on every word.")
with nav_col:
    user_name = st.session_state.get("user_name", "User")
    st.markdown(f"👤 **{user_name}**")
    if st.button("📊 My Profile", width="stretch"):
        st.switch_page("pages/3_Profile.py")
    if st.button("🚪 Logout", width="stretch"):
        for k in ["user_id","user_name","username","user_email","stage","report","analyzer"]:
            st.session_state.pop(k, None)
        st.switch_page("pages/0_Login.py")

st.divider()

# ==========================================================================
# STAGE 1 – RECORD
# ==========================================================================
if st.session_state.stage == "record":
    st.subheader("Step 1 — Record your sentence")
    st.info("Prepare your microphone and keep background noise low.")

    audio_input = st.audio_input("🎙️ Click to record", key="audio_recorder")

    st.markdown("**Or upload an existing recording (WAV/MP3)**")
    uploaded_file = st.file_uploader("Upload audio", type=["wav", "mp3"], label_visibility="collapsed")

    audio_to_process = audio_input or uploaded_file

    if audio_to_process is not None:
        with st.spinner("Transcribing…"):
            try:
                audio_path = _save_audio(audio_to_process)
                st.session_state.audio_path    = audio_path
                raw = analyzer.transcribe(audio_path)
                st.session_state.raw_transcript = raw
                st.session_state.corrected      = analyzer.correct_transcript(raw)
                st.session_state.stage          = "confirm"
                st.rerun()
            except Exception as exc:
                logger.exception("Transcription failed")
                st.error(f"Transcription failed: {exc}")

# ==========================================================================
# STAGE 2 – CONFIRM
# ==========================================================================
elif st.session_state.stage == "confirm":
    st.subheader("Step 2 — Confirm what you said")

    st.audio(_read_audio_bytes(st.session_state.audio_path), format="audio/wav")

    col_raw, col_edit = st.columns(2)
    with col_raw:
        st.markdown("**🎤 We heard:**")
        st.code(st.session_state.raw_transcript or "—", language=None)
    with col_edit:
        st.markdown("**✏️ Edit if needed:**")
        edited = st.text_input(
            "Corrected sentence",
            value=st.session_state.corrected,
            label_visibility="collapsed",
            key="edit_sentence",
        )
        st.session_state.corrected = edited

    st.markdown("")
    col_a, col_r, _ = st.columns([2, 1, 4])
    with col_a:
        if st.button("🔍 Analyse Pronunciation", type="primary", width="stretch"):
            with st.spinner("Analysing phonemes…"):
                try:
                    report = analyzer.analyze(
                        audio_path=st.session_state.audio_path,
                        sentence=st.session_state.corrected,
                        user_id=st.session_state.get("user_id"),
                    )
                    st.session_state.report = report
                    st.session_state.stage  = "report"
                    st.rerun()
                except Exception as exc:
                    logger.exception("Analysis failed")
                    st.error(f"Analysis failed: {exc}")
    with col_r:
        if st.button("🔄 Re-record", width="stretch"):
            _reset()
            st.rerun()

# ==========================================================================
# STAGE 3 – WORD-BY-WORD ANALYSIS
# ==========================================================================
elif st.session_state.stage == "report":
    report: PronunciationReport = st.session_state.report
    score = report.overall_score

    score_color = (
        "#27ae60" if score >= 75
        else "#f39c12" if score >= 50
        else "#e74c3c"
    )
    score_emoji = (
        "🌟" if score >= 90 else
        "✅" if score >= 75 else
        "👍" if score >= 50 else
        "💪"
    )

    banner_col, btn_col = st.columns([3, 1])
    with banner_col:
        st.markdown(
            f'<span class="score-pill" style="background:{score_color}22;'
            f'color:{score_color};border:2px solid {score_color}">'
            f'{score_emoji} Overall Score: {score} / 100</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f'**Sentence:** *"{report.sentence}"*')
    with btn_col:
        if st.button("📊 View Full Report →", type="primary",
                     width="stretch", key="report_btn_top"):
            st.switch_page("pages/2_Overall_Report.py")

    st.divider()

    st.subheader("📝 Word-by-Word Phoneme Analysis")

    for wr in report.word_reports:
        score_icon = "✅" if wr.score >= 80 else ("⚠️" if wr.score >= 50 else "❌")

        with st.expander(
            f"{score_icon}  **{wr.word.upper()}**  ·  {wr.score}/100",
            expanded=wr.has_errors,
        ):
            ph_col, det_col, audio_col = st.columns([3, 3, 2])

            with ph_col:
                st.markdown("**Expected:**")
                error_exp = {e["expected_phoneme"] for e in wr.errors if e.get("expected_phoneme")}
                tags = [
                    f'<span class="phoneme-tag {"ph-error" if ph in error_exp else "ph-correct"}">'
                    f'{ph}</span>'
                    for ph in wr.expected_phonemes
                ]
                st.markdown(" ".join(tags) or "*—*", unsafe_allow_html=True)

            with det_col:
                st.markdown("**You produced:**")
                if wr.detected_phonemes:
                    error_det = {e["detected_phoneme"] for e in wr.errors if e.get("detected_phoneme")}
                    tags = [
                        f'<span class="phoneme-tag {"ph-error" if ph in error_det else "ph-correct"}">'
                        f'{ph}</span>'
                        for ph in wr.detected_phonemes
                    ]
                    st.markdown(" ".join(tags), unsafe_allow_html=True)
                else:
                    st.markdown("*Not detected*")

            with audio_col:
                st.markdown(f"**Score: {wr.score}/100**")
                audio_payload = tts_audio_service.word_audio_payload(wr.word)
                if audio_payload:
                    st.markdown("🔊 **Correct:**")
                    st.audio(audio_payload.data, format=audio_payload.format)

            if wr.errors:
                st.markdown("**Errors:**")
                rows = [{
                    "Expected": e.get("expected_phoneme") or "—",
                    "Detected": e.get("detected_phoneme") or "—",
                    "Type":     e.get("error_type", "").capitalize(),
                    "Severity": e.get("severity", "").capitalize(),
                } for e in wr.errors]
                st.table(rows)

            if wr.suggestion:
                st.success(f"💡 {wr.suggestion}")

    st.divider()

    col1, col2, col3 = st.columns([2, 2, 3])
    with col1:
        if st.button("📊 View Full Report →", type="primary",
                     width="stretch", key="report_btn_bottom"):
            st.switch_page("pages/2_Overall_Report.py")
    with col2:
        if st.button("🔁 Try Another Sentence",
                     width="stretch", key="retry_btn_bottom"):
            _reset()
            st.rerun()
    with col3:
        if st.button("👤 View My Profile", width="stretch"):
            st.switch_page("pages/3_Profile.py")
