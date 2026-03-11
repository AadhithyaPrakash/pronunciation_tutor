"""
Presentation Layer - Streamlit Application
------------------------------------------
Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Ensure absolute imports work when running via:
#   streamlit run app/streamlit_app.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from project .env for Streamlit runtime.
load_dotenv(PROJECT_ROOT / ".env")

from domain.learning_logic import MAX_RETRIES
from infrastructure.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

from app.controller import ConversationController, SessionState
from services import tts_service


def _init_state() -> None:
    if "ctrl" not in st.session_state:
        logger.info("Initializing new ConversationController in session state")
        st.session_state.ctrl = ConversationController()


def _save_audio_bytes(audio_bytes: bytes) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(audio_bytes)
    tmp.close()
    logger.info("Saved audio bytes to temp file: %s (%d bytes)", tmp.name, len(audio_bytes))
    return Path(tmp.name)


def _show_articulation_image(errors: list) -> None:
    """Display articulation guide image for the first error, if available."""
    if not errors:
        return
    first_phoneme = errors[0].get("expected_phoneme", "")
    img_path = Path("assets/articulation_images") / f"{first_phoneme}.png"
    if img_path.exists():
        logger.debug("Showing articulation image for phoneme=%s", first_phoneme)
        st.image(str(img_path), caption=f"How to produce /{first_phoneme}/", width=300)


st.set_page_config(
    page_title="Pronunciation Tutor",
    page_icon="🗣️",
    layout="centered",
)

_init_state()
ctrl: ConversationController = st.session_state.ctrl
session = ctrl.session
logger.debug("UI rerun in state=%s", session.state.name)

st.title("🗣️ Pronunciation Tutor")
st.caption("Speak naturally. The tutor will help you correct each word, one at a time.")
st.divider()


if session.state == SessionState.LISTEN_SENTENCE:
    st.subheader("Step 1 - Record your sentence")
    st.info(
        "Click Record and read any English sentence aloud. "
        "Speak clearly at a natural pace."
    )

    audio_value = st.audio_input("🎙️ Click to record your sentence")
    if audio_value is not None:
        logger.info("Sentence audio captured from UI")
        audio_bytes = audio_value.read()
        with st.spinner("Transcribing..."):
            audio_path = _save_audio_bytes(audio_bytes)
            try:
                ctrl.handle_sentence_audio(audio_path)
                st.rerun()
            except Exception as exc:
                logger.exception("Sentence transcription flow failed")
                st.error(f"Transcription failed: {exc}")

elif session.state == SessionState.CONFIRM_SENTENCE:
    st.subheader("Step 2 - Confirm your sentence")
    st.write("**We heard:**")
    st.code(session.raw_transcript, language=None)

    st.write("**Corrected to:**")
    edited = st.text_input(
        "Edit if needed, then confirm:",
        value=session.corrected_sentence,
        key="confirm_input",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Confirm sentence", use_container_width=True):
            logger.info("User confirmed sentence; starting alignment")
            with st.spinner("Aligning phonemes..."):
                try:
                    ctrl.confirm_sentence(confirmed_sentence=edited)
                    st.rerun()
                except Exception as exc:
                    logger.exception("Sentence confirmation/alignment failed")
                    st.error(f"Alignment failed: {exc}")
    with col2:
        if st.button("Re-record", use_container_width=True):
            logger.info("User requested re-record; resetting to LISTEN_SENTENCE")
            session.state = SessionState.LISTEN_SENTENCE
            st.rerun()

elif session.state == SessionState.PROCESS_WORD:
    st.info("Analysing pronunciation...")
    logger.info("Transient PROCESS_WORD state reached; triggering rerun")
    st.rerun()

elif session.state in (SessionState.EXPLAIN_ERROR, SessionState.RETRY_WORD):
    word = session.current_word
    progress = session.current_progress

    st.subheader(f"Word {session.word_index + 1} of {len(session.words)}: **{word}**")
    st.progress((session.word_index) / max(len(session.words), 1))

    if session.current_errors:
        st.warning("Pronunciation errors detected:")
        rows = []
        for error in session.current_errors:
            rows.append(
                {
                    "Expected": error.get("expected_phoneme", "-"),
                    "Detected": error.get("detected_phoneme", "-"),
                    "Error Type": error.get("error_type", "-").capitalize(),
                    "Severity": error.get("severity", "-").capitalize(),
                }
            )
        st.table(rows)

    if session.current_explanation:
        st.info(f"**Tutor says:** {session.current_explanation}")
        col_tts, _ = st.columns([1, 3])
        with col_tts:
            if st.button("Hear explanation", key="tts_btn"):
                logger.info("TTS requested for current explanation")
                tts_service.speak(session.current_explanation)

    _show_articulation_image(session.current_errors)
    st.divider()

    attempt_label = (
        f"Attempt {progress.attempt_count + 1} / {MAX_RETRIES}" if progress else "Retry"
    )
    st.write(f"**{attempt_label} - Record just the word '{word}':**")
    retry_audio = st.audio_input(
        "🎙️ Record word",
        key=f"retry_{session.word_index}_{progress.attempt_count if progress else 0}",
    )

    if retry_audio is not None:
        logger.info("Retry audio captured for word='%s'", word)
        retry_bytes = retry_audio.read()
        with st.spinner("Evaluating..."):
            retry_path = _save_audio_bytes(retry_bytes)
            try:
                ctrl.handle_word_audio(retry_path)
                st.rerun()
            except Exception as exc:
                logger.exception("Word evaluation flow failed for word='%s'", word)
                st.error(f"Evaluation failed: {exc}")

    if st.button("Skip this word"):
        logger.info("User skipped word='%s'", word)
        ctrl.advance_to_next_word()
        st.rerun()

elif session.state == SessionState.NEXT_WORD:
    st.success(session.feedback_message)
    logger.info("NEXT_WORD state message: %s", session.feedback_message)
    st.rerun()

elif session.state == SessionState.SESSION_SUMMARY:
    st.subheader("Session Complete")
    logger.info("Rendering SESSION_SUMMARY for session_id=%s", session.session_id)
    st.balloons()

    progresses = session.word_progresses
    if progresses:
        st.write("### Word-by-Word Results")
        for progress in progresses:
            icon = "PASS" if progress.passed else "FAIL"
            accuracy_pct = round(progress.best_accuracy * 100, 1)
            st.write(
                f"{icon} **{progress.word}** - best accuracy: {accuracy_pct}% "
                f"({progress.attempt_count} attempt{'s' if progress.attempt_count != 1 else ''})"
            )

    st.divider()
    if session.summary:
        st.write("### Coach Feedback")
        st.write(session.summary)

    if st.button("Start a new session"):
        logger.info("User started a new session from summary screen")
        st.session_state.ctrl = ConversationController()
        st.rerun()
