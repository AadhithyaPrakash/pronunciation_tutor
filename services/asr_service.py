"""
Services Layer - ASR (Automatic Speech Recognition)
---------------------------------------------------
Provides speech-to-text with backend fallback.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_model = None
_backend_in_use = None

_model_size = os.getenv("WHISPER_MODEL", "base").split("#", 1)[0].strip()
_preferred_backend = os.getenv("ASR_BACKEND", "faster_whisper").strip().lower()
_device = os.getenv("WHISPER_DEVICE", "cpu").strip().lower()
_compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip().lower()


def _load_faster_whisper():
    from faster_whisper import WhisperModel  # pip install faster-whisper

    logger.info(
        "Loading faster-whisper model='%s' device='%s' compute_type='%s'",
        _model_size,
        _device,
        _compute_type,
    )
    return WhisperModel(_model_size, device=_device, compute_type=_compute_type)


def _load_openai_whisper():
    import whisper  # pip install openai-whisper

    logger.info("Loading openai-whisper model='%s'", _model_size)
    return whisper.load_model(_model_size)


def _get_model():
    global _model, _backend_in_use
    if _model is not None:
        return _model, _backend_in_use

    loaders = {
        "faster_whisper": _load_faster_whisper,
        "openai_whisper": _load_openai_whisper,
    }

    if _preferred_backend not in loaders:
        logger.warning(
            "Unknown ASR_BACKEND='%s'. Falling back to faster_whisper.",
            _preferred_backend,
        )
        backend_order = ["faster_whisper", "openai_whisper"]
    elif _preferred_backend == "faster_whisper":
        backend_order = ["faster_whisper", "openai_whisper"]
    else:
        backend_order = ["openai_whisper", "faster_whisper"]

    last_error = None
    for backend_name in backend_order:
        try:
            started = time.perf_counter()
            model = loaders[backend_name]()
            elapsed = time.perf_counter() - started
            _model = model
            _backend_in_use = backend_name
            logger.info("ASR backend ready: %s (loaded in %.2fs)", backend_name, elapsed)
            return _model, _backend_in_use
        except Exception as exc:
            last_error = exc
            logger.exception("Failed to initialize ASR backend '%s': %s", backend_name, exc)

    raise RuntimeError(
        "Could not initialize any ASR backend. "
        "Install dependencies and check network access for model download."
    ) from last_error


def transcribe_audio(audio_path: str | Path) -> str:
    """Transcribe a WAV/MP3 file and return transcript text."""
    model, backend = _get_model()
    logger.info("Starting transcription using backend=%s audio_path=%s", backend, audio_path)
    started = time.perf_counter()

    if backend == "faster_whisper":
        segments, _ = model.transcribe(str(audio_path), language="en")
        text = " ".join(segment.text.strip() for segment in segments).strip()
    else:
        result = model.transcribe(str(audio_path), language="en", fp16=False)
        text = result["text"].strip()

    elapsed = time.perf_counter() - started
    logger.info("Transcription complete in %.2fs (chars=%d)", elapsed, len(text))
    return text


def transcribe_bytes(audio_bytes: bytes, suffix: str = ".wav") -> str:
    """Write bytes to temp file, transcribe, then delete."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    logger.debug("Created temp audio file for byte transcription: %s", tmp_path)

    try:
        return transcribe_audio(tmp_path)
    finally:
        os.unlink(tmp_path)
        logger.debug("Deleted temp audio file: %s", tmp_path)
