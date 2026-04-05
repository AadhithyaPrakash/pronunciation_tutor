"""
Services Layer - ASR (Automatic Speech Recognition)
----------------------------------------------------
Improved version with:
- VAD (Voice Activity Detection) for poor mic conditions
- Audio normalisation before transcription
- Better silence handling
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_model = None
_backend_in_use = None

_model_size        = os.getenv("WHISPER_MODEL", "base").split("#", 1)[0].strip()
_preferred_backend = os.getenv("ASR_BACKEND", "faster_whisper").strip().lower()
_device            = os.getenv("WHISPER_DEVICE", "cpu").strip().lower()
_compute_type      = os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip().lower()
_vad_filter        = os.getenv("WHISPER_VAD_FILTER", "false").strip().lower() == "true"


@dataclass
class WordTimestamp:
    word:  str
    start: float
    end:   float


def _load_faster_whisper():
    from faster_whisper import WhisperModel
    logger.info("Loading faster-whisper model='%s' device='%s' compute_type='%s'",
                _model_size, _device, _compute_type)
    return WhisperModel(_model_size, device=_device, compute_type=_compute_type)


def _load_openai_whisper():
    import whisper
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
    backend_order = (
        ["faster_whisper", "openai_whisper"]
        if _preferred_backend != "openai_whisper"
        else ["openai_whisper", "faster_whisper"]
    )
    last_error = None
    for name in backend_order:
        try:
            started = time.perf_counter()
            model   = loaders[name]()
            _model, _backend_in_use = model, name
            logger.info("ASR ready: %s (%.2fs)", name, time.perf_counter() - started)
            return _model, _backend_in_use
        except Exception as exc:
            last_error = exc
            logger.error("ASR backend '%s' failed: %s", name, exc)
    raise RuntimeError("No ASR backend available.") from last_error


def _paths_match(left: str | Path, right: str | Path) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except Exception:
        return str(left) == str(right)


def _cleanup_preprocessed_audio(original_path: str | Path, clean_path: str | Path) -> None:
    try:
        if _paths_match(original_path, clean_path):
            return
        Path(clean_path).unlink(missing_ok=True)
    except Exception as exc:
        logger.debug("Could not clean up temp audio '%s': %s", clean_path, exc)


def _transcribe_preprocessed_audio(clean_path: str | Path) -> str:
    model, backend = _get_model()
    logger.info("Transcribing via %s: %s", backend, clean_path)
    t0 = time.perf_counter()

    if backend == "faster_whisper":
        segments, _ = model.transcribe(
            str(clean_path),
            language="en",
            vad_filter=_vad_filter,
            vad_parameters={"min_silence_duration_ms": 300} if _vad_filter else {},
            beam_size=5,
            best_of=5,
            temperature=0.0,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
    else:
        result = model.transcribe(
            str(clean_path),
            language="en",
            fp16=False,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.4,
            logprob_threshold=-1.0,
        )
        text = result["text"].strip()

    logger.info("Transcription done in %.2fs: %r", time.perf_counter() - t0, text)
    return text


# ── Audio preprocessing for poor mics ───────────────────────────────────────

def _preprocess_audio(audio_path: str | Path) -> str:
    """
    Normalize and denoise audio for better transcription on poor mics.
    Returns path to cleaned temp file.
    """
    tmp_path = None
    try:
        import subprocess, shutil
        if not shutil.which("ffmpeg"):
            return str(audio_path)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        tmp_path = tmp.name

        # FFmpeg: convert to 16kHz mono, apply volume normalisation,
        # high-pass filter (removes low rumble), and afftdn noise reduction
        cmd = [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ar", "16000",
            "-ac", "1",
            "-af", "highpass=f=80,afftdn=nf=-25,loudnorm",
            tmp_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode == 0:
            logger.info("Audio preprocessed via FFmpeg for poor-mic improvement")
            return tmp_path
        else:
            Path(tmp_path).unlink(missing_ok=True)
            logger.warning("FFmpeg preprocessing failed, using original audio")
            return str(audio_path)
    except Exception as exc:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
        logger.warning("Audio preprocessing skipped: %s", exc)
        return str(audio_path)


# ── Public API ───────────────────────────────────────────────────────────────

def transcribe_audio(audio_path: str | Path) -> str:
    clean_path = _preprocess_audio(audio_path)
    try:
        return _transcribe_preprocessed_audio(clean_path)
    finally:
        _cleanup_preprocessed_audio(audio_path, clean_path)


def transcribe_bytes(audio_bytes: bytes, suffix: str = ".wav") -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        return transcribe_audio(tmp_path)
    finally:
        os.unlink(tmp_path)


def transcribe_with_word_timestamps(
    audio_path: str | Path,
) -> Tuple[str, List[WordTimestamp]]:
    clean_path = _preprocess_audio(audio_path)
    try:
        model, backend = _get_model()
        logger.info("Transcribing with word timestamps via %s: %s", backend, clean_path)
        t0 = time.perf_counter()

        if backend == "faster_whisper":
            text, words = _timestamps_faster_whisper(model, str(clean_path))
        else:
            text, words = _timestamps_openai_whisper(model, str(clean_path))

        logger.info("Timestamps done in %.2fs: %d words, text=%r",
                    time.perf_counter() - t0, len(words), text)
        return text, words

    except Exception as exc:
        logger.warning("Word timestamp extraction failed (%s); falling back", exc)
        return _transcribe_preprocessed_audio(clean_path), []
    finally:
        _cleanup_preprocessed_audio(audio_path, clean_path)


def _timestamps_faster_whisper(model, audio_path: str) -> Tuple[str, List[WordTimestamp]]:
    segments, _ = model.transcribe(
        audio_path,
        language="en",
        word_timestamps=True,
        vad_filter=_vad_filter,
        beam_size=5,
        temperature=0.0,
    )
    words: List[WordTimestamp] = []
    parts: List[str] = []
    for seg in segments:
        parts.append(seg.text.strip())
        if seg.words:
            for w in seg.words:
                words.append(WordTimestamp(word=w.word.strip(), start=w.start, end=w.end))
    return " ".join(parts).strip(), words


def _timestamps_openai_whisper(model, audio_path: str) -> Tuple[str, List[WordTimestamp]]:
    result = model.transcribe(
        audio_path,
        language="en",
        fp16=False,
        word_timestamps=True,
        beam_size=5,
        temperature=0.0,
        no_speech_threshold=0.4,
    )
    words: List[WordTimestamp] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            words.append(WordTimestamp(word=w["word"].strip(), start=w["start"], end=w["end"]))
    return result["text"].strip(), words
