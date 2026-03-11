"""
Infrastructure Layer - Audio Processing
---------------------------------------
Utilities for recording, loading, and normalizing audio.
"""

from __future__ import annotations

import logging
import tempfile
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "int16"
DEFAULT_SECS = 5


def record_audio(duration: int = DEFAULT_SECS) -> np.ndarray:
    """Record audio from default microphone."""
    import sounddevice as sd

    logger.info("Recording audio for %d seconds", duration)
    frames = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
    )
    sd.wait()
    flattened = frames.flatten()
    logger.info("Recording complete (samples=%d)", len(flattened))
    return flattened


def record_to_file(path: str | Path, duration: int = DEFAULT_SECS) -> Path:
    """Record and write WAV file."""
    audio = record_audio(duration)
    path = Path(path)
    _save_wav(path, audio)
    logger.info("Audio recorded to file: %s", path)
    return path


def record_to_tempfile(duration: int = DEFAULT_SECS) -> Path:
    """Record audio to a temporary WAV file."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    return record_to_file(tmp.name, duration)


def _save_wav(path: Path, audio: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio.tobytes())


def bytes_to_wav_file(audio_bytes: bytes, suffix: str = ".wav") -> Path:
    """Write audio bytes to a temporary file."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(audio_bytes)
    tmp.close()
    logger.debug("Wrote %d bytes to temp wav: %s", len(audio_bytes), tmp.name)
    return Path(tmp.name)


def load_wav(path: str | Path) -> np.ndarray:
    """Load WAV file and normalize int16 to float32."""
    import scipy.io.wavfile as wavfile

    rate, data = wavfile.read(str(path))
    logger.debug("Loaded wav file: %s (rate=%s, samples=%d)", path, rate, len(data))
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    return data


def normalise(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize audio array."""
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    return audio / peak
