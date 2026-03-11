"""
Services Layer – Phoneme Recognition
--------------------------------------
Uses a wav2vec2 model fine-tuned on TIMIT phonemes to extract the actual
phoneme sequence directly from audio, without going through word-level ASR.

This is the critical component for detecting pronunciation errors:
  - Expected phonemes: CMUdict lookup on the CORRECT word (text-based)
  - Detected phonemes: wav2vec2 inference on the AUDIO (audio-based)  ← this file

Without this, the system must rely on ASR word transcription + CMUdict,
which fails because Whisper normalises mispronounced words to their correct
spelling (e.g. saying "tink" → Whisper outputs "think" → same phonemes as
expected → zero errors detected even though the pronunciation was wrong).

Model used: vitouphy/wav2vec2-xls-r-300m-timit-phoneme
  - Fine-tuned on TIMIT, outputs TIMIT phoneme labels
  - TIMIT phonemes are a subset of ARPAbet and map 1-to-1 or near 1-to-1

Requirements:
  pip install transformers torchaudio torch
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model loading (lazy, cached)
# ---------------------------------------------------------------------------

_processor = None
_model     = None

MODEL_ID = "vitouphy/wav2vec2-xls-r-300m-timit-phoneme"


def _get_model():
    global _processor, _model
    if _processor is None:
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        import torch

        logger.info("Loading phoneme recognition model: %s", MODEL_ID)
        _processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
        _model     = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
        _model.eval()
        logger.info("Phoneme recognition model loaded")
    return _processor, _model


# ---------------------------------------------------------------------------
# TIMIT → ARPAbet mapping
# ---------------------------------------------------------------------------
# TIMIT uses slightly different labels from CMUdict ARPAbet.
# This mapping normalises them so they can be compared directly.

_TIMIT_TO_ARPABET = {
    # Vowels – stress markers stripped from CMUdict, so plain label is fine
    "iy": "IY",   "ih": "IH",  "eh": "EH",  "ae": "AE",
    "aa": "AA",   "aw": "AW",  "ay": "AY",  "ah": "AH",
    "ao": "AO",   "oy": "OY",  "ow": "OW",  "uh": "UH",
    "uw": "UW",   "ux": "UW",  "er": "ER",  "ax": "AH",
    "ix": "IH",   "axr": "ER", "ax-h": "AH",
    # Consonants
    "b":  "B",   "ch": "CH",  "d":  "D",   "dh": "DH",
    "dx": "D",   "el": "L",   "em": "M",   "en": "N",
    "eng":"NG",  "f":  "F",   "g":  "G",   "hh": "HH",
    "jh": "JH",  "k":  "K",   "l":  "L",   "m":  "M",
    "n":  "N",   "ng": "NG",  "nx": "N",   "p":  "P",
    "q":  "K",   "r":  "R",   "s":  "S",   "sh": "SH",
    "t":  "T",   "th": "TH",  "v":  "V",   "w":  "W",
    "wh": "W",   "y":  "Y",   "z":  "Z",   "zh": "ZH",
    # Silence / non-speech → skip
    "sil": None, "sp": None,  "spn": None, "": None,
}


def _timit_to_arpabet(timit: str) -> Optional[str]:
    """Convert a TIMIT label to ARPAbet. Returns None for silence tokens."""
    return _TIMIT_TO_ARPABET.get(timit.lower())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recognize_phonemes(audio_path: str | Path) -> List[str]:
    """
    Run wav2vec2 phoneme recognition on an audio file.

    Returns a list of ARPAbet phoneme strings (stress-free), e.g.:
        ["T", "IH", "NG", "K"]   ← for a mispronounced "think" said as "tink"

    Returns an empty list if the model cannot be loaded (graceful degradation).
    """
    try:
        return _run_inference(audio_path)
    except Exception as exc:
        logger.error("Phoneme recognition failed: %s", exc)
        return []


def recognize_phonemes_for_word(
    audio_path: str | Path,
    word: str,
    expected_phonemes: List[str],
) -> List[str]:
    """
    Recognise phonemes for a single-word audio recording.

    Falls back to expected phonemes ONLY if the model is completely
    unavailable (import error).  Any other result — even an empty list —
    is returned as-is so that errors are correctly detected.

    Parameters
    ----------
    audio_path        : path to single-word WAV recording
    word              : the target word (used for logging only)
    expected_phonemes : CMUdict expected sequence (used as fallback)
    """
    try:
        detected = _run_inference(audio_path)
        logger.info(
            "Phoneme recognition word='%s'  detected=%s  expected=%s",
            word, detected, expected_phonemes,
        )
        return detected
    except ImportError:
        # transformers / torch not installed → return expected so the rest
        # of the pipeline doesn't crash (no errors detected, but at least
        # won't throw).
        logger.warning(
            "transformers/torch not installed; phoneme recognition unavailable for '%s'",
            word,
        )
        return expected_phonemes
    except Exception as exc:
        logger.error("Phoneme recognition failed for word='%s': %s", word, exc)
        return []


# ---------------------------------------------------------------------------
# Internal inference
# ---------------------------------------------------------------------------

def _run_inference(audio_path: str | Path) -> List[str]:
    import torch
    import torchaudio

    processor, model = _get_model()

    # Load and resample to 16 kHz (wav2vec2 requirement)
    waveform, sample_rate = torchaudio.load(str(audio_path))
    if sample_rate != 16_000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=16_000
        )
        waveform = resampler(waveform)

    # Convert to mono float32 numpy array
    audio_np = waveform.squeeze().numpy().astype(np.float32)

    # Tokenise
    inputs = processor(
        audio_np,
        sampling_rate=16_000,
        return_tensors="pt",
        padding=True,
    )

    # Inference (no gradient needed)
    with torch.no_grad():
        logits = model(**inputs).logits

    # Decode predicted phoneme ids → labels
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    # transcription is a space-separated string of TIMIT phoneme labels
    # e.g. "t ih ng k"
    raw_tokens = transcription.strip().lower().split()

    # Convert TIMIT → ARPAbet, drop silence tokens
    arpabet = []
    for token in raw_tokens:
        mapped = _timit_to_arpabet(token)
        if mapped is not None:
            arpabet.append(mapped)

    # Deduplicate consecutive identical phonemes (CTC output artefact)
    deduped = _ctc_collapse(arpabet)

    logger.debug("Raw TIMIT tokens: %s  →  ARPAbet: %s", raw_tokens, deduped)
    return deduped


def _ctc_collapse(phonemes: List[str]) -> List[str]:
    """Remove consecutive duplicate phonemes (standard CTC blank collapse)."""
    if not phonemes:
        return []
    result = [phonemes[0]]
    for ph in phonemes[1:]:
        if ph != result[-1]:
            result.append(ph)
    return result
