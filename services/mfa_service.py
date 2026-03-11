"""
Services Layer – MFA / Phoneme Alignment Service
-------------------------------------------------
Primary path  : Montreal Forced Aligner (MFA) – full timestamp alignment.
Fallback path : wav2vec2 phoneme recognition – audio-based phoneme detection
                with CMUdict for expected phonemes.

The fallback is what runs when MFA is not installed, which is the common
case during development.  It directly analyses the audio at the phoneme
level, so it correctly detects errors even when Whisper transcribes a
mispronounced word correctly (e.g. saying "tink" but Whisper still outputs
"think" – the fallback reads the audio itself and finds T IH NG K, not
TH IH NG K).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import nltk
from nltk.corpus import cmudict

from domain.phoneme_alignment import WordAlignment, build_word_alignment
from services import phoneme_recognition_service

logger = logging.getLogger(__name__)

_cmu: Optional[dict] = None

MFA_ACOUSTIC_MODEL = os.getenv("MFA_ACOUSTIC_MODEL", "english_us_arpa")
MFA_DICTIONARY     = os.getenv("MFA_DICTIONARY",     "english_us_arpa")


# ---------------------------------------------------------------------------
# CMUdict helpers
# ---------------------------------------------------------------------------

def _get_cmudict() -> dict:
    global _cmu
    if _cmu is None:
        try:
            _cmu = cmudict.dict()
        except LookupError:
            nltk.download("cmudict", quiet=True)
            _cmu = cmudict.dict()
    return _cmu


def get_expected_phonemes(word: str) -> List[str]:
    """Return first CMUdict pronunciation for a word, without stress digits."""
    entries = _get_cmudict().get(word.lower())
    if not entries:
        logger.debug("No CMUdict entry for word='%s'", word)
        return []
    return [ph.rstrip("012") for ph in entries[0]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def align_audio(
    audio_path: str | Path,
    transcript: str,
    asr_transcript: Optional[str] = None,   # kept for API compat, unused in fallback
) -> List[WordAlignment]:
    """
    Align audio against a reference transcript.

    Tries MFA first; falls back to wav2vec2 phoneme recognition.

    Parameters
    ----------
    audio_path     : WAV file (16 kHz mono preferred)
    transcript     : the confirmed/correct sentence or word
    asr_transcript : unused in the new fallback; kept for backward compatibility
    """
    logger.info(
        "align_audio called — transcript=%r  audio=%s", transcript, audio_path
    )
    try:
        alignments = _mfa_align(audio_path, transcript)
        logger.info("MFA alignment succeeded (%d words)", len(alignments))
        return alignments
    except Exception as exc:
        logger.warning("MFA unavailable (%s); using wav2vec2 phoneme fallback", exc)
        return _wav2vec2_fallback(audio_path, transcript)


# ---------------------------------------------------------------------------
# MFA path
# ---------------------------------------------------------------------------

def _mfa_align(audio_path: str | Path, transcript: str) -> List[WordAlignment]:
    import textgrid

    audio_path = Path(audio_path)
    work_dir   = Path(tempfile.mkdtemp(prefix="mfa_"))
    try:
        corpus_dir = work_dir / "corpus"
        corpus_dir.mkdir()
        output_dir = work_dir / "output"
        output_dir.mkdir()

        shutil.copy(audio_path, corpus_dir / "utterance.wav")
        (corpus_dir / "utterance.txt").write_text(transcript, encoding="utf-8")

        cmd = [
            "mfa", "align",
            str(corpus_dir), MFA_DICTIONARY, MFA_ACOUSTIC_MODEL,
            str(output_dir), "--clean",
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        tg = textgrid.TextGrid.fromFile(str(output_dir / "utterance.TextGrid"))
        return _parse_textgrid(tg)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _parse_textgrid(tg) -> List[WordAlignment]:
    word_tier    = next(t for t in tg.tiers if t.name.lower() == "words")
    phoneme_tier = next(
        t for t in tg.tiers if t.name.lower() in ("phones", "phonemes")
    )
    alignments = []
    for w in word_tier:
        label = w.mark.strip()
        if not label or label in ("<eps>", "sp"):
            continue
        ph_data = []
        for p in phoneme_tier:
            if p.maxTime <= w.minTime:
                continue
            if p.minTime >= w.maxTime:
                break
            ph_label = p.mark.strip().rstrip("012")
            if ph_label and ph_label not in ("<eps>", "sp"):
                ph_data.append({
                    "phoneme": ph_label, "start": p.minTime,
                    "end": p.maxTime,   "confidence": 1.0,
                })
        alignments.append(build_word_alignment(
            word=label, start=w.minTime, end=w.maxTime, phoneme_data=ph_data
        ))
    return alignments


# ---------------------------------------------------------------------------
# wav2vec2 fallback  ← THE KEY FIX
# ---------------------------------------------------------------------------

def _wav2vec2_fallback(
    audio_path: str | Path,
    transcript: str,
) -> List[WordAlignment]:
    """
    Fallback when MFA is unavailable.

    Uses wav2vec2 phoneme recognition on the FULL audio to get the phoneme
    sequence actually produced by the speaker, then heuristically assigns
    those phonemes to each word based on expected phoneme counts.

    This correctly detects errors because it reads the AUDIO, not the ASR
    text — so if the speaker says "tink" instead of "think", the audio
    gives back T IH NG K rather than TH IH NG K.
    """
    words = transcript.split()

    # Get the full detected phoneme sequence from audio
    all_detected = phoneme_recognition_service.recognize_phonemes(audio_path)
    logger.info(
        "wav2vec2 fallback — transcript=%r  detected=%s", transcript, all_detected
    )

    # Get expected phoneme counts per word (for slicing the detected sequence)
    expected_per_word = [get_expected_phonemes(w) for w in words]
    total_expected    = sum(len(e) for e in expected_per_word)

    alignments = []

    if not all_detected or total_expected == 0:
        # No audio signal at all → return empty detected for every word
        # so error_detection marks everything as deletion errors
        logger.warning(
            "No detected phonemes from wav2vec2; detected will be empty for all words"
        )
        for word in words:
            alignments.append(build_word_alignment(
                word=word, start=0.0, end=0.0, phoneme_data=[]
            ))
        return alignments

    # Distribute detected phonemes across words proportionally to their
    # expected phoneme counts (best we can do without timestamps)
    detected_cursor = 0
    for word, expected in zip(words, expected_per_word):
        n_expected = len(expected)
        if n_expected == 0:
            alignments.append(build_word_alignment(
                word=word, start=0.0, end=0.0, phoneme_data=[]
            ))
            continue

        # How many detected phonemes to assign to this word
        # Use proportion of expected length, clamped to remaining detected
        remaining_detected  = len(all_detected) - detected_cursor
        remaining_expected  = sum(len(e) for e in expected_per_word[words.index(word):])
        proportion          = n_expected / max(remaining_expected, 1)
        n_assign            = max(1, round(proportion * remaining_detected))
        n_assign            = min(n_assign, remaining_detected)

        word_detected = all_detected[detected_cursor : detected_cursor + n_assign]
        detected_cursor += n_assign

        ph_data = [
            {"phoneme": ph, "start": 0.0, "end": 0.0, "confidence": 0.8}
            for ph in word_detected
        ]
        alignments.append(build_word_alignment(
            word=word, start=0.0, end=0.0, phoneme_data=ph_data
        ))
        logger.debug(
            "word='%s'  expected=%s  assigned_detected=%s",
            word, expected, word_detected,
        )

    return alignments
