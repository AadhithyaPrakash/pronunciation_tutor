"""
tests/test_domain.py

Unit tests for the domain layer — no external services needed.
Run with: pytest tests/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from domain.error_detection import detect_errors, phoneme_error_rate
from domain.severity_scoring import score_error, score_word_analysis, overall_word_severity
from domain.phoneme_alignment import get_expected_phonemes, _strip_stress


class TestPhonemeAlignment:

    def test_known_word(self):
        phones = get_expected_phonemes("think")
        assert isinstance(phones, list)
        assert len(phones) > 0
        assert "TH" in phones

    def test_unknown_word_returns_empty(self):
        assert get_expected_phonemes("xyzqwerty") == []

    def test_stress_stripped(self):
        assert _strip_stress("AH1") == "AH"

    def test_no_stress_unchanged(self):
        assert _strip_stress("TH") == "TH"


class TestErrorDetection:

    def test_perfect_match_no_errors(self):
        analysis = detect_errors("think",
                                  expected=["TH", "IH", "NG", "K"],
                                  detected=["TH", "IH", "NG", "K"])
        assert analysis.is_correct is True
        assert analysis.error_count == 0

    def test_substitution_detected(self):
        analysis = detect_errors("think",
                                  expected=["TH", "IH", "NG", "K"],
                                  detected=["T", "IH", "NG", "K"])
        assert not analysis.is_correct
        subs = [e for e in analysis.errors if e.error_type == "substitution"]
        assert len(subs) == 1
        assert subs[0].expected_phoneme == "TH"
        assert subs[0].detected_phoneme == "T"

    def test_deletion_detected(self):
        analysis = detect_errors("think",
                                  expected=["TH", "IH", "NG", "K"],
                                  detected=["IH", "NG", "K"])
        deletions = [e for e in analysis.errors if e.error_type == "deletion"]
        assert len(deletions) >= 1

    def test_insertion_detected(self):
        analysis = detect_errors("bed",
                                  expected=["B", "EH", "D"],
                                  detected=["B", "EH", "D", "Z"])
        insertions = [e for e in analysis.errors if e.error_type == "insertion"]
        assert len(insertions) >= 1

    def test_empty_detected_all_deletions(self):
        analysis = detect_errors("think",
                                  expected=["TH", "IH", "NG", "K"],
                                  detected=[])
        assert not analysis.is_correct
        assert all(e.error_type == "deletion" for e in analysis.errors)

    def test_empty_expected_returns_correct(self):
        assert detect_errors("xyzqwerty", expected=[], detected=["T"]).is_correct

    def test_per_perfect(self):
        assert phoneme_error_rate(["TH", "IH", "NG", "K"], ["TH", "IH", "NG", "K"]) == 0.0

    def test_per_empty_expected(self):
        assert phoneme_error_rate([], ["TH"]) == 0.0


class TestSeverityScoring:

    def _make_error(self, exp, det, etype="substitution", pos=0):
        from domain.error_detection import PhonemeError
        return PhonemeError(word="test", expected_phoneme=exp,
                            detected_phoneme=det, error_type=etype, position=pos)

    def test_high_confidence_minor(self):
        err = self._make_error("TH", "T")
        assert score_error(err, confidence=0.9).severity == "minor"

    def test_low_confidence_severe(self):
        err = self._make_error("TH", "B")
        assert score_error(err, confidence=0.1).severity == "severe"

    def test_deletion_at_least_moderate(self):
        err = self._make_error("TH", None, etype="deletion")
        assert score_error(err, confidence=0.8).severity in ("moderate", "severe")

    def test_overall_no_errors_returns_none(self):
        from domain.error_detection import WordAnalysis
        analysis = WordAnalysis(word="think", expected_phonemes=["TH", "IH"],
                                detected_phonemes=["TH", "IH"], errors=[], is_correct=True)
        assert overall_word_severity(analysis) == "none"
