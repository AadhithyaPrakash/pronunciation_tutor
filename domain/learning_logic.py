"""
Domain Layer – Learning Logic
------------------------------
Decides when a word is "good enough" and tracks per-word attempt history.
Pure logic – no I/O.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from domain.severity_scoring import Severity


MAX_RETRIES = 3          # Give up after this many failed attempts per word
PASS_THRESHOLD = 0.75    # Phoneme accuracy ratio to consider a word "passed"


@dataclass
class WordAttempt:
    """Records one attempt at pronouncing a word."""
    word: str
    expected_phonemes: List[str]
    detected_phonemes: List[str]
    errors: List[dict]            # annotated error dicts from severity_scoring
    accuracy: float               # 0.0 – 1.0


@dataclass
class WordProgress:
    """Accumulates attempts for a single word across retries."""
    word: str
    attempts: List[WordAttempt] = field(default_factory=list)

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def passed(self) -> bool:
        if not self.attempts:
            return False
        return self.attempts[-1].accuracy >= PASS_THRESHOLD

    @property
    def give_up(self) -> bool:
        return self.attempt_count >= MAX_RETRIES and not self.passed

    @property
    def best_accuracy(self) -> float:
        if not self.attempts:
            return 0.0
        return max(a.accuracy for a in self.attempts)

    def add_attempt(self, attempt: WordAttempt) -> None:
        self.attempts.append(attempt)


def compute_accuracy(expected: List[str], detected: List[str]) -> float:
    """
    Simple phoneme accuracy: fraction of expected phonemes that are correct.
    Uses longest-common-subsequence matching.
    """
    if not expected:
        return 1.0

    # Build LCS length table
    m, n = len(expected), len(detected)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if expected[i - 1] == detected[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    return lcs_len / len(expected)


def should_explain(errors: List[dict], severity_threshold: str = "minor") -> bool:
    """Return True if any error meets or exceeds the severity threshold."""
    order = [Severity.MINOR, Severity.MODERATE, Severity.SEVERE]
    threshold_idx = order.index(Severity(severity_threshold))
    for e in errors:
        if order.index(Severity(e["severity"])) >= threshold_idx:
            return True
    return False
