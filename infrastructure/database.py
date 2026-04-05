"""
Infrastructure Layer - SQLite Database
--------------------------------------
Handles persistence for users (with auth), sessions, and word-level results.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(os.getenv("DB_PATH", "data/pronunciation_tutor.db")).expanduser()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    username    TEXT UNIQUE NOT NULL,
    email       TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER REFERENCES users(id),
    sentence   TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at   TEXT,
    summary    TEXT,
    overall_score INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS word_results (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER REFERENCES sessions(id),
    word          TEXT NOT NULL,
    attempts      INTEGER NOT NULL DEFAULT 0,
    passed        INTEGER NOT NULL DEFAULT 0,
    best_accuracy REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS phoneme_errors (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    word_result_id   INTEGER REFERENCES word_results(id),
    expected_phoneme TEXT,
    detected_phoneme TEXT,
    error_type       TEXT,
    severity         TEXT,
    confidence       REAL
);
"""


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA)
        # Migration: add missing columns if upgrading from older schema
        try:
            conn.execute("ALTER TABLE sessions ADD COLUMN overall_score INTEGER DEFAULT 0")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE users ADD COLUMN username TEXT")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE users ADD COLUMN email TEXT")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
        except Exception:
            pass
    logger.info("Database initialized at %s", DB_PATH)


# ── Password helpers ─────────────────────────────────────────────────────────

def _hash_password(password: str, salt: Optional[str] = None) -> str:
    if salt is None:
        salt = secrets.token_hex(16)
    h = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return f"{salt}:{h}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt, _ = stored.split(":", 1)
        return _hash_password(password, salt) == stored
    except Exception:
        return False


# ── User CRUD ────────────────────────────────────────────────────────────────

def register_user(name: str, username: str, email: str, password: str) -> Optional[int]:
    """Register a new user. Returns user_id or None if username/email taken."""
    with _connect() as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE username=? OR email=?", (username, email)
        ).fetchone()
        if existing:
            return None
        cur = conn.execute(
            "INSERT INTO users (name, username, email, password_hash, created_at) VALUES (?,?,?,?,?)",
            (name, username, email, _hash_password(password), _now()),
        )
        return cur.lastrowid


def login_user(username: str, password: str) -> Optional[dict]:
    """Verify credentials. Returns user dict or None."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username=?", (username,)
        ).fetchone()
    if not row:
        return None
    user = dict(row)
    if not _verify_password(password, user.get("password_hash", "")):
        return None
    return user


def get_user(user_id: int) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return dict(row) if row else None


def create_user(name: str) -> int:
    """Legacy: create user by name only (no auth). Used by old code paths."""
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO users (name, username, email, password_hash, created_at) VALUES (?,?,?,?,?)",
            (name, f"user_{secrets.token_hex(4)}", f"{secrets.token_hex(4)}@legacy.local",
             _hash_password("legacy"), _now()),
        )
        return cur.lastrowid


# ── Session CRUD ─────────────────────────────────────────────────────────────

def start_session(sentence: str, user_id: Optional[int] = None) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO sessions (user_id, sentence, started_at) VALUES (?,?,?)",
            (user_id, sentence, _now()),
        )
        return cur.lastrowid


def end_session(session_id: int, summary: str = "", overall_score: int = 0) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE sessions SET ended_at=?, summary=?, overall_score=? WHERE id=?",
            (_now(), summary, overall_score, session_id),
        )


def save_word_result(
    session_id: int,
    word: str,
    attempts: int,
    passed: bool,
    best_accuracy: float,
    errors: List[dict],
) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO word_results (session_id, word, attempts, passed, best_accuracy) VALUES (?,?,?,?,?)",
            (session_id, word, attempts, int(passed), round(best_accuracy, 4)),
        )
        wr_id = cur.lastrowid
        for e in errors:
            conn.execute(
                "INSERT INTO phoneme_errors (word_result_id, expected_phoneme, detected_phoneme, error_type, severity, confidence) VALUES (?,?,?,?,?,?)",
                (wr_id, e.get("expected_phoneme"), e.get("detected_phoneme"),
                 e.get("error_type"), e.get("severity"), e.get("confidence")),
            )
    return wr_id


def get_session_results(session_id: int) -> List[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM word_results WHERE session_id=?", (session_id,)
        ).fetchall()
    return [dict(r) for r in rows]


# ── Profile / history queries ────────────────────────────────────────────────

def get_user_sessions(user_id: int, limit: int = 50) -> List[dict]:
    """Return recent sessions for a user, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            """SELECT id, sentence, started_at, ended_at, overall_score, summary
               FROM sessions WHERE user_id=? AND ended_at IS NOT NULL
               ORDER BY started_at DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_session_detail(session_id: int) -> List[dict]:
    """Return word results with phoneme errors for one session."""
    with _connect() as conn:
        words = conn.execute(
            "SELECT * FROM word_results WHERE session_id=?", (session_id,)
        ).fetchall()
        detail = []
        for w in words:
            wd = dict(w)
            errs = conn.execute(
                "SELECT * FROM phoneme_errors WHERE word_result_id=?", (w["id"],)
            ).fetchall()
            wd["errors"] = [dict(e) for e in errs]
            detail.append(wd)
    return detail


def get_user_stats(user_id: int) -> dict:
    """Aggregate stats for the profile page."""
    with _connect() as conn:
        sessions = conn.execute(
            "SELECT overall_score, started_at FROM sessions WHERE user_id=? AND ended_at IS NOT NULL ORDER BY started_at",
            (user_id,),
        ).fetchall()
        total_words = conn.execute(
            "SELECT COUNT(*) as c FROM word_results wr JOIN sessions s ON wr.session_id=s.id WHERE s.user_id=?",
            (user_id,),
        ).fetchone()
        passed_words = conn.execute(
            "SELECT COUNT(*) as c FROM word_results wr JOIN sessions s ON wr.session_id=s.id WHERE s.user_id=? AND wr.passed=1",
            (user_id,),
        ).fetchone()
        errors = conn.execute(
            """SELECT pe.error_type, COUNT(*) as cnt
               FROM phoneme_errors pe
               JOIN word_results wr ON pe.word_result_id=wr.id
               JOIN sessions s ON wr.session_id=s.id
               WHERE s.user_id=?
               GROUP BY pe.error_type""",
            (user_id,),
        ).fetchall()

    scores = [dict(r) for r in sessions]
    return {
        "sessions": scores,
        "total_sessions": len(scores),
        "avg_score": round(sum(r["overall_score"] for r in scores) / max(len(scores), 1)),
        "best_score": max((r["overall_score"] for r in scores), default=0),
        "total_words": (total_words["c"] if total_words else 0),
        "passed_words": (passed_words["c"] if passed_words else 0),
        "error_types": {r["error_type"]: r["cnt"] for r in errors},
    }


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"