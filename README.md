# Pronunciation Tutor 🗣️

A phoneme-level, interactive English pronunciation correction system built as a final-year AI project.

---

## What It Does

The system behaves like a human tutor:

1. Records your spoken sentence.
2. Transcribes and spell-corrects it.
3. You confirm the sentence.
4. It aligns phonemes with your audio using Montreal Forced Aligner.
5. For each word it compares **expected** phonemes (CMUdict) with **detected** ones.
6. It classifies errors as **substitution / deletion / insertion** and scores severity.
7. A Gemini-powered tutor explains exactly how to fix the sound.
8. You retry the word; the loop continues until you pass or run out of attempts.
9. A session summary highlights your strengths and practice areas.

---

## Architecture

```
pronunciation_tutor/
│
├── app/                        ← Presentation + Application layers
│   ├── streamlit_app.py        ← Streamlit UI  (Presentation)
│   └── controller.py           ← State-machine controller (Application)
│
├── domain/                     ← Core logic (no I/O, fully testable)
│   ├── phoneme_alignment.py    ← WordAlignment / PhonemeToken data classes
│   ├── error_detection.py      ← Substitution / deletion / insertion detection
│   ├── severity_scoring.py     ← Minor / Moderate / Severe scoring
│   └── learning_logic.py       ← Accuracy, retry thresholds, pass/fail
│
├── services/                   ← Infrastructure integrations
│   ├── asr_service.py          ← Whisper speech-to-text
│   ├── mfa_service.py          ← MFA forced alignment + CMUdict lookup
│   ├── llm_service.py          ← Gemini: transcript correction + explanations
│   └── tts_service.py          ← pyttsx3 text-to-speech
│
├── infrastructure/
│   ├── database.py             ← SQLite persistence
│   └── audio_processing.py     ← Recording, loading, normalising audio
│
├── assets/
│   └── articulation_images/    ← Optional: PNG per ARPAbet phoneme (e.g. TH.png)
│
├── data/                       ← SQLite DB written here at runtime
├── requirements.txt
├── .env.example
└── README.md
```

### Layer Responsibilities

| Layer | Responsibility |
|---|---|
| **Presentation** | Streamlit UI, audio capture, display |
| **Application** | Session state machine, retry logic |
| **Domain** | Pure phoneme comparison, error typing, scoring |
| **Services / Infrastructure** | ASR, MFA, Gemini, SQLite, TTS |

---

## Quick Start

### 1. Clone and set up the environment

```bash
cd pronunciation_tutor
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Install Montreal Forced Aligner (via conda)

MFA requires conda. If you don't have conda, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) first.

```bash
conda create -n aligner python=3.10 -y
conda activate aligner
conda install -c conda-forge montreal-forced-aligner -y

mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
```

> **Tip:** You can run the app without MFA installed. The system automatically falls back to CMUdict-only expected phonemes (no time alignment, but error detection still works).

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY
```

### 4. Download NLTK data (first run only)

```python
import nltk
nltk.download("cmudict")
```

### 5. Run the app

```bash
streamlit run app/streamlit_app.py
```

---

## Session States

```
LISTEN_SENTENCE   → Record full sentence
CONFIRM_SENTENCE  → Confirm/edit corrected transcript
PROCESS_WORD      → Analyse current word (automatic)
EXPLAIN_ERROR     → Show error table + tutor message
RETRY_WORD        → Record word again
NEXT_WORD         → (transient) Advance word pointer
SESSION_SUMMARY   → Show overall results + coach feedback
```

---

## Error Object Format

```json
{
  "word": "think",
  "expected_phoneme": "TH",
  "detected_phoneme": "T",
  "error_type": "substitution",
  "severity": "moderate",
  "confidence": 0.62
}
```

---

## Adding Articulation Images

Place PNG files named after ARPAbet phonemes in `assets/articulation_images/`:

```
assets/articulation_images/
    TH.png
    R.png
    L.png
    ...
```

The UI will automatically display the relevant image when an error is detected.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Google Gemini API key |
| `WHISPER_MODEL` | `base` | Whisper model size |
| `MFA_ACOUSTIC_MODEL` | `english_us_arpa` | MFA acoustic model name |
| `MFA_DICTIONARY` | `english_us_arpa` | MFA dictionary name |
| `DB_PATH` | `data/pronunciation_tutor.db` | SQLite database path |

---

## Extending the System

| Goal | Where to change |
|---|---|
| Swap ASR engine | `services/asr_service.py` |
| Swap LLM provider | `services/llm_service.py` |
| Change pass threshold | `domain/learning_logic.py` → `PASS_THRESHOLD` |
| Add new error types | `domain/error_detection.py` |
| Change DB engine | `infrastructure/database.py` |
| Add new UI screens | `app/streamlit_app.py` |

---

## License

MIT – Free to use and modify for academic and personal projects.
