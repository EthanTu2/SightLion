# SightLion

### 2nd Place — Claremont Accelerator Hackathon 2026

AI-assisted ER intake monitoring that uses existing waiting room cameras to continuously track patients, detect clinical distress signals in real time, and surface a priority queue for staff review — so the sickest patients are seen first.

<!-- TODO: Add LinkedIn post link here -->

> **Disclaimer:** SightLion is for staff review only. It is **not** a diagnostic tool.

## How It Works

SightLion's pipeline runs entirely on-device with no GPU required:

1. **Landmark detection** — MediaPipe extracts 468 facial landmarks and 33 body landmarks per person, tracking up to 5 people simultaneously.
2. **Signal extraction** — Six clinical signals are computed from the landmarks each frame: slumped posture, body sway, tripod positioning, arm drift, hands-near-throat, facial asymmetry, and low alertness (eye aspect ratio).
3. **Clinical assessments** — Signals map to four nurse-recognizable categories: Stroke Risk (FAST screen), Fall Risk, Respiratory Distress, and Mental Status.
4. **Severity scoring** — A weighted 0–100 score determines triage priority (Critical ≥65, Urgent 35–64, Stable <35). If anyone hits critical for 5+ seconds, the system auto-alerts.
5. **Face re-identification** — Inspired by Apple Face ID, we use 21 geometric facial-proportion ratios (scale/lighting-invariant) with multi-template matching to recognize returning patients and prevent duplicate records.

## Tech Stack

- **Python 3.12+**
- **Streamlit** — real-time dashboard UI
- **MediaPipe** — pose estimation + face mesh landmark detection
- **OpenCV** — video capture, frame processing, annotation
- **NumPy** — signal computation and accumulation

## Project Structure

```
SightLion/
├── app.py          # Streamlit dashboard, capture loops, patient queue
├── signals.py      # MediaPipe detection, signal extraction, multi-person tracker
├── scorer.py       # Severity scoring and clinical assessment derivation
├── processor.py    # Patient record builder, demo profiles
├── utils.py        # Frame annotation, face crop, geometric embedding
├── logo.png        # SightLion logo
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-org>/SightLion.git
cd SightLion

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The dashboard opens in your browser. Grant webcam access when prompted to use Live Webcam mode, or switch to Video Upload mode to analyze `.mp4` files.

### Modes

| Mode | Description |
|---|---|
| **Live Webcam** | Real-time monitoring with landmark overlay and clinical HUD |
| **Continuous Monitoring** | Indefinite webcam feed with auto-save on sustained critical condition |
| **Video Upload** | Analyze pre-recorded `.mp4` files |
| **Demo** | Animated synthetic patients to explore the UI without a camera |

## Key Features

- Multi-person real-time tracking and per-person triage scoring
- Four clinical assessments: Stroke Risk, Fall Risk, Respiratory Distress, Mental Status
- Face ID–inspired re-identification that recognizes returning patients
- Stillness-based photo capture for clean patient thumbnails
- Priority queue sorted by severity with face photos, scores, and clinical findings
- Critical alert system with auto-save after 5 seconds of sustained critical score
- Body-relative sway normalization (scale-invariant, dead zone kills camera jitter)
- Persistence-gated fall risk (sway alone caps at Moderate)
