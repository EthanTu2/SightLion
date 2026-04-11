# SightLion

SightLion is an AI-assisted ER intake monitoring MVP that reviews short patient videos, extracts visible distress signals with rules-based computer vision, and surfaces a color-coded priority queue for staff review.

## Important disclaimer

SightLion is **not** a diagnostic tool. It only flags visible distress signals from video to help nurses and doctors triage patients faster.

## Tech stack

- Python 3.10+
- OpenCV (`cv2`)
- MediaPipe Face Mesh + Pose
- NumPy
- Streamlit

## Project structure

```text
triagevision/
├── app.py
├── processor.py
├── signals.py
├── scorer.py
├── utils.py
├── sample_videos/
└── README.md
```

## Install

```bash
pip install opencv-python mediapipe numpy streamlit
```

## Run

```bash
streamlit run app.py
```

## Features

- Multi-file `.mp4` intake upload flow
- Frame-by-frame signal extraction with MediaPipe
- Weighted scoring and red/yellow/green queue assignment
- Real-time priority dashboard sorted by severity
- Expandable per-patient signal breakdown
- Demo mode with sample patients and placeholder thumbnails
- Manual review warning when too few frames have valid landmarks
