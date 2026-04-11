"""Video processing pipeline for TriageVision."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Dict, List

import cv2

from scorer import WEIGHTS, compute_score, generate_explanation, get_priority
from signals import MIN_VALID_FRAMES, extract_signals
from utils import create_placeholder_thumbnail, resize_thumbnail

DEMO_PATIENT_PROFILES = [
    {
        "patient_id": "Patient A",
        "score": 82,
        "signals": {
            "slumped_posture": 0.34,
            "body_sway": 0.83,
            "tripod_position": 0.12,
            "arm_drift": 0.18,
            "hands_near_throat": 0.91,
            "facial_asymmetry": 0.12,
            "low_alertness": 0.22,
        },
    },
    {
        "patient_id": "Patient B",
        "score": 58,
        "signals": {
            "slumped_posture": 0.77,
            "body_sway": 0.41,
            "tripod_position": 0.18,
            "arm_drift": 0.74,
            "hands_near_throat": 0.08,
            "facial_asymmetry": 0.14,
            "low_alertness": 0.19,
        },
    },
    {
        "patient_id": "Patient C",
        "score": 71,
        "signals": {
            "slumped_posture": 0.28,
            "body_sway": 0.22,
            "tripod_position": 0.09,
            "arm_drift": 0.21,
            "hands_near_throat": 0.05,
            "facial_asymmetry": 0.78,
            "low_alertness": 0.87,
        },
    },
    {
        "patient_id": "Patient D",
        "score": 22,
        "signals": {
            "slumped_posture": 0.09,
            "body_sway": 0.11,
            "tripod_position": 0.04,
            "arm_drift": 0.07,
            "hands_near_throat": 0.03,
            "facial_asymmetry": 0.06,
            "low_alertness": 0.08,
        },
    },
]


def _display_signals(signals: dict) -> dict:
    return {key: float(value) for key, value in signals.items() if not key.startswith("_")}


def build_patient_record(
    patient_id: str,
    signals: dict,
    thumbnail=None,
    *,
    score_override: int | None = None,
    warning: str | None = None,
) -> Dict[str, object]:
    """Build a dashboard-ready patient record from signals and a thumbnail."""
    visible_signals = _display_signals(signals)
    score = score_override if score_override is not None else compute_score(visible_signals)
    priority = get_priority(score)
    valid_frames = int(signals.get("_valid_frames", MIN_VALID_FRAMES))
    final_warning = warning
    if valid_frames < MIN_VALID_FRAMES and final_warning is None:
        final_warning = "Insufficient data — manual review recommended"

    if thumbnail is None:
        thumbnail = create_placeholder_thumbnail(patient_id, priority["hex"])

    return {
        "patient_id": patient_id,
        "score": score,
        "color": priority["color"],
        "label": priority["label"],
        "hex": priority["hex"],
        "explanation": generate_explanation(visible_signals, WEIGHTS),
        "signals": visible_signals,
        "thumbnail": thumbnail,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "warning": final_warning,
    }


def extract_thumbnail(video_path: str):
    """Return a representative frame from the input video."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return create_placeholder_thumbnail("TV", "#5B6474")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)

    success, frame = capture.read()
    capture.release()

    if not success:
        return create_placeholder_thumbnail("TV", "#5B6474")

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return resize_thumbnail(rgb_frame)


def process_video(video_path: str, patient_id: str) -> Dict[str, object]:
    """Process a patient video and return dashboard-ready metadata."""
    signals = extract_signals(video_path)
    thumbnail = extract_thumbnail(video_path)
    return build_patient_record(patient_id=patient_id, signals=signals, thumbnail=thumbnail)


def build_demo_patients() -> List[Dict[str, object]]:
    """Return pre-scored demo patient records for offline demos."""
    patients = []
    for profile in DEMO_PATIENT_PROFILES:
        priority = get_priority(profile["score"])
        thumbnail = create_placeholder_thumbnail(profile["patient_id"], priority["hex"])
        patients.append(
            build_patient_record(
                patient_id=profile["patient_id"],
                signals=profile["signals"],
                thumbnail=thumbnail,
                score_override=profile["score"],
            )
        )
    return patients


def get_demo_profiles() -> List[Dict[str, object]]:
    """Return a safe copy of the animated demo feed profiles."""
    return deepcopy(DEMO_PATIENT_PROFILES)
