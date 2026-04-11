"""Video processing pipeline for SightLion."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Dict, List

import cv2

from scorer import (
    WEIGHTS,
    compute_score,
    derive_clinical_assessments,
    generate_explanation,
    get_priority,
)
from signals import MIN_VALID_FRAMES, extract_signals
from utils import create_placeholder_thumbnail, resize_thumbnail

DEMO_PATIENT_PROFILES = [
    {
        "patient_id": "Patient A",
        "score": 78,
        "signals": {
            "slumped_posture": 0.20,
            "body_sway": 0.35,
            "tripod_position": 0.08,
            "arm_drift": 0.62,
            "hands_near_throat": 0.10,
            "facial_asymmetry": 0.70,
            "low_alertness": 0.15,
        },
    },
    {
        "patient_id": "Patient B",
        "score": 52,
        "signals": {
            "slumped_posture": 0.65,
            "body_sway": 0.55,
            "tripod_position": 0.10,
            "arm_drift": 0.20,
            "hands_near_throat": 0.05,
            "facial_asymmetry": 0.10,
            "low_alertness": 0.40,
        },
    },
    {
        "patient_id": "Patient C",
        "score": 61,
        "signals": {
            "slumped_posture": 0.15,
            "body_sway": 0.12,
            "tripod_position": 0.45,
            "arm_drift": 0.10,
            "hands_near_throat": 0.72,
            "facial_asymmetry": 0.08,
            "low_alertness": 0.65,
        },
    },
    {
        "patient_id": "Patient D",
        "score": 18,
        "signals": {
            "slumped_posture": 0.08,
            "body_sway": 0.10,
            "tripod_position": 0.04,
            "arm_drift": 0.06,
            "hands_near_throat": 0.02,
            "facial_asymmetry": 0.05,
            "low_alertness": 0.07,
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
        final_warning = "Insufficient data \u2014 manual review recommended"

    if thumbnail is None:
        thumbnail = create_placeholder_thumbnail(patient_id, priority["hex"])

    return {
        "patient_id": patient_id,
        "score": score,
        "color": priority["color"],
        "label": priority["label"],
        "hex": priority["hex"],
        "explanation": generate_explanation(visible_signals, WEIGHTS),
        "assessments": derive_clinical_assessments(visible_signals),
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
