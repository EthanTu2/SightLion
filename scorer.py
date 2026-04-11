"""Scoring helpers for SightLion."""

from __future__ import annotations

from typing import Dict

WEIGHTS: Dict[str, float] = {
    "hands_near_throat": 0.25,
    "low_alertness": 0.20,
    "arm_drift": 0.18,
    "slumped_posture": 0.15,
    "body_sway": 0.12,
    "tripod_position": 0.06,
    "facial_asymmetry": 0.04,
}


def _percent(value: float) -> str:
    return f"{round(value * 100)}%"


def _describe_signal(signal_name: str, value: float) -> str:
    descriptions = {
        "hands_near_throat": f"hands near throat ({_percent(value)} of frames)",
        "low_alertness": f"low alertness (eyes closed in {_percent(value)} of frames)",
        "arm_drift": f"arm drift detected ({_percent(value)} of frames)",
        "slumped_posture": f"slumped posture observed ({_percent(value)} of frames)",
        "body_sway": f"body sway / instability elevated ({_percent(value)} severity)",
        "tripod_position": f"tripod position observed ({_percent(value)} of frames)",
        "facial_asymmetry": f"facial asymmetry noted ({_percent(value)} severity)",
    }
    return descriptions.get(signal_name, signal_name.replace("_", " "))


def compute_score(signals: dict) -> int:
    """Convert signal values into a weighted 0-100 severity score."""
    raw = sum(WEIGHTS[key] * float(signals.get(key, 0.0)) for key in WEIGHTS)
    return max(0, min(100, round(raw * 100)))


def get_priority(score: int) -> dict:
    """Map a numeric score onto a triage color and label."""
    if score >= 65:
        return {"color": "red", "label": "Critical", "hex": "#DC2626"}
    if score >= 35:
        return {"color": "yellow", "label": "Urgent", "hex": "#D97706"}
    return {"color": "green", "label": "Stable", "hex": "#16A34A"}


def generate_explanation(signals: dict, weights: dict | None = None) -> str:
    """Describe the top weighted signals in human-readable form."""
    selected_weights = weights or WEIGHTS
    contributions = []
    for signal_name, weight in selected_weights.items():
        signal_value = float(signals.get(signal_name, 0.0))
        contribution = weight * signal_value
        if contribution > 0:
            contributions.append((signal_name, signal_value, contribution))

    if not contributions:
        return "No visible distress signals flagged."

    top_signals = sorted(contributions, key=lambda item: item[2], reverse=True)[:2]
    parts = [_describe_signal(signal_name, signal_value) for signal_name, signal_value, _ in top_signals]
    return "Flagged for: " + ", ".join(parts)
