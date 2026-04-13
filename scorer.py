"""Scoring and clinical assessment helpers for SightLion."""

from __future__ import annotations

from typing import Dict

WEIGHTS: Dict[str, float] = {
    "low_alertness": 0.30,
    "hands_near_throat": 0.20,
    "slumped_posture": 0.15,
    "arm_drift": 0.10,
    "body_sway": 0.12,
    "tripod_position": 0.08,
    "facial_asymmetry": 0.05,
}

CLINICAL_HIGH = 0.50
CLINICAL_MODERATE = 0.25
CLINICAL_FINDING = 0.30

_PERSIST_REQUIRED = 0.30
_STRONG_SINGLE = 0.70

_FALL_SWAY_FINDING = 0.40
_FALL_SWAY_HIGH = 0.55
_FALL_POSTURE_HIGH = 0.50
_FALL_PERSIST_HIGH = 0.50


def _persist(signals: dict, key: str) -> float:
    """Persistence estimate for a signal (fraction of time above noise).

    The accumulated average already acts as a persistence proxy: a high
    running mean means the signal has been consistently elevated.
    """
    return float(signals.get(key, 0.0))

STROKE_HIGH = 0.75
STROKE_MODERATE = 0.50
STROKE_FINDING = 0.55


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


# ---------------------------------------------------------------------------
# Clinical assessment derivation
# ---------------------------------------------------------------------------

def _level_color(level: str) -> str:
    if level in ("HIGH", "YES", "Unresponsive"):
        return "#DC2626"
    if level in ("MODERATE", "Drowsy"):
        return "#D97706"
    return "#16A34A"


def derive_stroke_risk(signals: dict) -> dict:
    asym = float(signals.get("facial_asymmetry", 0))
    drift = float(signals.get("arm_drift", 0))
    peak = max(asym, drift)
    findings: list[str] = []
    if asym > STROKE_FINDING:
        findings.append("Facial droop detected")
    if drift > STROKE_FINDING:
        findings.append("Arm weakness detected")
    if peak > STROKE_HIGH:
        level = "HIGH"
    elif peak > STROKE_MODERATE:
        level = "MODERATE"
    else:
        level = "LOW"
    return {"level": level, "color": _level_color(level), "findings": findings}


def derive_fall_risk(signals: dict) -> dict:
    """Fall risk requires sustained, strong sway — ideally with postural collapse.

    Sway alone can reach at most MODERATE.  HIGH requires either
    (a) sway + posture both confirmed, or (b) severe postural collapse alone.
    """
    sway = float(signals.get("body_sway", 0))
    posture = float(signals.get("slumped_posture", 0))

    sway_persist = _persist(signals, "body_sway")
    posture_persist = _persist(signals, "slumped_posture")

    sway_strong = sway > _FALL_SWAY_HIGH and sway_persist >= _FALL_PERSIST_HIGH
    posture_strong = posture > _FALL_POSTURE_HIGH and posture_persist >= _FALL_PERSIST_HIGH

    findings: list[str] = []
    if sway > _FALL_SWAY_FINDING and sway_persist >= _PERSIST_REQUIRED:
        findings.append("Sustained balance instability")
    elif sway > _FALL_SWAY_FINDING:
        findings.append("Transient sway detected")
    if posture > CLINICAL_FINDING and posture_persist >= _PERSIST_REQUIRED:
        findings.append("Postural collapse")

    if (sway_strong and posture_strong) or (posture_strong and posture > _STRONG_SINGLE):
        level = "HIGH"
    elif findings:
        level = "MODERATE"
    else:
        level = "LOW"
        findings = []

    return {"level": level, "color": _level_color(level), "findings": findings}


def derive_respiratory(signals: dict) -> dict:
    tripod = float(signals.get("tripod_position", 0))
    throat = float(signals.get("hands_near_throat", 0))
    findings: list[str] = []
    if tripod > CLINICAL_FINDING:
        findings.append("Labored breathing posture")
    if throat > CLINICAL_FINDING:
        findings.append("Airway distress gestures")
    level = "YES" if findings else "NO"
    return {"level": level, "color": _level_color(level), "findings": findings}


def derive_mental_status(signals: dict) -> dict:
    alertness = float(signals.get("low_alertness", 0))
    if alertness > 0.60:
        return {"level": "Unresponsive", "color": _level_color("Unresponsive"),
                "findings": ["No eye opening"]}
    if alertness > 0.30:
        return {"level": "Drowsy", "color": _level_color("Drowsy"),
                "findings": ["Reduced eye opening"]}
    return {"level": "Alert", "color": _level_color("Alert"), "findings": []}


def derive_clinical_assessments(signals: dict) -> dict:
    """Derive the 4 clinical assessments from raw signal values."""
    return {
        "stroke_risk": derive_stroke_risk(signals),
        "fall_risk": derive_fall_risk(signals),
        "respiratory": derive_respiratory(signals),
        "mental_status": derive_mental_status(signals),
    }


ASSESSMENT_LABELS = {
    "stroke_risk": "Stroke Risk",
    "fall_risk": "Fall Risk",
    "respiratory": "Respiratory",
    "mental_status": "Mental Status",
}


def generate_explanation(signals: dict, weights: dict | None = None) -> str:
    """Clinical-language explanation from the derived assessments."""
    assessments = derive_clinical_assessments(signals)
    parts: list[str] = []
    for key in ("stroke_risk", "fall_risk", "respiratory", "mental_status"):
        a = assessments[key]
        if a["level"] in ("LOW", "NO", "Alert"):
            continue
        label = ASSESSMENT_LABELS[key]
        line = f"{label}: {a['level']}"
        if a["findings"]:
            line += f" \u2014 {', '.join(a['findings'])}"
        parts.append(line)
    if not parts:
        return "No clinical concerns identified."
    return " \u00b7 ".join(parts)
