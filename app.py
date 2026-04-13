"""Streamlit dashboard for SightLion."""

from __future__ import annotations

import base64
import html
import math
import os
import pathlib
import tempfile
import time

import cv2
import streamlit as st

from processor import build_demo_patients, build_patient_record, get_demo_profiles
from scorer import (
    ASSESSMENT_LABELS,
    compute_score,
    derive_clinical_assessments,
    get_priority,
)
from signals import (
    FRAME_SAMPLE_RATE,
    SIGNAL_KEYS,
    MultiPersonTracker,
    create_face_mesh_estimator,
    create_pose_estimator,
    process_frame_multi,
    zero_signals,
)
from utils import (
    add_template,
    annotate_frame,
    best_template_similarity,
    create_demo_frame,
    crop_face_thumbnail,
    face_similarity,
    image_to_base64,
    resize_thumbnail,
)

PREVIEW_SECONDS = 3.0
DEFAULT_CAPTURE_SECONDS = 10
PLAYBACK_SPEEDS = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0}
CRITICAL_SCORE_THRESHOLD = 65
CRITICAL_ALERT_SECONDS = 5.0

_ASSESS_ICONS = {
    "stroke_risk": "\U0001f9e0",
    "fall_risk": "\U0001f6b6",
    "respiratory": "\U0001fab7",
    "mental_status": "\U0001f441",
}

_ASSESS_TIPS = {
    "stroke_risk": "FAST screen: facial droop and arm weakness indicate stroke risk.",
    "fall_risk": "Balance instability and postural collapse indicate fall risk.",
    "respiratory": "Tripod posture and airway distress gestures suggest respiratory concern.",
    "mental_status": "Eye-opening and alertness indicate mental status level.",
}


def _level_css_color(level: str) -> str:
    if level in ("HIGH", "YES", "Unresponsive"):
        return "#DC2626"
    if level in ("MODERATE", "Drowsy"):
        return "#D97706"
    return "#16A34A"


_FACE_MATCH_THRESHOLD = 0.65


def _find_matching_patient(
    templates: list, patients: list[dict]
) -> int | None:
    """Return queue index of the patient whose face best matches *templates*.

    Apple Face ID–style: compare every query template against every stored
    template and take the maximum.
    """
    if not templates:
        return None
    best_sim, best_idx = 0.0, None
    for i, p in enumerate(patients):
        p_templates = p.get("face_embeddings", [])
        for q_emb in templates:
            sim = best_template_similarity(q_emb, p_templates)
            if sim > best_sim:
                best_sim, best_idx = sim, i
    return best_idx if best_sim >= _FACE_MATCH_THRESHOLD else None


def _save_or_merge_patient(record: dict) -> tuple[str, bool]:
    """Append *record* to queue, or merge into an existing match.

    On merge the stored template set grows (adaptive learning, like Face ID
    updating its model after each successful unlock).
    Returns (patient_id, was_merged).
    """
    patients = st.session_state["patients"]
    new_templates = record.get("face_embeddings", [])
    match_idx = _find_matching_patient(new_templates, patients)
    if match_idx is not None:
        existing = patients[match_idx]
        for key in (
            "score", "label", "hex", "color", "assessments", "explanation",
            "signals", "thumbnail", "timestamp", "warning",
        ):
            if key in record:
                existing[key] = record[key]
        old_t = existing.setdefault("face_embeddings", [])
        for emb in new_templates:
            add_template(old_t, emb)
        return existing["patient_id"], True
    patients.append(record)
    return record["patient_id"], False


def _relabel_from_queue(
    person_states: list[dict], face_emb_map: dict[int, list]
) -> None:
    """Override live track labels with queue patient IDs when faces match.

    Uses greedy 1-to-1 assignment so two tracks never claim the same patient.
    Compares each track's template *set* against each patient's template set.
    """
    patients = st.session_state.get("patients", [])
    if not patients:
        return

    pairs: list[tuple[float, int, int]] = []
    for ps in person_states:
        track_templates = face_emb_map.get(ps["person_id"], [])
        if not track_templates:
            continue
        for qi, p in enumerate(patients):
            p_templates = p.get("face_embeddings", [])
            if not p_templates:
                continue
            sim = max(
                best_template_similarity(t, p_templates)
                for t in track_templates
            )
            if sim >= _FACE_MATCH_THRESHOLD:
                pairs.append((sim, ps["person_id"], qi))

    pairs.sort(reverse=True)
    used_tracks: set[int] = set()
    used_patients: set[int] = set()
    for sim, track_pid, qi in pairs:
        if track_pid in used_tracks or qi in used_patients:
            continue
        for ps in person_states:
            if ps["person_id"] == track_pid:
                ps["queue_patient_id"] = patients[qi]["patient_id"]
                ps["label"] = patients[qi]["patient_id"]
                break
        used_tracks.add(track_pid)
        used_patients.add(qi)


def _enrich_person_states(raw_tracks: list[dict]) -> list[dict]:
    """Add score, assessments, hex, and label to raw tracker output."""
    enriched = []
    for t in raw_tracks:
        signals = t["signals"]
        score = compute_score(signals)
        priority = get_priority(score)
        assessments = derive_clinical_assessments(signals)
        enriched.append({
            **t,
            "label": f"P{t['person_id']:02d}",
            "score": score,
            "hex": priority["hex"],
            "priority_label": priority["label"],
            "assessments": assessments,
        })
    return enriched


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --slate-50: #F8FAFC; --slate-100: #F1F5F9;
                --slate-200: #E2E8F0; --slate-300: #CBD5E1;
                --slate-400: #94A3B8; --slate-500: #64748B;
                --slate-600: #475569; --slate-700: #334155;
                --slate-800: #1E293B; --slate-900: #0F172A;
                --red: #DC2626; --amber: #D97706; --green: #16A34A;
            }
            .stApp { background: var(--slate-100); }

            section[data-testid="stSidebar"] > div:first-child {
                padding-top: 1rem;
            }
            section[data-testid="stSidebar"] .stRadio > div { gap: 0.3rem; }
            section[data-testid="stSidebar"] .stSlider { margin-bottom: -0.3rem; }

            /* ── Header ─────────────────────────────────── */
            .tv-header {
                background: linear-gradient(135deg, var(--slate-900) 0%, var(--slate-800) 100%);
                color: #fff;
                padding: 18px 26px;
                border-radius: 12px;
                margin-bottom: 18px;
                display: flex; align-items: center; gap: 16px;
                box-shadow: 0 2px 8px rgba(15,23,42,0.12);
            }
            .tv-logo {
                height: 40px; object-fit: contain; flex-shrink: 0;
            }
            .tv-header-sub {
                font-size: 12.5px; color: var(--slate-400); margin-top: 2px;
            }

            /* ── Disclaimer ─────────────────────────────── */
            .tv-disclaimer {
                background: #FFFBEB; border: 1px solid #FDE68A; color: #92400E;
                border-radius: 8px; padding: 7px 14px;
                font-size: 12px; font-weight: 600; margin-bottom: 12px;
            }

            /* ── Section labels ─────────────────────────── */
            .tv-section-label {
                font-size: 11px; font-weight: 700; color: var(--slate-500);
                text-transform: uppercase; letter-spacing: 1px;
                margin-bottom: 8px;
            }

            /* ── Video frame wrapper ────────────────────── */
            .tv-frame-wrap {
                background: var(--slate-900);
                border-radius: 10px;
                padding: 4px;
                box-shadow: 0 2px 12px rgba(15,23,42,0.15);
                margin-bottom: 12px;
            }
            .tv-frame-wrap img {
                border-radius: 7px; display: block; width: 100%;
            }

            /* ── Clinical assessment tiles ────────────────── */
            .tv-assess-row {
                display: flex; gap: 8px;
                margin-bottom: 14px;
            }
            .tv-assess-cell {
                flex: 1; text-align: center;
                background: #fff;
                border: 1px solid var(--slate-200);
                border-radius: 10px;
                padding: 14px 6px 12px;
                transition: box-shadow 0.15s;
                position: relative;
            }
            .tv-assess-cell:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.08); z-index: 10; }
            .tv-assess-tip {
                visibility: hidden; opacity: 0;
                position: absolute; bottom: calc(100% + 8px); left: 50%;
                transform: translateX(-50%);
                background: var(--slate-900); color: #fff;
                padding: 8px 12px; border-radius: 6px;
                font-size: 11px; font-weight: 400; line-height: 1.45;
                width: 195px; text-align: left;
                text-transform: none; letter-spacing: normal;
                pointer-events: none;
                transition: opacity 0.15s, visibility 0.15s;
                box-shadow: 0 4px 12px rgba(0,0,0,0.18);
            }
            .tv-assess-tip::after {
                content: ""; position: absolute; top: 100%; left: 50%;
                transform: translateX(-50%);
                border: 6px solid transparent; border-top-color: var(--slate-900);
            }
            .tv-assess-cell:hover .tv-assess-tip { visibility: visible; opacity: 1; }
            .tv-assess-icon { font-size: 20px; margin-bottom: 4px; }
            .tv-assess-title {
                font-size: 9px; font-weight: 700; color: var(--slate-500);
                text-transform: uppercase; letter-spacing: 0.5px;
                margin-bottom: 6px;
            }
            .tv-assess-level {
                font-size: 16px; font-weight: 800;
                margin-bottom: 4px;
            }
            .tv-assess-findings {
                font-size: 10px; color: var(--slate-400);
                line-height: 1.4;
                min-height: 14px;
            }
            /* ── Score tile (wider) ──────────────────────── */
            .tv-score-tile {
                flex: 0.8; text-align: center;
                background: #fff;
                border: 1px solid var(--slate-200);
                border-radius: 10px;
                padding: 14px 10px 12px;
                display: flex; flex-direction: column;
                align-items: center; justify-content: center;
            }
            .tv-score-tile-label {
                font-size: 9px; font-weight: 700; color: var(--slate-500);
                text-transform: uppercase; letter-spacing: 0.5px;
                margin-bottom: 4px;
            }
            .tv-score-tile-value {
                font-size: 28px; font-weight: 900; line-height: 1;
            }
            .tv-score-tile-sub {
                font-size: 11px; font-weight: 600; margin-top: 2px;
            }
            .tv-score-tile-track {
                font-size: 9px; font-weight: 600; color: var(--slate-400);
                margin-top: 4px;
            }

            /* ── Queue header ───────────────────────────── */
            .tv-queue-header {
                display: flex; justify-content: space-between; align-items: center;
                margin-bottom: 14px; padding-bottom: 12px;
                border-bottom: 2px solid var(--slate-200);
            }
            .tv-queue-title { font-size: 17px; font-weight: 800; color: var(--slate-900); }
            .tv-count-badge {
                background: var(--slate-900); color: #fff;
                padding: 4px 12px; border-radius: 999px;
                font-size: 11px; font-weight: 700;
            }

            /* ── Patient card ───────────────────────────── */
            .tv-card {
                background: #fff;
                border: 1px solid var(--slate-200);
                border-radius: 10px; overflow: hidden;
                margin-bottom: 12px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                transition: box-shadow 0.15s;
            }
            .tv-card:hover { box-shadow: 0 3px 10px rgba(0,0,0,0.08); }
            .tv-card-inner { display: flex; align-items: stretch; }
            .tv-card-bar { width: 5px; flex-shrink: 0; border-radius: 10px 0 0 10px; }
            .tv-card-body { padding: 16px 18px; width: 100%; }
            .tv-card-head { display: flex; align-items: center; gap: 14px; }
            .tv-card-face {
                width: 64px; height: 64px;
                border-radius: 50%;
                object-fit: cover;
                flex-shrink: 0;
            }
            .tv-card-info { flex: 1; min-width: 0; }
            .tv-card-row1 {
                display: flex; justify-content: space-between; align-items: center;
            }
            .tv-card-pid { font-size: 14px; font-weight: 700; color: var(--slate-900); }
            .tv-card-pill {
                padding: 3px 10px; border-radius: 999px;
                font-size: 11px; font-weight: 700; white-space: nowrap;
            }
            .tv-card-row2 {
                display: flex; align-items: baseline; gap: 8px; margin-top: 8px;
            }
            .tv-card-score { font-size: 28px; font-weight: 800; line-height: 1; }
            .tv-card-score-sub { font-size: 13px; font-weight: 500; color: var(--slate-400); }
            .tv-card-track {
                margin-top: 10px; height: 6px;
                background: var(--slate-200); border-radius: 999px; overflow: hidden;
            }
            .tv-card-fill { height: 100%; border-radius: 999px; }

            /* ── Card clinical assessments ────────────────── */
            .tv-card-assess {
                margin-top: 12px;
                display: flex; flex-direction: column; gap: 4px;
            }
            .tv-card-assess-row {
                display: flex; align-items: baseline; gap: 6px;
                font-size: 12.5px; line-height: 1.5;
            }
            .tv-card-assess-label {
                font-weight: 600; color: var(--slate-600);
                min-width: 90px;
            }
            .tv-card-assess-level { font-weight: 800; }
            .tv-card-assess-findings {
                font-weight: 400; color: var(--slate-400);
                font-size: 11.5px;
            }
            .tv-card-ts { margin-top: 8px; font-size: 11px; color: var(--slate-400); }
            .tv-card-warning {
                margin-top: 8px; padding: 6px 10px; border-radius: 6px;
                background: #FFFBEB; border: 1px solid #FDE68A;
                color: #92400E; font-size: 11px; font-weight: 600;
            }

            /* ── Empty state ────────────────────────────── */
            .tv-empty {
                text-align: center; padding: 48px 24px;
                color: var(--slate-400); font-size: 13.5px;
                background: #fff;
                border: 1.5px dashed var(--slate-300);
                border-radius: 12px; line-height: 1.7;
            }
            .tv-empty-icon { font-size: 40px; margin-bottom: 12px; }
            .tv-empty b { color: var(--slate-600); }

            /* ── Legend ──────────────────────────────────── */
            .tv-legend {
                margin-top: 16px;
                background: #fff; border: 1px solid var(--slate-200);
                border-radius: 8px; padding: 12px 18px;
                color: var(--slate-600); font-size: 13px;
                line-height: 1.8; text-align: center;
            }
            .tv-legend-dot { font-weight: 800; font-size: 14px; }

            /* ── Sidebar build note ─────────────────────── */
            .tv-build-note {
                font-size: 10.5px; color: var(--slate-400);
                line-height: 1.45; margin-top: 8px;
                border-top: 1px solid var(--slate-200);
                padding-top: 10px;
            }

            /* ── Primary button overrides ───────────────── */
            .stButton > button[kind="primary"] {
                border-radius: 8px; font-weight: 700;
                padding: 0.55rem 1.2rem;
                font-size: 14px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _notify(message: str) -> None:
    if hasattr(st, "toast"):
        st.toast(message)
    else:
        st.info(message)


def _webcam_available() -> bool:
    try:
        capture = cv2.VideoCapture(0)
        available = capture.isOpened()
        if available:
            success, _ = capture.read()
            available = available and success
        capture.release()
        return available
    except Exception:
        return False


def _ensure_state() -> None:
    st.session_state.setdefault("patients", [])
    st.session_state.setdefault("next_patient_number", 1)
    st.session_state.setdefault("demo_profiles", get_demo_profiles())
    st.session_state.setdefault("demo_feed_patient", "Patient A")

    if "has_webcam" not in st.session_state:
        st.session_state["has_webcam"] = _webcam_available()

    default_mode = "Live Webcam" if st.session_state["has_webcam"] else "Video Upload"
    st.session_state.setdefault("selected_mode", default_mode)
    st.session_state.setdefault("continuous_mode", False)
    st.session_state.setdefault("monitoring_active", False)
    st.session_state.setdefault("monitoring_data", None)


def _consume_patient_id(override_value: str) -> str:
    patient_number = int(st.session_state["next_patient_number"])
    st.session_state["next_patient_number"] = patient_number + 1
    override = override_value.strip()
    if override:
        return override
    return f"Patient {patient_number:03d}"


# ---------------------------------------------------------------------------
# Clinical assessment tiles (below the video)
# ---------------------------------------------------------------------------

def _update_assess_tiles(slot, person_states: list[dict]) -> None:
    """Render clinical assessment tiles + severity score below the live feed."""
    if person_states:
        worst = max(person_states, key=lambda p: p["score"])
        assessments = worst["assessments"]
        score = worst["score"]
        priority = get_priority(score)
        n_tracked = len(person_states)
    else:
        assessments = derive_clinical_assessments(zero_signals())
        score = 0
        priority = get_priority(0)
        n_tracked = 0

    cells = ""
    for key in ("stroke_risk", "fall_risk", "respiratory", "mental_status"):
        a = assessments[key]
        level = a["level"]
        color = _level_css_color(level)
        icon = _ASSESS_ICONS.get(key, "")
        title = ASSESSMENT_LABELS[key].upper()
        tip = _ASSESS_TIPS.get(key, "")
        findings = " &middot; ".join(html.escape(f) for f in a.get("findings", []))
        if not findings:
            findings = "&mdash;"
        cells += (
            f'<div class="tv-assess-cell">'
            f'<div class="tv-assess-tip">{html.escape(tip)}</div>'
            f'<div class="tv-assess-icon">{icon}</div>'
            f'<div class="tv-assess-title">{title}</div>'
            f'<div class="tv-assess-level" style="color:{color};">{html.escape(level)}</div>'
            f'<div class="tv-assess-findings">{findings}</div>'
            f'</div>'
        )

    score_color = priority["hex"]
    score_label = priority["label"]
    track_note = ""
    if n_tracked > 1:
        track_note = (
            f'<div class="tv-score-tile-track">'
            f'\U0001f465 {n_tracked} people tracked'
            f'</div>'
        )
    cells += (
        f'<div class="tv-score-tile">'
        f'<div class="tv-score-tile-label">SEVERITY</div>'
        f'<div class="tv-score-tile-value" style="color:{score_color};">{score}</div>'
        f'<div class="tv-score-tile-sub" style="color:{score_color};">{html.escape(score_label)}</div>'
        f'{track_note}'
        f'</div>'
    )

    slot.markdown(
        f'<div class="tv-assess-row">{cells}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Patient card (right column)
# ---------------------------------------------------------------------------

def _render_patient_card(patient: dict) -> None:
    b64 = image_to_base64(patient.get("thumbnail"))
    pid = html.escape(str(patient.get("patient_id", "Unknown")))
    label = html.escape(str(patient.get("label", "Stable")))
    ts = html.escape(str(patient.get("timestamp", "")))
    hex_color = patient["hex"]
    score = patient["score"]

    assessments = patient.get("assessments") or derive_clinical_assessments(
        patient.get("signals", {})
    )

    warning = patient.get("warning")
    warn_block = ""
    if warning:
        warn_block = f'<div class="tv-card-warning">{html.escape(str(warning))}</div>'

    assess_rows = ""
    for key in ("stroke_risk", "fall_risk", "respiratory", "mental_status"):
        a = assessments.get(key, {"level": "\u2014", "findings": [], "color": "#94A3B8"})
        a_label = html.escape(ASSESSMENT_LABELS.get(key, key))
        level = html.escape(a["level"])
        color = _level_css_color(a["level"])
        findings = ""
        if a.get("findings"):
            findings = (
                f'<span class="tv-card-assess-findings">'
                f' \u2014 {html.escape(", ".join(a["findings"]))}'
                f'</span>'
            )
        assess_rows += (
            f'<div class="tv-card-assess-row">'
            f'<span class="tv-card-assess-label">{a_label}:</span>'
            f'<span class="tv-card-assess-level" style="color:{color};">{level}</span>'
            f'{findings}'
            f'</div>'
        )

    card_html = (
        f'<div class="tv-card">'
        f'<div class="tv-card-inner">'
        f'<div class="tv-card-bar" style="background:{hex_color};"></div>'
        f'<div class="tv-card-body">'
        f'<div class="tv-card-head">'
        f'<img class="tv-card-face" src="data:image/png;base64,{b64}" '
        f'style="border:2.5px solid {hex_color};" />'
        f'<div class="tv-card-info">'
        f'<div class="tv-card-row1">'
        f'<div class="tv-card-pid">{pid}</div>'
        f'<div class="tv-card-pill" style="background:{hex_color}18;color:{hex_color};">{label}</div>'
        f'</div>'
        f'<div class="tv-card-row2">'
        f'<div class="tv-card-score" style="color:{hex_color};">{score}</div>'
        f'<div class="tv-card-score-sub">/ 100 &nbsp; Triage Severity</div>'
        f'</div>'
        f'</div>'
        f'</div>'
        f'<div class="tv-card-track">'
        f'<div class="tv-card-fill" style="width:{score}%;background:{hex_color};"></div>'
        f'</div>'
        f'<div class="tv-card-assess">{assess_rows}</div>'
        f'<div class="tv-card-ts">{ts}</div>'
        f'{warn_block}'
        f'</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Video processing helpers
# ---------------------------------------------------------------------------

def _frame_to_rgb(frame_bgr):
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _stream_capture_loop(
    capture,
    frame_slot,
    signal_slot,
    *,
    duration_seconds: float | None,
    playback_speed: float = 1.0,
    countdown_prefix: str | None = None,
):
    """Multi-person capture loop. Returns (all_final, latest_rgb_frame).

    Each entry in all_final includes a 'face_crop' key (80x80 RGB array or None).
    """
    tracker = MultiPersonTracker()
    person_states: list[dict] = []
    face_crops: dict[int, object] = {}
    face_emb_map: dict[int, list] = {}
    latest_rgb_frame = None
    start_time = time.monotonic()
    frame_index = 0
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 15.0)
    sleep_seconds = (
        0.0
        if duration_seconds is not None
        else max(0.0, (1.0 / max(fps, 1.0)) / playback_speed)
    )

    with create_pose_estimator() as pose, create_face_mesh_estimator() as face_mesh:
        while True:
            if duration_seconds is not None and time.monotonic() - start_time >= duration_seconds:
                break

            success, frame = capture.read()
            if not success:
                break

            frame_index += 1
            pose_results = None
            face_results = None

            if frame_index % FRAME_SAMPLE_RATE == 0:
                per_face_data, pose_results, face_results = process_frame_multi(
                    frame, pose, face_mesh
                )
                raw_tracks = tracker.update(per_face_data, frame_index)
                person_states = _enrich_person_states(raw_tracks)
                if per_face_data:
                    latest_rgb_frame = _frame_to_rgb(frame)
                    for ps in person_states:
                        pid = ps["person_id"]
                        is_still = ps.get("is_still", False)
                        seen = ps.get("seen_count", 0)
                        geo_emb = ps.get("face_embedding")
                        add_template(face_emb_map.setdefault(pid, []), geo_emb)
                        if (is_still and seen >= 6) or (pid not in face_crops and seen >= 10):
                            face_crops[pid] = crop_face_thumbnail(
                                latest_rgb_frame,
                                face_bbox=ps.get("face_bbox"),
                                center_x=ps["face_center"][0],
                                center_y=ps["face_center"][1],
                            )
                    _relabel_from_queue(person_states, face_emb_map)

            countdown_text = None
            if duration_seconds is not None and countdown_prefix is not None:
                remaining = max(0.0, duration_seconds - (time.monotonic() - start_time))
                countdown_text = f"{countdown_prefix}: {remaining:0.1f}s"

            annotated = annotate_frame(
                frame,
                pose_results,
                face_results,
                person_states,
                countdown_text=countdown_text,
            )
            frame_slot.image(_frame_to_rgb(annotated), width="stretch")
            _update_assess_tiles(signal_slot, person_states)

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    all_final = tracker.get_all_final()
    for pdata in all_final:
        pid = pdata["person_id"]
        pdata["face_crop"] = face_crops.get(pid)
        pdata["face_embeddings"] = face_emb_map.get(pid, [])
    return all_final, latest_rgb_frame


def _capture_live_patients(
    frame_slot, signal_slot, patient_override: str, capture_seconds: int
) -> list[dict]:
    """Capture from webcam and return patient records for all tracked people."""
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        return []
    try:
        all_final, latest_rgb_frame = _stream_capture_loop(
            webcam,
            frame_slot,
            signal_slot,
            duration_seconds=float(capture_seconds),
            countdown_prefix="Capture window",
        )
    finally:
        webcam.release()

    fallback_thumb = resize_thumbnail(latest_rgb_frame) if latest_rgb_frame is not None else None
    records: list[dict] = []
    for pdata in all_final:
        override = patient_override if len(all_final) == 1 else ""
        templates = pdata.get("face_embeddings", [])
        match_idx = _find_matching_patient(templates, st.session_state["patients"])
        if match_idx is not None:
            pid = st.session_state["patients"][match_idx]["patient_id"]
        else:
            pid = _consume_patient_id(override)
        thumb = pdata.get("face_crop")
        if thumb is None:
            thumb = fallback_thumb
        record = build_patient_record(patient_id=pid, signals=pdata["signals"], thumbnail=thumb)
        record["face_embeddings"] = templates
        _save_or_merge_patient(record)
        records.append(record)
    return records


def _preview_live_webcam(frame_slot, signal_slot) -> None:
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        frame_slot.info("Webcam unavailable.")
        return
    try:
        _stream_capture_loop(
            webcam,
            frame_slot,
            signal_slot,
            duration_seconds=PREVIEW_SECONDS,
            countdown_prefix=None,
        )
    finally:
        webcam.release()


def _run_continuous_monitoring(frame_slot, signal_slot):
    """Run continuous webcam monitoring with per-person critical alert auto-save."""
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        st.session_state["monitoring_active"] = False
        frame_slot.warning("Webcam unavailable.")
        return

    tracker = MultiPersonTracker()
    person_states: list[dict] = []
    face_crops: dict[int, object] = {}
    face_emb_map: dict[int, list] = {}
    latest_rgb_frame = None
    frame_index = 0
    start_time = time.monotonic()
    critical_timers: dict[int, float | None] = {}

    try:
        with create_pose_estimator() as pose, create_face_mesh_estimator() as face_mesh:
            while True:
                success, frame = webcam.read()
                if not success:
                    break

                frame_index += 1
                pose_results = None
                face_results = None

                if frame_index % FRAME_SAMPLE_RATE == 0:
                    per_face_data, pose_results, face_results = process_frame_multi(
                        frame, pose, face_mesh
                    )
                    raw_tracks = tracker.update(per_face_data, frame_index)
                    person_states = _enrich_person_states(raw_tracks)
                    if per_face_data:
                        latest_rgb_frame = _frame_to_rgb(frame)
                        for ps in person_states:
                            pid = ps["person_id"]
                            is_still = ps.get("is_still", False)
                            seen = ps.get("seen_count", 0)
                            geo_emb = ps.get("face_embedding")
                            add_template(face_emb_map.setdefault(pid, []), geo_emb)
                            if (is_still and seen >= 6) or (pid not in face_crops and seen >= 10):
                                face_crops[pid] = crop_face_thumbnail(
                                    latest_rgb_frame,
                                    face_bbox=ps.get("face_bbox"),
                                    center_x=ps["face_center"][0],
                                    center_y=ps["face_center"][1],
                                )
                        _relabel_from_queue(person_states, face_emb_map)

                any_critical = False
                for ps in person_states:
                    pid = ps["person_id"]
                    score = ps["score"]
                    if score >= CRITICAL_SCORE_THRESHOLD:
                        if critical_timers.get(pid) is None:
                            critical_timers[pid] = time.monotonic()
                        crit_dur = time.monotonic() - critical_timers[pid]
                        if crit_dur >= CRITICAL_ALERT_SECONDS:
                            person_signals = tracker.get_person_final_and_reset(pid)
                            if person_signals:
                                thumb = face_crops.get(pid)
                                if thumb is None and latest_rgb_frame is not None:
                                    thumb = resize_thumbnail(latest_rgb_frame)
                                templates = face_emb_map.get(pid, [])
                                match_idx = _find_matching_patient(
                                    templates, st.session_state["patients"]
                                )
                                if match_idx is not None:
                                    patient_id = st.session_state["patients"][match_idx]["patient_id"]
                                else:
                                    pnum = int(st.session_state["next_patient_number"])
                                    st.session_state["next_patient_number"] = pnum + 1
                                    patient_id = f"Patient {pnum:03d}"
                                record = build_patient_record(
                                    patient_id=patient_id,
                                    signals=person_signals,
                                    thumbnail=thumb,
                                )
                                record["face_embeddings"] = templates
                                _, merged = _save_or_merge_patient(record)
                                verb = "updated" if merged else "auto-saved"
                                _notify(
                                    f"\U0001f6a8 CRITICAL ALERT \u2014 {patient_id} "
                                    f"(P{pid:02d}) {verb} (score {score} "
                                    f"for {CRITICAL_ALERT_SECONDS:.0f}s)"
                                )
                            critical_timers[pid] = None
                        any_critical = True
                    else:
                        critical_timers[pid] = None

                tracked_ids = {ps["person_id"] for ps in person_states}
                critical_timers = {k: v for k, v in critical_timers.items() if k in tracked_ids}

                elapsed = time.monotonic() - start_time
                mins, secs = int(elapsed // 60), int(elapsed % 60)
                status = f"Monitoring: {mins}:{secs:02d}"
                if any_critical:
                    max_crit = max(
                        (time.monotonic() - t for t in critical_timers.values() if t is not None),
                        default=0.0,
                    )
                    status += f"  |  CRITICAL: {max_crit:.1f}s"

                annotated = annotate_frame(
                    frame, pose_results, face_results,
                    person_states, countdown_text=status,
                )

                if any_critical:
                    h, w = annotated.shape[:2]
                    pulse = 0.5 + 0.5 * math.sin(time.monotonic() * 6)
                    color = (0, 0, int(255 * pulse))
                    cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), color, 6)

                frame_slot.image(_frame_to_rgb(annotated), width="stretch")
                _update_assess_tiles(signal_slot, person_states)

                st.session_state["monitoring_data"] = {
                    "all_final": tracker.get_all_final(),
                    "face_crops": dict(face_crops),
                    "face_emb_map": dict(face_emb_map),
                    "thumbnail": latest_rgb_frame,
                }
    finally:
        webcam.release()


def _play_uploaded_video(
    frame_slot, signal_slot, video_path: str, playback_speed: float
) -> tuple[list[dict], None]:
    """Play an uploaded video and return (all_final, latest_rgb_frame)."""
    capture = cv2.VideoCapture(video_path)
    try:
        all_final, latest_rgb_frame = _stream_capture_loop(
            capture,
            frame_slot,
            signal_slot,
            duration_seconds=None,
            playback_speed=playback_speed,
        )
    finally:
        capture.release()
    return all_final, latest_rgb_frame


def _animate_demo_feed(frame_slot, signal_slot, profile: dict) -> None:
    base_signals = profile["signals"]
    hex_color = (
        "#DC2626"
        if profile["score"] >= 65
        else "#D97706"
        if profile["score"] >= 35
        else "#16A34A"
    )
    for frame_number in range(90):
        phase = frame_number / 6.0
        animated_signals = {}
        for offset, key in enumerate(SIGNAL_KEYS):
            base = float(base_signals.get(key, 0.0))
            animated_signals[key] = max(0.0, min(1.0, base + 0.04 * math.sin(phase + offset)))

        score = compute_score(animated_signals)
        priority = get_priority(score)
        person_states = [{
            "person_id": 0,
            "label": profile["patient_id"],
            "face_center": (480, 230),
            "signals": animated_signals,
            "score": score,
            "hex": priority["hex"],
            "priority_label": priority["label"],
            "assessments": derive_clinical_assessments(animated_signals),
        }]

        frame = create_demo_frame(profile["patient_id"], hex_color, phase)
        annotated = annotate_frame(
            frame, None, None, person_states, countdown_text="Demo feed"
        )
        frame_slot.image(_frame_to_rgb(annotated), width="stretch")
        _update_assess_tiles(signal_slot, person_states)
        time.sleep(0.04)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _sidebar_controls() -> tuple[str, int, str, list, float, bool, bool]:
    with st.sidebar:
        st.markdown(
            '<div style="font-size:11px;font-weight:700;color:#64748B;'
            'text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">Controls</div>',
            unsafe_allow_html=True,
        )
        has_webcam = st.session_state.get("has_webcam", False)
        mode = st.radio("Mode", ["Live Webcam", "Video Upload"], key="selected_mode")
        if mode == "Live Webcam" and not has_webcam:
            st.warning("No webcam detected. Webcam features unavailable.")

        continuous = st.checkbox("Continuous monitoring", key="continuous_mode")
        if continuous:
            capture_seconds = 0
        else:
            capture_seconds = st.slider(
                "Capture duration (s)", 5, 60, DEFAULT_CAPTURE_SECONDS, 1
            )
        patient_override = st.text_input(
            "Patient ID override", placeholder="Auto-assigned if blank"
        )
        playback_label = st.selectbox(
            "Playback speed", list(PLAYBACK_SPEEDS.keys()), index=1
        )

        uploaded_files = []
        if mode == "Video Upload":
            uploaded_files = st.file_uploader(
                "Upload .mp4 videos",
                type=["mp4"],
                accept_multiple_files=True,
            )

        st.markdown(
            '<div style="margin:10px 0 6px;border-top:1px solid #E2E8F0;"></div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        load_demo = c1.button("Load Demo", type="secondary")
        clear_queue = c2.button("Clear Queue", type="secondary")

        st.selectbox(
            "Demo animation",
            [p["patient_id"] for p in st.session_state["demo_profiles"]],
            key="demo_feed_patient",
        )

        st.markdown(
            '<div class="tv-build-note">'
            "SightLion \u2014 AI-assisted ER intake triage with multi-person clinical "
            "assessments, severity scoring, and priority queue."
            "</div>",
            unsafe_allow_html=True,
        )

    return (
        mode,
        capture_seconds,
        patient_override,
        uploaded_files,
        PLAYBACK_SPEEDS[playback_label],
        load_demo,
        clear_queue,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_LOGO_PATH = pathlib.Path(__file__).with_name("logo.png")


@st.cache_data
def _logo_b64() -> str:
    if _LOGO_PATH.exists():
        return base64.b64encode(_LOGO_PATH.read_bytes()).decode()
    return ""


def main() -> None:
    """Render the SightLion dashboard."""
    st.set_page_config(page_title="SightLion", page_icon=str(_LOGO_PATH), layout="wide")
    _inject_styles()
    _ensure_state()

    (
        mode,
        capture_seconds,
        patient_override,
        uploaded_files,
        playback_speed,
        demo_clicked,
        clear_clicked,
    ) = _sidebar_controls()
    has_webcam = st.session_state.get("has_webcam", False)

    if clear_clicked:
        st.session_state["patients"] = []
        st.session_state["next_patient_number"] = 1
        _notify("Priority queue cleared.")

    if demo_clicked:
        st.session_state["patients"] = build_demo_patients()
        _notify("Loaded demo patients.")

    logo_data = _logo_b64()
    logo_tag = (
        f'<img class="tv-logo" src="data:image/png;base64,{logo_data}" alt="SightLion"/>'
        if logo_data
        else ""
    )
    st.markdown(
        f'<div class="tv-header">'
        f'{logo_tag}'
        f'<div class="tv-header-sub">AI-assisted ER intake monitoring</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.4, 1.0], gap="medium")

    with left_col:
        st.markdown(
            '<div class="tv-disclaimer">\u26a0\ufe0f For staff review only. Not a diagnostic tool.</div>',
            unsafe_allow_html=True,
        )

        use_webcam = mode == "Live Webcam" and has_webcam
        section_title = "Live Intake Monitor" if use_webcam else "Video Review"
        st.markdown(
            f'<div class="tv-section-label">{section_title}</div>',
            unsafe_allow_html=True,
        )

        frame_slot = st.empty()
        signal_slot = st.empty()

        if mode == "Live Webcam" and not has_webcam:
            st.markdown(
                '<div class="tv-empty">'
                '<div class="tv-empty-icon">\U0001f4f7</div>'
                "No webcam detected.<br>"
                "Switch to <b>Video Upload</b> mode or press <b>Load Demo</b> to explore."
                "</div>",
                unsafe_allow_html=True,
            )
            _update_assess_tiles(signal_slot, [])

        elif use_webcam:
            continuous = st.session_state.get("continuous_mode", False)
            monitoring = st.session_state.get("monitoring_active", False)

            if not continuous and monitoring:
                st.session_state["monitoring_active"] = False
                st.session_state["monitoring_data"] = None
                monitoring = False

            if continuous:
                _, btn_col, _ = st.columns([1, 2, 1])
                with btn_col:
                    if monitoring:
                        stop_clicked = st.button(
                            "\u23f9  Stop & Save", type="primary"
                        )
                    else:
                        stop_clicked = False
                        start_monitor = st.button(
                            "\u25b6  Start Monitoring", type="primary"
                        )

                if monitoring and stop_clicked:
                    st.session_state["monitoring_active"] = False
                    data = st.session_state.get("monitoring_data")
                    if data and data.get("all_final"):
                        saved_crops = data.get("face_crops", {})
                        saved_emb_map = data.get("face_emb_map", {})
                        fallback = (
                            resize_thumbnail(data["thumbnail"])
                            if data.get("thumbnail") is not None
                            else None
                        )
                        saved_count, merged_count = 0, 0
                        for pdata in data["all_final"]:
                            override = patient_override if len(data["all_final"]) == 1 else ""
                            track_pid = pdata["person_id"]
                            templates = saved_emb_map.get(track_pid, [])
                            match_idx = _find_matching_patient(
                                templates, st.session_state["patients"]
                            )
                            if match_idx is not None:
                                pid = st.session_state["patients"][match_idx]["patient_id"]
                            else:
                                pid = _consume_patient_id(override)
                            thumb = saved_crops.get(track_pid, fallback)
                            record = build_patient_record(
                                patient_id=pid, signals=pdata["signals"], thumbnail=thumb
                            )
                            record["face_embeddings"] = templates
                            _, merged = _save_or_merge_patient(record)
                            saved_count += 1
                            if merged:
                                merged_count += 1
                        parts = []
                        new_count = saved_count - merged_count
                        if new_count:
                            parts.append(f"{new_count} new")
                        if merged_count:
                            parts.append(f"{merged_count} updated")
                        _notify(f"Saved {' / '.join(parts)} patient(s) to queue.")
                    st.session_state["monitoring_data"] = None
                    st.rerun()
                elif monitoring:
                    _run_continuous_monitoring(frame_slot, signal_slot)
                elif start_monitor:
                    st.session_state["monitoring_active"] = True
                    st.session_state["monitoring_data"] = None
                    st.rerun()
                else:
                    _preview_live_webcam(frame_slot, signal_slot)
            else:
                _, btn_col, _ = st.columns([1, 2, 1])
                with btn_col:
                    start_capture = st.button(
                        "\u23fa  Start Capture", type="primary"
                    )

                if start_capture:
                    with st.spinner("Capturing webcam window..."):
                        records = _capture_live_patients(
                            frame_slot, signal_slot, patient_override, capture_seconds
                        )
                    if records:
                        _notify(f"Saved {len(records)} patient(s) to the queue.")
                else:
                    _preview_live_webcam(frame_slot, signal_slot)

        else:
            _, btn_col, _ = st.columns([1, 2, 1])
            with btn_col:
                analyze = st.button(
                    "\u25b6  Play & Analyze Uploads", type="primary"
                )

            if analyze and uploaded_files:
                with st.spinner("Playing uploads with landmark overlay..."):
                    for uf in uploaded_files:
                        tmp = None
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                                f.write(uf.getbuffer())
                                tmp = f.name
                            all_final, latest_rgb_frame = _play_uploaded_video(
                                frame_slot, signal_slot, tmp, playback_speed
                            )
                            fallback_thumb = (
                                resize_thumbnail(latest_rgb_frame)
                                if latest_rgb_frame is not None
                                else None
                            )
                            for pdata in all_final:
                                override = (
                                    patient_override
                                    if len(uploaded_files) == 1 and len(all_final) == 1
                                    else ""
                                )
                                templates = pdata.get("face_embeddings", [])
                                match_idx = _find_matching_patient(
                                    templates, st.session_state["patients"]
                                )
                                if match_idx is not None:
                                    pid = st.session_state["patients"][match_idx]["patient_id"]
                                else:
                                    pid = _consume_patient_id(override)
                                thumb = pdata.get("face_crop", fallback_thumb)
                                rec = build_patient_record(
                                    patient_id=pid,
                                    signals=pdata["signals"],
                                    thumbnail=thumb,
                                )
                                rec["face_embeddings"] = templates
                                _save_or_merge_patient(rec)
                        finally:
                            if tmp and os.path.exists(tmp):
                                os.remove(tmp)
                _notify(f"Processed {len(uploaded_files)} uploaded video(s).")

            elif uploaded_files:
                st.markdown(
                    '<div class="tv-empty">'
                    '<div class="tv-empty-icon">\U0001f3ac</div>'
                    "Videos ready. Press <b>Play &amp; Analyze Uploads</b> to begin."
                    "</div>",
                    unsafe_allow_html=True,
                )
                _update_assess_tiles(signal_slot, [])

            else:
                selected_profile = next(
                    p
                    for p in st.session_state["demo_profiles"]
                    if p["patient_id"] == st.session_state["demo_feed_patient"]
                )
                _animate_demo_feed(frame_slot, signal_slot, selected_profile)

    with right_col:
        patients = sorted(
            st.session_state["patients"],
            key=lambda p: p["score"],
            reverse=True,
        )
        st.session_state["patients"] = patients

        st.markdown(
            f'<div class="tv-queue-header">'
            f'<div class="tv-queue-title">Priority Queue</div>'
            f'<div class="tv-count-badge">{len(patients)} patient{"s" if len(patients) != 1 else ""}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        if not patients:
            st.markdown(
                '<div class="tv-empty">'
                '<div class="tv-empty-icon">\U0001f3e5</div>'
                "No patients in queue yet.<br>"
                "Start a capture, upload videos, or press <b>Load Demo</b>."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            for patient in patients:
                _render_patient_card(patient)

        st.markdown(
            '<div class="tv-legend">'
            '<span class="tv-legend-dot" style="color:#DC2626;">\u25cf</span> Critical \u2265 65 &nbsp;&nbsp;'
            '<span class="tv-legend-dot" style="color:#D97706;">\u25cf</span> Urgent 35\u201364 &nbsp;&nbsp;'
            '<span class="tv-legend-dot" style="color:#16A34A;">\u25cf</span> Stable &lt; 35'
            "</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
