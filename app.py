"""Streamlit dashboard for TriageVision."""

from __future__ import annotations

import html
import math
import os
import tempfile
import time

import cv2
import streamlit as st

from processor import build_demo_patients, build_patient_record, get_demo_profiles
from scorer import compute_score
from signals import (
    FRAME_SAMPLE_RATE,
    SIGNAL_KEYS,
    SignalAccumulator,
    create_face_mesh_estimator,
    create_pose_estimator,
    process_frame,
    zero_signals,
)
from utils import (
    annotate_frame,
    create_demo_frame,
    format_signal_name,
    image_to_base64,
    resize_thumbnail,
)

PREVIEW_SECONDS = 3.0
DEFAULT_CAPTURE_SECONDS = 10
PLAYBACK_SPEEDS = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0}


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp { background: #F4F6F9; }

            section[data-testid="stSidebar"] > div:first-child {
                padding-top: 1rem;
            }
            section[data-testid="stSidebar"] .stRadio > div { gap: 0.35rem; }
            section[data-testid="stSidebar"] .stSlider { margin-bottom: -0.4rem; }

            .tv-header {
                background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
                color: #FFFFFF;
                padding: 16px 24px;
                border-radius: 10px;
                margin-bottom: 16px;
                display: flex;
                align-items: center;
                gap: 14px;
            }
            .tv-cross {
                width: 32px; height: 32px;
                border-radius: 7px;
                background: #DC2626;
                display: inline-flex;
                align-items: center; justify-content: center;
                font-size: 20px; font-weight: 700; color: #fff;
                line-height: 1;
            }
            .tv-header-title {
                font-size: 22px; font-weight: 800;
                letter-spacing: -0.4px;
            }
            .tv-header-sub {
                font-size: 12.5px; color: #94A3B8;
                margin-top: 3px;
            }

            .tv-disclaimer {
                background: #FFFBEB;
                border: 1px solid #FDE68A;
                color: #92400E;
                border-radius: 8px;
                padding: 7px 12px;
                font-size: 12px;
                font-weight: 600;
                margin-bottom: 10px;
            }

            .tv-section-label {
                font-size: 14px;
                font-weight: 800;
                color: #1E293B;
                text-transform: uppercase;
                letter-spacing: 0.8px;
                margin-bottom: 8px;
            }

            .tv-queue-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 12px;
                padding-bottom: 10px;
                border-bottom: 1px solid #E2E8F0;
            }
            .tv-queue-title {
                font-size: 16px; font-weight: 800; color: #0F172A;
                letter-spacing: -0.2px;
            }
            .tv-count-badge {
                background: #0F172A; color: #FFFFFF;
                padding: 3px 10px; border-radius: 999px;
                font-size: 11px; font-weight: 700;
            }

            .tv-card {
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 10px;
                overflow: hidden;
                margin-bottom: 12px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }
            .tv-card-inner { display: flex; align-items: stretch; }
            .tv-card-bar { width: 5px; flex-shrink: 0; }
            .tv-card-body {
                padding: 16px; display: flex; gap: 14px; width: 100%;
            }
            .tv-card-thumb {
                width: 64px; height: 64px;
                object-fit: cover; border-radius: 8px;
                border: 1px solid #E2E8F0;
                flex-shrink: 0;
            }
            .tv-card-content { flex: 1; min-width: 0; }
            .tv-card-top {
                display: flex; justify-content: space-between;
                align-items: center; gap: 8px;
            }
            .tv-card-pid {
                font-size: 14px; font-weight: 700; color: #0F172A;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            }
            .tv-card-pill {
                padding: 3px 10px; border-radius: 999px;
                font-size: 11px; font-weight: 700;
                white-space: nowrap;
            }
            .tv-card-score {
                font-size: 24px; font-weight: 800; color: #0F172A;
                margin-top: 6px;
            }
            .tv-card-score span { font-size: 13px; font-weight: 500; color: #94A3B8; }
            .tv-card-bar-track {
                margin-top: 6px; height: 5px;
                background: #E2E8F0; border-radius: 999px; overflow: hidden;
            }
            .tv-card-bar-fill { height: 100%; border-radius: 999px; }
            .tv-card-explanation {
                margin-top: 8px; font-size: 12px; color: #64748B;
                line-height: 1.45;
            }
            .tv-card-ts { margin-top: 4px; font-size: 11px; color: #94A3B8; }
            .tv-card-warning {
                margin-top: 8px; padding: 6px 10px; border-radius: 6px;
                background: #FFFBEB; border: 1px solid #FDE68A;
                color: #92400E; font-size: 11px; font-weight: 600;
            }

            .tv-empty {
                text-align: center;
                padding: 44px 20px;
                color: #94A3B8; font-size: 13.5px;
                background: #FFFFFF;
                border: 1.5px dashed #CBD5E1;
                border-radius: 12px;
                line-height: 1.6;
            }
            .tv-empty-icon { font-size: 36px; margin-bottom: 10px; }

            .tv-legend {
                margin-top: 14px;
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
                padding: 12px 16px;
                color: #475569;
                font-size: 13px;
                line-height: 1.7;
                text-align: center;
            }

            .tv-build-note {
                font-size: 10.5px; color: #94A3B8;
                line-height: 1.45; margin-top: 6px;
                border-top: 1px solid #E2E8F0;
                padding-top: 8px;
            }

            .tv-signal-row {
                display: flex; gap: 8px;
                margin-top: 8px;
            }
            .tv-signal-cell {
                flex: 1; text-align: center;
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
                padding: 8px 4px 10px;
            }
            .tv-signal-cell-label {
                font-size: 10px; font-weight: 600;
                color: #64748B;
                text-transform: uppercase;
                letter-spacing: 0.3px;
                margin-bottom: 6px;
            }
            .tv-signal-cell-bar {
                height: 6px;
                border-radius: 999px;
                background: #E2E8F0;
                overflow: hidden;
                margin: 0 4px;
            }
            .tv-signal-cell-fill {
                height: 100%;
                border-radius: 999px;
            }
            .tv-signal-cell-val {
                font-size: 11px; font-weight: 700;
                color: #0F172A;
                margin-top: 4px;
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


def _consume_patient_id(override_value: str) -> str:
    patient_number = int(st.session_state["next_patient_number"])
    st.session_state["next_patient_number"] = patient_number + 1
    override = override_value.strip()
    if override:
        return override
    return f"Patient {patient_number:03d}"


def _render_patient_card(patient: dict) -> None:
    b64 = image_to_base64(patient.get("thumbnail"))
    pid = html.escape(str(patient.get("patient_id", "Unknown")))
    label = html.escape(str(patient.get("label", "Stable")))
    explanation = html.escape(str(patient.get("explanation", "")))
    ts = html.escape(str(patient.get("timestamp", "")))
    hex_color = patient["hex"]
    score = patient["score"]

    warning = patient.get("warning")
    warn_block = ""
    if warning:
        warn_block = f'<div class="tv-card-warning">{html.escape(str(warning))}</div>'

    card_html = (
        f'<div class="tv-card">'
        f'<div class="tv-card-inner">'
        f'<div class="tv-card-bar" style="background:{hex_color};"></div>'
        f'<div class="tv-card-body">'
        f'<img class="tv-card-thumb" src="data:image/png;base64,{b64}" />'
        f'<div class="tv-card-content">'
        f'<div class="tv-card-top">'
        f'<div class="tv-card-pid">{pid}</div>'
        f'<div class="tv-card-pill" style="background:{hex_color}18;color:{hex_color};">{label}</div>'
        f'</div>'
        f'<div class="tv-card-score">{score} <span>/ 100</span></div>'
        f'<div class="tv-card-bar-track">'
        f'<div class="tv-card-bar-fill" style="width:{score}%;background:{hex_color};"></div>'
        f'</div>'
        f'<div class="tv-card-explanation">{explanation}</div>'
        f'<div class="tv-card-ts">{ts}</div>'
        f'{warn_block}'
        f'</div>'
        f'</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

    with st.expander("Signal breakdown"):
        for signal_name in SIGNAL_KEYS:
            value = float(patient.get("signals", {}).get(signal_name, 0.0))
            st.progress(
                int(value * 100),
                text=f"{format_signal_name(signal_name)}  —  {value:.2f}",
            )


_SIGNAL_SHORT = {
    "slumped_posture": "Posture",
    "body_sway": "Sway",
    "tripod_position": "Tripod",
    "arm_drift": "Arm Drift",
    "hands_near_throat": "Throat",
    "facial_asymmetry": "Asym.",
    "low_alertness": "Alert",
}


def _bar_color(v: float) -> str:
    if v > 0.6:
        return "#DC2626"
    if v >= 0.3:
        return "#D97706"
    return "#16A34A"


def _update_signal_bars(signal_slot, signals: dict) -> None:
    cells = ""
    for key in SIGNAL_KEYS:
        v = float(signals.get(key, 0.0))
        color = _bar_color(v)
        label = _SIGNAL_SHORT.get(key, key)
        cells += (
            f'<div class="tv-signal-cell">'
            f'<div class="tv-signal-cell-label">{label}</div>'
            f'<div class="tv-signal-cell-bar">'
            f'<div class="tv-signal-cell-fill" style="width:{int(v*100)}%;background:{color};"></div>'
            f'</div>'
            f'<div class="tv-signal-cell-val" style="color:{color};">{v:.2f}</div>'
            f'</div>'
        )
    signal_slot.markdown(
        f'<div class="tv-signal-row">{cells}</div>',
        unsafe_allow_html=True,
    )


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
    accumulator = SignalAccumulator()
    latest_signals = zero_signals()
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
                frame_signals, pose_results, face_results = process_frame(
                    frame, pose, face_mesh
                )
                if frame_signals is not None:
                    accumulator.add_frame(frame_signals)
                    latest_signals = accumulator.current_signals()
                    latest_rgb_frame = _frame_to_rgb(frame)

            countdown_text = None
            if duration_seconds is not None and countdown_prefix is not None:
                remaining = max(0.0, duration_seconds - (time.monotonic() - start_time))
                countdown_text = f"{countdown_prefix}: {remaining:0.1f}s"

            annotated = annotate_frame(
                frame,
                pose_results,
                face_results,
                latest_signals,
                compute_score(latest_signals),
                countdown_text=countdown_text,
            )
            frame_slot.image(_frame_to_rgb(annotated), width="stretch")
            _update_signal_bars(signal_slot, latest_signals)

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    return accumulator.final_signals(), latest_rgb_frame


def _capture_live_patient(frame_slot, signal_slot, patient_id: str, capture_seconds: int):
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        return None
    try:
        signals, latest_rgb_frame = _stream_capture_loop(
            webcam,
            frame_slot,
            signal_slot,
            duration_seconds=float(capture_seconds),
            countdown_prefix="Capture window",
        )
    finally:
        webcam.release()
    thumbnail = resize_thumbnail(latest_rgb_frame) if latest_rgb_frame is not None else None
    return build_patient_record(patient_id=patient_id, signals=signals, thumbnail=thumbnail)


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


def _play_uploaded_video(
    frame_slot, signal_slot, video_path: str, patient_id: str, playback_speed: float
):
    capture = cv2.VideoCapture(video_path)
    try:
        signals, latest_rgb_frame = _stream_capture_loop(
            capture,
            frame_slot,
            signal_slot,
            duration_seconds=None,
            playback_speed=playback_speed,
        )
    finally:
        capture.release()
    thumbnail = resize_thumbnail(latest_rgb_frame) if latest_rgb_frame is not None else None
    return build_patient_record(patient_id=patient_id, signals=signals, thumbnail=thumbnail)


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

        frame = create_demo_frame(profile["patient_id"], hex_color, phase)
        annotated = annotate_frame(
            frame, None, None, animated_signals, profile["score"], countdown_text="Demo feed"
        )
        frame_slot.image(_frame_to_rgb(annotated), width="stretch")
        _update_signal_bars(signal_slot, animated_signals)
        time.sleep(0.04)


def _sidebar_controls() -> tuple[str, int, str, list, float, bool, bool]:
    with st.sidebar:
        st.markdown(
            '<div style="font-size:18px;font-weight:700;color:#0F172A;margin-bottom:12px;">Controls</div>',
            unsafe_allow_html=True,
        )
        has_webcam = st.session_state.get("has_webcam", False)
        mode = st.radio("Mode", ["Live Webcam", "Video Upload"], key="selected_mode")
        if mode == "Live Webcam" and not has_webcam:
            st.warning("No webcam detected. Webcam features unavailable.")

        capture_seconds = st.slider(
            "Capture duration (s)", 5, 20, DEFAULT_CAPTURE_SECONDS, 1
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

        st.markdown('<div style="margin:8px 0 4px;border-top:1px solid #E2E8F0;"></div>', unsafe_allow_html=True)
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
            "TriageVision MVP — rules-based signal extraction, weighted scoring, "
            "priority queue, live webcam, video playback, and demo mode."
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


def main() -> None:
    """Render the TriageVision dashboard."""
    st.set_page_config(page_title="TriageVision", layout="wide")
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

    st.markdown(
        '<div class="tv-header">'
        '<div class="tv-cross">+</div>'
        "<div>"
        '<div class="tv-header-title">TriageVision</div>'
        '<div class="tv-header-sub">AI-assisted ER intake monitoring for staff review</div>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.55, 1.0], gap="medium")

    # ── Left: live feed ──────────────────────────────────────────────
    with left_col:
        st.markdown(
            '<div class="tv-disclaimer">⚠️ For staff review only. Not a diagnostic tool.</div>',
            unsafe_allow_html=True,
        )

        use_webcam = mode == "Live Webcam" and has_webcam
        section_title = "Live Intake Monitor" if use_webcam else "Video Review"
        st.markdown(f'<div class="tv-section-label">{section_title}</div>', unsafe_allow_html=True)

        frame_slot = st.empty()
        signal_slot = st.empty()

        if mode == "Live Webcam" and not has_webcam:
            st.markdown(
                '<div class="tv-empty">'
                '<div class="tv-empty-icon">📷</div>'
                "No webcam detected.<br>Switch to <b>Video Upload</b> or load <b>Demo patients</b>."
                "</div>",
                unsafe_allow_html=True,
            )
            _update_signal_bars(signal_slot, zero_signals())

        elif use_webcam:
            start_capture = st.button("⏺  Start capture", type="primary")

            if start_capture:
                patient_id = _consume_patient_id(patient_override)
                with st.spinner("Capturing webcam window..."):
                    record = _capture_live_patient(
                        frame_slot, signal_slot, patient_id, capture_seconds
                    )
                if record is not None:
                    st.session_state["patients"].append(record)
                    _notify(f"Added {record['patient_id']} to the queue.")
            else:
                _preview_live_webcam(frame_slot, signal_slot)

        else:
            analyze = st.button("▶  Play and analyze uploads", type="primary")

            if analyze and uploaded_files:
                with st.spinner("Playing uploads with landmark overlay..."):
                    for uf in uploaded_files:
                        tmp = None
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                                f.write(uf.getbuffer())
                                tmp = f.name
                            pid = _consume_patient_id(
                                patient_override if len(uploaded_files) == 1 else ""
                            )
                            rec = _play_uploaded_video(
                                frame_slot, signal_slot, tmp, pid, playback_speed
                            )
                            st.session_state["patients"].append(rec)
                        finally:
                            if tmp and os.path.exists(tmp):
                                os.remove(tmp)
                _notify(f"Processed {len(uploaded_files)} uploaded video(s).")

            elif uploaded_files:
                st.markdown(
                    '<div class="tv-empty">'
                    '<div class="tv-empty-icon">🎬</div>'
                    "Press <b>Play and analyze uploads</b> to begin."
                    "</div>",
                    unsafe_allow_html=True,
                )
                _update_signal_bars(signal_slot, zero_signals())

            else:
                selected_profile = next(
                    p
                    for p in st.session_state["demo_profiles"]
                    if p["patient_id"] == st.session_state["demo_feed_patient"]
                )
                _animate_demo_feed(frame_slot, signal_slot, selected_profile)

    # ── Right: priority queue ────────────────────────────────────────
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
            f'<div class="tv-count-badge">{len(patients)}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        if not patients:
            st.markdown(
                '<div class="tv-empty">'
                '<div class="tv-empty-icon">🏥</div>'
                "No patients in queue.<br>"
                "Start a capture, upload videos, or load demo patients."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            for patient in patients:
                _render_patient_card(patient)

        st.markdown(
            '<div class="tv-legend">'
            '<span style="color:#DC2626;font-weight:700;">●</span> Critical ≥ 65 &nbsp;&nbsp;'
            '<span style="color:#D97706;font-weight:700;">●</span> Urgent 35–64 &nbsp;&nbsp;'
            '<span style="color:#16A34A;font-weight:700;">●</span> Stable &lt; 35'
            "</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
