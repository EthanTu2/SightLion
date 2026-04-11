"""Streamlit dashboard for SightLion."""

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

_SIGNAL_SHORT = {
    "slumped_posture": "Posture",
    "body_sway": "Sway",
    "tripod_position": "Tripod",
    "arm_drift": "Arm Drift",
    "hands_near_throat": "Throat",
    "facial_asymmetry": "Asymmetry",
    "low_alertness": "Alertness",
}


def _bar_color(v: float) -> str:
    if v > 0.6:
        return "#DC2626"
    if v >= 0.3:
        return "#D97706"
    return "#16A34A"


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
            .tv-cross {
                width: 34px; height: 34px; border-radius: 8px;
                background: var(--red);
                display: inline-flex; align-items: center; justify-content: center;
                font-size: 20px; font-weight: 700; color: #fff; line-height: 1;
            }
            .tv-header-title { font-size: 23px; font-weight: 800; letter-spacing: -0.5px; }
            .tv-header-sub   { font-size: 12.5px; color: var(--slate-400); margin-top: 4px; }

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

            /* ── Signal tiles ───────────────────────────── */
            .tv-signal-row {
                display: flex; gap: 6px;
                margin-bottom: 14px;
            }
            .tv-signal-cell {
                flex: 1; text-align: center;
                background: #fff;
                border: 1px solid var(--slate-200);
                border-radius: 8px;
                padding: 10px 2px 12px;
                transition: box-shadow 0.15s;
            }
            .tv-signal-cell:hover { box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
            .tv-signal-lbl {
                font-size: 9.5px; font-weight: 700; color: var(--slate-500);
                text-transform: uppercase; letter-spacing: 0.4px;
                margin-bottom: 8px;
            }
            .tv-signal-track {
                height: 6px; border-radius: 999px;
                background: var(--slate-200);
                overflow: hidden; margin: 0 6px;
            }
            .tv-signal-fill { height: 100%; border-radius: 999px; transition: width 0.2s; }
            .tv-signal-val {
                font-size: 12px; font-weight: 800; margin-top: 6px;
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
            .tv-card-explanation {
                margin-top: 10px; font-size: 12.5px; color: var(--slate-500);
                line-height: 1.5;
            }
            .tv-card-ts { margin-top: 4px; font-size: 11px; color: var(--slate-400); }
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
        f'<div class="tv-card-row1">'
        f'<div class="tv-card-pid">{pid}</div>'
        f'<div class="tv-card-pill" style="background:{hex_color}18;color:{hex_color};">{label}</div>'
        f'</div>'
        f'<div class="tv-card-row2">'
        f'<div class="tv-card-score" style="color:{hex_color};">{score}</div>'
        f'<div class="tv-card-score-sub">/ 100</div>'
        f'</div>'
        f'<div class="tv-card-track">'
        f'<div class="tv-card-fill" style="width:{score}%;background:{hex_color};"></div>'
        f'</div>'
        f'<div class="tv-card-explanation">{explanation}</div>'
        f'<div class="tv-card-ts">{ts}</div>'
        f'{warn_block}'
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


def _update_signal_bars(signal_slot, signals: dict) -> None:
    cells = ""
    for key in SIGNAL_KEYS:
        v = float(signals.get(key, 0.0))
        color = _bar_color(v)
        label = _SIGNAL_SHORT.get(key, key)
        cells += (
            f'<div class="tv-signal-cell">'
            f'<div class="tv-signal-lbl">{label}</div>'
            f'<div class="tv-signal-track">'
            f'<div class="tv-signal-fill" style="width:{int(v*100)}%;background:{color};"></div>'
            f'</div>'
            f'<div class="tv-signal-val" style="color:{color};">{v:.2f}</div>'
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
            '<div style="font-size:11px;font-weight:700;color:#64748B;'
            'text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">Controls</div>',
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
            "SightLion MVP — rules-based signal extraction, weighted scoring, "
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
    """Render the SightLion dashboard."""
    st.set_page_config(page_title="SightLion", layout="wide")
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
        '<div class="tv-header-title">SightLion</div>'
        '<div class="tv-header-sub">AI-assisted ER intake monitoring for staff review</div>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.4, 1.0], gap="medium")

    with left_col:
        st.markdown(
            '<div class="tv-disclaimer">⚠️ For staff review only. Not a diagnostic tool.</div>',
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
                '<div class="tv-empty-icon">📷</div>'
                "No webcam detected.<br>"
                "Switch to <b>Video Upload</b> mode or press <b>Load Demo</b> to explore."
                "</div>",
                unsafe_allow_html=True,
            )
            _update_signal_bars(signal_slot, zero_signals())

        elif use_webcam:
            _, btn_col, _ = st.columns([1, 2, 1])
            with btn_col:
                start_capture = st.button(
                    "⏺  Start Capture", type="primary"
                )

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
            _, btn_col, _ = st.columns([1, 2, 1])
            with btn_col:
                analyze = st.button(
                    "▶  Play & Analyze Uploads", type="primary"
                )

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
                    "Videos ready. Press <b>Play &amp; Analyze Uploads</b> to begin."
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
                '<div class="tv-empty-icon">🏥</div>'
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
            '<span class="tv-legend-dot" style="color:#DC2626;">●</span> Critical ≥ 65 &nbsp;&nbsp;'
            '<span class="tv-legend-dot" style="color:#D97706;">●</span> Urgent 35–64 &nbsp;&nbsp;'
            '<span class="tv-legend-dot" style="color:#16A34A;">●</span> Stable &lt; 35'
            "</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
