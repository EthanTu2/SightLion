"""Utility helpers for SightLion."""

from __future__ import annotations

import base64
import math

import cv2
import numpy as np

from signals import mp_drawing, mp_drawing_styles, mp_face_mesh, mp_pose

SHORT_SIGNAL_LABELS = {
    "slumped_posture": "Posture",
    "body_sway": "Sway",
    "tripod_position": "Tripod",
    "arm_drift": "Arm drift",
    "hands_near_throat": "Hands throat",
    "facial_asymmetry": "Face asym.",
    "low_alertness": "Alertness",
}


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    normalized = hex_color.lstrip("#")
    if len(normalized) != 6:
        return (91, 100, 116)
    return tuple(int(normalized[index : index + 2], 16) for index in (0, 2, 4))


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    red, green, blue = _hex_to_rgb(hex_color)
    return (blue, green, red)


def format_signal_name(signal_name: str) -> str:
    """Convert a snake_case signal name into a UI label."""
    return signal_name.replace("_", " ").title()


def get_signal_bgr(value: float) -> tuple[int, int, int]:
    if value > 0.6:
        return (74, 75, 226)
    if value >= 0.3:
        return (39, 159, 239)
    return (80, 175, 76)


def get_signal_label(signal_name: str) -> str:
    return SHORT_SIGNAL_LABELS.get(signal_name, format_signal_name(signal_name))


def create_placeholder_thumbnail(label: str, hex_color: str):
    """Create a simple placeholder thumbnail image."""
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    rgb = _hex_to_rgb(hex_color)
    image[:, :] = rgb

    initials = "".join(part[0] for part in label.split()[:2]).upper() or "SL"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(initials, font, 0.7, 2)
    origin = ((80 - text_size[0]) // 2, (80 + text_size[1]) // 2)
    cv2.putText(image, initials, origin, font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def resize_thumbnail(image, size: tuple[int, int] = (80, 80)):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def image_to_base64(image) -> str:
    """Encode a NumPy image into base64 for HTML rendering."""
    if image is None:
        image = create_placeholder_thumbnail("SL", "#5B6474")

    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".png", bgr_image)
    if not success:
        return ""
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def draw_landmark_overlay(frame_bgr, pose_results, face_results) -> None:
    """Draw MediaPipe face mesh and pose landmarks onto a BGR frame."""
    if face_results and getattr(face_results, "multi_face_landmarks", None):
        mp_drawing.draw_landmarks(
            frame_bgr,
            face_results.multi_face_landmarks[0],
            mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )

    if pose_results and getattr(pose_results, "pose_landmarks", None):
        mp_drawing.draw_landmarks(
            frame_bgr,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )


def _score_bgr(score: int) -> tuple[int, int, int]:
    if score >= 65:
        return (38, 38, 220)
    if score >= 35:
        return (6, 119, 217)
    return (74, 163, 22)


def draw_signal_hud(frame_bgr, signals: dict, score: int, countdown_text: str | None = None) -> None:
    """Draw a semi-transparent live HUD panel with signal values."""
    overlay = frame_bgr.copy()
    panel_width = 370
    line_height = 32
    header_h = 52
    footer_h = 70
    panel_height = header_h + (len(signals) * line_height) + footer_h
    x0, y0 = 14, 14
    x1, y1 = x0 + panel_width, y0 + panel_height

    cv2.rectangle(overlay, (x0, y0), (x1, y1), (10, 15, 25), -1)
    cv2.addWeighted(overlay, 0.80, frame_bgr, 0.20, 0, frame_bgr)
    cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (60, 70, 90), 1)

    cv2.putText(frame_bgr, "LIVE TRIAGE SIGNALS", (30, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 163, 184), 1, cv2.LINE_AA)
    cv2.line(frame_bgr, (30, 50), (x1 - 16, 50), (40, 50, 70), 1)

    y_pos = 74
    for signal_name, value in signals.items():
        v = float(value)
        color = get_signal_bgr(v)
        cv2.circle(frame_bgr, (34, y_pos - 4), 6, color, -1)
        label_text = f"{get_signal_label(signal_name)}"
        cv2.putText(frame_bgr, label_text, (50, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 190, 210), 1, cv2.LINE_AA)
        val_text = f"{v:.2f}"
        val_size, _ = cv2.getTextSize(val_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        cv2.putText(frame_bgr, val_text, (x1 - 22 - val_size[0], y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)

        bar_x0, bar_y0 = 50, y_pos + 6
        bar_w = panel_width - 72
        cv2.rectangle(frame_bgr, (bar_x0, bar_y0), (bar_x0 + bar_w, bar_y0 + 3), (40, 50, 70), -1)
        fill_w = max(1, int(bar_w * v))
        cv2.rectangle(frame_bgr, (bar_x0, bar_y0), (bar_x0 + fill_w, bar_y0 + 3), color, -1)

        y_pos += line_height

    cv2.line(frame_bgr, (30, y_pos + 2), (x1 - 16, y_pos + 2), (40, 50, 70), 1)

    score_color = _score_bgr(score)
    cv2.putText(frame_bgr, "SCORE", (30, y_pos + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, str(score), (100, y_pos + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, score_color, 3, cv2.LINE_AA)

    if countdown_text:
        cv2.putText(frame_bgr, countdown_text, (30, y_pos + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (253, 224, 71), 1, cv2.LINE_AA)


def annotate_frame(frame_bgr, pose_results, face_results, signals: dict, score: int, countdown_text: str | None = None):
    """Return a BGR frame with landmarks and HUD annotations."""
    annotated = frame_bgr.copy()
    draw_landmark_overlay(annotated, pose_results, face_results)
    draw_signal_hud(annotated, signals, score, countdown_text=countdown_text)
    return annotated


def create_demo_frame(patient_id: str, hex_color: str, phase: float):
    """Create a colored fake demo frame with animated landmark dots."""
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    frame[:, :] = _hex_to_bgr(hex_color)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (960, 540), (14, 18, 24), -1)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    cv2.putText(frame, patient_id, (32, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

    center_x = int(480 + 24 * math.sin(phase * 0.9))
    center_y = int(230 + 18 * math.cos(phase * 0.7))
    face_radius = 54
    cv2.circle(frame, (center_x, center_y), face_radius, (255, 255, 255), 2)

    face_points = [
        (center_x - 22, center_y - 8 + int(3 * math.sin(phase))),
        (center_x + 22, center_y - 8 + int(3 * math.cos(phase))),
        (center_x, center_y + 10 + int(5 * math.sin(phase * 1.1))),
        (center_x - 18, center_y + 30 + int(4 * math.sin(phase * 0.8))),
        (center_x + 18, center_y + 30 + int(4 * math.cos(phase * 0.8))),
    ]
    for point in face_points:
        cv2.circle(frame, point, 4, (255, 255, 255), -1)

    pose_points = [
        (center_x, center_y + 65),
        (center_x - 52, center_y + 120 + int(8 * math.sin(phase))),
        (center_x + 52, center_y + 120 + int(8 * math.cos(phase))),
        (center_x - 90 + int(10 * math.sin(phase * 0.9)), center_y + 205),
        (center_x + 90 + int(10 * math.cos(phase * 0.9)), center_y + 205),
        (center_x - 36, center_y + 250),
        (center_x + 36, center_y + 250),
        (center_x - 55 + int(6 * math.sin(phase)), center_y + 340),
        (center_x + 55 + int(6 * math.cos(phase)), center_y + 340),
    ]
    for point in pose_points:
        cv2.circle(frame, point, 5, (255, 255, 255), -1)

    for start, end in zip(pose_points[:-1], pose_points[1:]):
        cv2.line(frame, start, end, (255, 255, 255), 2)

    return frame
