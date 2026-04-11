"""Utility helpers for TriageVision."""

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

    initials = "".join(part[0] for part in label.split()[:2]).upper() or "TV"
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
        image = create_placeholder_thumbnail("TV", "#5B6474")

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


def draw_signal_hud(frame_bgr, signals: dict, score: int, countdown_text: str | None = None) -> None:
    """Draw a semi-transparent live HUD panel with signal values."""
    overlay = frame_bgr.copy()
    panel_width = 360
    line_height = 34
    panel_height = 56 + (len(signals) * line_height) + 60
    cv2.rectangle(overlay, (14, 14), (14 + panel_width, 14 + panel_height), (10, 15, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame_bgr, 0.25, 0, frame_bgr)

    cv2.rectangle(frame_bgr, (14, 14), (14 + panel_width, 14 + panel_height), (50, 60, 80), 1)

    cv2.putText(frame_bgr, "LIVE TRIAGE SIGNALS", (30, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (148, 163, 184), 2)

    y_position = 80
    for signal_name, value in signals.items():
        color = get_signal_bgr(float(value))
        cv2.circle(frame_bgr, (34, y_position - 5), 8, color, -1)
        cv2.putText(
            frame_bgr,
            f"{get_signal_label(signal_name)}: {float(value):.2f}",
            (52, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
        )
        y_position += line_height

    y_position += 4
    cv2.putText(
        frame_bgr,
        f"Score: {score}",
        (30, y_position + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3,
    )

    if countdown_text:
        cv2.putText(
            frame_bgr,
            countdown_text,
            (30, y_position + 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (253, 224, 71),
            2,
        )


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
