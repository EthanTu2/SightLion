"""Utility helpers for SightLion."""

from __future__ import annotations

import base64
import math

import cv2
import numpy as np

from signals import mp_drawing, mp_drawing_styles, mp_face_mesh, mp_pose


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


def crop_face_thumbnail(
    frame_rgb,
    face_bbox: tuple[float, float, float, float] | None = None,
    center_x: float | None = None,
    center_y: float | None = None,
    size: int = 120,
):
    """Crop a square face region.  Prefers *face_bbox* (landmark-derived)
    which scales naturally with distance; falls back to centre + fixed radius."""
    h, w = frame_rgb.shape[:2]

    if face_bbox is not None:
        bx0, by0, bx1, by1 = face_bbox
        fw, fh = bx1 - bx0, by1 - by0
        pad = max(fw, fh) * 0.55
        x0 = max(0, int(bx0 - pad * 0.6))
        y0 = max(0, int(by0 - pad * 0.85))
        x1 = min(w, int(bx1 + pad * 0.6))
        y1 = min(h, int(by1 + pad * 0.4))
    else:
        cx = int(center_x or w // 2)
        cy = int(center_y or h // 2)
        radius = max(int(h * 0.22), 90)
        cy -= int(radius * 0.15)
        x0, y0 = max(0, cx - radius), max(0, cy - radius)
        x1, y1 = min(w, cx + radius), min(h, cy + radius)

    cw, ch = x1 - x0, y1 - y0
    side = max(cw, ch)
    mid_x, mid_y = (x0 + x1) // 2, (y0 + y1) // 2
    sx0 = max(0, mid_x - side // 2)
    sy0 = max(0, mid_y - side // 2)
    sx1 = min(w, sx0 + side)
    sy1 = min(h, sy0 + side)

    crop = frame_rgb[sy0:sy1, sx0:sx1]
    if crop.size == 0:
        return create_placeholder_thumbnail("?", "#94A3B8")
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Face re-identification  (Apple Face ID–inspired multi-template approach)
# ---------------------------------------------------------------------------

_MAX_TEMPLATES = 5
_TEMPLATE_DIVERSITY = 0.95


_GEO_MAX_DIST = 0.50


def face_similarity(emb1, emb2) -> float:
    """Similarity between two geometric face embeddings (0–1).

    Uses Euclidean distance on raw landmark-ratio vectors, converted to
    a 0–1 score.  Distance 0 → 1.0, distance ≥ _GEO_MAX_DIST → 0.0.
    """
    if emb1 is None or emb2 is None:
        return 0.0
    dist = float(np.linalg.norm(emb1 - emb2))
    return max(0.0, 1.0 - dist / _GEO_MAX_DIST)


def best_template_similarity(
    query: np.ndarray | None, templates: list
) -> float:
    """Max cosine similarity between *query* and any template in list."""
    if query is None or not templates:
        return 0.0
    return max(face_similarity(query, t) for t in templates)


def add_template(templates: list, emb, max_n: int = _MAX_TEMPLATES) -> None:
    """Append *emb* to *templates* if sufficiently diverse and under cap."""
    if emb is None:
        return
    for existing in templates:
        if face_similarity(emb, existing) > _TEMPLATE_DIVERSITY:
            return
    if len(templates) < max_n:
        templates.append(emb)


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
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame_bgr,
                face_landmarks,
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


def draw_person_labels(frame_bgr, person_states: list[dict]) -> None:
    """Draw patient labels above each tracked person's face."""
    for person in person_states:
        cx = int(person["face_center"][0])
        label = person["label"]
        bgr = _hex_to_bgr(person["hex"])

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        text_size, _ = cv2.getTextSize(label, font, scale, thickness)

        bbox = person.get("face_bbox")
        if bbox is not None:
            label_y = max(10, int(bbox[1]) - 12)
        else:
            label_y = int(person["face_center"][1]) - 75
        label_x = cx - text_size[0] // 2
        pad = 8

        cv2.rectangle(
            frame_bgr,
            (label_x - pad, label_y - text_size[1] - pad),
            (label_x + text_size[0] + pad, label_y + pad + 2),
            bgr,
            -1,
        )
        cv2.rectangle(
            frame_bgr,
            (label_x - pad, label_y - text_size[1] - pad),
            (label_x + text_size[0] + pad, label_y + pad + 2),
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame_bgr, label, (label_x, label_y),
            font, scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )


_LEVEL_BGR = {
    "HIGH": (38, 38, 220),
    "MODERATE": (6, 119, 217),
    "YES": (38, 38, 220),
    "NO": (74, 163, 22),
    "Unresponsive": (38, 38, 220),
    "Drowsy": (6, 119, 217),
    "Alert": (74, 163, 22),
    "LOW": (74, 163, 22),
}

_HUD_ASSESS_ORDER = [
    ("stroke_risk", "STROKE RISK"),
    ("fall_risk", "FALL RISK"),
    ("respiratory", "RESPIRATORY"),
    ("mental_status", "MENTAL STATUS"),
]


def _score_bgr(score: int) -> tuple[int, int, int]:
    if score >= 65:
        return (38, 38, 220)
    if score >= 35:
        return (6, 119, 217)
    return (74, 163, 22)


def draw_clinical_hud(
    frame_bgr,
    assessments: dict,
    score: int,
    countdown_text: str | None = None,
    track_count: int = 0,
) -> None:
    """Draw a clinical-assessment HUD panel on the video frame."""
    overlay = frame_bgr.copy()
    panel_width = 340
    line_height = 28
    findings_lh = 18
    header_h = 48
    footer_h = 60
    x0, y0 = 14, 14

    total_findings = sum(len(a.get("findings", [])) for a in assessments.values())
    panel_height = header_h + (len(_HUD_ASSESS_ORDER) * line_height) + (total_findings * findings_lh) + footer_h
    if track_count > 1:
        panel_height += 24
    x1, y1 = x0 + panel_width, y0 + panel_height

    cv2.rectangle(overlay, (x0, y0), (x1, y1), (10, 15, 25), -1)
    cv2.addWeighted(overlay, 0.82, frame_bgr, 0.18, 0, frame_bgr)
    cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (60, 70, 90), 1)

    cv2.putText(frame_bgr, "TRIAGE ASSESSMENT", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 163, 184), 1, cv2.LINE_AA)
    cv2.line(frame_bgr, (30, 48), (x1 - 16, 48), (40, 50, 70), 1)

    y_pos = 72
    for key, label in _HUD_ASSESS_ORDER:
        a = assessments.get(key, {"level": "\u2014", "findings": []})
        level = a["level"]
        color = _LEVEL_BGR.get(level, (148, 163, 184))

        cv2.circle(frame_bgr, (34, y_pos - 4), 5, color, -1)
        cv2.putText(frame_bgr, label, (48, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 190, 210), 1, cv2.LINE_AA)
        lvl_size, _ = cv2.getTextSize(level, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
        cv2.putText(frame_bgr, level, (x1 - 20 - lvl_size[0], y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)
        y_pos += line_height

        for finding in a.get("findings", []):
            cv2.putText(frame_bgr, finding, (56, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (130, 140, 160), 1, cv2.LINE_AA)
            y_pos += findings_lh

    cv2.line(frame_bgr, (30, y_pos + 2), (x1 - 16, y_pos + 2), (40, 50, 70), 1)

    score_color = _score_bgr(score)
    cv2.putText(frame_bgr, "SEVERITY", (30, y_pos + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (148, 163, 184), 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, str(score), (110, y_pos + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.90, score_color, 3, cv2.LINE_AA)
    score_w, _ = cv2.getTextSize(str(score), cv2.FONT_HERSHEY_SIMPLEX, 0.90, 3)
    cv2.putText(frame_bgr, "/ 100", (112 + score_w[0] + 4, y_pos + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 130, 150), 1, cv2.LINE_AA)

    footer_y = y_pos + 40
    if track_count > 1:
        cv2.putText(
            frame_bgr,
            f"Tracking {track_count} people",
            (30, footer_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 200, 255), 1, cv2.LINE_AA,
        )
        footer_y += 24

    if countdown_text:
        cv2.putText(frame_bgr, countdown_text, (30, footer_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (253, 224, 71), 1, cv2.LINE_AA)


def annotate_frame(
    frame_bgr,
    pose_results,
    face_results,
    person_states: list[dict],
    countdown_text: str | None = None,
):
    """Return a BGR frame with landmarks, person labels, and clinical HUD."""
    from scorer import derive_clinical_assessments

    annotated = frame_bgr.copy()
    draw_landmark_overlay(annotated, pose_results, face_results)

    if person_states:
        draw_person_labels(annotated, person_states)
        worst = max(person_states, key=lambda p: p["score"])
        draw_clinical_hud(
            annotated, worst["assessments"], worst["score"],
            countdown_text, track_count=len(person_states),
        )
    else:
        empty = derive_clinical_assessments({})
        draw_clinical_hud(annotated, empty, 0, countdown_text)

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
