"""Signal extraction for SightLion."""

from __future__ import annotations

import math
import os
from typing import Dict

os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
import numpy as np  # noqa: E402

try:
    import absl.logging  # noqa: E402

    absl.logging.set_verbosity(absl.logging.ERROR)
    absl.logging.set_stderrthreshold(absl.logging.ERROR)
except ImportError:
    pass

FRAME_SAMPLE_RATE = 2
MIN_VALID_FRAMES = 5

POSTURE_UPRIGHT_RATIO = 0.50
POSTURE_RANGE = 0.35
BODY_SWAY_NORM_SCALE = 0.14
SWAY_DEAD_ZONE = 0.018
EMA_ALPHA = 0.35
ARM_DRIFT_SCALE = 0.80
THROAT_Y_RANGE_RATIO = 0.15
THROAT_X_RANGE_RATIO = 0.25
THROAT_FRAME_FRACTION_THRESHOLD = 0.30
FACIAL_ASYMMETRY_FACE_WIDTH_RATIO = 0.35
EAR_OPEN_THRESHOLD = 0.28
EAR_RANGE = 0.12

_LEFT_EYE_LANDMARKS = (33, 160, 158, 133, 153, 144)
_RIGHT_EYE_LANDMARKS = (263, 387, 385, 362, 380, 373)

SIGNAL_KEYS = (
    "slumped_posture",
    "body_sway",
    "tripod_position",
    "arm_drift",
    "hands_near_throat",
    "facial_asymmetry",
    "low_alertness",
)

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def zero_signals() -> Dict[str, float]:
    return {key: 0.0 for key in SIGNAL_KEYS}


def create_pose_estimator():
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def create_face_mesh_estimator(max_num_faces: int = 5):
    return mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=max_num_faces,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


class SignalAccumulator:
    """Accumulate frame-level signals into running averages."""

    _window = 90

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        from collections import deque
        self.valid_frames = 0
        self.nose_positions_x: deque[float] = deque(maxlen=self._window)
        self.shoulder_widths: deque[float] = deque(maxlen=self._window)
        self._ema: dict[str, float] = {}
        self.samples = {
            "slumped_posture": [],
            "tripod_position": [],
            "arm_drift": [],
            "hands_near_throat": [],
            "facial_asymmetry": [],
            "low_alertness": [],
        }

    def add_frame(self, frame_signals: dict) -> None:
        self.valid_frames += 1
        self.nose_positions_x.append(float(frame_signals["nose_x"]))
        sw = float(frame_signals.get("shoulder_width", 0.0))
        if sw > 0:
            self.shoulder_widths.append(sw)
        for key in self.samples:
            self.samples[key].append(float(frame_signals[key]))

    def current_signals(self) -> Dict[str, float]:
        signals = zero_signals()
        if self.valid_frames == 0:
            return signals

        signals["slumped_posture"] = _safe_mean(self.samples["slumped_posture"])
        signals["tripod_position"] = _safe_mean(self.samples["tripod_position"])
        signals["arm_drift"] = _safe_mean(self.samples["arm_drift"])
        signals["facial_asymmetry"] = _safe_mean(self.samples["facial_asymmetry"])
        signals["low_alertness"] = _safe_mean(self.samples["low_alertness"])

        throat_fraction = _safe_mean(self.samples["hands_near_throat"])
        signals["hands_near_throat"] = (
            throat_fraction if throat_fraction >= THROAT_FRAME_FRACTION_THRESHOLD else 0.0
        )

        if len(self.nose_positions_x) > 2:
            nose_std = float(np.std(list(self.nose_positions_x)))
            ref = _safe_mean(list(self.shoulder_widths)) if self.shoulder_widths else 0.0
            if ref > 1.0:
                norm_sway = nose_std / ref
            else:
                norm_sway = 0.0
            if norm_sway < SWAY_DEAD_ZONE:
                norm_sway = 0.0
            raw_sway = _clamp01(norm_sway / BODY_SWAY_NORM_SCALE)
            prev_sway = self._ema.get("body_sway")
            if prev_sway is not None:
                raw_sway = EMA_ALPHA * raw_sway + (1.0 - EMA_ALPHA) * prev_sway
            self._ema["body_sway"] = raw_sway
            signals["body_sway"] = raw_sway
        return signals

    def final_signals(self) -> Dict[str, float]:
        signals = self.current_signals()
        return {**signals, "_valid_frames": self.valid_frames}


def _pose_px(landmarks, index: int, frame_w: int, frame_h: int) -> tuple[float, float]:
    landmark = landmarks[index]
    return landmark.x * frame_w, landmark.y * frame_h


def _face_xy(landmarks, index: int) -> tuple[float, float]:
    landmark = landmarks[index]
    return landmark.x, landmark.y


def _facial_asymmetry(face_landmarks, frame_w: int) -> float:
    left_eye_x = face_landmarks[33].x * frame_w
    right_eye_x = face_landmarks[263].x * frame_w
    left_mouth_x = face_landmarks[61].x * frame_w
    right_mouth_x = face_landmarks[291].x * frame_w
    nose_x = face_landmarks[1].x * frame_w

    face_width = max(abs(right_eye_x - left_eye_x), 1.0)
    eye_asymmetry = abs(abs(nose_x - left_eye_x) - abs(right_eye_x - nose_x))
    mouth_asymmetry = abs(abs(nose_x - left_mouth_x) - abs(right_mouth_x - nose_x))
    average_asymmetry = (eye_asymmetry + mouth_asymmetry) / 2.0
    return _clamp01(average_asymmetry / (face_width * FACIAL_ASYMMETRY_FACE_WIDTH_RATIO))


def _single_eye_ear(face_landmarks, indices: tuple) -> float:
    """6-point Eye Aspect Ratio for one eye.

    indices: (lateral, upper_lat, upper_med, medial, lower_med, lower_lat)
    """
    pts = [_face_xy(face_landmarks, i) for i in indices]
    v1 = _euclidean(pts[1], pts[5])
    v2 = _euclidean(pts[2], pts[4])
    h = _euclidean(pts[0], pts[3])
    if h == 0:
        return 1.0
    return (v1 + v2) / (2.0 * h)


def _low_alertness(face_landmarks) -> float:
    """Continuous alertness score averaging both eyes."""
    left_ear = _single_eye_ear(face_landmarks, _LEFT_EYE_LANDMARKS)
    right_ear = _single_eye_ear(face_landmarks, _RIGHT_EYE_LANDMARKS)
    avg_ear = (left_ear + right_ear) / 2.0
    return _clamp01((EAR_OPEN_THRESHOLD - avg_ear) / EAR_RANGE)


def _hands_near_throat(
    hand_points: list[tuple[float, float]],
    nose_x: float,
    nose_y: float,
    shoulder_mid_y: float,
    frame_w: int,
    frame_h: int,
) -> float:
    min_x = nose_x - (frame_w * THROAT_X_RANGE_RATIO)
    max_x = nose_x + (frame_w * THROAT_X_RANGE_RATIO)
    min_y = nose_y
    throat_depth = (shoulder_mid_y - nose_y) * 0.75
    max_y = nose_y + max(throat_depth, frame_h * THROAT_Y_RANGE_RATIO)

    for px, py in hand_points:
        if min_x <= px <= max_x and min_y <= py <= max_y:
            return 1.0
    return 0.0


def analyze_landmarks(pose_landmarks, face_landmarks, frame_shape) -> Dict[str, float]:
    """Convert MediaPipe landmarks for one frame into signal values."""
    frame_h, frame_w = frame_shape[:2]

    left_shoulder = _pose_px(pose_landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value, frame_w, frame_h)
    right_shoulder = _pose_px(pose_landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, frame_w, frame_h)
    left_hip = _pose_px(pose_landmarks, mp_pose.PoseLandmark.LEFT_HIP.value, frame_w, frame_h)
    right_hip = _pose_px(pose_landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value, frame_w, frame_h)
    left_wrist = _pose_px(pose_landmarks, mp_pose.PoseLandmark.LEFT_WRIST.value, frame_w, frame_h)
    right_wrist = _pose_px(pose_landmarks, mp_pose.PoseLandmark.RIGHT_WRIST.value, frame_w, frame_h)
    nose_pose = _pose_px(pose_landmarks, mp_pose.PoseLandmark.NOSE.value, frame_w, frame_h)
    left_index = _pose_px(pose_landmarks, mp_pose.PoseLandmark.LEFT_INDEX.value, frame_w, frame_h)
    right_index = _pose_px(pose_landmarks, mp_pose.PoseLandmark.RIGHT_INDEX.value, frame_w, frame_h)
    left_thumb = _pose_px(pose_landmarks, mp_pose.PoseLandmark.LEFT_THUMB.value, frame_w, frame_h)
    right_thumb = _pose_px(pose_landmarks, mp_pose.PoseLandmark.RIGHT_THUMB.value, frame_w, frame_h)

    shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
    hip_mid_y = (left_hip[1] + right_hip[1]) / 2.0
    shoulder_width = abs(left_shoulder[0] - right_shoulder[0])

    torso_length = hip_mid_y - shoulder_mid_y
    head_to_shoulder = shoulder_mid_y - nose_pose[1]

    if torso_length > 0:
        posture_ratio = head_to_shoulder / torso_length
        slumped = _clamp01((POSTURE_UPRIGHT_RATIO - posture_ratio) / POSTURE_RANGE)
    else:
        posture_ratio = 1.0
        slumped = 0.0

    lean_score = _clamp01((0.42 - posture_ratio) / 0.22) if torso_length > 0 else 0.0
    closest_wrist_hip = min(
        abs(left_wrist[1] - hip_mid_y),
        abs(right_wrist[1] - hip_mid_y),
    )
    brace_score = _clamp01(1.0 - closest_wrist_hip / (frame_h * 0.25))
    tripod = lean_score * brace_score

    wrist_delta = abs(left_wrist[1] - right_wrist[1])
    ref_distance = max(shoulder_width, frame_h * 0.1)
    arm_drift = _clamp01(wrist_delta / (ref_distance * ARM_DRIFT_SCALE))

    hand_points = [
        left_wrist, right_wrist,
        left_index, right_index,
        left_thumb, right_thumb,
    ]

    return {
        "slumped_posture": slumped,
        "tripod_position": tripod,
        "arm_drift": arm_drift,
        "hands_near_throat": _hands_near_throat(
            hand_points=hand_points,
            nose_x=nose_pose[0],
            nose_y=nose_pose[1],
            shoulder_mid_y=shoulder_mid_y,
            frame_w=frame_w,
            frame_h=frame_h,
        ),
        "facial_asymmetry": _facial_asymmetry(face_landmarks, frame_w),
        "low_alertness": _low_alertness(face_landmarks),
        "nose_x": nose_pose[0],
        "shoulder_width": shoulder_width,
    }


def process_frame(frame_bgr, pose, face_mesh):
    """Run MediaPipe over one BGR frame and return signal data plus raw results."""
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)

    if not pose_results.pose_landmarks or not face_results.multi_face_landmarks:
        return None, pose_results, face_results

    frame_signals = analyze_landmarks(
        pose_results.pose_landmarks.landmark,
        face_results.multi_face_landmarks[0].landmark,
        frame_bgr.shape,
    )
    return frame_signals, pose_results, face_results


def analyze_face_only(face_landmarks, frame_shape) -> Dict[str, float]:
    """Compute face-only signals when no pose data is available for this person."""
    frame_h, frame_w = frame_shape[:2]
    nose_x = face_landmarks[1].x * frame_w
    return {
        "slumped_posture": 0.0,
        "tripod_position": 0.0,
        "arm_drift": 0.0,
        "hands_near_throat": 0.0,
        "facial_asymmetry": _facial_asymmetry(face_landmarks, frame_w),
        "low_alertness": _low_alertness(face_landmarks),
        "nose_x": nose_x,
    }


def _face_nose_px(face_landmarks, frame_w: int, frame_h: int) -> tuple[float, float]:
    return face_landmarks[1].x * frame_w, face_landmarks[1].y * frame_h


# ---------------------------------------------------------------------------
# Geometric face embedding  (Face ID–inspired)
# Ratios between landmark pairs are scale-/lighting-/expression-invariant.
# ---------------------------------------------------------------------------

_GEO_PAIRS = [
    (33, 263),   # outer eye to outer eye
    (133, 362),  # inner eye to inner eye
    (33, 133),   # left eye width
    (263, 362),  # right eye width
    (10, 152),   # forehead to chin (face height)
    (1, 10),     # nose tip to forehead
    (1, 152),    # nose tip to chin
    (61, 291),   # mouth width
    (129, 358),  # nostril width
    (107, 133),  # left brow inner to left eye inner
    (336, 362),  # right brow inner to right eye inner
    (1, 13),     # nose tip to upper lip
    (13, 14),    # lip opening height
    (70, 300),   # outer brow spread
    (33, 1),     # left outer eye to nose
    (263, 1),    # right outer eye to nose
    (61, 152),   # mouth left to chin
    (291, 152),  # mouth right to chin
]


def _lm_dist(lms, a: int, b: int) -> float:
    return math.hypot(lms[a].x - lms[b].x, lms[a].y - lms[b].y)


def compute_geometric_embedding(face_landmarks) -> np.ndarray | None:
    """Compact face-identity vector from landmark distance ratios.

    All distances are normalised by cheek-to-cheek width so the vector is
    scale-invariant.  Three vertical-position ratios are appended for extra
    discriminative power.  ~21 dimensions, very fast.
    """
    lms = face_landmarks
    fw = _lm_dist(lms, 234, 454)
    if fw < 1e-6:
        return None

    features = [_lm_dist(lms, a, b) / fw for a, b in _GEO_PAIRS]

    face_h = lms[152].y - lms[10].y
    if abs(face_h) > 1e-6:
        features.append((lms[1].y - lms[10].y) / face_h)
        features.append((lms[13].y - lms[10].y) / face_h)
        features.append(((lms[133].y + lms[362].y) / 2 - lms[10].y) / face_h)
    else:
        features.extend([0.5, 0.7, 0.35])

    return np.array(features, dtype=np.float32)


def _face_bbox_px(face_landmarks, frame_w: int, frame_h: int) -> tuple[float, float, float, float]:
    """Tight bounding box from all face mesh landmarks (x0, y0, x1, y1)."""
    xs = [lm.x * frame_w for lm in face_landmarks]
    ys = [lm.y * frame_h for lm in face_landmarks]
    return (min(xs), min(ys), max(xs), max(ys))


def process_frame_multi(frame_bgr, pose, face_mesh):
    """Run MediaPipe for multi-person detection.

    Returns:
        per_face_data: list of {"face_center": (px_x, px_y), "signals": {...}}
        pose_results: raw pose results for drawing
        face_results: raw face results for drawing
    """
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)

    if not face_results.multi_face_landmarks:
        return [], pose_results, face_results

    frame_h, frame_w = frame_bgr.shape[:2]

    pose_nose = None
    if pose_results.pose_landmarks:
        nose_lm = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value]
        pose_nose = (nose_lm.x * frame_w, nose_lm.y * frame_h)

    best_pose_face_idx = -1
    if pose_nose is not None:
        best_dist = float("inf")
        for fi, face_lms in enumerate(face_results.multi_face_landmarks):
            fn = _face_nose_px(face_lms.landmark, frame_w, frame_h)
            dist = math.hypot(fn[0] - pose_nose[0], fn[1] - pose_nose[1])
            if dist < best_dist:
                best_dist = dist
                best_pose_face_idx = fi
        if best_dist > frame_w * 0.15:
            best_pose_face_idx = -1

    raw_faces: list[dict] = []
    for fi, face_lms in enumerate(face_results.multi_face_landmarks):
        face_center = _face_nose_px(face_lms.landmark, frame_w, frame_h)
        face_bbox = _face_bbox_px(face_lms.landmark, frame_w, frame_h)
        geo_emb = compute_geometric_embedding(face_lms.landmark)
        if fi == best_pose_face_idx:
            signals = analyze_landmarks(
                pose_results.pose_landmarks.landmark,
                face_lms.landmark,
                frame_bgr.shape,
            )
        else:
            signals = analyze_face_only(face_lms.landmark, frame_bgr.shape)
        raw_faces.append({
            "face_center": face_center,
            "face_bbox": face_bbox,
            "face_embedding": geo_emb,
            "signals": signals,
        })

    per_face_data = _deduplicate_faces(raw_faces, frame_w)
    return per_face_data, pose_results, face_results


_DEDUP_FRACTION = 0.12


def _deduplicate_faces(faces: list[dict], frame_w: int) -> list[dict]:
    """Merge detections that are too close (same person detected twice)."""
    if len(faces) <= 1:
        return faces
    threshold = frame_w * _DEDUP_FRACTION
    keep: list[dict] = []
    used: set[int] = set()
    for i, face in enumerate(faces):
        if i in used:
            continue
        for j in range(i + 1, len(faces)):
            if j in used:
                continue
            dist = math.hypot(
                face["face_center"][0] - faces[j]["face_center"][0],
                face["face_center"][1] - faces[j]["face_center"][1],
            )
            if dist < threshold:
                used.add(j)
        keep.append(face)
    return keep


# ---------------------------------------------------------------------------
# Multi-person tracker
# ---------------------------------------------------------------------------

def _is_still(centers: list, threshold: float = 8.0) -> bool:
    if len(centers) < 5:
        return False
    recent = centers[-5:]
    xs = [c[0] for c in recent]
    ys = [c[1] for c in recent]
    return float(np.std(xs)) < threshold and float(np.std(ys)) < threshold


class _PersonTrack:
    __slots__ = (
        "person_id", "face_center", "face_bbox", "face_embedding",
        "accumulator", "last_seen", "seen_count", "recent_centers",
    )

    def __init__(self, person_id: int, face_center: tuple[float, float], frame_index: int):
        self.person_id = person_id
        self.face_center = face_center
        self.face_bbox: tuple[float, float, float, float] | None = None
        self.face_embedding = None
        self.accumulator = SignalAccumulator()
        self.last_seen = frame_index
        self.seen_count = 1
        self.recent_centers: list[tuple[float, float]] = [face_center]


class MultiPersonTracker:
    """Track multiple people across frames using face-position matching."""

    MAX_MATCH_FRACTION = 0.35
    STALE_FRAME_LIMIT = 45
    MIN_CONFIRM_FRAMES = 4
    EMA_ALPHA = 0.45

    def __init__(self) -> None:
        self._tracks: dict[int, _PersonTrack] = {}
        self._next_id = 1
        self._frame_w = 640

    def update(self, per_face_data: list[dict], frame_index: int) -> list[dict]:
        """Match detected faces to tracks and update accumulators.

        Returns list of confirmed tracks (seen >= MIN_CONFIRM_FRAMES):
        {"person_id", "face_center", "signals", "seen_count"} sorted by id.
        """
        if per_face_data:
            cx = per_face_data[0]["face_center"][0]
            if cx > 0:
                self._frame_w = max(self._frame_w, cx * 2.5)

        max_dist = self._frame_w * self.MAX_MATCH_FRACTION

        used_tracks: set[int] = set()
        used_faces: set[int] = set()

        pairs: list[tuple[float, int, int]] = []
        for fi, face in enumerate(per_face_data):
            for tid, track in self._tracks.items():
                dist = math.hypot(
                    face["face_center"][0] - track.face_center[0],
                    face["face_center"][1] - track.face_center[1],
                )
                pairs.append((dist, fi, tid))
        pairs.sort()

        for dist, fi, tid in pairs:
            if fi in used_faces or tid in used_tracks:
                continue
            if dist > max_dist:
                continue
            track = self._tracks[tid]
            raw_center = per_face_data[fi]["face_center"]
            a = self.EMA_ALPHA
            track.face_center = (
                a * raw_center[0] + (1 - a) * track.face_center[0],
                a * raw_center[1] + (1 - a) * track.face_center[1],
            )
            track.face_bbox = per_face_data[fi].get("face_bbox")
            emb = per_face_data[fi].get("face_embedding")
            if emb is not None:
                track.face_embedding = emb
            track.recent_centers.append(raw_center)
            if len(track.recent_centers) > 10:
                track.recent_centers = track.recent_centers[-10:]
            track.last_seen = frame_index
            track.seen_count += 1
            track.accumulator.add_frame(per_face_data[fi]["signals"])
            used_faces.add(fi)
            used_tracks.add(tid)

        for fi in range(len(per_face_data)):
            if fi in used_faces:
                continue
            pid = self._next_id
            self._next_id += 1
            face = per_face_data[fi]
            track = _PersonTrack(pid, face["face_center"], frame_index)
            track.face_bbox = face.get("face_bbox")
            track.face_embedding = face.get("face_embedding")
            track.accumulator.add_frame(face["signals"])
            self._tracks[pid] = track

        stale = [
            tid
            for tid, t in self._tracks.items()
            if frame_index - t.last_seen > self.STALE_FRAME_LIMIT
        ]
        for tid in stale:
            del self._tracks[tid]

        return [
            {
                "person_id": tid,
                "face_center": track.face_center,
                "face_bbox": track.face_bbox,
                "face_embedding": track.face_embedding,
                "signals": track.accumulator.current_signals(),
                "seen_count": track.seen_count,
                "is_still": _is_still(track.recent_centers),
            }
            for tid, track in sorted(self._tracks.items())
            if track.seen_count >= self.MIN_CONFIRM_FRAMES
        ]

    def get_all_final(self) -> list[dict]:
        """Get final signals for all confirmed tracked people."""
        return [
            {
                "person_id": tid,
                "face_center": t.face_center,
                "face_bbox": t.face_bbox,
                "face_embedding": t.face_embedding,
                "signals": t.accumulator.final_signals(),
            }
            for tid, t in sorted(self._tracks.items())
            if t.seen_count >= self.MIN_CONFIRM_FRAMES
        ]

    def get_person_final_and_reset(self, person_id: int) -> dict | None:
        """Get final signals for one person and reset their accumulator."""
        track = self._tracks.get(person_id)
        if track is None:
            return None
        signals = track.accumulator.final_signals()
        track.accumulator.reset()
        return signals

    @property
    def track_count(self) -> int:
        return sum(
            1 for t in self._tracks.values()
            if t.seen_count >= self.MIN_CONFIRM_FRAMES
        )


def extract_signals(video_path: str) -> Dict[str, float]:
    """Extract distress-related signals from a video clip."""
    capture = cv2.VideoCapture(video_path)
    zeroed = zero_signals()
    if not capture.isOpened():
        return {**zeroed, "_valid_frames": 0}

    accumulator = SignalAccumulator()
    frame_index = 0

    with create_pose_estimator() as pose, create_face_mesh_estimator() as face_mesh:
        while True:
            success, frame = capture.read()
            if not success:
                break

            frame_index += 1
            if frame_index % FRAME_SAMPLE_RATE != 0:
                continue

            frame_signals, _, _ = process_frame(frame, pose, face_mesh)
            if frame_signals is None:
                continue
            accumulator.add_frame(frame_signals)

    capture.release()

    if accumulator.valid_frames < MIN_VALID_FRAMES:
        return {**zeroed, "_valid_frames": accumulator.valid_frames}
    return accumulator.final_signals()
