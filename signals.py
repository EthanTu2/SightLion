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
BODY_SWAY_MAX_STD_PX = 50.0
ARM_DRIFT_SCALE = 0.30
THROAT_Y_RANGE_RATIO = 0.15
THROAT_X_RANGE_RATIO = 0.25
THROAT_FRAME_FRACTION_THRESHOLD = 0.30
FACIAL_ASYMMETRY_FACE_WIDTH_RATIO = 0.05
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


def create_face_mesh_estimator():
    return mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


class SignalAccumulator:
    """Accumulate frame-level signals into running averages."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.valid_frames = 0
        self.nose_positions_x: list[float] = []
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

        if len(self.nose_positions_x) > 1:
            signals["body_sway"] = _clamp01(float(np.std(self.nose_positions_x)) / BODY_SWAY_MAX_STD_PX)
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
