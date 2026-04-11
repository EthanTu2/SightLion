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

SLUMPED_RATIO_THRESHOLD = 0.85
BODY_SWAY_MAX_STD_PX = 50.0
TRIPOD_Y_TOLERANCE_RATIO = 0.10
ARM_DRIFT_THRESHOLD_RATIO = 0.15
THROAT_Y_RANGE_RATIO = 0.15
THROAT_X_RANGE_RATIO = 0.20
THROAT_FRAME_FRACTION_THRESHOLD = 0.30
FACIAL_ASYMMETRY_FACE_WIDTH_RATIO = 0.05
LOW_ALERTNESS_EAR_THRESHOLD = 0.20

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
        refine_landmarks=False,
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


def _eye_closed(face_landmarks) -> float:
    upper = _face_xy(face_landmarks, 159)
    lower = _face_xy(face_landmarks, 145)
    outer = _face_xy(face_landmarks, 33)
    inner = _face_xy(face_landmarks, 133)

    horizontal_distance = _euclidean(outer, inner)
    if horizontal_distance == 0:
        return 0.0

    eye_aspect_ratio = _euclidean(upper, lower) / horizontal_distance
    return 1.0 if eye_aspect_ratio < LOW_ALERTNESS_EAR_THRESHOLD else 0.0


def _hands_near_throat(
    left_wrist: tuple[float, float],
    right_wrist: tuple[float, float],
    nose_x: float,
    nose_y: float,
    frame_w: int,
    frame_h: int,
) -> float:
    min_x = nose_x - (frame_w * THROAT_X_RANGE_RATIO)
    max_x = nose_x + (frame_w * THROAT_X_RANGE_RATIO)
    min_y = nose_y
    max_y = nose_y + (frame_h * THROAT_Y_RANGE_RATIO)

    for wrist_x, wrist_y in (left_wrist, right_wrist):
        if min_x <= wrist_x <= max_x and min_y <= wrist_y <= max_y:
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
    left_knee = _pose_px(pose_landmarks, mp_pose.PoseLandmark.LEFT_KNEE.value, frame_w, frame_h)
    right_knee = _pose_px(pose_landmarks, mp_pose.PoseLandmark.RIGHT_KNEE.value, frame_w, frame_h)
    nose_pose = _pose_px(pose_landmarks, mp_pose.PoseLandmark.NOSE.value, frame_w, frame_h)

    shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
    hip_mid_y = (left_hip[1] + right_hip[1]) / 2.0
    knee_tolerance = frame_h * TRIPOD_Y_TOLERANCE_RATIO
    wrist_delta_ratio = abs(left_wrist[1] - right_wrist[1]) / max(frame_h, 1)

    return {
        "slumped_posture": 1.0 if shoulder_mid_y > hip_mid_y * SLUMPED_RATIO_THRESHOLD else 0.0,
        "tripod_position": 1.0
        if abs(left_wrist[1] - left_knee[1]) <= knee_tolerance
        and abs(right_wrist[1] - right_knee[1]) <= knee_tolerance
        else 0.0,
        "arm_drift": 1.0 if wrist_delta_ratio > ARM_DRIFT_THRESHOLD_RATIO else 0.0,
        "hands_near_throat": _hands_near_throat(
            left_wrist=left_wrist,
            right_wrist=right_wrist,
            nose_x=nose_pose[0],
            nose_y=nose_pose[1],
            frame_w=frame_w,
            frame_h=frame_h,
        ),
        "facial_asymmetry": _facial_asymmetry(face_landmarks, frame_w),
        "low_alertness": _eye_closed(face_landmarks),
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
