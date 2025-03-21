# src/pose_estimator.py
import cv2
import numpy as np

class MediapipePoseEstimator:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        import mediapipe as mp  # Delayed import here
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.prev_landmarks = None

    def estimate_pose(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        return results

    def classify_pose(self, landmarks):
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        if right_wrist.y < right_shoulder.y:
            return "Serve"
        return "Unknown"

    def calculate_speed(self, current, previous):
        if previous is None:
            return 0.0
        return np.linalg.norm(np.array([current.x, current.y]) - np.array([previous.x, previous.y]))
