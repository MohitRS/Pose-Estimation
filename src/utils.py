# src/utils.py
import cv2
import mediapipe as mp

def convert_landmarks_to_array(landmarks):
    return [[landmark.x, landmark.y] for landmark in landmarks]

def draw_landmarks(image, results):
    mp_pose = mp.solutions.pose
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
        )
    return image
