# src/pose_estimation_worker.py
import sys
import cv2
import time
import json
import base64
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.pose_estimation_worker <video_path>")
        sys.exit(1)
    video_path = sys.argv[1]
    
    # Delayed import of mediapipe
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Define allowed joints (only the key tennis joints):
    allowed_indices = {
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_ELBOW.value,
        mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        mp_pose.PoseLandmark.LEFT_WRIST.value,
        mp_pose.PoseLandmark.RIGHT_WRIST.value,
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.LEFT_KNEE.value,
        mp_pose.PoseLandmark.RIGHT_KNEE.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value,
        mp_pose.PoseLandmark.RIGHT_ANKLE.value
    }
    # Filter connections so that only connections between allowed joints are drawn.
    allowed_connections = set()
    for connection in mp_pose.POSE_CONNECTIONS:
        if connection[0] in allowed_indices and connection[1] in allowed_indices:
            allowed_connections.add(connection)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(json.dumps({"error": "Failed to open video"}), flush=True)
        sys.exit(1)
    
    # Limit processing to ~10 seconds (assume 30 fps = 300 frames)
    max_frames = 300
    prev_landmarks = None
    frame_no = 0
    
    while cap.isOpened() and frame_no < max_frames:
        ret, frame = cap.read()
        if not ret:
            # If video ends before 10 seconds, break out.
            break
        frame_no += 1
        
        height, width, _ = frame.shape
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        # Prepare output dictionary.
        output = {"frame": frame_no, "pose": "Unknown", "wrist_speed": 0.0, "frame_data": ""}
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # Draw only the allowed joints.
            for idx, lm in enumerate(landmarks):
                if idx in allowed_indices:
                    x, y = int(lm.x * width), int(lm.y * height)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            for connection in allowed_connections:
                a, b = connection
                lm_a = landmarks[a]
                lm_b = landmarks[b]
                x_a, y_a = int(lm_a.x * width), int(lm_a.y * height)
                x_b, y_b = int(lm_b.x * width), int(lm_b.y * height)
                cv2.line(frame, (x_a, y_a), (x_b, y_b), (0, 255, 0), 2)
            
            # Simple classification: if right wrist is above right shoulder, mark as "Serve"
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            if right_wrist.y < right_shoulder.y:
                output["pose"] = "Serve"
            # Calculate wrist speed.
            if prev_landmarks:
                prev_wrist = prev_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                speed = np.linalg.norm(
                    np.array([right_wrist.x, right_wrist.y]) - np.array([prev_wrist.x, prev_wrist.y])
                )
                output["wrist_speed"] = speed
            prev_landmarks = landmarks
        
        # Encode the processed frame to JPEG then to base64.
        ret2, buffer = cv2.imencode('.jpg', frame)
        if ret2:
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            output["frame_data"] = jpg_as_text
        
        print(json.dumps(output), flush=True)
        time.sleep(0.03)
    
    cap.release()

if __name__ == '__main__':
    main()
