# src/data_logger.py
import csv
from datetime import datetime

class DataLogger:
    def __init__(self, filename='data/tennis_pose_log.csv'):
        self.filename = filename
        self._initialize_file()
    
    def _initialize_file(self):
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'pose', 'wrist_speed', 'landmarks'])

    def log_data(self, pose, speed, landmarks):
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            landmarks_str = ';'.join([f"{lm.x},{lm.y}" for lm in landmarks])
            writer.writerow([datetime.now().isoformat(), pose, speed, landmarks_str])
