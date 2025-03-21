# src/gui.py
import sys
import json
import base64
import os
import cv2
import numpy as np
from PyQt6.QtCore import QProcess, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QLabel, QPushButton, QHBoxLayout
)

# Use the current Python executable.
PYTHON_EXECUTABLE = sys.executable

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mediapipe Pose Estimation")
        self.setGeometry(100, 100, 1280, 720)
        self._setup_ui()
        
        self.worker_process = QProcess(self)
        self.worker_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.worker_process.readyReadStandardOutput.connect(self.handle_worker_output)
        self.is_running = False
        self.recorded_frames = []  # List to store processed frames (as numpy arrays)
        self.fps = 30  # Assumed frame rate
        self.frame_width = None
        self.frame_height = None
    
    def _setup_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Video display label.
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label)

        # Control buttons.
        btn_layout = QHBoxLayout()
        self.control_btn = QPushButton("Start Analysis", self)
        self.control_btn.clicked.connect(self.toggle_analysis)
        btn_layout.addWidget(self.control_btn)

        self.download_btn = QPushButton("Download Processed Video", self)
        self.download_btn.clicked.connect(self.download_video)
        btn_layout.addWidget(self.download_btn)

        layout.addLayout(btn_layout)

        # Status label for analysis information.
        self.status_label = QLabel("Status: Ready", self)
        layout.addWidget(self.status_label)
    
    def toggle_analysis(self):
        if not self.is_running:
            video_path = r"C:\Users\mohit\Downloads\C0069.mp4"  # Update the path if needed.
            if not os.path.exists(video_path):
                self.status_label.setText("Error: Video file not found!")
                return
            # Reset recorded frames for this analysis run.
            self.recorded_frames = []
            self.worker_process.start(PYTHON_EXECUTABLE, ["-m", "src.pose_estimation_worker", video_path])
            if not self.worker_process.waitForStarted(3000):
                self.status_label.setText("Error: Worker process failed to start.")
                return
            self.control_btn.setText("Stop Analysis")
            self.is_running = True
        else:
            self.worker_process.kill()
            self.control_btn.setText("Start Analysis")
            self.is_running = False
    
    def handle_worker_output(self):
        data = self.worker_process.readAllStandardOutput()
        text = bytes(data).decode("utf-8").strip()
        if not text:
            return
        # Debug: print raw output.
        print("Worker output:", text)
        for line in text.splitlines():
            try:
                output = json.loads(line)
                self.status_label.setText(
                    f"Frame: {output.get('frame')} | Pose: {output.get('pose')} | Wrist Speed: {output.get('wrist_speed'):.2f}"
                )
                if output.get("frame_data"):
                    # Decode image data for display.
                    img_data = base64.b64decode(output["frame_data"])
                    qt_image = QImage.fromData(img_data, "JPG")
                    self.video_label.setPixmap(QPixmap.fromImage(qt_image))
                    
                    # Also decode to a numpy array and store for video saving.
                    nparr = np.frombuffer(base64.b64decode(output["frame_data"]), np.uint8)
                    frame_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    self.recorded_frames.append(frame_np)
                    # Record frame dimensions if not already set.
                    if self.frame_width is None or self.frame_height is None:
                        self.frame_height, self.frame_width = frame_np.shape[:2]
            except Exception as e:
                print("Error parsing worker output:", e)
    
    def download_video(self):
        if not self.recorded_frames:
            self.status_label.setText("No video recorded yet!")
            return
        output_filename = "processed_output.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_filename, fourcc, self.fps, (self.frame_width, self.frame_height))
        for frame in self.recorded_frames:
            out.write(frame)
        out.release()
        self.status_label.setText(f"Video saved as {output_filename}")
    
    def closeEvent(self, event):
        if self.worker_process.state() != QProcess.ProcessState.NotRunning:
            self.worker_process.kill()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
