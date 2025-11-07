# SPORTSENGAGE
# 🏏 Cricket Batsman Stance Detection (MediaPipe Pose)

This project detects whether a batsman is **Right-Handed** or **Left-Handed** from a cricket video using **MediaPipe Pose** and **OpenCV**. It crops a fixed region (ROI) around the batsman, detects pose landmarks, classifies stance, smooths flicker across frames, and outputs a labeled video, CSV, and stance timeline plot.

---

## ⚙️ Setup

```bash
# 1️⃣ Install dependencies
pip install opencv-python mediapipe matplotlib numpy

# 2️⃣ Place your input video
# Example: data/net_session.mp4

# 3️⃣ Run the script
python src/pipeline_mediapipe.py

How It Works

ROI Crop: Focus only on batsman using predefined normalized coordinates.
Pose Estimation: Uses MediaPipe’s lightweight Pose model.
Landmark Mapping: Converts ROI keypoints to full-frame coordinates.
Stance Detection: Compares shoulders, wrists, hips, and ankles to infer handedness.
Temporal Smoothing: Uses a 25-frame history to reduce flicker.
Visualization: Draws skeleton and stance label on each frame.

Exports:
batsman_labeled.mp4 → annotated video
batsman_stance.csv → timestamped stance log
stance_plot.png → stance timeline plot

Terminal Log Example:

Loading MediaPipe Pose...
Video: 1280×720 @ 30.0 fps – 1800 frames
Processed 600/1800 (33.3%) – 22.4 fps
Processed 1200/1800 (66.6%) – 23.1 fps
CSV → CRICKET_ANALYSIS_RESULTS/batsman_stance.csv
Plot → CRICKET_ANALYSIS_RESULTS/stance_plot.png
Finished – 1800 frames processed


Notes

Adjust ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 in the script for different camera angles.

Increase model_complexity for higher accuracy if needed.

Works best for side-view cricket footage with visible batsman posture.



Tech: MediaPipe · OpenCV · NumPy · Matplotlib
