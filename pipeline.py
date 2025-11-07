# --------------------------------------------------------------
#  src/pipeline_mediapipe.py
# --------------------------------------------------------------
import cv2, csv, os, time, matplotlib.pyplot as plt, numpy as np
import mediapipe as mp
from collections import Counter
from mediapipe.framework.formats import landmark_pb2

# ================= CONFIG =================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

INPUT_VIDEO   = os.path.join(PROJECT_ROOT, "data", "net_session.mp4")
OUTPUT_DIR    = os.path.join(PROJECT_ROOT, "CRICKET_ANALYSIS_RESULTS")
CSV_OUT       = os.path.join(OUTPUT_DIR, "batsman_stance.csv")
PLOT_OUT      = os.path.join(OUTPUT_DIR, "stance_plot.png")
VIDEO_OUT     = os.path.join(OUTPUT_DIR, "batsman_labeled.mp4")

# 🟨 ROI (fixed region focusing on batsman only)
ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 0.46, 0.54, 0.56, 0.65

# Increased smoothing for flicker reduction
FRAME_HISTORY = 25
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= MediaPipe SETUP =================
print("Loading MediaPipe Pose...")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.4,   # smoother tracking
)

# ================= VIDEO INPUT =================
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video at:\n   {INPUT_VIDEO}")

fps   = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_vid = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (width, height))
print(f"Video: {width}×{height} @ {fps:.1f} fps – {total_frames} frames")

# ================= HELPERS =================
def remap(landmarks, crop_x, crop_y, crop_w, crop_h):
    """Map ROI coordinates back to full-frame coordinates."""
    mapped = []
    for lm in landmarks:
        mapped.append(landmark_pb2.NormalizedLandmark(
            x=(lm.x * crop_w + crop_x) / width,
            y=(lm.y * crop_h + crop_y) / height,
            z=lm.z,
            visibility=getattr(lm, 'visibility', 1.0)
        ))
    return mapped


def get_stance(landmarks):
    """Improved stance classification with stability buffer for right-handers."""
    try:
        # Important key points for stance detection
        ls, rs, lw, rw, lh, rh, lk, rk, la, ra = [landmarks[i] for i in [11,12,15,16,23,24,25,26,27,28]]
        score = {'right':0, 'left':0}

        # Shoulder alignment — with tolerance to reduce flicker
        if (rs.x - ls.x) > 0.03: score['right'] += 2
        elif (ls.x - rs.x) > 0.03: score['left'] += 2

        # Hand vertical position (lower wrist = top hand)
        if (lw.y - rw.y) > 0.02: score['right'] += 2
        elif (rw.y - lw.y) > 0.02: score['left'] += 2

        # Hip alignment
        if (rh.x - lh.x) > 0.02: score['right'] += 1
        elif (lh.x - rh.x) > 0.02: score['left'] += 1

        # Leg/ankle alignment
        if (ra.x - la.x) < -0.02: score['right'] += 1
        elif (la.x - ra.x) < -0.02: score['left'] += 1

        # Confidence buffer
        diff = abs(score['right'] - score['left'])
        if diff < 2:
            return None

        return "Right-Handed" if score['right'] > score['left'] else "Left-Handed"
    except Exception:
        return None


# ================= MAIN LOOP =================
recent_stances, results = [], []
frame_idx = 0
start_time = time.time()
mp_drawing = mp.solutions.drawing_utils

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ---- ROI crop ----
    x1 = int(ROI_X1 * width)
    y1 = int(ROI_Y1 * height)
    x2 = int(ROI_X2 * width)
    y2 = int(ROI_Y2 * height)
    roi_rgb = rgb[y1:y2, x1:x2]

    # ---- Pose Detection ----
    res = pose.process(roi_rgb)
    stance = None

    if res.pose_landmarks:
        mapped = remap(res.pose_landmarks.landmark, x1, y1, x2-x1, y2-y1)
        stance = get_stance(mapped)

        # 🟡 Draw landmarks (yellow dots)
        mp_drawing.draw_landmarks(
            frame,
            landmark_pb2.NormalizedLandmarkList(landmark=mapped),
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,255), thickness=1)
        )

    # ---- Smoothing ----
    recent_stances.append(stance)
    if len(recent_stances) > FRAME_HISTORY:
        recent_stances.pop(0)
    final_stance = Counter([s for s in recent_stances if s]).most_common(1)
    final_stance = final_stance[0][0] if final_stance else None

    # ---- Draw ROI ----
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 🟢 Only show text if confident
    if final_stance:
        color = (0,255,0) if "Right" in final_stance else (0,0,255)
        cv2.putText(frame, f"BATSMAN: {final_stance}",
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    out_vid.write(frame)
    results.append((round(frame_idx/fps, 1), final_stance or ""))

    if frame_idx % 600 == 0:
        elapsed = time.time() - start_time
        print(f"Processed {frame_idx}/{total_frames} "
              f"({frame_idx/total_frames*100:.1f}%) – {frame_idx/elapsed:.2f} fps")

cap.release()
out_vid.release()

# ================= SAVE CSV =================
with open(CSV_OUT, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(["time_sec", "stance"])
    w.writerows(results)
print("\nCSV →", CSV_OUT)

# ================= PLOT =================
if results:
    times, numeric = zip(*[(t, 1 if "Right" in s else 0) for t, s in results if s])
    plt.figure(figsize=(12,4))
    plt.plot(times, numeric, color='green', linewidth=1.5)
    plt.yticks([0,1], ["Left-Handed","Right-Handed"])
    plt.xlabel("Time (s)")
    plt.title("Batsman Stance Detection Timeline")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=150)
    plt.close()
    print("Plot →", PLOT_OUT)

print(f"\nFinished – {len(results)} frames processed")
