# Reads video + CSVs from stage 1, then tracks a single rat with a Kalman filter and saves MP4
# Outputs the kalman tracked video file with blue dots representing the prediction of the filter
# green dot represeting the measurement and red dot and consequent red track represeting the tracking
# Note: This file should be run after outputs from stage1_mog2_relations.py have been generated

import cv2
import csv
import numpy as np
from collections import defaultdict, deque

# Inputs from previous stage 1
video_path       = r"../ES30_1_13_24_cropped.mp4"
blobs_csv_path   = "ES30_1_13_24_cropped_blobs.csv"
relations_path   = "ES30_1_13_24_cropped_relations.csv"
out_kalman_mp4   = "ES30_1_13_24_cropped_kalman.mp4"

# Kalman selection params
kf_base_R        = 20.0       # base measurement noise (px), scaled by overlap proxy
kf_Q_pos_acc     = 1e-1
kf_Q_vel_acc     = 1e-1
init_box_wh      = 60
pred_box_alpha   = 0.25       # EMA for w/h
kalman_gate_norm = 0.60       # gate measurement by normalized distance

# helper functions
def make_mp4_writer(path, fps, size):
    W, H = size
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    w = cv2.VideoWriter(path, fourcc, fps, (W, H))
    if w.isOpened(): return w
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(path, fourcc, fps, (W, H))
    if w.isOpened(): return w
    raise RuntimeError("Could not open MP4 writer.")

def norm_distance(p, q, W, H):
    return float(np.hypot((p[0]-q[0]) / max(1.0, W), (p[1]-q[1]) / max(1.0, H)))

def bbox_intersection(ax, ay, aw, ah, bx, by, bw, bh):
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    return iw * ih

def overlap_ratio(prev_stat, curr_stat):
    x1, y1, w1, h1, a1 = prev_stat
    x2, y2, w2, h2, a2 = curr_stat
    inter = bbox_intersection(x1, y1, w1, h1, x2, y2, w2, h2)
    if inter <= 0: return 0.0
    denom = float(min(a1, a2)) if min(a1, a2) > 0 else 1.0
    return inter / denom

# Kalman: state [x, y, vx, vy], measurements [x, y]
class RatKF:
    def __init__(self, dt=1.0, init_box=60):
        self.kf = cv2.KalmanFilter(4, 2, 0)
        self.kf.transitionMatrix = np.array([[1,0,dt,0],
                                             [0,1,0,dt],
                                             [0,0,1 ,0],
                                             [0,0,0 ,1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0],
                                              [0,1,0,0]], dtype=np.float32)
        q = np.array([[0.25*dt**4, 0, 0.5*dt**3, 0],
                      [0, 0.25*dt**4, 0, 0.5*dt**3],
                      [0.5*dt**3, 0, dt**2, 0],
                      [0, 0.5*dt**3, 0, dt**2]], dtype=np.float32)
        self.kf.processNoiseCov = (kf_Q_pos_acc * q + kf_Q_vel_acc * np.eye(4, dtype=np.float32))
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 50.0
        self.kf.statePost = np.zeros((4,1), dtype=np.float32)

        self.initialized = False
        self.box_w = float(init_box)
        self.box_h = float(init_box)
        self.trail = deque(maxlen=128)

    def init_from_detection(self, cx, cy, w=None, h=None):
        self.kf.statePost[:] = np.array([[cx],[cy],[0.0],[0.0]], dtype=np.float32)
        if w: self.box_w = float(w)
        if h: self.box_h = float(h)
        self.initialized = True
        self.trail.clear()
        self.trail.append((int(cx), int(cy)))

    def predict(self):
        pred = self.kf.predict()
        return float(pred[0,0]), float(pred[1,0])

    def correct(self, mx, my, ov_proxy=1.0):
        ov = max(1e-3, min(1.0, ov_proxy))
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * (kf_base_R / ov)
        meas = np.array([[mx],[my]], dtype=np.float32)
        corr = self.kf.correct(meas)
        cx, cy = float(corr[0,0]), float(corr[1,0])
        self.trail.append((int(cx), int(cy)))
        return cx, cy

# loading the csvs
def load_blobs_csv(path):
    # returns: blobs[frame] = {blob_id: {"area":..., "bbox":(x,y,w,h,a), "cent":(cx,cy)}}
    blobs = defaultdict(dict)
    with open(path, "r", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            fr = int(r["frame"]); bid = int(r["blob_id"])
            x = int(r["bbox_x"]); y = int(r["bbox_y"])
            w = int(r["bbox_w"]); h = int(r["bbox_h"]); a = int(r["area"])
            cx = float(r["centroid_x"]); cy = float(r["centroid_y"])
            blobs[fr][bid] = {"area": a, "bbox": (x,y,w,h,a), "cent": (cx,cy)}
    return blobs

def load_relations(path):
    # returns: pairs[frame] = dict(prev_id -> curr_id), plus appear/vanish if needed
    pairs = defaultdict(dict)
    with open(path, "r", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            if r["relation"] != "pair": continue
            fr = int(r["frame"])
            val = r["value"].strip()     # like "12->7"
            if "->" in val:
                p, c = val.split("->")
                try:
                    pid, cid = int(p.strip(" []")), int(c.strip(" []"))
                    pairs[fr][pid] = cid
                except:  # robust to formatting
                    pass
    return pairs

# the main function
def main():
    # Load CSVs
    blobs = load_blobs_csv(blobs_csv_path)
    pairs = load_relations(relations_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (W, H)
    writer = make_mp4_writer(out_kalman_mp4, fps, size)

    kf = RatKF(dt=1.0, init_box=init_box_wh)

    # Selection state: the blob ID are we following
    prev_chosen_id = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_blobs = blobs.get(frame_idx, {})
        kf_frame = frame.copy()

        # predict
        px, py = kf.predict()

        # Choose a measurement:
        chosen_id = None
        meas = None   # (cx,cy)
        meas_box = None  # (x,y,w,h,a)

        if not kf.initialized:
            # initialize from largest blob (by area)
            if frame_blobs:
                chosen_id = max(frame_blobs.keys(), key=lambda bid: frame_blobs[bid]["area"])
                d = frame_blobs[chosen_id]
                cx, cy = d["cent"]; x,y,w,h,a = d["bbox"]
                kf.init_from_detection(cx, cy, w, h)
                px, py = cx, cy
                meas = (cx, cy); meas_box = (x,y,w,h,a)
        else:
            # if we had a chosen id in prev frame, follow its link in pairs[frame_idx]
            if prev_chosen_id is not None and frame_idx in pairs and prev_chosen_id in pairs[frame_idx]:
                cid = pairs[frame_idx][prev_chosen_id]
                if cid in frame_blobs:
                    chosen_id = cid
            # fallback: pick blob nearest to prediction (within gate), else largest
            if chosen_id is None and frame_blobs:
                # nearest to (px,py)
                nearest = sorted(frame_blobs.items(),
                                 key=lambda kv: norm_distance((px,py), kv[1]["cent"], W, H))
                if nearest and norm_distance((px,py), nearest[0][1]["cent"], W, H) <= kalman_gate_norm:
                    chosen_id = nearest[0][0]
                else:
                    chosen_id = max(frame_blobs.keys(), key=lambda bid: frame_blobs[bid]["area"])

            if chosen_id is not None:
                d = frame_blobs[chosen_id]
                meas = d["cent"]; meas_box = d["bbox"]
                # smooth box size
                kf.box_w = (1 - pred_box_alpha) * kf.box_w + pred_box_alpha * meas_box[2]
                kf.box_h = (1 - pred_box_alpha) * kf.box_h + pred_box_alpha * meas_box[3]

        # draw prediction in blue
        cv2.circle(kf_frame, (int(px), int(py)), 6, (255, 0, 0), -1)

        # Build predicted box for overlap proxy
        pb_w = max(10.0, kf.box_w)
        pb_h = max(10.0, kf.box_h)
        pbx = int(px - pb_w/2); pby = int(py - pb_h/2)
        pred_box = (pbx, pby, int(pb_w), int(pb_h), int(pb_w*pb_h))
        cv2.rectangle(kf_frame, (pred_box[0], pred_box[1]),
                      (pred_box[0]+pred_box[2], pred_box[1]+pred_box[3]),
                      (255,200,0), 2)

        # Correct KF if we have a measurement
        if kf.initialized and meas is not None:
            mx, my = meas
            # overlap proxy for adaptive R
            ov = overlap_ratio(pred_box, meas_box) if meas_box is not None else 1.0
            cx_corr, cy_corr = kf.correct(mx, my, ov_proxy=max(0.01, ov))

            # draw measurement (green) & corrected (red)
            cv2.circle(kf_frame, (int(mx), int(my)), 6, (0, 220, 0), -1)
            cv2.circle(kf_frame, (int(cx_corr), int(cy_corr)), 7, (0, 0, 255), -1)

            # keep which ID we used
            prev_chosen_id = chosen_id
        else:
            # if initialized but no measurement accepted this frame, keep last id
            pass

        # Draw KF trail
        if kf.trail and len(kf.trail) >= 2:
            for i in range(1, len(kf.trail)):
                cv2.line(kf_frame, kf.trail[i-1], kf.trail[i], (0,0,255), 2)

        writer.write(kf_frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print("[DONE] Wrote:", out_kalman_mp4)

if __name__ == "__main__":
    main()
