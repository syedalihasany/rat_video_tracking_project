# stage1_mog2_relations.py
# MOG2 segmentation + greedy overlap associations + CSVs + overlay MP4

import cv2
import csv
import numpy as np

# ------------- USER INPUTS -------------
video_path = r"../ES30_1_13_24_cropped.mp4"
out_prefix = "ES30_1_13_24_cropped"

min_area_px   = 300         # ignore tiny specks
kernel_size   = 10           # morphology kernel
history       = 600         # MOG2 memory
varThreshold  = 25          # higher = less sensitive
detectShadows = False
warmup_frames = 200         # warmup MOG2, no outputs

# Association gates (greedy max-overlap)
overlap_gate   = 0.00       # allow 0 to be permissive
dist_gate_norm = 0.25       # normalized centroid distance gate

# ------------- OUTPUTS -------------
out_overlay_mp4   = f"{out_prefix}_overlay.mp4"
out_blobs_csv     = f"{out_prefix}_blobs.csv"
out_relations_csv = f"{out_prefix}_relations.csv"

# ------------- HELPERS -------------
def make_mp4_writer(path, fps, size):
    W, H = size
    # Try H.264
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    w = cv2.VideoWriter(path, fourcc, fps, (W, H))
    if w.isOpened():
        return w
    # Fallback to mp4v
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(path, fourcc, fps, (W, H))
    if w.isOpened():
        return w
    raise RuntimeError("Could not open MP4 writer (avc1/mp4v).")

def bbox_intersection(ax, ay, aw, ah, bx, by, bw, bh):
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    return iw * ih

def overlap_ratio(prev_stat, curr_stat):
    # overlap / min(area_prev, area_curr) — robust to splits/merges
    x1, y1, w1, h1, a1 = prev_stat
    x2, y2, w2, h2, a2 = curr_stat
    inter = bbox_intersection(x1, y1, w1, h1, x2, y2, w2, h2)
    if inter <= 0:
        return 0.0
    denom = float(min(a1, a2)) if min(a1, a2) > 0 else 1.0
    return inter / denom

def norm_distance(p, q, W, H):
    return float(np.hypot((p[0]-q[0]) / max(1.0, W), (p[1]-q[1]) / max(1.0, H)))

# ------------- MAIN -------------
def main():
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (W, H)

    # Background model and warmup
    fgbg = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold, detectShadows=detectShadows)
    for _ in range(warmup_frames):
        ret, f = cap.read()
        if not ret: break
        fgbg.apply(f)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    writer = make_mp4_writer(out_overlay_mp4, fps, size)

    # CSVs
    blobs_f = open(out_blobs_csv, "w", newline="")
    blobs_wr = csv.writer(blobs_f)
    # Per-blob schema (IDs are per-frame CC labels)
    blobs_wr.writerow(["frame","blob_id","area","bbox_x","bbox_y","bbox_w","bbox_h","centroid_x","centroid_y"])

    rel_f = open(out_relations_csv, "w", newline="")
    rel_wr = csv.writer(rel_f)
    # Relations schema
    rel_wr.writerow(["frame","relation","value","cost"])  # value examples: "12->7", "[3,4]->6", "appear:8", "vanish:5"

    prev_stats = None
    prev_cents = None
    prev_ids   = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- segmentation ---
        fg = fgbg.apply(frame)
        fg = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)[1]
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  kernel, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)

        # --- components ---
        num_labels, labels, stats, cents = cv2.connectedComponentsWithStats(fg)
        curr_ids = [L for L in range(1, num_labels) if stats[L][4] >= min_area_px]

        # draw + write blobs
        overlay = frame.copy()
        for L in curr_ids:
            x,y,w,h,a = map(int, stats[L])
            cx, cy = cents[L]
            cv2.rectangle(overlay, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.circle(overlay, (int(cx),int(cy)), 3, (0,0,255), -1)
            blobs_wr.writerow([frame_idx, L, a, x, y, w, h, float(cx), float(cy)])

        # --- relations (GREEDY max-overlap, current→best previous) ---
        if prev_stats is not None and len(prev_ids) > 0:
            curr_best_prev = {}   # curr Lc -> chosen prev Lp
            used_prev = set()
            used_curr = set()

            for Lc in curr_ids:
                x2,y2,w2,h2,a2 = stats[Lc]
                cx2,cy2 = cents[Lc]
                best_prev = None
                best_ov   = -1.0
                for Lp in prev_ids:
                    x1,y1,w1,h1,a1 = prev_stats[Lp]
                    cx1,cy1 = prev_cents[Lp]
                    # optional distance gate
                    if norm_distance((cx1,cy1), (cx2,cy2), W, H) > dist_gate_norm:
                        continue
                    ov = overlap_ratio(prev_stats[Lp], stats[Lc])
                    if ov > best_ov:
                        best_ov, best_prev = ov, Lp
                if best_prev is not None and best_ov >= overlap_gate:
                    if best_prev not in used_prev and Lc not in used_curr:
                        curr_best_prev[Lc] = best_prev
                        used_prev.add(best_prev)
                        used_curr.add(Lc)
                        # draw match line
                        p = (int(prev_cents[best_prev][0]), int(prev_cents[best_prev][1]))
                        q = (int(cents[Lc][0]), int(cents[Lc][1]))
                        cv2.line(overlay, p, q, (255,200,0), 2)
                        rel_wr.writerow([frame_idx, "pair", f"{best_prev}->{Lc}", f"{1.0 - best_ov:.6f}"])

            # splits: one previous to many current (check overlaps > 0)
            prev_to_currs = {Lp: [] for Lp in prev_ids}
            for Lc in curr_ids:
                for Lp in prev_ids:
                    if bbox_intersection(*prev_stats[Lp][:4], *stats[Lc][:4]) > 0:
                        prev_to_currs[Lp].append(Lc)
            for Lp, lst in prev_to_currs.items():
                u = sorted(set(lst))
                if len(u) >= 2:
                    rel_wr.writerow([frame_idx, "split", f"{Lp}->{u}", ""])

            # merges: many previous to one current
            curr_to_prevs = {}
            for Lc, Lp in curr_best_prev.items():
                curr_to_prevs.setdefault(Lc, []).append(Lp)
            for Lc, lst in curr_to_prevs.items():
                u = sorted(set(lst))
                if len(u) >= 2:
                    rel_wr.writerow([frame_idx, "merge", f"{u}->{Lc}", ""])

            # appeared / vanished
            matched_prev = set(curr_best_prev.values())
            matched_curr = set(curr_best_prev.keys())
            appeared = [Lc for Lc in curr_ids  if Lc not in matched_curr]
            vanished = [Lp for Lp in prev_ids if Lp not in matched_prev]
            for Lc in appeared:
                rel_wr.writerow([frame_idx, "appear", f"{Lc}", ""])
            for Lp in vanished:
                rel_wr.writerow([frame_idx, "vanish", f"{Lp}", ""])

        writer.write(overlay)

        # slide
        prev_stats = stats
        prev_cents = cents
        prev_ids   = curr_ids
        frame_idx += 1

    cap.release()
    writer.release()
    blobs_f.close()
    rel_f.close()
    print("[DONE] Wrote:", out_overlay_mp4, out_blobs_csv, out_relations_csv)

if __name__ == "__main__":
    main()
