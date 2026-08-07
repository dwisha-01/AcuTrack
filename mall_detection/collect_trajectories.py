#  collect_trajectories.py — Standalone data collection script
#  Run this on your "normal" videos BEFORE training.
#  It runs the full AcuTrack pipeline and saves trajectory sequences to a .npy file.
#
#  Usage:
#    python collect_trajectories.py --video crowd_video.mp4
#    python collect_trajectories.py --video moderate_video.mp4 --append
#    python collect_trajectories.py --video mot17_video.mp4 --append
#
#  Output: training_data.npy  (shape: N x SEQ_LEN x FEAT_DIM)
#
#  Tip: run on at least 2-3 videos to get diverse normal behaviour.
#  200+ sequences is enough to train; 500+ is better.

import cv2
import numpy as np
import torch
import argparse
import os
from pathlib import Path
from collections import deque

from ultralytics import YOLO
from boxmot.trackers.strongsort.strongsort import StrongSort
from trajectory_ae import extract_features, SEQ_LEN, FEAT_DIM

# ── CONFIG ─────────────────────────────────────────────────────────────────────
FRAME_W        = 1060
FRAME_H        = 660
DETECT_EVERY_N = 3
OUTPUT_FILE    = "training_data.npy"

ZONES = {
    "Zone A": (10,  80, 390, 660),
    "Zone B": (390, 80, 720, 660),
    "Zone C": (720, 80, 1050, 660),
}


def point_in_zone(px, py):
    for name, (x1, y1, x2, y2) in ZONES.items():
        if x1 < px < x2 and y1 < py < y2:
            return name
    return None


def collect(video_path, append=False):
    print(f"\n{'='*55}")
    print(f"  AcuTrack — Trajectory Collection Mode")
    print(f"  Video  : {video_path}")
    print(f"  Output : {OUTPUT_FILE}")
    print(f"  Mode   : {'append' if append else 'new file'}")
    print(f"{'='*55}\n")

    # ── Load models ────────────────────────────────────────────────────────────
    print("Loading YOLO (ONNX)...")
    YOLO("yolov8n.pt").export(format="onnx")
    det_model = YOLO("yolov8n.onnx")

    print("Loading StrongSORT + OSNet...")
    tracker = StrongSort(
        reid_weights=Path("osnet_x0_25_msmt17.pt"),
        device=torch.device("cpu"),
        half=False,
        max_age=60,
        cmc_off=True,
        max_cos_dist=0.25,
    )

    # ── Per-track buffers ──────────────────────────────────────────────────────
    feat_buffers  = {}   # track_id -> deque(maxlen=SEQ_LEN)
    prev_positions = {}  # track_id -> (foot_x, foot_y)
    dwell_start   = {}   # track_id -> {zone: start_time}
    completed_seqs = []  # finished (SEQ_LEN, FEAT_DIM) arrays

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    frame_count = 0
    prev_dets   = np.empty((0, 6))

    print("Processing frames... (press Ctrl+C to stop early)\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            frame_count += 1
            now = frame_count / 9.0   # approximate timestamp at ~9 FPS

            # YOLO every N frames
            if frame_count % DETECT_EVERY_N == 0:
                results = det_model(frame, classes=[0], verbose=False, conf=0.2, iou=0.45)
                dets = np.empty((0, 6))
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    cls   = results[0].boxes.cls.cpu().numpy()
                    dets  = np.column_stack([boxes, confs, cls])
                prev_dets = dets

            tracks = tracker.update(prev_dets, frame)
            if len(tracks) == 0:
                continue

            active_ids = set()
            for track in tracks:
                tid   = int(track[4])
                fx    = int((track[0] + track[2]) / 2)
                fy    = int(track[3])
                active_ids.add(tid)

                zone  = point_in_zone(fx, fy)

                # Dwell tracking
                if tid not in dwell_start:
                    dwell_start[tid] = {}
                if zone and zone not in dwell_start[tid]:
                    dwell_start[tid][zone] = now
                dwell = (now - dwell_start[tid].get(zone, now)) if zone else 0.0

                # Previous position
                prev_pos = prev_positions.get(tid)
                prev_fx  = prev_pos[0] if prev_pos else None
                prev_fy  = prev_pos[1] if prev_pos else None
                prev_positions[tid] = (fx, fy)

                # Build feature vector
                fvec = extract_features(
                    fx, fy, prev_fx, prev_fy,
                    zone, dwell,
                    frame_w=FRAME_W, frame_h=FRAME_H
                )

                # Add to buffer
                if tid not in feat_buffers:
                    feat_buffers[tid] = deque(maxlen=SEQ_LEN)
                feat_buffers[tid].append(fvec)

                # When buffer is full, save a sequence and slide window by SEQ_LEN//2
                if len(feat_buffers[tid]) == SEQ_LEN:
                    completed_seqs.append(np.array(feat_buffers[tid]))
                    # Slide: remove first half so next sequence overlaps
                    for _ in range(SEQ_LEN // 2):
                        feat_buffers[tid].popleft()

            # Remove buffers for tracks that disappeared
            gone = set(feat_buffers.keys()) - active_ids
            for tid in gone:
                del feat_buffers[tid]
                prev_positions.pop(tid, None)
                dwell_start.pop(tid, None)

            if frame_count % 500 == 0:
                print(f"  Frame {frame_count:>5}  |  sequences collected: {len(completed_seqs)}")

    except KeyboardInterrupt:
        print("\nStopped early by user.")

    cap.release()

    if len(completed_seqs) == 0:
        print("\nNo sequences collected — check that the video file exists and has detections.")
        return

    new_data = np.array(completed_seqs, dtype=np.float32)  # (N, SEQ_LEN, FEAT_DIM)
    print(f"\n  Collected {len(completed_seqs)} sequences from this video.")

    if append and os.path.exists(OUTPUT_FILE):
        existing = np.load(OUTPUT_FILE)
        new_data  = np.concatenate([existing, new_data], axis=0)
        print(f"  Appended to existing file — total sequences: {len(new_data)}")

    np.save(OUTPUT_FILE, new_data)
    print(f"  Saved → {OUTPUT_FILE}  (shape: {new_data.shape})")
    print(f"\n  Ready to train when you have ≥ 200 sequences.")
    print(f"  Run:  python train_ae.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect normal trajectory sequences for AE training.")
    parser.add_argument("--video",  required=True, help="Path to video file")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing training_data.npy instead of overwriting")
    args = parser.parse_args()
    collect(args.video, append=args.append)
