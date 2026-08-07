import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import cv2
import torch
from collections import defaultdict

# Add path for loading models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
from boxmot.trackers.strongsort.strongsort import StrongSort

def bbox_iou(box1, box2):
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection + 1e-8
    
    return intersection / union

def get_wildtrack_gt(annotations_dir, frame_idx, camera_id):
    """
    Parses the ground truth bounding boxes for a given frame and camera ID.
    camera_id is 0-indexed (0 to 6) corresponding to C1-C7/cam1-cam7.
    """
    json_path = annotations_dir / f"{frame_idx:08d}.json"
    if not json_path.exists():
        return None
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    gt_boxes = []
    for person in data:
        person_id = person["personID"]
        view = person["views"][camera_id]
        # xmin, ymin, xmax, ymax are -1 if not visible in this view
        if view["xmin"] != -1:
            box = [view["xmin"], view["ymin"], view["xmax"], view["ymax"]]
            gt_boxes.append((person_id, box))
    return gt_boxes

def print_banner():
    print("=" * 65)
    print("         ACUTRACK ACCURACY & RELIABILITY EVALUATION TOOL")
    print("=" * 65)
    print("This script evaluates the accuracy of the YOLOv8 + StrongSort")
    print("pipeline against the Wildtrack Ground Truth annotations.")
    print("=" * 65)

def main():
    print_banner()
    
    # ── PATH CONFIGURATION ───────────────────────────────────────────────────
    dataset_dir = Path(__file__).parent / "Wildtrack"
    annotations_dir = dataset_dir / "annotations_positions"
    
    if not annotations_dir.exists():
        print(f"Error: Could not find annotations directory at: {annotations_dir}")
        print("Please ensure the 'Wildtrack' folder is placed inside 'mall_detection'.")
        return

    # ── USER INPUT ───────────────────────────────────────────────────────────
    print("\nAvailable Cameras:")
    for i in range(1, 8):
        video_file = dataset_dir / f"cam{i}.mp4"
        exists = " [Exists]" if video_file.exists() else " [Missing]"
        print(f"  {i}. Camera {i} (cam{i}.mp4) {exists}")
        
    cam_choice = input("\nSelect Camera to evaluate (1-7) [Default: 1]: ").strip()
    camera_num = int(cam_choice) if cam_choice.isdigit() and 1 <= int(cam_choice) <= 7 else 1
    camera_id = camera_num - 1 # 0-indexed for viewNum
    
    video_path = dataset_dir / f"cam{camera_num}.mp4"
    if not video_path.exists():
        print(f"Error: Video file '{video_path}' does not exist.")
        return

    # Select evaluation frames limit
    limit_choice = input("Enter number of annotated frames to evaluate (1-400) [Default: 50]: ").strip()
    max_eval_frames = int(limit_choice) if limit_choice.isdigit() and 1 <= int(limit_choice) <= 400 else 50

    print("\nInitializing models and tracking framework...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load YOLOv8 model
    if device.type == "cuda":
        model = YOLO("yolov8n.pt")
        model.to(device)
        print("YOLOv8 loaded on GPU.")
    else:
        if not os.path.exists("yolov8n.onnx"):
            print("Exporting YOLOv8 to ONNX for fast CPU inference...")
            YOLO("yolov8n.pt").export(format="onnx")
        model = YOLO("yolov8n.onnx")
        print("YOLOv8 loaded on CPU (ONNX format).")
        
    # Load StrongSort tracker
    tracker = StrongSort(
        reid_weights=Path(__file__).parent / "osnet_x0_25_msmt17.pt",
        device=device,
        half=(device.type == "cuda"),
        max_age=60,
        cmc_off=True,
        max_cos_dist=0.25,
    )
    
    cap = cv2.VideoCapture(str(video_path))
    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # We resize internally to matching application settings
    FRAME_W, FRAME_H = 800, 500
    scale_x = original_w / FRAME_W
    scale_y = original_h / FRAME_H
    
    # Get sorted list of all annotated frame indexes (e.g. 0, 5, 10, ...)
    all_jsons = sorted(list(annotations_dir.glob("*.json")))
    eval_frames = [int(f.stem) for f in all_jsons][:max_eval_frames]
    
    print(f"Evaluating {len(eval_frames)} frames on {video_path.name}...")
    
    # ── METRIC STATS counters ────────────────────────────────────────────────
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Tracking consistency mapping
    pred_to_gt = defaultdict(set) # pred_id -> set of matched gt_ids (more than 1 means ID switches/drifts)
    gt_to_pred = defaultdict(set) # gt_id -> set of matched pred_ids
    
    total_latency = 0.0
    processed_count = 0
    
    print("\nProcessing frames:")
    t_start = time.time()
    
    eval_set = set(eval_frames)
    max_target = max(eval_frames)
    
    current_frame_idx = 0
    eval_idx = 0
    
    while current_frame_idx <= max_target:
        ret, frame = cap.read()
        if not ret:
            print(f"Reached end of video stream at frame {current_frame_idx}")
            break
            
        t_inference_start = time.time()
        
        # Resize to application processing size
        resized_frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        
        # 1. Run YOLOv8 detection
        results = model(resized_frame, classes=[0], verbose=False, conf=0.2, iou=0.45)
        dets = np.empty((0, 6))
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            dets = np.column_stack([
                results[0].boxes.xyxy.cpu().numpy(),
                results[0].boxes.conf.cpu().numpy(),
                results[0].boxes.cls.cpu().numpy(),
            ])
            
        # 2. Run StrongSort tracker (updates on every frame for tracking continuity)
        tracks = tracker.update(dets, resized_frame)
        latency = time.time() - t_inference_start
        
        # 3. Only evaluate metric accuracy if this is an annotated target frame
        if current_frame_idx in eval_set:
            total_latency += latency
            processed_count += 1
            eval_idx += 1
            
            # Get ground truth detections
            gt_boxes = get_wildtrack_gt(annotations_dir, current_frame_idx, camera_id)
            if gt_boxes is not None:
                predictions = []
                for track in tracks:
                    px1 = track[0] * scale_x
                    py1 = track[1] * scale_y
                    px2 = track[2] * scale_x
                    py2 = track[3] * scale_y
                    pred_id = int(track[4])
                    predictions.append((pred_id, [px1, py1, px2, py2]))
                    
                if eval_idx == 1:
                    print("\n--- DIAGNOSTIC FOR FIRST EVALUATED FRAME ---")
                    print(f"Current frame index: {current_frame_idx}")
                    print(f"Ground Truth count: {len(gt_boxes)}")
                    print(f"Predictions count: {len(predictions)}")
                    if len(gt_boxes) > 0:
                        print(f"Sample GT box: {gt_boxes[0]}")
                    if len(predictions) > 0:
                        print(f"Sample Pred box: {predictions[0]}")
                    for p_id, p_box in predictions[:5]:
                        ious = []
                        for gt_id, gt_box in gt_boxes:
                            ious.append((gt_id, bbox_iou(p_box, gt_box)))
                        ious = sorted(ious, key=lambda x: x[1], reverse=True)
                        print(f"  Pred {p_id} {[round(coord, 1) for coord in p_box]} -> Best IoUs: {[(gt_id, round(iou_val, 3)) for gt_id, iou_val in ious[:3]]}")
                    print("--------------------------------------------\n")
                    
                # Perform Bounding Box Matching via IoU
                matched_gt = set()
                matched_pred = set()
                
                # Greedy matching based on maximum IoU
                for pred_id, pred_box in predictions:
                    best_iou = 0.0
                    best_gt_id = None
                    
                    for gt_id, gt_box in gt_boxes:
                        if gt_id in matched_gt:
                            continue
                        iou = bbox_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_id = gt_id
                            
                    if best_iou >= 0.40: # Match threshold
                        true_positives += 1
                        matched_gt.add(best_gt_id)
                        matched_pred.add(pred_id)
                        
                        # Re-ID tracking mapping
                        pred_to_gt[pred_id].add(best_gt_id)
                        gt_to_pred[best_gt_id].add(pred_id)
                    else:
                        false_positives += 1
                        
                # Any ground truth box not matched is a False Negative
                false_negatives += (len(gt_boxes) - len(matched_gt))
                
            # Print progress
            if eval_idx % 10 == 0 or current_frame_idx == max_target:
                print(f"  [{eval_idx}/{len(eval_frames)}] Frames evaluated...")
                
        current_frame_idx += 1

    cap.release()
    
    # ── COMPUTE QUANTITATIVE METRICS ─────────────────────────────────────────
    precision = true_positives / (true_positives + false_positives + 1e-9)
    recall = true_positives / (true_positives + false_negatives + 1e-9)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)
    
    avg_fps = processed_count / (total_latency + 1e-9)
    avg_latency_ms = (total_latency / (processed_count + 1e-9)) * 1000
    
    # Count identity switches
    # An ID switch is defined when a predicted tracker ID is matched with multiple different ground truth IDs
    # or vice-versa (tracker swaps identity).
    id_switches = 0
    for pred_id, matched_gts in pred_to_gt.items():
        if len(matched_gts) > 1:
            id_switches += (len(matched_gts) - 1)
            
    # Calculate ID match rate (reliability)
    # The percentage of ground truth individuals that were consistently mapped to a single main predicted ID
    stable_gt_ids = 0
    for gt_id, matched_preds in gt_to_pred.items():
        if len(matched_preds) == 1:
            stable_gt_ids += 1
            
    reid_consistency = (stable_gt_ids / (len(gt_to_pred) + 1e-9)) * 100.0

    # ── DISPLAY REPORT ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("               ACUTRACK PERFORMANCE & ACCURACY REPORT")
    print("=" * 65)
    print(f" Target Sequence      : {video_path.name}")
    print(f" Frames Evaluated     : {processed_count}")
    print(f" Target Resolution    : {original_w}x{original_h} px (Rescaled from {FRAME_W}x{FRAME_H})")
    print("-" * 65)
    print(" [PEDESTRIAN DETECTION ACCURACY]")
    print(f"  - True Positives (TP)   : {true_positives}")
    print(f"  - False Positives (FP)  : {false_positives}")
    print(f"  - False Negatives (FN)  : {false_negatives}")
    print(f"  - Detection Precision   : {precision * 100.2:.2f}%")
    print(f"  - Detection Recall      : {recall * 100.0:.2f}%")
    print(f"  - Detection F1-Score    : {f1_score * 100.0:.2f}%")
    print("-" * 65)
    print(" [RE-ID & TRACKING RELIABILITY]")
    print(f"  - Unique GT Identities  : {len(gt_to_pred)}")
    print(f"  - Unique Tracker IDs    : {len(pred_to_gt)}")
    print(f"  - ID Switches Detected  : {id_switches}")
    print(f"  - Re-ID Consistency     : {reid_consistency:.2f}% (Stable matches across time)")
    print("-" * 65)
    print(" [SYSTEM PERFORMANCE & EFFICIENCY]")
    print(f"  - Average Inference     : {avg_latency_ms:.2f} ms per frame")
    print(f"  - Pipeline Frame Rate   : {avg_fps:.2f} FPS")
    print(f"  - Acceleration Hardware : {'GPU (CUDA)' if device.type == 'cuda' else 'CPU (ONNX Optimized)'}")
    print("=" * 65)
    print(" Validation successfully completed.\n")

if __name__ == "__main__":
    main()
