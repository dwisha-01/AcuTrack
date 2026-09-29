#  ACUTRACK — ID-Persistent Retail Analytics Platform
#  app.py — StrongSort + OSNet + LSTM Autoencoder Suspicious Tracking
#
#  ML Pipeline:
#    1. Collect normal sequences  →  python collect_trajectories.py --video your_video.mp4
#    2. Train autoencoder         →  python train_ae.py
#    3. Restart app.py            →  model auto-loaded on startup
#    Or: use the in-app controls  →  Dashboard "Behavior Model" panel
#
#  Changes from previous version:
#    - FIX: All camera threads now run in parallel (not exit when inactive)
#    - FIX: Removed torch.set_num_threads(1) — now uses all CPU cores
#    - FIX: Per-camera YOLO models — no global yolo_lock contention
#    - FIX: Per-camera behavior locks — no cross-camera lock contention
#    - FIX: conf=0.35 (was 0.2) — eliminates false positive detections
#    - FIX: Re-ID threshold=0.28 (was 0.40) — prevents wrong identity merges
#    - FIX: max_age=20 (was 60) — eliminates ghost track false positives
#    - FIX: Start all camera threads at startup, not on-demand
#
#  Changes in this revision:
#    - FIX: Cross-camera global_id COLLISION. Previously, whenever a track had
#      no OSNet embedding available yet (brand-new track, or the features
#      lookup threw), global_id fell back to the raw per-camera track_id.
#      Every camera's StrongSort tracker numbers tracks starting at 1, so
#      cam1's track_id=1 and cam3's track_id=1 both resolved to the SAME
#      global_id=1 — and since global_flagged_ids / tracked_global_ids /
#      global_gallery are shared across all cameras, two different people
#      ended up sharing flag state, target-lock, and even blended gallery
#      embeddings. Fallback is now a camera-namespaced NEGATIVE int that can
#      never collide with a real (gallery-assigned) global_id — real IDs are
#      always positive, starting at 1.
#    - FIX: Camera threads no longer all start at boot. Only the initially
#      active camera's thread starts at startup; other cameras' threads are
#      started on-demand the first time you switch to them (and then stay
#      warm exactly as before), so idle cameras don't burn CPU on YOLO +
#      StrongSort + OSNet before you've ever looked at them.
from models import (
    init_db, db_session, Camera, Zone, TrackedPerson, DwellTime, Alert, 
    Sighting, PersonEmbedding, shutdown_session
)
from flask import Flask, render_template, Response, request, jsonify
from reid_search import detect_face, extract_face_embedding, compute_face_similarity
from datetime import datetime, timedelta

def to_ist_str(dt):
    if not dt:
        return "N/A"
    ist_dt = dt + timedelta(hours=5, minutes=30)
    return ist_dt.strftime("%Y-%m-%d %H:%M:%S")

from flask_socketio import SocketIO
from ultralytics import YOLO
from boxmot.trackers.strongsort.strongsort import StrongSort
from pathlib import Path
from collections import deque
import queue
import json
import cv2
import time
import threading
import subprocess
import sys
import os
import numpy as np
import torch
import psutil

from trajectory_ae import (TrajectoryAutoencoder, extract_features,
                            seq_to_tensor, SEQ_LEN, FEAT_DIM)

# ── Inference-only mode ────────────────────────────────────────────────────────
# NOTE: Do NOT call torch.set_num_threads(1) here — that limits ALL PyTorch
# ops (YOLO + OSNet + AE) to 1 CPU core across every thread. Removed.
torch.set_grad_enabled(False)
# Let PyTorch use available cores (default). Cap if you want to limit:
# torch.set_num_threads(max(4, os.cpu_count() // 2))

app = Flask(__name__)

@app.teardown_appcontext
def shutdown_session_holder(exception=None):
    shutdown_session(exception)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', ping_timeout=60, ping_interval=25)

# ── YOLO ───────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"AcuTrack: Using device '{device}' for model inference.")

# Export ONNX once if needed (CPU path)
if device.type != "cuda" and not os.path.exists("yolov8n.onnx"):
    print("Exporting YOLOv8 to ONNX for CPU inference...")
    YOLO("yolov8n.pt").export(format="onnx")

def _load_yolo():
    """Load one YOLO model instance (called once per camera thread)."""
    m = YOLO("yolov8n.pt")
    if device.type == "cuda":
        m.to(device)
    return m

# ── FRAME RESOLUTION ───────────────────────────────────────────────────────────
FRAME_W = 1060
FRAME_H = 660

# ── PERFORMANCE CONSTANTS ──────────────────────────────────────────────────────
DETECT_EVERY_N   = 2
EMIT_EVERY_N     = 5
MOTION_THRESHOLD = 15
JPEG_QUALITY     = 70
# The dashboard only needs a modest refresh rate for an annotated analytics
# view.  Keeping this below the source rate avoids encoding frames the browser
# cannot paint and bounds MJPEG bandwidth.
STREAM_FPS        = 12
STATE_TTL_SECONDS = 30 * 86400  # 30 days to keep profiles in memory for matching
MAX_GALLERY_IDENTITIES = 1000
FRAME_STEP        = 2  # Process every Nth frame of video file (constant step size) to reduce lag and preserve tracking continuity

# Set this to True to run the 3-laptop live demo with webcams,
# or False to run the Wildtrack dataset.
LIVE_DEMO_MODE = True

# ── VIDEO SOURCES ─────────────────────────────────────────────────────────────
if LIVE_DEMO_MODE:
    VIDEO_SOURCES = {
        "cam1": {"file": 0, "label": "Camera 1 (Local)", "desc": "Local Laptop Webcam Feed"},
        "cam2": {"file": "upload", "label": "Camera 2 (Remote)", "desc": "Webcam Feed uploaded from Laptop 2"},
        "cam3": {"file": "upload", "label": "Camera 3 (Remote)", "desc": "Webcam Feed uploaded from Laptop 3"},
    }
else:
    VIDEO_SOURCES = {
        "cam1": {"file": "Wildtrack/cam1.mp4", "label": "Camera 1", "desc": "Wildtrack Cam 1 — Courtyard view 1"},
        "cam2": {"file": "Wildtrack/cam2.mp4", "label": "Camera 2", "desc": "Wildtrack Cam 2 — Courtyard view 2"},
        "cam3": {"file": "Wildtrack/cam3.mp4", "label": "Camera 3", "desc": "Wildtrack Cam 3 — Courtyard view 3"},
        "cam4": {"file": "Wildtrack/cam4.mp4", "label": "Camera 4", "desc": "Wildtrack Cam 4 — Courtyard view 4"},
        "cam5": {"file": "Wildtrack/cam5.mp4", "label": "Camera 5", "desc": "Wildtrack Cam 5 — Courtyard view 5"},
        "cam6": {"file": "Wildtrack/cam6.mp4", "label": "Camera 6", "desc": "Wildtrack Cam 6 — Courtyard view 6"},
        "cam7": {"file": "Wildtrack/cam7.mp4", "label": "Camera 7", "desc": "Wildtrack Cam 7 — Courtyard view 7"},
        "live": {"file": 0,                    "label": "Live Camera", "desc": "Live webcam feed"},
    }

# Thread-safe queues to receive webcam frames uploaded from Laptop 2 & Laptop 3
upload_frame_queues = {k: queue.Queue(maxsize=2) for k in VIDEO_SOURCES}

# Per-source health stays independent of the rolling process-wide performance
# counters. Timestamps are wall-clock values for the dashboard and diagnostics.
camera_health_lock = threading.Lock()
camera_health = {
    key: {
        "status": "waiting", "last_frame_at": None, "last_processed_at": None,
        "last_stream_frame_at": None,
        "arrival_fps": 0.0, "processing_fps": 0.0, "frames_received": 0,
        "worker_status": "stopped", "frames_dropped": 0,
        "processing_latency_ms": 0.0,
        "frames_processed": 0, "consecutive_read_failures": 0,
        "reconnection_attempts": 0, "client_reconnection_attempts": 0,
        "client_last_read_failure_burst": 0, "last_error": None,
        "_last_arrival_mono": None, "_last_process_mono": None,
        "_arrival_samples": deque(maxlen=20), "_process_samples": deque(maxlen=20),
    }
    for key in VIDEO_SOURCES
}


def update_camera_health(camera_key, **updates):
    with camera_health_lock:
        camera_health[camera_key].update(updates)


def record_camera_frame(camera_key, processed=False):
    now_wall = time.time()
    now_mono = time.monotonic()
    with camera_health_lock:
        health = camera_health[camera_key]
        if processed:
            health["frames_processed"] += 1
            health["last_processed_at"] = now_wall
            health["status"] = "connected"
            health["worker_status"] = "running"
            health["last_error"] = None
            samples = health["_process_samples"]
            previous = health["_last_process_mono"]
            health["_last_process_mono"] = now_mono
            if previous is not None:
                samples.append(now_mono - previous)
                health["processing_fps"] = round(len(samples) / sum(samples), 2) if sum(samples) else 0.0
        else:
            health["frames_received"] += 1
            health["last_frame_at"] = now_wall
            health["consecutive_read_failures"] = 0
            health["last_error"] = None
            samples = health["_arrival_samples"]
            previous = health["_last_arrival_mono"]
            health["_last_arrival_mono"] = now_mono
            if previous is not None:
                samples.append(now_mono - previous)
                health["arrival_fps"] = round(len(samples) / sum(samples), 2) if sum(samples) else 0.0


def enqueue_latest_camera_frame(camera_key, frame, captured_at=None):
    """Nonblocking, bounded latest-frame insertion for remote camera uploads."""
    q = upload_frame_queues[camera_key]
    item = (time.monotonic() if captured_at is None else captured_at, frame)
    dropped = 0
    while True:
        try:
            q.put_nowait(item)
            break
        except queue.Full:
            try:
                q.get_nowait()
                dropped += 1
            except queue.Empty:
                continue
    if dropped:
        with camera_health_lock:
            camera_health[camera_key]["frames_dropped"] += dropped
    return dropped


def camera_retry_delay(current_delay):
    """Exponential recovery backoff, capped so retries stay periodic."""
    return min(10.0, max(1.0, current_delay * 1.5))


def record_camera_read_failure(camera_key):
    with camera_health_lock:
        health = camera_health[camera_key]
        health["consecutive_read_failures"] += 1
        failures = health["consecutive_read_failures"]
        if failures >= 10:
            health["status"] = "recovering"
            health["worker_status"] = "recovering"
            health["last_error"] = f"{failures} consecutive camera read failures"
        return failures

# FIX: stable per-camera index used to namespace fallback IDs (see below) so
# they can never collide across cameras.
CAMERA_INDEX = {k: i for i, k in enumerate(VIDEO_SOURCES)}

# ── ZONES ─────────────────────────────────────────────────────────────────────
ZONES = {
    "Zone A": {"coords": (10,  80, 390, 660), "color": (74,  144, 217), "label": "Store Entrance"},
    "Zone B": {"coords": (390, 80, 720, 660), "color": (56,  161, 105), "label": "Food Court"},
    "Zone C": {"coords": (720, 80, 1050, 660), "color": (221, 107, 32),  "label": "Exit Corridor"},
}

ZONE_CAPACITY = {"Zone A": 8, "Zone B": 10, "Zone C": 8}

# ── DUMMY CAMERA MOTION COMPENSATION ──────────────────────────────────────────
class DummyCMC:
    def apply(self, img, dets):
        # Returns identity warp matrix to bypass OpenCV's findTransformECC
        return np.eye(2, 3, dtype=np.float32)

# ── STRONGSORT + OSNET (one tracker per camera) ───────────────────────────────
# FIX: max_age reduced from 60 → 20 to eliminate ghost tracks that inflate FP.
# At ~5 FPS effective rate, max_age=20 = ~4 seconds of track persistence — realistic.
trackers = {k: None for k in VIDEO_SOURCES}
tracker_init_lock = threading.Lock()


def get_tracker(camera_key):
    """Create a StrongSORT instance only when its camera is first used."""
    with tracker_init_lock:
        tracker = trackers[camera_key]
        if tracker is None:
            print(f"[{camera_key}] Loading StrongSort + OSNet Re-ID...")
            tracker = StrongSort(
                reid_weights=Path("osnet_x0_25_msmt17.pt"),
                device=device,
                half=(device.type == "cuda"),
                max_age=90,
                max_cos_dist=0.55,
            )
            tracker.cmc = DummyCMC()
            trackers[camera_key] = tracker
        return tracker

# ── STATE MANAGEMENT ──────────────────────────────────────────────────────────
video_positions = {k: 0 for k in VIDEO_SOURCES}
# The dashboard presents one synchronized view at a time.  When changing
# cameras, seek the new view to the active view's source frame before it can
# contribute embeddings to the shared Re-ID gallery.
camera_seek_targets = {k: None for k in VIDEO_SOURCES}
camera_states = {
    k: {
        "zone_counts":        {z: 0   for z in ZONES},
        "zone_footfall":      {z: 0   for z in ZONES},
        "zone_avg_dwell":     {z: 0.0 for z in ZONES},
        "zone_dwell_samples": {z: []  for z in ZONES},
        "prev_zone_ids":      {z: set() for z in ZONES},
        "zone_alerts":        {z: False for z in ZONES},
        "dwell_entry_times":  {},
        "person_history":     {},
        "known_ids":          set(),
        "frame_count":        0,
        "prev_tracks":        [],
        "prev_dets":          np.empty((0, 6)),
        # FIX: Per-camera behavior lock — eliminates cross-camera lock contention.
        # Previously a single global behavior_lock blocked all 7 cameras.
        "behavior_lock":      threading.Lock(),
    }
    for k in VIDEO_SOURCES
}

# ── RE-ID ─────────────────────────────────────────────────────────────────────
known_ids       = set()
id_switch_count = 0
reid_lock       = threading.Lock()
db_lock         = threading.Lock()
db_queue        = queue.Queue()
last_db_sighting_time = {}

def db_worker():
    """Background worker thread to handle all SQLite database writes and file I/O crops asynchronously."""
    while True:
        try:
            task = db_queue.get()
            if task is None:
                db_queue.task_done()
                break
            
            task_type, data = task
            with db_lock:
                try:
                    if task_type == "register_person":
                        gid = data["gid"]
                        camera_id = data["camera_id"]
                        
                        db_camera = db_session.query(Camera).filter(Camera.camera_key == camera_id).first()
                        camera_db_id = db_camera.id if db_camera else 1
                        
                        crop = data.get("snapshot_crop")
                        full_save_path = data.get("full_save_path")
                        snapshot_web_path = data.get("snapshot_web_path")
                        if crop is not None and full_save_path is not None:
                            os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
                            cv2.imwrite(full_save_path, crop)
                        
                        db_person = db_session.query(TrackedPerson).filter(TrackedPerson.person_id == gid).first()
                        if db_person is None:
                            db_person = TrackedPerson(
                                person_id=gid,
                                best_image_path=snapshot_web_path,
                                first_seen=datetime.utcnow(),
                                last_seen=datetime.utcnow(),
                                visit_count=1,
                                total_dwell=0
                            )
                            db_session.add(db_person)
                            
                            new_emb = PersonEmbedding(
                                person_id=gid,
                                embedding_data=data["embedding_bytes"],
                                camera_key=camera_id,
                                confidence=float(data["confidence"])
                            )
                            db_session.add(new_emb)
                        else:
                            db_person.visit_count += 1
                            db_person.last_seen = datetime.utcnow()
                            
                            box_area = data.get("box_area")
                            confidence = data.get("confidence", 1.0)
                            if snapshot_web_path:
                                if not db_person.best_image_path:
                                    db_person.best_image_path = snapshot_web_path
                                elif box_area is not None and box_area > 8000 and confidence >= 0.6:
                                    db_person.best_image_path = snapshot_web_path

                            existing_embs = db_session.query(PersonEmbedding).filter(
                                PersonEmbedding.person_id == gid,
                                PersonEmbedding.camera_key != "face"
                            ).all()
                            if len(existing_embs) < GALLERY_TEMPLATES_PER_ID:
                                embedding = np.frombuffer(data["embedding_bytes"], dtype=np.float32)
                                distinct = True
                                for ext in existing_embs:
                                    ext_vec = np.frombuffer(ext.embedding_data, dtype=np.float32)
                                    dist = _cos_dist(embedding, ext_vec)
                                    if dist < 0.12:
                                        distinct = False
                                        break
                                if distinct:
                                    new_emb = PersonEmbedding(
                                        person_id=gid,
                                        embedding_data=data["embedding_bytes"],
                                        camera_key=camera_id,
                                        confidence=float(confidence)
                                    )
                                    db_session.add(new_emb)
                        
                        # Extract and cache face embedding asynchronously in the background thread
                        if crop is not None:
                            try:
                                from reid_search import detect_face, extract_face_embedding
                                c_face = detect_face(crop)
                                if c_face is not None:
                                    face_emb = extract_face_embedding(crop, c_face)
                                    if face_emb is not None:
                                        existing_face_emb = db_session.query(PersonEmbedding).filter(
                                            PersonEmbedding.person_id == gid,
                                            PersonEmbedding.camera_key == "face"
                                        ).first()
                                        if not existing_face_emb:
                                            new_face_emb = PersonEmbedding(
                                                person_id=gid,
                                                embedding_data=face_emb.astype(np.float32).tobytes(),
                                                camera_key="face",
                                                confidence=1.0
                                            )
                                            db_session.add(new_face_emb)
                                            with gallery_lock:
                                                global_face_gallery.setdefault(gid, []).append(face_emb)
                            except Exception as e_face:
                                print(f"[DB WORKER FACE EXTRACTION EXCEPTION] {e_face}")
                                
                        if snapshot_web_path:
                            new_sighting = Sighting(
                                person_id=gid,
                                camera_id=camera_db_id,
                                timestamp=datetime.utcnow(),
                                confidence=float(data["confidence"]),
                                zone=data.get("current_zone"),
                                snapshot_path=snapshot_web_path
                            )
                            db_session.add(new_sighting)
                            
                        db_session.commit()
                        
                    elif task_type == "merge_persons":
                        old_gid = data["old_gid"]
                        new_gid = data["new_gid"]
                        
                        # 1. Update sightings
                        db_session.query(Sighting).filter(Sighting.person_id == old_gid).update({Sighting.person_id: new_gid})
                        
                        # 2. Update dwell times
                        db_session.query(DwellTime).filter(DwellTime.person_id == old_gid).update({DwellTime.person_id: new_gid})
                        
                        # 3. Update alerts
                        db_session.query(Alert).filter(Alert.person_id == old_gid).update({Alert.person_id: new_gid})
                        
                        # 4. Update embeddings
                        db_session.query(PersonEmbedding).filter(PersonEmbedding.person_id == old_gid).update({PersonEmbedding.person_id: new_gid})
                        
                        # 5. Merge TrackedPerson statistics
                        old_person = db_session.query(TrackedPerson).filter(TrackedPerson.person_id == old_gid).first()
                        new_person = db_session.query(TrackedPerson).filter(TrackedPerson.person_id == new_gid).first()
                        if old_person and new_person:
                            new_person.visit_count += old_person.visit_count
                            new_person.total_dwell += old_person.total_dwell
                            if old_person.first_seen < new_person.first_seen:
                                new_person.first_seen = old_person.first_seen
                            if old_person.last_seen > new_person.last_seen:
                                new_person.last_seen = old_person.last_seen
                            db_session.delete(old_person)
                        elif old_person:
                            old_person.person_id = new_gid
                            
                        db_session.commit()
                        
                    elif task_type == "add_dwell":
                        person_id = data["person_id"]
                        camera_key = data["camera_key"]
                        zone_name = data["zone_name"]
                        duration = data["duration"]
                        entry_time = data["entry_time"]
                        
                        db_camera = db_session.query(Camera).filter(Camera.camera_key == camera_key).first()
                        camera_db_id = db_camera.id if db_camera else 1
                        
                        db_zone = db_session.query(Zone).filter(
                            Zone.camera_id == camera_db_id, 
                            Zone.zone_name == zone_name
                        ).first()
                        
                        if db_zone:
                            new_dt = DwellTime(
                                person_id=person_id,
                                zone_id=db_zone.id,
                                entry_time=entry_time,
                                exit_time=datetime.utcnow(),
                                dwell_duration_seconds=int(duration)
                            )
                            db_session.add(new_dt)
                            
                            db_person = db_session.query(TrackedPerson).filter(TrackedPerson.person_id == person_id).first()
                            if db_person:
                                db_person.total_dwell = (db_person.total_dwell or 0) + int(duration)
                                
                            db_session.commit()
                            
                    elif task_type == "add_alert":
                        person_id = data["person_id"]
                        camera_key = data["camera_key"]
                        zone_name = data["zone_name"]
                        anomaly_score = data["anomaly_score"]
                        message = data["message"]
                        alert_type = data["alert_type"]
                        
                        db_camera = db_session.query(Camera).filter(Camera.camera_key == camera_key).first()
                        camera_db_id = db_camera.id if db_camera else 1
                        
                        db_zone = None
                        if zone_name:
                            db_zone = db_session.query(Zone).filter(
                                Zone.camera_id == camera_db_id,
                                Zone.zone_name == zone_name
                            ).first()
                        zone_db_id = db_zone.id if db_zone else None
                        
                        new_alert = Alert(
                            person_id=person_id,
                            camera_id=camera_db_id,
                            zone_id=zone_db_id,
                            alert_type=alert_type,
                            anomaly_score=float(anomaly_score),
                            message=message,
                            status="PENDING",
                            timestamp=datetime.utcnow()
                        )
                        db_session.add(new_alert)
                        
                        db_person = db_session.query(TrackedPerson).filter(TrackedPerson.person_id == person_id).first()
                        if db_person:
                            db_person.is_flagged_suspicious = True
                            db_person.flagged_reason = message
                            
                        db_session.commit()
                        
                except Exception as ex:
                    db_session.rollback()
                    print(f"[DB WORKER TASK EXCEPTION] {ex}")
                finally:
                    db_session.close()
        except Exception as e_queue:
            print(f"[DB WORKER QUEUE EXCEPTION] {e_queue}")
        finally:
            db_queue.task_done()

threading.Thread(target=db_worker, daemon=True).start()

global_flagged_ids = set()
global_manually_flagged_ids = set()

# ── CROSS-CAMERA GALLERY ──────────────────────────────────────────────────────
gallery_lock    = threading.Lock()
global_gallery  = {}
global_face_gallery = {}
local_to_global = {}
global_last_seen = {}
next_global_id  = 1
_global_id_diagnostic_seen = set()
face_aligned_tracks = set()
track_last_face_check = {}
# FIX: yolo_lock REMOVED — each camera now has its own YOLO model instance,
# so no shared resource contention. Cameras run YOLO truly in parallel.

# ── CAMERA THREAD LIFECYCLE ────────────────────────────────────────────────────
# FIX: cameras no longer all start at boot. Threads are started on-demand
# (see ensure_camera_started) the first time a camera becomes active, and
# then keep running so switching back to it doesn't lose warm state.
camera_threads_started = {k: False for k in VIDEO_SOURCES}
camera_threads_lock    = threading.Lock()
camera_worker_threads   = {k: None for k in VIDEO_SOURCES}

# ── CROSS-CAMERA RE-ID MATCHING PARAMS ────────────────────────────────────────
# FIX (this revision): the gallery used to store ONE blended running-average
# embedding per global id (alpha=0.9 old + 0.1 new). Averaging embeddings
# taken from very different camera angles produces a "blurry" template that
# doesn't strongly resemble ANY single viewpoint — so a genuine re-appearance
# from a new angle often fell outside the match threshold and got a brand new
# id. We now keep up to GALLERY_TEMPLATES_PER_ID separate embeddings per
# person and match against the BEST (min-distance) template, not an average.
GALLERY_TEMPLATES_PER_ID = 15

# REID matching thresholds and ambiguity margins (Phase 2 & 3)
# In live demo mode with multiple consumer webcams, lighting, auto-exposure, and sensors vary.
# We adapt thresholds dynamically for live webcam demo vs benchmark dataset.
if LIVE_DEMO_MODE:
    REID_MATCH_THRESHOLD = 0.58       # 0.58 allows cross-camera matching across different laptop webcams
    REID_RECENT_WINDOW_SECONDS = 30 * 86400.0
    REID_AMBIGUITY_MARGIN = 0.04      # lower ambiguity margin prevents splitting same person into new IDs
    GALLERY_MIN_CONFIDENCE = 0.50     # 0.50 ensures live webcam crops are admitted to gallery
    GALLERY_MIN_BOX_AREA = 1000       # 1000 px allows people 2-3 meters away to register in gallery
    GALLERY_MIN_DIST = 0.05           # accepts varied poses
    GALLERY_MAX_DIST = 0.68           # allows cross-angle templates
else:
    REID_MATCH_THRESHOLD = 0.48
    REID_RECENT_WINDOW_SECONDS = 30 * 86400.0
    REID_AMBIGUITY_MARGIN = 0.08
    GALLERY_MIN_CONFIDENCE = 0.70
    GALLERY_MIN_BOX_AREA = 4000
    GALLERY_MIN_DIST = 0.08
    GALLERY_MAX_DIST = 0.60

# ── RULE-BASED BEHAVIOR PARAMS (tuned for retail) ─────────────────────────────
RESTRICTED_ZONES = set()
LOITER_THRESHOLD = 90
ZONE_HOP_LIMIT   = 3
SCORE_DECAY      = 0.90
FLAG_THRESHOLD   = 70

# ── LSTM AUTOENCODER ───────────────────────────────────────────────────────────
ae_lock          = threading.Lock()
ae_model_obj     = None      # TrajectoryAutoencoder instance, None until loaded
ae_threshold     = 0.05      # MSE threshold — updated after training
ae_trained       = False     # True once weights are loaded
ae_training_now  = False     # True while train_ae.py subprocess is running

# Sequence collection (for in-app training workflow)
TRAINING_MODE    = False     # when True, collect seqs instead of scoring
training_seqs    = []        # list of (SEQ_LEN, FEAT_DIM) numpy arrays
training_lock    = threading.Lock()
MIN_TRAIN_SEQS   = 200       # minimum before training is allowed

# ML/rule hybrid weighting
ML_WEIGHT   = 0.6    # weight for ML score (0.0 = rules only, 1.0 = ML only)
RULE_WEIGHT = 0.4

WEIGHTS_FILE   = "trajectory_ae.pt"
THRESHOLD_FILE = "ae_threshold.npy"


def _try_load_ae_model():
    """Load AE weights + threshold from disk if they exist. Called at startup."""
    global ae_model_obj, ae_threshold, ae_trained
    if not os.path.exists(WEIGHTS_FILE):
        print("  No AE weights found — run data collection + train_ae.py first.")
        return
    try:
        m = TrajectoryAutoencoder(feat_dim=FEAT_DIM, seq_len=SEQ_LEN)
        m.load_state_dict(torch.load(WEIGHTS_FILE, map_location="cpu"))
        m.eval()
        with ae_lock:
            ae_model_obj = m
            ae_trained   = True
        if os.path.exists(THRESHOLD_FILE):
            t = float(np.load(THRESHOLD_FILE)[0])
            with ae_lock:
                ae_threshold = t
            print(f"  AE model loaded — threshold: {t:.5f}")
        else:
            print("  AE model loaded — using default threshold (no threshold file).")
    except Exception as e:
        print(f"  AE model load failed: {e}")


# ── TARGET LOCK ───────────────────────────────────────────────────────────────
tracked_global_ids = set()

# ── GLOBAL STATE ──────────────────────────────────────────────────────────────
state_lock          = threading.Lock()
frame_condition     = threading.Condition(state_lock)
current_video       = "cam1"
latest_frame        = None
latest_frame_seq    = 0

# Rolling metrics are intentionally small and bounded. They expose the real
# workload without retaining frames, detections, or per-request samples.
performance_metrics = {
    "frames": 0, "inference_ms": 0.0, "tracking_ms": 0.0,
    "reid_ms": 0.0, "encode_ms": 0.0, "loop_ms": 0.0,
    "last_report_at": time.monotonic(), "fps": 0.0,
}


def record_performance(inference_ms, tracking_ms, reid_ms, encode_ms, loop_ms):
    """Maintain constant-memory EWMA timing metrics for the health endpoint."""
    with state_lock:
        metrics = performance_metrics
        metrics["frames"] += 1
        alpha = 0.08
        for name, value in (("inference_ms", inference_ms), ("tracking_ms", tracking_ms),
                            ("reid_ms", reid_ms), ("encode_ms", encode_ms),
                            ("loop_ms", loop_ms)):
            if value is None:
                continue
            metrics[name] = value if metrics["frames"] == 1 else (1 - alpha) * metrics[name] + alpha * value
        elapsed = time.monotonic() - metrics["last_report_at"]
        if elapsed >= 1.0:
            metrics["fps"] = metrics["frames"] / elapsed
            metrics["frames"] = 0
            metrics["last_report_at"] = time.monotonic()

# ── ANNOTATIONS CACHE ─────────────────────────────────────────────────────────
annotations_cache = {}


# ── HELPERS ───────────────────────────────────────────────────────────────────

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


def evaluate_live_frame(tracks, frame_idx, camera_key):
    """Compares tracks to ground truth JSON for frame_idx and camera_key."""
    import json
    cam_map = {"cam1": 0, "cam2": 1, "cam3": 2, "cam4": 3, "cam5": 4, "cam6": 5, "cam7": 6}
    if camera_key not in cam_map:
        return 0, 0, 0
    camera_id = cam_map[camera_key]

    # Use memory cache to avoid disk reads
    if frame_idx in annotations_cache:
        data = annotations_cache[frame_idx]
    else:
        json_path = Path("Wildtrack/annotations_positions") / f"{frame_idx:08d}.json"
        if not json_path.exists():
            return 0, 0, 0
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            annotations_cache[frame_idx] = data
        except Exception:
            return 0, 0, 0

    gt_boxes = []
    for person in data:
        view = person["views"][camera_id]
        if view["xmin"] != -1:
            gt_boxes.append([view["xmin"], view["ymin"], view["xmax"], view["ymax"]])

    if not gt_boxes and not tracks:
        return 0, 0, 0

    # Scale track coordinates back to 1920×1080 (original Wildtrack resolution)
    scale_x = 1920.0 / FRAME_W
    scale_y = 1080.0 / FRAME_H

    predictions = []
    for track in tracks:
        px1 = track[0] * scale_x
        py1 = track[1] * scale_y
        px2 = track[2] * scale_x
        py2 = track[3] * scale_y
        predictions.append([px1, py1, px2, py2])

    tp = 0
    matched_gts = set()
    for pred_box in predictions:
        best_iou = 0.0
        best_gt_idx = None
        for i, gt_box in enumerate(gt_boxes):
            if i in matched_gts:
                continue
            iou = bbox_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        if best_iou >= 0.40:
            tp += 1
            matched_gts.add(best_gt_idx)

    fp = len(predictions) - tp
    fn = len(gt_boxes) - len(matched_gts)
    return tp, fp, fn


def point_in_zone(px, py, zone_name):
    # NOTE: previously hardcoded to (1060.0 / FRAME_W) and (660.0 / FRAME_H).
    # Since ZONES coordinates are authored directly against FRAME_W/FRAME_H,
    # this scaling is always exactly 1.0 as long as those two constants stay
    # in sync — which they always will now, since we removed the duplicate
    # hardcoded numbers. If FRAME_W/FRAME_H ever change, zone detection still
    # lines up correctly instead of silently drifting.
    x1, y1, x2, y2 = ZONES[zone_name]["coords"]
    return x1 < px < x2 and y1 < py < y2


def _cos_dist(a, b):
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def emit_global_id_diagnostic(event):
    """Emit one grep-friendly JSON record for a Global ID decision."""
    print("GLOBAL_ID_DIAG " + json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False))


def _global_id_diag_once(key, event):
    """Avoid repeating the same local-track decision on every detection frame."""
    diagnostic_key = (key[0], key[1], event.get("assigned_gid"))
    if diagnostic_key in _global_id_diagnostic_seen:
        return
    _global_id_diagnostic_seen.add(diagnostic_key)
    emit_global_id_diagnostic(event)


def match_or_register(camera_id, local_id, embedding, threshold=REID_MATCH_THRESHOLD,
                       exclude_gids=None, box_area=None, frame=None, bbox=None, confidence=1.0, current_zone=None,
                       run_face_detection=True, ghost_claims=None):
    """
    Match local track to a global ID via OSNet embedding gallery.
    Then, queue the database sync task to offload all I/O from the real-time loop.
    """
    global next_global_id, face_aligned_tracks, track_last_face_check
    if ghost_claims is None:
        ghost_claims = {}
    key = (camera_id, local_id)
    seen_at = time.monotonic()
    embedding_dim = int(np.asarray(embedding).size) if embedding is not None else None
    embedding_norm = float(np.linalg.norm(embedding)) if embedding is not None else None
    embedding_valid = bool(
        embedding is not None and embedding_dim > 0 and np.isfinite(embedding_norm) and embedding_norm > 1e-8
    )
    diagnostic = {
        "event": "global_id_decision",
        "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "camera_key": camera_id,
        "local_track_id": int(local_id),
        "box_area": box_area,
        "detection_confidence": float(confidence) if confidence is not None else None,
        "embedding_available": embedding is not None,
        "embedding_dimension": embedding_dim,
        "embedding_norm": embedding_norm,
        "embedding_valid": embedding_valid,
        "candidate_gid_count": 0,
        "gallery_gid_count": 0,
        "gallery_template_count": 0,
        "empty_gallery_gid_count": 0,
        "empty_gallery_gids": [],
        "excluded_gid_count": 0,
        "excluded_gids": [],
        "best_candidate_gid": None,
        "best_cosine_distance": None,
        "second_best_candidate_gid": None,
        "second_best_cosine_distance": None,
        "actual_threshold": None,
        "required_ambiguity_margin": REID_AMBIGUITY_MARGIN,
        "actual_margin": None,
        "best_candidate_excluded": False,
        "best_candidate_template_count": 0,
        "best_candidate_gallery_empty": None,
        "assigned_gid_template_count": 0,
        "assigned_gid_gallery_empty": None,
        "new_gid_template_added": None,
        "new_gid_template_rejection_reason": None,
        "decision_source": "osnet",
        "osnet_matching_performed": False,
        "local_binding_status": "unbound",
        "matching_result": None,
        "decision": None,
        "assigned_gid": None,
        "reason": None,
    }
    
    # Adaptive threshold for small (far-away) crops starting from central configured threshold
    actual_threshold = threshold
    if box_area is not None and box_area < 6000:
        fraction = max(0.0, min(1.0, (6000.0 - box_area) / 6000.0))
        actual_threshold = threshold + fraction * 0.08  # scale threshold up slightly for smaller, noisier crops
    diagnostic["actual_threshold"] = actual_threshold
        
    # Quality filter: completely bypass gallery matching for extremely small boxes
    if box_area is not None and box_area < 100:
        gid = -(CAMERA_INDEX[camera_id] * 100000 + local_id + 1)
        diagnostic.update(decision_source="osnet", decision="small_box", matching_result="not_attempted", assigned_gid=gid,
                          reason="box_area_below_minimum_for_matching")
        _global_id_diag_once(key, diagnostic)
        return gid
        
    allow_new = (box_area is None or box_area >= 1200)
    fallback_id = -(CAMERA_INDEX[camera_id] * 100000 + local_id + 1)

    if embedding is None:
        diagnostic.update(decision="embedding_unavailable", matching_result="embedding_unavailable",
                          reason="track_has_no_embedding_feature")
        _global_id_diag_once(key, diagnostic)
    
    # Try to match face if visible
    face_gid = None
    if run_face_detection and key not in face_aligned_tracks:
        now_mono = time.monotonic()
        if now_mono - track_last_face_check.get(key, 0.0) >= 1.0:
            track_last_face_check[key] = now_mono
            if (LIVE_DEMO_MODE or camera_id == "live" or (box_area is not None and box_area >= 4000)) and frame is not None and bbox is not None:
                try:
                    x1, y1, x2, y2 = bbox
                    h_img, w_img, _ = frame.shape
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))
                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2]
                        with reid_lock:
                            q_face = detect_face(crop)
                            if q_face is not None:
                                q_face_emb = extract_face_embedding(crop, q_face)
                            else:
                                q_face_emb = None
                        
                        if q_face_emb is not None:
                            best_face_gid = None
                            best_face_sim = 0.0
                            with gallery_lock:
                                for g_id, f_embs in global_face_gallery.items():
                                    for f_emb in f_embs:
                                        sim = compute_face_similarity(q_face_emb, f_emb)
                                        if sim > best_face_sim:
                                            best_face_sim = sim
                                            best_face_gid = g_id
                                            
                            # Mark that face extraction has been completed successfully for this track
                            face_aligned_tracks.add(key)
                            
                            if best_face_sim >= 0.45:
                                face_gid = best_face_gid
                                print(f"  [FACE MATCH] {camera_id} local {local_id} -> GID {face_gid} (sim: {best_face_sim:.3f})")
                except Exception as e_face:
                    print(f"[LIVE FACE MATCH EXCEPTION] {e_face}")

    with gallery_lock:
        had_positive_binding = key in local_to_global and local_to_global[key] > 0
        if key in local_to_global and local_to_global[key] <= 0:
            diagnostic["local_binding_status"] = "negative_fallback_recheck"
        diagnostic["gallery_gid_count"] = len(global_gallery)
        diagnostic["gallery_template_count"] = sum(
            len(templates) for cam_templates in global_gallery.values()
            for templates in cam_templates.values()
        )
        if had_positive_binding:
            diagnostic["decision_source"] = "existing_binding"
            diagnostic["decision"] = "existing_binding"
            diagnostic["local_binding_status"] = "positive_binding"
        if face_gid is not None:
            diagnostic["decision_source"] = "face"
            old_gid = local_to_global.get(key)
            if old_gid is not None and old_gid > 0 and old_gid != face_gid:
                print(f"  [FACE ID CORRECTION] Correcting {camera_id} local {local_id} from GID {old_gid} to GID {face_gid}")
                
                # 1. Update in-memory local_to_global mappings for all keys mapped to old_gid
                for k, v in list(local_to_global.items()):
                    if v == old_gid:
                        local_to_global[k] = face_gid
                        
                # 2. Merge in-memory history states for the current camera
                state = camera_states.get(camera_id)
                if state:
                    beh_lock = state["behavior_lock"]
                    with beh_lock:
                        old_h = state["person_history"].get(old_gid)
                        new_h = state["person_history"].setdefault(face_gid, _empty_history())
                        if old_h:
                            new_h["positions"].extend(old_h["positions"])
                            new_h["feat_buffer"].extend(old_h["feat_buffer"])
                            new_h["zones_visited"].update(old_h["zones_visited"])
                            for z, count in old_h["zone_entry_count"].items():
                                new_h["zone_entry_count"][z] = new_h["zone_entry_count"].get(z, 0) + count
                            if old_h["flagged"]: new_h["flagged"] = True
                            if old_h["manually_flagged"]: new_h["manually_flagged"] = True
                            new_h["prev_foot"] = old_h["prev_foot"] or new_h["prev_foot"]
                            new_h["last_seen"] = max(old_h.get("last_seen", 0.0), new_h.get("last_seen", 0.0))
                            state["person_history"].pop(old_gid, None)
                            
                        # Merge dwell entry times
                        old_dwell = state["dwell_entry_times"].get(old_gid)
                        if old_dwell:
                            new_dwell = state["dwell_entry_times"].setdefault(face_gid, {})
                            for zone, t in old_dwell.items():
                                new_dwell.setdefault(zone, t)
                            state["dwell_entry_times"].pop(old_gid, None)
                            
                # 3. Merge galleries
                if old_gid in global_gallery:
                    new_cam_templates = global_gallery.setdefault(face_gid, {})
                    for cam_id, templates in global_gallery[old_gid].items():
                        dest = new_cam_templates.setdefault(cam_id, deque(maxlen=GALLERY_TEMPLATES_PER_ID))
                        dest.extend(templates)
                    global_gallery.pop(old_gid, None)
                    
                if old_gid in global_face_gallery:
                    new_face_embs = global_face_gallery.setdefault(face_gid, [])
                    new_face_embs.extend(global_face_gallery[old_gid])
                    global_face_gallery.pop(old_gid, None)
                    
                if old_gid in global_last_seen:
                    global_last_seen[face_gid] = max(global_last_seen.get(face_gid, 0.0), global_last_seen.get(old_gid, 0.0))
                    global_last_seen.pop(old_gid, None)
                    
                # 4. Queue the DB merge task
                db_queue.put(("merge_persons", {
                    "old_gid": old_gid,
                    "new_gid": face_gid
                }))
                
                # Structured Logging for FACE_MATCH correction (Phase 11)
                gallery_size = sum(len(v) for v in global_gallery[face_gid].values()) if face_gid in global_gallery else 0
                timestamp_str = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp_str}]")
                print(f"Camera: {camera_id}")
                print(f"Local ID: {local_id}")
                print(f"Area: {box_area}")
                print(f"Conf: {confidence:.2f}")
                print(f"Best GID: {face_gid}")
                print(f"Best Distance: 0.00")
                print(f"Best Similarity: {best_face_sim:.2f}")
                print(f"Second GID: N/A")
                print(f"Second Distance: N/A")
                print(f"Margin: N/A")
                print(f"Gallery Size: {gallery_size}")
                print(f"Decision: FACE_MATCH (CORRECTION from GID {old_gid})")
            else:
                local_to_global[key] = face_gid
                
                # Structured Logging for FACE_MATCH (Phase 11)
                gallery_size = sum(len(v) for v in global_gallery[face_gid].values()) if face_gid in global_gallery else 0
                timestamp_str = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp_str}]")
                print(f"Camera: {camera_id}")
                print(f"Local ID: {local_id}")
                print(f"Area: {box_area}")
                print(f"Conf: {confidence:.2f}")
                print(f"Best GID: {face_gid}")
                print(f"Best Distance: 0.00")
                print(f"Best Similarity: {best_face_sim:.2f}")
                print(f"Second GID: N/A")
                print(f"Second Distance: N/A")
                print(f"Margin: N/A")
                print(f"Gallery Size: {gallery_size}")
                print(f"Decision: FACE_MATCH")

        # Check if already resolved to a positive global ID
        if key in local_to_global and local_to_global[key] > 0:
            gid = local_to_global[key]
            if face_gid is not None:
                diagnostic.update(decision="face_match", matching_result="face_match", reason="face_similarity_passed")
            elif not had_positive_binding:
                diagnostic.update(decision="face_match", matching_result="face_match", reason="face_similarity_passed")
            else:
                diagnostic.update(decision="existing_binding", matching_result="existing_binding",
                                  reason="positive_local_binding_reused")
            diagnostic["assigned_gid"] = gid
            global_last_seen[gid] = seen_at
            
            # Phase 5/6: Prevent gallery contamination by validating crop quality before inserting
            is_quality_ok = (confidence >= GALLERY_MIN_CONFIDENCE) and (box_area is not None and box_area >= GALLERY_MIN_BOX_AREA)
            if is_quality_ok:
                cam_templates = global_gallery.setdefault(gid, {})
                templates = cam_templates.setdefault(camera_id, deque(maxlen=GALLERY_TEMPLATES_PER_ID))
                
                if len(templates) == 0:
                    templates.append(embedding)
                    timestamp_str = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp_str}] [GALLERY UPDATE] Appended initial template to GID {gid} on camera {camera_id}")
                else:
                    min_dist = min(_cos_dist(embedding, t) for t in templates)
                    if GALLERY_MIN_DIST < min_dist < GALLERY_MAX_DIST:
                        templates.append(embedding)
                        timestamp_str = datetime.now().strftime("%H:%M:%S")
                        print(f"[{timestamp_str}] [GALLERY UPDATE] Appended new template to GID {gid} on camera {camera_id} (min_dist: {min_dist:.3f})")
        else:
            # Not yet matched or mapped to negative fallback. Try to match it against gallery.
            candidates = []
            empty_gallery_gid_count = 0
            excluded_gid_count = 0
            for m_gid, cam_templates in global_gallery.items():
                if exclude_gids and m_gid in exclude_gids:
                    excluded_gid_count += 1
                    diagnostic["excluded_gids"].append(m_gid)
                    continue
                if seen_at - global_last_seen.get(m_gid, 0.0) > REID_RECENT_WINDOW_SECONDS:
                    continue
                
                # Find closest distance across all templates for all cameras of this GID
                all_dists = []
                for c_id, templates in cam_templates.items():
                    for t in templates:
                        all_dists.append(_cos_dist(embedding, t))
                if not all_dists:
                    empty_gallery_gid_count += 1
                    diagnostic["empty_gallery_gids"].append(m_gid)
                    continue
                
                min_dist = min(all_dists)
                candidates.append((min_dist, m_gid))
                
            candidates.sort(key=lambda c: c[0])
            diagnostic["osnet_matching_performed"] = True
            diagnostic["candidate_gid_count"] = len(candidates)
            diagnostic["empty_gallery_gid_count"] = empty_gallery_gid_count
            diagnostic["excluded_gid_count"] = excluded_gid_count
            diagnostic["actual_threshold"] = actual_threshold

            best_gid = None
            best_dist = 1.0
            best_sim = 0.0
            second_best_gid = None
            second_best_dist = 1.0
            margin = 1.0

            if candidates:
                best_dist, best_gid = candidates[0]
                best_sim = 1.0 - best_dist
                if len(candidates) >= 2:
                    second_best_dist, second_best_gid = candidates[1]
                    margin = second_best_dist - best_dist
                else:
                    second_best_dist = 1.0
                    second_best_gid = None
                    margin = 1.0 - best_dist
            else:
                margin = 0.0

            # Phase 3: Implement proper ambiguity checking
            is_confident_match = (best_dist <= actual_threshold) and (margin >= REID_AMBIGUITY_MARGIN)
            diagnostic.update(
                best_candidate_gid=best_gid,
                best_cosine_distance=best_dist if best_gid is not None else None,
                second_best_candidate_gid=second_best_gid,
                second_best_cosine_distance=second_best_dist if second_best_gid is not None else None,
                actual_margin=margin if best_gid is not None else None,
                best_candidate_template_count=(
                    sum(len(v) for v in global_gallery.get(best_gid, {}).values()) if best_gid is not None else 0
                ),
                best_candidate_gallery_empty=(
                    not any(global_gallery.get(best_gid, {}).values()) if best_gid is not None else None
                ),
            )

            if is_confident_match:
                # Check if this match recovers an old coasting ghost track's Global ID
                if best_gid in ghost_claims:
                    old_local_id, old_status = ghost_claims[best_gid]
                    decision = "RECOVER_EXISTING_GLOBAL_ID"
                    log_ghost_track_transition(
                        camera_id=camera_id,
                        old_local_id=old_local_id,
                        old_gid=best_gid,
                        new_local_id=local_id,
                        best_gid=best_gid,
                        best_dist=best_dist,
                        second_gid=second_best_gid,
                        second_dist=second_best_dist,
                        margin=margin,
                        decision="RECOVER_EXISTING_GLOBAL_ID",
                        reason="STRONG_REID_MATCH"
                    )
                    # Release/rebind the old ghost track mapping so it cannot reclaim GID
                    old_key = (camera_id, old_local_id)
                    if old_key in local_to_global:
                        del local_to_global[old_key]
                else:
                    # Classify match decision (Phase 9 & 11)
                    has_other_cameras = any(cid != camera_id for cid in global_gallery[best_gid].keys())
                    if has_other_cameras:
                        decision = "CROSS_CAMERA_MATCH"
                    else:
                        decision = "MATCH_EXISTING"
                
                gid = best_gid
                diagnostic.update(decision="match", matching_result="match",
                                  reason="distance_and_ambiguity_margin_passed")
                local_to_global[key] = gid
                global_last_seen[gid] = seen_at
                
                # Quality check before gallery insertion
                is_quality_ok = (confidence >= GALLERY_MIN_CONFIDENCE) and (box_area is not None and box_area >= GALLERY_MIN_BOX_AREA)
                if is_quality_ok:
                    cam_templates = global_gallery.setdefault(gid, {})
                    templates = cam_templates.setdefault(camera_id, deque(maxlen=GALLERY_TEMPLATES_PER_ID))
                    if len(templates) == 0:
                        templates.append(embedding)
                        timestamp_str = datetime.now().strftime("%H:%M:%S")
                        print(f"[{timestamp_str}] [GALLERY UPDATE] Appended initial template to GID {gid} on camera {camera_id}")
                    else:
                        min_dist_to_templates = min(_cos_dist(embedding, t) for t in templates)
                        if GALLERY_MIN_DIST < min_dist_to_templates < GALLERY_MAX_DIST:
                            templates.append(embedding)
                            timestamp_str = datetime.now().strftime("%H:%M:%S")
                            print(f"[{timestamp_str}] [GALLERY UPDATE] Appended new template to GID {gid} on camera {camera_id} (min_dist: {min_dist_to_templates:.3f})")
            else:
                # If a ghost track candidate was evaluated but could not be confidently matched, log transition
                if best_gid is not None and best_gid in ghost_claims:
                    old_local_id, old_status = ghost_claims[best_gid]
                    if best_dist <= actual_threshold and margin < REID_AMBIGUITY_MARGIN:
                        log_ghost_track_transition(
                            camera_id=camera_id,
                            old_local_id=old_local_id,
                            old_gid=best_gid,
                            new_local_id=local_id,
                            best_gid=best_gid,
                            best_dist=best_dist,
                            second_gid=second_best_gid,
                            second_dist=second_best_dist,
                            margin=margin,
                            decision="AMBIGUOUS_MATCH",
                            reason="AMBIGUITY_MARGIN_NOT_MET"
                        )
                    else:
                        log_ghost_track_transition(
                            camera_id=camera_id,
                            old_local_id=old_local_id,
                            old_gid=best_gid,
                            new_local_id=local_id,
                            best_gid=best_gid,
                            best_dist=best_dist,
                            second_gid=second_best_gid,
                            second_dist=second_best_dist,
                            margin=margin,
                            decision="NEW_GLOBAL_ID" if allow_new else "FALLBACK_ID",
                            reason="DISTANCE_EXCEEDS_THRESHOLD"
                        )

                # No confident match: check if registration is allowed or fallback
                if best_dist <= actual_threshold and margin < REID_AMBIGUITY_MARGIN:
                    decision = "UNCERTAIN_MATCH"
                elif candidates:
                    decision = "NEW_GLOBAL_ID" if allow_new else "FALLBACK_ID"
                else:
                    decision = "NEW_GLOBAL_ID" if allow_new else "FALLBACK_ID"

                if allow_new and decision != "FALLBACK_ID":
                    # Register new Global ID
                    gid = next_global_id
                    next_global_id += 1
                    local_to_global[key] = gid
                    global_last_seen[gid] = seen_at
                    
                    # Phase 6: Ensure initial gallery embedding passes quality checks
                    is_quality_ok = (confidence >= GALLERY_MIN_CONFIDENCE) and (box_area is not None and box_area >= GALLERY_MIN_BOX_AREA)
                    if is_quality_ok:
                        global_gallery[gid] = {camera_id: deque([embedding], maxlen=GALLERY_TEMPLATES_PER_ID)}
                        diagnostic["new_gid_template_added"] = True
                        timestamp_str = datetime.now().strftime("%H:%M:%S")
                        print(f"[{timestamp_str}] [GALLERY UPDATE] Created new GID {gid} gallery on camera {camera_id} with initial template")
                    else:
                        global_gallery[gid] = {}
                        diagnostic["new_gid_template_added"] = False
                        diagnostic["new_gid_template_rejection_reason"] = (
                            "confidence_below_gallery_minimum" if confidence < GALLERY_MIN_CONFIDENCE
                            else "box_area_below_gallery_minimum"
                        )
                        timestamp_str = datetime.now().strftime("%H:%M:%S")
                        print(f"[{timestamp_str}] [GALLERY WARNING] Created new GID {gid} with empty gallery (initial crop failed quality checks)")
                    diagnostic.update(
                        decision="new_gid",
                        matching_result=("rejected_ambiguity" if candidates and best_dist <= actual_threshold
                                         and margin < REID_AMBIGUITY_MARGIN else
                                         "rejected_distance" if candidates else
                                         "empty_gallery" if empty_gallery_gid_count else "no_candidate"),
                        reason=("rejected_ambiguity_new_gid" if candidates and best_dist <= actual_threshold
                                and margin < REID_AMBIGUITY_MARGIN else
                                "rejected_distance_new_gid" if candidates else
                                "empty_gallery_templates_unavailable" if empty_gallery_gid_count else
                                "no_eligible_gallery_candidates"),
                    )
                else:
                    gid = fallback_id
                    local_to_global[key] = gid
                    if not allow_new:
                        decision = "FALLBACK_ID"
                    diagnostic.update(
                        decision="negative_fallback",
                        matching_result=("rejected_ambiguity" if candidates and best_dist <= actual_threshold
                                         and margin < REID_AMBIGUITY_MARGIN else
                                         "rejected_distance" if candidates else
                                         "empty_gallery" if empty_gallery_gid_count else "no_candidate"),
                        reason=("rejected_ambiguity_new_gid_disallowed" if candidates and best_dist <= actual_threshold
                                and margin < REID_AMBIGUITY_MARGIN else
                                "rejected_distance_new_gid_disallowed" if candidates else
                                "no_eligible_gallery_candidate_new_gid_disallowed"),
                    )

            diagnostic["assigned_gid"] = gid

            # Structured Logging for new decisions (Phase 11)
            gallery_size = sum(len(v) for v in global_gallery[best_gid].values()) if (best_gid is not None and best_gid in global_gallery) else 0
            timestamp_str = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp_str}]")
            print(f"Camera: {camera_id}")
            print(f"Local ID: {local_id}")
            print(f"Area: {box_area}")
            print(f"Conf: {confidence:.2f}")
            print(f"Best GID: {best_gid if best_gid is not None else 'N/A'}")
            print(f"Best Distance: {best_dist:.2f}")
            print(f"Best Similarity: {best_sim:.2f}")
            print(f"Second GID: {second_best_gid if second_best_gid is not None else 'N/A'}")
            print(f"Second Distance: {second_best_dist:.2f}")
            print(f"Margin: {margin:.2f}")
            print(f"Gallery Size: {gallery_size}")
            print(f"Decision: {decision}")

        if diagnostic["decision"] is None:
            diagnostic.update(decision="negative_fallback", matching_result="no_candidate",
                              reason="no_positive_binding_or_match")
        diagnostic["assigned_gid"] = gid
        if gid > 0:
            assigned_templates = global_gallery.get(gid, {})
            diagnostic["assigned_gid_template_count"] = sum(len(v) for v in assigned_templates.values())
            diagnostic["assigned_gid_gallery_empty"] = diagnostic["assigned_gid_template_count"] == 0
        _global_id_diag_once(key, diagnostic)

    # If it is positive, queue the database sync sighting task
    if gid > 0:
        sighting_key = (camera_id, gid)
        last_seen_info = last_db_sighting_time.get(sighting_key)
        now_monotonic = time.monotonic()
        
        should_sighting = (
            last_seen_info is None
            or last_seen_info[1] != current_zone
            or now_monotonic - last_seen_info[0] > 180.0
        )
        
        if should_sighting:
            last_db_sighting_time[sighting_key] = (now_monotonic, current_zone)
            
            snapshot_crop = None
            full_save_path = None
            snapshot_web_path = None
            
            if frame is not None and bbox is not None:
                try:
                    x1, y1, x2, y2 = bbox
                    h_img, w_img, _ = frame.shape
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))
                    if x2 > x1 and y2 > y1:
                        snapshot_crop = frame[y1:y2, x1:x2].copy()
                        app_dir = os.path.dirname(os.path.abspath(__file__))
                        static_dir = os.path.join(app_dir, "static", "snapshots")
                        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
                        snapshot_filename = f"person_{gid}_{timestamp_str}.jpg"
                        full_save_path = os.path.join(static_dir, snapshot_filename)
                        snapshot_web_path = f"static/snapshots/{snapshot_filename}"
                except Exception as e_crop:
                    print(f"[CROP SNAPSHOT PREPARE EXCEPTION] {e_crop}")
                    
            db_queue.put(("register_person", {
                "gid": gid,
                "camera_id": camera_id,
                "embedding_bytes": embedding.astype(np.float32).tobytes(),
                "confidence": float(confidence),
                "box_area": box_area,
                "snapshot_crop": snapshot_crop,
                "full_save_path": full_save_path,
                "snapshot_web_path": snapshot_web_path,
                "current_zone": current_zone
            }))
        
    return gid


def classify_track_status(tracker, track_id):
    """Classify track state as ACTIVE, COASTING, or STALE based on tracker time_since_update."""
    if hasattr(tracker, 'tracker') and hasattr(tracker.tracker, 'tracks'):
        for t in tracker.tracker.tracks:
            if t.id == track_id:
                tsu = getattr(t, 'time_since_update', 0)
                if tsu == 0:
                    return "ACTIVE"
                elif tsu <= 15:
                    return "COASTING"
                else:
                    return "STALE"
    return "ACTIVE"


def log_ghost_track_transition(camera_id, old_local_id, old_gid, new_local_id,
                               best_gid, best_dist, second_gid, second_dist,
                               margin, decision, reason):
    """Targeted diagnostic logging for ghost track transitions."""
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[GHOST_TRACK_TRANSITION]")
    print(f"timestamp={timestamp_str}")
    print(f"camera={camera_id}")
    print(f"old_local={old_local_id}")
    print(f"old_gid={old_gid}")
    print(f"new_local={new_local_id}")
    print(f"best_gid={best_gid if best_gid is not None else 'N/A'}")
    print(f"best_distance={best_dist:.2f}")
    print(f"second_gid={second_gid if second_gid is not None else 'N/A'}")
    print(f"second_distance={second_dist:.2f}")
    print(f"margin={margin:.2f}")
    print(f"old_gid_excluded_before_fix=true")
    print(f"old_gid_considered_after_fix=true")
    print(f"final_decision={decision}")
    print(f"decision_reason={reason}\n")


def reserve_existing_global_ids(camera_key, tracks, track_status_map=None):
    """Reserve mapped IDs before matching new tracks in the same frame.

    Differentiates between:
    - active_claimed: GIDs bound to local tracks actively detected in this frame.
      These GIDs are strictly protected against duplicate assignment.
    - ghost_claims: {gid: (old_local_id, status)} bound to coasting/stale tracks that
      were Kalman-predicted without a real detection in this frame. These GIDs are
      allowed to be reclaimed by a new track with strong Re-ID evidence.
    """
    if track_status_map is None:
        track_status_map = {}

    active_claimed = set()
    ghost_claims = {}
    claimed_all = set()

    with gallery_lock:
        for track in tracks:
            local_id = int(track[4])
            key = (camera_key, local_id)
            gid = local_to_global.get(key)
            if gid is None or gid <= 0:
                continue

            # Remove duplicate mappings for the same camera view
            if gid in claimed_all:
                del local_to_global[key]
                continue

            claimed_all.add(gid)
            status = track_status_map.get(local_id, "ACTIVE")
            if status == "ACTIVE":
                active_claimed.add(gid)
            else:
                ghost_claims[gid] = (local_id, status)

    return active_claimed, ghost_claims


def prune_identity_state(now):
    """Bound Re-ID and behavior state while retaining recent cross-camera IDs."""
    protected = global_flagged_ids | global_manually_flagged_ids | tracked_global_ids
    now_monotonic = time.monotonic()
    with gallery_lock:
        candidates = sorted(global_last_seen, key=global_last_seen.get)
        for gid in candidates:
            expired = now_monotonic - global_last_seen[gid] > STATE_TTL_SECONDS
            over_limit = len(global_gallery) > MAX_GALLERY_IDENTITIES
            if not (expired or over_limit) or gid in protected:
                continue
            global_gallery.pop(gid, None)
            global_last_seen.pop(gid, None)
            for key, mapped_gid in list(local_to_global.items()):
                if mapped_gid == gid:
                    del local_to_global[key]

    for state in camera_states.values():
        with state["behavior_lock"]:
            for gid, history in list(state["person_history"].items()):
                if gid not in protected and now - history.get("last_seen", now) > STATE_TTL_SECONDS:
                    del state["person_history"][gid]


def deduplicate_tracks(tracks, iou_threshold=0.95):
    """Drop invalid or near-identical tracker outputs without suppressing people close together."""
    if tracks is None or len(tracks) == 0:
        return tracks
    kept = []
    seen_track_ids = set()
    for track in tracks:
        x1, y1, x2, y2 = map(float, track[:4])
        track_id = int(track[4])
        if x2 <= x1 or y2 <= y1 or track_id in seen_track_ids:
            continue
        box = (x1, y1, x2, y2)
        if any(bbox_iou(box, tuple(map(float, other[:4])) ) >= iou_threshold for other in kept):
            continue
        kept.append(track)
        seen_track_ids.add(track_id)
    return np.asarray(kept) if kept else np.empty((0, 7))



def compute_rule_score_camera(camera_key, global_id, foot_x, foot_y, zone_name, now,
                               prev_position=None):
    """
    Rule-based suspicion scoring (tuned thresholds).
    Returns (newly_flagged: bool, rule_score: float).
    Must be called with camera_states[camera_key]["behavior_lock"] held.
    """
    state = camera_states[camera_key]
    h = state["person_history"].setdefault(global_id, _empty_history())

    h["positions"].append((foot_x, foot_y))

    last_time = h.get("last_time") or now
    dt = min(1.0, max(0.0, now - last_time))
    h["last_time"] = now
    if dt == 0:
        dt = 0.1

    transition_penalty = 0
    if zone_name != h["last_zone"]:
        if zone_name == h["candidate_zone"]:
            h["zone_settle_count"] += 1
            if h["zone_settle_count"] >= 10:
                h["zone_entry_count"][zone_name] = h["zone_entry_count"].get(zone_name, 0) + 1
                h["zones_visited"].add(zone_name)
                h["last_zone"] = zone_name
                h["zone_settle_count"] = 0
                h["candidate_zone"] = None

                total_reentries = sum(v for v in h["zone_entry_count"].values())
                if total_reentries >= ZONE_HOP_LIMIT:
                    transition_penalty += 20
                reentry_count = h["zone_entry_count"].get(zone_name, 0)
                if reentry_count >= 2:
                    transition_penalty += 15
        else:
            h["candidate_zone"] = zone_name
            h["zone_settle_count"] = 1
    else:
        h["candidate_zone"] = None
        h["zone_settle_count"] = 0

    h["score"] = h["score"] * (SCORE_DECAY ** (dt * 10))

    frame_rate_score = 0
    breakdown = {}

    # 1 — Loitering
    if global_id in state["dwell_entry_times"] and zone_name in state["dwell_entry_times"][global_id]:
        dwell = now - state["dwell_entry_times"][global_id][zone_name]
        if dwell > LOITER_THRESHOLD:
            frame_rate_score += 4.0 * dt
            breakdown["loitering"] = round(dwell, 1)

    # 2 — Erratic movement
    if len(h["positions"]) >= 10:
        pts    = np.array(h["positions"])
        deltas = np.diff(pts, axis=0)
        mags   = np.sqrt(deltas[:, 0]**2 + deltas[:, 1]**2)
        moving = deltas[mags > 5]
        if len(moving) >= 6:
            angles    = np.arctan2(moving[:, 1], moving[:, 0])
            angle_std = np.std(angles)
            if angle_std > 1.8:
                frame_rate_score += 3.0 * dt
                breakdown["erratic"] = round(angle_std, 2)

    # 3 & 4 — Zone hop and reentry
    if transition_penalty > 0:
        h["score"] += transition_penalty
        if "zone_hop" in h["zone_entry_count"]:
            breakdown["zone_hop"] = sum(h["zone_entry_count"].values())
        if zone_name in h["zone_entry_count"] and h["zone_entry_count"][zone_name] >= 2:
            breakdown["reentry"] = h["zone_entry_count"][zone_name]


    # 6 — Restricted zone
    if zone_name in RESTRICTED_ZONES:
        frame_rate_score += 10.0 * dt
        breakdown["restricted_zone"] = zone_name

    h["score"] = min(100.0, h["score"] + frame_rate_score)
    h["score_breakdown"] = breakdown

    newly_flagged = (
        h["score"] >= FLAG_THRESHOLD
        and global_id not in global_flagged_ids
        and global_id not in global_manually_flagged_ids
    )
    return newly_flagged, h["score"]


def compute_ml_score_camera(camera_key, global_id):
    """
    Returns ML anomaly score (0–100) for a person on a specific camera.
    Uses the last SEQ_LEN feature vectors in their feat_buffer.
    Returns 0.0 if model not trained or buffer not full yet.
    """
    with ae_lock:
        if not ae_trained or ae_model_obj is None:
            return 0.0
        m   = ae_model_obj
        thr = ae_threshold

    state = camera_states[camera_key]
    with state["behavior_lock"]:
        h = state["person_history"].get(global_id)
        if h is None or len(h["feat_buffer"]) < SEQ_LEN:
            return 0.0
        buf = list(h["feat_buffer"])

    x = seq_to_tensor(buf)
    err = m.reconstruction_error(x)[0]
    return float(min(100.0, (err / (thr + 1e-9)) * 100.0))


def _empty_history():
    return {
        "positions":         deque(maxlen=30),
        "zones_visited":     set(),
        "zone_entry_count":  {},
        "last_zone":         None,
        "candidate_zone":    None,
        "zone_settle_count": 0,
        "score":             0.0,
        "score_breakdown":   {},
        "flagged":           False,
        "manually_flagged":  False,
        "feat_buffer":       deque(maxlen=SEQ_LEN),
        "ml_score":          0.0,
        "hybrid_score":      0.0,
        "prev_foot":         None,
        "last_time":         None,
    }



def draw_alert_banner(frame, zone_name, zone_data):
    x1, y1, x2, y2 = zone_data["coords"]
    sx1 = int(x1 * (FRAME_W / 1060.0))
    sy1 = int(y1 * (FRAME_H / 660.0))
    sx2 = int(x2 * (FRAME_W / 1060.0))
    sy2 = int(y2 * (FRAME_H / 660.0))
    overlay = frame.copy()
    cv2.rectangle(overlay, (sx1, sy1), (sx2, sy2), (0, 0, 220), -1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cx = (sx1 + sx2) // 2
    cv2.putText(frame, "! OVERCROWDED", (cx - 80, sy2 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame


def reset_tracker_tracks(vid):
    t = trackers[vid]
    if t is None:
        return
    try:
        t.tracker.tracks = []
    except AttributeError:
        pass
    try:
        t.tracker.metric.samples = {}
    except AttributeError:
        pass


def clear_camera_track_bindings(camera_key):
    """Forget only local tracker IDs; preserve global gallery identities."""
    with gallery_lock:
        for key in list(local_to_global):
            if key[0] == camera_key:
                del local_to_global[key]
    for key in list(face_aligned_tracks):
        if key[0] == camera_key:
            face_aligned_tracks.discard(key)
    for key in list(track_last_face_check):
        if key[0] == camera_key:
            del track_last_face_check[key]
    with gallery_lock:
        _global_id_diagnostic_seen.difference_update(
            diagnostic_key for diagnostic_key in list(_global_id_diagnostic_seen)
            if diagnostic_key[0] == camera_key
        )


def reset_camera_stats(camera_key, keep_flagged=True):
    state = camera_states[camera_key]
    state["prev_dets"] = np.empty((0, 6))
    state["known_ids"] = set()
    state["eval_tp"] = 0
    state["eval_fp"] = 0
    state["eval_fn"] = 0

    with gallery_lock:
        for key in list(local_to_global.keys()):
            if key[0] == camera_key:
                gid = local_to_global[key]
                if not (keep_flagged and (gid in global_flagged_ids or gid in global_manually_flagged_ids)):
                    del local_to_global[key]
                    _global_id_diagnostic_seen.difference_update(
                        diagnostic_key for diagnostic_key in list(_global_id_diagnostic_seen)
                        if diagnostic_key[0] == camera_key and diagnostic_key[1] == key[1]
                    )

    for z in ZONES:
        state["zone_counts"][z] = 0
        state["zone_footfall"][z] = 0
        state["zone_avg_dwell"][z] = 0.0
        state["zone_dwell_samples"][z] = []
        state["prev_zone_ids"][z] = set()
        state["zone_alerts"][z] = False

    state["dwell_entry_times"].clear()
    with state["behavior_lock"]:
        if keep_flagged:
            for gid in list(state["person_history"].keys()):
                ph = state["person_history"][gid]
                if gid in global_flagged_ids or gid in global_manually_flagged_ids:
                    ph["positions"].clear()
                    ph["feat_buffer"].clear()
                    ph["prev_foot"] = None
                    ph["score"] = 0.0
                else:
                    del state["person_history"][gid]
        else:
            state["person_history"].clear()


def open_camera_capture(source, is_live):
    if is_live and sys.platform.startswith('win'):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    if is_live:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# ── MAIN CAMERA PROCESSING LOOP ───────────────────────────────────────────────

def camera_processing_loop(camera_key):
    """
    Runs continuously for a single camera.

    Only the selected dashboard camera reads, detects, tracks, and updates
    the shared gallery. Started but hidden cameras pause so they cannot drift
    through unrelated footage and corrupt cross-camera identity templates.
    """
    global latest_frame, latest_frame_seq, known_ids, TRAINING_MODE

    state  = camera_states[camera_key]
    source = VIDEO_SOURCES[camera_key]["file"]
    is_live = isinstance(source, int)
    is_upload = (source == "upload")
    cam_idx = CAMERA_INDEX[camera_key]
    update_camera_health(camera_key, status="connecting", worker_status="starting")

    cap = None
    if not is_upload:
        cap = open_camera_capture(source, is_live)

        if is_live:
            print(f"[{camera_key}] Live camera buffer size set to 1 to prevent lag.")

        if not cap.isOpened():
            cap.release()
            update_camera_health(camera_key, status="reconnecting", last_error=f"Could not open source {source}")
            raise RuntimeError(f"could not open source {source}")
        update_camera_health(camera_key, status="connected", worker_status="running", last_error=None)

    if is_upload:
        fps = 15.0
        frame_delay = 1.0 / fps
    else:
        fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_delay = 1.0 / fps
    frame_count = 0
    processed_count = 0
    prev_tracks = []
    last_socket_emit_time = 0.0

    # FIX: Each camera thread loads its own YOLO model — no shared lock needed.
    # Previously all cameras shared one model behind yolo_lock, which serialized
    # all YOLO calls. Now each runs its own inference in true parallel.
    try:
        local_model = _load_yolo()
    except Exception:
        if cap is not None:
            cap.release()
        raise
    print(f"[{camera_key}] YOLO model loaded — running independently.")

    state["prev_dets"]          = np.empty((0, 6))
    state["eval_tp"]            = 0
    state["eval_fp"]            = 0
    state["eval_fn"]            = 0

    try:
        while True:
            t_start = time.time()

            # Hidden cameras pause instead of racing through unrelated video
            # frames and contaminating the shared Re-ID gallery.
            with state_lock:
                is_active = (camera_key == current_video)
                seek_target = camera_seek_targets[camera_key]
                if seek_target is not None:
                    camera_seek_targets[camera_key] = None
            # Modified pause logic for live/uploaded streams so they run in the background
            if not is_active and not (is_live or is_upload):
                time.sleep(0.05)
                continue
            if seek_target is not None and not is_live and not is_upload:
                cap.set(cv2.CAP_PROP_POS_FRAMES, seek_target)
                reset_tracker_tracks(camera_key)
                clear_camera_track_bindings(camera_key)
                reset_camera_stats(camera_key, keep_flagged=True)
                prev_tracks = []
                frame_count = seek_target

            # Constant frame stepping to reduce CPU load and match real-time speed
            if is_active and not is_live and not is_upload and FRAME_STEP > 1 and frame_count > 0:
                for _ in range(FRAME_STEP - 1):
                    cap.grab()
                frame_count += FRAME_STEP - 1

            if is_upload:
                q = upload_frame_queues.get(camera_key)
                if q is not None:
                    try:
                        frame_item = q.get(timeout=0.03)
                        if isinstance(frame_item, tuple) and len(frame_item) == 2:
                            captured_at, frame = frame_item
                            latency_ms = max(0.0, (time.monotonic() - captured_at) * 1000.0)
                            update_camera_health(camera_key, processing_latency_ms=round(latency_ms, 2))
                        else:  # tolerate a frame queued by older in-process code
                            frame = frame_item
                        ret = True
                    except queue.Empty:
                        ret = False
                else:
                    ret = False
            else:
                ret, frame = cap.read()

            if not ret:
                if is_live or is_upload:
                    if is_live:
                        failures = record_camera_read_failure(camera_key)
                        if failures >= 10:
                            raise RuntimeError(f"camera read failed {failures} consecutive times")
                    time.sleep(0.01)
                    continue
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    reset_tracker_tracks(camera_key)
                    reset_camera_stats(camera_key, keep_flagged=True)
                    prev_tracks = []
                    frame_count = 0
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.01)
                        continue

            if not is_upload:
                record_camera_frame(camera_key)
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            frame_count += 1
            processed_count += 1
            record_camera_frame(camera_key, processed=True)
            update_camera_health(camera_key, status="connected", worker_status="running")
            with state_lock:
                if is_upload or is_live:
                    video_positions[camera_key] = frame_count
                else:
                    video_positions[camera_key] = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            # Determine if this camera is currently shown in the browser.
            # Inactive cameras still process fully — they just skip encode + emit.
            with state_lock:
                is_active = (camera_key == current_video)
            now = time.time()



            # ── YOLO detection ──────────────────────────────────────────────────
            # FIX: conf raised from 0.2 → 0.35.
            # At conf=0.2, YOLO outputs many partial/blurred/background detections
            # that become FP in evaluation. 0.35 is the Wildtrack sweet spot —
            # keeps real detections, discards marginal ones.
            run_detection = (processed_count % DETECT_EVERY_N == 0)
            inference_started = time.perf_counter()
            if run_detection:
                results = local_model(frame, classes=[0], verbose=False, conf=0.35, iou=0.50, imgsz=480)
                dets = np.empty((0, 6))
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    dets = np.column_stack([
                        results[0].boxes.xyxy.cpu().numpy(),
                        results[0].boxes.conf.cpu().numpy(),
                        results[0].boxes.cls.cpu().numpy(),
                    ])
                state["prev_dets"] = dets
            else:
                dets = state["prev_dets"]
            inference_ms = (time.perf_counter() - inference_started) * 1000.0 if run_detection else 0.0

            # ── StrongSORT ──────────────────────────────────────────────────────
            tracker = get_tracker(camera_key)
            tracking_started = time.perf_counter()
            if run_detection:
                tracks = tracker.update(dets, frame)
                prev_tracks = tracks
            else:
                tracks = prev_tracks
            tracks = deduplicate_tracks(tracks)
            tracking_ms = (time.perf_counter() - tracking_started) * 1000.0 if run_detection else 0.0

            # ── Live validation scoring (Wildtrack GT) ──────────────────────────
            if not LIVE_DEMO_MODE and camera_key in ["cam1", "cam2", "cam3", "cam4", "cam5", "cam6", "cam7"]:
                if frame_count % 5 == 0:
                    tp, fp, fn = evaluate_live_frame(tracks, frame_count, camera_key)
                    state["eval_tp"] = state.get("eval_tp", 0) + tp
                    state["eval_fp"] = state.get("eval_fp", 0) + fp
                    state["eval_fn"] = state.get("eval_fn", 0) + fn

            # ── Zone overlays (only for active camera — skip drawing for bg cams) ─
            if is_active:
                overlay = frame.copy()
                for zone_name, zone_data in ZONES.items():
                    x1, y1, x2, y2 = zone_data["coords"]
                    sx1 = int(x1 * (FRAME_W / 1060.0))
                    sy1 = int(y1 * (FRAME_H / 660.0))
                    sx2 = int(x2 * (FRAME_W / 1060.0))
                    sy2 = int(y2 * (FRAME_H / 660.0))
                    color = zone_data["color"]
                    bgr   = (int(color[2]), int(color[1]), int(color[0]))
                    cv2.rectangle(overlay, (sx1, sy1), (sx2, sy2), bgr, -1)
                cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
                for zone_name, zone_data in ZONES.items():
                    x1, y1, x2, y2 = zone_data["coords"]
                    sx1 = int(x1 * (FRAME_W / 1060.0))
                    sy1 = int(y1 * (FRAME_H / 660.0))
                    sx2 = int(x2 * (FRAME_W / 1060.0))
                    sy2 = int(y2 * (FRAME_H / 660.0))
                    color = zone_data["color"]
                    bgr   = (int(color[2]), int(color[1]), int(color[0]))
                    cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), bgr, 2)
                    cv2.putText(frame, zone_name, (sx1 + 8, sy1 + 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)
                    cv2.putText(frame, zone_data["label"], (sx1 + 8, sy1 + 52),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 1)

            # ── Per-track processing (ALL cameras — needed for Re-ID + scoring) ──
            detection_data     = []
            current_zone_ids   = {z: set() for z in ZONES}
            frame_people_count = 0

            # Classify all tracks in this camera's tracker (ACTIVE, COASTING, STALE)
            track_status_map = {}
            if hasattr(tracker, 'tracker') and hasattr(tracker.tracker, 'tracks'):
                for t in tracker.tracker.tracks:
                    tsu = getattr(t, 'time_since_update', 0)
                    if tsu == 0:
                        track_status_map[t.id] = "ACTIVE"
                    elif tsu <= 15:
                        track_status_map[t.id] = "COASTING"
                    else:
                        track_status_map[t.id] = "STALE"

            # Reserve active assignments and identify coasting ghost claims
            active_claimed_gids, ghost_claims = reserve_existing_global_ids(
                camera_key, tracks, track_status_map
            )
            frame_claimed_gids = set(active_claimed_gids)

            # Process ACTIVE tracks first so genuine observations resolve and reclaim GIDs
            # before coasting ghost tracks are evaluated
            sorted_tracks = sorted(
                tracks,
                key=lambda trk: 0 if track_status_map.get(int(trk[4]), "ACTIVE") == "ACTIVE" else 1
            )

            reid_started = time.perf_counter()
            if len(sorted_tracks) > 0:
                for track in sorted_tracks:
                    x1, y1   = int(track[0]), int(track[1])
                    x2, y2   = int(track[2]), int(track[3])
                    track_id = int(track[4])
                    conf     = float(track[6]) if len(track) > 6 else 1.0
                    track_status = track_status_map.get(track_id, "ACTIVE")

                    foot_x = (x1 + x2) // 2
                    foot_y = y2

                    # Determine zone first so it can be saved in the sighting record
                    current_zone = None
                    for zone_name in ZONES:
                        if point_in_zone(foot_x, foot_y, zone_name):
                            current_zone = zone_name
                            break

                    # ── Global ID via OSNet gallery ─────────────────────────────
                    key = (camera_key, track_id)
                    with gallery_lock:
                        global_id = local_to_global.get(
                            key, -(cam_idx * 100000 + track_id + 1)
                        )

                    if run_detection:
                        try:
                            # Only evaluate Re-ID if the track is ACTIVE, or if already positively mapped
                            # (coasting ghost tracks that have been superseded should not register new GIDs)
                            should_match = (track_status == "ACTIVE") or (key in local_to_global and local_to_global[key] > 0)
                            if should_match:
                                for t in tracker.tracker.tracks:
                                    if t.id == track_id and t.features is not None and len(t.features) > 0:
                                        emb       = np.array(t.features[-1])
                                        emb       = emb / (np.linalg.norm(emb) + 1e-8)
                                        box_area  = (x2 - x1) * (y2 - y1)
                                        global_id = match_or_register(
                                            camera_key, track_id, emb,
                                            exclude_gids=frame_claimed_gids,
                                            box_area=box_area,
                                            frame=frame,
                                            bbox=(x1, y1, x2, y2),
                                            confidence=conf,
                                            current_zone=current_zone,
                                            run_face_detection=True,
                                            ghost_claims=ghost_claims
                                        )
                                        frame_claimed_gids.add(global_id)
                                        if global_id in ghost_claims:
                                            del ghost_claims[global_id]
                                        break
                        except Exception as e:
                            print(f"[REID EXCEPTION] {e}")

                    frame_people_count += 1
                    with reid_lock:
                        known_ids.add(global_id)
                        state["known_ids"].add(global_id)

                    # Update zone mapping with the resolved global_id
                    if current_zone is not None:
                        current_zone_ids[current_zone].add(global_id)
                        if global_id not in state["dwell_entry_times"]:
                            state["dwell_entry_times"][global_id] = {}
                        if current_zone not in state["dwell_entry_times"][global_id]:
                            state["dwell_entry_times"][global_id][current_zone] = now

                    dwell_now = 0.0
                    if current_zone and global_id in state["dwell_entry_times"]:
                        dwell_now = now - state["dwell_entry_times"][global_id].get(current_zone, now)

                    # ── Feature vector for AE ──────────────────────────────────
                    beh_lock = state["behavior_lock"]
                    with beh_lock:
                        h = state["person_history"].setdefault(global_id, _empty_history())
                        h["last_seen"] = now
                        prev_foot = h["prev_foot"]

                    fvec = extract_features(
                        foot_x, foot_y,
                        prev_foot[0] if prev_foot else None,
                        prev_foot[1] if prev_foot else None,
                        current_zone, dwell_now,
                        frame_w=FRAME_W, frame_h=FRAME_H,
                    )

                    with beh_lock:
                        state["person_history"][global_id]["feat_buffer"].append(fvec)
                        state["person_history"][global_id]["prev_foot"] = (foot_x, foot_y)

                    # ── Training mode: collect sequences ───────────────────────
                    if TRAINING_MODE:
                        with beh_lock:
                            buf = state["person_history"][global_id]["feat_buffer"]
                            if len(buf) == SEQ_LEN:
                                seq = np.array(buf)
                                with training_lock:
                                    training_seqs.append(seq)
                                for _ in range(SEQ_LEN // 2):
                                    buf.popleft()

                    # ── Rule-based score ───────────────────────────────────────
                    newly_flagged = False
                    with beh_lock:
                        prev_pos = None
                        positions = state["person_history"][global_id]["positions"]
                        if len(positions) > 0:
                            prev_pos = positions[-1]

                    if current_zone is not None:
                        with beh_lock:
                            newly_flagged, rule_score = compute_rule_score_camera(
                                camera_key, global_id, foot_x, foot_y, current_zone, now,
                                prev_position=prev_pos
                            )
                    else:
                        with beh_lock:
                            h = state["person_history"].setdefault(global_id, _empty_history())
                            h["positions"].append((foot_x, foot_y))
                            h["score"] *= SCORE_DECAY
                        rule_score = state["person_history"][global_id]["score"]

                    # ── ML score ───────────────────────────────────────────────
                    ml_score = compute_ml_score_camera(camera_key, global_id)

                    # ── Hybrid score ───────────────────────────────────────────
                    with ae_lock:
                        trained = ae_trained
                    if trained:
                        hybrid = ML_WEIGHT * ml_score + RULE_WEIGHT * rule_score
                    else:
                        hybrid = rule_score

                    with beh_lock:
                        h = state["person_history"].get(global_id, {})
                        h["ml_score"]     = round(ml_score, 1)
                        h["hybrid_score"] = round(hybrid, 1)
                        if hybrid >= FLAG_THRESHOLD and global_id not in global_flagged_ids and global_id not in global_manually_flagged_ids:
                            newly_flagged = True
                            global_flagged_ids.add(global_id)
                        elif hybrid < FLAG_THRESHOLD * 0.6 and global_id not in global_manually_flagged_ids:
                            if global_id in global_flagged_ids:
                                global_flagged_ids.remove(global_id)

                    if newly_flagged:
                        with beh_lock:
                            ph = state["person_history"].get(global_id, {})
                        socketio.emit("suspicious_alert", {
                            "global_id":    global_id,
                            "score":        round(hybrid, 1),
                            "ml_score":     round(ml_score, 1),
                            "breakdown":    ph.get("score_breakdown", {}),
                            "camera_key":   camera_key,
                            "camera_label": VIDEO_SOURCES[camera_key]["label"],
                        })

                        # Database logging for real (positive) IDs
                        if global_id > 0:
                            alert_type = "SUSPICIOUS_MOVEMENT"
                            if ph.get("score_breakdown", {}).get("loitering", 0) > LOITER_THRESHOLD:
                                alert_type = "LOITERING"
                            db_queue.put(("add_alert", {
                                "person_id": global_id,
                                "camera_key": camera_key,
                                "zone_name": current_zone,
                                "anomaly_score": ml_score or 0.0,
                                "message": f"Suspicious behavior detected. Hybrid Score: {hybrid:.1f}.",
                                "alert_type": alert_type
                            }))

                    with beh_lock:
                        ph           = state["person_history"].get(global_id, {})
                        is_flagged   = global_id in global_flagged_ids or global_id in global_manually_flagged_ids
                        is_manually  = global_id in global_manually_flagged_ids
                        display_score = ph.get("hybrid_score", 0.0)

                    detection_data.append({
                        "track_id":    track_id,
                        "global_id":   global_id,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "foot_x":      foot_x,
                        "foot_y":      foot_y,
                        "is_flagged":  is_flagged,
                        "is_manually": is_manually,
                        "score":       display_score,
                        "zone":        current_zone,
                    })
            reid_ms = (time.perf_counter() - reid_started) * 1000.0 if run_detection else 0.0

            # ── Bounding boxes (active camera only) ──────────────────────────────
            if is_active:
                for d in detection_data:
                    global_id  = d["global_id"]
                    x1, y1     = d["x1"], d["y1"]
                    x2, y2     = d["x2"], d["y2"]
                    is_flagged = d["is_flagged"]
                    is_manually= d["is_manually"]
                    score      = d["score"]
                    is_target  = (global_id in tracked_global_ids)

                    if is_target:
                        color, thickness = (0, 0, 255), 3
                    elif is_manually:
                        color, thickness = (0, 140, 255), 2
                    elif is_flagged:
                        color, thickness = (0, 0, 200), 2
                    else:
                        color, thickness = (50, 200, 50), 2

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    label = f"GID:{global_id} [{int(score)}]" if (is_flagged or is_target) else f"GID:{global_id}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    cv2.circle(frame, (d["foot_x"], d["foot_y"]), 5, (255, 255, 255), -1)

                    if is_target:
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        cv2.line(frame, (cx - 22, cy), (cx + 22, cy), (0, 0, 255), 2)
                        cv2.line(frame, (cx, cy - 22), (cx, cy + 22), (0, 0, 255), 2)
                        cv2.circle(frame, (cx, cy), 32, (0, 0, 255), 1)

            # ── Zone stats (ALL cameras — needed for cross-camera analytics) ──────
            for zone_name in ZONES:
                curr_ids  = current_zone_ids[zone_name]
                prev_ids  = state["prev_zone_ids"][zone_name]
                count     = len(curr_ids)
                state["zone_counts"][zone_name] = count
                state["zone_alerts"][zone_name] = (count >= ZONE_CAPACITY.get(zone_name, 10))
                state["zone_footfall"][zone_name] += len(curr_ids - prev_ids)
                for tid in prev_ids - curr_ids:
                    if tid in state["dwell_entry_times"] and zone_name in state["dwell_entry_times"][tid]:
                        duration = now - state["dwell_entry_times"][tid][zone_name]
                        state["zone_dwell_samples"][zone_name].append(duration)
                        state["zone_dwell_samples"][zone_name] = state["zone_dwell_samples"][zone_name][-20:]
                        state["zone_avg_dwell"][zone_name] = round(
                            sum(state["zone_dwell_samples"][zone_name]) /
                            len(state["zone_dwell_samples"][zone_name]), 1)
                        
                        # Database logging for real (positive) IDs
                        if tid > 0:
                            entry_timestamp = datetime.fromtimestamp(state["dwell_entry_times"][tid][zone_name])
                            db_queue.put(("add_dwell", {
                                "person_id": tid,
                                "camera_key": camera_key,
                                "zone_name": zone_name,
                                "duration": duration,
                                "entry_time": entry_timestamp
                            }))

                        del state["dwell_entry_times"][tid][zone_name]
            state["prev_zone_ids"] = {z: set(current_zone_ids[z]) for z in ZONES}

            # ── Alert banners (active camera only) ───────────────────────────────
            if is_active:
                for zone_name, zone_data in ZONES.items():
                    if state["zone_alerts"][zone_name]:
                        frame = draw_alert_banner(frame, zone_name, zone_data)

            # ── Suspicious count ─────────────────────────────────────────────────
            beh_lock = state["behavior_lock"]
            with beh_lock:
                suspicious_count = sum(
                    1 for ph in state["person_history"].values()
                    if ph.get("flagged") or ph.get("manually_flagged")
                       or ph.get("global_id") in global_flagged_ids
                )

            # ── HUD (active camera only) ─────────────────────────────────────────
            if is_active:
                with reid_lock:
                    unique_ids = len(state["known_ids"])
                with ae_lock:
                    model_status = "ML+Rules" if ae_trained else "Rules only"

                cv2.rectangle(frame, (0, 0), (FRAME_W, 60), (245, 247, 250), -1)
                hud_text = (f"AcuTrack  |  People: {frame_people_count}  |  "
                            f"IDs: {unique_ids}  |  Suspicious: {suspicious_count}  |  {model_status}")
                cv2.putText(frame, hud_text, (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (30, 30, 30), 2)
                cv2.putText(frame, VIDEO_SOURCES[camera_key]["label"],
                            (FRAME_W - 160, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 80), 2)
                if isinstance(VIDEO_SOURCES[camera_key]["file"], int):
                    cv2.circle(frame, (FRAME_W - 20, 30), 8, (0, 0, 220), -1)
                if len(tracked_global_ids) > 0:
                    locked_str = ", ".join(str(i) for i in sorted(tracked_global_ids))
                    cv2.putText(frame, f"TARGET LOCK: GID {locked_str}",
                                (14, FRAME_H - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 220), 2)

            # ── Terminal every 100 frames ─────────────────────────────────────────
            if frame_count % 100 == 0:
                prune_identity_state(now)
                precision = state["eval_tp"] / (state["eval_tp"] + state["eval_fp"] + 1e-9)
                recall    = state["eval_tp"] / (state["eval_tp"] + state["eval_fn"] + 1e-9)
                f1        = 2 * (precision * recall) / (precision + recall + 1e-9)
                tag = "[ACTIVE]" if is_active else "[BG    ]"
                print(f"\n{'='*60}")
                print(f"  {tag} [{camera_key}] FRAME {frame_count}  |  People: {frame_people_count}  |  "
                      f"IDs: {len(state['known_ids'])}")
                if state["eval_tp"] + state["eval_fp"] + state["eval_fn"] > 0:
                    print(f"  LIVE METRICS (Wildtrack GT)  |  "
                          f"Precision: {precision*100:.1f}%  |  "
                          f"Recall: {recall*100:.1f}%  |  "
                          f"F1-Score: {f1*100:.1f}%")
                print('='*60)

            # ── Encode frame + broadcast stats (active camera only) ───────────────
            with state_lock:
                is_active = camera_key == current_video
            if is_active:
                encode_started = time.perf_counter()
                ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                encode_ms = (time.perf_counter() - encode_started) * 1000.0
                if ok:
                    with frame_condition:
                        # A camera switch may race with this encode. Publish only
                        # if this frame still belongs to the selected camera.
                        if camera_key == current_video:
                            latest_frame = buffer.tobytes()
                            latest_frame_seq += 1
                            update_camera_health(camera_key, last_stream_frame_at=time.time())
                            frame_condition.notify_all()

                # Stream actual frame via Socket.IO is commented out to optimize memory & CPU.
                # The frontend now uses native MJPEG streaming from the /video_feed HTTP route.
                # now_monotonic = time.monotonic()
                # if now_monotonic - last_socket_emit_time >= (1.0 / STREAM_FPS):
                #     ok_stream, stream_buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                #     if ok_stream:
                #         import base64
                #         encoded_img = base64.b64encode(stream_buffer).decode('utf-8')
                #         socketio.emit("video_frame", {"image": encoded_img})
                #         last_socket_emit_time = now_monotonic

                if frame_count % EMIT_EVERY_N == 0:
                    stats = {"total": frame_people_count, "video": camera_key, "zones": {}}
                    for zone_name in ZONES:
                        stats["zones"][zone_name] = {
                            "count":    state["zone_counts"][zone_name],
                            "footfall": state["zone_footfall"][zone_name],
                            "dwell":    state["zone_avg_dwell"][zone_name],
                            "label":    ZONES[zone_name]["label"],
                            "alert":    state["zone_alerts"][zone_name],
                            "capacity": ZONE_CAPACITY.get(zone_name, 10),
                        }
                    socketio.emit("stats", stats)

                    with beh_lock:
                        persons_payload = [
                            {
                                "global_id":        gid,
                                "score":            round(ph.get("hybrid_score", ph.get("score", 0)), 1),
                                "rule_score":       round(ph.get("score", 0), 1),
                                "ml_score":         round(ph.get("ml_score", 0), 1),
                                "flagged":          gid in global_flagged_ids or gid in global_manually_flagged_ids,
                                "manually_flagged": gid in global_manually_flagged_ids,
                                "score_breakdown":  ph.get("score_breakdown", {}),
                                "zone":             ph.get("last_zone"),
                                "zones_visited":    list(ph.get("zones_visited", set())),
                                "zone_entry_count": ph.get("zone_entry_count", {}),
                                "present":          True,
                            }
                            for gid, ph in state["person_history"].items()
                        ]
                        
                        # Add flagged targets who are either offline or visible on another camera feed
                        active_gids = set(state["person_history"].keys())
                        all_flagged_gids = global_flagged_ids | global_manually_flagged_ids
                        for fgid in all_flagged_gids:
                            if fgid not in active_gids:
                                other_camera = None
                                now_monotonic = time.time()
                                for ocam_key, ostate in camera_states.items():
                                    if ocam_key != camera_key:
                                        with ostate["behavior_lock"]:
                                            if fgid in ostate["person_history"]:
                                                oph = ostate["person_history"][fgid]
                                                last_t = oph.get("last_time")
                                                if last_t and now_monotonic - last_t < 5.0:
                                                    other_camera = VIDEO_SOURCES.get(ocam_key, {}).get("label", ocam_key)
                                                    break
                                
                                if other_camera:
                                    persons_payload.append({
                                        "global_id":        fgid,
                                        "score":            0.0,
                                        "rule_score":       0.0,
                                        "ml_score":         0.0,
                                        "flagged":          True,
                                        "manually_flagged": fgid in global_manually_flagged_ids,
                                        "score_breakdown":  {},
                                        "zone":             f"On {other_camera}",
                                        "zones_visited":    [],
                                        "zone_entry_count": {},
                                        "present":          True,
                                    })
                                else:
                                    persons_payload.append({
                                        "global_id":        fgid,
                                        "score":            0.0,
                                        "rule_score":       0.0,
                                        "ml_score":         0.0,
                                        "flagged":          True,
                                        "manually_flagged": fgid in global_manually_flagged_ids,
                                        "score_breakdown":  {},
                                        "zone":             "Not Present",
                                        "zones_visited":    [],
                                        "zone_entry_count": {},
                                        "present":          False,
                                    })
                    with training_lock:
                        n_seqs_now = len(training_seqs)
                    with ae_lock:
                        ae_status_payload = {
                            "trained":       ae_trained,
                            "training_now":  ae_training_now,
                            "training_mode": TRAINING_MODE,
                            "seq_count":     n_seqs_now,
                            "min_seqs":      MIN_TRAIN_SEQS,
                            "threshold":     round(ae_threshold, 5),
                            "ml_weight":     ML_WEIGHT,
                        }
                    socketio.emit("person_data", {"camera_key": camera_key, "persons": persons_payload})

            record_performance(
                inference_ms if run_detection else None,
                tracking_ms if run_detection else None,
                reid_ms if run_detection else None,
                encode_ms if is_active else 0.0,
                (time.time() - t_start) * 1000.0,
            )

            # ── Pacing: active camera matches real-time; background runs free ──────
            # Background cameras run at full speed to stay warm. If CPU is overloaded,
            # add: if not is_active: time.sleep(0.01)
            if is_active:
                elapsed_proc = time.time() - t_start
                step_mult = 1 if (is_live or is_upload) else FRAME_STEP
                sleep_time = (frame_delay * step_mult) - elapsed_proc
                if sleep_time > 0:
                    time.sleep(sleep_time)
            else:
                # Small yield to avoid starving the active camera on single-core systems
                time.sleep(0.001)

    finally:
        try:
            cap.release()
        except Exception:
            pass


def ensure_camera_started(camera_key):
    """
    FIX: cameras no longer all spin up at boot. This starts a camera's
    processing thread the first time it's needed (startup for the initial
    active camera, or on switch_video for anything else) and never starts
    the same camera twice. Once started, a camera keeps running in the
    background (same warm-state behavior as before) — it's just not forced
    to start before you've ever looked at it.
    """
    with camera_threads_lock:
        existing = camera_worker_threads.get(camera_key)
        if camera_threads_started.get(camera_key) and existing and existing.is_alive():
            return
        camera_threads_started[camera_key] = True
        update_camera_health(camera_key, worker_status="starting")
        t = threading.Thread(target=camera_worker, args=(camera_key,), daemon=True,
                             name=f"camera-worker-{camera_key}")
        camera_worker_threads[camera_key] = t
        t.start()
    print(f"  [on-demand] Started camera thread: {camera_key}")


def prepare_camera_recovery(camera_key, error):
    """Reset only camera-local tracker bindings and publish its recovery state."""
    print(f"[{camera_key}] Camera worker recovering: {error}")
    update_camera_health(camera_key, status="reconnecting", worker_status="recovering", last_error=error)
    with camera_health_lock:
        camera_health[camera_key]["reconnection_attempts"] += 1
        camera_health[camera_key]["consecutive_read_failures"] = 0
    clear_camera_track_bindings(camera_key)
    with tracker_init_lock:
        trackers[camera_key] = None


def camera_worker_cycle(camera_key):
    """Run one processing lifetime and convert exits/errors into recoveries."""
    try:
        camera_processing_loop(camera_key)
        error = "Camera processing loop exited unexpectedly"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    prepare_camera_recovery(camera_key, error)
    return error


def camera_worker(camera_key):
    """Supervise one camera pipeline with bounded periodic reconnect attempts."""
    retry_delay = 1.0
    update_camera_health(camera_key, worker_status="starting")
    try:
        while True:
            try:
                camera_worker_cycle(camera_key)
            except Exception as exc:
                # Recovery itself can fail (for example, cleanup state was
                # corrupted). Keep this camera retryable and report the failure.
                error = f"Recovery error: {type(exc).__name__}: {exc}"
                print(f"[{camera_key}] {error}")
                update_camera_health(camera_key, status="recovering", worker_status="recovering", last_error=error)
                with camera_health_lock:
                    camera_health[camera_key]["reconnection_attempts"] += 1
            time.sleep(retry_delay)
            retry_delay = camera_retry_delay(retry_delay)
    finally:
        update_camera_health(camera_key, worker_status="stopped")
        with camera_threads_lock:
            camera_threads_started[camera_key] = False


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


def generate_frames():
    """Yield one new JPEG at a bounded rate; never spin on an unchanged frame."""
    last_seq = -1
    last_sent = 0.0
    while True:
        delay = (1.0 / STREAM_FPS) - (time.monotonic() - last_sent)
        if delay > 0:
            time.sleep(delay)
        with frame_condition:
            while latest_frame is None or latest_frame_seq == last_seq:
                frame_condition.wait(timeout=1.0)
            frame = latest_frame
            last_seq = latest_frame_seq
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        last_sent = time.monotonic()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    headers={'Cache-Control': 'no-store, no-cache, must-revalidate'})


@app.route('/health/metrics')
def health_metrics():
    """Process and per-camera stream health snapshot."""
    process = psutil.Process(os.getpid())
    with state_lock:
        metrics = {
            key: round(value, 2) if isinstance(value, float) else value
            for key, value in performance_metrics.items() if key != "last_report_at"
        }
        metrics["active_camera"] = current_video
        metrics["stream_fps_cap"] = STREAM_FPS
    metrics["rss_mb"] = round(process.memory_info().rss / (1024 * 1024), 2)
    metrics["threads"] = process.num_threads()
    with gallery_lock:
        metrics["gallery_identities"] = len(global_gallery)
        metrics["local_identity_mappings"] = len(local_to_global)
    now = time.time()
    with camera_health_lock:
        camera_metrics = {}
        for key, health in camera_health.items():
            snapshot = {name: value for name, value in health.items() if not name.startswith("_")}
            last_frame_at = health["last_frame_at"]
            age = round(max(0.0, now - last_frame_at), 2) if last_frame_at else None
            snapshot["last_frame_age_seconds"] = age
            last_processed_at = health["last_processed_at"]
            processed_age = round(max(0.0, now - last_processed_at), 2) if last_processed_at else None
            snapshot["last_processed_age_seconds"] = processed_age
            last_stream_frame_at = health["last_stream_frame_at"]
            stream_age = round(max(0.0, now - last_stream_frame_at), 2) if last_stream_frame_at else None
            snapshot["last_stream_frame_age_seconds"] = stream_age
            q = upload_frame_queues[key]
            snapshot["queue_depth"] = q.qsize() if VIDEO_SOURCES[key]["file"] == "upload" else 0
            oldest_age = None
            if VIDEO_SOURCES[key]["file"] == "upload":
                with q.mutex:
                    if q.queue:
                        oldest_item = q.queue[0]
                        if isinstance(oldest_item, tuple) and len(oldest_item) == 2:
                            oldest_age = max(0.0, time.monotonic() - oldest_item[0])
            snapshot["oldest_queue_frame_age_seconds"] = round(oldest_age, 3) if oldest_age is not None else None
            if VIDEO_SOURCES[key]["file"] == "upload" and (age is None or age > 5.0):
                snapshot["status"] = "stale" if last_frame_at else "waiting"
            elif VIDEO_SOURCES[key]["file"] != "upload" and last_frame_at and age > 5.0:
                snapshot["status"] = "stale"
            elif key == current_video and (stream_age is None or stream_age > 5.0):
                snapshot["status"] = "stale" if last_stream_frame_at else "waiting"
            elif health["worker_status"] in {"starting", "recovering", "stopped"}:
                snapshot["status"] = "reconnecting" if last_frame_at else "waiting"
            camera_metrics[key] = snapshot
    metrics["cameras"] = camera_metrics
    return jsonify(metrics)


@app.route('/switch_video', methods=['POST'])
def switch_video_route():
    """
    Switch the active camera view.
    FIX: starts the target camera's thread on-demand (only once, ever) if
    it hasn't run yet. Already-running cameras just get current_video
    updated; the processing loops pick it up on their next iteration.
    """
    global current_video, latest_frame, latest_frame_seq
    data    = request.get_json()
    new_vid = data.get("video")
    if new_vid in VIDEO_SOURCES:
        with state_lock:
            if new_vid == current_video:
                return jsonify({"status": "ok", "frame": video_positions[new_vid]})
            source_frame = video_positions[current_video]
            # Wildtrack views share frame numbering. Align the incoming view
            # before allowing it to emit or update Re-ID state.
            if not isinstance(VIDEO_SOURCES[new_vid]["file"], int):
                camera_seek_targets[new_vid] = source_frame
            current_video = new_vid
            latest_frame = None
            latest_frame_seq += 1
            frame_condition.notify_all()
            update_camera_health(new_vid, status="waiting", last_stream_frame_at=None)
        ensure_camera_started(new_vid)
        return jsonify({"status": "ok", "frame": source_frame})
    return jsonify({"status": "error"}), 400




@app.route('/track_person', methods=['POST'])
def track_person():
    global tracked_global_ids
    data = request.get_json()
    gid  = data.get("global_id")
    if gid is not None:
        gid = int(gid)
        if gid in tracked_global_ids:
            tracked_global_ids.remove(gid)
        else:
            tracked_global_ids.add(gid)
    return jsonify({"status": "ok", "tracking": list(tracked_global_ids)})


@app.route('/flag_person', methods=['POST'])
def flag_person():
    data = request.get_json()
    gid  = data.get("global_id")
    flag = bool(data.get("flag", True))
    if gid is None:
        return jsonify({"status": "error", "message": "global_id required"}), 400
    gid = int(gid)
    beh_lock = camera_states[current_video]["behavior_lock"]
    with beh_lock:
        if flag:
            global_manually_flagged_ids.add(gid)
            global_flagged_ids.add(gid)
        else:
            if gid in global_manually_flagged_ids:
                global_manually_flagged_ids.remove(gid)
            if gid in global_flagged_ids:
                global_flagged_ids.remove(gid)

        h = camera_states[current_video]["person_history"].setdefault(gid, _empty_history())
        h["manually_flagged"] = flag
        if flag:
            h["flagged"] = True
            
    # Sync with SQLite database
    with db_lock:
        try:
            db_person = db_session.query(TrackedPerson).filter(TrackedPerson.person_id == gid).first()
            if db_person:
                db_person.is_flagged_suspicious = flag
                db_person.flagged_reason = "Manually flagged by administrator." if flag else None
                db_session.commit()
        except Exception as db_ex:
            db_session.rollback()
            print(f"[DB EXCEPTION in manual flag_person] {db_ex}")
        finally:
            db_session.close()
            
    return jsonify({"status": "ok", "global_id": gid, "manually_flagged": flag})


@app.route('/api/camera/upload/<camera_key>', methods=['POST'])
def camera_upload(camera_key):
    if camera_key not in VIDEO_SOURCES:
        return jsonify({"status": "error", "message": "Unknown camera key"}), 400
    
    file = request.files.get('frame')
    if not file:
        return jsonify({"status": "error", "message": "No frame file uploaded"}), 400
        
    try:
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            h, w = img.shape[:2]
            if h > 480:
                scale = 480.0 / h
                img = cv2.resize(img, (int(w * scale), 480))
            
            q = upload_frame_queues.get(camera_key)
            if q is not None:
                enqueue_latest_camera_frame(camera_key, img)
                record_camera_frame(camera_key)
                try:
                    client_reconnects = int(request.headers.get("X-Camera-Reconnection-Attempts", "0"))
                    client_read_failures = int(request.headers.get("X-Camera-Last-Read-Failure-Burst", "0"))
                except ValueError:
                    client_reconnects = client_read_failures = 0
                update_camera_health(
                    camera_key, status="connected",
                    client_reconnection_attempts=max(0, client_reconnects),
                    client_last_read_failure_burst=max(0, client_read_failures),
                )
                # Ensure the background processing thread is running for this camera
                ensure_camera_started(camera_key)
                return jsonify({"status": "ok"})
        return jsonify({"status": "error", "message": "Failed to decode image"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/reset_database', methods=['POST'])
def reset_database():
    """Clear all dynamic database tracking data (persons, embeddings, sightings, dwell times, alerts) to start fresh."""
    global next_global_id, global_gallery, global_face_gallery, local_to_global, global_last_seen, search_history_log, face_aligned_tracks, track_last_face_check
    session = db_session()
    try:
        session.query(Alert).delete()
        session.query(DwellTime).delete()
        session.query(Sighting).delete()
        session.query(PersonEmbedding).delete()
        session.query(TrackedPerson).delete()
        session.commit()
        
        # 1. Reset trackers to force them to reload clean motion states
        with tracker_init_lock:
            for k in trackers:
                trackers[k] = None
                
        # 2. Reset local camera histories and statistics
        for k in VIDEO_SOURCES:
            reset_camera_stats(k, keep_flagged=False)
            
        # 3. Reset global Re-ID gallery, search logs, and locked target lists
        with gallery_lock:
            global_gallery.clear()
            global_face_gallery.clear()
            local_to_global.clear()
            global_last_seen.clear()
            face_aligned_tracks.clear()
            track_last_face_check.clear()
            next_global_id = 1
            
        tracked_global_ids.clear()
        global_flagged_ids.clear()
        global_manually_flagged_ids.clear()
        search_history_log.clear()
        
        # 4. Clean up physical snapshot and query files on disk
        snapshots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "snapshots")
        if os.path.exists(snapshots_dir):
            import shutil
            for name in os.listdir(snapshots_dir):
                path = os.path.join(snapshots_dir, name)
                try:
                    if os.path.isfile(path) or os.path.islink(path):
                        os.unlink(path)
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
                except Exception as e_del:
                    print(f"Failed to delete {path}: {e_del}")
            
        print("Database, local trackers, and memory galleries reset successfully.")
        return jsonify({"status": "ok", "message": "Database and trackers reset successfully."})
    except Exception as e:
        session.rollback()
        print(f"Error resetting database: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        session.close()


# ── PERSON SEARCH / QUERY BY IMAGE ENDPOINTS ───────────────────────────────────

search_history_log = []

@app.route('/upload_query_image', methods=['POST'])
def upload_query_image():
    """
    Receives an uploaded image, crops the primary person, extracts ReID embedding, 
    and returns matched database profiles sorted by similarity.
    """
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400
        
    try:
        from reid_search import crop_query_person, extract_reid_embedding, search_person_by_embedding
        
        # Save temp file
        app_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(app_dir, "static", "snapshots")
        os.makedirs(temp_dir, exist_ok=True)
        
        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"query_{timestamp_str}_{file.filename}"
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)
        
        # Crop person
        crop, bbox = crop_query_person(temp_path)
        if crop is None:
            return jsonify({
                "status": "error", 
                "message": "No person detected in the uploaded image. Please upload a clear image of a person."
            }), 400
            
        # Save the cropped query image so the admin can see what was searched
        cropped_filename = f"crop_{timestamp_str}_{file.filename}"
        cropped_path = os.path.join(temp_dir, cropped_filename)
        cv2.imwrite(cropped_path, crop)
        
        # Extract embedding
        emb = extract_reid_embedding(crop)
        
        # Perform similarity search
        threshold = float(request.form.get("threshold", 0.78))
        matches = search_person_by_embedding(emb, query_crop=crop, similarity_threshold=threshold, top_k=5)
        
        # Log to in-memory search history
        search_history_log.append({
            "timestamp": to_ist_str(datetime.utcnow()),
            "query_image": f"static/snapshots/{filename}",
            "cropped_image": f"static/snapshots/{cropped_filename}",
            "matches_found": len(matches),
            "top_match_similarity": matches[0]["similarity"] if matches else None,
            "top_match_id": matches[0]["person_id"] if matches else None
        })
        
        return jsonify({
            "status": "success",
            "query_image": f"static/snapshots/{filename}",
            "cropped_image": f"static/snapshots/{cropped_filename}",
            "matches": matches
        })
    except Exception as ex:
        print(f"[SEARCH ENDPOINT EXCEPTION] {ex}")
        return jsonify({"status": "error", "message": f"Server error: {str(ex)}"}), 500


@app.route('/person/<int:person_id>', methods=['GET'])
def get_person_details(person_id):
    """Retrieves full profile details for a registered person."""
    try:
        person = db_session.query(TrackedPerson).filter(TrackedPerson.person_id == person_id).first()
        if not person:
            return jsonify({"status": "error", "message": "Person not found"}), 404
            
        # Unique cameras visited
        cameras = db_session.query(Camera.label).join(Sighting).filter(Sighting.person_id == person_id).distinct().all()
        cameras_list = [c[0] for c in cameras]
        
        return jsonify({
            "status": "success",
            "person": {
                "person_id": person.person_id,
                "best_image_path": person.best_image_path,
                "first_seen": to_ist_str(person.first_seen),
                "last_seen": to_ist_str(person.last_seen),
                "total_dwell_seconds": person.total_dwell or 0,
                "is_flagged_suspicious": person.is_flagged_suspicious,
                "flagged_reason": person.flagged_reason,
                "visit_count": person.visit_count or 1,
                "camera_history": cameras_list
            }
        })
    except Exception as ex:
        return jsonify({"status": "error", "message": str(ex)}), 500


@app.route('/person/<int:person_id>/timeline', methods=['GET'])
def get_person_timeline(person_id):
    """Retrieves chronological sightings timeline for a person."""
    try:
        sightings = db_session.query(Sighting, Camera.label).join(Camera).filter(
            Sighting.person_id == person_id
        ).order_by(Sighting.timestamp.asc()).all()
        
        timeline = []
        last_zone_key = None
        for s, cam_label in sightings:
            # Group sightings by camera + zone transition points
            zone_key = (cam_label, s.zone)
            if zone_key != last_zone_key:
                timeline.append({
                    "sighting_id": s.sighting_id,
                    "timestamp": to_ist_str(s.timestamp),
                    "camera_label": cam_label,
                    "zone": s.zone or "Main Floor",
                    "confidence": round(s.confidence, 2) if s.confidence else 1.0,
                    "snapshot_path": s.snapshot_path
                })
                last_zone_key = zone_key
                
        # Limit to 25 most recent transition nodes to avoid flooding frontend memory
        if len(timeline) > 25:
            timeline = timeline[-25:]
            
        return jsonify({
            "status": "success",
            "timeline": timeline
        })
    except Exception as ex:
        return jsonify({"status": "error", "message": str(ex)}), 500


@app.route('/persons_list', methods=['GET'])
def get_all_persons_list():
    """Retrieves list of all tracked persons in the database."""
    try:
        persons = db_session.query(TrackedPerson).order_by(TrackedPerson.last_seen.desc()).all()
        out = []
        for p in persons:
            sightings_count = db_session.query(Sighting).filter(Sighting.person_id == p.person_id).count()
            out.append({
                "person_id": p.person_id,
                "best_image_path": p.best_image_path,
                "first_seen": to_ist_str(p.first_seen),
                "last_seen": to_ist_str(p.last_seen),
                "visit_count": p.visit_count or 1,
                "total_dwell_seconds": p.total_dwell or 0,
                "is_flagged_suspicious": p.is_flagged_suspicious,
                "flagged_reason": p.flagged_reason,
                "total_sightings": sightings_count
            })
        return jsonify({"status": "success", "persons": out})
    except Exception as ex:
        return jsonify({"status": "error", "message": str(ex)}), 500


@app.route('/search/history', methods=['GET'])
def get_search_history():
    """Retrieves in-memory query history logs."""
    return jsonify({
        "status": "success",
        "history": list(reversed(search_history_log))
    })


@app.route('/get_persons')
def get_persons():
    state    = camera_states[current_video]
    beh_lock = state["behavior_lock"]
    with beh_lock:
        persons = [
            {
                "global_id":        gid,
                "score":            round(ph.get("hybrid_score", ph.get("score", 0)), 1),
                "rule_score":       round(ph.get("score", 0), 1),
                "ml_score":         round(ph.get("ml_score", 0), 1),
                "flagged":          gid in global_flagged_ids or gid in global_manually_flagged_ids,
                "manually_flagged": gid in global_manually_flagged_ids,
                "score_breakdown":  ph.get("score_breakdown", {}),
                "zone":             ph.get("last_zone"),
                "zones_visited":    list(ph.get("zones_visited", set())),
                "zone_entry_count": ph.get("zone_entry_count", {}),
            }
            for gid, ph in state["person_history"].items()
        ]
    persons.sort(key=lambda p: p["score"], reverse=True)
    return jsonify({"persons": persons})


@app.route('/db_status')
def db_status_route():
    from models import Camera, Zone, TrackedPerson, PersonEmbedding, Sighting
    try:
        num_cameras = db_session.query(Camera).count()
        num_zones = db_session.query(Zone).count()
        num_persons = db_session.query(TrackedPerson).count()
        num_embeddings = db_session.query(PersonEmbedding).count()
        num_sightings = db_session.query(Sighting).count()
        
        sample_embeddings = []
        embs = db_session.query(PersonEmbedding).limit(5).all()
        for e in embs:
            vec = np.frombuffer(e.embedding_data, dtype=np.float32)
            sample_embeddings.append({
                "id": e.id,
                "person_id": e.person_id,
                "bytes_len": len(e.embedding_data),
                "shape": list(vec.shape) if hasattr(vec, "shape") else None
            })
            
        return jsonify({
            "status": "success",
            "cameras": num_cameras,
            "zones": num_zones,
            "persons": num_persons,
            "embeddings": num_embeddings,
            "sightings": num_sightings,
            "sample_embeddings": sample_embeddings
        })
    except Exception as ex:
        return jsonify({"status": "error", "error": str(ex)})


@app.route('/ae_status')
def ae_status_route():
    with ae_lock:
        trained     = ae_trained
        training    = ae_training_now
        thr         = ae_threshold
    with training_lock:
        n_seqs = len(training_seqs)
    return jsonify({
        "trained":       trained,
        "training_now":  training,
        "training_mode": TRAINING_MODE,
        "seq_count":     n_seqs,
        "min_seqs":      MIN_TRAIN_SEQS,
        "threshold":     round(thr, 5),
        "ml_weight":     ML_WEIGHT,
        "weights_file":  os.path.exists(WEIGHTS_FILE),
    })


@app.route('/toggle_training', methods=['POST'])
def toggle_training_route():
    """Toggle sequence collection mode on/off."""
    global TRAINING_MODE
    TRAINING_MODE = not TRAINING_MODE
    return jsonify({"training_mode": TRAINING_MODE})


@app.route('/clear_training_data', methods=['POST'])
def clear_training_data():
    """Clear collected sequences from memory (does NOT delete .npy file)."""
    with training_lock:
        training_seqs.clear()
    return jsonify({"status": "ok", "seq_count": 0})


@app.route('/train_model', methods=['POST'])
def train_model_route():
    """
    Save collected sequences to training_data.npy and launch train_ae.py
    as a background subprocess. Returns immediately.
    """
    global ae_training_now

    with training_lock:
        n_seqs = len(training_seqs)
        if n_seqs < MIN_TRAIN_SEQS:
            return jsonify({
                "status":  "error",
                "message": f"Need at least {MIN_TRAIN_SEQS} sequences. Have {n_seqs}."
            }), 400
        data = np.array(training_seqs, dtype=np.float32)

    if os.path.exists("training_data.npy"):
        existing = np.load("training_data.npy")
        data     = np.concatenate([existing, data], axis=0)
    np.save("training_data.npy", data)
    print(f"  Saved {len(data)} sequences to training_data.npy")

    with ae_lock:
        ae_training_now = True

    def _run_training():
        global ae_training_now, ae_model_obj, ae_threshold, ae_trained
        try:
            subprocess.run(
                [sys.executable, "train_ae.py"],
                check=True
            )
            _try_load_ae_model()
            socketio.emit("ae_training_done", {
                "success":   True,
                "threshold": ae_threshold,
            })
        except subprocess.CalledProcessError as e:
            print(f"  Training failed: {e}")
            socketio.emit("ae_training_done", {"success": False, "error": str(e)})
        finally:
            with ae_lock:
                globals()['ae_training_now'] = False

    threading.Thread(target=_run_training, daemon=True).start()
    return jsonify({"status": "ok", "message": "Training started in background."})


def _load_gallery_from_db():
    """
    On startup, load all registered person profiles and their OSNet embeddings 
    from the SQLite database into memory to maintain persistence across runs.
    """
    global next_global_id, global_gallery, global_last_seen
    session = db_session()
    try:
        persons = session.query(TrackedPerson).all()
        if not persons:
            print("  Database: No registered profiles found to load.")
            return
            
        max_id = 0
        loaded_count = 0
        now_monotonic = time.monotonic()
        
        with gallery_lock:
            for p in persons:
                gid = p.person_id
                if gid > max_id:
                    max_id = gid
                
                if p.is_flagged_suspicious:
                    if p.flagged_reason and "Manually" in p.flagged_reason:
                        global_manually_flagged_ids.add(gid)
                    else:
                        global_flagged_ids.add(gid)
                
                embs = session.query(PersonEmbedding).filter(
                    PersonEmbedding.person_id == gid,
                    PersonEmbedding.camera_key != "face"
                ).all()
                
                if embs:
                    cam_templates = {}
                    for e in embs:
                        cam_key = e.camera_key
                        vec = np.frombuffer(e.embedding_data, dtype=np.float32)
                        vec = vec / (np.linalg.norm(vec) + 1e-8)
                        
                        templates = cam_templates.setdefault(cam_key, deque(maxlen=GALLERY_TEMPLATES_PER_ID))
                        templates.append(vec)
                        
                    global_gallery[gid] = cam_templates
                    global_last_seen[gid] = now_monotonic
                    loaded_count += 1

                # Load face embeddings as well
                face_embs = session.query(PersonEmbedding).filter(
                    PersonEmbedding.person_id == gid,
                    PersonEmbedding.camera_key == "face"
                ).all()
                for fe in face_embs:
                    f_vec = np.frombuffer(fe.embedding_data, dtype=np.float32)
                    global_face_gallery.setdefault(gid, []).append(f_vec)
            
            next_global_id = max_id + 1
            
        print(f"  Database: Loaded {loaded_count} profiles. Next global ID: {next_global_id}")
    except Exception as ex:
        print(f"  Database Gallery Load Error: {ex}")
    finally:
        session.close()


# ── STARTUP ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  AcuTrack — Hybrid ML + Rule Suspicious Tracking")
    print("  StrongSort + OSNet Re-ID + LSTM Autoencoder")
    print("=" * 60)

    # Initialize the database and seed defaults
    print("  Initializing SQLite database...")
    init_db(VIDEO_SOURCES, ZONES)

    # Try to load pre-trained AE model
    _try_load_ae_model()

    # Load database gallery to prevent ID collisions
    _load_gallery_from_db()

    # FIX: Only the initially active camera's thread starts here. The rest
    # start on-demand the first time the user switches to them (see
    # ensure_camera_started / switch_video_route), so idle cameras don't
    # burn CPU on YOLO + StrongSort + OSNet before anyone's looking at them.
    print(f"  Starting active camera thread only: {current_video}")
    ensure_camera_started(current_video)

    print(f"  Active camera view : {current_video}")
    print("  Open browser → http://localhost:5000")
    print("=" * 60)

    socketio.run(app, host='0.0.0.0', port=5000, debug=False,
                 allow_unsafe_werkzeug=True)
