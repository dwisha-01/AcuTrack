import os
import cv2
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timedelta
import threading

def to_ist_str(dt):
    if not dt:
        return "N/A"
    ist_dt = dt + timedelta(hours=5, minutes=30)
    return ist_dt.strftime("%Y-%m-%d %H:%M:%S")


from ultralytics import YOLO
from boxmot.reid.core import ReID
from models import db_session, PersonEmbedding, TrackedPerson, Sighting, Camera

# Select CPU or CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lazy loaders for models to keep startup clean and save memory
_yolo_model = None
_reid_model = None

def get_query_yolo():
    """Lazily loads and returns the YOLOv8 person detector."""
    global _yolo_model
    if _yolo_model is None:
        app_dir = os.path.dirname(os.path.abspath(__file__))
        if device.type == "cuda":
            _yolo_model = YOLO(os.path.join(app_dir, "yolov8n.pt"))
            _yolo_model.to(device)
        elif os.path.exists(os.path.join(app_dir, "yolov8n.onnx")):
            _yolo_model = YOLO(os.path.join(app_dir, "yolov8n.onnx"))
        else:
            _yolo_model = YOLO(os.path.join(app_dir, "yolov8n.pt"))
    return _yolo_model

def get_query_reid():
    """Lazily loads and returns the OSNet Re-ID model backend."""
    global _reid_model
    if _reid_model is None:
        app_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(app_dir, "osnet_x0_25_msmt17.pt")
        _reid_model = ReID(
            weights=Path(weights_path),
            device=device,
            half=(device.type == "cuda")
        )
    return _reid_model

def _cos_dist(a, b):
    """Cosine distance between two vectors."""
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

import ssl
import urllib.request

_face_detector = None
_face_recognizer = None

def download_face_models():
    """Checks and downloads YuNet and SFace ONNX models if not present."""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    yunet_path = os.path.join(app_dir, "face_detection_yunet_2023mar.onnx")
    sface_path = os.path.join(app_dir, "face_recognition_sface_2021dec.onnx")
    
    if os.path.exists(yunet_path) and os.path.exists(sface_path):
        return yunet_path, sface_path

    # Use unverified context to avoid SSL errors with urllib on some systems
    try:
        context = ssl._create_unverified_context()
    except AttributeError:
        context = None
        
    urls = {
        yunet_path: "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        sface_path: "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    }
    
    for local_path, url in urls.items():
        if not os.path.exists(local_path):
            print(f"Face Verification: Downloading {url}...")
            try:
                if context:
                    with urllib.request.urlopen(url, context=context, timeout=30) as response, open(local_path, 'wb') as out_file:
                        out_file.write(response.read())
                else:
                    with urllib.request.urlopen(url, timeout=30) as response, open(local_path, 'wb') as out_file:
                        out_file.write(response.read())
                print(f"Face Verification: Downloaded {os.path.basename(local_path)} successfully.")
            except Exception as e:
                print(f"Face Verification Error: Failed to download {url}: {e}")
                
    return yunet_path, sface_path

def get_face_detector(w=320, h=320):
    """Lazily loads and returns the YuNet face detector model."""
    global _face_detector
    if _face_detector is None:
        yunet_path, _ = download_face_models()
        if not os.path.exists(yunet_path):
            print("Face Verification: YuNet ONNX file is missing.")
            return None
        try:
            # Use positional arguments for robust compatibility across OpenCV versions
            _face_detector = cv2.FaceDetectorYN.create(
                yunet_path,
                "",
                (w, h),
                0.4,
                0.3,
                10
            )
        except Exception as e:
            print(f"Face Verification: Failed to create FaceDetectorYN: {e}")
            _face_detector = None
    return _face_detector

def get_face_recognizer():
    """Lazily loads and returns the SFace face recognizer model."""
    global _face_recognizer
    if _face_recognizer is None:
        _, sface_path = download_face_models()
        if not os.path.exists(sface_path):
            print("Face Verification: SFace ONNX file is missing.")
            return None
        try:
            # Use positional arguments for robust compatibility across OpenCV versions
            _face_recognizer = cv2.FaceRecognizerSF.create(
                sface_path,
                ""
            )
        except Exception as e:
            print(f"Face Verification: Failed to create FaceRecognizerSF: {e}")
            _face_recognizer = None
    return _face_recognizer

_face_lock = threading.Lock()

def detect_face(img):
    """Detects the primary face in the image and returns face box + landmarks."""
    with _face_lock:
        if img is None or img.size == 0:
            return None
        h, w, _ = img.shape
        detector = get_face_detector(w, h)
        if detector is None:
            return None
        
        # Try setting input size dynamically
        try:
            detector.setInputSize((w, h))
        except Exception as e_size:
            # Recreate detector dynamically if setInputSize fails (due to some cv2 binding versions)
            try:
                yunet_path, _ = download_face_models()
                detector = cv2.FaceDetectorYN.create(yunet_path, "", (w, h), 0.4, 0.3, 10)
            except Exception as e_recreate:
                print(f"Face Verification: Failed to recreate detector: {e_recreate}")
                return None
                
        try:
            retval, faces = detector.detect(img)
            if retval and faces is not None and len(faces) > 0:
                # Pick the face with the highest confidence score (at index 14)
                best_face = max(faces, key=lambda f: f[14])
                
                # Check for non-finite values to avoid float-to-int conversion exceptions
                if not np.isfinite(best_face).all():
                    return None
                
                # Enforce size and confidence filters to prevent noisy/distant matches
                try:
                    f_w = int(best_face[2])
                    f_h = int(best_face[3])
                    f_conf = float(best_face[14])
                except (ValueError, OverflowError):
                    return None
                    
                if f_w >= 14 and f_h >= 14 and f_conf >= 0.75:
                    return best_face
                return None
        except Exception as e:
            print(f"Face Verification: Error during face detection: {e}")
        return None

def extract_face_embedding(img, face):
    """Aligns the face crop and extracts its 128-dimensional embedding."""
    with _face_lock:
        recognizer = get_face_recognizer()
        if recognizer is None or face is None:
            return None
        try:
            aligned_face = recognizer.alignCrop(img, face)
            feature = recognizer.feature(aligned_face)
            return feature
        except Exception as e:
            print(f"Face Verification: Error during face embedding extraction: {e}")
        return None

def compute_face_similarity(feat1, feat2):
    """Calculates cosine similarity between two face feature vectors."""
    if feat1 is None or feat2 is None:
        return 0.0
    try:
        f1 = feat1.flatten()
        f2 = feat2.flatten()
        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        return float(np.dot(f1, f2) / (norm1 * norm2))
    except Exception as e:
        print(f"Face Verification: Error calculating face similarity: {e}")
    return 0.0

def crop_query_person(image_path):
    """
    Loads an image from image_path, runs YOLOv8 person detection, 
    and crops the bounding box containing the largest person.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
        
    model = get_query_yolo()
    results = model(img, classes=[0], verbose=False, conf=0.35)
    
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        # Fallback: if no person is detected, return the entire image as the crop
        return img, (0, 0, img.shape[1], img.shape[0])
        
    best_box = None
    max_area = 0
    
    for box in results[0].boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, xyxy)
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            best_box = (x1, y1, x2, y2)
            
    if best_box is None:
        return img, (0, 0, img.shape[1], img.shape[0])
        
    x1, y1, x2, y2 = best_box
    h, w, _ = img.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    crop = img[y1:y2, x1:x2]
    return crop, best_box

def extract_reid_embedding(crop):
    """
    Passes a BGR crop through the OSNet backend and 
    returns a normalized 512-dimensional embedding vector.
    """
    model = get_query_reid()
    with torch.no_grad():
        features = model([crop])
        if torch.is_tensor(features):
            features = features.detach().cpu().numpy()
        elif hasattr(features, "numpy"):
            features = features.numpy()
        emb = features[0]
    # L2 Normalization
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    return emb

def normalize_face_similarity(face_sim):
    """Maps SFace similarity (-0.2 to 1.0) to a scale where match (>=0.36) maps to >= 0.78."""
    if face_sim >= 0.36:
        # Linear map: 0.36 -> 0.78, 1.0 -> 1.0
        return 0.78 + (face_sim - 0.36) * (1.0 - 0.78) / (1.0 - 0.36)
    elif face_sim >= 0.28:
        # Uncertain range: map 0.28 to 0.40 and 0.36 to 0.78
        return 0.40 + (face_sim - 0.28) * (0.78 - 0.40) / (0.36 - 0.28)
    else:
        # Definite mismatch! Strict zero match score.
        return 0.0

def search_person_by_embedding(query_emb, query_crop=None, similarity_threshold=0.78, top_k=5):
    """
    Performs database matching for a query target by combining face and clothing Re-ID.
    - If a face is visible in both query and database profile, uses face similarity.
    - Otherwise, falls back to clothing Re-ID matching.
    """
    app_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Try to extract face embedding from query crop
    query_face_emb = None
    if query_crop is not None:
        q_face = detect_face(query_crop)
        if q_face is not None:
            query_face_emb = extract_face_embedding(query_crop, q_face)

    # Fetch all registered persons and embeddings from DB
    all_persons = db_session.query(TrackedPerson).all()
    if not all_persons:
        return []
        
    db_embs = db_session.query(PersonEmbedding).all()
    
    # Map face and clothing embeddings by person ID
    person_face_map = {}
    person_clothing_map = {}
    for rec in db_embs:
        vec = np.frombuffer(rec.embedding_data, dtype=np.float32)
        if rec.camera_key == "face":
            if vec.shape[0] == 128:
                person_face_map[rec.person_id] = vec
        else:
            if vec.shape[0] == 512:
                person_clothing_map.setdefault(rec.person_id, []).append(vec)

    # Self-healing: extract and cache face embeddings for candidates missing them
    db_updated = False
    for person in all_persons:
        pid = person.person_id
        if pid not in person_face_map and person.best_image_path:
            cand_img_path = os.path.join(app_dir, person.best_image_path)
            if os.path.exists(cand_img_path):
                cand_img = cv2.imread(cand_img_path)
                if cand_img is not None:
                    c_face = detect_face(cand_img)
                    if c_face is not None:
                        cand_face_emb = extract_face_embedding(cand_img, c_face)
                        if cand_face_emb is not None:
                            person_face_map[pid] = cand_face_emb
                            new_rec = PersonEmbedding(
                                person_id=pid,
                                embedding_data=cand_face_emb.astype(np.float32).tobytes(),
                                camera_key="face",
                                confidence=1.0
                            )
                            db_session.add(new_rec)
                            db_updated = True
    if db_updated:
        try:
            db_session.commit()
        except Exception as e:
            db_session.rollback()
            print(f"Failed to commit self-healing face cache: {e}")

    # Calculate match scores
    results = []
    for person in all_persons:
        pid = person.person_id
        cand_face_emb = person_face_map.get(pid)
        cand_clothing = person_clothing_map.get(pid, [])
        
        final_sim_score = 0.0
        face_match_status = "no_face"
        face_sim = None
        
        # Use face similarity if face is visible in both query and database profile
        if query_face_emb is not None and cand_face_emb is not None:
            face_sim = compute_face_similarity(query_face_emb, cand_face_emb)
            final_sim_score = normalize_face_similarity(face_sim)
            if face_sim >= 0.36:
                face_match_status = "matched"
            elif face_sim < 0.28:
                face_match_status = "mismatched"
            else:
                face_match_status = "uncertain"
            print(f"  Candidate GID {pid} (Face): score={final_sim_score:.3f}, face_sim={face_sim:.3f}")
        else:
            # Fallback to clothing similarity matching
            if cand_clothing:
                final_sim_score = max(float(np.dot(query_emb, vec)) for vec in cand_clothing)
                face_match_status = "clothing_only" if query_face_emb is not None else "no_face_in_query"
                print(f"  Candidate GID {pid} (Clothing): score={final_sim_score:.3f}")
                
        # Filter matches based on similarity_threshold
        if final_sim_score >= similarity_threshold:
            sightings_count = db_session.query(Sighting).filter(Sighting.person_id == pid).count()
            cameras = db_session.query(Camera.label).join(Sighting).filter(Sighting.person_id == pid).distinct().all()
            cameras_list = [c[0] for c in cameras]
            
            results.append({
                "person_id": pid,
                "is_face_match": (face_match_status == "matched"),
                "similarity": round(final_sim_score, 3),
                "confidence_percentage": round(final_sim_score * 100, 1),
                "best_image_path": person.best_image_path,
                "first_seen": to_ist_str(person.first_seen),
                "last_seen": to_ist_str(person.last_seen),
                "total_dwell_seconds": person.total_dwell or 0,
                "total_sightings": sightings_count,
                "camera_history": cameras_list,
                "face_match_status": face_match_status,
                "face_similarity": round(face_sim, 3) if face_sim is not None else None
            })
            
    results = sorted(results, key=lambda x: (x["is_face_match"], x["similarity"]), reverse=True)[:top_k]
    return results
