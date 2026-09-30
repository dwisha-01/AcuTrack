"""Compatibility adapter between DeepSORT and AccuTrack's tracker contract."""

import os

import numpy as np


def selected_tracker_algorithm():
    algorithm = os.getenv("TRACKER_ALGORITHM", "strongsort").strip().lower()
    if algorithm not in {"strongsort", "deepsort"}:
        raise ValueError(
            f"Unsupported TRACKER_ALGORITHM {algorithm!r}; use 'strongsort' or 'deepsort'."
        )
    return algorithm


class _DeepTrackView:
    """Expose the StrongSORT-style fields read by AccuTrack's shared loop."""

    def __init__(self, track, embedding):
        self.id = int(track.track_id)
        self.time_since_update = int(track.time_since_update)
        self.features = [embedding] if embedding is not None else []


class _DeepTrackerState:
    def __init__(self, owner):
        self._owner = owner

    @property
    def tracks(self):
        return [
            _DeepTrackView(track, self._owner._last_embeddings.get(int(track.track_id)))
            for track in self._owner._tracker.tracker.tracks
        ]


class DeepSortAdapter:
    """Accept YOLO Nx6 detections and emit AccuTrack's normalized Nx7 rows."""

    algorithm = "deepsort"

    def __init__(self, tracker=None, embedding_extractor=None, *, max_age=90):
        if tracker is None:
            from deep_sort_realtime.deepsort_tracker import DeepSort

            # AccuTrack OSNet vectors supply DeepSORT's local appearance metric;
            # its packaged embedder is disabled to avoid a second Re-ID model.
            tracker = DeepSort(
                max_age=max_age,
                n_init=1,
                max_cosine_distance=0.55,
                nn_budget=100,
                embedder=None,
            )
        self._tracker = tracker
        self._embedding_extractor = embedding_extractor
        self.max_age = max_age
        self.tracker = _DeepTrackerState(self)
        self._last_embeddings = {}
        self._last_confidences = {}
        self._seen_local_ids = set()
        self._previous_track_ids = set()
        self.last_diagnostics = {
            "active_tracks": 0,
            "new_local_tracks": 0,
            "removed_tracks": 0,
        }

    def _extract_embedding(self, crop):
        extractor = self._embedding_extractor
        if extractor is None:
            from reid_search import extract_reid_embedding

            extractor = extract_reid_embedding
        embedding = np.asarray(extractor(crop), dtype=np.float32).reshape(-1)
        return embedding / (np.linalg.norm(embedding) + 1e-8)

    def update(self, detections, frame):
        dets = np.asarray(detections, dtype=np.float32).reshape(-1, 6)
        height, width = frame.shape[:2]
        raw_detections = []
        embeddings = []
        supplementary = []

        for x1, y1, x2, y2, confidence, class_id in dets:
            left = max(0, min(width, int(x1)))
            top = max(0, min(height, int(y1)))
            right = max(0, min(width, int(x2)))
            bottom = max(0, min(height, int(y2)))
            if right <= left or bottom <= top:
                continue
            embedding = self._extract_embedding(frame[top:bottom, left:right])
            conf = float(confidence)
            raw_detections.append(
                ([left, top, right - left, bottom - top], conf, str(int(class_id)))
            )
            embeddings.append(embedding)
            supplementary.append({"embedding": embedding, "confidence": conf})

        self._tracker.update_tracks(
            raw_detections, embeds=embeddings, others=supplementary
        )

        live_tracks = list(self._tracker.tracker.tracks)
        current_ids = {int(track.track_id) for track in live_tracks}
        rows = []
        for track in live_tracks:
            local_id = int(track.track_id)
            details = track.get_det_supplementary()
            if track.time_since_update == 0 and isinstance(details, dict):
                self._last_embeddings[local_id] = details["embedding"]
                self._last_confidences[local_id] = details["confidence"]

            if not track.is_confirmed() or track.time_since_update > self.max_age:
                continue
            bbox = track.to_ltrb()
            confidence = self._last_confidences.get(local_id, 1.0)
            rows.append([*bbox, local_id, 0.0, confidence])

        active_ids = {int(track.track_id) for track in live_tracks if track.time_since_update == 0}
        output_ids = {int(row[4]) for row in rows}
        new_ids = output_ids - self._seen_local_ids
        self._seen_local_ids.update(output_ids)
        self.last_diagnostics = {
            "active_tracks": len(active_ids),
            "new_local_tracks": len(new_ids),
            "removed_tracks": len(self._previous_track_ids - current_ids),
        }
        self._previous_track_ids = current_ids

        removed_ids = self._seen_local_ids - current_ids
        for local_id in removed_ids:
            self._last_embeddings.pop(local_id, None)
            self._last_confidences.pop(local_id, None)
        return np.asarray(rows, dtype=np.float32).reshape(-1, 7)

    def reset(self):
        next_id = self._tracker.tracker._next_id
        self._tracker.delete_all_tracks()
        # Keep camera-local IDs monotonic so old Global ID bindings cannot alias.
        self._tracker.tracker._next_id = next_id
        self._tracker.tracker.metric.samples = {}
        self._last_embeddings.clear()
        self._last_confidences.clear()
        self._seen_local_ids.clear()
        self._previous_track_ids.clear()
        self.last_diagnostics = {
            "active_tracks": 0,
            "new_local_tracks": 0,
            "removed_tracks": 0,
        }


def create_tracker(algorithm, *, strongsort_factory, device, weights_path):
    if algorithm == "deepsort":
        return DeepSortAdapter(max_age=90)
    if algorithm != "strongsort":
        raise ValueError(f"Unsupported tracker algorithm: {algorithm}")
    tracker = strongsort_factory(
        reid_weights=weights_path,
        device=device,
        half=(device.type == "cuda"),
        max_age=90,
        max_cos_dist=0.55,
    )
    return tracker
