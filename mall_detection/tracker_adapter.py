"""Small compatibility adapter for the tracker output used by AccuTrack."""

import os
from types import SimpleNamespace

import numpy as np


def selected_tracker_algorithm():
    algorithm = os.getenv("TRACKER_ALGORITHM", "strongsort").strip().lower()
    if algorithm not in {"strongsort", "ocsort"}:
        raise ValueError(
            f"Unsupported TRACKER_ALGORITHM {algorithm!r}; use 'strongsort' or 'ocsort'."
        )
    return algorithm


class _OCTrackView:
    """Expose the track fields AccuTrack reads from StrongSORT internals."""

    def __init__(self, track):
        self._track = track
        self.id = int(track.id) + 1  # BoxMOT OC-SORT output IDs are one-based.
        self.time_since_update = int(track.time_since_update)
        self.features = None


class _OCTrackerState:
    def __init__(self, owner):
        self._owner = owner

    @property
    def tracks(self):
        return [_OCTrackView(track) for track in self._owner._tracker.active_tracks]


class OCSortAdapter:
    """Normalize BoxMOT OC-SORT outputs to AccuTrack's Nx7 track rows."""

    algorithm = "ocsort"

    def __init__(self, tracker=None, *, max_age=90):
        if tracker is None:
            from boxmot.trackers.ocsort.ocsort import OcSort

            tracker = OcSort(det_thresh=0.35, max_age=max_age, min_hits=1)
        self._tracker = tracker
        self.tracker = _OCTrackerState(self)
        self._previous_active_ids = set()
        self._seen_local_ids = set()
        self.last_diagnostics = {
            "active_tracks": 0,
            "new_local_tracks": 0,
            "removed_tracks": 0,
        }

    def update(self, detections, frame):
        # BoxMOT expects [x1, y1, x2, y2, confidence, class], matching YOLO.
        dets = np.asarray(detections, dtype=np.float32).reshape(-1, 6)
        raw_tracks = self._tracker.update(dets, frame)
        if raw_tracks is None or np.asarray(raw_tracks).size == 0:
            normalized = np.empty((0, 7), dtype=np.float32)
        else:
            raw_tracks = np.asarray(raw_tracks, dtype=np.float32).reshape(-1, 8)
            # OC-SORT: xyxy, local id, confidence, class, detection index.
            normalized = np.column_stack(
                (raw_tracks[:, :5], np.zeros(len(raw_tracks)), raw_tracks[:, 5])
            ).astype(np.float32, copy=False)

        active_ids = {int(track.id) + 1 for track in self._tracker.active_tracks}
        output_ids = {int(row[4]) for row in normalized}
        new_ids = output_ids - self._seen_local_ids
        self._seen_local_ids.update(output_ids)
        self.last_diagnostics = {
            "active_tracks": len(output_ids),
            "new_local_tracks": len(new_ids),
            "removed_tracks": len(self._previous_active_ids - active_ids),
        }
        self._previous_active_ids = active_ids
        return normalized

    def get_embedding(self, frame, bbox):
        """Extract the existing OSNet embedding for an OC-SORT track crop."""
        from reid_search import extract_reid_embedding

        x1, y1, x2, y2 = map(int, bbox)
        height, width = frame.shape[:2]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return extract_reid_embedding(frame[y1:y2, x1:x2])

    def reset(self):
        self._tracker.active_tracks.clear()
        self._tracker.frame_count = 0
        self._previous_active_ids.clear()
        self._seen_local_ids.clear()
        self.last_diagnostics = {
            "active_tracks": 0,
            "new_local_tracks": 0,
            "removed_tracks": 0,
        }


def create_tracker(algorithm, *, strongsort_factory, device, weights_path):
    if algorithm == "ocsort":
        return OCSortAdapter(max_age=90)
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
