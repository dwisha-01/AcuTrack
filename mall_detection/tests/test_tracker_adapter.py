import os
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tracker_adapter import OCSortAdapter, create_tracker, selected_tracker_algorithm


class FakeTrack:
    def __init__(self, track_id, time_since_update=0):
        self.id = track_id
        self.time_since_update = time_since_update


class FakeOCSort:
    def __init__(self, rows):
        self.rows = rows
        self.active_tracks = []
        self.frame_count = 0
        self.received = None

    def update(self, detections, frame):
        self.received = detections.copy()
        self.active_tracks = [FakeTrack(0)] if len(self.rows) else []
        return np.asarray(self.rows, dtype=np.float32)


class TrackerAdapterTests(unittest.TestCase):
    def test_tracker_selection_defaults_to_strongsort_and_accepts_ocsort(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(selected_tracker_algorithm(), "strongsort")
        with patch.dict(os.environ, {"TRACKER_ALGORITHM": "OCSORT"}, clear=True):
            self.assertEqual(selected_tracker_algorithm(), "ocsort")

    def test_tracker_selection_rejects_unknown_algorithm(self):
        with patch.dict(os.environ, {"TRACKER_ALGORITHM": "sort"}, clear=True):
            with self.assertRaises(ValueError):
                selected_tracker_algorithm()

    def test_ocsort_receives_yolo_rows_and_returns_accutrack_shape(self):
        detections = np.array([[1, 2, 11, 22, 0.9, 0]], dtype=np.float32)
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        raw = [[1, 2, 11, 22, 1, 0.9, 0, 0]]
        underlying = FakeOCSort(raw)
        adapter = OCSortAdapter(underlying)

        tracks = adapter.update(detections, frame)

        np.testing.assert_array_equal(underlying.received, detections)
        np.testing.assert_allclose(tracks, [[1, 2, 11, 22, 1, 0, 0.9]])
        self.assertEqual(adapter.tracker.tracks[0].id, int(tracks[0, 4]))
        self.assertEqual(adapter.last_diagnostics["new_local_tracks"], 1)

    def test_empty_ocsort_output_has_stable_shape_and_reset_clears_state(self):
        underlying = FakeOCSort([])
        adapter = OCSortAdapter(underlying)
        tracks = adapter.update(np.empty((0, 6)), np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertEqual(tracks.shape, (0, 7))
        underlying.active_tracks = [FakeTrack(3)]
        adapter._seen_local_ids.add(4)

        adapter.reset()

        self.assertEqual(underlying.active_tracks, [])
        self.assertFalse(adapter._seen_local_ids)

    def test_ocsort_factory_does_not_construct_strongsort(self):
        strongsort_factory = Mock(side_effect=AssertionError("unexpected"))
        tracker = create_tracker(
            "ocsort", strongsort_factory=strongsort_factory,
            device=type("Device", (), {"type": "cpu"})(), weights_path="unused.pt"
        )
        self.assertIsInstance(tracker, OCSortAdapter)
        strongsort_factory.assert_not_called()

    def test_strongsort_factory_preserves_existing_configuration(self):
        strongsort = object()
        strongsort_factory = Mock(return_value=strongsort)
        device = type("Device", (), {"type": "cpu"})()

        result = create_tracker(
            "strongsort", strongsort_factory=strongsort_factory,
            device=device, weights_path="osnet.pt"
        )

        self.assertIs(result, strongsort)
        strongsort_factory.assert_called_once_with(
            reid_weights="osnet.pt", device=device, half=False,
            max_age=90, max_cos_dist=0.55
        )

    def test_environment_selection_constructs_ocsort(self):
        with patch.dict(os.environ, {"TRACKER_ALGORITHM": "ocsort"}, clear=True):
            algorithm = selected_tracker_algorithm()
        tracker = create_tracker(
            algorithm, strongsort_factory=Mock(),
            device=type("Device", (), {"type": "cpu"})(), weights_path="unused.pt"
        )
        self.assertIsInstance(tracker, OCSortAdapter)


if __name__ == "__main__":
    unittest.main()
