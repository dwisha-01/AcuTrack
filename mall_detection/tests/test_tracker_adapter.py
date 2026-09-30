import os
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tracker_adapter import DeepSortAdapter, create_tracker, selected_tracker_algorithm


class FakeDeepTrack:
    def __init__(self, track_id, details, time_since_update=0):
        self.track_id = track_id
        self.time_since_update = time_since_update
        self.details = details

    def get_det_supplementary(self):
        return self.details

    def is_confirmed(self):
        return True

    def to_ltrb(self):
        return np.array([10, 12, 30, 52], dtype=np.float32)


class FakeDeepSort:
    def __init__(self):
        self.tracker = type("Core", (), {"tracks": [], "_next_id": 1})()
        self.received = None

    def update_tracks(self, raw_detections, embeds=None, frame=None, today=None, others=None):
        self.received = (raw_detections, embeds, others)
        if raw_detections:
            self.tracker.tracks = [FakeDeepTrack("17", others[0])]
        else:
            self.tracker.tracks = []

    def delete_all_tracks(self):
        self.tracker.tracks = []
        self.tracker._next_id = 1


class TrackerAdapterTests(unittest.TestCase):
    def test_selection_defaults_to_strongsort_and_accepts_deepsort(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(selected_tracker_algorithm(), "strongsort")
        with patch.dict(os.environ, {"TRACKER_ALGORITHM": "DeepSort"}, clear=True):
            self.assertEqual(selected_tracker_algorithm(), "deepsort")

    def test_selection_rejects_unknown_algorithm(self):
        with patch.dict(os.environ, {"TRACKER_ALGORITHM": "sort"}, clear=True):
            with self.assertRaises(ValueError):
                selected_tracker_algorithm()

    def test_deepsort_adapter_converts_yolo_rows_and_preserves_local_id_features(self):
        fake_tracker = FakeDeepSort()
        embedding = np.array([3.0, 4.0], dtype=np.float32)
        adapter = DeepSortAdapter(
            fake_tracker, embedding_extractor=lambda crop: embedding
        )
        detections = np.array([[10, 12, 30, 52, 0.87, 0]], dtype=np.float32)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        tracks = adapter.update(detections, frame)

        raw, embeds, others = fake_tracker.received
        self.assertEqual(raw[0][0], [10, 12, 20, 40])
        self.assertAlmostEqual(raw[0][1], 0.87)
        self.assertEqual(raw[0][2], "0")
        np.testing.assert_allclose(embeds[0], [0.6, 0.8])
        self.assertAlmostEqual(others[0]["confidence"], 0.87)
        np.testing.assert_allclose(tracks, [[10, 12, 30, 52, 17, 0, 0.87]])
        view = adapter.tracker.tracks[0]
        self.assertEqual(view.id, int(tracks[0, 4]))
        np.testing.assert_allclose(view.features[-1], [0.6, 0.8])
        self.assertEqual(adapter.last_diagnostics["new_local_tracks"], 1)

    def test_empty_detections_return_accutrack_empty_shape(self):
        fake_tracker = FakeDeepSort()
        adapter = DeepSortAdapter(fake_tracker, embedding_extractor=Mock())
        tracks = adapter.update(np.empty((0, 6)), np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertEqual(tracks.shape, (0, 7))
        self.assertEqual(fake_tracker.received, ([], [], []))

    def test_real_deepsort_initializes_and_returns_compatible_rows(self):
        adapter = DeepSortAdapter(embedding_extractor=lambda crop: np.array([1, 0], dtype=np.float32))
        detections = np.array([[4, 6, 24, 46, 0.9, 0]], dtype=np.float32)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        adapter.update(detections, frame)
        tracks = adapter.update(detections, frame)
        self.assertEqual(tracks.shape, (1, 7))
        self.assertEqual(int(tracks[0, 4]), 1)
        self.assertAlmostEqual(float(tracks[0, 6]), 0.9)

    def test_reset_keeps_camera_local_ids_monotonic(self):
        adapter = DeepSortAdapter(embedding_extractor=lambda crop: np.array([1, 0], dtype=np.float32))
        detections = np.array([[4, 6, 24, 46, 0.9, 0]], dtype=np.float32)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        adapter.update(detections, frame)
        first_tracks = adapter.update(detections, frame)
        adapter.reset()
        adapter.update(detections, frame)
        next_tracks = adapter.update(detections, frame)
        self.assertEqual(int(first_tracks[0, 4]), 1)
        self.assertEqual(int(next_tracks[0, 4]), 2)

    def test_factory_preserves_strongsort_and_constructs_deepsort(self):
        device = type("Device", (), {"type": "cpu"})()
        strongsort = object()
        strongsort_factory = Mock(return_value=strongsort)
        self.assertIs(
            create_tracker("strongsort", strongsort_factory=strongsort_factory,
                           device=device, weights_path="osnet.pt"),
            strongsort,
        )
        strongsort_factory.assert_called_once_with(
            reid_weights="osnet.pt", device=device, half=False,
            max_age=90, max_cos_dist=0.55
        )
        deepsort = create_tracker("deepsort", strongsort_factory=strongsort_factory,
                                  device=device, weights_path="unused.pt")
        self.assertIsInstance(deepsort, DeepSortAdapter)


if __name__ == "__main__":
    unittest.main()
