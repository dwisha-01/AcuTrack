"""Focused checks for Global ID decision diagnostic records."""

import os
import sys
import unittest
from collections import deque
from unittest.mock import patch

import numpy as np

mall_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tests_dir = os.path.dirname(os.path.abspath(__file__))
for path in (mall_dir, tests_dir):
    if path not in sys.path:
        sys.path.insert(0, path)

from db_test_support import initialize_test_database
import app as acu


class TestGlobalIdDiagnostics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        initialize_test_database(acu)

    @classmethod
    def tearDownClass(cls):
        acu.db_queue.join()

    def setUp(self):
        with acu.gallery_lock:
            acu.global_gallery.clear()
            acu.global_last_seen.clear()
            acu.local_to_global.clear()
            acu._global_id_diagnostic_seen.clear()
            acu.next_global_id = 100
        self.events = []
        self.emit_patch = patch.object(acu, "emit_global_id_diagnostic", side_effect=self.events.append)
        self.emit_patch.start()
        self.queue_patch = patch.object(acu.db_queue, "put")
        self.queue_patch.start()
        self.embedding = np.zeros(512, dtype=np.float32)
        self.embedding[0] = 1.0

    def tearDown(self):
        self.queue_patch.stop()
        self.emit_patch.stop()

    def decide(self, camera, local_id, embedding=None, area=6000, confidence=0.9, **kwargs):
        if embedding is None:
            embedding = self.embedding
        kwargs.setdefault("run_face_detection", False)
        return acu.match_or_register(
            camera, local_id, embedding, box_area=area, confidence=confidence,
            **kwargs
        )

    def add_template(self, gid, camera, embedding=None):
        with acu.gallery_lock:
            acu.global_gallery[gid] = {camera: deque([embedding if embedding is not None else self.embedding], maxlen=15)}
            acu.global_last_seen[gid] = __import__("time").monotonic()

    def test_successful_gallery_match_and_cross_camera_event(self):
        self.add_template(17, "cam1")
        gid = self.decide("cam2", 14)
        self.assertEqual(gid, 17)
        event = self.events[-1]
        self.assertEqual(event["decision"], "match")
        self.assertEqual(event["matching_result"], "match")
        self.assertEqual(event["decision_source"], "osnet")
        self.assertEqual(event["camera_key"], "cam2")
        self.assertEqual(event["local_track_id"], 14)
        self.assertEqual(event["best_candidate_gid"], 17)
        self.assertEqual(event["assigned_gid"], 17)
        self.assertEqual(event["candidate_gid_count"], 1)
        self.assertEqual(event["best_candidate_template_count"], 1)
        self.assertTrue(event["embedding_valid"])
        self.assertEqual(event["embedding_dimension"], 512)

    def test_distance_rejection_records_candidate_and_new_gid(self):
        orthogonal = np.zeros(512, dtype=np.float32)
        orthogonal[1] = 1.0
        self.add_template(17, "cam1", orthogonal)
        gid = self.decide("cam2", 14)
        event = self.events[-1]
        self.assertEqual(gid, 100)
        self.assertEqual(event["decision"], "new_gid")
        self.assertEqual(event["reason"], "rejected_distance_new_gid")
        self.assertEqual(event["matching_result"], "rejected_distance")
        self.assertEqual(event["best_candidate_gid"], 17)
        self.assertAlmostEqual(event["best_cosine_distance"], 1.0, places=5)
        self.assertEqual(event["assigned_gid"], gid)

    def test_ambiguity_rejection_records_both_candidates(self):
        first = np.array([0.8, np.sqrt(1 - 0.8**2)], dtype=np.float32)
        second = np.array([0.78, np.sqrt(1 - 0.78**2)], dtype=np.float32)
        q = np.array([1.0, 0.0], dtype=np.float32)
        self.add_template(17, "cam1", first)
        self.add_template(23, "cam3", second)
        gid = self.decide("cam2", 14, embedding=q)
        event = self.events[-1]
        self.assertEqual(gid, 100)
        self.assertEqual(event["decision"], "new_gid")
        self.assertEqual(event["reason"], "rejected_ambiguity_new_gid")
        self.assertEqual(event["matching_result"], "rejected_ambiguity")
        self.assertEqual(event["best_candidate_gid"], 17)
        self.assertEqual(event["second_best_candidate_gid"], 23)
        self.assertLess(event["actual_margin"], event["required_ambiguity_margin"])

    def test_new_gid_with_empty_gallery_records_template_rejection(self):
        gid = self.decide("cam1", 4, confidence=0.4)
        event = self.events[-1]
        self.assertEqual(gid, 100)
        self.assertEqual(event["decision"], "new_gid")
        self.assertFalse(event["new_gid_template_added"])
        self.assertTrue(event["assigned_gid_gallery_empty"])
        self.assertEqual(event["new_gid_template_rejection_reason"], "confidence_below_gallery_minimum")
        self.assertEqual(acu.global_gallery[gid], {})

    def test_existing_empty_gallery_is_counted_as_unusable_candidate(self):
        with acu.gallery_lock:
            acu.global_gallery[17] = {}
            acu.global_last_seen[17] = __import__("time").monotonic()
        gid = self.decide("cam2", 14)
        event = self.events[-1]
        self.assertEqual(gid, 100)
        self.assertEqual(event["empty_gallery_gid_count"], 1)
        self.assertEqual(event["candidate_gid_count"], 0)
        self.assertEqual(event["empty_gallery_gids"], [17])
        self.assertEqual(event["reason"], "empty_gallery_templates_unavailable")

    def test_face_match_source_is_recorded(self):
        face_embedding = np.ones(128, dtype=np.float32)
        with acu.gallery_lock:
            acu.global_face_gallery[17] = [face_embedding]
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch.object(acu, "detect_face", return_value=object()), \
             patch.object(acu, "extract_face_embedding", return_value=face_embedding), \
             patch.object(acu, "compute_face_similarity", return_value=0.9):
            gid = self.decide(
                "cam1", 4, area=2500, frame=frame, bbox=(0, 0, 50, 50),
                run_face_detection=True,
            )
        event = self.events[-1]
        self.assertEqual(gid, 17)
        self.assertEqual(event["decision"], "face_match")
        self.assertEqual(event["decision_source"], "face")

    def test_small_box_fallback_is_diagnosed(self):
        gid = self.decide("cam1", 4, area=50)
        event = self.events[-1]
        self.assertLess(gid, 0)
        self.assertEqual(event["decision"], "small_box")
        self.assertEqual(event["assigned_gid"], gid)
        self.assertFalse(event["osnet_matching_performed"])

    def test_new_local_track_id_reassignment_is_visible_and_deduplicated(self):
        first_gid = self.decide("cam1", 4)
        second_gid = self.decide("cam1", 12)
        # Repeated detection frames for the same local track do not repeat the event.
        self.decide("cam1", 12)
        self.assertEqual(first_gid, 100)
        self.assertEqual(second_gid, first_gid)
        self.assertEqual([(e["local_track_id"], e["assigned_gid"]) for e in self.events], [(4, 100), (12, 100)])
        self.assertEqual(self.events[1]["local_binding_status"], "unbound")


if __name__ == "__main__":
    unittest.main(verbosity=2)
