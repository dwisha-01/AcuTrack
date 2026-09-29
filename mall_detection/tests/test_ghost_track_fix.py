import sys
import os
import time
import unittest
from collections import deque
import numpy as np

# Ensure mall_detection directory is on python path
mall_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if mall_dir not in sys.path:
    sys.path.insert(0, mall_dir)
tests_dir = os.path.dirname(os.path.abspath(__file__))
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)

from db_test_support import initialize_test_database
from app import (
    reserve_existing_global_ids,
    match_or_register,
    classify_track_status,
    local_to_global,
    global_gallery,
    global_last_seen,
    gallery_lock,
    REID_MATCH_THRESHOLD,
    REID_AMBIGUITY_MARGIN
)
import app as acu


class TestGhostTrackFix(unittest.TestCase):
    """
    Deterministic regression tests for the ghost-track lockout fix in AccuTrack.
    Verifies that:
    1. A new local track can recover an existing GID when the old track is coasting and Re-ID strongly matches.
    2. Two actively observed different people in the same frame maintain duplicate protection.
    3. Weak/ambiguous matches do not blindly steal an existing GID.
    """

    @classmethod
    def setUpClass(cls):
        initialize_test_database(acu)

    @classmethod
    def tearDownClass(cls):
        # Ensure the asynchronous persistence tasks have completed before the
        # test process exits or another test starts using the same test DB.
        acu.db_queue.join()

    def setUp(self):
        # Reset state before each test
        with gallery_lock:
            local_to_global.clear()
            global_gallery.clear()
            global_last_seen.clear()

        # Create two distinct unit-normalized embeddings
        np.random.seed(42)
        v1 = np.random.randn(512).astype(np.float32)
        self.emb_person_a = v1 / np.linalg.norm(v1)

        # A small perturbation remains a close cosine match in 512 dimensions.
        v1_pert = self.emb_person_a + 0.03 * np.random.randn(512).astype(np.float32)
        self.emb_person_a_variant = v1_pert / np.linalg.norm(v1_pert)

        # Distant vector for person B (orthogonal/uncorrelated, cosine distance ~0.7 - 1.0)
        v2 = np.random.randn(512).astype(np.float32)
        v2 = v2 - np.dot(v2, self.emb_person_a) * self.emb_person_a  # orthogonalize
        self.emb_person_b = v2 / np.linalg.norm(v2)

        v2_pert = self.emb_person_b + 0.03 * np.random.randn(512).astype(np.float32)
        self.emb_person_b_variant = v2_pert / np.linalg.norm(v2_pert)

    def test_case_a_same_person_local_track_transition(self):
        """
        Test A — Same person after local track transition:
        Initial state:
          cam1: local_id = 7 -> global_id = 27
        Then simulate:
          local_id = 7 (COASTING)
          local_id = 18 (ACTIVE, with embedding that strongly matches GID 27)
        Expected:
          local_id 18 -> GID 27 (recovered)
          local_to_global[(cam1, 7)] is released
        """
        # 1. Setup initial gallery state for GID 27
        now_mono = time.monotonic()
        with gallery_lock:
            local_to_global[("cam1", 7)] = 27
            global_gallery[27] = {"cam1": deque([self.emb_person_a], maxlen=15)}
            global_last_seen[27] = now_mono

        # 2. Simulate both tracks present: 7 (coasting), 18 (active new track)
        tracks = np.array([
            [100, 100, 200, 300, 7, 0, 0.85],
            [105, 105, 205, 305, 18, 0, 0.92]
        ])
        track_status_map = {7: "COASTING", 18: "ACTIVE"}

        # 3. Call reserve_existing_global_ids
        active_claimed, ghost_claims = reserve_existing_global_ids(
            "cam1", tracks, track_status_map
        )

        # Verify GID 27 is in ghost_claims and NOT in active_claimed
        self.assertNotIn(27, active_claimed, "Coasting track GID must NOT be in active_claimed!")
        self.assertIn(27, ghost_claims, "Coasting track GID must be in ghost_claims!")
        self.assertEqual(ghost_claims[27][0], 7)

        # 4. Call match_or_register for new local track 18 with embedding close to GID 27
        resolved_gid = match_or_register(
            camera_id="cam1",
            local_id=18,
            embedding=self.emb_person_a_variant,
            exclude_gids=set(active_claimed),
            box_area=15000,
            confidence=0.92,
            run_face_detection=False,
            ghost_claims=ghost_claims
        )

        # 5. Assert that local_id 18 successfully recovered GID 27
        print(f"\n[TEST A RESULT] New local track 18 resolved to Global ID: {resolved_gid}")
        self.assertEqual(resolved_gid, 27, f"Expected GID 27, but got {resolved_gid}")

        # 6. Assert local_to_global bindings: 18 owns 27, old ghost 7 is removed
        with gallery_lock:
            self.assertEqual(local_to_global.get(("cam1", 18)), 27)
            self.assertNotIn(("cam1", 7), local_to_global, "Old coasting track mapping should be removed")

    def test_case_b_different_people_duplicate_protection(self):
        """
        Test B — Different people duplicate protection:
        Person A: local_id = 7, GID = 27 (ACTIVE)
        Person B: local_id = 18, GID = 31 (ACTIVE)
        Expected:
          local_id 7 -> GID 27
          local_id 18 -> GID 31
          Two different actively observed people are NOT merged.
        """
        now_mono = time.monotonic()
        with gallery_lock:
            local_to_global[("cam1", 7)] = 27
            global_gallery[27] = {"cam1": deque([self.emb_person_a], maxlen=15)}
            global_last_seen[27] = now_mono

            global_gallery[31] = {"cam1": deque([self.emb_person_b], maxlen=15)}
            global_last_seen[31] = now_mono

        # Both tracks are ACTIVE in the frame
        tracks = np.array([
            [50, 100, 150, 300, 7, 0, 0.90],
            [350, 100, 450, 300, 18, 0, 0.92]
        ])
        track_status_map = {7: "ACTIVE", 18: "ACTIVE"}

        # 1. Call reserve_existing_global_ids
        active_claimed, ghost_claims = reserve_existing_global_ids(
            "cam1", tracks, track_status_map
        )

        # GID 27 belongs to an actively visible person, so it MUST be in active_claimed
        self.assertIn(27, active_claimed, "Active track GID 27 must be strictly reserved in active_claimed!")
        self.assertEqual(len(ghost_claims), 0, "No ghost claims should exist for active tracks")

        # 2. Call match_or_register for Person B (local_id = 18)
        resolved_gid_b = match_or_register(
            camera_id="cam1",
            local_id=18,
            embedding=self.emb_person_b_variant,
            exclude_gids=set(active_claimed),
            box_area=15000,
            confidence=0.92,
            run_face_detection=False,
            ghost_claims=ghost_claims
        )

        print(f"[TEST B RESULT] Person B (local 18) resolved to Global ID: {resolved_gid_b}")
        self.assertEqual(resolved_gid_b, 31, f"Expected GID 31 for Person B, but got {resolved_gid_b}")

        # Both local tracks have their distinct global IDs
        with gallery_lock:
            self.assertEqual(local_to_global[("cam1", 7)], 27)
            self.assertEqual(local_to_global[("cam1", 18)], 31)

    def test_case_c_weak_match_does_not_blindly_assign(self):
        """
        Test C — Weak match:
        old_gid = 27 is in coasting state (track 7).
        new_local_id = 18 appears, but its embedding is completely different from GID 27.
        Expected:
          Do not blindly assign GID 27 to local 18.
          Local 18 should be registered as a new Global ID (or fallback).
        """
        now_mono = time.monotonic()
        with gallery_lock:
            local_to_global[("cam1", 7)] = 27
            global_gallery[27] = {"cam1": deque([self.emb_person_a], maxlen=15)}
            global_last_seen[27] = now_mono

        tracks = np.array([
            [100, 100, 200, 300, 7, 0, 0.85],
            [105, 105, 205, 305, 18, 0, 0.92]
        ])
        track_status_map = {7: "COASTING", 18: "ACTIVE"}

        active_claimed, ghost_claims = reserve_existing_global_ids(
            "cam1", tracks, track_status_map
        )

        # Give local 18 an embedding completely distant from GID 27
        resolved_gid = match_or_register(
            camera_id="cam1",
            local_id=18,
            embedding=self.emb_person_b,
            exclude_gids=set(active_claimed),
            box_area=15000,
            confidence=0.92,
            run_face_detection=False,
            ghost_claims=ghost_claims
        )

        print(f"[TEST C RESULT] Weak match candidate resolved to Global ID: {resolved_gid}")
        # Must NOT be 27
        self.assertNotEqual(resolved_gid, 27, "Weak match must NOT blindly recover GID 27!")
        self.assertGreater(resolved_gid, 0, "Expected a valid newly registered GID")


if __name__ == "__main__":
    unittest.main(verbosity=2)
