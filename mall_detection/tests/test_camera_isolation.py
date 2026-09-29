import copy
import os
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np

mall_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if mall_dir not in sys.path:
    sys.path.insert(0, mall_dir)
tests_dir = os.path.dirname(os.path.abspath(__file__))
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)

from db_test_support import initialize_test_database
import app as acu


class TestCameraIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        initialize_test_database(acu)

    def setUp(self):
        with acu.camera_health_lock:
            self.health_before = copy.deepcopy(acu.camera_health)
        with acu.gallery_lock:
            self.bindings_before = dict(acu.local_to_global)
        with acu.tracker_init_lock:
            self.trackers_before = dict(acu.trackers)
        for q in acu.upload_frame_queues.values():
            while True:
                try:
                    q.get_nowait()
                except acu.queue.Empty:
                    break

    def tearDown(self):
        with acu.camera_health_lock:
            acu.camera_health.clear()
            acu.camera_health.update(self.health_before)
        with acu.gallery_lock:
            acu.local_to_global.clear()
            acu.local_to_global.update(self.bindings_before)
        with acu.tracker_init_lock:
            acu.trackers.update(self.trackers_before)
        for q in acu.upload_frame_queues.values():
            while True:
                try:
                    q.get_nowait()
                except acu.queue.Empty:
                    break

    def test_queue_drops_oldest_without_blocking_or_growing(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        q = acu.upload_frame_queues["cam2"]
        acu.enqueue_latest_camera_frame("cam2", frame, captured_at=10.0)
        acu.enqueue_latest_camera_frame("cam2", frame, captured_at=11.0)
        acu.enqueue_latest_camera_frame("cam2", frame, captured_at=12.0)

        self.assertEqual(q.qsize(), 2)
        self.assertEqual(q.get_nowait()[0], 11.0)
        self.assertEqual(q.get_nowait()[0], 12.0)
        self.assertEqual(acu.camera_health["cam2"]["frames_dropped"], 1)
        self.assertEqual(acu.camera_health["cam1"]["frames_dropped"], 0)
        self.assertEqual(acu.camera_health["cam3"]["frames_dropped"], 0)

    def test_repeated_read_failures_recover_only_the_failed_camera(self):
        other_status = (acu.camera_health["cam1"]["status"], acu.camera_health["cam3"]["status"])
        for _ in range(10):
            failures = acu.record_camera_read_failure("cam2")
        self.assertEqual(failures, 10)
        self.assertEqual(acu.camera_health["cam2"]["status"], "recovering")
        self.assertEqual(
            (acu.camera_health["cam1"]["status"], acu.camera_health["cam3"]["status"]),
            other_status,
        )

    def test_camera_one_failure_does_not_change_camera_two_or_three(self):
        other_status = {
            key: acu.camera_health[key]["worker_status"]
            for key in ("cam2", "cam3")
        }
        with patch.object(acu, "camera_processing_loop", side_effect=RuntimeError("cam1 unplugged")):
            acu.camera_worker_cycle("cam1")
        self.assertEqual(acu.camera_health["cam1"]["status"], "reconnecting")
        self.assertEqual(
            {key: acu.camera_health[key]["worker_status"] for key in ("cam2", "cam3")},
            other_status,
        )

    def test_worker_exception_is_logged_and_camera_local_recovery_is_scheduled(self):
        with patch.object(acu, "camera_processing_loop", side_effect=RuntimeError("synthetic camera error")):
            error = acu.camera_worker_cycle("cam2")
        self.assertIn("synthetic camera error", error)
        self.assertEqual(acu.camera_health["cam2"]["status"], "reconnecting")
        self.assertEqual(acu.camera_health["cam2"]["worker_status"], "recovering")
        self.assertEqual(acu.camera_health["cam2"]["reconnection_attempts"], 1)
        self.assertEqual(acu.camera_health["cam1"]["worker_status"], self.health_before["cam1"]["worker_status"])
        self.assertEqual(acu.camera_health["cam3"]["worker_status"], self.health_before["cam3"]["worker_status"])

    def test_camera_returns_to_running_after_a_successful_frame(self):
        acu.record_camera_frame("cam2", processed=True)
        self.assertEqual(acu.camera_health["cam2"]["status"], "connected")
        self.assertEqual(acu.camera_health["cam2"]["worker_status"], "running")
        self.assertIsNotNone(acu.camera_health["cam2"]["last_processed_at"])

    def test_database_worker_uses_initialized_camera_schema(self):
        message = f"db-worker-schema-check-{os.getpid()}"
        acu.db_queue.put(("add_alert", {
            "person_id": None,
            "camera_key": "cam1",
            "zone_name": None,
            "anomaly_score": 0.0,
            "message": message,
            "alert_type": "TEST",
        }))
        acu.db_queue.join()

        session = acu.db_session()
        try:
            alert = session.query(acu.Alert).filter(acu.Alert.message == message).one()
            camera = session.query(acu.Camera).filter(acu.Camera.camera_key == "cam1").one()
            self.assertEqual(alert.camera_id, camera.id)
        finally:
            session.close()

    def test_stale_selected_camera_is_reported_by_health_endpoint(self):
        with acu.camera_health_lock:
            health = acu.camera_health["cam1"]
            health["last_frame_at"] = time.time()
            health["last_processed_at"] = time.time() - 20
            health["last_stream_frame_at"] = time.time() - 20
            health["worker_status"] = "running"
        response = acu.app.test_client().get("/health/metrics")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["cameras"]["cam1"]["status"], "stale")
        self.assertGreater(payload["cameras"]["cam1"]["last_stream_frame_age_seconds"], 5)

    def test_retry_backoff_is_bounded_and_not_tight(self):
        delay = 1.0
        values = []
        for _ in range(12):
            delay = acu.camera_retry_delay(delay)
            values.append(delay)
        self.assertGreaterEqual(values[0], 1.0)
        self.assertTrue(all(a <= b for a, b in zip(values, values[1:])))
        self.assertEqual(values[-1], 10.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
