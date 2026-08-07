# AccuTrack Optimization Report

## Architecture and data flow

`camera_processing_loop` owns one OpenCV capture and one lazily-created YOLO
and StrongSORT/OSNet pipeline per started camera. It produces local tracks,
maps them to global Re-ID identities, computes zones/dwell/anomaly state, and
only the selected camera encodes an annotated JPEG. The browser consumes that
JPEG through `/video_feed` and receives analytics over Socket.IO.

The processing model is one daemon thread per camera that has been selected.
Shared state is protected by dedicated locks for gallery, per-camera behavior,
and dashboard state. Before this change, the browser stream generator could
write the same encoded frame continuously for every connected client.

## Changes made

| File | Change | Reason |
| --- | --- | --- |
| `app.py` | Lazy StrongSORT/OSNet construction | Avoids loading seven unused tracker/Re-ID instances at startup. |
| `app.py` | Condition-driven, 12 FPS-capped MJPEG producer | Eliminates busy-loop retransmission of unchanged JPEGs and bounds client bandwidth. |
| `app.py` | Paused hidden cameras plus frame-aligned switch | Prevents out-of-sync cameras from polluting the global Re-ID gallery; the selected Wildtrack view seeks to the last active frame before matching resumes. |
| `app.py` | Recent-window Re-ID and existing-ID reservation | Prevents stale gallery identities and same-frame duplicate assignment from causing ID switches. |
| `app.py` | Conservative duplicate-track filter | Removes invalid, repeated-ID, and nearly identical (IoU >= 0.95) boxes without suppressing nearby people. |
| `app.py` | 15-minute identity/history TTL and 1,000-identity gallery cap | Prevents unbounded Re-ID, local mapping, and behavior-history growth while preserving short cross-camera disappearances. |
| `app.py` | `/health/metrics` and rolling timing telemetry | Supplies repeatable FPS, timing, resident-memory, thread, and Re-ID-cardinality measurements. |
| `app.py` | Werkzeug compatibility flag | Fixes the verified startup failure with the installed Flask-SocketIO version. |
| `templates/index.html` | Hides the Behavior Model controls and stops status updates | Removes that panel from the operational dashboard and avoids its status WebSocket traffic. |
| `requirements.txt` | Adds `psutil` | Used solely by the metrics endpoint for process measurements. |

## Measurements

The unmodified server could not complete a valid baseline run in this
environment: Flask-SocketIO raised a RuntimeError that rejected Werkzeug.
Therefore no before/after percentage is reported.

After the compatibility fix, with `cam1` on the supplied CPU-only machine:

| Metric | Observed value |
| --- | ---: |
| Inference EWMA | 124.65 ms per detection pass |
| StrongSORT EWMA | 363.25 ms per detection pass |
| Re-ID EWMA | 41.84 ms per detection pass |
| JPEG encode EWMA | 5.55 ms |
| Processing rate | 4.86 FPS |
| Resident memory | 706.19 MB |
| Python threads | 34 |
| Gallery identities / local mappings | 17 / 21 |
| 3-second MJPEG transfer | 990,956 bytes (about 330 KB/s) |

Use `GET /health/metrics` during a controlled workload to capture the same
figures before and after configuration changes. Browser CPU, layout, paint,
and GPU-process measurements require Chrome DevTools against a real browser
session and were not guessed.

## Validation performed

- Python syntax compilation passed for `app.py`, `models.py`, and
  `trajectory_ae.py`.
- The Flask index and metrics endpoints returned HTTP 200 from a live server.
- A focused check verified duplicate-track suppression and same-frame Re-ID
  exclusion/mapping stability.
- The MJPEG endpoint transferred only encoded stream data from the live
  server; it no longer loops immediately over an unchanged `latest_frame`.

## Remaining bottlenecks and next steps

- CPU-only StrongSORT dominates the measured detection cycle. GPU execution or
  a smaller/faster Re-ID backend is the largest likely throughput improvement,
  but must be benchmarked against ID-switch accuracy.
- The dashboard intentionally processes one selected camera at a time. This
  makes camera-switch Re-ID reliable and avoids background CPU use. A genuine
  simultaneous multi-camera wall needs a synchronized multi-camera scheduler
  rather than independently running camera threads.
- The frontend is vanilla HTML/JavaScript rather than React. Its existing
  person-list reconciliation already avoids rebuilding unchanged rows; a
  Chrome performance trace is needed before further UI changes.
- Multi-camera Re-ID quality must be evaluated using synchronized annotated
  footage (IDF1/HOTA plus cross-camera false-merge and false-split rates).
  Appearance embeddings alone cannot guarantee identity correctness across
  arbitrary viewpoints and identical clothing.
