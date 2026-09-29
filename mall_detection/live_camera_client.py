import cv2
import requests
import time
import argparse
import sys

# Camera keys the central server actually reads from the upload queue
# (see VIDEO_SOURCES in app.py — cam1 is the server's own local webcam,
# so uploading to it here would be silently dropped and never displayed).
VALID_UPLOAD_CAMERAS = {"cam2", "cam3"}


def main():
    parser = argparse.ArgumentParser(description="AcuTrack Live Camera Streamer Client")
    parser.add_argument("--server", required=True, help="IP address or hostname of the Central Server (e.g. 192.168.1.10)")
    parser.add_argument("--camera", required=True, help="Camera key (e.g., cam2, cam3)")
    parser.add_argument("--webcam", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--video", help="Optional path to a video file to stream (simulates webcam)")
    args = parser.parse_args()

    if args.camera not in VALID_UPLOAD_CAMERAS:
        print(f"WARNING: '{args.camera}' is not one of the server's upload-type cameras "
              f"({sorted(VALID_UPLOAD_CAMERAS)}). If this key isn't marked \"upload\" in the "
              f"server's VIDEO_SOURCES, frames will be accepted but never processed or shown.")

    url = f"http://{args.server}:5000/api/camera/upload/{args.camera}"
    
    if args.video:
        print(f"Streaming from video file {args.video} to {url}...")
        cap = cv2.VideoCapture(args.video)
    else:
        print(f"Streaming from webcam {args.webcam} to {url}...")
        # On Windows, OpenCV's default backend (MSMF) is frequently slow to
        # initialize or flat-out fails to grab frames from some webcams.
        # DirectShow (CAP_DSHOW) is what app.py already uses for its own local
        # webcam on Windows — use the same backend here for the same reason.
        if sys.platform.startswith('win'):
            cap = cv2.VideoCapture(args.webcam, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(args.webcam)
        # Set frame resolution to speed up capture (640x480 is ideal)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Keep only the newest webcam frame when a network post takes longer than
    # capture. This bounds latency instead of uploading an increasingly old queue.
    if not args.video:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened() and args.video:
        print(f"ERROR: Could not open {'video file ' + args.video if args.video else 'webcam index ' + str(args.webcam)}.")
        return

    if not args.video:
        # Some webcams report as "opened" immediately but return failed reads
        # for their first several frames while auto-exposure/focus settles.
        # Warm up here so the main loop doesn't start by spamming failure logs.
        warmup_ok = False
        for _ in range(30):
            ret, _ = cap.read()
            if ret:
                warmup_ok = True
                break
            time.sleep(0.1)
        if not warmup_ok:
            print(f"WARNING: Webcam {args.webcam} did not return a warmup frame; "
                  "continuing so the recovery loop can retry it.")

    # Use persistent session with connection pooling to prevent Windows socket exhaustion (WinError 10053/10054)
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=2, max_retries=2)
    session.mount('http://', adapter)

    consecutive_failures = 0
    MAX_BACKOFF = 5.0
    consecutive_grab_failures = 0
    reconnect_attempts = 0
    total_reconnect_attempts = 0
    last_read_failure_burst = 0

    try:
        while True:
            try:
                ret, frame = cap.read()
            except Exception as exc:
                ret, frame = False, None
                if consecutive_grab_failures == 0:
                    print(f"Webcam read raised {type(exc).__name__}: {exc}; entering recovery.")
            if not ret:
                if args.video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                consecutive_grab_failures += 1
                if consecutive_grab_failures == 1 or consecutive_grab_failures % 50 == 0:
                    print(f"Failed to grab frame ({consecutive_grab_failures} in a row). "
                          f"Retrying...")
                if consecutive_grab_failures == 100:
                    print("Camera has failed to produce a frame for ~10s straight. "
                          "It may have been disconnected, or another application just "
                          "took control of it — check that before assuming this script "
                          "is broken.")
                time.sleep(0.1)
                if consecutive_grab_failures >= 10:
                    reconnect_attempts += 1
                    total_reconnect_attempts += 1
                    backoff = min(MAX_BACKOFF, 0.5 * reconnect_attempts)
                    print(f"Reopening webcam after {consecutive_grab_failures} failed reads; "
                          f"attempt {reconnect_attempts} in {backoff:.1f}s.")
                    cap.release()
                    time.sleep(backoff)
                    if sys.platform.startswith('win'):
                        cap = cv2.VideoCapture(args.webcam, cv2.CAP_DSHOW)
                    else:
                        cap = cv2.VideoCapture(args.webcam)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if cap.isOpened():
                        consecutive_grab_failures = 0
                        reconnect_attempts = 0
                        print("Webcam reopened; resuming upload.")
                continue
            if consecutive_grab_failures:
                last_read_failure_burst = max(last_read_failure_burst, consecutive_grab_failures)
            consecutive_grab_failures = 0

            # Encode frame to JPEG (quality 70 is lightweight and fast)
            encoded, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not encoded:
                print("JPEG encoding failed; dropping frame.")
                continue
            
            # Send frame to central server via persistent session
            try:
                response = session.post(
                    url,
                    files={'frame': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')},
                    headers={
                        'X-Camera-Reconnection-Attempts': str(total_reconnect_attempts),
                        'X-Camera-Last-Read-Failure-Burst': str(last_read_failure_burst),
                    },
                    timeout=3.0
                )
                if response.status_code != 200:
                    consecutive_failures += 1
                    if consecutive_failures == 1 or consecutive_failures % 20 == 0:
                        print(f"Server returned HTTP {response.status_code}: {response.text[:200]}")
                    time.sleep(min(MAX_BACKOFF, 0.5 * consecutive_failures))
                    continue
                else:
                    consecutive_failures = 0
            except requests.exceptions.RequestException as e:
                # Network hiccups or brief server busy - back off gracefully and reconnect.
                # Backoff escalates (capped) instead of a fixed 0.5s retry-forever loop,
                # and prints periodically so a dead server connection isn't silent.
                consecutive_failures += 1
                backoff = min(MAX_BACKOFF, 0.5 * consecutive_failures)
                if consecutive_failures == 1 or consecutive_failures % 20 == 0:
                    print(f"Connection issue ({consecutive_failures} in a row): {e}. "
                          f"Retrying in {backoff:.1f}s...")
                time.sleep(backoff)

            # ~10-12 FPS matches server STREAM_FPS and eliminates buffer lag & socket aborts
            time.sleep(0.09)
            
    except KeyboardInterrupt:
        print("Streaming stopped by user.")
    finally:
        cap.release()
        session.close()

if __name__ == "__main__":
    main()
