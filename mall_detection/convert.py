import cv2
import glob

def make_video(output, frames_pattern, fps=10):
    frames = sorted(glob.glob(frames_pattern))
    if not frames:
        print(f"No frames found for {output}")
        return
    first = cv2.imread(frames[0])
    if first is None:
        print(f"Could not read frames for {output}")
        return
    h, w = first.shape[:2]
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        img = cv2.imread(f)
        if img is not None:
            out.write(img)
    out.release()
    print(f"Done! {output} created with {len(frames)} frames")

# Mall dataset — your main video (already exists, skip if you want)
make_video("crowd_video.mp4", "mall_dataset/frames/frames/*.jpg")

# UCSD pedestrian — moderate crowd (outdoor walkway)
make_video("moderate_video.mp4", "UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/Train001/*.tif")

# Low light — simulated from mall dataset
import numpy as np
frames = sorted(glob.glob("mall_dataset/frames/frames/*.jpg"))
first = cv2.imread(frames[0])
h, w = first.shape[:2]
out = cv2.VideoWriter("lowlight_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
for f in frames:
    img = cv2.imread(f)
    dark = cv2.convertScaleAbs(img, alpha=0.3, beta=0)
    out.write(dark)
out.release()
print("Done! lowlight_video.mp4 created")
# MOT17-09 sequence
frames = sorted(glob.glob("MOT17-09-FRCNN/img1/*.jpg"))
first = cv2.imread(frames[0])
h, w = first.shape[:2]
out = cv2.VideoWriter("mot17_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
for f in frames:
    img = cv2.imread(f)
    out.write(img)
out.release()
print("Done! mot17_video.mp4 created")



