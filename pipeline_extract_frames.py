#Extract evenly spaced frames from videoInput.mp4 for reconstruction

import argparse
from pathlib import Path

import cv2


parser = argparse.ArgumentParser()
parser.add_argument("--count", type=int, default=24,
                    help="Number of views to extract across the video.")
args = parser.parse_args()

if args.count < 2:
    raise ValueError("--count must be at least 2.")

folder = Path(__file__).resolve().parent
video_path = folder / "videoInput.mp4"
frames_dir = folder / "frames"
frames_dir.mkdir(exist_ok=True)

video = cv2.VideoCapture(str(video_path))
if not video.isOpened():
    raise RuntimeError(f"Could not open video file: {video_path}")

total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
if total < 2:
    raise RuntimeError("The video does not contain enough frames.")

indices = [round(i * (total - 1) / (args.count - 1)) for i in range(args.count)]

for output_index, frame_index in enumerate(indices):
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    found, frame = video.read()
    if not found:
        raise RuntimeError(f"Could not read frame {frame_index}.")

    output_path = frames_dir / f"frame_{output_index:04d}.jpg"
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Could not write {output_path}.")

video.release()
print(f"Extracted {len(indices)} views to {frames_dir}")
