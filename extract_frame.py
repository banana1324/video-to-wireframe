from pathlib import Path
import cv2

#paths
folder = Path(__file__).resolve().parent
video_path = folder / "videoInput.mp4"
image_path = folder / "frame.jpg"

#open video
video = cv2.VideoCapture(str(video_path))

#see if video is opened and found
if not video.isOpened(): 
    raise  RuntimeError(f"Could not open video file: {video_path}")


#see if frame found
frameFound, frame = video.read()

if not frameFound:
    raise RuntimeError(f"Could not read frame from video file: {video_path}")



#save frame
if not cv2.imwrite(str(image_path), frame):
    raise RuntimeError(f"Could not write image file: {image_path}")


height, width, channels = frame.shape
print(f"Frame size: {width} x {height}")
print(f"Colour channels: {channels}")
print(f"Saved to: {image_path}")