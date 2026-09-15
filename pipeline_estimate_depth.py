"""Estimate one depth map per extracted frame."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


folder = Path(__file__).resolve().parent
frames_dir = folder / "frames"
depth_dir = folder / "depth_frames"
depth_dir.mkdir(exist_ok=True)

frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
if not frame_paths:
    raise RuntimeError("No extracted frames found. Run pipeline_extract_frames.py first.")

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "depth-anything/Depth-Anything-V2-Small-hf"
print(f"Loading {model_name} on {device}")
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device)
model.eval()

for frame_path in frame_paths:
    image = Image.open(frame_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        prediction = torch.nn.functional.interpolate(
            outputs.predicted_depth.unsqueeze(1),
            size=(image.height, image.width),
            mode="bicubic",
            align_corners=False,
        )

    depth = prediction.squeeze().cpu().numpy().astype(np.float32)
    output_path = depth_dir / f"{frame_path.stem.replace('frame_', 'depth_')}.npy"
    np.save(output_path, depth)
    print(f"Saved {output_path.name}")

print(f"Estimated depth for {len(frame_paths)} views.")
