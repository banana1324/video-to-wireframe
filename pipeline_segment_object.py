"""Select the object once, then segment it in every extracted frame."""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor


folder = Path(__file__).resolve().parent
frames_dir = folder / "frames"
masks_dir = folder / "mask_frames"
masks_dir.mkdir(exist_ok=True)
frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
if not frame_paths:
    raise RuntimeError("No extracted frames found. Run pipeline_extract_frames.py first.")

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "facebook/sam2.1-hiera-tiny"
print(f"Loading {model_name} on {device}")
processor = Sam2Processor.from_pretrained(model_name)
model = Sam2Model.from_pretrained(model_name).to(device)
model.eval()

first_image = Image.open(frame_paths[0]).convert("RGB")
scale = min(900 / first_image.width, 600 / first_image.height, 1.0)
preview_size = (max(1, round(first_image.width * scale)),
                max(1, round(first_image.height * scale)))
preview = cv2.cvtColor(np.asarray(first_image.resize(preview_size)), cv2.COLOR_RGB2BGR)

selected_point = None
window_name = "Click inside the object, then press Enter"


def on_click(event, x, y, flags, param):
    global selected_point
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_point = (
            min(first_image.width - 1, int(x * first_image.width / preview_size[0])),
            min(first_image.height - 1, int(y * first_image.height / preview_size[1])),
        )
        marked = preview.copy()
        cv2.circle(marked, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow(window_name, marked)


cv2.namedWindow(window_name)
cv2.imshow(window_name, preview)
cv2.setMouseCallback(window_name, on_click)
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 13 and selected_point is not None:
        break
    if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        cv2.destroyAllWindows()
        raise SystemExit("Selection cancelled.")
cv2.destroyAllWindows()

# Use the same stable image-space point for all views. For this turntable video
# the banana rotates in place, so its center remains on the object.
point_x, point_y = selected_point
input_points = [[[[point_x, point_y]]]]
input_labels = [[[1]]]

for frame_path in frame_paths:
    image = Image.open(frame_path).convert("RGB")
    inputs = processor(
        images=image,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        outputs = model(**inputs)

    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu()
    )[0]
    best_index = outputs.iou_scores[0, 0].argmax().item()
    foreground = masks[0, best_index].numpy().astype(bool)
    np.save(masks_dir / f"{frame_path.stem.replace('frame_', 'mask_')}.npy", foreground)
    Image.fromarray((foreground * 255).astype(np.uint8)).save(
        masks_dir / f"{frame_path.stem.replace('frame_', 'mask_')}.png"
    )
    print(f"Saved mask for {frame_path.name}")

print(f"Segmented {len(frame_paths)} views.")
