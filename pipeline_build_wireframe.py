"""Merge rotating depth views into one canonical wireframe model.

Assumption: the camera is fixed and the object rotates around the image-space
vertical axis. The default video makes one complete turn.
"""

import argparse
import re
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("object_name", nargs="?", default="object")
parser.add_argument("--rotation-degrees", type=float, default=360.0,
                    help="Total object turn represented by the video.")
parser.add_argument("--spacing", type=int, default=14,
                    help="Pixel spacing between sampled points.")
parser.add_argument("--depth-strength", type=float, default=1.0)
args = parser.parse_args()

if args.spacing < 1:
    raise ValueError("--spacing must be positive.")

folder = Path(__file__).resolve().parent
depth_dir = folder / "depth_frames"
masks_dir = folder / "mask_frames"
depth_paths = sorted(depth_dir.glob("depth_*.npy"))
mask_paths = sorted(masks_dir.glob("mask_*.npy"))
if not depth_paths or len(depth_paths) != len(mask_paths):
    raise RuntimeError("depth_frames and mask_frames must contain the same number of views.")

depths = [np.load(path).astype(np.float32) for path in depth_paths]
masks = [np.load(path).astype(bool) for path in mask_paths]
for depth, mask in zip(depths, masks):
    if depth.ndim != 2 or depth.shape != mask.shape:
        raise ValueError("Every depth map and mask must have matching 2D shapes.")

valid_depths = [depth[mask & np.isfinite(depth)] for depth, mask in zip(depths, masks)]
valid_depths = [values for values in valid_depths if values.size]
if not valid_depths:
    raise ValueError("No valid foreground depth pixels found.")
all_object_depth = np.concatenate(valid_depths)
minimum = float(all_object_depth.min())
maximum = float(all_object_depth.max())

height, width = depths[0].shape
focal_length = max(width, height)
center_x = (width - 1) / 2
center_y = (height - 1) / 2
vertices = []
edges = []

def rotate_y(point, angle):
    cosine = np.cos(angle)
    sine = np.sin(angle)
    x, y, z = point
    return np.array([x * cosine - z * sine, y, x * sine + z * cosine])

for frame_index, (depth, foreground) in enumerate(zip(depths, masks)):
    foreground = foreground & np.isfinite(depth)
    normalized = np.zeros_like(depth, dtype=np.float32)
    normalized[foreground] = (depth[foreground] - minimum) / max(maximum - minimum, 1e-8)
    distance = 3.0 + (1.0 - normalized) * args.depth_strength

    angle = np.deg2rad(args.rotation_degrees * frame_index / max(len(depths) - 1, 1))
    frame_indices = {}
    for y in range(0, height, args.spacing):
        for x in range(0, width, args.spacing):
            if not foreground[y, x]:
                continue
            z = float(distance[y, x])
            camera_point = np.array([
                (x - center_x) * z / focal_length,
                (center_y - y) * z / focal_length,
                z,
            ])
            # Undo this frame's known turn so all views share frame 0's pose.
            canonical_point = rotate_y(camera_point, -angle)
            frame_indices[(x, y)] = len(vertices)
            vertices.append(canonical_point)

    for (x, y), start_index in frame_indices.items():
        for next_x, next_y in ((x + args.spacing, y), (x, y + args.spacing)):
            end_index = frame_indices.get((next_x, next_y))
            if end_index is None:
                continue
            edges.append([start_index, end_index])

vertices = np.asarray(vertices, dtype=np.float64)
lower = vertices.min(axis=0)
upper = vertices.max(axis=0)
vertices -= (lower + upper) / 2
vertices *= 2.0 / max(float((upper - lower).max()), 1e-8)

words = [word for word in re.split(r"[^0-9a-zA-Z]+", args.object_name.strip()) if word]
camel_name = (words[0].lower() + "".join(word.capitalize() for word in words[1:])) if words else "object"
vertices_var = f"{camel_name}Vertices"
edges_var = f"{camel_name}Edges"
data_path = folder / f"{camel_name}-data.js"
data_path.write_text(
    f"const {vertices_var} = [\n" +
    "\n".join(f"    {{ x: {x:.8f}, y: {y:.8f}, z: {z:.8f} }}," for x, y, z in vertices) +
    f"\n];\n\nconst {edges_var} = [\n" +
    "\n".join(f"    [{a}, {b}]," for a, b in edges) +
    "\n];\n",
    encoding="utf-8",
)
print(f"Views merged: {len(depths)}")
print(f"Vertices: {len(vertices)}")
print(f"Edges: {len(edges)}")
print(f"Saved: {data_path}")
