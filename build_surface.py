"""Build ONE closed wireframe from rotating silhouettes (a visual hull).

This is a shape approximation, not metric depth fusion or photogrammetry.
Assumptions: fixed, approximately level orthographic camera; rigid object;
vertical rotation axis; correctly supplied view angles. Do not center masks
individually: that would erase the object's motion relative to its pivot.

For the original 949-frame banana clip, measured repeats are about 464 frames
apart. The first-to-last span is approximately 735.5 degrees, not 360.
"""

import argparse
import json
from pathlib import Path
import re
import shutil

import numpy as np
from scipy import ndimage
from skimage.measure import marching_cubes


def load_masks(directory):
    paths = sorted(directory.glob("mask_*.npy"))
    if len(paths) < 4:
        raise ValueError(f"Need at least four mask_*.npy files in {directory}")
    matches = [re.fullmatch(r"mask_(\d+)\.npy", p.name) for p in paths]
    if not all(matches):
        raise ValueError("Expected numbered mask_0000.npy, mask_0001.npy, ... files")
    ids = [int(m[1]) for m in matches]
    if ids != list(range(len(ids))):
        raise ValueError("Mask numbers must run from 0000 without missing views.")
    masks = []
    shape = None
    for path in paths:
        values = np.load(path, allow_pickle=False)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError(f"{path.name}: expected a finite 2D mask")
        if not np.isin(values, [0, 1, 255]).all():
            raise ValueError(f"{path.name}: expected a binary mask (0/1 or 0/255)")
        mask = values != 0
        if shape is not None and mask.shape != shape:
            raise ValueError(f"{path.name}: all masks must have the same dimensions")
        shape = mask.shape
        if not mask.any() or mask.all():
            raise ValueError(f"{path.name}: mask is empty or fills the entire image")
        if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
            raise ValueError(f"{path.name}: object touches an image edge; check segmentation/cropping")
        masks.append(mask)
    return masks


def masks_from_black_background(directory):
    """Banana-clip shortcut: largest bright object, filling dark surface spots.

    Only use for one unoccluded object on solid black. Do not use for objects
    with real silhouette holes or other backgrounds; supply segmented masks.
    No input files or saved SAM masks are changed.
    """
    from PIL import Image

    paths = sorted(directory.glob("frame_*.jpg"))
    if len(paths) < 4:
        raise ValueError(f"Need at least four frame_*.jpg files in {directory}")
    matches = [re.fullmatch(r"frame_(\d+)\.jpg", p.name) for p in paths]
    if not all(matches) or [int(m[1]) for m in matches] != list(range(len(paths))):
        raise ValueError("Frame numbers must run from 0000 without missing views")
    masks = []
    for path in paths:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        mask = ndimage.binary_fill_holes(rgb.max(axis=2) > 24)
        labels, count = ndimage.label(mask)
        if not count:
            raise ValueError(f"{path.name}: no foreground found")
        sizes = np.bincount(labels.ravel()); sizes[0] = 0
        mask = labels == sizes.argmax()
        if masks and mask.shape != masks[0].shape:
            raise ValueError("All frames must have matching dimensions")
        if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
            raise ValueError(f"{path.name}: foreground touches the border; check black-background assumption")
        masks.append(mask)
    return masks


def carve(masks, angles, resolution=72, axis_x=None, padding=3.0, verbose=True):
    """Intersect silhouette cones in a single object-centered coordinate grid.

    Units here are image pixels, under an orthographic camera. Z is already
    relative to the object pivot, so no artificial camera distance is rotated.
    """
    height, width = masks[0].shape
    bounds = []
    for mask in masks:
        yy, xx = np.nonzero(mask)
        bounds.append([xx.min(), xx.max(), yy.min(), yy.max()])
    bounds = np.asarray(bounds)
    if axis_x is None:
        # Across complete turns, the horizontal silhouette envelope is centered
        # on the axis. This is an estimate, not recovered camera calibration.
        axis_x = float((bounds[:, 0].min() + bounds[:, 1].max()) / 2)
    if not 0 <= axis_x < width:
        raise ValueError("--axis-x must be inside the image")
    cy = float((bounds[:, 2].min() + bounds[:, 3].max()) / 2)
    radius = max(axis_x - bounds[:, 0].min(), bounds[:, 1].max() - axis_x)
    radius += padding + 4.0
    step = 2 * radius / (resolution - 1)
    xs = np.linspace(-radius - 2 * step, radius + 2 * step, resolution + 4)
    zs = xs.copy()
    ymin = cy - bounds[:, 3].max() - padding - 3 * step
    ymax = cy - bounds[:, 2].min() + padding + 3 * step
    ys = ymin + np.arange(int(np.ceil((ymax - ymin) / step)) + 1) * step
    x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")
    field = np.full(x.shape, np.inf, dtype=np.float32)
    image_y = cy - y.ravel()
    for i, (mask, angle) in enumerate(zip(masks, angles)):
        sdf = (ndimage.distance_transform_edt(mask)
               - ndimage.distance_transform_edt(~mask)).astype(np.float32)
        # Same positive-Y rotation convention as index.js. Reversing all angles
        # mirrors the result in Z; silhouettes alone cannot resolve that ambiguity.
        image_x = axis_x + x.ravel() * np.cos(angle) + z.ravel() * np.sin(angle)
        distances = ndimage.map_coordinates(
            sdf, [image_y, image_x], order=1, mode="constant", cval=-max(width, height)
        ).reshape(field.shape)
        np.minimum(field, distances + padding, out=field)
        if verbose:
            print(f"Carved view {i + 1}/{len(masks)}", flush=True)

    # Retain one connected solid. Report discarded fragments rather than hiding
    # bad input or calibration behind the phrase 'views merged'.
    labels, count = ndimage.label(field > 0)
    if count == 0:
        raise ValueError("No common volume. Check masks, view angles, and rotation axis.")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    keep = labels == sizes.argmax()
    removed = int(sizes.sum() - sizes.max())
    if removed / sizes.sum() > 0.05:
        raise ValueError("Views produce large disconnected fragments. Check angles and masks before rebuilding.")
    field[(labels != 0) & ~keep] = -step
    if any((field.take(i, axis=a) > 0).any() for a in range(3) for i in [0, -1]):
        raise ValueError("Reconstruction reaches the grid boundary; check mask/axis calibration.")
    # Light smoothing removes voxel stair-steps, then extract only the exterior
    # surface. All views contribute to this one volume, not separate sheets.
    field = ndimage.gaussian_filter(field, sigma=0.55)
    for a in range(3):
        for i in [0, -1]:
            face = [slice(None)] * 3
            face[a] = i
            field[tuple(face)] = -step
    if field.max() <= 0:
        raise ValueError("No resolvable surface; increase resolution or check the masks.")
    vertices, faces, _, _ = marching_cubes(
        field, level=0, spacing=(step, step, step), allow_degenerate=False
    )
    vertices += np.array([xs[0], ys[0], zs[0]])
    return vertices.astype(np.float64), faces, {
        "axis_x_pixels": axis_x,
        "center_y_pixels": cy,
        "voxel_size_pixels": float(step),
        "discarded_fragment_voxels": removed,
        "projection": "orthographic approximation",
    }


def mesh_edges(faces):
    pairs = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges, counts = np.unique(np.sort(pairs, axis=1), axis=0, return_counts=True)
    if not np.all(counts == 2):
        raise ValueError("Mesh is not closed: an edge does not have exactly two adjacent faces.")
    return edges


def validate_connected(vertex_count, edges):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                       shape=(vertex_count, vertex_count)).tocsr()
    components = connected_components(graph, directed=False, return_labels=False)
    if components != 1:
        raise ValueError(f"Expected one connected surface; got {components}. Check angles/masks.")


def smooth_mesh(vertices, edges, iterations=3):
    """Small alternating Laplacian steps; topology and closed edges are retained."""
    degree = np.bincount(edges.ravel(), minlength=len(vertices)).astype(float)
    for _ in range(iterations):
        for strength in (0.5, -0.53):
            total = np.zeros_like(vertices)
            np.add.at(total, edges[:, 0], vertices[edges[:, 1]])
            np.add.at(total, edges[:, 1], vertices[edges[:, 0]])
            vertices += strength * (total / degree[:, None] - vertices)
    return vertices


def js_name(name):
    words = re.findall(r"[a-zA-Z0-9]+", name)
    result = words[0].lower() + "".join(w.capitalize() for w in words[1:]) if words else "object"
    return "object" + result if result[0].isdigit() else result


def write_data(path, name, vertices, edges, faces, metadata):
    """Preserve any previous generated file; replace only after a complete write."""
    data = {"vertices": [dict(zip(["x", "y", "z"], map(float, v))) for v in np.round(vertices, 7)],
            "edges": edges.tolist(), "faces": faces.tolist(), "metadata": metadata}
    contents = "// One closed visual-hull approximation; not metric depth fusion.\n"
    for suffix, key in [("Vertices", "vertices"), ("Edges", "edges"),
                        ("Faces", "faces"), ("Metadata", "metadata")]:
        contents += f"const {name}{suffix} = " + json.dumps(data[key], separators=(",", ":"), allow_nan=False) + ";\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")
    temporary.write_text(contents, encoding="utf-8")
    if path.exists():
        backup = path.with_name(path.name + ".bak")
        index = 1
        while backup.exists():
            backup = path.with_name(path.name + f".bak{index}")
            index += 1
        shutil.copy2(path, backup)
        print(f"Previous data preserved: {backup}")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("object_name", nargs="?", default="model")
    parser.add_argument("--masks", type=Path, default=Path("mask_frames"))
    parser.add_argument("--black-background", action="store_true",
                        help="Use the supplied banana clip's black background to segment frames without ML")
    parser.add_argument("--frames", type=Path, default=Path("frames"), help="Input for --black-background only")
    rotation = parser.add_mutually_exclusive_group(required=True)
    rotation.add_argument("--rotation-degrees", type=float,
                          help="Signed turn from FIRST extracted view to LAST; assumes constant speed.")
    rotation.add_argument("--angles", type=Path,
                          help="JSON array of measured degrees, one per mask, for nonuniform rotation.")
    parser.add_argument("--resolution", type=int, default=72, help="Grid samples across model (32-160)")
    parser.add_argument("--axis-x", type=float, help="Rotation-axis image X; default estimates silhouette envelope")
    parser.add_argument("--mask-padding", type=float, default=3, help="Outward mask tolerance in original pixels")
    parser.add_argument("--output", type=Path, help="Default: <object>-data.js in current directory")
    args = parser.parse_args()
    if not 32 <= args.resolution <= 160:
        parser.error("--resolution must be between 32 and 160")
    if not np.isfinite(args.mask_padding) or not 0 <= args.mask_padding <= 20:
        parser.error("--mask-padding must be between 0 and 20 pixels")
    if args.axis_x is not None and not np.isfinite(args.axis_x):
        parser.error("--axis-x must be finite")
    masks = (masks_from_black_background(args.frames.resolve()) if args.black_background
             else load_masks(args.masks.resolve()))
    if args.angles:
        degrees = np.asarray(json.loads(args.angles.read_text(encoding="utf-8")), dtype=float)
        if degrees.shape != (len(masks),) or not np.isfinite(degrees).all():
            parser.error("--angles must contain one finite angle per mask")
    else:
        if not np.isfinite(args.rotation_degrees) or abs(args.rotation_degrees) < 180:
            parser.error("Supply a finite turn of at least 180 degrees, preferably 360 or more")
        degrees = np.linspace(0, args.rotation_degrees, len(masks))
    if np.ptp(degrees) < 180:
        parser.error("Views must span at least half a turn; full coverage is preferable")
    print(f"Building one visual hull from {len(masks)} masks; angle span {np.ptp(degrees):.2f} degrees", flush=True)
    vertices, faces, metadata = carve(masks, np.deg2rad(degrees), args.resolution,
                                     args.axis_x, args.mask_padding)
    edges = mesh_edges(faces)
    validate_connected(len(vertices), edges)
    vertices = smooth_mesh(vertices, edges)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    vertices = (vertices - (low + high) / 2) * (2 / (high - low).max())
    metadata.update({"method": "silhouette visual hull", "views": len(masks),
                     "mask_source": "black background" if args.black_background else "saved segmentation masks",
                     "angles_degrees": degrees.tolist(), "closed": True, "connected_components": 1,
                     "limitation": "Approximate outer shape. Hidden concavities, camera calibration and metric scale are not recovered."})
    name = js_name(args.object_name)
    path = (args.output or Path(f"{name}-data.js")).resolve()
    write_data(path, name, vertices, edges, faces, metadata)
    print(f"Vertices: {len(vertices)}; edges: {len(edges)}; triangles: {len(faces)}")
    print("Mesh check: one connected surface, zero open boundary edges")
    print(f"Estimated axis X: {metadata['axis_x_pixels']:.2f} pixels")
    print(f"Saved: {path}")
    print("Approximate shape only. Keep index.html and index.js; reload with Ctrl+F5.")


if __name__ == "__main__":
    main()
