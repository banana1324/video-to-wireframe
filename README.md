# Video to Wireframe

A project for converting objects in video into animated 3D wireframes.

The browser displays predefined meshes. The Python scripts extract video frames and estimate depth. Full video-to-wireframe playback is not available yet.

## Setup on Windows

Use Python 3.10 and an NVIDIA GPU with a compatible driver. The current ML scripts use CUDA explicitly. GPU computation and single-frame depth estimation have been tested on a GTX 1660 with 6 GB VRAM.

Open PowerShell in the repository's main folder and run:

```powershell
cd video-wireframe
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install opencv-python==4.12.0.88 transformers==4.57.1
```

Model files download on first use, so an internet connection is needed initially.

## Usage

### View a mesh

Open `index.html` in a browser. After changing `index.js`, save it and refresh the page.

### Extract a frame and estimate depth

1. Copy an MP4 into `video-wireframe` and name it `videoInput.mp4`.
2. In PowerShell, enter `video-wireframe` and activate the environment if needed:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Run:

```powershell
python extract_frame.py
python estimate_depth.py
```

4. Open the generated images in `video-wireframe`:

| File | Contents |
| --- | --- |
| `frame.jpg` | First video frame |
| `depth.png` | Depth preview; brighter areas are predicted to be nearer |
| `depth.npy` | Numerical depth predictions |

Depth values are estimates, not measured distances in metres. These outputs are not connected to the browser viewer yet.

Object selection is WIP.
