# Video to Wireframe

Convert a video of a spinning object into an approximate 3D wireframe. The browser rotates the generated model; scroll over it to zoom in or out.

## Setup on Windows

Use Python 3.10. Open PowerShell in the `video-wireframe` folder:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install opencv-python==4.12.0.88 numpy scipy scikit-image pillow
```

The black-background workflow below does not require a GPU or ML models.

## Usage

1. Save your video as `videoInput.mp4` in the project folder.
2. With the environment activated, run:

```powershell
python extract_frames.py --count 24
python build_surface.py model --black-background --rotation-degrees 360
python -m http.server 8000
```

3. Open [localhost:8000](http://localhost:8000). Scroll up to zoom in and down to zoom out. Press **Ctrl+F5** after rebuilding.

Use the actual rotation angle from the first extracted frame to the last. `360` means one complete turn; the original sample uses `735.5`.

Keep `index.html`, `index.js`, and the generated `model-data.js` together. The HTML must load `model-data.js` before `index.js`.

To use another video, replace `videoInput.mp4` and repeat the commands with its rotation angle. Rename the old `frames` and `mask_frames` folders first to avoid mixing old and new views.

## Other backgrounds

Install the optional ML dependencies. These PyTorch commands are for an NVIDIA GPU with a compatible driver:

```powershell
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install transformers==4.57.1
```

After extracting frames, run:

```powershell
python segment_object.py
python build_surface.py model --rotation-degrees 360
```

Click inside the object and press Enter. Check the generated masks before building. Model files download on first use. Depth estimation is not needed for this reconstruction method.

## Limitations

Use a fixed camera and a rigid object spinning steadily around a vertical axis. The black-background shortcut works best with a bright, solid object without holes. Reconstruction uses silhouettes, so it approximates the outer shape and cannot recover hidden dents or precise measurements.
