# DM detector demo app

`dm_demo_app.py` is a single-window GUI (tkinter + Pillow `ImageTk`) for trying the DM detector release files. It imports **only** a release file, `release/build<N>/dm_detector_build<N_>.py`, via `importlib`. It never imports the project modules (`dm_roi`, `print_feature`, `print_type_classifier`, ...).

## Run

```bash
# from the repository root (the folder that contains release/)
python release/demo/dm_demo_app.py
```

1. **Release build** drop-down: pick a build (the newest is selected by default). **Refresh** rescans `release/`.
2. **Open images…**: select one or more HEIC/JPG/PNG files.
3. **Process**: images are scored in a background thread, so the window stays responsive; a progress bar and the status line show progress.

The table has one row per image:

| column | content |
|---|---|
| 1 | thumbnail of the input image (EXIF orientation applied) and its file name |
| 2 | crop of the primary (largest valid) code: the normalised 25 px/module crop from `score_file(..., return_crop=True)`. If there is no crop, it shows the error status (for example `E_RESOLUTION_TOO_LOW`) or `crop not supported by this build` |
| 3 | horizontal bar chart of the class scores, using the class names and order from the selected module's `CLASSES`. The top class is red with a black outline and bold text, and every bar shows its value |

## How version selection works

- **Scanning.** At startup and on **Refresh**, the app scans `release/` for subfolders whose names match `build<N>` exactly (for example `build0.01`, `build0.02`, `build7`). This skips `demo/` and backup folders such as `build0.01.bak-*`.
- **Validity.** A folder is valid only if it contains `dm_detector_build<N with dots replaced by underscores>.py`. That file is picked up automatically. Folders without it are listed as `UNAVAILABLE`; selecting one shows the reason and disables processing.
- **Sort order.** Builds are sorted numerically by build number (as a decimal), newest first. The newest valid build is selected by default.
- **Import and caching.** Selecting a build imports its release file under a unique module name (`dm_release_<N_>_<hash>`) and caches the module, so switching back does not re-import it. The status line shows the file name, the number `get_build_number()` reports, the classes, and whether crops are supported. If the import fails, the error appears in the status line.
- **Crop support.** This is detected from the signature of `score_file` (a `return_crop` parameter). Build 0.01 has none, so column 2 shows `crop not supported by this build` while the scores still display. Build 0.02 and later return the crop.

## Command-line helpers (no clicking)

```bash
PY=.venv/bin/python
$PY release/demo/dm_demo_app.py --list-builds                       # show discovered builds
$PY release/demo/dm_demo_app.py --headless IMG1.HEIC IMG2.HEIC      # score with the newest build, print a summary
$PY release/demo/dm_demo_app.py --build 0.01 --headless IMG.HEIC    # a specific build
$PY release/demo/dm_demo_app.py --smoke-ms 2000                     # open the window, close it after 2 s
$PY release/demo/dm_demo_app.py --release-dir /other/release        # scan another folder
```

## Requirements and limitations

- **Requirements.** The project `.venv` (Python 3.14.7, Tk 9.0.4) with numpy, opencv-python, pillow, pillow-heif, scipy and scikit-image, which the release file needs anyway. No extra packages are needed.
- **Speed.** Processing takes about 2 to 4 s per 12 MP HEIC and runs one image at a time.
- **Memory.** Results are shown as thumbnails only. Large batches (hundreds of images) keep every row in memory.
- **Class names.** Older builds may have other classes; the bar chart always follows the selected module's `CLASSES`.
- **Test status.** The GUI was smoke-tested programmatically: the window opened, processed images and closed. The file dialog and the clicking have not been tested by hand.
