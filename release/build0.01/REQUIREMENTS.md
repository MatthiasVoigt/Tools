# Requirements for dm_detector_build0_01.py

## Python
- Built and tested with **Python 3.14.7** (CPython).
- Older Python versions have not been tested.

## Tested platform
- macOS on Apple M4 (arm64).
- Other operating systems and CPUs have not been tested. All dependencies ship prebuilt wheels for Linux, Windows and macOS.

## Dependencies
The versions in `requirements.txt` are pinned to the tested environment. Everything else the file uses is in the Python standard library.

| package | used for |
|---|---|
| numpy | arrays and the embedded logistic-regression model |
| opencv-python | detection, geometry and image processing |
| scipy (`ndimage`) | morphology and labelling |
| scikit-image | LBP and GLCM texture features |
| pillow | reading image files |
| pillow-heif | reading `.HEIC` photos only; it bundles libheif, so no system library is needed |

These are **not** needed at runtime: scikit-learn, joblib, pandas, torch or a GPU. The model weights are embedded in the `.py` file.

On a headless server you can swap `opencv-python` for `opencv-python-headless` (same version).

## Setup on a fresh machine
With uv:

    uv venv --python 3.14 .venv
    uv pip install --python .venv -r requirements.txt

Or with venv and pip:

    python3.14 -m venv .venv
    .venv/bin/pip install -r requirements.txt

## Check that it works

    .venv/bin/python dm_detector_build0_01.py --selftest
    .venv/bin/python dm_detector_build0_01.py path/to/photo.HEIC

## Runtime
Scoring takes about 2–3 s per 12 MP photo on an Apple M4, on a single process.
