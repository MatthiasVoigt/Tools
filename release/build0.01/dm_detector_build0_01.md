# DM detector — release build 0.01

> **Note:** this document was written before the class set was changed. The release file now uses `CLASSES = ("TT", "IJ", "IJ_PC", "LA")`: TT_PC was removed from training and LA was added. Where the class list, counts or scores below differ, the `.py` file is authoritative; an updated document will follow.

`dm_detector_build0_01.py` is one self-contained Python file. It finds the Data Matrix (ECC200) codes in a photo or image array, measures their module size, and gives a print-type score per class for every code that is large enough. The classes are print types, one per catalog class folder: `TT`, `IJ`, `IJ_PC` and `LA`.

- **Build number:** `"0.01"` (a string). `get_build_number()` and `get_version()` return it, and `--version` prints it.
- **File name:** `dm_detector_build0_01` uses an underscore instead of the dot. A dot in a module name breaks `import` and `python -m unittest`.
- **Class order:** `CLASSES = ("TT", "TT_PC", "IJ", "IJ_PC")`. The sklearn model stores its classes alphabetically (`IJ, IJ_PC, TT, TT_PC`). The embedded coefficient rows were re-ordered to `CLASSES`, and the file checks this at import.
- **Constants:** `BUILD_NUMBER = "0.01"`, `MIN_PX_PER_MODULE = 25.0`, `ERRORS`, `MODEL_FEATURES` (107 names) and `MODEL_SHA256`.

## Provenance

- **Pipeline.** The file was generated on 2026-09-30 from the DM_detector pipeline, which was not modified:
  - detection, module size and deskewed crop from `dm_roi.py` via `dm_detector.py`;
  - `prepare_roi`, with baseline masks (`min_structure_modules=None`), and the print-feature extractor from `print_feature.py`;
  - the 25 px/module rule from `print_type_classifier.py`.

  Those functions are copied verbatim, using the AST, by `unit test/build0.01/build_tools/make_release.py`. Only these parts were removed: AdaIN/torch, the min-structure mask variant, crop annotation, debug returns, and the table/HTML/CLI helpers. The module docstring lists each edit.
- **Resolution rule.** Codes below 25 px/module are refused and never upscaled. Kept codes are analysed at exactly 25 px/module, using an INTER_AREA downscale of the deskewed crop (pad 0.2), the same as the classifier's crops and features.
- **Model.** The model uses the same setup as `print_type_classifier.py train`: median imputation (`keep_empty_features=True`), then `StandardScaler`, then a multinomial `LogisticRegression(class_weight="balanced", max_iter=10000)`. It was **retrained without the 12 `adain_*` features**, which leaves 107 features, so torch is not needed.
- **Training data.** The 438 codes in `Data/catalog modified/features.csv`, from 386 photos:

  | class | codes | photos |
  |---|---|---|
  | TT | 193 | 160 |
  | TT_PC | 85 | 73 |
  | IJ | 99 | 97 |
  | IJ_PC | 61 | 56 |

  The labels are the catalog subfolders. The model was trained on 2026-09-30 at 10:59 EDT with sklearn 1.9.1.
- **Model file and embedding.**
  - Source model: `model_noadain.joblib`, SHA-256 `4898e0b8766e0549b14da0c42f400c30176c25fec2cd02091487d50528c3706e`. Its CV metrics are in `metrics_noadain.json`, both in this folder.
  - The model is embedded as NumPy constants: imputer values, scaler mean/scale, coefficients and intercepts. `predict_proba` is re-implemented in NumPy and matches sklearn to within 4.4e-16.
- **Cross-validation.** The classifier's own CV code and seeds were used: `StratifiedGroupKFold(5, shuffle=True, random_state=0)`, with *block* = 10 consecutive image numbers per class and *photo* = source photo.

| model | CV | code acc | code macro-F1 | photo acc | photo macro-F1 |
|---|---|---|---|---|---|
| **build 0.01 (107 features, no AdaIN)** | block-grouped | 0.872 | **0.880** | 0.873 | 0.880 |
| **build 0.01 (107 features, no AdaIN)** | photo-grouped | 0.936 | 0.942 | 0.946 | 0.950 |
| previous `models/print_type_classifier.joblib` (119 features, with AdaIN) | block-grouped | 0.858 | 0.864 | 0.855 | 0.860 |
| previous (119 features) | photo-grouped | 0.952 | 0.955 | 0.959 | 0.961 |

Per-class F1 in the block-grouped CV (build 0.01): TT 0.866, TT_PC 0.772, IJ 0.924, IJ_PC 0.959.

**The block-grouped figure is the honest estimate.**

## Known limits

- **Confound.** Each class was photographed in a single capture session, as one contiguous range of image numbers. Session effects (lighting, item, distance) are therefore confounded with the class. Even the block-grouped CV cannot remove this, so treat all metrics as optimistic for new material and new sessions.
- **Weakest pair.** TT vs TT_PC is the weakest pair. In the block CV, 26 TT codes were predicted as TT_PC and 14 TT_PC codes as TT, giving TT_PC an F1 of 0.77.
- **Resolution.** Codes below **25 px/module** are rejected with `E_RESOLUTION_TOO_LOW`; they are never upscaled or guessed. In the catalog, 121 of 525 photos have no code of sufficient resolution.
- **Error texts.** They are **PROVISIONAL**. The texts, and possibly the conditions, are still to be decided by Matthias, so key on the status code, not on the text.
- **Codes per photo.** At most 5 codes are scored per photo, the same limit as the detector's default.
- **Primary code.** The primary result is the **largest valid code**, meaning the largest side length in pixels; ties go to the higher px/module. The mean over all valid codes, which is the convention of `print_type_classifier predict`, is also returned as `scores_mean_all_codes`.
- **Speed.** Detection plus features take about 2 to 4 s per 12 MP HEIC on an Apple-silicon Mac (single process).

## Requirements

Tested with Python **3.14.7** and these packages:

| package | version | used for |
|---|---|---|
| numpy | 2.5.3 | everything, model |
| opencv-python (`cv2`) | 5.0.0.93 | detection, crops, filters |
| pillow | 12.3.0 | image files |
| pillow-heif | 1.8.0 | HEIC/HEIF files (imported only when reading them) |
| scipy | 1.18.1 | `scipy.ndimage`: blob properties, distance transform, hole filling, median filter |
| scikit-image | 0.26.0 | `skimage.feature`: LBP and GLCM texture features |

Everything else comes from the standard library. The file imports no project modules and needs no torch, sklearn, joblib or pandas.

## Use as an importable module

```python
import sys; sys.path.insert(0, "/path/to/release/build0.01")   # or copy the .py next to your code
import dm_detector_build0_01 as dm

print(dm.get_build_number())         # "0.01"
print(dm.CLASSES)                    # ('TT', 'TT_PC', 'IJ', 'IJ_PC')

# 1) from a file (HEIC / JPEG / PNG ...; EXIF orientation applied)
r = dm.score_file("IMG_2349.HEIC")
if r["status"] == "ok":
    print(r["predicted_class"], r["scores"])      # {'TT': ..., 'TT_PC': ..., 'IJ': ..., 'IJ_PC': ...}
    print(r["scores_array"])                      # np.ndarray, CLASSES order
    for c in r["codes"]:                          # every detected code (up to 5)
        print(c["index"], c["status"], round(c["px_per_module"], 1), c["bbox"], c["scores"])
else:
    print(r["status"], r["error"])                # e.g. E_RESOLUTION_TOO_LOW + text

# 2) from a NumPy array: uint8 gray (HxW), BGR/RGB (HxWx3) or BGRA/RGBA (HxWx4)
import cv2
bgr = cv2.imread("photo.jpg")
r = dm.score_image(bgr)                           # color_order="BGR" (OpenCV default)
r = dm.score_image(rgb_array, color_order="RGB")  # PIL / matplotlib order
r = dm.score_image(gray_array)                    # 2-D uint8

# 3) JSON instead of a dict (arrays become lists, NaN becomes null)
s = dm.score_file("IMG_2349.HEIC", as_json=True)  # or dm.score_file_json(path)
s = dm.score_image(bgr, as_json=True)             # or dm.score_image_json(bgr, "BGR")
```

The result is a dict. The JSON has the same keys.

```text
build                 "0.01"
status                "ok" or an error code from ERRORS
error                 None, or the error text (PROVISIONAL)
scores                {class: probability} of the primary code, or None
scores_array          np.ndarray (4,) in CLASSES order (NaN on error)
predicted_class       argmax class of the primary code, or None
primary_code_index    1-based index into codes (detector rank), or None
scores_mean_all_codes {class: mean probability over all valid codes}, or None
n_codes, n_valid_codes
codes                 list, one dict per detected code:
                      index, status, error, px_per_module, module_px_detector, bbox [x, y, w, h],
                      polygon, side_px, detector_score, rotation, tilt_deg, polarity, grid_n,
                      module_method, scores, scores_array, predicted_class
                      (+ "features": 107 model features if return_features=True)
```

**Error handling.** Expected problems are never raised; they come back as `status` + `error`, with `scores=None` and `scores_array` all NaN:

```python
r = dm.score_file("missing.heic")
assert r["status"] == "E_FILE_NOT_FOUND" and r["scores"] is None
if r["status"] != "ok":
    log.warning("%s: %s", r["status"], r["error"])
```

A code that is too small, or whose features fail, is reported in its own `codes[i]["status"]`. The photo is `ok` as long as at least one code is valid.

Other public helpers:

- `predict_proba(X)`: the NumPy model; `X` is (n, 107) in `MODEL_FEATURES` order.
- `get_model_parameters()`: copies of the embedded constants.
- `to_json(result)`
- `run_selftest()`

## Use as a standalone application and for unit testing

```bash
PY=python   # a Python with the packages from requirements.txt, e.g. .venv/bin/python
cd release/build0.01

$PY dm_detector_build0_01.py IMG_2349.HEIC IMG_1997.HEIC   # JSON array, one object per image (+ "path")
$PY dm_detector_build0_01.py --features IMG_2349.HEIC       # include the 107 features per code
$PY dm_detector_build0_01.py --indent -1 *.HEIC             # compact JSON
$PY dm_detector_build0_01.py --version                      # 0.01
$PY dm_detector_build0_01.py --selftest                     # embedded unittest suite
$PY -m unittest -v dm_detector_build0_01                     # same tests via unittest
DM_RELEASE_TEST_IMAGE=/path/IMG_1997.HEIC DM_RELEASE_TEST_CLASS=TT $PY -m unittest dm_detector_build0_01
```

| exit code | meaning |
|---|---|
| 0 | every image scored (`status == "ok"`) / self-test passed |
| 1 | at least one image returned an error status (the JSON still contains all results) / self-test failed |
| 2 | usage error (no image given, unknown option) |

The embedded tests (`ReleaseTests`, 14 tests) contain no images. They draw a synthetic Data Matrix-like symbol in code and check:

- the build number;
- model shapes and class order;
- that scores sum to 1, the class keys, and the JSON round trip;
- that `score_file` and `score_image` agree;
- gray, BGR, RGB and BGRA input;
- each error code: missing file, garbage file, bad arrays, a blank image (no code), and a too-low resolution.

The golden test runs only when `DM_RELEASE_TEST_IMAGE` is set. `DM_RELEASE_TEST_CLASS` optionally sets the expected class. `DM_RELEASE_TMPDIR` optionally sets the folder for the temporary test files.

## Errors (PROVISIONAL)

| status | when | text (PROVISIONAL) |
|---|---|---|
| `E_FILE_NOT_FOUND` | path does not exist or is not a file | PROVISIONAL: file not found |
| `E_UNREADABLE_IMAGE` | file exists but cannot be decoded | PROVISIONAL: file could not be read as an image |
| `E_INVALID_ARRAY` | not an ndarray, dtype not uint8, bad shape/channels, < 16 px, unknown `color_order` | PROVISIONAL: invalid image array (expected uint8 gray HxW, BGR/RGB HxWx3 or BGRA/RGBA HxWx4) |
| `E_NO_CODE_DETECTED` | detector finds no Data Matrix | PROVISIONAL: no Data Matrix code detected |
| `E_RESOLUTION_TOO_LOW` | codes found, all below 25 px/module | PROVISIONAL: resolution too low (all detected codes below 25 px/module; not upscaled, no guess) |
| `E_FEATURE_EXTRACTION_FAILED` | usable codes found, but feature extraction failed for all of them | PROVISIONAL: print feature extraction failed for every usable code |
| `E_INTERNAL` | unexpected exception (detection or last-resort guard) | PROVISIONAL: internal error |

Some messages get details appended, such as the path, the dtype or the px/module value. Compare with `startswith(ERRORS[code])` or, better, compare `status`.

## Test results

The build was verified with a separate release test harness (parity against the training pipeline, the self-test, the command-line tool and a full catalog run). The test results are kept privately and are not published here.
