# DM detector — release build 0.02

`dm_detector_build0_02.py` is one self-contained Python file. It finds the Data Matrix (ECC200) codes in a photo or image array, measures their module size, and gives a print-type score per class for every code that is large enough. The classes are four catalog subfolders (one per print type): `TT`, `IJ`, `IJ_PC` and `LA`. Since 2026-09-30, `LA` is new and `TT_PC` is no longer a class: its photos stay in the catalog but were left out of training.

- **Build number:** `"0.02"` (a string). `get_build_number()` and `get_version()` return it, and `--version` prints it.
- **File name:** `dm_detector_build0_02` uses an underscore instead of the dot. A dot in a module name breaks `import` and `python -m unittest`.
- **Class order:** `CLASSES = ("TT", "IJ", "IJ_PC", "LA")`. The sklearn model stores its classes alphabetically (`IJ, IJ_PC, LA, TT`). The embedded coefficient rows were re-ordered to `CLASSES`, and the file checks this at import.
- **Constants:** `BUILD_NUMBER = "0.02"`, `MIN_PX_PER_MODULE = 25.0`, `ERRORS`, `MODEL_FEATURES` (107 names) and `MODEL_SHA256`.
- **New in 0.02:** `score_file` and `score_image` accept `return_crop=True` and then also return the normalised 25 px/module code crop as an RGB uint8 array (see [Code crops](#code-crops-return_croptrue-new-in-002)). The default output is unchanged, and crops never go into JSON. Detection, features, model and scores are the same as in 0.01.

## Provenance

- **Pipeline.** The file was generated on 2026-09-30 from the DM_detector pipeline. The detection and feature code was not modified; `print_type_classifier.py` only gained the LA folder and a training-class filter that leaves out TT_PC:
  - detection, module size and deskewed crop from `dm_roi.py` via `dm_detector.py`;
  - `prepare_roi`, with baseline masks (`min_structure_modules=None`), and the print-feature extractor from `print_feature.py`;
  - the 25 px/module rule from `print_type_classifier.py`.

  Those functions are copied verbatim, using the AST, by `unit test/build0.02/build_tools/make_release.py`. Only these parts were removed: AdaIN/torch, the min-structure mask variant, crop annotation, debug returns, and the table/HTML/CLI helpers. The module docstring lists each edit.
- **Resolution rule.** Codes below 25 px/module are refused and never upscaled. Kept codes are analysed at exactly 25 px/module, using an INTER_AREA downscale of the deskewed crop (pad 0.2), the same as the classifier's crops and features.
- **Model.** The model uses the same setup as `print_type_classifier.py train`: median imputation (`keep_empty_features=True`), then `StandardScaler`, then a multinomial `LogisticRegression(class_weight="balanced", max_iter=10000)`. It was **retrained without the 12 `adain_*` features**, which leaves 107 features, so torch is not needed.
- **Training data.** 527 codes from 478 photos in `Data/catalog modified/features.csv`. The TT_PC rows (85 codes) are still in the file but excluded from training.

  | class | codes | photos |
  |---|---|---|
  | TT | 193 | 160 |
  | IJ | 99 | 97 |
  | IJ_PC | 61 | 56 |
  | LA | 174 | 165 |

  The labels are the catalog subfolders. LA was processed by the normal pipeline: 245 photos, of which 165 have a valid code, 59 are `E_RESOLUTION_TOO_LOW` and 21 have no code, giving 174 valid codes. One edited duplicate of another LA photo and the `.AAE` sidecars were skipped. The model was trained on 2026-09-30 at 13:34 EDT with sklearn 1.9.1 for build 0.01 and is **reused unchanged** in 0.02 (same file, same SHA-256).
- **Model file and embedding.**
  - Source model: `model_noadain.joblib`, SHA-256 `1fb6992135c3cc38417adfe621901aa0589744b0423149e3453715b4c620b2ed`. The `.joblib` file is not published because it is not needed at runtime. Its CV metrics are in `metrics_noadain.json` in this folder.
  - The model is embedded as NumPy constants: imputer values, scaler mean/scale, coefficients and intercepts. `predict_proba` is re-implemented in NumPy and matches sklearn to within 2.2e-16.
- **Cross-validation.** The classifier's own CV code and seeds were used: `StratifiedGroupKFold(5, shuffle=True, random_state=0)`, with *block* = 10 consecutive image numbers per class and *photo* = source photo.

| model | CV | code acc | code macro-F1 | photo acc | photo macro-F1 |
|---|---|---|---|---|---|
| **build 0.02 (107 features, no AdaIN; TT, IJ, IJ_PC, LA)** | block-grouped | 0.958 | **0.949** | 0.962 | 0.955 |
| **build 0.02 (107 features, no AdaIN; TT, IJ, IJ_PC, LA)** | photo-grouped | 0.983 | 0.982 | 0.985 | 0.984 |
| `models/print_type_classifier.joblib` (119 features with AdaIN, same 4 classes) | block-grouped | 0.954 | 0.948 | 0.958 | 0.954 |
| `models/print_type_classifier.joblib` (119 features, same 4 classes) | photo-grouped | 0.985 | 0.984 | 0.985 | 0.984 |
| earlier draft of 0.01 (107 features; TT, TT_PC, IJ, IJ_PC) | block-grouped | 0.872 | 0.880 | 0.873 | 0.880 |

Per-class F1 in the block-grouped CV (build 0.02): TT 0.964, IJ 0.907, IJ_PC 0.937, LA 0.989. The block-grouped confusion matrix (rows true, columns predicted, in the order TT, IJ, IJ_PC, LA) is TT [186, 6, 0, 1], IJ [6, 88, 5, 0], IJ_PC [0, 1, 59, 1], LA [1, 0, 1, 172]. The class set differs from the earlier draft, which had the hard TT/TT_PC pair, so its numbers are not directly comparable.

**The block-grouped figure is the honest estimate.**

## Known limits

- **Confound.** Each class was photographed in a single capture session, as one contiguous range of image numbers. Session effects (lighting, item, distance) are therefore confounded with the class. Even the block-grouped CV cannot remove this, so treat all metrics as optimistic for new material and new sessions.
- **Weakest pairs.** TT vs IJ and IJ vs IJ_PC are the weakest pairs (block CV: 6 TT codes predicted as IJ, 6 IJ as TT, 5 IJ as IJ_PC).
- **TT_PC is not a class.** A TT_PC print is scored as one of the four classes. In the catalog run, all 85 valid TT_PC codes were predicted as TT (mean p(TT) 0.983).
- **Resolution.** Codes below **25 px/module** are rejected with `E_RESOLUTION_TOO_LOW`; they are never upscaled or guessed. In the catalog, 180 of 770 photos have no code of sufficient resolution.
- **Error texts.** They are **PROVISIONAL**. The texts, and possibly the conditions, are still to be decided by Matthias, so key on the status code, not on the text.
- **Codes per photo.** At most 5 codes are scored per photo, the same limit as the detector's default.
- **Primary code.** The primary result is the **largest valid code**, meaning the largest side length in pixels; ties go to the higher px/module. The mean over all valid codes, which is the convention of `print_type_classifier predict`, is also returned as `scores_mean_all_codes`.
- **Speed.** In the build 0.02 catalog run (5 parallel workers, `cv2.setNumThreads(2)`), one image took mean 4.41 s, median 4.34 s, p90 5.84 s (min 1.33 s, max 10.19 s), measured inside `score_file`. `return_crop=True` was not used in that run.

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
import sys; sys.path.insert(0, "/path/to/release/build0.02")   # or copy the .py next to your code
import dm_detector_build0_02 as dm

print(dm.get_build_number())         # "0.02"
print(dm.CLASSES)                    # ('TT', 'IJ', 'IJ_PC', 'LA')

# 1) from a file (HEIC / JPEG / PNG ...; EXIF orientation applied)
r = dm.score_file("photo1.HEIC")
if r["status"] == "ok":
    print(r["predicted_class"], r["scores"])      # {'TT': ..., 'IJ': ..., 'IJ_PC': ..., 'LA': ...}
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
s = dm.score_file("photo1.HEIC", as_json=True)  # or dm.score_file_json(path)
s = dm.score_image(bgr, as_json=True)             # or dm.score_image_json(bgr, "BGR")
```

The result is a dict. The JSON has the same keys.

```text
build                 "0.02"
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
                      (+ "crop": RGB uint8 array for valid codes if return_crop=True)
crop                  only with return_crop=True: crop of the primary code
                      (RGB uint8 HxWx3), or None on error; never in the JSON
```

**Error handling.** Expected problems are never raised; they come back as `status` + `error`, with `scores=None` and `scores_array` all NaN:

```python
r = dm.score_file("missing.heic")
assert r["status"] == "E_FILE_NOT_FOUND" and r["scores"] is None
if r["status"] != "ok":
    log.warning("%s: %s", r["status"], r["error"])
```

A code that is too small, or whose features fail, is reported in its own `codes[i]["status"]`. The photo is `ok` as long as at least one code is valid.

## Code crops (`return_crop=True`, new in 0.02)

Both input functions take an optional keyword `return_crop` (default `False`):

```python
score_file(path, as_json=False, return_features=False, return_crop=False)
score_image(array, color_order="BGR", as_json=False, return_features=False, return_crop=False)
```

With `return_crop=True` the result gets these extra entries:

| key | content |
|---|---|
| `result["crop"]` | crop of the **primary** code (the same object as `codes[primary_code_index - 1]["crop"]`); `None` if the status is not `ok` (no code, too low resolution, file/array error ...) |
| `codes[i]["crop"]` | crop of every **valid** code (`status == "ok"`). Codes that are not valid (for example `E_RESOLUTION_TOO_LOW`) have no `"crop"` key. If building the crop itself fails, the value is `None`; the scores are never affected |

**The crop image:**

- type `numpy.ndarray`, dtype **`uint8`**, shape **`(H, W, 3)`**, channel order **RGB** (not OpenCV's BGR), C-contiguous;
- it is the classifier's crop: the detected code deskewed and upright, with 0.2 padding, resampled with `INTER_AREA` to exactly **25 px/module** (never upscaled). H and W therefore depend on the code's module count, not on the photo resolution. Measured on 2026-09-30: an LA photo (26-module code) gave `(920, 920, 3)`, a TT photo (16-module code) gave `(564, 564, 3)`;
- it is pixel-identical to the normalised crop PNGs written by the pipeline (`print_type_classifier.py build`, `Data/catalog modified`), apart from the BGR to RGB channel order.

**Unchanged behaviour:**

- Without `return_crop` (the default) the result contains no `"crop"` key at all, exactly as in 0.01.
- Scores are identical with and without `return_crop`.
- Crops are **never** put into JSON: `as_json=True`, `to_json()`, `score_*_json()` and the CLI drop every `"crop"` key. There is no CLI option for crops.

```python
import dm_detector_build0_02 as dm
from PIL import Image

r = dm.score_file("photo2.HEIC", return_crop=True)
if r["status"] == "ok":
    crop = r["crop"]                          # np.ndarray, uint8, (H, W, 3), RGB, 25 px/module
    print(r["predicted_class"], crop.shape, crop.dtype)
    Image.fromarray(crop).save("primary_code.png")          # RGB -> PIL directly
    for c in r["codes"]:
        if c.get("crop") is not None:                        # valid codes only
            Image.fromarray(c["crop"]).save("code_%d.png" % c["index"])
else:
    assert r["crop"] is None
    print(r["status"], r["error"])

# OpenCV wants BGR:
import cv2
cv2.imwrite("primary_code_cv.png", cv2.cvtColor(r["crop"], cv2.COLOR_RGB2BGR))

# score_image works the same way
r2 = dm.score_image(cv2.imread("photo.jpg"), return_crop=True)
```

Build 0.01 does not have the parameter (a `TypeError` if passed). Callers that must support both can check `"return_crop" in inspect.signature(dm.score_file).parameters`, as the demo app does.

Other public helpers:

- `predict_proba(X)`: the NumPy model; `X` is (n, 107) in `MODEL_FEATURES` order.
- `get_model_parameters()`: copies of the embedded constants.
- `to_json(result)`
- `run_selftest()`

## Use as a standalone application and for unit testing

```bash
PY=python   # a Python with the packages from requirements.txt, e.g. .venv/bin/python
cd release/build0.02

$PY dm_detector_build0_02.py photo1.HEIC photo2.HEIC       # JSON array, one object per image (+ "path")
$PY dm_detector_build0_02.py --features photo1.HEIC         # include the 107 features per code
$PY dm_detector_build0_02.py --indent -1 *.HEIC             # compact JSON
$PY dm_detector_build0_02.py --version                      # 0.02
$PY dm_detector_build0_02.py --selftest                     # embedded unittest suite
$PY -m unittest -v dm_detector_build0_02                     # same tests via unittest
DM_RELEASE_TEST_IMAGE=path/to/tt_photo.HEIC DM_RELEASE_TEST_CLASS=TT $PY -m unittest dm_detector_build0_02
```

| exit code | meaning |
|---|---|
| 0 | every image scored (`status == "ok"`) / self-test passed |
| 1 | at least one image returned an error status (the JSON still contains all results) / self-test failed |
| 2 | usage error (no image given, unknown option) |

The embedded tests (`ReleaseTests`, 15 tests: 14 that need no images plus the optional golden test) contain no images apart from the golden test. They draw a synthetic Data Matrix-like symbol in code and check:

- the build number;
- model shapes and class order;
- that scores sum to 1, the class keys, and the JSON round trip;
- that `score_file` and `score_image` agree;
- gray, BGR, RGB and BGRA input;
- each error code: missing file, garbage file, bad arrays, a blank image (no code), and a too-low resolution;
- `return_crop` (new in 0.02): default output has no crop, scores unchanged, crop is RGB uint8 HxWx3 at about 25 px/module, `result["crop"]` is the primary code's crop, only valid codes carry a crop, crops never appear in JSON, and error results give `crop = None`.

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

## Test results and outputs

The test results for this build were produced by the project's reusable release test harness (`unit test/run_release_tests.py --build 0.02`, run from the project root). The harness and its outputs are kept privately and are not published here. The harness writes these outputs:

| output | files |
|---|---|
| parity | `parity.md`, `parity.csv` |
| selftest, unittest and CLI logs | `selftest.log`, `unittest.log`, `cli.log`, `cli_check.json` |
| self-contained import check | `import_test.log` |
| full catalog run | `catalog_run.csv`, `catalog_run_codes.csv`, `catalog_run_code_features.csv`, `catalog_run_summary.md` |
| PCA and LDA 2D scatter plots of the top-40 non-adain features | `pca_top_features.png/.csv`, `lda_top_features.png/.csv` |
| integrity check of the pipeline sources and data | `integrity.txt` |
| summary | `test_report.md` |

## Build 0.02 test results

All numbers below are from `unit test/build0.02/test_report.md` and `catalog_run_summary.md` (harness run finished 2026-09-30 14:49 EDT, exit 0).

- **Parity** (training codes): 527 of 527 matched and compared; max |feature diff| 2.84e-14 (NaN mismatches 0), max |probability diff| 2.04e-14, argmax agreement 527 / 527; NumPy model vs sklearn max diff 2.22e-16. **PASS** (target <= 1e-6).
- **Built-in tests and CLI:** `--selftest` exit 0, ran 15 tests, OK (skipped=1, the golden test); `python -m unittest` exit 0, ran 15 tests, OK (skipped=1). CLI: 3 images rc 0, status ok/ok/ok, codes 2/1/3 (all valid); missing path rc 1 with `E_FILE_NOT_FOUND`; no arguments rc 2; bad option rc 2; `--version` prints `0.02`. **PASS**.
- **Self-contained import** from a neutral folder: build `0.02`, status `ok` on a TT photo (p(TT) 0.99991), no forbidden modules loaded (`dm_roi`, `dm_detector`, `print_feature`, `print_type_classifier`, `sklearn`, `joblib`, `torch`, `pandas`).
- **Catalog run** (770 photos, all 5 folders; in-sample):
  - status: 551 ok, 180 `E_RESOLUTION_TOO_LOW`, 39 `E_NO_CODE_DETECTED`; 1065 codes detected, 612 valid.
  - Model classes TT, IJ, IJ_PC, LA (680 photos): 478 ok, 164 `E_RESOLUTION_TOO_LOW`, 38 `E_NO_CODE_DETECTED`; 527 valid codes in 478 photos.
  - Photo level (primary code, n=478): accuracy **1.0000**, macro-F1 **1.0000**. Code level (n=527): accuracy **1.0000**, macro-F1 **1.0000**. This is **in-sample**, because the catalog is the training data.

    | class | photos | ok | E_RESOLUTION_TOO_LOW | E_NO_CODE_DETECTED | codes detected | valid codes | photo recall | code recall |
    |---|---|---|---|---|---|---|---|---|
    | TT | 218 | 160 | 47 | 11 | 321 | 193 | 100.0% | 100.0% |
    | IJ | 129 | 97 | 30 | 2 | 165 | 99 | 100.0% | 100.0% |
    | IJ_PC | 88 | 56 | 28 | 4 | 118 | 61 | 100.0% | 100.0% |
    | LA | 245 | 165 | 59 | 21 | 316 | 174 | 100.0% | 100.0% |
    | TT_PC (not in model) | 90 | 73 | 16 | 1 | 145 | 85 | n/a | n/a |

  - TT_PC (not a model class, left out of the accuracy): all 85 valid codes and all 73 primary codes predicted as TT; mean code probability TT 0.983, IJ 0.008, IJ_PC 0.008, LA 0.000.
  - Timing: mean 4.41 s, median 4.34 s, p90 5.84 s per image (5 workers); wall time 682 s for 770 images.
- **Honest estimate (model CV):** block-grouped code macro-F1 0.9490, accuracy 0.9583; photo-grouped macro-F1 0.9817. Same model as 0.01, so these equal the 0.01 figures.
- **Scatter plots** (top 40 non-adain features, 527 codes): PCA PC1 12.6 %, PC2 10.2 % explained variance; 22 codes misclassified in the block-grouped CV (out of fold), 0 in-sample errors. Closest class pairs: in PCA IJ/IJ_PC (1.45 pooled SD), in LDA TT/IJ (2.08).
- **Integrity:** all SHA-256 of the pipeline sources, `models/print_type_classifier.joblib`, `Data/catalog modified/features.csv` and `manifest.csv`, and `labels.json` identical to before the release work.

## Data manifest (not published)

A private `data_manifest.json` (generated 2026-09-30 13:40 EDT) records the input data that build 0.02 was built and tested against: the model file with its SHA-256, origin (trained for build 0.01, reused unchanged), classes and the classes left out of training (TT_PC); per catalog folder the file count, total bytes, whether the folder is a model class, and size + SHA-256 of every file; size, SHA-256 and mtime of `features.csv`, `manifest.csv` and `labels.json`; and the row counts `features_rows_per_class`, `training_rows_per_class` (527 in total), `training_photos_per_class` and `manifest_rows_per_class`. It holds no image data. It is not published here because it lists local file paths and photo file names. The per-folder `n_files` counts every file in the folder, so it can be higher than the number of photos the harness scored (LA: 248 files listed, 245 photos in the catalog run).

`model_constants.json` holds the model parameters embedded in the release file (classes in sklearn order, feature names, imputer, scaler and coefficients), for reference.

This folder (`release/build0.02/`) contains:

- `dm_detector_build0_02.py`
- `dm_detector_build0_02.md`
- `requirements.txt` and `REQUIREMENTS.md`
- `metrics_noadain.json`
- `model_constants.json`
