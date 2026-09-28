# dm_roi: find the Data Matrix ROI in a photo

`dm_roi.py` finds square Data Matrix (ECC200) codes in a photo and returns their regions of interest (ROIs). For every code it gives a tight, slightly tilted square (`polygon`), the enclosing axis-aligned `bbox`, the coarse `rotation`, the residual `tilt_deg`, a `score`, the `polarity`, and an estimated **module size in pixels per module**, measured in ORIGINAL full-resolution pixels. It uses classical OpenCV and NumPy only. It runs on Python 3.9 or newer. The project `.venv` now uses Python 3.14.7 with opencv-python 5.0.0.93, numpy 2.5.3, pandas 3.0.6, Pillow 12.3.0, pillow-heif 1.8.0, notebook 7.6.3 and ipykernel 7.3.0. The code avoids deprecated NumPy APIs and also runs on the earlier 3.9 set (numpy 2.0.2, Pillow 11.3.0, pillow-heif 1.1.1). Decoding is out of scope.

A stage-by-stage description of the algorithm (loading, preprocessing, candidate regions, location and straightening including the dotted-code fallback, cropping, pixels-per-module, scoring, rotation and tilt) is in [DMDetectorAlgorithm.md](DMDetectorAlgorithm.md).

For everyday use, import **`dm_detector`** (next to `dm_roi.py`). It's a small public API that wraps `dm_roi`, so detection results are identical. See [dm_detector: public API](#dm_detector-public-api) below.

## Assumptions (by design)

* Codes are **square**. Rectangular DM variants aren't searched for.
* Coarse orientation comes in **90° steps**: 0, 90, 180 or 270. It is read from the L-shaped finder when it can be verified, otherwise it's `None`.
* Around each step the code may be tilted by up to about **±10°** (`max_tilt_deg=10`, configurable), e.g. 80–100° or 170–190°. There is no search at arbitrary angles. Candidates whose edges point into the band between 10° and 80° (mod 90) are rejected.

## Command line

Run it from the project folder. Quote paths if they contain spaces.

```bash
cd "/Users/matthiasvoigt/work/projects/DM_detector"
source .venv/bin/activate

python dm_roi.py images/IMG_0066.HEIC --out out/IMG_0066_roi.png            # best ROI + annotated PNG
python dm_roi.py images/IMG_0066.HEIC --crop out/IMG_0066_crop.png           # + deskewed crop with px/mod text
python dm_roi.py images/IMG_0050.HEIC --all --json                           # all candidates as JSON
python dm_roi.py --batch images/                                             # CSV/JSON + thumbnails in out/batch/
```

* It prints polygon, bbox, rotation, tilt, angle, score, polarity and `module_px` (px/module, full resolution), plus the grid size N when known.
* By default the annotated PNG goes to `out/<name>_roi.png` next to `dm_roi.py`. The script refuses to write into the input image folder, so `images/` is never modified.
* `--batch` writes `out/batch/summary.csv`, `out/batch/summary.json` (with a `module_px` column) and annotated thumbnails.
* Options: `--max-results N`, `--max-tilt DEG`, `--min-score S`.

## dm_detector: public API

```python
import dm_detector as d

img = d.load_photo("images/IMG_0066.HEIC")   # BGR uint8, EXIF-rotated
results = d.detect(img)                        # ranked list of DMRoi, best first (DMConfig overrides as kwargs)
top = d.best(img)                              # best DMRoi or None
crop = d.crop(img, top, annotate=True)         # deskewed, padded crop with the "px/mod" text
df = d.results_table("images", fields=["image", "score", "rotation", "tilt_deg",
                                        "module_px", "grid_n", "dotted_fallback"])
df1 = d.rois_table(results, fields=d.PRESENTATION_FIELDS, image="IMG_0066.HEIC")
```

Per result: `score`, `rotation`, `tilt_deg`, `angle`, `module_px` (pixels per module, full resolution), `module_px_xy`, `grid_n`, `polarity`, `side_px`, `module_conf`, `module_method`, and three booleans:

* `dotted_located`: the ROI was located by the dotted-code fallback (`details['dotted'] > 0`).
* `dotted_pitch`: the module size came from the dot-pitch estimate (`module_method == 'dot-pitch'`).
* `dotted_fallback`: either of the two.

`d.ROI_OUTPUT_FIELDS` and `fields=` are re-exported, so you can pick table columns, including the three dotted flags. `d.draw(img, results)` and `d.module_size(img, roi)` are also available.

## Python API (dm_roi)

```python
from dm_roi import load_image, find_dm_rois, find_dm_roi, crop_roi, draw_rois, estimate_module_size

img = load_image("images/IMG_0066.HEIC")   # BGR uint8, EXIF-rotated (HEIC via pillow-heif; JPEG/PNG/ndarray also fine)
rois = find_dm_rois(img, max_results=5)     # ranked list of DMRoi, best first
best = find_dm_roi(img)                     # best one or None
crop = crop_roi(img, best, pad=0.2, deskew=True, annotate=True)   # upright crop with "px/mod" overlay
vis = draw_rois(img, rois)                  # annotated, downscaled copy
est = estimate_module_size(img, best)       # dict: module_px, module_px_xy, grid_n, module_conf, module_method
```

`DMRoi` fields:

| Field | Meaning |
|---|---|
| `polygon` | 4×2 corners TL, TR, BR, BL |
| `bbox` | (x, y, w, h) |
| `rotation` | 0/90/180/270 or `None` |
| `tilt_deg` | residual tilt; positive is clockwise on screen |
| `angle` | rotation + tilt |
| `score` | 0–1 |
| `polarity` | `dark_on_light` / `light_on_dark` |
| `side_px` | side length in pixels |
| `module_px`, `module_px_xy` | module size in full-resolution pixels (overall, and x/y) |
| `grid_n` | modules per side, when confident |
| `module_conf`, `module_method` | confidence and method of the module estimate |
| `dotted_fallback`, `dotted_located`, `dotted_pitch` | dotted-code fallback flags (see above) |
| `details` | sub-scores |

All parameters are in the `DMConfig` dataclass. You can override any field as a keyword argument, e.g. `find_dm_rois(img, max_tilt_deg=8, min_score=0.6)`.

### Choosing output fields (tables)

`ROI_OUTPUT_FIELDS` lists every available column: `image`, `rank`, `bbox`, `polygon`, `center`, `rotation`, `tilt_deg`, `angle`, `score`, `polarity`, `side_px`, `module_px`, `module_px_xy`, `grid_n`, `module_conf`, `module_method`, `dotted_fallback`, `dotted_located`, `dotted_pitch` and `details`. Any `details_<key>` also works, e.g. `details_finder` or `details_timing`. `DEFAULT_TABLE_FIELDS` is the default table set, used with `fields=None`.

```python
from dm_roi import rois_to_dataframe, batch_results_to_dataframe, ROI_OUTPUT_FIELDS

row = best.to_row(["tilt_deg", "angle", "score", "module_px"])            # dict
df  = rois_to_dataframe(rois, fields=["score", "rotation", "tilt_deg", "module_px"], image="IMG_0066.HEIC")
df_all = batch_results_to_dataframe("images", fields=["image", "rank", "score", "angle", "module_px", "grid_n"])
```

The table helpers return a pandas DataFrame, which needs pandas (installed in the project `.venv`). Without pandas they return a `list[dict]`. On the command line, `--fields score,rotation,tilt_deg,module_px` selects the fields for `--json` and for the `--batch` CSV.

## Notebooks

* **`results_presentation.ipynb`** imports `dm_detector` and contains no detector code. It runs over `images/` and shows a pandas table with image, score, rotation, tilt, pixels per module, grid size and `dotted_fallback`. Add columns in the `FIELDS` / `EXTRA_FIELDS` cell. Below the table is a grid of cropped codes with the same px/mod overlay. The table is also saved as `out/results_presentation.csv`.
* **`dm_roi_overview.ipynb`** is the visual overview, described below.

Both are delivered already executed.

### Visual overview notebook

`dm_roi_overview.ipynb` imports `dm_roi.py` from this folder and runs over all photos in `images/`. It builds one table row per image:

1. a thumbnail with the ROI outlines;
2. the primary crop (padded, deskewed, px/mod text on the crop);
3. up to 3 more candidate crops.

Each crop's caption shows score · rotation · tilt · px/mod. The same table is saved as `dm_roi_overview.html`, which opens in any browser. The notebook is delivered already executed.

To re-run either notebook, start Jupyter from the venv so the notebook uses its Python:

```bash
cd "/Users/matthiasvoigt/work/projects/DM_detector"
source .venv/bin/activate
jupyter notebook          # then open dm_roi_overview.ipynb or results_presentation.ipynb
```

The kernel "DM Detector" (the project `.venv`) is already registered. Both notebooks are set to open with it.

## OpenCV 5.0 and Data Matrix

`cv2.barcode.BarcodeDetector` handles 1D barcodes only (EAN/UPC...), and `cv2.QRCodeDetector` / `cv2.QRCodeDetectorAruco` handle QR codes only. OpenCV 5.0.0 has no Data Matrix detector. `dm_roi` uses the 1D barcode detector only to rank down candidates that sit on a linear barcode.
