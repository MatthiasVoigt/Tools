# DM detector: algorithm, stage by stage

This document describes the detection flow in `dm_roi.py` in processing order, including the functions, OpenCV calls, parameter defaults and score weights. `dm_detector.py` is a thin public wrapper around it (see the last section), so everything here applies to both modules.

**Design assumptions**

* Codes are square Data Matrix (ECC200) symbols.
* The coarse rotation comes in 90° steps (0/90/180/270), with at most ±`max_tilt_deg` (default 10°) of residual tilt.
* There is no rotation-invariant search.
* All output geometry is in ORIGINAL full-resolution pixels, after EXIF orientation is applied.
* Decoding is out of scope.

All tunables live in the frozen dataclass `DMConfig`. Any field can be overridden as a keyword, e.g. `find_dm_rois(img, max_tilt_deg=8)`, or a whole `DMConfig` can be passed as `config=` (keywords then override its fields). Unknown keywords raise `TypeError` (`_make_config`).

| `DMConfig` field | Default | Used in stage |
|---|---|---|
| `work_long_side` | 1600 | 2 |
| `max_tilt_deg` | 10.0 | 4 |
| `min_side_frac` / `max_side_frac` | 0.025 / 0.7 | `min_side_frac`: 3, 4, 7; `max_side_frac`: 3 only |
| `min_squareness` | 0.78 | 4, 7 |
| `texture_windows` | (0.01, 0.022, 0.035) × long side | 3 |
| `texture_thresholds` | (0.2, 0.35, 0.5) × p99.5 | 3 |
| `min_balance` | 0.35 | 3 |
| `binary_close_fracs` | (0.004, 0.008, 0.014) × long side | 3 |
| `refine_side_px` | 360 | 4 |
| `max_proposals` / `max_seeds` | 60 / 30 | 3 |
| `min_score` | 0.55 | 7 |
| `nms_overlap` | 0.4 | 7 |
| `use_barcode_suppression` | True | 2 (runs the barcode detector), 7 (uses its boxes) |
| `estimate_modules` | True | 6 |
| `module_max_side_px` | 1200 | 6 |

(`refine_close_fracs` = (1/40, 1/24, 1/14) is also defined, but nothing in the code reads it. The refinement derives its closing sizes from the pitch estimate (`1.3·p`, `2.2·p`) plus a hard-coded `side/14`, as described in stage 4a.)

---

## 1. Photo loading: `load_image(src, max_long_side=None)`

* **Input types:** a path (`str` or `os.PathLike`) or a NumPy array. A path that is not an existing file raises `FileNotFoundError`.
* **HEIC/HEIF/AVIF:** `pillow_heif.register_heif_opener()` is called lazily, then the file is opened with `PIL.Image.open`.
* **Orientation:** `ImageOps.exif_transpose` applies the EXIF orientation, so iPhone portrait shots come out upright.
* **Paths with `max_long_side`:** after the EXIF transpose, PIL downsizes the image with `Image.thumbnail(..., LANCZOS)` (not `cv2.resize`), before the colour conversion.
* **Colour:** the image is converted to RGB and then to BGR `uint8` with `cv2.cvtColor(..., COLOR_RGB2BGR)`.
* **Arrays:** they go through `_to_bgr_u8`, which accepts gray, BGR or BGRA and any dtype (floats with max ≤ 1 are scaled ×255, then values are clipped to 0–255). They are downsized with `cv2.resize(INTER_AREA)` only if `max_long_side` is given.
* **Detection input:** `find_dm_rois` always loads the image at full resolution (`max_long_side=None`). Test photos are 3024×4032 or 4284×5712.

## 2. Preprocessing (in `find_dm_rois`)

1. **Grayscale:** `_gray(img)` uses `cv2.cvtColor(BGR2GRAY)`. The full-resolution gray image `gray` is kept for refinement (stage 4) and module estimation (stage 6). Cropping (stage 5) does not use it: `crop_roi` reloads the caller's image and warps the colour (BGR) image.
2. **Working copy:** `gw = cv2.resize(gray, fx=s, fy=s, INTER_AREA)` with `s = min(1, work_long_side / long side)`, i.e. 1600 px on the long side (no resize if the photo is already ≤ 1600 px).
3. **1D barcode boxes:** with `use_barcode_suppression`, `_barcode_boxes(gw)` runs `cv2.barcode.BarcodeDetector().detectMulti(gw)`. In the code this call comes right after `_propose` (stage 3); the order doesn't matter, because these boxes are used only in stage 7. Each result quad is turned into a box with `cv2.boundingRect`, and only elongated boxes (aspect > 1.2) are kept. If `cv2.barcode` is not available, if `detectMulti` raises `cv2.error`, or if nothing is found, it silently returns `[]` and no barcode down-ranking happens.

## 3. Candidate region detection: `_propose(gw, cfg)`

This stage works on the 1600 px copy and returns axis-aligned boxes `(x, y, w, h)` from three sources. `L` is the long side of `gw`. Boxes whose long side is outside `[min_side_frac·L, 1.2·max_side_frac·L]` are dropped. So are boxes with a side < 3 px, and boxes whose aspect (short/long side) is below 0.45 for texture-contour proposals (3a) or below 0.6 for seed (3b) and binary (3c) proposals.

### 3a. Balanced 2D texture map

* **Gradients:** `cv2.GaussianBlur(σ=1)`, then `|cv2.Sobel(dx)|` and `|cv2.Sobel(dy)|` (ksize 3).
* **Local energies:** for each window `k = odd(wf·L)`, with `wf` in `texture_windows` (≈ 17 / 35 / 57 px), `cv2.blur` gives the local energies `Ex` and `Ey`.
* **Balance:** `bal = min(Ex,Ey) / max(Ex,Ey)`. It is about 1 for DM modules and about 0 for 1D barcodes and long strokes.
* **Texture map:** `m = ½(Ex+Ey) · clip((bal − ½·min_balance) / min_balance, 0, 1)`.
* **Contour proposals:** for each map and each threshold `t` in `texture_thresholds`:
  * binarise `m > t · p99.5(m)`;
  * apply `cv2.morphologyEx(MORPH_OPEN, k×k rect)`;
  * take `cv2.findContours(RETR_EXTERNAL)`, then `cv2.boundingRect`.
* **Proposal score:** each proposal, from all three sources, gets `mean_texture · (0.4 + 0.6·squareness) · sqrt(side / L)`.
  * `mean_texture` comes from a `cv2.integral` of the finest map normalised by its p99.5.
  * The `sqrt(side)` factor favours the whole code over a patch of it.

### 3b. Multi-scale square seeds

* **Peaks:** local maxima of the coarsest texture map (`cv2.dilate` with an `odd(max(5, 0.6·min_side))` kernel, value > 0.3·p99).
* **Seeds:** the top `max_seeds` = 30 peaks are used.
* **Windows:** each peak gets 12 square windows with sides geometrically spaced from `1.1·min_side` to `0.9·max_side`.
* **Clipping:** windows are clipped to the image, and a clipped window must still have aspect ≥ 0.6.
* **Kept per peak:** the 2 best-scoring windows. This covers large, clean codes that the contour maps fragment or merge with nearby text.

### 3c. Binary ink components (both polarities)

* **Threshold:** `cv2.adaptiveThreshold(ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, block = odd(0.04·L), C = 10)`, on `gw` and on `255 − gw`.
* **Closing:** `cv2.morphologyEx(MORPH_CLOSE)` with rect kernels `binary_close_fracs · L` (≈ 6 / 13 / 22 px).
* **Contours:** `cv2.findContours(RETR_LIST)`. `RETR_LIST` finds codes printed inside frames or boxes, not just the outer contours.
* **Filters:** contour area ≥ 0.35·box area, squareness ≥ 0.6.

### 3d. De-duplication

* **Rule:** proposals are sorted by score and greedily de-duplicated with IoU > 0.5.
* **Quotas:** contour and binary proposals get a quota of `max_proposals` = 60. Seed proposals get a separate quota of `2·max_seeds` = 60. They are de-duplicated against the boxes already kept from the contour and binary pass, not only against each other.
* **Result:** up to about 120 boxes. They are scaled back to full resolution (÷ `s`) for stage 4.

## 4. Code location and straightening: `_refine(gray, rect, cfg)` → `_fit_and_verify(...)`

Each proposal is refined on a full-resolution crop.

### 4a. Crop and binarise (`_refine`)

1. **Crop:** the box is padded by `0.5·side + 4` px, cut from `gray`, and downscaled so the candidate side is ≤ `refine_side_px` = 360 px (`cv2.resize INTER_AREA`, factor `r`).
2. **Otsu threshold:** `cv2.GaussianBlur(σ=0.8)`, then `cv2.threshold(THRESH_BINARY + THRESH_OTSU)` computed on the proposal core only.
3. **Polarities:** both are tried (`dark_on_light`: ink = blur < t; `light_on_dark`: ink = blur > t).
4. **Pitch guess:** `_pitch_guess` uses the autocorrelation of the edge projection profiles via `_autocorr_period`, clipped to `[2, side/6]`.
5. **Base mask:** `cv2.morphologyEx(MORPH_CLOSE, rect 0.6·p)` merges dots but keeps module gaps.
6. **Grouping:** the base mask is closed again with rect kernels `{1.3·p, 2.2·p, side/14}`. `cv2.connectedComponentsWithStats(connectivity=8)` then groups the code into one component.
7. **Components tried:** the 6 components with the largest overlap with the proposal core. Components touching ≥ 2 crop borders, smaller than `0.7·min_side_frac`, or with aspect < 0.55 are skipped.
8. **Fit:** every remaining component goes to `_fit_and_verify`. The best score wins.

### 4b. Dotted-code fallback (a step inside `_refine`)

This step runs after each polarity pass if no candidate has been found yet or the best score is below 0.7. It targets inkjet, laser and dot-peen codes, whose isolated dots don't close into one solid component.

1. **Remove panels:** for both polarities, components larger than 5 % of the crop (printed panels) are removed from the ink mask.
2. **Cluster the dots:** `_dotted_cluster(ink, core_mask, pitch_hint = side/22)` runs:
   * `cv2.connectedComponentsWithStats`, keeping dot-sized blobs (0.2–6× the median area) centred in the core. At least 30 dots are required.
   * **Pitch:** the median nearest-neighbour distance between dot centres (up to 400 sampled), averaged with the hint.
   * **Density map:** a dot-centre density map, box-blurred with `cv2.blur` (k = 4·pitch), thresholded at its 70th percentile, and closed with `MORPH_CLOSE`.
   * **Cluster box:** the `cv2.findContours` blob with side ≥ 8·pitch and squareness ≥ 0.75 that contains the most dots (score = dots × squareness).
3. **Fit:** the box is padded by 5 %. Its dots are closed with an elliptical kernel (`MORPH_ELLIPSE`, 0.7·p with `p = box side / 22`) so the L finder reads solid. The box is then passed to `_fit_and_verify` as one solid component.
4. **Flag:** a result from this step gets `dotted = 1`. It is reported as `details['dotted']` and drives `DMRoi.dotted_located` / `dotted_fallback`.

*Known code-side nit (behaviour unaffected):* the fallback's inner loop reuses the names `polarity`, `ink`, `p` and `base` from the enclosing polarity loop. Because it sits inside that loop, it can run twice with identical inputs (after each pass), which only repeats work.

### 4c. Tilt and tight square (`_fit_and_verify`)

1. **Residual tilt:** `_orientation4` uses Sobel gradients of the blurred base mask inside the dilated component. It computes the 4-fold gradient angle `φ = ¼·atan2(Σ m²·sin 4θ, Σ m²·cos 4θ)`, which gives the edge orientation modulo 90°.
   * If the 4-fold coherence is < 0.12, the angle of `cv2.minAreaRect` / `cv2.boxPoints` is used instead.
2. **Tilt constraint:** if `|tilt| > max_tilt_deg`, the candidate is rejected. This is where the "90° steps ± 10°" assumption is enforced.
3. **Trim appendages:** attached text or barcodes are removed in the code-aligned frame, up to 2 passes:
   * take column and row pixel histograms, smoothed with a 3-tap filter;
   * keep the longest run of bins > 0.4 × the 80th-percentile height.
4. **Tight box:** the 0.1–99.9 percentile extents give `w`, `h` and the centre. Reject if squareness < `min_squareness` (0.78) or fill < 0.45.
5. **Upright copy:** `_warp_upright` (`cv2.getRotationMatrix2D` + `cv2.warpAffine`, `BORDER_REPLICATE`) at ≤ 220 px side. Three masks are made from it:
   * the upright gray image and the Otsu ink;
   * the upright base mask `upb`;
   * `upb_dot`, which is `upb` closed with an ellipse of 0.9·pitch, for dotted modules.

## 5. Crop production: `crop_roi(img, roi, pad=0.2, deskew=True, normalize_rotation=True, out_size=None, annotate=False)`

* **Crop:** `crop_roi` first reloads the input with `load_image` (a path is read again; an array is converted to BGR `uint8` if needed). With `deskew`, `_warp_upright` (`cv2.warpAffine`, `INTER_CUBIC`) cuts a square of side `max(w, h)`, plus `pad` × side on each side, from the full-resolution colour (BGR) image, undoing the tilt.
* **Rotation:** only with `deskew`: with `normalize_rotation` and a known, non-zero `rotation`, `np.rot90(k = rotation/90)` turns the code so the L finder is at left + bottom. Without `deskew`, `normalize_rotation` is ignored.
* **Without deskew:** you get a padded axis-aligned bbox crop (`pad` × the bbox long side, clipped to the image).
* **Resize:** `out_size` resizes the crop (`INTER_AREA` down, `INTER_NEAREST` up).
* **Text overlay:** with `annotate=True`, `annotate_crop` writes `"12.4 px/mod 18x18"` in `cv2.FONT_HERSHEY_SIMPLEX` (`"px/mod n/a"` if `module_px` is None; the `NxN` part only when `grid_n` is known):
  * The text goes on a semi-opaque dark box in the top-left padding.
  * If the padding is thinner than the text box, a 40-gray bar is added above the crop instead, so the code is never covered. The returned crop is then taller by the bar height.
  * Crops made without `deskew` always get the bar, because `crop_roi` then passes `pad=None` and the padding counts as 0.
* **Outline drawing:** `draw_rois` returns a copy downscaled to `max_long_side` = 1600 px by default (`INTER_AREA`; pass `None` to keep full resolution), so its pixel coordinates are not full resolution. It draws the outlines with `cv2.polylines` (best result green, others orange), a red dot at the finder corner only when `rotation` is known, and `#rank score px/m` labels (`px/m` only when `module_px` is set).

## 6. Pixels-per-module estimation: `estimate_module_size(img, roi)`

This runs for every returned ROI when `estimate_modules=True`, via `_apply_module_estimate`. Its errors never break detection: an exception is swallowed and the ROI keeps its defaults (`module_px`/`grid_n` None, `module_conf` 0, `module_method` `''`, not `'none'`). Output is in ORIGINAL full-resolution pixels.

1. **Deskew:** `_warp_upright(..., INTER_CUBIC)` at scale `sc = min(1, 1200/side)`, raised to at most 4× so small codes reach ≥ 240 px.
2. **Binarise:** Gaussian blur (σ = 0.4 % of the side), then Otsu. Ink is chosen by the ROI polarity.
3. **Closing sizes:** for each closing size `kf` in (0.01, 0.02, 0.035) × side (`MORPH_ELLIPSE`, `2k+1`):
   * **Autocorrelation:** `_autocorr_period` of the gradient projection profiles along x and y. This is FFT autocorrelation; it takes the smallest lag with a peak ≥ 0.6 × the strongest, refined parabolically, searched in `[max(2, side/150, 1.6k), side/8]`. It gives `n_a = side / period`.
   * **Timing borders:** `_border_analysis` of the closed mask. The run counts on the two timing sides with timing > 0.2 are averaged into `n_t`. The timing sides are those not in the L of the ROI's rotation; if `roi.rotation` is None, the rotation that `_border_analysis` finds on this closed mask is used instead.
   * **Method and confidence:**
     * `n_t` and `n_a` agree within max(1.5, 10 %): `grid_n = _snap_n(mean, 1.5)` to the nearest value in `VALID_SQUARE_SIZES` (10…144). Method `timing+autocorr`, confidence 0.95 if both timing sides agree (both have timing > 0.2 and their counts differ by ≤ max(1.5, 8 % of `n_t`)), else 0.8.
     * Otherwise, if both timing sides agree (same test): `grid_n = _snap_n(n_t, 1.2)`, method `timing`, confidence 0.6.
     * If snapping fails, `grid_n` stays None and the next rules apply.
     * If `grid_n` is found, `module_px_xy = (w/grid_n, h/grid_n)`.
     * If there is no grid but there is a period: `module_px = period / sc`, method `autocorr`, confidence `0.3 + 0.5·strength`, capped at 0.6 and halved if `n_a < 8`.
     * With neither a grid nor a period, this closing size gives no result.
   * **Selection:** the closing size with the highest confidence wins.
4. **Early return:** if no closing size gave any result (no timing and no autocorr period), the function returns at once with `module_px` None, `grid_n` None, confidence 0 and method `none`. `_dot_pitch` is not tried in that case.
5. **Dot-pitch fallback (`_dot_pitch`):** if there is a winner but it has no `grid_n` (i.e. an `autocorr` result), the unclosed ink is checked for dots:
   * **Test:** ≥ 40 blobs (≥ 41 labels from `cv2.connectedComponentsWithStats`, counting the background); median blob area > 3 px and ≤ (min side / 8)²; ≥ 40 and ≥ 50 % of the blobs within 0.3–3× the median area; median bbox aspect ≥ 0.6.
   * **Pitch:** the median nearest-neighbour distance of up to 600 centres.
   * **Result**, with `n_d = side/pitch`:
     * `n_d` in 8–150 and snaps to a valid size within 0.6: method `dot-pitch`, that `grid_n`, `module_px_xy = (w/grid_n, h/grid_n)`, confidence 0.7.
     * `n_d` in 8–150 but no snap: method `dot-pitch` with the raw pitch (`pitch / sc`), `grid_n` None, confidence 0.5.
     * `n_d` outside 8–150, or the ink is not dotted: the `autocorr` result from step 3 is kept.
     * A `dot-pitch` result replaces the `autocorr` result even if the autocorr confidence was higher (it can reach 0.6).
   * A `dot-pitch` result sets `DMRoi.dotted_pitch` / `dotted_fallback`.
6. **Output fields:** `module_px` (mean of x/y), `module_px_xy`, `grid_n` (or None), `module_conf`, and `module_method` (`timing+autocorr` | `timing` | `autocorr` | `dot-pitch` | `none`). Except on the `none` early return, the dict also has `n_timing` and `n_autocorr` (the winner's `n_t` and `n_a`); these are not copied to the `DMRoi`.

*Known code-side nit:* the inline comment on `DMRoi.module_method` lists only `timing`, `autocorr` and `timing+autocorr`.

## 7. Scoring, rotation and tilt

### 7a. Finder / timing: `_border_analysis(mask)`

* **Profiles:** for each side, 2-px wide profiles at offsets 0…6 % of the side are thresholded at 0.5 (ends trimmed by 1.5 %). Gaps < side/70 are filled and specks < side/120 are removed.
* **Solidity:** the best filled fraction.
* **Timing regularity:** needs ≥ 7 runs. It is `(1 − cv/0.4) · (1 − |fill − 0.5|/0.3)` of the inner run lengths, and it also yields a module count `n`.
* **L hypotheses:** the 4 hypotheses (left+bottom → 0°, top+left → 90°, right+top → 180°, bottom+right → 270°, clockwise) are scored:
  * `finder = clip((min solidity of the two L sides − 0.72)/0.22, 0, 1)`
  * `timing = 0.7·(agree·min) + 0.3·mean` of the two opposite sides' regularity. `agree` = 1 if both sides have a module count > 0 and the counts differ by ≤ max(1.5, 12 % of the larger count), else 0.6, so the 0.6 penalty scales only the `min` term, before the blend.
  * The blended timing is then × `max(0, 1 − (max solidity of the opposite sides − 0.85)/0.15)`, a penalty if an opposite side is also solid (> 0.85).
* **Result:** the best hypothesis gives `finder`, `timing`, `rotation`, the module count (mean of the two opposite sides' counts if they agree, else the larger) and the `margin` to the runner-up.
* **Masks tried:** `_fit_and_verify` runs this on `upb_dot` first, then on `upb` and the raw ink, unless finder + timing already exceeds 1.7. The best sum is kept.

### 7b. Candidate score (`_fit_and_verify`)

```
score = 0.30·finder + 0.34·timing + 0.12·density + 0.10·balance + 0.06·ink + 0.08·squareness
```

* **Sub-scores:**
  * `density = clip((transitions per row/col − 2)/6, 0, 1)`
  * `balance = clip((axis balance − 0.4)/0.4, 0, 1)`
  * `ink = clip(1 − |ink fraction − 0.5|/0.35, 0, 1)`
  * `squareness = clip((sq − 0.78)/0.22, 0, 1)`
* **Adjustments:**
  * timing < 0.15 and finder < 0.5 → score × 0.7.
  * Dotted boost: finder ≥ 0.9, density ≥ 0.4 and timing < 0.25 → `score = max(score, 0.55·finder + 0.25·density + 0.2·squareness)`.
  * Ink/background contrast < 25 grey levels → score × max(0.3, contrast/25).

### 7c. Barcode down-ranking (`find_dm_rois`)

* The tilted polygon is built with `_polygon_from`, and `bbox = cv2.boundingRect(polygon)`.
* Candidates smaller than 0.8·`min_side_frac` of the long side are dropped.
* If a candidate overlaps a `cv2.barcode` box by > 0.6 (intersection / smaller box) and its finder is < 0.8, score × 0.4.

### 7d. Timing-first NMS, soft gate and reject rules

Candidates are sorted by (timing ≥ 0.35 first, then score). This way a tight code with a verified timing pattern wins over a label or quiet-zone box that contains it. Each candidate then goes through these checks in order:

1. **Soft gate:**
   * timing ≥ 0.25: needs score ≥ `min_score` (0.55).
   * Dotted exception (finder ≥ 0.9, density ≥ 0.4, squareness ≥ 0.88, barcode overlap < 0.3, fill ≥ 0.45): needs ≥ `min_score` (same as the timing ≥ 0.25 case). The test uses these features only; it does not look at the `dotted` flag from 4b.
   * Otherwise: needs ≥ max(`min_score`, 0.72).
2. **Weak-timing rejects:** these apply when timing < 0.25.
   * **1D barcode / halftone:** reject if axis balance < 0.5, barcode overlap > 0.5, or contrast < 75.
   * **Solid panel:** reject if the side > 0.4 × the long side. For non-dotted candidates, also reject if ink fraction < 0.2 or > 0.8, or if fill > 0.97 with density < 0.5.
3. **NMS:** drop the candidate if its bbox overlaps an already kept one by > `nms_overlap` (0.4, intersection / smaller box). Stop at `max_results`.

Kept results are re-sorted by score (capped at 1.0) and returned as `DMRoi`s.

### 7e. Rotation and tilt in the result

* `rotation` is the L-hypothesis rotation, but only if finder ≥ 0.5, timing ≥ 0.25 and margin > 0.15. Otherwise it is `None`.
* `tilt_deg` is the residual tilt from 4c, in [−`max_tilt_deg`, +`max_tilt_deg`] (default [−10, 10]); positive means clockwise on screen.
* `angle = rotation + tilt_deg`, or just `tilt_deg` when rotation is unknown.
* `polygon` holds the corners TL, TR, BR, BL of the tilted tight box (`w` × `h`; not forced square, only squareness ≥ `min_squareness`).
* `side_px = (w + h)/2`.
* `details` has `finder`, `timing` and `dotted` (with `debug=True`, all sub-scores and timings).

---

## Entry points in `dm_detector.py`

`dm_detector.py` imports `dm_roi` and only renames and forwards calls. It contains no algorithm code, so its results are identical to `dm_roi.find_dm_rois`.

| Function | Forwards to | Returns |
|---|---|---|
| `load_photo(path, max_long_side=None)` | `load_image` (stage 1) | BGR `uint8` array |
| `detect(image_or_path, *, max_results=5, **params)` | `find_dm_rois` (stages 2–7); `max_results` is keyword-only; `params` are `DMConfig` fields, and `debug=` / `config=` also pass through; unknown keys raise `TypeError` | ranked `list[DMRoi]` |
| `best(image_or_path, **params)` | `detect(...)[0]`, with `detect`'s default `max_results=5` (see below) | `DMRoi` or `None` |
| `crop(image, roi, annotate=True, pad=0.2, out_size=None, normalize_rotation=True)` | `crop_roi(..., deskew=True, annotate=annotate)`, which calls `annotate_crop` (stage 5) | deskewed colour crop with px/mod text (taller by the bar if the padding is too thin) |
| `draw(image, rois, max_long_side=1600)` | `draw_rois` | annotated copy, downscaled to 1600 px on the long side by default (`None` = full resolution) |
| `module_size(image, roi)` | `estimate_module_size` (stage 6) with the default `DMConfig`; overrides used in `detect` (e.g. `module_max_side_px`) are not applied | dict, including `n_timing` / `n_autocorr` |
| `rois_table(rois, fields=None, image=None)` | `rois_to_dataframe` | DataFrame for one photo (`rank` = 1…n in list order); `list[dict]` if pandas is not installed |
| `results_table(images, fields=None, *, include_empty=True, max_results=5, **params)` | `batch_results_to_dataframe`; `params` are `DMConfig` fields plus `debug=` / `config=`, and `progress=True` prints one line per image | DataFrame for a folder (image files sorted by name) or list, one row per code; with `include_empty`, a photo without a code gets one row with only `image` filled; `list[dict]` if pandas is not installed |

**`best()` vs `dm_roi.find_dm_roi()`:** they can return different ROIs. `find_dm_roi` sets `max_results=1`, so the NMS loop in 7d stops at the first candidate that passes, in timing-first order (timing ≥ 0.35 first, then score). `best()` calls `detect` with `max_results=5`, keeps up to five, re-sorts them by score and returns the first. So if a candidate with timing < 0.35 scores higher than the first timing-verified one and doesn't overlap it (overlap ≤ `nms_overlap`), `best()` returns it, while `find_dm_roi` returns the timing-verified one. `best(img, max_results=1)` behaves like `find_dm_roi`.

**Result attributes** (`DMRoi`): `score`, `rotation`, `tilt_deg`, `angle`, `module_px`, `module_px_xy`, `grid_n`, `polarity`, `side_px`, `module_conf`, `module_method`, `polygon`, `bbox`, `details`, the property `center` (mean of the polygon corners, `(x, y)`), and three derived booleans:

* `dotted_located`: `details['dotted'] > 0`, i.e. the ROI came from the dotted-code fallback in 4b.
* `dotted_pitch`: `module_method == 'dot-pitch'`, i.e. the module size came from `_dot_pitch` in stage 6.
* `dotted_fallback`: `dotted_located or dotted_pitch`.

**Also re-exported:** `ROI_OUTPUT_FIELDS` (every named column, including `center` and the three flags), `DEFAULT_TABLE_FIELDS` (the default columns; no dotted flags), `PRESENTATION_FIELDS` (image, rank, score, rotation, tilt_deg, module_px, grid_n, dotted_fallback), `VALID_SQUARE_SIZES`, `DMConfig`, `DMRoi` and `annotate_crop`.

**Field selection:** every table helper takes `fields=`: `None` = `DEFAULT_TABLE_FIELDS`, `"all"` = `ROI_OUTPUT_FIELDS`, or a comma-separated string or list of names. Besides the names in `ROI_OUTPUT_FIELDS`, any `details_<key>` (e.g. `details_finder`) selects one entry of `details` (None if missing). This is a naming pattern, not an entry in the tuple. Any other name raises `KeyError`.
