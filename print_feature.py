"""print_feature -- ordered print-characteristic vector for one Data Matrix code.

Implements Proposal B (section 9 of ``PrintTypeClusteringLiterature.md``): a
single extractor that turns a deskewed Data Matrix ROI (from ``dm_detector``)
into an ORDERED list of ``(property_name, float)`` tuples following the
119-row feature specification table of that section (same names, same order).
Undefined values are ``math.nan`` (never a silent zero), e.g. ``dot_*`` for
codes that are not dotted.

Pipeline (the shared front-end of section 9):

1. Luminance of the padded, deskewed crop (``dm_detector.crop``); the code
   square is centred in it (``code_pad`` = the crop padding fraction).
2. Resample so the module pitch is ``canonical_px_per_module`` (default 24).
3. Glare mask (near-saturated pixels), local ink / substrate levels by
   normalised convolution, a polarity-aware "markness" image ``N`` (ink = 1,
   substrate = 0, whatever the print polarity) and the binary mark ``N > 0.5``.
4. **ink** (mark eroded by 1/4 module), **substrate** (background eroded by
   1/4 module) and **edge** (the 1/2-module band between them) masks inside
   the code square + 1-module quiet zone, all glare-masked.
5. Module grid fit (comb fit of the edge-energy projections), module states,
   straight edge runs on grid lines, sub-pixel 50 % crossings per pixel row.
6. Print-direction vote (banding, structure tensor, edge-class contrast, dot
   elongation) -> ``print_axis``; lead/trail sign vote (blur, satellites,
   profile skew) -> edge classes ``leading / trailing / orth_left / orth_right``.
7. Every feature of the table, in order. Spatial quantities are in module
   units or computed at the canonical pitch.

Conventions (upright code frame of the crop, image y axis pointing down):

* ``print_axis`` 0 = code x, 1 = code y. The process direction is +axis when
  the lead/trail vote is positive. The *leading* edge of a mark is the edge
  whose outward normal points against the process direction, *trailing* along
  it; ``orth_left`` / ``orth_right`` are the edges on the left / right hand
  when looking along the process direction.
* Orientation features are in degrees relative to the print axis, wrapped to
  (-90, 90]; Gabor orientations are the wave-vector direction (0 deg responds to
  intensity variation ALONG the print axis).
* When ``print_polarity_confidence`` is ~0 the lead/trail sign is effectively
  arbitrary (it follows the sign of the weak vote); treat lead/trail
  asymmetry features of such codes as unordered.

Deviations from the specification (documented, not silent):

* ``edge_bias_leading`` / ``_trailing`` (and ``_orth_left`` / ``_orth_right``)
  are the per-axis *symmetric* bias: the grid phase has no reference other
  than the marks themselves, so a lead-vs-trail split of the bias is not
  identifiable (a common shift of all marks is indistinguishable from a grid
  shift). Both edges of an axis therefore carry the same value, and
  ``edge_bias_lead_trail_diff`` is **dropped** (always NaN).
* ``jpeg_blockiness`` is computed on the native, un-deskewed bbox crop with
  the 8x8 lattice aligned to the photo origin. The iPhone photos are HEIC
  (HEVC, variable block sizes), so this is only a weak compression indicator.
* The optional AdaIN block (rows 108-119) needs torch + torchvision (VGG-19
  ImageNet weights, downloaded once to the torch cache in the user's home);
  without them, or with ``include_adain=False``, those 12 values are NaN.
* The ink / edge / substrate masks partition the analysis region with a
  1/4-module erosion on each side (edge band = 1/2 module), so single-module
  marks keep an ink core of 1/2 module at 24 px/module.

Usage::

    import dm_detector as d
    import print_feature as pf

    img = d.load_photo("images/IMG_0066.HEIC")
    roi = d.best(img)
    feats = pf.features_for_roi(img, roi)          # [(name, value), ...] 119 items
    df = pf.features_table("images", limit=3)      # one row per detected code

Command line (writes ``out/features/print_features.csv`` and ``.html``)::

    python print_feature.py                        # all photos in images/
    python print_feature.py --limit 3 --best-only
    python print_feature.py --no-adain             # skip the torch/VGG block

Works on Python 3.9+ (tested 3.14) with numpy, opencv-python, scipy,
scikit-image and pandas; torch/torchvision only for the optional AdaIN block.
"""
from __future__ import annotations

import argparse
import base64
import html
import math
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from scipy import ndimage as ndi

import dm_detector as d

__all__ = [
    "FeatureTuple",
    "PrintFeatureConfig",
    "PreparedROI",
    "FEATURE_NAMES",
    "DROPPED_FEATURES",
    "TABLE_ID_COLUMNS",
    "feature_names",
    "features_to_dict",
    "prepare_roi",
    "extract_print_features",
    "features_for_roi",
    "features_table",
    "main",
]

FeatureTuple = Tuple[str, float]
EDGE_CLASSES: Tuple[str, ...] = ("leading", "trailing", "orth_left", "orth_right")
_ORIENTS: Tuple[str, ...] = ("000", "045", "090", "135")

#: Stable output order (section 9 feature specification table, rows 1-119).
FEATURE_NAMES: Tuple[str, ...] = tuple(
    ["capture_px_per_module", "resample_factor", "upsampled_flag", "module_pitch_anisotropy",
     "sharpness_score", "glare_fraction", "jpeg_blockiness", "bimodality_separability",
     "mark_polarity", "print_axis", "print_axis_confidence", "print_polarity_confidence"]
    + ["edge_raggedness_%s" % c for c in EDGE_CLASSES]
    + ["edge_blur_width_%s" % c for c in EDGE_CLASSES]
    + ["edge_bias_%s" % c for c in EDGE_CLASSES]
    + ["edge_profile_skew_%s" % c for c in EDGE_CLASSES]
    + ["edge_satellite_density_%s" % c for c in EDGE_CLASSES]
    + ["edge_raggedness_lead_trail_log2ratio", "edge_blur_lead_trail_log2ratio",
       "edge_bias_lead_trail_diff", "edge_satellite_lead_trail_log2ratio",
       "edge_raggedness_process_vs_orth_log2ratio", "edge_blur_process_vs_orth_log2ratio",
       "corner_rounding_radius", "st_coherence_ink", "st_orientation_ink", "st_coherence_edge"]
    + ["gabor_ink_s%d_%s" % (s, o) for s in (1, 2) for o in _ORIENTS]
    + ["gabor_ink_orientation_of_max", "gabor_ink_anisotropy",
       "gabor_substrate_orientation_of_max", "gabor_substrate_anisotropy",
       "banding_peak_process", "banding_peak_cross", "banding_period_process",
       "lbp_riu2_ink_entropy", "lbp_riu2_ink_uniform_fraction", "lbp_var_ink",
       "lbp_rv_ink_process_vs_cross", "clbp_m_ink_mean",
       "lbp_riu2_substrate_entropy", "lbp_riu2_substrate_uniform_fraction", "lbp_var_substrate",
       "lbp_rv_substrate_process_vs_cross", "lbp_riu2_edge_entropy",
       "ldp_edge_direction_entropy", "hog_contour_orientation_entropy"]
    + ["glcm_%s_%s" % (r, p) for r in ("ink", "substrate") for p in (
        "contrast_d1", "contrast_d2", "homogeneity_d1", "energy_d1", "correlation_d1",
        "correlation_d2", "contrast_along_vs_across_log2ratio")]
    + ["ink_density_mean", "ink_density_std", "ink_mottle", "ink_graininess",
       "ink_void_density", "ink_void_area_fraction", "ink_void_orientation", "ink_streak_score",
       "substrate_brightness", "substrate_grain_energy", "substrate_fiber_coherence",
       "substrate_fiber_orientation", "substrate_speck_density", "substrate_speck_size_median",
       "spatter_density_near_edge", "spatter_decay_length", "spatter_size_median",
       "dot_detected_flag", "dot_roundness_mean", "dot_roundness_std", "dot_diameter",
       "dot_pitch_regularity", "dot_merge_fraction", "dot_elongation_orientation"]
    + ["adain_%s_rp%d" % (r, i) for r in ("ink", "substrate", "edge") for i in range(4)]
)
assert len(FEATURE_NAMES) == 119, len(FEATURE_NAMES)

#: Features of the specification that are always NaN (see module docstring).
DROPPED_FEATURES: Tuple[str, ...] = ("edge_bias_lead_trail_diff",)

#: Identification / detector columns that precede the features in tables.
TABLE_ID_COLUMNS: Tuple[str, ...] = ("image", "roi_index", "score", "px_per_module", "rotation",
                                     "dotted_fallback")

_EPS = 1e-9

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PrintFeatureConfig:
    """Tunable parameters of the extractor (module units unless noted)."""

    #: Canonical pitch the ROI is resampled to (px per module).
    canonical_px_per_module: float = 24.0
    #: Padding fraction of the crop around the code (as in ``dm_detector.crop``).
    code_pad: float = 0.2
    #: Quiet-zone margin included in the analysis region (modules).
    quiet_zone_modules: float = 1.0
    #: Erosion radius that separates ink / edge / substrate (modules).
    region_erode_modules: float = 0.25
    #: Sigma of the local ink / substrate level maps (modules).
    level_sigma_modules: float = 2.0
    #: Pixels with max(B, G, R) >= this are glare (native 8-bit values).
    glare_level: int = 250
    #: Relative search range of the grid-pitch comb fit.
    pitch_search: float = 0.08
    #: Minimum run length (modules) of a straight edge used for raggedness.
    min_run_modules: int = 2
    #: Trim at each end of an edge segment (modules), avoids corners.
    edge_trim_modules: float = 0.25
    #: Satellite / spatter: blob area below this (module^2) is "small".
    satellite_max_area: float = 0.05
    #: Satellite distance window from the contour (modules).
    satellite_dist: Tuple[float, float] = (0.1, 1.0)
    #: Voids: holes in the mark smaller than this (module^2).
    void_max_area: float = 0.25
    #: Largest canonical crop side (px); bigger codes are processed at lower pitch.
    max_canonical_side: int = 2400
    #: Compute the optional AdaIN block (torch + torchvision VGG-19).
    include_adain: bool = False
    #: Fixed seed of the AdaIN random projection.
    adain_seed: int = 1234


@dataclass
class PreparedROI:
    """Canonical-pitch ROI with masks (shared by print_feature and the notebooks)."""

    bgr: np.ndarray                 #: resampled BGR crop (uint8)
    lum: np.ndarray                 #: luminance 0..255 (float32)
    markness: np.ndarray            #: N: ink = 1, substrate = 0 (float32)
    mark: np.ndarray                #: binary mark N > 0.5 (bool)
    ink: np.ndarray                 #: uniform-print interior mask (bool)
    substrate: np.ndarray           #: substrate interior mask (bool)
    edge: np.ndarray                #: edge band mask (bool)
    glare: np.ndarray               #: glare mask (bool)
    region: np.ndarray              #: analysis region (code + quiet zone, no glare)
    code_box: Tuple[float, float, float, float]  #: x0, y0, x1, y1 of the code square
    ppm: float                      #: pitch of the canonical crop (px / module)
    capture_ppm: float              #: native pitch (px / module)
    polarity: str                   #: "dark_on_light" or "light_on_dark"
    ink_level: float                #: median luminance of the ink
    substrate_level: float          #: median luminance of the substrate
    extra: Dict[str, Any] = field(default_factory=dict)


def feature_names() -> List[str]:
    """Stable column order (same as the names returned by :func:`extract_print_features`)."""
    return list(FEATURE_NAMES)


def features_to_dict(feats: Sequence[FeatureTuple]) -> Dict[str, float]:
    """``[(name, value), ...]`` -> ``{name: value}`` (order preserved)."""
    return {k: float(v) for k, v in feats}


# --------------------------------------------------------------------------
# Small numeric helpers
# --------------------------------------------------------------------------
def _nan() -> float:
    return float("nan")


def _f(v: Any) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return _nan()
    return v if math.isfinite(v) else _nan()


def _log2ratio(a: float, b: float, eps: float = 1e-6) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return _nan()
    return float(math.log2((a + eps) / (b + eps)))


def _entropy_norm(hist: np.ndarray) -> float:
    h = np.asarray(hist, dtype=np.float64).ravel()
    s = h.sum()
    if s <= 0 or h.size < 2:
        return _nan()
    p = h[h > 0] / s
    return float(-(p * np.log2(p)).sum() / math.log2(h.size))


def _wrap90(deg: float) -> float:
    """Wrap an axial angle to (-90, 90]."""
    if not math.isfinite(deg):
        return _nan()
    a = (deg + 90.0) % 180.0 - 90.0
    return 90.0 if a == -90.0 else float(a)


def _axial_mean_deg(angles_deg: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    a = np.radians(np.asarray(angles_deg, dtype=np.float64)) * 2.0
    w = np.ones_like(a) if weights is None else np.asarray(weights, dtype=np.float64)
    if a.size == 0 or w.sum() <= 0:
        return _nan()
    return float(np.degrees(0.5 * math.atan2((w * np.sin(a)).sum(), (w * np.cos(a)).sum())))


def _disk(r: float) -> np.ndarray:
    k = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))


def _masked_fill(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace pixels outside ``mask`` by the mean inside, so filters see no edges."""
    out = img.astype(np.float32).copy()
    if mask.any():
        out[~mask] = float(img[mask].mean())
    return out


def _skew(x: np.ndarray, w: np.ndarray) -> float:
    w = np.clip(np.asarray(w, dtype=np.float64), 0, None)
    if w.sum() <= _EPS:
        return _nan()
    w = w / w.sum()
    m = (w * x).sum()
    v = (w * (x - m) ** 2).sum()
    if v <= _EPS:
        return _nan()
    return float((w * (x - m) ** 3).sum() / v ** 1.5)


def _gray_f(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return bgr.astype(np.float32)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

# --------------------------------------------------------------------------
# Front-end: resample, levels, masks
# --------------------------------------------------------------------------
def prepare_roi(roi_bgr: np.ndarray, module_px: float, *, polarity: Optional[str] = None,
                code_size_px: Optional[Tuple[float, float]] = None,
                config: Optional[PrintFeatureConfig] = None) -> PreparedROI:
    """Resample a padded, deskewed ROI crop to the canonical pitch and build the masks.

    Args:
        roi_bgr: crop from ``dm_detector.crop(img, roi, annotate=False, pad=code_pad)``
            (BGR uint8 or gray).
        module_px: native module pitch (px per module) from the detector.
        polarity: ``"dark_on_light"`` / ``"light_on_dark"``; ``None`` = auto.
        code_size_px: native (width, height) of the code square inside the crop;
            ``None`` = square of side ``crop_side / (1 + 2 * code_pad)``.
    """
    cfg = config or PrintFeatureConfig()
    if not module_px or module_px <= 0:
        raise ValueError("module_px must be positive")
    bgr = roi_bgr if roi_bgr.ndim == 3 else cv2.cvtColor(roi_bgr, cv2.COLOR_GRAY2BGR)
    H0, W0 = bgr.shape[:2]
    ppm = float(cfg.canonical_px_per_module)
    f = ppm / float(module_px)
    if max(H0, W0) * f > cfg.max_canonical_side:
        f = cfg.max_canonical_side / float(max(H0, W0))
        ppm = f * float(module_px)
    W, H = max(8, int(round(W0 * f))), max(8, int(round(H0 * f)))
    interp = cv2.INTER_AREA if f < 1.0 else cv2.INTER_CUBIC
    rs = cv2.resize(bgr, (W, H), interpolation=interp)
    glare_native = (bgr.max(axis=2) >= cfg.glare_level).astype(np.uint8)
    glare = cv2.resize(glare_native, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    if glare.any():
        glare = cv2.dilate(glare.astype(np.uint8), _disk(max(1.0, 0.08 * ppm))).astype(bool)
    lum = _gray_f(rs)

    # code square in canonical coordinates
    if code_size_px is None:
        side0 = min(H0, W0) / (1.0 + 2.0 * cfg.code_pad)
        cw, ch = side0 * f, side0 * f
    else:
        cw, ch = code_size_px[0] * f, code_size_px[1] * f
    cx, cy = W / 2.0, H / 2.0
    code_box = (cx - cw / 2.0, cy - ch / 2.0, cx + cw / 2.0, cy + ch / 2.0)
    qz = cfg.quiet_zone_modules * ppm
    yy, xx = np.mgrid[0:H, 0:W]
    in_code = ((xx >= code_box[0]) & (xx < code_box[2]) & (yy >= code_box[1]) & (yy < code_box[3]))
    in_reg = ((xx >= code_box[0] - qz) & (xx < code_box[2] + qz) &
              (yy >= code_box[1] - qz) & (yy < code_box[3] + qz))
    valid_code = in_code & ~glare
    if valid_code.sum() < 50:
        valid_code = in_code

    # polarity + initial Otsu on the code square
    sm = cv2.GaussianBlur(lum, (0, 0), 0.7)
    vals = sm[valid_code].astype(np.uint8)
    t0, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if polarity not in ("dark_on_light", "light_on_dark"):
        dark_frac = float((vals < t0).mean())
        polarity = "dark_on_light" if dark_frac <= 0.55 else "light_on_dark"
    dark_mark = polarity == "dark_on_light"
    mark0 = (sm < t0) if dark_mark else (sm > t0)

    # local ink / substrate levels by normalised convolution (lighting gradients)
    sig = cfg.level_sigma_modules * ppm
    base = (in_reg & ~glare).astype(np.float32)

    def _local_level(m: np.ndarray, fallback: float) -> np.ndarray:
        w = cv2.GaussianBlur(m.astype(np.float32) * base, (0, 0), sig)
        v = cv2.GaussianBlur(sm * m.astype(np.float32) * base, (0, 0), sig)
        out = np.where(w > 1e-3, v / np.maximum(w, 1e-6), fallback)
        return out.astype(np.float32)

    ink_med = float(np.median(sm[mark0 & valid_code])) if (mark0 & valid_code).any() else float(t0)
    sub_med = float(np.median(sm[~mark0 & valid_code])) if (~mark0 & valid_code).any() else float(t0)
    I = _local_level(mark0, ink_med)
    S = _local_level(~mark0, sub_med)
    den = I - S
    tiny = np.abs(den) < 3.0
    den = np.where(tiny, np.sign(ink_med - sub_med + 1e-6) * 3.0, den)
    N = ((sm - S) / den).astype(np.float32)
    mark = (N > 0.5) & in_reg

    r = cfg.region_erode_modules * ppm
    k = _disk(r)
    region = in_reg & ~glare
    mark_u8 = mark.astype(np.uint8)
    ink = cv2.erode(mark_u8, k).astype(bool) & region & in_code
    bg_u8 = ((~mark) | ~in_reg).astype(np.uint8)
    substrate = cv2.erode(bg_u8, k).astype(bool) & region
    edge = region & ~ink & ~substrate & (cv2.dilate(mark_u8, k).astype(bool))
    edge &= ~cv2.erode(mark_u8, k).astype(bool)
    ink_level = float(np.median(lum[ink])) if ink.any() else ink_med
    sub_level = float(np.median(lum[substrate])) if substrate.any() else sub_med
    return PreparedROI(bgr=rs, lum=lum, markness=N, mark=mark, ink=ink, substrate=substrate,
                       edge=edge, glare=glare & in_reg, region=region, code_box=code_box, ppm=ppm,
                       capture_ppm=float(module_px), polarity=polarity, ink_level=ink_level,
                       substrate_level=sub_level,
                       extra={"in_code": in_code, "in_reg": in_reg, "otsu": float(t0),
                              "sm": sm, "resample_factor": f})


# --------------------------------------------------------------------------
# Module grid and edges
# --------------------------------------------------------------------------
def _comb_fit(profile: np.ndarray, p0: float, rel: float, lo: int) -> Tuple[float, float]:
    """Best pitch / phase of a periodic peak train in ``profile`` (x offset ``lo``)."""
    x = np.arange(profile.size, dtype=np.float64) + lo
    g = profile.astype(np.float64) - profile.mean()
    best = (p0, 0.0, -1.0)
    for p in np.linspace(p0 * (1 - rel), p0 * (1 + rel), 161):
        z = (g * np.exp(-2j * np.pi * x / p)).sum()
        a = abs(z)
        if a > best[2]:
            best = (p, -float(np.angle(z)) * p / (2 * np.pi), a)  # peaks at x = ph + k p
    p, ph, _ = best
    return float(p), float(ph % p)


@dataclass
class _Grid:
    x_lines: np.ndarray    # positions of vertical grid lines (n+1)
    y_lines: np.ndarray    # positions of horizontal grid lines (n+1)
    states: np.ndarray     # (ny+2, nx+2) bool, ring of quiet-zone modules (False)
    px: float
    py: float


def _fit_grid(P: PreparedROI) -> Optional[_Grid]:
    N = P.markness
    x0, y0, x1, y1 = P.code_box
    H, W = N.shape
    xa, xb = max(0, int(x0 - 0.5 * P.ppm)), min(W, int(x1 + 0.5 * P.ppm))
    ya, yb = max(0, int(y0 - 0.5 * P.ppm)), min(H, int(y1 + 0.5 * P.ppm))
    Nc = np.clip(cv2.GaussianBlur(N, (0, 0), 1.0), -0.5, 1.5)
    gx = np.abs(np.diff(Nc, axis=1))[ya:yb, xa:xb - 1].sum(axis=0)
    gy = np.abs(np.diff(Nc, axis=0))[ya:yb - 1, xa:xb].sum(axis=1)
    if gx.size < 3 * P.ppm or gy.size < 3 * P.ppm:
        return None
    px, phx = _comb_fit(gx, P.ppm, 0.08, xa)
    py, phy = _comb_fit(gy, P.ppm, 0.08, ya)
    phx += 0.5  # diff() sits between pixels
    phy += 0.5

    def lines(a0: float, a1: float, p: float, ph: float) -> np.ndarray:
        n = max(8, int(round((a1 - a0) / p)))
        k0 = round((a0 - ph) / p)
        start = ph + k0 * p
        return start + p * np.arange(n + 1)

    xl = lines(x0, x1, px, phx)
    yl = lines(y0, y1, py, phy)
    nx, ny = xl.size - 1, yl.size - 1
    states = np.zeros((ny + 2, nx + 2), dtype=bool)
    for i in range(ny):
        for j in range(nx):
            xs, xe = xl[j] + 0.25 * px, xl[j + 1] - 0.25 * px
            ys, ye = yl[i] + 0.25 * py, yl[i + 1] - 0.25 * py
            a, b = int(round(ys)), int(round(ye))
            c, e = int(round(xs)), int(round(xe))
            a, c = max(0, a), max(0, c)
            if b <= a or e <= c or b > H or e > W:
                continue
            states[i + 1, j + 1] = float(np.median(N[a:b, c:e])) > 0.5
    return _Grid(x_lines=xl, y_lines=yl, states=states, px=px, py=py)


def _interp_line(N: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return cv2.remap(N, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


# Normal directions: "-x" (mark on the right, edge faces left) ... outward normals
_DIRS = ("-x", "+x", "-y", "+y")

def _edge_measurements(P: PreparedROI, G: _Grid, cfg: PrintFeatureConfig) -> Dict[str, Dict[str, Any]]:
    """Per outward-normal direction: raggedness, blur, bias, skew, edge length."""
    N = P.markness
    ppm = P.ppm
    half = int(round(0.5 * ppm))
    offs = np.arange(-half, half + 0.001, 0.5)
    trim = cfg.edge_trim_modules
    st = G.states
    ny, nx = st.shape[0] - 2, st.shape[1] - 2
    out: Dict[str, Dict[str, Any]] = {}
    for dname in _DIRS:
        vertical = dname in ("-x", "+x")      # edge lies on a vertical grid line
        sign = 1.0 if dname[0] == "+" else -1.0
        runs: List[List[Tuple[int, int]]] = []
        seg_count = 0
        if vertical:
            for j in range(nx + 1):
                cur: List[Tuple[int, int]] = []
                for i in range(ny):
                    left, right = st[i + 1, j], st[i + 1, j + 1]
                    is_edge = (left and not right) if sign > 0 else (right and not left)
                    if is_edge:
                        cur.append((i, j))
                        seg_count += 1
                    elif cur:
                        runs.append(cur)
                        cur = []
                if cur:
                    runs.append(cur)
        else:
            for i in range(ny + 1):
                cur = []
                for j in range(nx):
                    up, down = st[i, j + 1], st[i + 1, j + 1]
                    is_edge = (up and not down) if sign > 0 else (down and not up)
                    if is_edge:
                        cur.append((i, j))
                        seg_count += 1
                    elif cur:
                        runs.append(cur)
                        cur = []
                if cur:
                    runs.append(cur)
        profiles: List[np.ndarray] = []
        offsets: List[float] = []
        rag: List[Tuple[float, float]] = []
        for run in runs:
            if vertical:
                j = run[0][1]
                line = G.x_lines[j]
                t_a = G.y_lines[run[0][0]] + trim * G.py
                t_b = G.y_lines[run[-1][0] + 1] - trim * G.py
            else:
                i = run[0][0]
                line = G.y_lines[i]
                t_a = G.x_lines[run[0][1]] + trim * G.px
                t_b = G.x_lines[run[-1][1] + 1] - trim * G.px
            ts = np.arange(math.ceil(t_a), math.floor(t_b) + 1, 1.0)
            if ts.size < 3:
                continue
            o = line + sign * offs[None, :]
            tt = np.repeat(ts[:, None], offs.size, axis=1)
            if vertical:
                prof = _interp_line(N, o.repeat(ts.size, 0), tt)
            else:
                prof = _interp_line(N, tt, o.repeat(ts.size, 0))
            prof = np.asarray(prof, dtype=np.float64).reshape(ts.size, offs.size)
            cross = np.full(ts.size, np.nan)
            for r_, pr in enumerate(prof):
                idx = np.where((pr[:-1] >= 0.5) & (pr[1:] < 0.5))[0]
                if idx.size == 0:
                    continue
                c0 = idx[np.argmin(np.abs(offs[idx]))]
                a, b = pr[c0], pr[c0 + 1]
                cross[r_] = offs[c0] + (a - 0.5) / (a - b + _EPS) * (offs[c0 + 1] - offs[c0])
            ok = np.isfinite(cross)
            if ok.sum() < 3:
                continue
            offsets.extend(cross[ok].tolist())
            for r_ in np.where(ok)[0]:
                profiles.append(np.interp(offs + cross[r_], offs, prof[r_]))
            if len(run) >= cfg.min_run_modules and ok.sum() >= 0.6 * ppm:
                A = np.vstack([ts[ok], np.ones(ok.sum())]).T
                coef, *_ = np.linalg.lstsq(A, cross[ok], rcond=None)
                res = cross[ok] - A @ coef
                rag.append((float(res.std()) / ppm, float(ok.sum())))
        rec: Dict[str, Any] = {"length_modules": float(seg_count), "n_rows": len(offsets)}
        rec["raggedness"] = (float(np.average([r for r, _ in rag], weights=[w for _, w in rag]))
                             if rag else _nan())
        rec["offset_mean"] = float(np.mean(offsets)) / ppm if offsets else _nan()
        if profiles:
            mp = np.mean(np.vstack(profiles), axis=0)
            inside = mp[offs <= -0.35 * ppm].mean() if (offs <= -0.35 * ppm).any() else mp[0]
            outside = mp[offs >= 0.35 * ppm].mean() if (offs >= 0.35 * ppm).any() else mp[-1]
            span = inside - outside
            if span > 0.2:
                q = (mp - outside) / span
                c = offs.size // 2
                w90 = _first_cross(offs, q, 0.9, c, -1)
                w10 = _first_cross(offs, q, 0.1, c, +1)
                rec["blur"] = (w10 - w90) / ppm if math.isfinite(w90) and math.isfinite(w10) else _nan()
                dq = -np.gradient(q, offs)
                rec["skew"] = _skew(offs, dq)
            else:
                rec["blur"] = _nan()
                rec["skew"] = _nan()
        else:
            rec["blur"] = _nan()
            rec["skew"] = _nan()
        out[dname] = rec
    return out


def _first_cross(offs: np.ndarray, q: np.ndarray, level: float, c: int, step: int) -> float:
    """Walk from the centre index ``c`` in direction ``step`` until q crosses ``level``."""
    i = c
    n = q.size
    while 0 <= i + step < n:
        a, b = q[i], q[i + step]
        if (a - level) * (b - level) <= 0 and a != b:
            return float(offs[i] + (a - level) / (a - b) * (offs[i + step] - offs[i]))
        i += step
    return _nan()


def _corner_radius(P: PreparedROI, G: _Grid, bias_x: float, bias_y: float) -> float:
    """Mean fitted radius of convex mark corners (module units)."""
    N = P.markness
    st = G.states
    ny, nx = st.shape[0] - 2, st.shape[1] - 2
    radii: List[float] = []
    bx = (bias_x if math.isfinite(bias_x) else 0.0) * P.ppm
    by = (bias_y if math.isfinite(bias_y) else 0.0) * P.ppm
    ts = np.arange(-0.35 * P.ppm, 0.6 * P.ppm, 0.25)
    for i in range(ny):
        for j in range(nx):
            if not st[i + 1, j + 1]:
                continue
            for dy, dx in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                if st[i + 1 + dy, j + 1] or st[i + 1, j + 1 + dx] or st[i + 1 + dy, j + 1 + dx]:
                    continue
                cxp = G.x_lines[j + (1 if dx > 0 else 0)] + dx * bx
                cyp = G.y_lines[i + (1 if dy > 0 else 0)] + dy * by
                ux, uy = -dx / math.sqrt(2), -dy / math.sqrt(2)
                xs = cxp + ts * ux
                ys = cyp + ts * uy
                v = np.asarray(_interp_line(N, xs[None, :], ys[None, :]), dtype=np.float64).ravel()
                idx = np.where((v[:-1] < 0.5) & (v[1:] >= 0.5))[0]
                if idx.size == 0:
                    continue
                k = idx[0]
                t = ts[k] + (0.5 - v[k]) / (v[k + 1] - v[k] + _EPS) * (ts[k + 1] - ts[k])
                radii.append(max(0.0, t) / (math.sqrt(2) - 1) / P.ppm)
    return float(np.mean(radii)) if len(radii) >= 3 else _nan()


# --------------------------------------------------------------------------
# Texture helpers
# --------------------------------------------------------------------------
def _structure_tensor(img: np.ndarray, sigma_int: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = cv2.GaussianBlur(img, (0, 0), 1.0)
    ix = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    iy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    jxx = cv2.GaussianBlur(ix * ix, (0, 0), sigma_int)
    jxy = cv2.GaussianBlur(ix * iy, (0, 0), sigma_int)
    jyy = cv2.GaussianBlur(iy * iy, (0, 0), sigma_int)
    return jxx, jxy, jyy


def _st_stats(img: np.ndarray, mask: np.ndarray, sigma_int: float) -> Tuple[float, float]:
    """(mean local coherence, dominant LINE orientation in image degrees) over ``mask``."""
    if mask.sum() < 20:
        return _nan(), _nan()
    jxx, jxy, jyy = _structure_tensor(img.astype(np.float32), sigma_int)
    tr = jxx + jyy
    disc = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    coh = np.where(tr > 1e-12, disc / np.maximum(tr, 1e-12), 0.0)
    a, b, c = float(jxx[mask].sum()), float(jxy[mask].sum()), float(jyy[mask].sum())
    grad_ang = 0.5 * math.degrees(math.atan2(2 * b, a - c))
    return float(coh[mask].mean()), grad_ang + 90.0


def _gabor_energy(img: np.ndarray, mask: np.ndarray, lam: float, theta_deg: float) -> float:
    sigma = 0.56 * lam
    ks = int(2 * math.ceil(2.5 * sigma) + 1)
    th = math.radians(theta_deg)
    kr = cv2.getGaborKernel((ks, ks), sigma, th, lam, 1.0, 0.0, ktype=cv2.CV_32F)
    ki = cv2.getGaborKernel((ks, ks), sigma, th, lam, 1.0, math.pi / 2, ktype=cv2.CV_32F)
    kr -= kr.mean()
    ki -= ki.mean()
    nr = float(np.abs(kr).sum()) + _EPS
    r = cv2.filter2D(img, cv2.CV_32F, kr / nr)
    i = cv2.filter2D(img, cv2.CV_32F, ki / nr)
    return float((r[mask] ** 2 + i[mask] ** 2).mean())


def _lbp_stats(img: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float]:
    """(riu2 entropy pooled over R=1,2,3, uniform fraction, log VAR) over ``mask``."""
    from skimage.feature import local_binary_pattern
    if mask.sum() < 30:
        return _nan(), _nan(), _nan()
    hists = []
    uni = []
    im = img.astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for R in (1, 2, 3):
            codes = local_binary_pattern(im, 8, R, method="uniform")[mask]
            hists.append(np.bincount(codes.astype(np.int64), minlength=10)[:10])
            uni.append(float((codes < 9).mean()))
        var = local_binary_pattern(im, 8, 1, method="var")[mask]
    var = var[np.isfinite(var)]
    lv = float(math.log(float(np.mean(var)) + 1e-8)) if var.size else _nan()
    return _entropy_norm(np.concatenate(hists)), float(np.mean(uni)), lv


_NB8 = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]  # (dy, dx)


def _neighbour_stack(img: np.ndarray) -> np.ndarray:
    p = np.pad(img, 1, mode="edge")
    H, W = img.shape
    return np.stack([p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] for dy, dx in _NB8], axis=0)


def _lbp_rv_ratio(img: np.ndarray, mask: np.ndarray, axis_deg: float) -> float:
    """log2(#edge-like LBP codes whose boundary runs along the print axis / across).

    Neighbour k sits at image angle -45*k degrees (y down), i.e. the direction
    (dx, dy) = (cos, -sin) of 45*k; an edge-like code has 4 consecutive ones and
    its boundary is perpendicular to the mean direction of those ones.
    """
    if mask.sum() < 30:
        return _nan()
    nb = _neighbour_stack(img)
    bits = (nb >= img[None]).astype(np.uint8)[:, mask]
    along = across = 0
    for s in range(8):
        pat = np.zeros(8, np.uint8)
        pat[[(s + k) % 8 for k in range(4)]] = 1
        hit = int(np.all(bits == pat[:, None], axis=0).sum())
        centre = -(s * 45.0 + 67.5)          # image-frame angle of the ones' mean direction
        rel = (centre - axis_deg) % 180.0
        if 45.0 < rel < 135.0:               # ones point across the axis -> boundary along it
            along += hit
        else:
            across += hit
    return _log2ratio(float(along), float(across), 1.0)


def _glcm_feats(img: np.ndarray, mask: np.ndarray, axis_is_x: bool) -> Dict[str, float]:
    from skimage.feature import graycomatrix, graycoprops
    keys = ("contrast_d1", "contrast_d2", "homogeneity_d1", "energy_d1", "correlation_d1",
            "correlation_d2", "contrast_along_vs_across_log2ratio")
    res = {k: _nan() for k in keys}
    if mask.sum() < 100:
        return res
    med = float(np.median(img[mask]))
    q = np.clip(np.floor((img - (med - 0.5)) * 32.0), 0, 31).astype(np.uint8)
    q[~mask] = 32
    along = 0.0 if axis_is_x else np.pi / 2
    across = np.pi / 2 if axis_is_x else 0.0
    g = graycomatrix(q, distances=[3, 6], angles=[along, across], levels=33, symmetric=True,
                     normed=False).astype(np.float64)
    g = g[:32, :32]
    s = g.sum(axis=(0, 1), keepdims=True)
    if np.any(s <= 0):
        return res
    g = g / s
    con = graycoprops(g, "contrast")
    hom = graycoprops(g, "homogeneity")
    asm = graycoprops(g, "ASM")
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cor = graycoprops(g, "correlation")
    res["contrast_d1"] = float(con[0].mean())
    res["contrast_d2"] = float(con[1].mean())
    res["homogeneity_d1"] = float(hom[0].mean())
    res["energy_d1"] = float(asm[0].mean())
    res["correlation_d1"] = _f(np.nanmean(cor[0]))
    res["correlation_d2"] = _f(np.nanmean(cor[1]))
    res["contrast_along_vs_across_log2ratio"] = _log2ratio(float(con[0, 0]), float(con[0, 1]), 1e-4)
    return res


def _banding(img: np.ndarray, mask: np.ndarray, along_x: bool, ppm: float) -> Tuple[float, float]:
    """(normalised spectral peak, period in modules) of the masked mean profile."""
    m = mask.astype(np.float64)
    v = img.astype(np.float64) * m
    if along_x:
        cnt, sm = m.sum(axis=0), v.sum(axis=0)
    else:
        cnt, sm = m.sum(axis=1), v.sum(axis=1)
    if cnt.size == 0 or cnt.max() <= 0:
        return _nan(), _nan()
    ok = cnt >= max(3.0, 0.1 * cnt.max())
    if ok.sum() < 4 * ppm:
        return _nan(), _nan()
    idx = np.arange(cnt.size)
    first, last = idx[ok][0], idx[ok][-1]
    prof = np.interp(idx[first:last + 1], idx[ok], sm[ok] / cnt[ok])
    x = np.arange(prof.size)
    prof = prof - np.polyval(np.polyfit(x, prof, 1), x)
    prof *= np.hanning(prof.size)
    pw = np.abs(np.fft.rfft(prof)) ** 2
    freqs = np.fft.rfftfreq(prof.size, d=1.0)
    periods = np.full(freqs.shape, np.inf)
    periods[freqs > 0] = 1.0 / freqs[freqs > 0] / ppm
    band = (periods >= 0.25) & (periods <= 8.0)
    if band.sum() < 3 or pw[band].sum() <= 0:
        return _nan(), _nan()
    k = int(np.argmax(np.where(band, pw, -1)))
    return float(pw[k] / pw[band].sum()), float(periods[k])


def _blob_table(binary: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, lab, stats, cents = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    return lab, stats, cents


def _region_props(lab: np.ndarray, ids: Sequence[int]) -> List[Dict[str, float]]:
    """Area, perimeter, roundness, equivalent diameter, orientation, eccentricity."""
    props: List[Dict[str, float]] = []
    if not len(ids):
        return props
    objs = ndi.find_objects(lab)
    for i in ids:
        sl = objs[i - 1] if i - 1 < len(objs) else None
        if sl is None:
            continue
        sub = (lab[sl] == i).astype(np.uint8)
        area = float(sub.sum())
        cs, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        per = float(sum(cv2.arcLength(c, True) for c in cs)) if cs else 0.0
        ys, xs = np.nonzero(sub)
        orient, ecc = _nan(), 0.0
        if area >= 5:
            cov = np.cov(np.vstack([xs, ys]))
            ev, evec = np.linalg.eigh(cov)
            if ev[1] > 0:
                ecc = float(math.sqrt(max(0.0, 1 - ev[0] / ev[1])))
                orient = float(math.degrees(math.atan2(evec[1, 1], evec[0, 1])))
        props.append({"area": area, "perimeter": per,
                      "roundness": float(4 * math.pi * area / per ** 2) if per > 0 else _nan(),
                      "eqd": float(math.sqrt(4 * area / math.pi)), "orient": orient, "ecc": ecc,
                      "cx": float(xs.mean() + sl[1].start), "cy": float(ys.mean() + sl[0].start)})
    return props


def _kirsch_dominant(img: np.ndarray) -> np.ndarray:
    """Index (0..7) of the strongest Kirsch compass response per pixel."""
    ring = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
    vals = [5, 5, 5, -3, -3, -3, -3, -3]
    resp = []
    for k in range(8):
        kk = np.zeros((3, 3), np.float32)
        for i, r in enumerate(ring):
            kk[r] = vals[(i - k) % 8]
        resp.append(np.abs(cv2.filter2D(img.astype(np.float32), cv2.CV_32F, kk)))
    return np.argmax(np.stack(resp, 0), axis=0).astype(np.int64)


# --------------------------------------------------------------------------
# Print direction vote
# --------------------------------------------------------------------------
def _axis_vote(cues: Sequence[Tuple[float, float]]) -> Tuple[int, float]:
    """cues: (value in [-1, 1] with + = code x, weight) -> (axis, confidence)."""
    num = sum(w * v for v, w in cues if math.isfinite(v))
    den = sum(w for v, w in cues if math.isfinite(v))
    if den <= 0:
        return 0, 0.0
    return (0 if num >= 0 else 1), float(min(1.0, abs(num) / den))


def _rel_angle(img_deg: float, axis: int) -> float:
    return _wrap90(img_deg - (0.0 if axis == 0 else 90.0))


# --------------------------------------------------------------------------
# Optional AdaIN block
# --------------------------------------------------------------------------
_VGG_CACHE: Dict[str, Any] = {}


def _vgg_features() -> Any:
    if "net" not in _VGG_CACHE:
        from torchvision.models import VGG19_Weights, vgg19
        net = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:7].eval()
        for p in net.parameters():
            p.requires_grad_(False)
        _VGG_CACHE["net"] = net
    return _VGG_CACHE["net"]


def _adain_block(P: PreparedROI, cfg: PrintFeatureConfig) -> List[float]:
    try:
        import torch
        net = _vgg_features()
    except Exception as exc:  # torch missing / no weights: leave NaN
        warnings.warn("AdaIN block disabled: %s" % exc)
        return [_nan()] * 12
    rgb = cv2.cvtColor(P.bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    x = torch.from_numpy(((rgb - mean) / std).transpose(2, 0, 1)[None].copy())
    feats = []
    with torch.no_grad():
        h = x
        for idx, layer in enumerate(net):
            h = layer(h)
            if idx in (1, 6):  # relu1_1, relu2_1
                feats.append(h[0].numpy())
    rng = np.random.default_rng(cfg.adain_seed)
    out: List[float] = []
    proj = None
    for reg in (P.ink, P.substrate, P.edge):
        vec = []
        for fm in feats:
            m = cv2.resize(reg.astype(np.uint8), (fm.shape[2], fm.shape[1]),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
            if m.sum() < 4:
                vec = []
                break
            v = fm[:, m]
            vec.append(v.mean(axis=1))
            vec.append(np.log(v.std(axis=1) + 1e-3))
        if not vec:
            out.extend([_nan()] * 4)
            continue
        z = np.concatenate(vec)
        if proj is None:
            proj = rng.standard_normal((4, z.size)) / math.sqrt(z.size)
        out.extend([float(v) for v in proj @ z])
    return out


def _jpeg_blockiness(native: Optional[np.ndarray], origin: Tuple[int, int] = (0, 0)) -> float:
    """8x8 block-boundary / interior mean absolute gradient ratio (1 = no blocking)."""
    if native is None or native.size == 0 or min(native.shape[:2]) < 32:
        return _nan()
    g = _gray_f(native)
    ox, oy = origin
    ratios = []
    for axis, o in ((1, ox), (0, oy)):
        dif = np.abs(np.diff(g, axis=axis))
        prof = dif.mean(axis=0 if axis == 1 else 1)
        pos = (np.arange(prof.size) + o) % 8 == 7
        if pos.sum() == 0 or (~pos).sum() == 0:
            continue
        ratios.append(float(prof[pos].mean() / (prof[~pos].mean() + _EPS)))
    return float(np.mean(ratios)) if ratios else _nan()


# --------------------------------------------------------------------------
# Main extractor
# --------------------------------------------------------------------------
def extract_print_features(
    roi_bgr_or_gray: np.ndarray,
    *,
    module_px: float,
    angle_deg: float = 0.0,
    polarity: Optional[str] = None,
    module_px_xy: Optional[Tuple[float, float]] = None,
    dotted_hint: Optional[bool] = None,
    canonical_px_per_module: float = 24.0,
    include_adain: bool = False,
    native_crop_for_jpeg_metric: Optional[np.ndarray] = None,
    native_crop_origin: Tuple[int, int] = (0, 0),
    code_size_px: Optional[Tuple[float, float]] = None,
    config: Optional[PrintFeatureConfig] = None,
    return_debug: bool = False,
) -> Any:
    """Return the ORDERED list of ``(name, value)`` covering the 119-row table.

    Args:
        roi_bgr_or_gray: padded, deskewed, upright crop (``dm_detector.crop`` with
            ``annotate=False``); the code square is centred in it.
        module_px: native module pitch (px / module).
        angle_deg: detector angle, informational only (the crop is already upright).
        polarity: ``"dark_on_light"`` / ``"light_on_dark"`` / ``None`` (auto).
        module_px_xy: native pitch along crop x / y (for ``module_pitch_anisotropy``).
        dotted_hint: the detector's ``dotted_fallback`` flag (``None`` = blob test only).
        canonical_px_per_module: resampling target (default 24).
        include_adain: compute rows 108-119 (needs torch + torchvision).
        native_crop_for_jpeg_metric: native un-deskewed crop for ``jpeg_blockiness``.
        native_crop_origin: (x, y) of that crop in the photo (8x8 lattice alignment).
        code_size_px: native (w, h) of the code square in ``roi_bgr_or_gray``.
        config: :class:`PrintFeatureConfig` overrides.
        return_debug: also return a dict with the prepared ROI and intermediate data.

    Returns:
        ``[(name, value), ...]`` in :data:`FEATURE_NAMES` order (and the debug
        dict with ``return_debug=True``).
    """
    base = config or PrintFeatureConfig()
    cfg = PrintFeatureConfig(**{**base.__dict__, "canonical_px_per_module": canonical_px_per_module,
                                "include_adain": include_adain or base.include_adain})
    P = prepare_roi(roi_bgr_or_gray, module_px, polarity=polarity, code_size_px=code_size_px,
                    config=cfg)
    ppm = P.ppm
    F: Dict[str, float] = {k: _nan() for k in FEATURE_NAMES}
    N = P.markness
    contrast = abs(P.ink_level - P.substrate_level)
    in_code = P.extra["in_code"]

    # --- QC block -----------------------------------------------------------
    F["capture_px_per_module"] = float(module_px)
    F["resample_factor"] = float(P.extra["resample_factor"])
    F["upsampled_flag"] = 1.0 if P.extra["resample_factor"] > 1.0 else 0.0
    if module_px_xy and module_px_xy[1]:
        F["module_pitch_anisotropy"] = float(module_px_xy[0]) / float(module_px_xy[1])
    if P.edge.sum() > 20:
        lap = cv2.Laplacian(cv2.GaussianBlur(N, (0, 0), 0.5), cv2.CV_32F)
        v = float(lap[P.edge].var())
        F["sharpness_score"] = v / (v + 0.01)
    F["glare_fraction"] = float(P.glare.sum() / max(1, P.extra["in_reg"].sum()))
    F["jpeg_blockiness"] = _jpeg_blockiness(native_crop_for_jpeg_metric, native_crop_origin)
    vals = P.extra["sm"][in_code & ~P.glare]
    if vals.size > 50 and vals.var() > 0:
        t = P.extra["otsu"]
        a, b = vals[vals < t], vals[vals >= t]
        if a.size and b.size:
            w0, w1 = a.size / vals.size, b.size / vals.size
            F["bimodality_separability"] = float(w0 * w1 * (a.mean() - b.mean()) ** 2 / vals.var())
    F["mark_polarity"] = 0.0 if P.polarity == "dark_on_light" else 1.0

    # --- grid and edge measurements -----------------------------------------
    G = _fit_grid(P)
    E = _edge_measurements(P, G, cfg) if G is not None else {}
    ink_f = _masked_fill(N, P.ink)
    sub_f = _masked_fill(N, P.substrate)

    # --- dots (also a print-axis cue) ------------------------------------------
    lab, stats, _ = _blob_table(P.mark & in_code)
    mod_a = ppm * ppm
    n_mark_modules = int(G.states.sum()) if G is not None else 0
    cand = [i for i in range(1, stats.shape[0])
            if 0.05 * mod_a <= stats[i, cv2.CC_STAT_AREA] <= 1.5 * mod_a]
    dprops = [p for p in _region_props(lab, cand) if p["roundness"] > 0.55]
    blob_dotted = len(dprops) >= max(20, 0.4 * n_mark_modules)
    dotted = bool(dotted_hint) or blob_dotted

    # --- print-axis vote ---------------------------------------------------------
    bx_peak, bx_per = _banding(ink_f, P.ink, True, ppm)
    by_peak, by_per = _banding(ink_f, P.ink, False, ppm)
    coh_ink, line_ang = _st_stats(ink_f, P.ink, 0.25 * ppm)
    cues: List[Tuple[float, float]] = []
    if math.isfinite(bx_peak) and math.isfinite(by_peak) and bx_peak + by_peak > 0:
        cues.append(((bx_peak - by_peak) / (bx_peak + by_peak), 1.0))
    if math.isfinite(coh_ink) and math.isfinite(line_ang):
        cues.append((coh_ink * math.cos(2 * math.radians(line_ang)), 1.0))
    if E:
        rx = np.nanmean([E["-x"]["raggedness"], E["+x"]["raggedness"]])
        ry = np.nanmean([E["-y"]["raggedness"], E["+y"]["raggedness"]])
        if np.isfinite(rx) and np.isfinite(ry) and rx + ry > 0:
            # ragged edges with normals along x (vertical edges) point to an x process axis
            cues.append((float((rx - ry) / (rx + ry)), 0.5))
    if dotted and len(dprops) >= 5:
        el = [p for p in dprops if p["ecc"] > 0.5 and math.isfinite(p["orient"])]
        if len(el) >= 5:
            mo = _axial_mean_deg(np.array([p["orient"] for p in el]))
            cues.append((float(np.mean([p["ecc"] for p in el])) * math.cos(2 * math.radians(mo)), 1.0))
    axis, axis_conf = _axis_vote(cues)
    axis_deg = 0.0 if axis == 0 else 90.0
    F["print_axis"] = float(axis)
    F["print_axis_confidence"] = axis_conf

    # --- satellites / spatter / specks (per outward normal) ------------------------
    small_max = cfg.satellite_max_area * mod_a
    lab_all, st_all, _ = _blob_table(P.mark & P.region)
    areas = st_all[:, cv2.CC_STAT_AREA]
    big_ids = np.where(areas > small_max)[0]
    big_ids = big_ids[big_ids > 0]
    big = np.isin(lab_all, big_ids)
    sat_ids = [i for i in range(1, st_all.shape[0]) if 3 <= areas[i] <= small_max]
    sats: List[Dict[str, Any]] = []
    specks: List[Dict[str, Any]] = []
    if big.any() and sat_ids:
        dist, (iy, ix) = ndi.distance_transform_edt(~big, return_indices=True)
        for p in _region_props(lab_all, sat_ids):
            yi = min(max(int(round(p["cy"])), 0), dist.shape[0] - 1)
            xi = min(max(int(round(p["cx"])), 0), dist.shape[1] - 1)
            # tiny, faint blobs are threshold noise
            if float(N[yi, xi]) < 0.6 and p["area"] < 6:
                continue
            dd = float(dist[yi, xi]) / ppm
            vx, vy = xi - ix[yi, xi], yi - iy[yi, xi]
            if abs(vx) >= abs(vy):
                dname = "+x" if vx > 0 else "-x"
            else:
                dname = "+y" if vy > 0 else "-y"
            p.update({"dist": dd, "dir": dname})
            lo, hi = cfg.satellite_dist
            if lo <= dd <= hi:
                sats.append(p)
            elif dd > 1.0:
                specks.append(p)
    sat_density: Dict[str, float] = {}
    for dname in _DIRS:
        L = E[dname]["length_modules"] if E else 0.0
        n = sum(1 for p in sats if p["dir"] == dname)
        sat_density[dname] = n / L if L > 0 else _nan()

    # --- lead / trail sign vote ----------------------------------------------------
    if axis == 0:
        neg, pos, ol, orr = "-x", "+x", "-y", "+y"
    else:
        neg, pos, ol, orr = "-y", "+y", "+x", "-x"
    s_num, s_den = 0.0, 0.0
    if E:
        bn, bp = E[neg]["blur"], E[pos]["blur"]
        if math.isfinite(bn) and math.isfinite(bp) and bn + bp > 0:
            s_num += (bp - bn) / (bp + bn)      # wider transition on the trailing side
            s_den += 1.0
        kn, kp = E[neg]["skew"], E[pos]["skew"]
        if math.isfinite(kn) and math.isfinite(kp):
            s_num += 0.5 * float(np.tanh(kp - kn))  # outward tail on the trailing side
            s_den += 0.5
    sn, sp = sat_density.get(neg, _nan()), sat_density.get(pos, _nan())
    if math.isfinite(sn) and math.isfinite(sp) and sn + sp > 0:
        s_num += (sp - sn) / (sp + sn)          # satellites land downstream
        s_den += 1.0
    direction = 1 if s_num >= 0 else -1          # +1: process along +axis
    F["print_polarity_confidence"] = float(min(1.0, abs(s_num) / s_den)) if s_den > 0 else 0.0
    if direction > 0:
        cls = {"leading": neg, "trailing": pos, "orth_left": ol, "orth_right": orr}
    else:
        cls = {"leading": pos, "trailing": neg, "orth_left": orr, "orth_right": ol}

    if E:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            bias_x = float(np.nanmean([E["-x"]["offset_mean"], E["+x"]["offset_mean"]]))
            bias_y = float(np.nanmean([E["-y"]["offset_mean"], E["+y"]["offset_mean"]]))
        for c, dname in cls.items():
            F["edge_raggedness_%s" % c] = _f(E[dname]["raggedness"])
            F["edge_blur_width_%s" % c] = _f(E[dname]["blur"])
            F["edge_bias_%s" % c] = _f(bias_x if dname in ("-x", "+x") else bias_y)
            F["edge_profile_skew_%s" % c] = _f(E[dname]["skew"])
            F["edge_satellite_density_%s" % c] = _f(sat_density[dname])
        F["edge_raggedness_lead_trail_log2ratio"] = _log2ratio(F["edge_raggedness_trailing"],
                                                               F["edge_raggedness_leading"], 1e-4)
        F["edge_blur_lead_trail_log2ratio"] = _log2ratio(F["edge_blur_width_trailing"],
                                                         F["edge_blur_width_leading"], 1e-4)
        F["edge_satellite_lead_trail_log2ratio"] = _log2ratio(
            F["edge_satellite_density_trailing"], F["edge_satellite_density_leading"], 0.05)

        def _mean2(a: str, b: str) -> float:
            v = [x for x in (F[a], F[b]) if math.isfinite(x)]
            return float(np.mean(v)) if v else _nan()

        F["edge_raggedness_process_vs_orth_log2ratio"] = _log2ratio(
            _mean2("edge_raggedness_leading", "edge_raggedness_trailing"),
            _mean2("edge_raggedness_orth_left", "edge_raggedness_orth_right"), 1e-4)
        F["edge_blur_process_vs_orth_log2ratio"] = _log2ratio(
            _mean2("edge_blur_width_leading", "edge_blur_width_trailing"),
            _mean2("edge_blur_width_orth_left", "edge_blur_width_orth_right"), 1e-4)
        F["corner_rounding_radius"] = _corner_radius(P, G, bias_x, bias_y)

    # --- structure tensor ------------------------------------------------------------
    F["st_coherence_ink"] = _f(coh_ink)
    F["st_orientation_ink"] = _rel_angle(line_ang, axis)
    coh_e, _ = _st_stats(N, P.edge, 0.25 * ppm)
    F["st_coherence_edge"] = _f(coh_e)

    # --- Gabor -------------------------------------------------------------------------
    if P.ink.sum() > 50:
        pooled = np.zeros(4)
        for si, lam in ((1, 0.25 * ppm), (2, 0.5 * ppm)):
            for oi, o in enumerate(_ORIENTS):
                e = _gabor_energy(ink_f, P.ink, lam, float(o) + axis_deg)
                F["gabor_ink_s%d_%s" % (si, o)] = e
                pooled[oi] += e
        if pooled.sum() > 0:
            F["gabor_ink_orientation_of_max"] = float(int(_ORIENTS[int(np.argmax(pooled))]))
            F["gabor_ink_anisotropy"] = float((pooled.max() - pooled.min()) / pooled.sum())
    if P.substrate.sum() > 50:
        pooled = np.zeros(4)
        for lam in (0.25 * ppm, 0.5 * ppm):
            for oi, o in enumerate(_ORIENTS):
                pooled[oi] += _gabor_energy(sub_f, P.substrate, lam, float(o) + axis_deg)
        if pooled.sum() > 0:
            F["gabor_substrate_orientation_of_max"] = float(int(_ORIENTS[int(np.argmax(pooled))]))
            F["gabor_substrate_anisotropy"] = float((pooled.max() - pooled.min()) / pooled.sum())

    # --- banding --------------------------------------------------------------------------
    F["banding_peak_process"] = _f(bx_peak if axis == 0 else by_peak)
    F["banding_peak_cross"] = _f(by_peak if axis == 0 else bx_peak)
    F["banding_period_process"] = _f(bx_per if axis == 0 else by_per)

    # --- LBP family ---------------------------------------------------------------------
    e_, u_, v_ = _lbp_stats(N, P.ink)
    F["lbp_riu2_ink_entropy"], F["lbp_riu2_ink_uniform_fraction"], F["lbp_var_ink"] = e_, u_, v_
    F["lbp_rv_ink_process_vs_cross"] = _lbp_rv_ratio(N, P.ink, axis_deg)
    if P.ink.sum() > 30:
        nb = _neighbour_stack(N)
        F["clbp_m_ink_mean"] = float(np.abs(nb - N[None])[:, P.ink].mean())
    e_, u_, v_ = _lbp_stats(N, P.substrate)
    F["lbp_riu2_substrate_entropy"], F["lbp_riu2_substrate_uniform_fraction"] = e_, u_
    F["lbp_var_substrate"] = v_
    F["lbp_rv_substrate_process_vs_cross"] = _lbp_rv_ratio(N, P.substrate, axis_deg)
    F["lbp_riu2_edge_entropy"] = _lbp_stats(N, P.edge)[0]
    if P.edge.sum() > 30:
        kir = _kirsch_dominant(N)
        F["ldp_edge_direction_entropy"] = _entropy_norm(np.bincount(kir[P.edge], minlength=8))
    contour = P.mark & ~cv2.erode(P.mark.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    contour &= P.region
    if contour.sum() > 30:
        gx = cv2.Sobel(N, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(N, cv2.CV_32F, 0, 1, ksize=3)
        ang = np.degrees(np.arctan2(gy[contour], gx[contour])) % 180.0
        F["hog_contour_orientation_entropy"] = _entropy_norm(np.histogram(ang, 18, (0, 180))[0])

    # --- GLCM (32 levels = 1/32 of the ink-substrate contrast per level) ---------------
    for reg, m in (("ink", P.ink), ("substrate", P.substrate)):
        for k_, v_ in _glcm_feats(N, m, axis == 0).items():
            F["glcm_%s_%s" % (reg, k_)] = v_

    # --- ink density / mottle / grain / voids / streaks ----------------------------------
    if P.ink.sum() > 30 and contrast > 0:
        ink_vals = P.lum[P.ink]
        if P.polarity == "dark_on_light":
            F["ink_density_mean"] = float((P.substrate_level - ink_vals.mean())
                                          / max(P.substrate_level, 1.0))
        else:
            F["ink_density_mean"] = float((ink_vals.mean() - P.substrate_level)
                                          / max(float(ink_vals.mean()), 1.0))
        F["ink_density_std"] = float(ink_vals.std() / contrast)
        tile = max(4, int(round(ppm)))
        H, W = N.shape
        Hc, Wc = (H // tile) * tile, (W // tile) * tile
        m_t = P.ink[:Hc, :Wc].reshape(Hc // tile, tile, Wc // tile, tile)
        n_t = N[:Hc, :Wc].reshape(Hc // tile, tile, Wc // tile, tile)
        cnt = m_t.sum(axis=(1, 3))
        ssum = (n_t * m_t).sum(axis=(1, 3))
        good = cnt >= 0.3 * tile * tile
        F["ink_mottle"] = float(np.var(ssum[good] / cnt[good])) if good.sum() >= 4 else _nan()
        hp = ink_f - cv2.GaussianBlur(ink_f, (0, 0), 0.25 * ppm / 2.0)
        F["ink_graininess"] = float(hp[P.ink].var())
    mark_in = P.mark & in_code
    holes = ndi.binary_fill_holes(mark_in) & ~mark_in
    lab_h, st_h, _ = _blob_table(holes)
    void_ids = [i for i in range(1, st_h.shape[0])
                if 2 <= st_h[i, cv2.CC_STAT_AREA] <= cfg.void_max_area * mod_a]
    mark_area_mod = float(mark_in.sum()) / mod_a
    if mark_area_mod > 1:
        vprops = _region_props(lab_h, void_ids)
        va = sum(p["area"] for p in vprops)
        F["ink_void_density"] = len(vprops) / mark_area_mod
        F["ink_void_area_fraction"] = float(va / (mark_in.sum() + va))
        el = [p for p in vprops if p["ecc"] > 0.6 and math.isfinite(p["orient"])]
        if len(el) >= 3:
            F["ink_void_orientation"] = _rel_angle(
                _axial_mean_deg(np.array([p["orient"] for p in el])), axis)
    if P.ink.sum() > 100:
        # profile ACROSS the print axis, averaged along it: light lines along the process
        m = P.ink.astype(np.float64)
        red = 1 if axis == 0 else 0
        s_ = (ink_f * m).sum(axis=red)
        c_ = m.sum(axis=red)
        ok = c_ >= max(3.0, 0.2 * c_.max())
        if ok.sum() > 2 * ppm:
            prof = s_[ok] / c_[ok]
            base_ = ndi.median_filter(prof, size=max(3, int(round(ppm))) | 1, mode="nearest")
            F["ink_streak_score"] = float(np.clip((base_ - prof).max(), 0.0, 1.0))

    # --- substrate --------------------------------------------------------------------------
    if P.substrate.sum() > 30:
        F["substrate_brightness"] = float(np.median(P.lum[P.substrate]) / 255.0)
        hp = sub_f - cv2.GaussianBlur(sub_f, (0, 0), 0.25 * ppm / 2.0)
        F["substrate_grain_energy"] = float(hp[P.substrate].var())
        coh_s, ang_s = _st_stats(sub_f, P.substrate, 0.25 * ppm)
        F["substrate_fiber_coherence"] = _f(coh_s)
        F["substrate_fiber_orientation"] = _rel_angle(ang_s, axis)
        far = P.substrate & (ndi.distance_transform_edt(~big) > ppm) if big.any() else P.substrate
        far_mod = float(far.sum()) / mod_a
        if far_mod > 0.5:
            F["substrate_speck_density"] = len(specks) / far_mod
            F["substrate_speck_size_median"] = (float(np.median([p["eqd"] for p in specks])) / ppm
                                                if specks else _nan())

    # --- spatter (pooled over edge classes) ----------------------------------------------
    if E:
        L_tot = sum(E[dn]["length_modules"] for dn in _DIRS)
        if L_tot > 0:
            F["spatter_density_near_edge"] = len(sats) / L_tot
        if len(sats) >= 5:
            # exponential MLE (mean excess over the inner distance limit)
            F["spatter_decay_length"] = float(np.mean([p["dist"] for p in sats])
                                              - cfg.satellite_dist[0])
        if sats:
            F["spatter_size_median"] = float(np.median([p["eqd"] for p in sats])) / ppm

    # --- dots -----------------------------------------------------------------------------
    F["dot_detected_flag"] = 1.0 if dotted else 0.0
    if dotted and len(dprops) >= 5:
        rnd = np.clip(np.array([p["roundness"] for p in dprops]), 0, 1)
        F["dot_roundness_mean"] = float(np.nanmean(rnd))
        F["dot_roundness_std"] = float(np.nanstd(rnd))
        F["dot_diameter"] = float(np.median([p["eqd"] for p in dprops])) / ppm
        pts = np.array([[p["cx"], p["cy"]] for p in dprops])
        dm = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
        np.fill_diagonal(dm, np.inf)
        nn = dm.min(axis=1)
        F["dot_pitch_regularity"] = float(1.0 - nn.std() / (nn.mean() + _EPS))
        a_med = float(np.median([p["area"] for p in dprops]))
        all_areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
        merged = all_areas[all_areas > 1.8 * a_med]
        n_merged = float(np.round(merged / a_med).sum())
        F["dot_merge_fraction"] = n_merged / (n_merged + len(dprops))
        el = [p for p in dprops if p["ecc"] > 0.5 and math.isfinite(p["orient"])]
        if len(el) >= 5:
            F["dot_elongation_orientation"] = _rel_angle(
                _axial_mean_deg(np.array([p["orient"] for p in el])), axis)

    # --- AdaIN (optional) -----------------------------------------------------------------
    if cfg.include_adain:
        for name, v in zip(FEATURE_NAMES[-12:], _adain_block(P, cfg)):
            F[name] = v

    for k in DROPPED_FEATURES:
        F[k] = _nan()
    feats = [(k, _f(F[k])) for k in FEATURE_NAMES]
    if return_debug:
        return feats, {"prepared": P, "grid": G, "edges": E, "classes": cls, "satellites": sats,
                       "specks": specks, "dots": dprops, "axis_cues": cues}
    return feats


# --------------------------------------------------------------------------
# dm_detector integration and tables
# --------------------------------------------------------------------------
def roi_geometry(roi: Any) -> Tuple[Tuple[float, float], Optional[Tuple[float, float]]]:
    """Native code (w, h) and ``module_px_xy`` in the upright (rotation-normalised) crop frame."""
    poly = np.asarray(roi.polygon, dtype=np.float64)
    w = 0.5 * (np.linalg.norm(poly[1] - poly[0]) + np.linalg.norm(poly[2] - poly[3]))
    h = 0.5 * (np.linalg.norm(poly[3] - poly[0]) + np.linalg.norm(poly[2] - poly[1]))
    xy = (float(roi.module_px_xy[0]), float(roi.module_px_xy[1])) if roi.module_px_xy else None
    if roi.rotation in (90, 270):
        w, h = h, w
        if xy:
            xy = (xy[1], xy[0])
    return (float(w), float(h)), xy


def roi_module_px(roi: Any) -> float:
    """Module pitch of a ROI; falls back to side / 20 when the detector has none."""
    return float(roi.module_px) if roi.module_px else float(roi.side_px) / 20.0


def prepare_detected(image: np.ndarray, roi: Any,
                     config: Optional[PrintFeatureConfig] = None) -> PreparedROI:
    """:func:`prepare_roi` for a :class:`dm_detector.DMRoi` (crop + resample + masks)."""
    cfg = config or PrintFeatureConfig()
    crop = d.crop(image, roi, annotate=False, pad=cfg.code_pad)
    size, _ = roi_geometry(roi)
    return prepare_roi(crop, roi_module_px(roi), polarity=roi.polarity, code_size_px=size,
                       config=cfg)


def features_for_roi(image: np.ndarray, roi: Any, *, include_adain: bool = False,
                     config: Optional[PrintFeatureConfig] = None,
                     return_debug: bool = False) -> Any:
    """Feature list for one :class:`dm_detector.DMRoi` of a loaded photo (BGR)."""
    cfg = config or PrintFeatureConfig()
    crop = d.crop(image, roi, annotate=False, pad=cfg.code_pad)
    size, xy = roi_geometry(roi)
    x, y, w, h = roi.bbox
    x8, y8 = (max(0, x) // 8) * 8, (max(0, y) // 8) * 8
    native = image[y8:y + h, x8:x + w]
    return extract_print_features(
        crop, module_px=roi_module_px(roi), angle_deg=roi.angle, polarity=roi.polarity,
        module_px_xy=xy, dotted_hint=bool(roi.dotted_fallback),
        canonical_px_per_module=cfg.canonical_px_per_module, include_adain=include_adain,
        native_crop_for_jpeg_metric=native, native_crop_origin=(x8, y8), code_size_px=size,
        config=cfg, return_debug=return_debug)


def list_images(folder: Union[str, Path]) -> List[Path]:
    """Photos in ``folder`` (HEIC / JPEG / PNG ...), sorted by name."""
    exts = (".heic", ".heif", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in exts)


def _thumb(P: PreparedROI, classes: Dict[str, str], size: int = 150
           ) -> Tuple[np.ndarray, np.ndarray]:
    """(plain canonical crop, mask overlay) thumbnails; edge band coloured by class."""
    x0, y0, x1, y1 = [int(round(v)) for v in P.code_box]
    q = int(round(0.6 * P.ppm))
    H, W = P.lum.shape
    sl = (slice(max(0, y0 - q), min(H, y1 + q)), slice(max(0, x0 - q), min(W, x1 + q)))
    img = P.bgr[sl].copy()
    ov = (0.45 * img).astype(np.uint8)
    ink, sub = P.ink[sl], P.substrate[sl]
    ov[ink] = np.clip(0.5 * ov[ink] + np.array([0, 0, 120]), 0, 255).astype(np.uint8)
    ov[sub] = np.clip(0.5 * ov[sub] + np.array([110, 60, 0]), 0, 255).astype(np.uint8)
    cols = {"leading": (0, 200, 0), "trailing": (0, 0, 220), "orth_left": (200, 200, 0),
            "orth_right": (200, 0, 200)}
    gy, gx = np.gradient(cv2.GaussianBlur(P.markness, (0, 0), 1.5))
    em = P.edge[sl]
    ogx, ogy = -gx[sl], -gy[sl]                       # outward normal of the mark
    horiz = np.abs(ogx) >= np.abs(ogy)
    dir_map = {"-x": em & horiz & (ogx < 0), "+x": em & horiz & (ogx >= 0),
               "-y": em & ~horiz & (ogy < 0), "+y": em & ~horiz & (ogy >= 0)}
    for c, dn in classes.items():
        ov[dir_map[dn]] = cols[c]
    s = size / float(max(img.shape[:2]))
    dims = (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s)))
    return (cv2.resize(img, dims, interpolation=cv2.INTER_AREA),
            cv2.resize(ov, dims, interpolation=cv2.INTER_AREA))


def code_thumbnail(P: PreparedROI, margin_modules: float = 0.6) -> np.ndarray:
    """Canonical crop cut to the code square plus a small margin (BGR)."""
    x0, y0, x1, y1 = [int(round(v)) for v in P.code_box]
    q = int(round(margin_modules * P.ppm))
    H, W = P.lum.shape
    return P.bgr[max(0, y0 - q):min(H, y1 + q), max(0, x0 - q):min(W, x1 + q)].copy()


def features_table(images: Union[str, Path, Sequence[Union[str, Path]]] = "images", *,
                   limit: Optional[int] = None, best_only: bool = False,
                   include_adain: bool = False, max_results: int = 5,
                   config: Optional[PrintFeatureConfig] = None, verbose: bool = False,
                   thumbs: Optional[Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]]] = None,
                   crops_dir: Optional[Union[str, Path]] = None) -> Any:
    """One row per detected code: :data:`TABLE_ID_COLUMNS`, then the 119 features.

    Extra bookkeeping columns: ``_error`` (empty if OK), ``_detect_s`` (photo
    load + detection time, repeated per code) and ``_feature_s`` (extraction).

    Args:
        images: folder or list of photo paths.
        limit: only the first ``limit`` photos (sorted by name).
        best_only: only the best ROI of each photo.
        include_adain: compute the optional AdaIN block (torch + torchvision).
        max_results: maximum ROIs per photo passed to ``dm_detector.detect``.
        thumbs: optional dict that receives ``(image, roi_index) -> (crop, overlay)``.
        crops_dir: optional folder for code thumbnails (canonical pitch, JPEG).
    """
    import pandas as pd
    cfg = config or PrintFeatureConfig()
    if isinstance(images, (str, Path)) and Path(images).is_dir():
        paths = list_images(images)
    elif isinstance(images, (str, Path)):
        paths = [Path(images)]
    else:
        paths = [Path(p) for p in images]
    if limit:
        paths = paths[:limit]
    if crops_dir:
        Path(crops_dir).mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for p in paths:
        t0 = time.perf_counter()
        img = d.load_photo(p)
        rois = d.detect(img, max_results=max_results)
        t1 = time.perf_counter()
        if best_only:
            rois = rois[:1]
        for k, roi in enumerate(rois):
            t2 = time.perf_counter()
            row: Dict[str, Any] = {"image": p.name, "roi_index": k, "score": float(roi.score),
                                   "px_per_module": _f(roi.module_px),
                                   "rotation": float(roi.rotation) if roi.rotation is not None
                                   else np.nan,
                                   "dotted_fallback": bool(roi.dotted_fallback)}
            try:
                feats, dbg = features_for_roi(img, roi, include_adain=include_adain, config=cfg,
                                              return_debug=True)
                row.update(dict(feats))
                row["_error"] = ""
                if thumbs is not None:
                    thumbs[(p.name, k)] = _thumb(dbg["prepared"], dbg["classes"])
                if crops_dir:
                    th = code_thumbnail(dbg["prepared"])
                    s = 256.0 / max(th.shape[:2])
                    th = cv2.resize(th, (max(1, int(th.shape[1] * s)), max(1, int(th.shape[0] * s))),
                                    interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(Path(crops_dir) / ("%s_%d.jpg" % (p.stem, k))), th,
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
            except Exception as exc:  # keep the batch going, record the failure
                row.update({n: np.nan for n in FEATURE_NAMES})
                row["_error"] = "%s: %s" % (type(exc).__name__, exc)
            row["_detect_s"] = t1 - t0
            row["_feature_s"] = time.perf_counter() - t2
            rows.append(row)
            if verbose:
                print("%-16s roi %d  score %.2f  %5.1f px/mod  features %.2fs %s" % (
                    p.name, k, roi.score, row["px_per_module"], row["_feature_s"], row["_error"]),
                    flush=True)
        if verbose and not rois:
            print("%-16s no code detected (%.1fs)" % (p.name, t1 - t0), flush=True)
    cols = list(TABLE_ID_COLUMNS) + list(FEATURE_NAMES)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=cols)
    extra = [c for c in df.columns if c.startswith("_")]
    return df[cols + extra]


# --------------------------------------------------------------------------
# HTML report
# --------------------------------------------------------------------------
def _img_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return ("data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")) if ok else ""


def write_html(df: Any, path: Union[str, Path],
               thumbs: Optional[Dict[Tuple[str, int], Any]] = None,
               title: str = "Print features (Proposal B)") -> None:
    """Readable HTML: per-code table with thumbnails, then a per-feature summary."""
    import pandas as pd
    feats = [c for c in FEATURE_NAMES if c in df.columns]
    summ = pd.DataFrame({
        "#": np.arange(1, len(feats) + 1),
        "valid": [int(df[c].notna().sum()) for c in feats],
        "mean": [df[c].mean() for c in feats],
        "std": [df[c].std() for c in feats],
        "min": [df[c].min() for c in feats],
        "median": [df[c].median() for c in feats],
        "max": [df[c].max() for c in feats],
    }, index=feats)
    css = (
        "body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:18px;color:#222}"
        "h1{font-size:20px} h2{font-size:16px;margin-top:28px}"
        "table{border-collapse:collapse;font-size:11px}"
        "th,td{border:1px solid #ddd;padding:2px 5px;text-align:right;white-space:nowrap}"
        "th{background:#f2f2f2;position:sticky;top:0;z-index:1}"
        "td.l{text-align:left} tr:nth-child(even){background:#fafafa}"
        ".wrap{overflow:auto;max-height:80vh;border:1px solid #ccc}"
        ".nan{color:#bbb} img{display:block}"
        ".legend span{display:inline-block;padding:1px 6px;margin-right:6px;color:#fff;font-size:12px}"
    )
    n_img = df["image"].nunique() if len(df) else 0
    out = ["<!DOCTYPE html><html><head><meta charset='utf-8'><title>%s</title><style>%s</style>"
           "</head><body>" % (html.escape(title), css)]
    out.append("<h1>%s</h1><p>%d codes from %d photos, %d features per code (canonical pitch "
               "24 px/module). Dropped (always NaN): %s. Units and definitions: section 9 of "
               "PrintTypeClusteringLiterature.md and the print_feature.py docstring.</p>" % (
                   html.escape(title), len(df), n_img, len(feats),
                   html.escape(", ".join(DROPPED_FEATURES))))
    out.append("<p class='legend'>Overlay: ink = red tint, substrate = blue tint, edge band by "
               "class: <span style='background:#0a0'>leading</span>"
               "<span style='background:#c00'>trailing</span>"
               "<span style='background:#0aa'>orth_left</span>"
               "<span style='background:#a0a'>orth_right</span></p>")
    out.append("<h2>Per-code feature table</h2><div class='wrap'><table><thead><tr>")
    head = ["crop", "overlay"] + list(TABLE_ID_COLUMNS) + feats
    out.append("".join("<th>%s</th>" % html.escape(h) for h in head) + "</tr></thead><tbody>")
    for _, r in df.iterrows():
        cells = []
        t = thumbs.get((r["image"], int(r["roi_index"]))) if thumbs else None
        for im in (t if t is not None else (None, None)):
            cells.append("<td>%s</td>" % ("<img src='%s'>" % _img_b64(im) if im is not None else ""))
        for c in TABLE_ID_COLUMNS:
            v = r[c]
            txt = "%.3f" % v if isinstance(v, float) and math.isfinite(v) else str(v)
            cells.append("<td class='l'>%s</td>" % html.escape(txt))
        for c in feats:
            v = r[c]
            ok = isinstance(v, (int, float, np.floating)) and math.isfinite(float(v))
            cells.append("<td>%.4g</td>" % float(v) if ok else "<td class='nan'>NaN</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</tbody></table></div>")
    out.append("<h2>Feature summary</h2><div class='wrap'>")
    out.append(summ.to_html(float_format=lambda v: "%.4g" % v, na_rep="NaN", border=0))
    out.append("</div></body></html>")
    Path(path).write_text("\n".join(out), encoding="utf-8")


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def _adain_available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except Exception:
        return False
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the extractor over a folder of photos; write CSV + HTML (see module docstring)."""
    ap = argparse.ArgumentParser(description="Print-characteristic features per Data Matrix code.")
    ap.add_argument("--images", default="images", help="folder with photos (default: images)")
    ap.add_argument("--out", default="out/features", help="output folder (default: out/features)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N photos")
    ap.add_argument("--best-only", action="store_true", help="only the best ROI per photo")
    ap.add_argument("--no-adain", action="store_true",
                    help="skip the optional AdaIN block (rows 108-119 become NaN)")
    ap.add_argument("--no-crops", action="store_true",
                    help="do not save code thumbnails to <out>/crops")
    args = ap.parse_args(argv)
    img_dir, out_dir = Path(args.images), Path(args.out)
    if not img_dir.is_dir():
        ap.error("image folder not found: %s" % img_dir)
    o, i = out_dir.resolve(), img_dir.resolve()
    if o == i or i in o.parents:
        raise SystemExit("refusing to write output into the input image folder: %s" % o)
    out_dir.mkdir(parents=True, exist_ok=True)
    adain = (not args.no_adain) and _adain_available()
    if not args.no_adain and not adain:
        print("torch/torchvision not installed: AdaIN block (rows 108-119) will be NaN")
    thumbs: Dict[Tuple[str, int], Any] = {}
    t0 = time.perf_counter()
    df = features_table(img_dir, limit=args.limit, best_only=args.best_only, include_adain=adain,
                        verbose=True, thumbs=thumbs,
                        crops_dir=None if args.no_crops else out_dir / "crops")
    total = time.perf_counter() - t0
    csv_path, html_path = out_dir / "print_features.csv", out_dir / "print_features.html"
    df.drop(columns=[c for c in ("_detect_s", "_feature_s") if c in df.columns]).to_csv(
        csv_path, index=False)
    write_html(df, html_path, thumbs)
    feats = list(FEATURE_NAMES)
    n = len(df)
    print("\n%d codes from %d photos, %d feature columns (%d dropped: %s), AdaIN %s" % (
        n, df["image"].nunique() if n else 0, len(feats), len(DROPPED_FEATURES),
        ", ".join(DROPPED_FEATURES), "on" if adain else "off"))
    if n:
        print("failed codes: %d" % int(df["_error"].astype(bool).sum()))
        print("time: total %.1fs; load+detect %.2fs/photo; features %.2fs/code mean, %.2fs max" % (
            total, df.groupby("image")["_detect_s"].first().mean(), df["_feature_s"].mean(),
            df["_feature_s"].max()))
        nan_counts = df[feats].isna().sum()
        nz = nan_counts[nan_counts > 0]
        print("features with NaN values (count out of %d codes):" % n)
        for k, v in nz.items():
            print("  %-45s %d" % (k, v))
        import pandas as pd
        with pd.option_context("display.width", 160, "display.max_rows", 200, "display.max_columns", 20,
                               "display.float_format", "{:.4g}".format):
            print(df[feats].describe().T[["count", "mean", "std", "min", "50%", "max"]])
    print("wrote %s and %s" % (csv_path, html_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
