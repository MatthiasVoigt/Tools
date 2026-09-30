#!/usr/bin/env python3
"""dm_detector_build0_02 -- DM detector release, build 0.02 (single self-contained file).

Data Matrix detection + print-type classification (TT, IJ, IJ_PC, LA) for one photo or image array.
Generated 2026-09-30 from the DM_detector pipeline (dm_detector.py -> dm_roi.py, print_feature.py,
print_type_classifier.py): the code paths used for scoring are inlined verbatim, apart from
these documented edits:
* DMRoi: removed to_dict()/to_row() (pipeline table helpers)
* crop_roi: removed the annotate option (only annotate=False is used)
* PrintFeatureConfig: removed min_structure_modules, include_adain, adain_seed (baseline masks, no AdaIN)
* prepare_roi: min-structure mask variant stripped (baseline masks, min_structure_modules=None)
* extract_print_features: removed include_adain / AdaIN block, return_debug, min_structure override
* features_for_roi: dm_detector.crop replaced by the inlined crop_roi with identical arguments; no adain/debug

Model: the print_type_classifier setup (median imputer -> StandardScaler -> balanced multinomial
LogisticRegression, default hyper-parameters), RETRAINED WITHOUT the 12 adain_* features so no
torch is needed; embedded as NumPy constants, predict_proba re-implemented in NumPy.

Codes below 25 px/module are rejected (never upscaled); kept codes are analysed at exactly
25 px/module (INTER_AREA). All codes (up to 5) of a photo are scored; the primary result is the
largest valid code (side length in pixels), per-code details are in ``codes``.

Usage::

    import dm_detector_build0_02 as dm
    r = dm.score_file("IMG_1234.HEIC")          # dict; r["scores"] -> {class: probability}
    s = dm.score_image(bgr_array, as_json=True)  # JSON string
    r = dm.score_file("IMG_1234.HEIC", return_crop=True)
    r["crop"]                                    # primary code crop, RGB uint8 HxWx3 (25 px/module)
    [c.get("crop") for c in r["codes"]]          # crop of every valid code (None / absent otherwise)

    python dm_detector_build0_02.py IMG_1234.HEIC [...]   # JSON to stdout
    python dm_detector_build0_02.py --selftest            # embedded unit tests
    python -m unittest dm_detector_build0_02               # same tests

Dependencies: Python 3.14, numpy, opencv-python (cv2), pillow + pillow-heif (HEIC), scipy
(ndimage) and scikit-image (LBP / GLCM features). Error texts are PROVISIONAL (see ERRORS).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import unittest
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from scipy import ndimage as ndi

BUILD_NUMBER = "0.02"
CLASSES: Tuple[str, ...] = ('TT', 'IJ', 'IJ_PC', 'LA')   # output order; model rows re-ordered to it (verified below)
MIN_PX_PER_MODULE = 25.0
TARGET_PPM = 25.0

# ==========================================================================
# Detection, module size and crop (inlined from dm_roi.py)
# ==========================================================================
VALID_SQUARE_SIZES: Tuple[int, ...] = (
    10, 12, 14, 16, 18, 20, 22, 24, 26, 32, 36, 40, 44, 48, 52,
    64, 72, 80, 88, 96, 104, 120, 132, 144,
)

ImageInput = Union[str, "os.PathLike[str]", np.ndarray]

@dataclass(frozen=True)
class DMConfig:
    """All tunable parameters of the detector (pass overrides as keyword args).

    Sizes given as fractions refer to the long side of the working image.
    """

    #: Long side (px) of the downscaled working copy used for the search.
    work_long_side: int = 1600
    #: Max residual tilt (deg) around each 90-degree step. Candidates whose
    #: edge orientation lies in the forbidden band (max_tilt..90-max_tilt,
    #: modulo 90) are rejected.
    max_tilt_deg: float = 10.0
    #: Smallest / largest accepted code side, as fraction of the long side.
    min_side_frac: float = 0.025
    max_side_frac: float = 0.7
    #: Minimum short/long side ratio of the tight box (codes are square).
    min_squareness: float = 0.78
    #: Box-filter windows (fractions of long side) for the texture map.
    texture_windows: Tuple[float, ...] = (0.01, 0.022, 0.035)
    #: Thresholds relative to the 99.5th percentile of the texture map.
    texture_thresholds: Tuple[float, ...] = (0.2, 0.35, 0.5)
    #: Minimum ratio min(Ex, Ey) / max(Ex, Ey) of axis-aligned gradient
    #: energies for a texture pixel (1D barcodes have ~0, DM ~1).
    min_balance: float = 0.35
    #: Closing kernel sizes (fraction of long side) for binary proposals.
    binary_close_fracs: Tuple[float, ...] = (0.004, 0.008, 0.014)
    #: Closing kernels used during refinement, as fraction of candidate side.
    refine_close_fracs: Tuple[float, ...] = (1 / 40.0, 1 / 24.0, 1 / 14.0)
    #: Candidates are refined on a crop scaled so the code side is <= this.
    refine_side_px: int = 360
    #: Maximum number of proposals refined per image (ranked by texture).
    max_proposals: int = 60
    #: Number of texture peaks used as multi-scale square seeds.
    max_seeds: int = 30
    #: Minimum final score for a candidate to be reported.
    min_score: float = 0.55
    #: Overlap (intersection over the smaller box) above which the weaker
    #: of two candidates is suppressed.
    nms_overlap: float = 0.4
    #: Down-rank candidates lying on a 1D barcode found by cv2.barcode.
    use_barcode_suppression: bool = True
    #: Estimate the module size / grid for every returned ROI.
    estimate_modules: bool = True
    #: Max side (px) of the deskewed full-res crop used for module estimation.
    module_max_side_px: int = 1200


@dataclass
class DMRoi:
    """One detected Data Matrix region (full-resolution pixel coordinates)."""

    polygon: np.ndarray                    #: (4, 2) float32 corners TL, TR, BR, BL
    bbox: Tuple[int, int, int, int]        #: axis-aligned (x, y, w, h) enclosing polygon
    rotation: Optional[int]                #: 0/90/180/270 clockwise, None if unknown
    tilt_deg: float                        #: residual tilt in [-max_tilt, max_tilt]
    score: float                           #: confidence in [0, 1]
    polarity: str                          #: "dark_on_light" or "light_on_dark"
    side_px: float                         #: mean side length of the square
    module_px: Optional[float] = None      #: module pitch (px per module), full res
    module_px_xy: Optional[Tuple[float, float]] = None  #: pitch along code x / y
    grid_n: Optional[int] = None           #: modules per side if confident
    module_conf: float = 0.0               #: 0..1 confidence of the module estimate
    module_method: str = ""                #: "timing", "autocorr", "timing+autocorr"
    details: Dict[str, float] = field(default_factory=dict)  #: sub-scores

    @property
    def angle(self) -> float:
        """Full orientation in degrees: ``rotation + tilt_deg`` (tilt only if unknown)."""
        return float((self.rotation or 0) + self.tilt_deg)

    @property
    def dotted_located(self) -> bool:
        """True if the ROI was located by the dotted-code fallback."""
        return bool(self.details.get("dotted", 0.0) > 0)

    @property
    def dotted_pitch(self) -> bool:
        """True if the module size came from the dot-pitch estimate."""
        return self.module_method == "dot-pitch"

    @property
    def dotted_fallback(self) -> bool:
        """True if the dotted-code fallback located the ROI or measured its pitch."""
        return self.dotted_located or self.dotted_pitch

    @property
    def center(self) -> Tuple[float, float]:
        c = self.polygon.mean(axis=0)
        return float(c[0]), float(c[1])


def _to_bgr_u8(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        if np.issubdtype(a.dtype, np.floating) and a.size and float(np.nanmax(a)) <= 1.0:
            a = a * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    if a.ndim == 3 and a.shape[2] == 1:
        return cv2.cvtColor(a[:, :, 0], cv2.COLOR_GRAY2BGR)
    if a.ndim == 3 and a.shape[2] == 4:
        return cv2.cvtColor(a, cv2.COLOR_BGRA2BGR)
    if a.ndim == 3 and a.shape[2] == 3:
        return np.ascontiguousarray(a)
    raise ValueError("unsupported image shape %r" % (a.shape,))


def load_image(src: ImageInput, max_long_side: Optional[int] = None) -> np.ndarray:
    """Load an image as a BGR ``uint8`` array (EXIF orientation applied).

    Accepts a path to HEIC/HEIF (via pillow-heif), JPEG, PNG, ... or an
    already-loaded numpy array (gray, BGR or BGRA; returned as BGR).
    ``max_long_side`` optionally downsizes early (e.g. for thumbnails).
    """
    if isinstance(src, np.ndarray):
        img = _to_bgr_u8(src)
    else:
        path = os.fspath(src)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        from PIL import Image, ImageOps  # local import: no work at module import

        if path.lower().endswith((".heic", ".heif", ".avif")):
            try:
                import pillow_heif

                pillow_heif.register_heif_opener()
            except ImportError as exc:  # pragma: no cover
                raise ImportError("pillow-heif is required to read HEIC files") from exc
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            if max_long_side is not None and max(im.size) > max_long_side:
                im.thumbnail((max_long_side, max_long_side), Image.Resampling.LANCZOS)
            rgb = np.asarray(im.convert("RGB"))
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return img
    if max_long_side is not None and max(img.shape[:2]) > max_long_side:
        s = max_long_side / float(max(img.shape[:2]))
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img if img.dtype == np.uint8 else _to_bgr_u8(img)[:, :, 1]
    return cv2.cvtColor(_to_bgr_u8(img), cv2.COLOR_BGR2GRAY)


def _odd(v: float, lo: int = 3) -> int:
    k = max(lo, int(round(v)))
    return k if k % 2 == 1 else k + 1


def _runs(b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run-length encode a 1D bool array -> (values, lengths)."""
    b = np.asarray(b, dtype=bool)
    if b.size == 0:
        return np.zeros(0, bool), np.zeros(0, int)
    idx = np.flatnonzero(np.diff(b.astype(np.int8))) + 1
    starts = np.concatenate(([0], idx))
    lengths = np.diff(np.concatenate((starts, [b.size])))
    return b[starts], lengths


def _fill_short(b: np.ndarray, value: bool, max_len: int) -> np.ndarray:
    """Flip interior runs equal to ``value`` that are shorter than ``max_len``."""
    if max_len <= 0:
        return b
    vals, lens = _runs(b)
    out = b.copy()
    pos = 0
    for i, (v, n) in enumerate(zip(vals, lens)):
        if v == value and n < max_len and 0 < i < len(vals) - 1:
            out[pos:pos + n] = not value
        pos += n
    return out


def _rect_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """Intersection area divided by the smaller box area."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = min(ax + aw, bx + bw) - max(ax, bx)
    ih = min(ay + ah, by + bh) - max(ay, by)
    if iw <= 0 or ih <= 0:
        return 0.0
    return float(iw * ih) / max(1e-6, min(aw * ah, bw * bh))


def _polygon_from(center: Tuple[float, float], w: float, h: float, tilt_deg: float) -> np.ndarray:
    """Corners TL, TR, BR, BL of a w x h box rotated clockwise by tilt_deg."""
    c, s = math.cos(math.radians(tilt_deg)), math.sin(math.radians(tilt_deg))
    cx, cy = center
    pts = []
    for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        pts.append((cx + c * dx - s * dy, cy + s * dx + c * dy))
    return np.array(pts, dtype=np.float32)


def _warp_upright(gray: np.ndarray, center: Tuple[float, float], w: float, h: float,
                  tilt_deg: float, pad: float = 0.0, scale: float = 1.0,
                  border_value: int = 0, interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Cut the (w x h, tilted) box plus ``pad`` (fraction of side) out upright."""
    ow = max(4, int(round(w * (1 + 2 * pad) * scale)))
    oh = max(4, int(round(h * (1 + 2 * pad) * scale)))
    M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), float(tilt_deg), float(scale))
    M[0, 2] += ow / 2.0 - center[0]
    M[1, 2] += oh / 2.0 - center[1]
    return cv2.warpAffine(gray, M, (ow, oh), flags=interp, borderMode=cv2.BORDER_REPLICATE)


def _orientation4(gray: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
    """Dominant edge orientation modulo 90 deg via the 4-fold gradient angle.

    Returns (tilt_deg in (-45, 45], 4-fold coherence 0..1, axis balance 0..1).
    Axis balance = min(Ex, Ey)/max(Ex, Ey) of gradient energy measured in the
    rotated (code-aligned) frame; 1D barcodes give ~0, DM codes ~1.
    """
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m2 = gx * gx + gy * gy
    if mask is not None:
        m2 = m2 * (mask > 0)
    th = np.arctan2(gy, gx)
    sw = float(m2.sum()) + 1e-9
    c4 = float((m2 * np.cos(4 * th)).sum())
    s4 = float((m2 * np.sin(4 * th)).sum())
    phi = 0.25 * math.degrees(math.atan2(s4, c4))
    coh = math.hypot(c4, s4) / sw
    # axis balance in the rotated frame
    ang = th - math.radians(phi)
    ex = float((m2 * np.abs(np.cos(ang))).sum())
    ey = float((m2 * np.abs(np.sin(ang))).sum())
    bal = min(ex, ey) / max(ex, ey, 1e-9)
    return phi, coh, bal


_SIDES = ("top", "right", "bottom", "left")

_L_TO_ROTATION = {("left", "bottom"): 0, ("top", "left"): 90, ("right", "top"): 180, ("bottom", "right"): 270}

def _side_profiles(ink: np.ndarray, side: str, max_off: int) -> List[np.ndarray]:
    h, w = ink.shape
    out = []
    for d in range(0, max_off + 1):
        if side == "top":
            p = ink[d:d + 2, :].mean(axis=0)
        elif side == "bottom":
            p = ink[max(0, h - 2 - d):h - d, :].mean(axis=0)
        elif side == "left":
            p = ink[:, d:d + 2].mean(axis=1)
        else:
            p = ink[:, max(0, w - 2 - d):w - d].mean(axis=1)
        out.append(p)
    return out


def _border_analysis(ink: np.ndarray) -> Dict[str, Any]:
    """Analyse the 4 borders of an upright, tight code crop (ink = True).

    A Data Matrix has a solid "L" finder on two adjacent borders and an
    alternating timing pattern (N modules, equal run lengths) on the two
    other borders. Returns per-side solidity / timing regularity / module
    count, the best L hypothesis (-> coarse rotation) and 0..1 scores.
    """
    ink = ink.astype(np.float32)
    h, w = ink.shape
    L = float(min(h, w))
    max_off = max(2, int(round(0.06 * L)))
    gap = max(1, int(round(L / 70.0)))
    speck = max(1, int(round(L / 120.0)))
    trim = max(1, int(round(0.015 * L)))
    per: Dict[str, Dict[str, float]] = {}
    for side in _SIDES:
        best_solid, best_t, best_n = 0.0, 0.0, 0.0
        for p in _side_profiles(ink, side, max_off):
            b = p[trim:len(p) - trim] > 0.5
            if b.size < 8:
                continue
            b = _fill_short(b, False, gap)      # close gaps between dots
            b = _fill_short(b, True, speck)     # drop specks
            f = float(b.mean())
            best_solid = max(best_solid, f)
            vals, lens = _runs(b)
            if len(lens) >= 7:
                mid = lens[1:-1].astype(np.float32)
                med = float(np.median(mid))
                cv = float(np.mean(np.abs(mid - med)) / max(med, 1e-6))
                reg = max(0.0, 1.0 - cv / 0.4) * max(0.0, 1.0 - abs(f - 0.5) / 0.3)
                if reg > best_t:
                    best_t = reg
                    best_n = float(len(p)) / max(med, 1e-6)
        per[side] = {"solid": best_solid, "timing": best_t, "n": best_n}
    hyps = []
    for (a, b), rot in _L_TO_ROTATION.items():
        o1, o2 = [s for s in _SIDES if s not in (a, b)]
        solid = min(per[a]["solid"], per[b]["solid"])
        n1, n2 = per[o1]["n"], per[o2]["n"]
        agree = 1.0 if (n1 > 0 and n2 > 0 and abs(n1 - n2) <= max(1.5, 0.12 * max(n1, n2))) else 0.6
        timing = min(per[o1]["timing"], per[o2]["timing"]) * agree
        timing = 0.7 * timing + 0.3 * 0.5 * (per[o1]["timing"] + per[o2]["timing"])
        not_solid = max(0.0, 1.0 - max(0.0, max(per[o1]["solid"], per[o2]["solid"]) - 0.85) / 0.15)
        fscore = float(np.clip((solid - 0.72) / 0.22, 0.0, 1.0))
        hyps.append((fscore + timing * not_solid, fscore, timing * not_solid, rot, (o1, o2),
                     0.5 * (n1 + n2) if agree == 1.0 else max(n1, n2)))
    hyps.sort(key=lambda t: -t[0])
    best = hyps[0]
    margin = best[0] - hyps[1][0]
    return {"sides": per, "finder": best[1], "timing": best[2], "rotation": best[3],
            "timing_sides": best[4], "margin": margin, "n": best[5]}


def _propose(gw: np.ndarray, cfg: DMConfig) -> List[Tuple[int, int, int, int]]:
    """Axis-aligned candidate boxes on the working image.

    Three complementary sources:
    (1) contours of a balanced 2D-texture map (gradient energy high on BOTH
        image axes -> DM modules; 1D barcodes / long strokes have one axis);
    (2) multi-scale square windows seeded at texture peaks (covers large,
        clean codes that the contour map tends to fragment);
    (3) connected components of adaptively thresholded, morphologically
        closed ink (both polarities; RETR_LIST so codes inside printed
        frames are found).

    Proposals are ranked by mean texture evidence and de-duplicated.
    """
    H, W = gw.shape
    L = max(H, W)
    min_side = cfg.min_side_frac * L
    max_side = cfg.max_side_frac * L
    props: List[Tuple[float, Tuple[int, int, int, int]]] = []

    g = cv2.GaussianBlur(gw, (0, 0), 1.0).astype(np.float32)
    ax = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    ay = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    tex_maps = []
    for wf in cfg.texture_windows:
        k = _odd(wf * L)
        ex = cv2.blur(ax, (k, k))
        ey = cv2.blur(ay, (k, k))
        bal = np.minimum(ex, ey) / (np.maximum(ex, ey) + 1e-3)
        m = 0.5 * (ex + ey) * np.clip((bal - 0.5 * cfg.min_balance) / cfg.min_balance, 0, 1)
        tex_maps.append((k, m))
    ref_map = tex_maps[0][1]
    ref_val = float(np.percentile(ref_map, 99.5)) + 1e-3
    integ = cv2.integral(ref_map / ref_val)

    def add(x: int, y: int, w: int, h: int, loose: float = 0.45, bonus: float = 1.0,
            target: Optional[list] = None) -> None:
        if w < 3 or h < 3:
            return
        if max(w, h) < min_side or max(w, h) > max_side * 1.2:
            return
        if min(w, h) / float(max(w, h)) < loose:
            return
        mean_t = (integ[y + h, x + w] - integ[y, x + w] - integ[y + h, x] + integ[y, x]) / float(w * h)
        sq = min(w, h) / float(max(w, h))
        # texture density x sqrt(size): favours the whole code over a patch of it
        sc = float(mean_t) * (0.4 + 0.6 * sq) * bonus * math.sqrt(max(w, h) / float(L))
        (props if target is None else target).append((sc, (int(x), int(y), int(w), int(h))))

    for k, m in tex_maps:
        ref = float(np.percentile(m, 99.5))
        if ref <= 1e-3:
            continue
        for t in cfg.texture_thresholds:
            b = (m > t * ref).astype(np.uint8)
            b = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
            cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                add(x, y, w, h)

    # Multi-scale square seeds centred on local maxima of the coarse texture
    # map (covers large clean codes that the contour maps fragment or merge
    # with neighbouring text / barcodes).
    coarse = tex_maps[-1][1]
    cref = float(np.percentile(coarse, 99.0)) + 1e-6
    dk = _odd(max(5.0, 0.6 * min_side))
    dil = cv2.dilate(coarse, cv2.getStructuringElement(cv2.MORPH_RECT, (dk, dk)))
    pys, pxs = np.nonzero((coarse >= dil) & (coarse > 0.3 * cref))
    order = np.argsort(-coarse[pys, pxs])[: cfg.max_seeds]
    sides = list(np.geomspace(min_side * 1.1, max_side * 0.9, 12))
    seed_props: List[Tuple[float, Tuple[int, int, int, int]]] = []
    for i in order:
        cx, cy = float(pxs[i]), float(pys[i])
        per_seed: List[Tuple[float, Tuple[int, int, int, int]]] = []
        for side in sides:
            half = 0.5 * side
            x0 = int(round(max(0.0, cx - half))); y0 = int(round(max(0.0, cy - half)))
            x1 = int(round(min(float(W), cx + half))); y1 = int(round(min(float(H), cy + half)))
            add(x0, y0, x1 - x0, y1 - y0, loose=0.6, target=per_seed)
        per_seed.sort(key=lambda t: -t[0])
        seed_props.extend(per_seed[:2])

    blk = _odd(0.04 * L)
    for inv in (False, True):
        src = 255 - gw if inv else gw
        ink = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blk, 10)
        for cf in cfg.binary_close_fracs:
            k = max(2, int(round(cf * L)))
            m = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
            cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                if max(w, h) < min_side:
                    continue
                area = cv2.contourArea(c)
                if area < 0.35 * w * h:
                    continue
                add(x, y, w, h, loose=0.6)

    # Rank by texture evidence; IoU-based de-duplication (a small patch inside
    # a code does not suppress the enclosing square, and vice versa).
    # Contour proposals and seed proposals get separate quotas.
    def dedupe(items: List[Tuple[float, Tuple[int, int, int, int]]], quota: int,
               keep: List[Tuple[int, int, int, int]]) -> None:
        items = sorted(items, key=lambda t: -t[0])
        added = 0
        for _, r in items:
            ra = float(r[2] * r[3])
            dup = False
            for q in keep:
                qa = float(q[2] * q[3])
                inter = _rect_overlap(r, q) * min(ra, qa)
                if inter / max(ra + qa - inter, 1e-6) > 0.5:
                    dup = True
                    break
            if not dup:
                keep.append(r)
                added += 1
            if added >= quota:
                break

    keep: List[Tuple[int, int, int, int]] = []
    dedupe(props, cfg.max_proposals, keep)
    dedupe(seed_props, cfg.max_seeds * 2, keep)
    return keep


def _dotted_cluster(ink_u8: np.ndarray, core_mask: np.ndarray, pitch_hint: float
                     ) -> Optional[Tuple[int, int, int, int]]:
    """Find a dense square cluster of ink dots (inkjet / laser / dot-peen).

    Returns the axis-aligned box of the cluster in crop coordinates, or None.
    """
    h, w = ink_u8.shape
    # isolate individual dots
    n, lab, stats, cents = cv2.connectedComponentsWithStats(ink_u8, connectivity=8)
    if n <= 8:
        return None
    areas = stats[1:, 4]
    if areas.size < 8:
        return None
    med = float(np.median(areas[areas > 0]))
    keep = []
    for i in range(1, n):
        a = stats[i, 4]
        if a < 0.2 * med or a > 6 * med:
            continue
        if core_mask[int(cents[i, 1]), int(cents[i, 0])] <= 0:
            continue
        keep.append(i)
    if len(keep) < 30:
        return None
    pts = cents[keep].astype(np.float32)
    # estimate pitch from nearest-neighbour distances
    from numpy.linalg import norm
    # subsample for speed
    sample = pts if len(pts) <= 400 else pts[np.linspace(0, len(pts) - 1, 400).astype(int)]
    dlist = []
    for p in sample:
        d = np.sqrt(((pts - p) ** 2).sum(axis=1))
        d.sort()
        if d.size >= 3 and d[1] > 1:
            dlist.append(d[1])
    if len(dlist) < 10:
        return None
    pitch = float(np.median(dlist))
    if pitch_hint > 0:
        pitch = 0.5 * (pitch + pitch_hint)
    # density map of dots
    dens = np.zeros((h, w), np.float32)
    for i in keep:
        dens[int(cents[i, 1]), int(cents[i, 0])] = 1.0
    k = max(3, int(round(4 * pitch))) | 1
    dens = cv2.blur(dens, (k, k))
    thr = float(np.percentile(dens[dens > 0], 70)) if (dens > 0).any() else 0
    mask = (dens >= thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = 0.0
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        if max(bw, bh) < 8 * pitch:
            continue
        sq = min(bw, bh) / float(max(bw, bh))
        if sq < 0.75:
            continue
        # count dots inside
        inside = ((pts[:, 0] >= x) & (pts[:, 0] < x + bw) & (pts[:, 1] >= y) & (pts[:, 1] < y + bh)).sum()
        sc = float(inside) * sq
        if sc > best_score:
            best_score = sc
            best = (x, y, bw, bh)
    return best


def _pitch_guess(ink: np.ndarray, default: float) -> float:
    """Rough module pitch of a binary patch (autocorrelation of edge profiles)."""
    f = ink.astype(np.float32)
    h, w = f.shape
    px, sx = _autocorr_period(np.abs(np.diff(f, axis=1)).sum(axis=0), 2.5, w / 5.0)
    py, sy = _autocorr_period(np.abs(np.diff(f, axis=0)).sum(axis=1), 2.5, h / 5.0)
    cands = [(s, p) for p, s in ((px, sx), (py, sy)) if p > 0 and s > 0.05]
    if not cands:
        return default
    if len(cands) == 2 and 0.75 < cands[0][1] / cands[1][1] < 1.33:
        return 0.5 * (cands[0][1] + cands[1][1])
    return max(cands)[1]


def _refine(gray_full: np.ndarray, rect: Tuple[float, float, float, float], cfg: DMConfig
            ) -> Optional[Dict[str, Any]]:
    """Fit a tight tilted square to the code inside/near ``rect`` and score it."""
    H, W = gray_full.shape
    x, y, w, h = rect
    side = max(w, h)
    pad = 0.5 * side + 4
    X0, Y0 = int(max(0, x - pad)), int(max(0, y - pad))
    X1, Y1 = int(min(W, x + w + pad)), int(min(H, y + h + pad))
    if X1 - X0 < 12 or Y1 - Y0 < 12:
        return None
    r = min(1.0, cfg.refine_side_px / float(side))
    crop = gray_full[Y0:Y1, X0:X1]
    if r < 1.0:
        crop = cv2.resize(crop, None, fx=r, fy=r, interpolation=cv2.INTER_AREA)
    ch, cw = crop.shape
    cx0, cy0 = int((x - X0) * r), int((y - Y0) * r)
    cx1, cy1 = int(min(cw, (x + w - X0) * r)), int(min(ch, (y + h - Y0) * r))
    core = crop[cy0:cy1, cx0:cx1]
    if core.size < 64:
        return None
    blur = cv2.GaussianBlur(crop, (0, 0), 0.8)
    t, _ = cv2.threshold(blur[cy0:cy1, cx0:cx1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    core_mask = np.zeros((ch, cw), np.float64)
    core_mask[cy0:cy1, cx0:cx1] = 1
    s_side = side * r
    min_comp = max(10.0, cfg.min_side_frac * max(H, W) * r * 0.7)
    best: Optional[Dict[str, Any]] = None
    tried: List[Tuple[int, int, int, int]] = []
    for polarity in ("dark_on_light", "light_on_dark"):
        ink = (blur < t) if polarity == "dark_on_light" else (blur > t)
        ink_u8 = ink.astype(np.uint8)
        p = _pitch_guess(ink[cy0:cy1, cx0:cx1], s_side / 20.0)
        p = float(np.clip(p, 2.0, s_side / 6.0))
        kb = max(1, int(round(0.6 * p)))
        base = cv2.morphologyEx(ink_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kb, kb)))
        ks = sorted(set(max(2, int(round(v))) for v in (1.3 * p, 2.2 * p, s_side / 14.0)))
        for k in ks:
            m = cv2.morphologyEx(base, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            if n <= 1:
                continue
            ov = np.bincount(lab.ravel(), weights=core_mask.ravel(), minlength=n)
            ov[0] = 0
            order = [i for i in np.argsort(-ov)[:6] if ov[i] > 0]
            for li in order:
                bx, by, bw, bh, area = stats[li]
                touches = int(bx <= 0) + int(by <= 0) + int(bx + bw >= cw) + int(by + bh >= ch)
                if touches >= 2 or max(bw, bh) < min_comp:
                    continue
                if min(bw, bh) < 0.55 * max(bw, bh):
                    continue
                key = (bx // 3, by // 3, bw // 3, bh // 3)
                if key in tried:
                    continue
                tried.append(key)
                comp = (lab[by:by + bh, bx:bx + bw] == li).astype(np.uint8)
                res = _fit_and_verify(blur[by:by + bh, bx:bx + bw], base[by:by + bh, bx:bx + bw],
                                      comp, float(t), polarity, p, cfg)
                if res is None:
                    continue
                res["center"] = (X0 + (bx + res["center"][0]) / r, Y0 + (by + res["center"][1]) / r)
                res["w"] /= r
                res["h"] /= r
                res["pitch"] = p / r
                if best is None or res["score"] > best["score"]:
                    best = res
        # Dotted-module fallback: cluster individual ink dots into a square.
        if best is None or best["score"] < 0.7:
            for polarity in ("dark_on_light", "light_on_dark"):
                ink = ((blur < t) if polarity == "dark_on_light" else (blur > t)).astype(np.uint8)
                # remove very large solid blobs (printed panels) so only dots remain
                n, lab, st, _ = cv2.connectedComponentsWithStats(ink)
                for i in range(1, n):
                    if st[i, 4] > 0.05 * ink.size:
                        ink[lab == i] = 0
                box = _dotted_cluster(ink, core_mask.astype(np.uint8), s_side / 22.0)
                if box is None:
                    continue
                bx, by, bw, bh = box
                # pad a bit then treat the box as a solid component for fitting
                pad2 = max(2, int(round(0.05 * max(bw, bh))))
                bx0, by0 = max(0, bx - pad2), max(0, by - pad2)
                bx1, by1 = min(cw, bx + bw + pad2), min(ch, by + bh + pad2)
                comp = np.ones((by1 - by0, bx1 - bx0), np.uint8)
                # lightly close the dots inside the box for border analysis
                sub = ink[by0:by1, bx0:bx1]
                p = max(2.0, 0.5 * ((bx1 - bx0) / 22.0 + (by1 - by0) / 22.0))
                kb = max(2, int(round(0.7 * p)))
                base = cv2.morphologyEx(sub, cv2.MORPH_CLOSE,
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kb, kb)))
                res = _fit_and_verify(blur[by0:by1, bx0:bx1], base, comp, float(t), polarity, p, cfg)
                if res is None:
                    continue
                res["center"] = (X0 + (bx0 + res["center"][0]) / r, Y0 + (by0 + res["center"][1]) / r)
                res["w"] /= r
                res["h"] /= r
                res["pitch"] = p / r
                res["dotted"] = 1.0
                if best is None or res["score"] > best["score"]:
                    best = res
    return best


def _fit_and_verify(blur: np.ndarray, base: np.ndarray, comp: np.ndarray, t: float,
                    polarity: str, pitch: float, cfg: DMConfig) -> Optional[Dict[str, Any]]:
    """Fit a tilted square to one component and verify Data Matrix structure.

    ``base`` is the ink mask lightly closed (dots merged, module gaps kept).
    """
    ys, xs = np.nonzero(comp)
    if xs.size < 50:
        return None
    # --- residual tilt: 4-fold gradient orientation of the (dot-merged) ink
    dil = cv2.dilate(comp, np.ones((5, 5), np.uint8))
    bf = cv2.GaussianBlur(base.astype(np.float32), (0, 0), 1.0)
    tilt, coh4, _ = _orientation4(bf, dil)
    if coh4 < 0.12:
        rect = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        bp = cv2.boxPoints(rect)
        e = bp[1] - bp[0]
        tilt = ((math.degrees(math.atan2(float(e[1]), float(e[0]))) + 45.0) % 90.0) - 45.0
    # Constraint: square codes, coarse rotation in 90-degree steps, residual
    # tilt of at most +/- max_tilt_deg. Orientations in the forbidden band
    # (max_tilt .. 90-max_tilt, modulo 90) are rejected here.
    if abs(tilt) > cfg.max_tilt_deg:
        return None
    c, s = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))
    mx, my = xs.mean(), ys.mean()
    xr = c * (xs - mx) + s * (ys - my)
    yr = -s * (xs - mx) + c * (ys - my)
    # Trim appendages (adjacent text / barcode touching the code through thin
    # bridges): keep the longest run of well-filled columns / rows in the
    # code-aligned frame.
    for _ in range(2):
        keep = np.ones(xr.size, bool)
        for arr in (xr, yr):
            lo = float(arr.min())
            hist = np.bincount(np.round(arr - lo).astype(int)).astype(np.float32)
            if hist.size < 8:
                continue
            hist = np.convolve(hist, np.ones(3, np.float32) / 3.0, mode="same")
            ref = float(np.percentile(hist, 80))
            good = hist > 0.4 * ref
            vals, lens = _runs(good)
            starts = np.concatenate(([0], np.cumsum(lens)[:-1]))
            cand = [(n, st) for v, n, st in zip(vals, lens, starts) if v]
            if not cand:
                continue
            n, st = max(cand)
            if n < 0.97 * hist.size:
                idx = np.round(arr - lo).astype(int)
                keep &= (idx >= st) & (idx < st + n)
        if keep.all() or keep.sum() < 50:
            break
        xr, yr = xr[keep], yr[keep]
    x0, x1 = np.percentile(xr, [0.1, 99.9])
    y0, y1 = np.percentile(yr, [0.1, 99.9])
    bw, bh = float(x1 - x0 + 1), float(y1 - y0 + 1)
    squareness = min(bw, bh) / max(bw, bh)
    if squareness < cfg.min_squareness:
        return None
    fill = xr.size / (bw * bh)
    if fill < 0.45:
        return None
    ccx, ccy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    center = (mx + c * ccx - s * ccy, my + s * ccx + c * ccy)
    # verification runs on an upright copy of at most ~220 px side (speed)
    vs = min(1.0, 220.0 / max(bw, bh))
    up = _warp_upright(blur, center, bw, bh, tilt, scale=vs)
    upb = _warp_upright(base.astype(np.uint8) * 255, center, bw, bh, tilt, scale=vs) > 127
    pitch = pitch * vs
    # Extra close for dotted (inkjet / laser) modules so the L finder reads solid.
    kdot = max(2, int(round(0.9 * pitch)))
    upb_dot = cv2.morphologyEx(upb.astype(np.uint8), cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kdot, kdot))) > 0
    ink = (up < t) if polarity == "dark_on_light" else (up > t)
    ba = _border_analysis(upb_dot)
    for alt in (upb, ink):
        if ba["finder"] + ba["timing"] > 1.7:
            break
        ba2 = _border_analysis(alt)
        if ba2["finder"] + ba2["timing"] > ba["finder"] + ba["timing"]:
            ba = ba2
    inkf = float(upb.mean())
    tr = 0.5 * (np.abs(np.diff(upb.astype(np.int8), axis=0)).sum(axis=0).mean()
                + np.abs(np.diff(upb.astype(np.int8), axis=1)).sum(axis=1).mean())
    _, _, bal = _orientation4(cv2.GaussianBlur(upb.astype(np.float32), (0, 0), 1.0))
    dens = float(np.clip((tr - 2.0) / 6.0, 0, 1))
    inkscore = float(np.clip(1.0 - abs(inkf - 0.5) / 0.35, 0, 1))
    balscore = float(np.clip((bal - 0.4) / 0.4, 0, 1))
    sqscore = float(np.clip((squareness - cfg.min_squareness) / (1 - cfg.min_squareness), 0, 1))
    score = (0.30 * ba["finder"] + 0.34 * ba["timing"] + 0.12 * dens + 0.10 * balscore
             + 0.06 * inkscore + 0.08 * sqscore)
    if ba["timing"] < 0.15 and ba["finder"] < 0.5:
        score *= 0.7
    elif ba["finder"] >= 0.9 and dens >= 0.4 and ba["timing"] < 0.25:
        # dotted codes: solid L, dense interior, weak timing readout
        score = max(score, 0.55 * ba["finder"] + 0.25 * dens + 0.2 * sqscore)
    cont = float(abs(up[ink].mean() - up[~ink].mean())) if 0 < ink.sum() < ink.size else 0.0
    if cont < 25:
        score *= max(0.3, cont / 25.0)
    return {"center": center, "w": bw, "h": bh, "tilt": tilt, "polarity": polarity,
            "score": float(score), "rotation": ba["rotation"], "finder": ba["finder"],
            "timing": ba["timing"], "margin": ba["margin"], "n_timing": ba["n"], "density": dens,
            "balance": bal, "ink": inkf, "squareness": squareness, "fill": fill, "coh4": coh4,
            "contrast": cont}


def _barcode_boxes(img_small: np.ndarray) -> List[Tuple[float, float, float, float]]:
    if not hasattr(cv2, "barcode"):
        return []
    try:
        det = cv2.barcode.BarcodeDetector()
        ok, pts = det.detectMulti(img_small)
    except cv2.error:
        return []
    if not ok or pts is None:
        return []
    out = []
    for q in np.asarray(pts).reshape(-1, 4, 2):
        x, y, w, h = cv2.boundingRect(q.astype(np.float32))
        if w * h > 0 and max(w, h) / max(1.0, min(w, h)) > 1.2:
            out.append((float(x), float(y), float(w), float(h)))
    return out


def _make_config(config: Optional[DMConfig], params: Dict[str, Any]) -> DMConfig:
    cfg = config if config is not None else DMConfig()
    if params:
        unknown = set(params) - set(DMConfig.__dataclass_fields__)
        if unknown:
            raise TypeError("unknown DMConfig parameter(s): %s" % ", ".join(sorted(unknown)))
        cfg = replace(cfg, **params)
    return cfg


def find_dm_rois(image: ImageInput, *, max_results: int = 5, debug: bool = False,
                 config: Optional[DMConfig] = None, **params: Any) -> List[DMRoi]:
    """Find Data Matrix code regions, ranked by confidence (best first).

    Args:
        image: path (HEIC/JPEG/PNG...) or numpy image (BGR, BGRA or gray).
        max_results: maximum number of ROIs returned (codes may occur several
            times per image).
        debug: if True, every ROI's ``details`` dict contains all sub-scores
            and ``details['n_proposals']`` etc.; rejected candidates are not
            returned.
        config: a :class:`DMConfig`; keyword ``params`` override its fields,
            e.g. ``find_dm_rois(img, max_tilt_deg=8, min_score=0.5)``.

    Returns:
        list of :class:`DMRoi` in full-resolution coordinates.
    """
    cfg = _make_config(config, params)
    img = load_image(image)
    gray = _gray(img)
    H, W = gray.shape
    s = min(1.0, cfg.work_long_side / float(max(H, W)))
    gw = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    t0 = time.perf_counter()
    props = _propose(gw, cfg)
    bars = _barcode_boxes(gw) if cfg.use_barcode_suppression else []
    cands: List[Dict[str, Any]] = []
    for (x, y, w, h) in props:
        res = _refine(gray, (x / s, y / s, w / s, h / s), cfg)
        if res is None:
            continue
        poly = _polygon_from(res["center"], res["w"], res["h"], res["tilt"])
        bx, by, bw, bh = cv2.boundingRect(poly)
        res["poly"] = poly
        res["bbox"] = (int(bx), int(by), int(bw), int(bh))
        side = 0.5 * (res["w"] + res["h"])
        if side < cfg.min_side_frac * max(H, W) * 0.8:
            continue
        sb = (bx * s, by * s, bw * s, bh * s)
        on_bar = max([_rect_overlap(sb, bb) for bb in bars] + [0.0])
        if on_bar > 0.6 and res["finder"] < 0.8:
            res["score"] *= 0.4
        res["on_barcode"] = on_bar
        cands.append(res)
    # Non-maximum suppression (overlap relative to the smaller box). Candidates
    # with a verified timing pattern go first, so a tight code wins over a
    # solid label patch / quiet-zone box that merely contains it.
    cands.sort(key=lambda d: (-(1 if d["timing"] >= 0.35 else 0), -d["score"]))
    kept: List[Dict[str, Any]] = []
    for c in cands:
        # Soft gate: weak timing needs stronger evidence. Exception: a clear
        # L finder on a near-square, balanced, high-density patch (dotted codes
        # often have a solid L after closing but a weak timing score).
        if c["timing"] >= 0.25:
            thr = cfg.min_score
        elif (c["finder"] >= 0.9 and c.get("density", 0) >= 0.4 and c.get("squareness", 0) >= 0.88
              and c.get("on_barcode", 0) < 0.3 and c.get("fill", 0) >= 0.45):
            thr = cfg.min_score
        else:
            thr = max(cfg.min_score, 0.72)
        if c["score"] < thr:
            continue
        # Without a readable timing pattern, reject 1D barcodes (one dominant
        # gradient axis / overlap with a cv2.barcode detection) and
        # low-contrast halftone / print-screen textures.
        if c["timing"] < 0.25 and (c.get("balance", 1.0) < 0.5 or c.get("on_barcode", 0.0) > 0.5
                                   or c.get("contrast", 255.0) < 75.0):
            continue
        # reject solid panels / stickers mistaken for codes
        if c["timing"] < 0.25 and (
            max(c["w"], c["h"]) > 0.4 * max(H, W)
            or (not c.get("dotted") and (c.get("ink", 0.5) < 0.2 or c.get("ink", 0.5) > 0.8
                                         or (c.get("fill", 0) > 0.97 and c.get("density", 0) < 0.5)))
        ):
            continue
        if any(_rect_overlap(c["bbox"], k["bbox"]) > cfg.nms_overlap for k in kept):
            continue
        kept.append(c)
        if len(kept) >= max_results:
            break
    kept.sort(key=lambda d: -d["score"])
    rois: List[DMRoi] = []
    for c in kept:
        rot = c["rotation"] if (c["finder"] >= 0.5 and c["timing"] >= 0.25 and c["margin"] > 0.15) else None
        det_keys = ("finder", "timing", "density", "balance", "ink", "squareness", "fill",
                    "coh4", "contrast", "on_barcode", "margin", "n_timing", "pitch")
        details = {k: float(c[k]) for k in det_keys} if debug else {
            "finder": float(c["finder"]), "timing": float(c["timing"])}
        # 1.0 when the ROI was located by the dotted-code fallback (_dotted_cluster)
        details["dotted"] = float(c.get("dotted", 0.0))
        if debug:
            details["n_proposals"] = float(len(props))
            details["detect_s"] = time.perf_counter() - t0
        roi = DMRoi(polygon=c["poly"], bbox=c["bbox"], rotation=rot, tilt_deg=float(c["tilt"]),
                    score=float(min(1.0, c["score"])), polarity=c["polarity"],
                    side_px=float(0.5 * (c["w"] + c["h"])), details=details)
        if cfg.estimate_modules:
            _apply_module_estimate(gray, roi, cfg)
        rois.append(roi)
    return rois


def _autocorr_period(profile: np.ndarray, lo: float, hi: float) -> Tuple[float, float]:
    """Fundamental period of a 1D profile via autocorrelation -> (period, strength)."""
    p = profile.astype(np.float64) - float(profile.mean())
    n = p.size
    if n < 8:
        return 0.0, 0.0
    f = np.fft.rfft(p, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    if ac[0] <= 0:
        return 0.0, 0.0
    ac = ac / ac[0]
    lo_i, hi_i = max(2, int(math.floor(lo))), min(n - 2, int(math.ceil(hi)))
    if hi_i <= lo_i + 1:
        return 0.0, 0.0
    peaks = [i for i in range(lo_i, hi_i + 1) if ac[i] >= ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > 0]
    if not peaks:
        return 0.0, 0.0
    top = max(ac[i] for i in peaks)
    for i in peaks:  # smallest lag with a strong peak = fundamental
        if ac[i] >= 0.6 * top:
            y0, y1, y2 = ac[i - 1], ac[i], ac[i + 1]
            den = y0 - 2 * y1 + y2
            off = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
            return float(i + np.clip(off, -0.5, 0.5)), float(ac[i])
    return 0.0, 0.0


def _snap_n(n: float, tol: float = 1.2) -> Optional[int]:
    best = min(VALID_SQUARE_SIZES, key=lambda v: abs(v - n))
    return best if abs(best - n) <= tol else None


def _dot_pitch(ink: np.ndarray) -> Optional[float]:
    """Median nearest-neighbour distance of dot-like blobs (None if not dotted)."""
    n, lab, st, cents = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    if n < 41:
        return None
    areas = st[1:, 4].astype(np.float64)
    med = float(np.median(areas))
    h, w = ink.shape
    if med <= 3 or med > (min(h, w) / 8.0) ** 2:
        return None
    sel = (areas > 0.3 * med) & (areas < 3.0 * med)
    # dots must dominate: most blobs of similar size, compact bounding boxes
    if sel.sum() < 40 or sel.mean() < 0.5:
        return None
    bw, bh = st[1:, 2][sel].astype(np.float64), st[1:, 3][sel].astype(np.float64)
    if float(np.median(np.minimum(bw, bh) / np.maximum(bw, bh))) < 0.6:
        return None
    pts = cents[1:][sel]
    if len(pts) > 600:
        pts = pts[np.linspace(0, len(pts) - 1, 600).astype(int)]
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    return float(np.median(nn))


def estimate_module_size(img: ImageInput, roi: DMRoi, config: Optional[DMConfig] = None
                         ) -> Dict[str, Any]:
    """Estimate the module pitch of a detected code in FULL-RES pixels/module.

    The ROI is deskewed at full resolution, binarised (Otsu, polarity aware)
    and analysed in two ways: (1) the fundamental period of the gradient
    projection profiles along both code axes (autocorrelation), and (2) the
    run lengths along the alternating timing borders. If the timing count is
    consistent it is snapped to a valid square DM size N and the pitch is
    side / N.

    Returns a dict with ``module_px``, ``module_px_xy`` (x, y), ``grid_n``
    (or None), ``module_conf`` (0..1) and ``module_method``.
    """
    cfg = config or DMConfig()
    gray = _gray(load_image(img)) if not (isinstance(img, np.ndarray) and img.ndim == 2) else img
    poly = np.asarray(roi.polygon, dtype=np.float64)
    wpx = 0.5 * (np.linalg.norm(poly[1] - poly[0]) + np.linalg.norm(poly[2] - poly[3]))
    hpx = 0.5 * (np.linalg.norm(poly[3] - poly[0]) + np.linalg.norm(poly[2] - poly[1]))
    side = max(wpx, hpx)
    sc = min(1.0, cfg.module_max_side_px / side) if side > 0 else 1.0
    sc = max(sc, min(4.0, 240.0 / max(side, 1.0)))  # upsample tiny codes
    ctr = poly.mean(axis=0)
    up = _warp_upright(gray, (float(ctr[0]), float(ctr[1])), wpx, hpx, roi.tilt_deg, scale=sc,
                       interp=cv2.INTER_CUBIC)
    upb = cv2.GaussianBlur(up, (0, 0), max(0.5, 0.004 * up.shape[0]))
    t, _ = cv2.threshold(upb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (upb < t) if roi.polarity == "dark_on_light" else (upb > t)
    uh, uw = ink.shape
    rot_hint = roi.rotation
    best: Optional[Dict[str, Any]] = None
    # Several closing sizes: 1 % of the side suits solid prints; 2-4 % merges
    # the dots of inkjet / laser / dot-peen modules (pitch, not dot diameter).
    for kf in (0.01, 0.02, 0.035):
        k = max(1, int(round(kf * min(uh, uw))))
        closed = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
        # (1) autocorrelation of gradient projection profiles along both axes
        cf = cv2.GaussianBlur(closed.astype(np.float32), (0, 0), 0.7)
        prof_x = np.abs(np.diff(cf, axis=1)).sum(axis=0)
        prof_y = np.abs(np.diff(cf, axis=0)).sum(axis=1)
        px, sx = _autocorr_period(prof_x, max(2.0, uw / 150.0, 1.6 * k), uw / 8.0)
        py, sy = _autocorr_period(prof_y, max(2.0, uh / 150.0, 1.6 * k), uh / 8.0)
        # (2) timing borders (N = modules along the alternating edges)
        ba = _border_analysis(closed.astype(bool))
        rot = rot_hint if rot_hint is not None else ba["rotation"]
        tsides = [s2 for s2 in _SIDES if s2 not in [k2 for k2, v in _L_TO_ROTATION.items() if v == rot][0]]
        ns = [ba["sides"][s2]["n"] for s2 in tsides if ba["sides"][s2]["timing"] > 0.2]
        n_t = float(np.mean(ns)) if ns else 0.0
        n_ac = ([uw / px] if px > 0 else []) + ([uh / py] if py > 0 else [])
        n_a = float(np.mean(n_ac)) if n_ac else 0.0
        grid_n: Optional[int] = None
        method, conf = "", 0.0
        agree_t = len(ns) == 2 and abs(ns[0] - ns[1]) <= max(1.5, 0.08 * n_t)
        if n_t > 0 and n_a > 0 and abs(n_t - n_a) <= max(1.5, 0.1 * n_t):
            grid_n = _snap_n(0.5 * (n_t + n_a), 1.5)
            method, conf = "timing+autocorr", (0.95 if agree_t else 0.8)
        elif n_t > 0 and agree_t:
            grid_n = _snap_n(n_t, 1.2)
            method, conf = "timing", 0.6
        if grid_n is not None:
            mx, my = wpx / grid_n, hpx / grid_n
        elif n_a > 0:
            mx = (px / sc) if px > 0 else (py / sc)
            my = (py / sc) if py > 0 else (px / sc)
            method, conf = "autocorr", float(np.clip(0.3 + 0.5 * min(sx or sy, sy or sx), 0, 0.6))
            if n_a < 8:  # fewer than 8 modules per side is not a DM
                conf *= 0.5
        else:
            continue
        cand = {"mx": mx, "my": my, "grid_n": grid_n, "conf": conf, "method": method,
                "n_t": n_t, "n_a": n_a}
        if best is None or cand["conf"] > best["conf"] + 1e-6:
            best = cand
    if best is None:
        return {"module_px": None, "module_px_xy": None, "grid_n": None,
                "module_conf": 0.0, "module_method": "none"}
    mx, my, grid_n, conf, method = best["mx"], best["my"], best["grid_n"], best["conf"], best["method"]
    n_t, n_a = best["n_t"], best["n_a"]
    if grid_n is None:
        # Dotted modules (inkjet / laser / dot-peen): the median nearest-
        # neighbour distance between dot centres is the module pitch.
        dp = _dot_pitch(ink)
        if dp is not None:
            n_d = min(uh, uw) / dp
            if 8 <= n_d <= 150:
                snapped = _snap_n(n_d, 0.6)
                if snapped is not None:
                    grid_n = snapped
                    mx, my = wpx / snapped, hpx / snapped
                    method, conf = "dot-pitch", 0.7
                else:
                    mx = my = dp / sc
                    method, conf = "dot-pitch", 0.5
    return {"module_px": float(0.5 * (mx + my)), "module_px_xy": (float(mx), float(my)),
            "grid_n": grid_n, "module_conf": float(conf), "module_method": method,
            "n_timing": n_t, "n_autocorr": n_a}


def _apply_module_estimate(gray: np.ndarray, roi: DMRoi, cfg: DMConfig) -> None:
    try:
        est = estimate_module_size(gray, roi, cfg)
    except Exception:  # never let the estimate break detection
        return
    roi.module_px = est["module_px"]
    roi.module_px_xy = est["module_px_xy"]
    roi.grid_n = est["grid_n"]
    roi.module_conf = est["module_conf"]
    roi.module_method = est["module_method"]


def crop_roi(img: ImageInput, roi: DMRoi, pad: float = 0.2, deskew: bool = True,
             normalize_rotation: bool = True, out_size: Optional[int] = None) -> np.ndarray:
    """Cut a ROI out of the full-resolution image.

    Args:
        pad: padding around the code as a fraction of its side (0.2 = 20 %).
        deskew: undo the small tilt (warpAffine to an upright square). If
            False, an axis-aligned padded bbox crop is returned.
        normalize_rotation: with ``deskew``, additionally rotate by 90-degree
            steps (np.rot90) so the L finder ends up left+bottom, if known.
        out_size: optionally resize the (square) result to this side length.
    """
    im = load_image(img)
    if deskew:
        poly = np.asarray(roi.polygon, dtype=np.float64)
        wpx = 0.5 * (np.linalg.norm(poly[1] - poly[0]) + np.linalg.norm(poly[2] - poly[3]))
        hpx = 0.5 * (np.linalg.norm(poly[3] - poly[0]) + np.linalg.norm(poly[2] - poly[1]))
        s = max(wpx, hpx)
        c = poly.mean(axis=0)
        out = _warp_upright(im, (float(c[0]), float(c[1])), s, s, roi.tilt_deg, pad=pad,
                            interp=cv2.INTER_CUBIC)
        if normalize_rotation and roi.rotation:
            out = np.ascontiguousarray(np.rot90(out, k=roi.rotation // 90))
    else:
        x, y, w, h = roi.bbox
        p = int(round(pad * max(w, h)))
        H, W = im.shape[:2]
        out = im[max(0, y - p):min(H, y + h + p), max(0, x - p):min(W, x + w + p)].copy()
    if out_size:
        interp = cv2.INTER_AREA if max(out.shape[:2]) > out_size else cv2.INTER_NEAREST
        out = cv2.resize(out, (out_size, int(round(out_size * out.shape[0] / max(out.shape[1], 1)))),
                         interpolation=interp)
    return out


# ==========================================================================
# Print features at 25 px/module, baseline masks (inlined from print_feature.py)
# ==========================================================================
FeatureTuple = Tuple[str, float]


EDGE_CLASSES: Tuple[str, ...] = ("leading", "trailing", "orth_left", "orth_right")


_ORIENTS: Tuple[str, ...] = ("000", "045", "090", "135")


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

DROPPED_FEATURES: Tuple[str, ...] = ("edge_bias_lead_trail_diff",)


_EPS = 1e-9


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
    #: Optional edge-band width (modules); None = baseline 2 * region_erode_modules.
    edge_band_modules: Optional[float] = None
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
    mask_extra: Dict[str, Any] = {"mask_method": "baseline"}
    # (release: baseline masks only; the min-structure variant is stripped)

    r = cfg.region_erode_modules * ppm
    if cfg.edge_band_modules is not None:
        r = 0.5 * float(cfg.edge_band_modules) * ppm
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
                              "sm": sm, "resample_factor": f, **mask_extra})


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


def _axis_vote(cues: Sequence[Tuple[float, float]]) -> Tuple[int, float]:
    """cues: (value in [-1, 1] with + = code x, weight) -> (axis, confidence)."""
    num = sum(w * v for v, w in cues if math.isfinite(v))
    den = sum(w for v, w in cues if math.isfinite(v))
    if den <= 0:
        return 0, 0.0
    return (0 if num >= 0 else 1), float(min(1.0, abs(num) / den))


def _rel_angle(img_deg: float, axis: int) -> float:
    return _wrap90(img_deg - (0.0 if axis == 0 else 90.0))


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


def extract_print_features(
    roi_bgr_or_gray: np.ndarray,
    *,
    module_px: float,
    angle_deg: float = 0.0,
    polarity: Optional[str] = None,
    module_px_xy: Optional[Tuple[float, float]] = None,
    dotted_hint: Optional[bool] = None,
    canonical_px_per_module: float = 24.0,
    native_crop_for_jpeg_metric: Optional[np.ndarray] = None,
    native_crop_origin: Tuple[int, int] = (0, 0),
    code_size_px: Optional[Tuple[float, float]] = None,
    config: Optional[PrintFeatureConfig] = None,
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
        native_crop_for_jpeg_metric: native un-deskewed crop for ``jpeg_blockiness``.
        native_crop_origin: (x, y) of that crop in the photo (8x8 lattice alignment).
        code_size_px: native (w, h) of the code square in ``roi_bgr_or_gray``.
        config: :class:`PrintFeatureConfig` overrides.

    Returns:
        ``[(name, value), ...]`` in :data:`FEATURE_NAMES` order (release: the 12
        adain_* values are always NaN).
    """
    base = config or PrintFeatureConfig()
    cfg = PrintFeatureConfig(**{**base.__dict__, "canonical_px_per_module": canonical_px_per_module})
    P = prepare_roi(roi_bgr_or_gray, module_px, polarity=polarity, code_size_px=code_size_px,
                    config=cfg)
    ppm = P.ppm
    F: Dict[str, float] = {k: _nan() for k in FEATURE_NAMES}
    N = P.markness
    contrast = abs(P.ink_level - P.substrate_level)
    in_code = P.extra["in_code"]
    # structure-measuring features (dots, satellites / specks, contour, voids) use the
    # RAW mark; with the min-structure mask variant P.mark is the cleaned mark
    mark_s = P.extra.get("mark_raw", P.mark)

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
    lab, stats, _ = _blob_table(mark_s & in_code)
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
    lab_all, st_all, _ = _blob_table(mark_s & P.region)
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
    contour = mark_s & ~cv2.erode(mark_s.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
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
    mark_in = mark_s & in_code
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

    # (release: AdaIN block stripped -> adain_* stay NaN; not used by the model)

    for k in DROPPED_FEATURES:
        F[k] = _nan()
    feats = [(k, _f(F[k])) for k in FEATURE_NAMES]
    return feats


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


def features_for_roi(image: np.ndarray, roi: Any, *,
                     config: Optional[PrintFeatureConfig] = None) -> Any:
    """Feature list for one :class:`dm_detector.DMRoi` of a loaded photo (BGR)."""
    cfg = config or PrintFeatureConfig()
    # == dm_detector.crop(image, roi, annotate=False, pad=cfg.code_pad)
    crop = crop_roi(image, roi, pad=cfg.code_pad, deskew=True, normalize_rotation=True)
    size, xy = roi_geometry(roi)
    x, y, w, h = roi.bbox
    x8, y8 = (max(0, x) // 8) * 8, (max(0, y) // 8) * 8
    native = image[y8:y + h, x8:x + w]
    return extract_print_features(
        crop, module_px=roi_module_px(roi), angle_deg=roi.angle, polarity=roi.polarity,
        module_px_xy=xy, dotted_hint=bool(roi.dotted_fallback),
        canonical_px_per_module=cfg.canonical_px_per_module,
        native_crop_for_jpeg_metric=native, native_crop_origin=(x8, y8), code_size_px=size,
        config=cfg)


# ==========================================================================
# Embedded model: balanced multinomial LogisticRegression, 107 non-adain features
# (trained for build 0.01, reused unchanged in build 0.02; trained on the 527 codes (TT 193, IJ 99, IJ_PC 61, LA 174) of 'Data/catalog modified/features.csv';
#  model_noadain.joblib SHA-256 1fb6992135c3cc38417adfe621901aa0589744b0423149e3453715b4c620b2ed, sklearn 1.9.1)
# block-grouped CV macro-F1 0.9490, photo-grouped CV macro-F1 0.9817
# ==========================================================================
MODEL_SHA256 = '1fb6992135c3cc38417adfe621901aa0589744b0423149e3453715b4c620b2ed'
#: class order of the trained sklearn model; coefficient rows below are re-ordered to CLASSES
SKLEARN_CLASS_ORDER: Tuple[str, ...] = ('IJ', 'IJ_PC', 'LA', 'TT')
MODEL_CLASSES: Tuple[str, ...] = ('TT', 'IJ', 'IJ_PC', 'LA')   # row order of _COEF / _INTERCEPT
MODEL_FEATURES: Tuple[str, ...] = (
    'capture_px_per_module',
    'resample_factor',
    'upsampled_flag',
    'module_pitch_anisotropy',
    'sharpness_score',
    'glare_fraction',
    'jpeg_blockiness',
    'bimodality_separability',
    'mark_polarity',
    'print_axis',
    'print_axis_confidence',
    'print_polarity_confidence',
    'edge_raggedness_leading',
    'edge_raggedness_trailing',
    'edge_raggedness_orth_left',
    'edge_raggedness_orth_right',
    'edge_blur_width_leading',
    'edge_blur_width_trailing',
    'edge_blur_width_orth_left',
    'edge_blur_width_orth_right',
    'edge_bias_leading',
    'edge_bias_trailing',
    'edge_bias_orth_left',
    'edge_bias_orth_right',
    'edge_profile_skew_leading',
    'edge_profile_skew_trailing',
    'edge_profile_skew_orth_left',
    'edge_profile_skew_orth_right',
    'edge_satellite_density_leading',
    'edge_satellite_density_trailing',
    'edge_satellite_density_orth_left',
    'edge_satellite_density_orth_right',
    'edge_raggedness_lead_trail_log2ratio',
    'edge_blur_lead_trail_log2ratio',
    'edge_bias_lead_trail_diff',
    'edge_satellite_lead_trail_log2ratio',
    'edge_raggedness_process_vs_orth_log2ratio',
    'edge_blur_process_vs_orth_log2ratio',
    'corner_rounding_radius',
    'st_coherence_ink',
    'st_orientation_ink',
    'st_coherence_edge',
    'gabor_ink_s1_000',
    'gabor_ink_s1_045',
    'gabor_ink_s1_090',
    'gabor_ink_s1_135',
    'gabor_ink_s2_000',
    'gabor_ink_s2_045',
    'gabor_ink_s2_090',
    'gabor_ink_s2_135',
    'gabor_ink_orientation_of_max',
    'gabor_ink_anisotropy',
    'gabor_substrate_orientation_of_max',
    'gabor_substrate_anisotropy',
    'banding_peak_process',
    'banding_peak_cross',
    'banding_period_process',
    'lbp_riu2_ink_entropy',
    'lbp_riu2_ink_uniform_fraction',
    'lbp_var_ink',
    'lbp_rv_ink_process_vs_cross',
    'clbp_m_ink_mean',
    'lbp_riu2_substrate_entropy',
    'lbp_riu2_substrate_uniform_fraction',
    'lbp_var_substrate',
    'lbp_rv_substrate_process_vs_cross',
    'lbp_riu2_edge_entropy',
    'ldp_edge_direction_entropy',
    'hog_contour_orientation_entropy',
    'glcm_ink_contrast_d1',
    'glcm_ink_contrast_d2',
    'glcm_ink_homogeneity_d1',
    'glcm_ink_energy_d1',
    'glcm_ink_correlation_d1',
    'glcm_ink_correlation_d2',
    'glcm_ink_contrast_along_vs_across_log2ratio',
    'glcm_substrate_contrast_d1',
    'glcm_substrate_contrast_d2',
    'glcm_substrate_homogeneity_d1',
    'glcm_substrate_energy_d1',
    'glcm_substrate_correlation_d1',
    'glcm_substrate_correlation_d2',
    'glcm_substrate_contrast_along_vs_across_log2ratio',
    'ink_density_mean',
    'ink_density_std',
    'ink_mottle',
    'ink_graininess',
    'ink_void_density',
    'ink_void_area_fraction',
    'ink_void_orientation',
    'ink_streak_score',
    'substrate_brightness',
    'substrate_grain_energy',
    'substrate_fiber_coherence',
    'substrate_fiber_orientation',
    'substrate_speck_density',
    'substrate_speck_size_median',
    'spatter_density_near_edge',
    'spatter_decay_length',
    'spatter_size_median',
    'dot_detected_flag',
    'dot_roundness_mean',
    'dot_roundness_std',
    'dot_diameter',
    'dot_pitch_regularity',
    'dot_merge_fraction',
    'dot_elongation_orientation',
)
_IMPUTER_STATISTICS = np.array([
    50.48772309346173, 0.4951698842453355, 0.0, 1.007185652014785,
    0.5399470156488924, 0.0, 0.9919622838497162, 0.9303608536720276,
    0.0, 0.0, 0.1478204625019724, 0.2995796758559433,
    0.0344475723431109, 0.0559300888772783, 0.0293200825141487, 0.027873392296151052,
    0.1427548556976591, 0.15563437854471565, 0.1286281948985422, 0.1246954379537676,
    -0.0107462744381152, -0.0107462744381152, -0.0185625983596068, -0.0185625983596068,
    -0.4957975048141311, 0.8339269645027159, 0.1506803737301388, 0.1419351123924819,
    0.0, 0.0, 0.0, 0.0,
    0.0443666101515013, 0.11143811271692375, 0.0, 0.0,
    0.7204519117761025, 0.16466284415280663, 0.2782458468470934, 0.2895653247833252,
    -0.4258545028855423, 0.5514100193977356, 8.114107185974717e-05, 8.927669114200398e-05,
    0.0001313860702794, 9.956023859558628e-05, 8.129830530378968e-05, 7.808126247255132e-05,
    0.0001268256164621, 8.273038838524371e-05, 90.0, 0.1387238380886092,
    90.0, 0.1300946021099684, 0.1799235967106583, 0.134051408648418,
    1.8076923076923075, 0.9455968622150656, 0.8733927169218688, -7.648513307635516,
    0.334877708921571, 0.0191260278224945, 0.9549633392513311, 0.8502205470356355,
    -9.053704354496505, 0.1712032199334904, 0.8983863406585634, 0.97807271981245,
    0.9578148665564395, 3.411589724976579, 5.4922155628795775, 0.5474187885822399,
    0.0396134711273744, 0.4883706980661098, 0.2107691919683779, -0.2031195481978146,
    0.9192158093917636, 1.351264633381123, 0.7144771678694554, 0.1043860867191008,
    0.5982797282306923, 0.3938118819086378, -0.080298573351269, 0.8744479417800903,
    0.0675992891192436, 0.0004810553711364, 0.0016607460565865, 0.0799430599108763,
    0.0014007434295867, 0.8359859288633658, 0.0602814336482628, 0.6745098233222961,
    0.0003809886693488, 0.2792032361030578, 0.8895191152734583, 0.0,
    0.1276615297284584, 0.0014204545454545, 0.1152723459705871, 0.1276615297284584,
    0.0, 0.9508103227024391, 0.0414964479365793, 0.8634865820595659,
    0.8092675882575943, 0.1010452961672473, 2.491285833929396,
], dtype=np.float64)

_SCALER_MEAN = np.array([
    58.541391622027554, 0.5392410075813155, 0.0, 1.0130871129637564,
    0.5249846022031163, 0.02176283177007956, 1.004012680935905, 0.9197066418360035,
    0.3567362428842505, 0.33396584440227706, 0.1671755726376752, 0.3037464883738265,
    0.0534379281951678, 0.058708382543635786, 0.04103900664589061, 0.03918241936708156,
    0.15745483484587272, 0.1741281526606663, 0.14376239795360218, 0.14315502715884176,
    -0.014954345729735512, -0.014954345729735512, -0.02829274589027841, -0.02829274589027841,
    -0.4789673924142938, 0.8353272326660286, 0.21369902441942995, 0.07272591972096146,
    0.04540202772053477, 0.07700349167520057, 0.0366678814462925, 0.06506174688148189,
    0.07717861919812093, 0.12277891285594847, 0.0, 0.19813489165690526,
    0.6410360996314336, 0.2058314809949304, 0.28513912955216913, 0.3068304673487593,
    -1.0190055333573573, 0.5298498305553956, 0.00020336339783119585, 0.0005720293272691918,
    0.0002879016313028966, 0.0002426832453086055, 0.0001507558781972175, 0.0001536024591605551,
    0.00023627519138705934, 0.0001485413190416114, 81.97343453510436, 0.18621062547881592,
    62.93168880455408, 0.14576609904500648, 0.21895571185014567, 0.18482288845269232,
    2.9223945930921036, 0.9384581828298116, 0.8717711246806614, -7.57912080537837,
    0.3736471955613625, 0.026923959046925603, 0.9508502723831711, 0.8492613999279265,
    -8.957874196635839, 0.1904719158436546, 0.8801381794438465, 0.972381051974812,
    0.9234554752623935, 10.1352060369703, 14.029256346145152, 0.5347375985442868,
    0.0710519681298102, 0.4693435591691206, 0.1958656104634274, -0.30777111579925837,
    1.4762257449365361, 2.074312105088626, 0.6903396757928864, 0.11221139217238105,
    0.5743774419951607, 0.3898712433621928, -0.08661509528545872, 0.8306926309610001,
    0.07897429699272593, 0.0008088458199187051, 0.004608527437996459, 0.3094883098612973,
    0.007455014798748545, -3.114042747997276, 0.07721483351468583, 0.5022361233788039,
    0.000636785000821793, 0.28568591016293476, 2.615923231338705, 0.07159157885858398,
    0.12819882693536264, 0.055842234381746146, 0.12645132137363468, 0.13189246331181795,
    0.10056925996204934, 0.9495177039409582, 0.04310753759177253, 0.8551717369301154,
    0.8055067331451329, 0.1149489005224904, 2.780427506630178,
], dtype=np.float64)

_SCALER_SCALE = np.array([
    29.98690023716086, 0.24490836934157365, 1.0, 0.07004492878304716,
    0.10942241991773512, 0.06629073935809332, 0.04668660963295036, 0.03834399622762138,
    0.47903600688996184, 0.47162767006925194, 0.11133215645073168, 0.17470141377094678,
    0.037591001858534215, 0.03937599913837224, 0.03258980233286882, 0.031711769128228984,
    0.058201699136179195, 0.07098105635631395, 0.05627574419631568, 0.056323626562705724,
    0.057737687723261384, 0.057737687723261384, 0.05201803261250891, 0.05201803261250891,
    1.1342085958001915, 0.9566207301293655, 1.431621936394096, 1.4617704467744104,
    0.44539299822441714, 0.7090843356592981, 0.1982553741583697, 0.6226315937248229,
    0.7046172225087595, 0.40642744114109175, 1.0, 0.40135470854696526,
    0.8524453649578638, 0.35469699209016897, 0.09972648644550769, 0.08248662745471816,
    29.335967560327614, 0.09735156394228643, 0.00035593724916966154, 0.00218456464835504,
    0.0004404523001190701, 0.0006776391074375295, 0.00018888403609904422, 0.00018946409098184733,
    0.0002890158527343158, 0.00018261558373928983, 31.193757831858914, 0.15613548712682887,
    43.75833974829765, 0.08404909107967634, 0.13422673650039968, 0.12674855057736048,
    2.2285930049203495, 0.026973178206241434, 0.04155086348768292, 1.3541546074588005,
    0.5440606419268817, 0.024818390906991578, 0.013384578024000945, 0.0334664519667695,
    0.8985588824242006, 0.5417136079150863, 0.053432323472371354, 0.02378487990505693,
    0.07545370687584381, 20.259772376332922, 23.26627524585373, 0.19207661378660093,
    0.07769975669549728, 0.19435389976214368, 0.1995501132913769, 0.41492423335448597,
    1.7611752509547136, 2.2322137331644374, 0.10936379352822889, 0.05941762129574871,
    0.14694827599538102, 0.16476507062216583, 0.28025665486212514, 0.12913992641792763,
    0.04904995769976381, 0.0009818966958451796, 0.008257357659071245, 0.6464927631049503,
    0.012475310160035854, 27.990859879118954, 0.05742640591866049, 0.31964540224680904,
    0.0007930582452244458, 0.052303140103098264, 44.98690054782327, 0.5434220720565833,
    0.009787313940554637, 0.49013908341510426, 0.06287270788777344, 0.023911202859057467,
    0.300757516801717, 0.014682953192380378, 0.013838267249743637, 0.057331315028134,
    0.03359113444079231, 0.08851546124437133, 3.8415604750903696,
], dtype=np.float64)

_COEF = np.array([
    [-0.02400077933332244, -0.00422184520080863, 0.0, 0.2233466892709253, -0.24978906527452144, 0.06581725948548561, 0.07005615437265472, 0.14866153341290028, 0.32842516206633365, 0.23306361425108524, -0.028842573862682644, -0.05665171475827637, -0.32865433289853413, -0.44372541595916815, -0.2580179378471694, -0.2168513967443779, 0.08386888465575047, -0.05127572944437338, -0.2687783621580713, -0.05437243784866168, -0.05062554269014944, -0.05062554269014944, 0.14125942914693662, 0.14125942914693662, 0.08407964197879146, 0.1536337367608163, 0.004257572474777157, -0.030473034510224904, -0.021918543033113393, 0.010694234340509343, -0.08905857842423763, -0.019448618570920416, -0.23349007797661947, 0.0257788692714662, 0.0, -0.038142421525481675, 0.07457287009675832, 0.24226633041367626, -0.3510937076929284, -0.06624397713189875, 0.010928360955881553, 0.2858433651389415, -0.08935330580498432, 0.0009177949177446412, -0.08231524761282404, -0.0012214459725978157, -0.17359981294173105, -0.1657650690547507, -0.07027217801890917, 0.07180342473458437, 0.10550647600498125, -0.11208606212538774, 0.1217115453287788, 0.29124761316903586, 0.027367057610276302, -0.3257001998612891, 0.1295113522730811, -0.07122679617206015, -0.25094323982901023, -0.009917324779146038, 0.25974358424934607, -0.09600803792157028, -0.1488462602895661, -0.2788954074047688, 0.10689146167856296, 0.11665526681506032, 0.060021135608378726, -0.6029977209829687, -0.5959374308063308, -0.021340110721276777, -0.03111757064557508, 0.3900964378643075, 0.6372595560219222, -0.3606551657389772, -0.27581975519472735, -0.21191820483676865, 0.05372941617245984, 0.12016213204313646, 0.036064954884149646, 0.16468552151368113, -0.10536007064971732, 0.1185549223552085, 0.06713299175485425, 1.131484578368475, -0.0004899229913860617, -0.03861407503049666, -0.028533654578633477, 0.09602853826812831, 0.11861465414235874, -0.15083356138888626, -0.46681865657266913, 0.17226512294170804, 0.12914573268115137, 0.229310398578374, 0.06635953976852205, -0.09790585078754988, 0.0005683092049541795, -0.02184146182187734, 0.2155825337955283, 0.1743955188857847, -0.13337928675310243, -0.03039536971593587, 0.02805372091021367, 0.07802603606238227, -0.013108889255263706, 0.012200746288213939, 0.018395472466765296],
    [0.13097760220423546, -0.18433199334710224, 0.0, -0.16975821761414897, 0.3301718129724388, -0.14992686047148504, 0.03036412301352174, -0.06458585926930276, -0.8586190498932535, -0.3033201459864926, 0.036654953298590504, 0.10503157173817977, -0.036681689698719946, 0.09680900146194832, 0.021086587657961578, -0.11394289090220397, 0.16264995649551509, 0.12687177656543502, 0.1919776481024114, 0.042894005849060966, 0.1753769126057614, 0.1753769126057614, 0.004780257612836677, 0.004780257612836677, -0.1620560494359363, -0.17165175003496114, -0.0549122883230596, -0.0993090059486222, -0.00857116173654993, -0.007601495600124049, -0.012365552942993092, -0.018310992413019644, 0.3422284335045069, -0.11088997302279789, 0.0, 0.20483641888042298, -0.15371290736683715, -0.14606885450527288, 0.039995604796051534, -0.04183115655747108, 0.027302914197413902, -0.07576660223234964, -0.0897445562432745, -0.1397870344794446, 0.02644683209960765, -0.14065181752567638, -0.027341186099540206, -0.04481964725381562, 0.38958711063856377, -0.1197331235091578, -0.09268931729865891, 0.26112894056519703, 0.14885568094406132, -0.19351900723801005, -0.28651273605238536, 0.03805231733090445, 0.3142282945766077, 0.19114728630439995, 0.16994460844528092, -0.13952498688711334, -0.18768212187590314, -0.1149139197878091, 0.2707718245178661, 0.37997004518951916, -0.04386993096104269, -0.11333511849585304, 0.11106477852241516, 0.14237252432810374, 0.27063136793765885, -0.1332151992455249, -0.14984331412932636, -0.0769767850790751, -0.4616938837815915, 0.4873945934370048, 0.4736284292841192, -0.0482813230925568, -0.1512270359710831, -0.1439773190826857, -0.12465639515106683, -0.3429836354832756, 0.22395078182262645, -0.1054822610442645, -0.17757141956364614, -1.0970998722144838, -0.2184794212358642, -0.2919811899654315, -0.15020321261704572, -0.2803784957949977, -0.39003960198252363, 0.17569395606873126, 0.5938086412356484, 0.2135057452747274, -0.1906369836607366, -0.25657367770092554, -0.006139413929328779, -0.020442008076190402, -0.0017032225596737543, -0.010607256528869095, -0.1503184938874589, -0.09797501282983719, -0.05574125071608017, 0.016358601779964482, -0.017971574774769333, 0.04676767287580268, 0.021583497039192934, -0.020104532069595404, -0.004407478117574434],
    [-0.04737291373461631, 0.10243617511459865, 0.0, -0.0199723487212245, -0.01305784858437851, -0.10452337343277365, 0.009314127234113333, -0.08803545974058902, -0.06080105098129387, 0.05677056254572011, -0.04050707895403542, 0.07737781017216827, -0.14456002684735478, -0.057230082210009255, -0.014376450845409068, -0.04348811595491521, -0.2384601048971828, -0.11277293246926716, -0.045086433468990474, -0.07720483236692402, -0.09525658896018067, -0.09525658896018067, -0.11645621028859071, -0.11645621028859071, 0.10289817556315181, 0.1850496066537503, 0.13690749495861834, 0.12933408138381589, -0.003606105531525587, -0.009721042498466158, -0.017447527971567914, -0.0007294718221030572, 0.048843244858782164, 0.09379347138523644, 0.0, -0.12134547126730702, -0.020955426984780888, -0.06025426549483869, -0.13090401185023426, 0.10104787973797644, -0.04746397518221149, -0.09442189034395113, 0.23392464704501797, 0.15008520423409993, 0.16739964351255962, 0.18132789502949104, 0.3469002816658673, 0.31661778510374816, -0.0991190703715076, 0.25250202124058774, -0.08649573691237145, -0.10836932913830594, -0.3010864239407703, -0.01765107201628428, 0.04800800458673741, 0.017319452937904514, -0.1969454073877669, -0.08832172634293078, 0.013775855378468644, 0.16121368837553782, -0.05464727777567783, 0.22353873476024916, -0.1522427216723315, 0.05671856229675635, 0.01856709178853607, 0.01530166719206717, -0.009444621626787523, 0.14761559121119613, 0.1103027369351064, 0.19197749614547452, 0.22635157352604846, -0.2149917996052756, -0.06523781371552885, -0.10915874002256234, -0.10261108119937878, 0.22373104558110338, 0.0823041854206933, 0.08838517945770276, -0.00042027576194791455, -0.022986075793450883, 0.04272080173351963, 0.06351548443530208, 0.02043370440548301, -0.16042040751391343, 0.26829890333779977, 0.24673414094728877, 0.22171366264520553, 0.24562395137471119, 0.2288155277014354, 0.05445870199173863, -0.07225325385519318, 0.14345239884889877, 0.09329586016720386, 0.14950193234313863, -0.0978423310993558, -0.0014629807692442417, 0.0013356971810914212, -0.0059232605695249034, -0.1145940010370637, 0.04714022076678158, -0.005022400018350744, -0.015062222253705895, 0.006992102435278321, -0.006267052547313614, -0.01796045765914038, 0.008821799010705511, -0.0005058879435147644],
    [-0.05960390913629728, 0.08611766343331229, 0.0, -0.033616122935554535, -0.0673248991135423, 0.18863297441877355, -0.1097344046202873, 0.003959785596989468, 0.5909949388082166, 0.013485969189687148, 0.03269469951812608, -0.1257576671520781, 0.5098960494446128, 0.4041464967072316, 0.2513078010346204, 0.37428240360149867, -0.008058736254080606, 0.037176885348205534, 0.12188714752465436, 0.08868326436652663, -0.02949478095543177, -0.02949478095543177, -0.02958347647118474, -0.02958347647118474, -0.02492176810600456, -0.1670315933796074, -0.08625277911033337, 0.00044795907502990937, 0.03409581030118638, 0.006628303758078093, 0.11887165933879694, 0.03848908280604006, -0.15758160038667252, -0.008682367633905624, 0.0, -0.04534852608763922, 0.1000954642548618, -0.03594321041356618, 0.44200211474711326, 0.0070272539513941525, 0.009232700028919993, -0.11565487256264227, -0.0548267849967587, -0.011215964672398845, -0.11153122799934324, -0.039454631531214486, -0.14595928262459448, -0.1060330687951797, -0.2201958622481451, -0.2045723224660115, 0.07367857820605188, -0.040673549301502535, 0.0305191976679289, -0.08007753391474416, 0.21113767385537166, 0.27032842959248043, -0.24679423946192616, -0.03159876378940884, 0.06722277600526096, -0.011771376709276474, -0.01741418459776859, -0.01261677705086868, 0.030317157444029946, -0.15779320008150652, -0.08158862250605303, -0.018621815511272847, -0.16164129250400974, 0.3130096054436718, 0.21500332593356855, -0.03742218617867257, -0.045390688751144624, -0.09812785317995953, -0.11032785852480441, -0.01758068767546656, -0.09519759289001639, 0.036468482348224085, 0.015193434377934011, -0.06456999241814933, 0.08901171602886164, 0.20128418976304152, -0.16131151290642937, -0.07658814574624756, 0.09000472340330767, 0.1260357013599251, -0.04932955911054836, 0.08386112404863878, -0.04297679544952554, -0.061273993847841456, 0.042609420138731195, -0.07931909667158527, -0.05473673080778424, -0.5292232670653372, -0.03180460918761433, -0.1222386532205904, 0.03762220526015969, 0.11981083963298667, -0.00020078382637323916, 0.038371978920268605, 0.04932996112899565, -0.12356072682272629, 0.19414293748753264, 0.029098990189678493, -0.01707424857072223, -0.11852665639086862, 0.009485849875214112, -0.0009180132293252548, -0.013482106405676564],
], dtype=np.float64)

_INTERCEPT = np.array([
    1.3809637161820483, 0.23886939700812349, -2.0589875211993887, 0.439154408009214,
], dtype=np.float64)

assert MODEL_CLASSES == CLASSES and sorted(SKLEARN_CLASS_ORDER) == sorted(CLASSES)
assert _COEF.shape == (len(CLASSES), len(MODEL_FEATURES))


# ==========================================================================
# Release API (build-specific; not taken from the pipeline sources)
# ==========================================================================
#: PROVISIONAL error codes and texts -- the final conditions and wording are still to be
#: decided by Matthias. Callers should key on the code (dict key), not on the text.
ERRORS: Dict[str, str] = {  # PROVISIONAL
    "E_FILE_NOT_FOUND": "PROVISIONAL: file not found",
    "E_UNREADABLE_IMAGE": "PROVISIONAL: file could not be read as an image",
    "E_INVALID_ARRAY": "PROVISIONAL: invalid image array (expected uint8 gray HxW, BGR/RGB HxWx3 or BGRA/RGBA HxWx4)",
    "E_NO_CODE_DETECTED": "PROVISIONAL: no Data Matrix code detected",
    "E_RESOLUTION_TOO_LOW": "PROVISIONAL: resolution too low (all detected codes below 25 px/module; not upscaled, no guess)",
    "E_FEATURE_EXTRACTION_FAILED": "PROVISIONAL: print feature extraction failed for every usable code",
    "E_INTERNAL": "PROVISIONAL: internal error",
}
STATUS_OK = "ok"
MAX_CODES_PER_IMAGE = 5          # dm_detector.detect default, as in print_type_classifier
_COLOR_ORDERS = ("BGR", "RGB", "BGRA", "RGBA", "GRAY")


def get_build_number() -> str:
    """Build number of this release file (a string, e.g. "0.02")."""
    return BUILD_NUMBER


def get_version() -> str:
    """Alias of :func:`get_build_number`."""
    return get_build_number()


def predict_proba(X: np.ndarray) -> np.ndarray:
    """NumPy re-implementation of the embedded sklearn pipeline (median imputer ->
    StandardScaler -> multinomial LogisticRegression). ``X``: (n, 107) in MODEL_FEATURES order.
    Returns (n, 4) probabilities in CLASSES order."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if X.shape[1] != len(MODEL_FEATURES):
        raise ValueError("expected %d features, got %d" % (len(MODEL_FEATURES), X.shape[1]))
    X = np.where(np.isnan(X), _IMPUTER_STATISTICS[None, :], X)
    Z = (X - _SCALER_MEAN[None, :]) / _SCALER_SCALE[None, :]
    logits = Z @ _COEF.T + _INTERCEPT[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def get_model_parameters() -> Dict[str, Any]:
    """Copies of the embedded model constants (class order, feature names, imputer values,
    scaler mean/scale, coefficients, intercepts, SHA-256 of the source joblib)."""
    return {"classes": tuple(CLASSES), "feature_names": tuple(MODEL_FEATURES),
            "imputer_statistics": _IMPUTER_STATISTICS.copy(), "scaler_mean": _SCALER_MEAN.copy(),
            "scaler_scale": _SCALER_SCALE.copy(), "coef": _COEF.copy(), "intercept": _INTERCEPT.copy(),
            "model_sha256": MODEL_SHA256, "min_px_per_module": MIN_PX_PER_MODULE, "target_ppm": TARGET_PPM}


def model_features_for_roi(image_bgr: np.ndarray, roi: DMRoi) -> Dict[str, float]:
    """The 107 model features (non-adain print features at 25 px/module) of one detected code."""
    with warnings.catch_warnings():   # e.g. "Mean of empty slice" from np.nanmean on all-NaN pairs
        warnings.simplefilter("ignore", category=RuntimeWarning)
        feats = features_for_roi(image_bgr, roi, config=PrintFeatureConfig(canonical_px_per_module=TARGET_PPM))
    d_ = dict(feats)
    return {n: float(d_[n]) for n in MODEL_FEATURES}


def normalized_crop(image_bgr: np.ndarray, roi: DMRoi) -> np.ndarray:
    """Crop of one code resampled to exactly 25 px/module with INTER_AREA (never upscaled);
    identical to the PNGs written by print_type_classifier build."""
    mpx = roi_module_px(roi)
    f = TARGET_PPM / mpx
    if f > 1.0:
        raise ValueError("refusing to upscale (module %.2f px < %.1f)" % (mpx, TARGET_PPM))
    crop = crop_roi(image_bgr, roi, pad=PrintFeatureConfig().code_pad)
    H0, W0 = crop.shape[:2]
    W, H = max(8, int(round(W0 * f))), max(8, int(round(H0 * f)))
    return cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA)


def code_crop_rgb(image_bgr: np.ndarray, roi: DMRoi) -> np.ndarray:
    """``return_crop`` image of one code: the classifier's crop (deskewed, pad 0.2 modules-relative as
    in print_type_classifier build) resampled to exactly 25 px/module with INTER_AREA, i.e. the same
    pixels as :func:`normalized_crop` / the PNGs in 'catalog modified', converted to RGB uint8 (HxWx3)."""
    crop = normalized_crop(image_bgr, roi)
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(crop[:, :, :3][:, :, ::-1]).astype(np.uint8, copy=False)


def _nan_scores() -> np.ndarray:
    return np.full(len(CLASSES), np.nan)


def _result(status: str = STATUS_OK, error: Optional[str] = None, **kw: Any) -> Dict[str, Any]:
    r: Dict[str, Any] = {"build": BUILD_NUMBER, "status": status,
                         "error": error if error is not None else (ERRORS.get(status) if status != STATUS_OK else None),
                         "scores": None, "scores_array": _nan_scores(), "predicted_class": None,
                         "primary_code_index": None, "scores_mean_all_codes": None,
                         "n_codes": 0, "n_valid_codes": 0, "codes": []}
    r.update(kw)
    return r


def _to_jsonable(o: Any) -> Any:
    if isinstance(o, dict):   # "crop" images (return_crop=True) are never put into JSON
        return {str(k): _to_jsonable(v) for k, v in o.items() if k != "crop"}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_to_jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return v if math.isfinite(v) else None
    return o


def to_json(result: Dict[str, Any], indent: Optional[int] = None) -> str:
    """Result dict -> JSON string (arrays as lists, NaN/inf as null)."""
    return json.dumps(_to_jsonable(result), indent=indent, allow_nan=False)


def _check_array(arr: Any, color_order: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """-> (BGR uint8 image, None) or (None, error message)."""
    if not isinstance(arr, np.ndarray):
        return None, "%s (got %s)" % (ERRORS["E_INVALID_ARRAY"], type(arr).__name__)
    if arr.dtype != np.uint8:
        return None, "%s (dtype %s)" % (ERRORS["E_INVALID_ARRAY"], arr.dtype)
    co = str(color_order).upper()
    if co not in _COLOR_ORDERS:
        return None, "%s (unknown color_order %r)" % (ERRORS["E_INVALID_ARRAY"], color_order)
    a = arr
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    if a.ndim == 2:
        bgr = cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_GRAY2BGR)
    elif a.ndim == 3 and a.shape[2] == 3:
        bgr = np.ascontiguousarray(a[:, :, ::-1] if co in ("RGB", "RGBA") else a)
    elif a.ndim == 3 and a.shape[2] == 4:
        bgr = cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGBA2BGR if co in ("RGB", "RGBA") else cv2.COLOR_BGRA2BGR)
    else:
        return None, "%s (shape %r)" % (ERRORS["E_INVALID_ARRAY"], tuple(a.shape))
    if min(bgr.shape[:2]) < 16:
        return None, "%s (image too small: %r)" % (ERRORS["E_INVALID_ARRAY"], tuple(arr.shape))
    return bgr, None


def _score_bgr(bgr: np.ndarray, return_features: bool = False, return_crop: bool = False) -> Dict[str, Any]:
    try:
        rois = find_dm_rois(bgr, max_results=MAX_CODES_PER_IMAGE)
    except Exception as exc:  # unexpected
        return _result("E_INTERNAL", "%s (detection: %s: %s)" % (ERRORS["E_INTERNAL"], type(exc).__name__, exc))
    if not rois:
        return _result("E_NO_CODE_DETECTED")
    codes: List[Dict[str, Any]] = []
    for k, roi in enumerate(rois, 1):
        mpx = roi_module_px(roi)
        c: Dict[str, Any] = {
            "index": k, "px_per_module": float(mpx),
            "module_px_detector": None if roi.module_px is None else float(roi.module_px),
            "bbox": [int(v) for v in roi.bbox],
            "polygon": [[float(x), float(y)] for x, y in roi.polygon],
            "side_px": float(roi.side_px), "detector_score": float(roi.score),
            "rotation": roi.rotation, "tilt_deg": float(roi.tilt_deg), "polarity": roi.polarity,
            "grid_n": roi.grid_n, "module_method": roi.module_method,
            "status": STATUS_OK, "error": None, "scores": None, "scores_array": _nan_scores(),
            "predicted_class": None}
        if mpx < MIN_PX_PER_MODULE:
            c["status"] = "E_RESOLUTION_TOO_LOW"
            c["error"] = "%s (%.2f px/module)" % (ERRORS["E_RESOLUTION_TOO_LOW"], mpx)
        else:
            try:
                feats = model_features_for_roi(bgr, roi)
                x = np.array([feats[n] for n in MODEL_FEATURES], dtype=np.float64)
                p = predict_proba(x)[0]
                c["scores"] = {cl: float(v) for cl, v in zip(CLASSES, p)}
                c["scores_array"] = p
                c["predicted_class"] = CLASSES[int(np.argmax(p))]
                if return_features:
                    c["features"] = feats
                if return_crop:
                    try:
                        c["crop"] = code_crop_rgb(bgr, roi)
                    except Exception:   # a crop failure never changes the scores
                        c["crop"] = None
            except Exception as exc:
                c["status"] = "E_FEATURE_EXTRACTION_FAILED"
                c["error"] = "%s (%s: %s)" % (ERRORS["E_FEATURE_EXTRACTION_FAILED"], type(exc).__name__, exc)
        codes.append(c)
    ok = [c for c in codes if c["status"] == STATUS_OK]
    common = {"codes": codes, "n_codes": len(codes), "n_valid_codes": len(ok)}
    if return_crop:
        common["crop"] = None
    if not ok:
        if any(c["status"] == "E_FEATURE_EXTRACTION_FAILED" for c in codes):
            return _result("E_FEATURE_EXTRACTION_FAILED", **common)
        return _result("E_RESOLUTION_TOO_LOW", **common)
    # primary = largest valid code (side length in image pixels); ties -> higher px/module, lower index
    prim = max(ok, key=lambda c: (c["side_px"], c["px_per_module"], -c["index"]))
    mean = np.mean([c["scores_array"] for c in ok], axis=0)
    if return_crop:
        common["crop"] = prim.get("crop")
    return _result(STATUS_OK, None, scores=dict(prim["scores"]), scores_array=prim["scores_array"].copy(),
                   predicted_class=prim["predicted_class"], primary_code_index=prim["index"],
                   scores_mean_all_codes={cl: float(v) for cl, v in zip(CLASSES, mean)}, **common)


def score_image(array: Any, color_order: str = "BGR", as_json: bool = False,
                return_features: bool = False, return_crop: bool = False) -> Union[Dict[str, Any], str]:
    """Score a uint8 image array (gray HxW, BGR/RGB HxWx3, BGRA/RGBA HxWx4).

    ``return_crop=True`` adds ``result["crop"]`` (primary code) and ``codes[i]["crop"]`` (every valid
    code): the normalised 25 px/module code crop as an RGB uint8 array (see :func:`code_crop_rgb`),
    ``None`` on error / for invalid codes. Scores are identical; crops are never included in JSON.

    ``color_order`` names the channel order of 3/4-channel input ("BGR" = OpenCV default,
    "RGB" = PIL/matplotlib). Returns the result dict (or its JSON string with ``as_json``).
    Expected problems are reported via ``status`` / ``error``, never raised.
    """
    try:
        bgr, err = _check_array(array, color_order)
        res = _result("E_INVALID_ARRAY", err) if bgr is None else _score_bgr(bgr, return_features, return_crop)
        if return_crop:
            res.setdefault("crop", None)
    except Exception as exc:  # pragma: no cover - last resort
        res = _result("E_INTERNAL", "%s (%s: %s)" % (ERRORS["E_INTERNAL"], type(exc).__name__, exc))
    return to_json(res) if as_json else res


def score_file(path: Union[str, "os.PathLike[str]"], as_json: bool = False,
               return_features: bool = False, return_crop: bool = False) -> Union[Dict[str, Any], str]:
    """Score an image file (HEIC/HEIF via pillow-heif, JPEG, PNG, ...; EXIF orientation applied).
    ``return_crop``: as in :func:`score_image`."""
    try:
        p = os.fspath(path)
        if not os.path.isfile(p):
            res = _result("E_FILE_NOT_FOUND", "%s: %s" % (ERRORS["E_FILE_NOT_FOUND"], p))
        else:
            try:
                img = load_image(p)
            except Exception as exc:
                img = None
                res = _result("E_UNREADABLE_IMAGE", "%s: %s (%s)" % (ERRORS["E_UNREADABLE_IMAGE"], p, type(exc).__name__))
            if img is not None:
                res = _score_bgr(img, return_features, return_crop)
    except Exception as exc:  # pragma: no cover - last resort
        res = _result("E_INTERNAL", "%s (%s: %s)" % (ERRORS["E_INTERNAL"], type(exc).__name__, exc))
    if return_crop and isinstance(res, dict):
        res.setdefault("crop", None)
    return to_json(res) if as_json else res


def score_file_json(path: Union[str, "os.PathLike[str]"]) -> str:
    """``score_file(path, as_json=True)``."""
    return score_file(path, as_json=True)  # type: ignore[return-value]


def score_image_json(array: Any, color_order: str = "BGR") -> str:
    """``score_image(array, color_order, as_json=True)``."""
    return score_image(array, color_order=color_order, as_json=True)  # type: ignore[return-value]


# ==========================================================================
# Embedded unit tests (no images embedded; a synthetic Data Matrix-like pattern is drawn)
# ==========================================================================
def _synthetic_code(ppm: int = 30, n: int = 18, seed: int = 7) -> np.ndarray:
    """Gray uint8 photo-like image with a Data Matrix-like symbol (L finder, timing borders,
    random interior), ``ppm`` px/module, slightly blurred and noisy. Not a decodable code."""
    rng = np.random.default_rng(seed)
    g = rng.random((n, n)) < 0.5
    g[:, 0] = True            # solid left
    g[n - 1, :] = True        # solid bottom
    g[0, :] = np.arange(n) % 2 == 0          # timing top
    g[:, n - 1] = np.arange(n) % 2 == 1      # timing right
    g[0, n - 1] = False
    code = np.where(np.kron(g, np.ones((ppm, ppm), bool)), 30, 225).astype(np.uint8)
    H, W = 1100, 1400
    img = np.full((H, W), 225, np.uint8)
    y0, x0 = (H - n * ppm) // 2, (W - n * ppm) // 2
    img[y0:y0 + n * ppm, x0:x0 + n * ppm] = code
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    noise = rng.normal(0, 3, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


class ReleaseTests(unittest.TestCase):
    _cache: Dict[str, Any] = {}

    @classmethod
    def synth(cls) -> Tuple[np.ndarray, Dict[str, Any]]:
        if "res" not in cls._cache:
            g = _synthetic_code()
            cls._cache["img"] = g
            cls._cache["res"] = score_image(g)
        return cls._cache["img"], cls._cache["res"]

    def _tmpdir(self) -> str:
        import tempfile
        return tempfile.mkdtemp(prefix="dm_release_test_", dir=os.environ.get("DM_RELEASE_TMPDIR") or None)

    def test_build_number(self):
        self.assertEqual(get_build_number(), "0.02")
        self.assertEqual(get_version(), BUILD_NUMBER)
        self.assertIsInstance(get_build_number(), str)

    def test_model_shapes(self):
        nf = len(MODEL_FEATURES)
        self.assertEqual(nf, 107)
        self.assertFalse(any(n.startswith("adain_") for n in MODEL_FEATURES))
        self.assertEqual(tuple(MODEL_CLASSES), CLASSES)
        self.assertEqual(CLASSES, ("TT", "IJ", "IJ_PC", "LA"))
        self.assertEqual(sorted(SKLEARN_CLASS_ORDER), sorted(CLASSES))
        self.assertEqual(_COEF.shape, (len(CLASSES), nf))
        self.assertEqual(_INTERCEPT.shape, (len(CLASSES),))
        for a in (_IMPUTER_STATISTICS, _SCALER_MEAN, _SCALER_SCALE):
            self.assertEqual(a.shape, (nf,))
            self.assertTrue(np.all(np.isfinite(a)))
        self.assertTrue(np.all(_SCALER_SCALE > 0))
        self.assertTrue(set(MODEL_FEATURES) <= set(FEATURE_NAMES))
        self.assertEqual(MIN_PX_PER_MODULE, 25.0)

    def test_get_model_parameters(self):
        mp = get_model_parameters()
        self.assertEqual(mp["classes"], CLASSES)
        self.assertEqual(len(mp["feature_names"]), mp["coef"].shape[1])
        mp["coef"][:] = 0          # a copy: the embedded model is unchanged
        self.assertTrue(np.any(_COEF != 0))

    def test_predict_proba_rows_sum_to_one(self):
        rng = np.random.default_rng(0)
        X = _SCALER_MEAN + rng.normal(size=(5, len(MODEL_FEATURES))) * _SCALER_SCALE
        X[0, :10] = np.nan
        P = predict_proba(X)
        self.assertEqual(P.shape, (5, len(CLASSES)))
        np.testing.assert_allclose(P.sum(1), 1.0, atol=1e-12)

    def test_synthetic_scores_keys_sum_json(self):
        _, r = self.synth()
        self.assertEqual(r["status"], "ok", r["error"])
        self.assertIsNone(r["error"])
        self.assertEqual(tuple(r["scores"].keys()), CLASSES)
        self.assertAlmostEqual(sum(r["scores"].values()), 1.0, places=9)
        self.assertEqual(r["scores_array"].shape, (len(CLASSES),))
        np.testing.assert_allclose(r["scores_array"], [r["scores"][c] for c in CLASSES])
        self.assertGreaterEqual(r["n_codes"], 1)
        c = r["codes"][r["primary_code_index"] - 1]
        self.assertGreaterEqual(c["px_per_module"], MIN_PX_PER_MODULE)
        self.assertEqual(len(c["bbox"]), 4)
        j = json.loads(to_json(r))
        self.assertEqual(j["build"], "0.02")
        self.assertEqual(list(j["scores"].keys()), list(CLASSES))
        np.testing.assert_allclose(j["scores_array"], r["scores_array"])
        s = score_image(self.synth()[0], as_json=True)
        self.assertIsInstance(s, str)
        self.assertEqual(json.loads(s)["scores"], j["scores"])
        self.assertEqual(json.loads(score_image_json(self.synth()[0])), json.loads(s))

    def test_gray_bgr_rgb_bgra(self):
        g, r = self.synth()
        bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        for arr, co in ((bgr, "BGR"), (rgb, "RGB"), (bgra, "BGRA"), (g[:, :, None], "BGR")):
            r2 = score_image(arr, color_order=co)
            self.assertEqual(r2["status"], "ok")
            np.testing.assert_allclose(r2["scores_array"], r["scores_array"], atol=1e-12)
        # a real colour image: RGB input with color_order="RGB" equals the BGR input
        col = bgr.copy()
        col[:, :, 0] = np.clip(col[:, :, 0].astype(int) + 20, 0, 255).astype(np.uint8)
        a = score_image(col, color_order="BGR")
        b = score_image(np.ascontiguousarray(col[:, :, ::-1]), color_order="RGB")
        self.assertEqual(a["status"], b["status"])
        np.testing.assert_allclose(a["scores_array"], b["scores_array"], atol=1e-12)

    def test_score_file_matches_score_image(self):
        g, r = self.synth()
        d_ = self._tmpdir()
        p = os.path.join(d_, "synthetic.png")
        try:
            self.assertTrue(cv2.imwrite(p, g))
            rf = score_file(p)
            self.assertEqual(rf["status"], "ok")
            np.testing.assert_allclose(rf["scores_array"], r["scores_array"], atol=1e-12)
            self.assertEqual(json.loads(score_file_json(p))["scores"], json.loads(to_json(r))["scores"])
        finally:
            if os.path.exists(p):
                os.remove(p)
            os.rmdir(d_)

    def test_error_missing_file(self):
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "does_not_exist_%d.heic" % os.getpid())
        r = score_file(p)
        self.assertEqual(r["status"], "E_FILE_NOT_FOUND")
        self.assertTrue(r["error"].startswith(ERRORS["E_FILE_NOT_FOUND"]))
        self.assertIsNone(r["scores"])
        self.assertTrue(np.all(np.isnan(r["scores_array"])))
        j = json.loads(score_file(p, as_json=True))
        self.assertEqual(j["status"], "E_FILE_NOT_FOUND")
        self.assertEqual(j["scores_array"], [None] * len(CLASSES))

    def test_error_garbage_file(self):
        d_ = self._tmpdir()
        p = os.path.join(d_, "garbage.heic")
        try:
            with open(p, "wb") as fh:
                fh.write(b"this is not an image" * 50)
            r = score_file(p)
            self.assertEqual(r["status"], "E_UNREADABLE_IMAGE")
            self.assertTrue(r["error"].startswith(ERRORS["E_UNREADABLE_IMAGE"]))
            self.assertIsNone(r["scores"])
        finally:
            os.remove(p)
            os.rmdir(d_)

    def test_error_bad_arrays(self):
        bad = [np.zeros((100, 100), np.float32), np.zeros((100, 100, 2), np.uint8),
               np.zeros((4, 100, 100, 3), np.uint8), np.zeros((5, 5), np.uint8), "not an array", None]
        for a in bad:
            r = score_image(a)
            self.assertEqual(r["status"], "E_INVALID_ARRAY", repr(type(a)))
            self.assertTrue(r["error"].startswith(ERRORS["E_INVALID_ARRAY"]))
            self.assertIsNone(r["scores"])
        self.assertEqual(score_image(np.zeros((50, 50, 3), np.uint8), color_order="XYZ")["status"],
                         "E_INVALID_ARRAY")

    def test_error_blank_image_no_code(self):
        r = score_image(np.full((600, 800, 3), 200, np.uint8))
        self.assertEqual(r["status"], "E_NO_CODE_DETECTED")
        self.assertEqual(r["error"], ERRORS["E_NO_CODE_DETECTED"])
        self.assertEqual(r["codes"], [])
        self.assertIsNone(r["scores"])

    def test_error_resolution_too_low(self):
        g, _ = self.synth()
        small = cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)  # ~15 px/module
        r = score_image(small)
        self.assertEqual(r["status"], "E_RESOLUTION_TOO_LOW")
        self.assertEqual(r["error"], ERRORS["E_RESOLUTION_TOO_LOW"])
        self.assertGreaterEqual(r["n_codes"], 1)
        self.assertTrue(all(c["px_per_module"] < MIN_PX_PER_MODULE for c in r["codes"]))
        self.assertTrue(all(c["status"] == "E_RESOLUTION_TOO_LOW" for c in r["codes"]))
        self.assertIsNone(r["scores"])
        self.assertTrue(np.all(np.isnan(r["scores_array"])))

    def test_return_crop(self):
        g, r = self.synth()
        self.assertNotIn("crop", r)                       # default output unchanged
        self.assertTrue(all("crop" not in c for c in r["codes"]))
        rc = score_image(g, return_crop=True)
        self.assertEqual(rc["status"], "ok")
        np.testing.assert_array_equal(rc["scores_array"], r["scores_array"])
        cr = rc["crop"]
        self.assertIsInstance(cr, np.ndarray)
        self.assertEqual(cr.dtype, np.uint8)
        self.assertEqual(cr.ndim, 3)
        self.assertEqual(cr.shape[2], 3)
        prim = rc["codes"][rc["primary_code_index"] - 1]
        self.assertIs(prim["crop"], cr)
        for c in rc["codes"]:
            if c["status"] == "ok":
                self.assertIsInstance(c["crop"], np.ndarray)
            else:
                self.assertNotIn("crop", c)
        # same pixels as the normalised crop (BGR -> RGB), about 25 px/module
        n = prim.get("grid_n") or 18
        side = max(cr.shape[:2])
        self.assertTrue(0.8 * 25 * n <= side <= 2.0 * 25 * n, (side, n))
        self.assertEqual(json.loads(to_json(rc)), json.loads(to_json(r)))   # crops never go into JSON
        rf = score_image(g, return_crop=True, as_json=True)
        self.assertNotIn("crop", json.loads(rf))
        # error cases: crop key present and None
        e = score_image(np.full((600, 800, 3), 200, np.uint8), return_crop=True)
        self.assertEqual(e["status"], "E_NO_CODE_DETECTED")
        self.assertIsNone(e["crop"])
        self.assertIsNone(score_file("/nonexistent/x_%d.heic" % os.getpid(), return_crop=True)["crop"])
        self.assertIsNone(score_image(None, return_crop=True)["crop"])

    def test_errors_dict(self):
        for k in ("E_FILE_NOT_FOUND", "E_UNREADABLE_IMAGE", "E_INVALID_ARRAY", "E_NO_CODE_DETECTED",
                  "E_RESOLUTION_TOO_LOW", "E_FEATURE_EXTRACTION_FAILED", "E_INTERNAL"):
            self.assertIn(k, ERRORS)
            self.assertTrue(ERRORS[k].startswith("PROVISIONAL"))

    @unittest.skipUnless(os.environ.get("DM_RELEASE_TEST_IMAGE"), "DM_RELEASE_TEST_IMAGE not set")
    def test_golden_image(self):
        p = os.environ["DM_RELEASE_TEST_IMAGE"]
        r = score_file(p)
        self.assertEqual(r["status"], "ok", r["error"])
        self.assertAlmostEqual(sum(r["scores"].values()), 1.0, places=9)
        exp = os.environ.get("DM_RELEASE_TEST_CLASS")
        if exp:
            self.assertEqual(r["predicted_class"], exp)


def run_selftest(verbosity: int = 2) -> bool:
    suite = unittest.TestLoader().loadTestsFromTestCase(ReleaseTests)
    return unittest.TextTestRunner(verbosity=verbosity).run(suite).wasSuccessful()


# ==========================================================================
# Command line
# ==========================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog=os.path.basename(__file__),
        description="DM detector build %s: print-type scores (%s) for Data Matrix codes in photos. "
                    "Prints a JSON array (one object per image). Exit codes: 0 ok, 1 any image errored, "
                    "2 usage error." % (BUILD_NUMBER, ", ".join(CLASSES)))
    ap.add_argument("images", nargs="*", help="image files (HEIC/JPEG/PNG ...)")
    ap.add_argument("--version", action="version", version=BUILD_NUMBER)
    ap.add_argument("--selftest", action="store_true", help="run the embedded unit tests")
    ap.add_argument("--indent", type=int, default=1, help="JSON indent (default 1; -1 = compact)")
    ap.add_argument("--features", action="store_true", help="include the 107 model features per code")
    a = ap.parse_args(argv)
    if a.selftest:
        if a.images:
            ap.error("--selftest takes no images")
        return 0 if run_selftest() else 1
    if not a.images:
        ap.error("give at least one image path (or --selftest / --version)")
    out, rc = [], 0
    for p in a.images:
        r = score_file(p, return_features=a.features)
        r = {"path": p, **r}
        if r["status"] != STATUS_OK:
            rc = 1
        out.append(_to_jsonable(r))
    print(json.dumps(out, indent=None if a.indent < 0 else a.indent, allow_nan=False))
    return rc


if __name__ == "__main__":
    sys.exit(main())
