"""dm_roi -- locate Data Matrix (ECC200) codes in a photo and return their ROIs.

Classical computer vision with OpenCV + NumPy only (no pylibdmtx / zxing /
scipy / scikit-image). Works on Python 3.9+ (tested 3.9 and 3.14) with opencv-python 5.0,
numpy 2.x, Pillow and pillow-heif (for iPhone HEIC files).

Geometric assumptions (by design, see ``DMConfig``):

* Codes are **square** Data Matrix symbols (no rectangular DM variants).
* The coarse orientation is one of 0, 90, 180 or 270 degrees.
* Around each of those steps the code may be tilted by up to about
  +/-10 degrees (``DMConfig.max_tilt_deg``, configurable), e.g. 80-100 or
  170-190 degrees. There is no general rotation-invariant search: candidates
  whose dominant edge orientation falls in the forbidden band
  (between ``max_tilt_deg`` and ``90 - max_tilt_deg`` modulo 90) are rejected.

Usage example::

    from dm_roi import find_dm_rois, find_dm_roi, load_image, crop_roi, draw_rois

    img = load_image("images/IMG_0066.HEIC")      # BGR uint8, EXIF-rotated
    rois = find_dm_rois(img, max_results=5)          # ranked, best first
    best = rois[0] if rois else None
    if best is not None:
        print(best.bbox, best.polygon, best.rotation, best.tilt_deg,
              best.score, best.polarity, best.module_px, best.grid_n)
        crop = crop_roi(img, best, pad=0.2, deskew=True, annotate=True)  # upright, "px/mod" text
        vis = draw_rois(img, rois)                          # annotated copy

    # tables with a configurable selection of output fields (pandas)
    from dm_roi import rois_to_dataframe, batch_results_to_dataframe, ROI_OUTPUT_FIELDS
    df = rois_to_dataframe(rois, fields=["score", "rotation", "tilt_deg", "module_px"])
    df_all = batch_results_to_dataframe("images", fields=["image", "rank", "score", "angle",
                                                            "module_px", "grid_n"])

Command line::

    python dm_roi.py images/IMG_0066.HEIC --out out/IMG_0066_roi.png
    python dm_roi.py images/IMG_0050.HEIC --all --json
    python dm_roi.py --batch images/            # CSV/JSON + thumbnails in out/batch/
    python dm_roi.py images/IMG_0066.HEIC --json --fields score,rotation,tilt_deg,module_px

Scope: localisation, cropping and module-size estimation only; the code
content is not decoded. Finder/timing checks are used to verify candidates,
read the coarse rotation and count modules.

All coordinates (``polygon``, ``bbox``, ``side_px``, ``module_px``...) are in
ORIGINAL full-resolution image pixels (after EXIF orientation is applied).

Conventions:

* ``polygon``: 4x2 float32 corners TL, TR, BR, BL of the tight (slightly
  tilted) square in image coordinates.
* ``tilt_deg``: residual tilt in [-max_tilt_deg, +max_tilt_deg]; positive
  means the code is turned clockwise as seen on screen.
* ``rotation``: coarse orientation {0, 90, 180, 270} (clockwise) derived from
  the L-shaped finder pattern; 0 means the solid "L" is on the left and bottom
  (canonical DM orientation). ``None`` if the finder could not be verified.
* ``angle``: ``rotation + tilt_deg`` (or just ``tilt_deg`` if rotation unknown).
* ``module_px``: estimated module pitch (size of one cell) in full-resolution
  pixels; for dotted (inkjet / dot-peen) codes this is the pitch, not the dot
  diameter. ``grid_n`` is the number of modules per side when confident.

What OpenCV 5.0 itself offers: ``cv2.barcode.BarcodeDetector`` handles 1D
symbologies only (EAN-8/13, UPC-A/E, ...); ``cv2.QRCodeDetector`` and
``cv2.QRCodeDetectorAruco`` detect QR codes only. There is no Data Matrix
detector in the main package, so this module uses its own texture/finder
based detector and (optionally) uses the 1D barcode detector to down-rank
candidates that sit on a linear barcode.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, replace, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

__all__ = [
    "DMConfig",
    "DMRoi",
    "VALID_SQUARE_SIZES",
    "ROI_OUTPUT_FIELDS",
    "DEFAULT_TABLE_FIELDS",
    "rois_to_dataframe",
    "batch_results_to_dataframe",
    "load_image",
    "find_dm_rois",
    "find_dm_roi",
    "estimate_module_size",
    "crop_roi",
    "annotate_crop",
    "draw_rois",
    "main",
]

#: Symbol sizes (modules per side) of square ECC200 Data Matrix codes.
VALID_SQUARE_SIZES: Tuple[int, ...] = (
    10, 12, 14, 16, 18, 20, 22, 24, 26, 32, 36, 40, 44, 48, 52,
    64, 72, 80, 88, 96, 104, 120, 132, 144,
)

#: Every output field a result row can contain (see :meth:`DMRoi.to_row`).
#: ``image`` and ``rank`` describe the source photo and the rank of the ROI in
#: it; the rest are :class:`DMRoi` attributes. Any ``details_<key>`` name
#: (e.g. ``details_finder``, ``details_timing``) selects one sub-score.
ROI_OUTPUT_FIELDS: Tuple[str, ...] = (
    "image", "rank", "bbox", "polygon", "center", "rotation", "tilt_deg", "angle", "score",
    "polarity", "side_px", "module_px", "module_px_xy", "grid_n", "module_conf",
    "module_method", "dotted_fallback", "dotted_located", "dotted_pitch", "details",
)

#: Default columns for tables (``fields=None``).
DEFAULT_TABLE_FIELDS: Tuple[str, ...] = (
    "image", "rank", "score", "rotation", "tilt_deg", "angle", "polarity", "side_px",
    "module_px", "module_px_xy", "grid_n", "module_conf", "module_method", "bbox",
)

IMAGE_EXTENSIONS = (".heic", ".heif", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

ImageInput = Union[str, "os.PathLike[str]", np.ndarray]


# --------------------------------------------------------------------------
# Configuration and result types
# --------------------------------------------------------------------------
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

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable representation."""
        return {
            "polygon": [[round(float(x), 1), round(float(y), 1)] for x, y in self.polygon],
            "bbox": [int(v) for v in self.bbox],
            "rotation": self.rotation,
            "tilt_deg": round(float(self.tilt_deg), 2),
            "angle": round(self.angle, 2),
            "score": round(float(self.score), 4),
            "polarity": self.polarity,
            "side_px": round(float(self.side_px), 1),
            "module_px": None if self.module_px is None else round(float(self.module_px), 2),
            "module_px_xy": None if self.module_px_xy is None
            else [round(float(v), 2) for v in self.module_px_xy],
            "grid_n": self.grid_n,
            "module_conf": round(float(self.module_conf), 3),
            "module_method": self.module_method,
            "dotted_fallback": self.dotted_fallback,
            "dotted_located": self.dotted_located,
            "dotted_pitch": self.dotted_pitch,
            "details": {k: round(float(v), 4) for k, v in self.details.items()},
        }

    def to_row(self, fields: Optional[Sequence[str]] = None, *, image: Optional[str] = None,
               rank: Optional[int] = None) -> Dict[str, Any]:
        """Return a flat dict with the selected output fields.

        Args:
            fields: names from :data:`ROI_OUTPUT_FIELDS` and/or ``details_<key>``;
                ``None`` = :data:`DEFAULT_TABLE_FIELDS`, ``"all"`` = every field.
            image: value for the ``image`` column (e.g. the file name).
            rank: value for the ``rank`` column (1 = best ROI of the image).
        """
        sel = _resolve_fields(fields)
        d = self.to_dict()
        row: Dict[str, Any] = {}
        for f in sel:
            if f == "image":
                row[f] = image
            elif f == "rank":
                row[f] = rank
            elif f == "center":
                row[f] = [round(v, 1) for v in self.center]
            elif f in ("dotted_fallback", "dotted_located", "dotted_pitch"):
                row[f] = bool(getattr(self, f))
            elif f.startswith("details_"):
                v = self.details.get(f[len("details_"):])
                row[f] = None if v is None else round(float(v), 4)
            elif f in d:
                row[f] = d[f]
            else:
                raise KeyError("unknown output field %r (see ROI_OUTPUT_FIELDS)" % f)
        return row


# --------------------------------------------------------------------------
# Image loading
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Border (finder / timing) analysis on an upright code crop
# --------------------------------------------------------------------------
_SIDES = ("top", "right", "bottom", "left")
# (solid side A, solid side B) of the L finder -> clockwise rotation of the code
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


# --------------------------------------------------------------------------
# Candidate proposals (working scale)
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# 1D barcode suppression (uses cv2.barcode from OpenCV 5)
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Output field selection / tables
# --------------------------------------------------------------------------
def _resolve_fields(fields: Optional[Union[str, Sequence[str]]]) -> Tuple[str, ...]:
    if fields is None:
        return DEFAULT_TABLE_FIELDS
    if isinstance(fields, str):
        if fields == "all":
            return ROI_OUTPUT_FIELDS
        fields = [f.strip() for f in fields.split(",") if f.strip()]
    out = tuple(fields)
    for f in out:
        if f not in ROI_OUTPUT_FIELDS and not f.startswith("details_"):
            raise KeyError("unknown output field %r (see ROI_OUTPUT_FIELDS)" % f)
    return out


def rois_to_dataframe(rois: Sequence["DMRoi"], fields: Optional[Union[str, Sequence[str]]] = None,
                      image: Optional[str] = None) -> Any:
    """Table of ROIs (one row per ROI) with the selected ``fields``.

    Returns a ``pandas.DataFrame`` if pandas is installed, else ``list[dict]``.
    """
    rows = [r.to_row(fields, image=image, rank=i + 1) for i, r in enumerate(rois)]
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        return rows
    return _int_columns(pd.DataFrame(rows, columns=list(_resolve_fields(fields))))


def _int_columns(df: Any) -> Any:
    """Keep integer columns integer (nullable Int64) even with empty rows."""
    for col in ("rank", "rotation", "grid_n"):
        if col in df.columns:
            try:
                df[col] = df[col].astype("Int64")
            except (TypeError, ValueError):
                pass
    return df


def batch_results_to_dataframe(images: Union[str, "os.PathLike[str]", Sequence[Any]],
                               fields: Optional[Union[str, Sequence[str]]] = None, *,
                               include_empty: bool = True, max_results: int = 5,
                               progress: bool = False, **find_kwargs: Any) -> Any:
    """Run :func:`find_dm_rois` over a folder or list of images; return one table.

    Args:
        images: folder path (all HEIC/JPEG/PNG files, sorted by name) or a list
            of paths / numpy arrays.
        fields: output columns (see :meth:`DMRoi.to_row`); ``None`` = default set.
        include_empty: add a row (image name only) for photos without a code.
        max_results, find_kwargs: forwarded to :func:`find_dm_rois`
            (e.g. ``max_tilt_deg=8``, ``min_score=0.6``).
        progress: print one line per image.

    Returns a ``pandas.DataFrame`` (``list[dict]`` if pandas is missing), one
    row per ROI. Per-image timings are in the CLI ``--batch`` summary.
    """
    if isinstance(images, (str, os.PathLike)) and Path(images).is_dir():
        items: List[Any] = list(_list_images(Path(images)))
    else:
        items = list(images)  # type: ignore[arg-type]
    sel = _resolve_fields(fields)
    rows: List[Dict[str, Any]] = []
    for i, item in enumerate(items):
        name = Path(item).name if not isinstance(item, np.ndarray) else "array_%d" % i
        img = load_image(item)
        rois = find_dm_rois(img, max_results=max_results, **find_kwargs)
        del img
        if progress:
            print("%-20s %d ROI(s)" % (name, len(rois)))
        if not rois and include_empty:
            rows.append({f: (name if f == "image" else None) for f in sel})
        for k, r in enumerate(rois):
            rows.append(r.to_row(sel, image=name, rank=k + 1))
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        return rows
    return _int_columns(pd.DataFrame(rows, columns=list(sel)))


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
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


def find_dm_roi(image: ImageInput, **kwargs: Any) -> Optional[DMRoi]:
    """Return the best :class:`DMRoi` or ``None`` (same arguments as find_dm_rois)."""
    kwargs.setdefault("max_results", 1)
    rois = find_dm_rois(image, **kwargs)
    return rois[0] if rois else None


# --------------------------------------------------------------------------
# Module size estimation
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Cropping and drawing
# --------------------------------------------------------------------------
def annotate_crop(crop: np.ndarray, roi: DMRoi, *, pad: Optional[float] = None,
                  bar: Optional[bool] = None) -> np.ndarray:
    """Draw the module-size estimate as text on a (deskewed, padded) crop.

    Text is e.g. ``"12.4 px/mod 18x18"`` (module pitch in ORIGINAL full-res
    pixels; grid size appended when known). It goes into the top-left corner
    of the padding on a semi-opaque dark box (readable on light and dark
    codes). If the padding is too thin to hold the text (``pad`` = padding
    fraction used for the crop) or ``bar=True``, a thin bar is added above the
    crop instead, so the code itself is never covered.
    """
    out = crop if crop.ndim == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    out = out.copy()
    if roi.module_px is None:
        text = "px/mod n/a"
    elif roi.grid_n:
        text = "%.1f px/mod %dx%d" % (roi.module_px, roi.grid_n, roi.grid_n)
    else:
        text = "%.1f px/mod" % roi.module_px
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.42 if w < 260 else min(1.2, w / 600.0)
    thick = 1 if fs < 0.8 else 2
    (tw, th), bl = cv2.getTextSize(text, font, fs, thick)
    while tw > w - 8 and fs > 0.3:
        fs *= 0.9
        (tw, th), bl = cv2.getTextSize(text, font, fs, thick)
    m = 3
    box_h = th + bl + 2 * m
    margin = h * pad / (1.0 + 2.0 * pad) if pad is not None else 0.0
    use_bar = bar if bar is not None else (margin < box_h + 2)
    if use_bar:
        canvas = np.full((h + box_h, w, 3), 40, np.uint8)
        canvas[box_h:] = out
        cv2.putText(canvas, text, (m, m + th), font, fs, (255, 255, 255), thick, cv2.LINE_AA)
        return canvas
    x0, y0 = 1, 1
    x1, y1 = min(w - 1, x0 + tw + 2 * m), y0 + box_h
    patch = out[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = (0.35 * patch + 0.65 * 20).astype(np.uint8)
    cv2.putText(out, text, (x0 + m, y0 + m + th), font, fs, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def crop_roi(img: ImageInput, roi: DMRoi, pad: float = 0.2, deskew: bool = True,
             normalize_rotation: bool = True, out_size: Optional[int] = None,
             annotate: bool = False) -> np.ndarray:
    """Cut a ROI out of the full-resolution image.

    Args:
        pad: padding around the code as a fraction of its side (0.2 = 20 %).
        deskew: undo the small tilt (warpAffine to an upright square). If
            False, an axis-aligned padded bbox crop is returned.
        normalize_rotation: with ``deskew``, additionally rotate by 90-degree
            steps (np.rot90) so the L finder ends up left+bottom, if known.
        out_size: optionally resize the (square) result to this side length.
        annotate: if True, overlay the module-size estimate on the crop
            (see :func:`annotate_crop`).
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
    if annotate:
        out = annotate_crop(out, roi, pad=pad if deskew else None)
    return out


def draw_rois(img: ImageInput, rois: Sequence[DMRoi], max_long_side: Optional[int] = 1600,
              thickness: Optional[int] = None, labels: bool = True) -> np.ndarray:
    """Return an annotated (optionally downscaled) BGR copy of ``img``.

    The best ROI is green, others orange; a dot marks the finder corner
    (junction of the solid L) when the rotation is known.
    """
    im = load_image(img)
    H, W = im.shape[:2]
    s = 1.0
    if max_long_side and max(H, W) > max_long_side:
        s = max_long_side / float(max(H, W))
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    else:
        im = im.copy()
    th = thickness or max(2, int(round(max(im.shape[:2]) / 400.0)))
    for i, r in enumerate(rois):
        col = (0, 220, 0) if i == 0 else (0, 160, 255)
        pts = np.round(r.polygon * s).astype(np.int32)
        cv2.polylines(im, [pts], True, col, th, cv2.LINE_AA)
        if r.rotation is not None:
            # canonical finder corner is bottom-left; rotate with the code
            idx = {0: 3, 90: 0, 180: 1, 270: 2}[r.rotation]
            cv2.circle(im, tuple(int(v) for v in pts[idx]), th * 3, (0, 0, 255), -1, cv2.LINE_AA)
        if labels:
            txt = "#%d %.2f" % (i + 1, r.score)
            if r.module_px:
                txt += " %.1fpx/m" % r.module_px
            org = (int(pts[:, 0].min()), max(12, int(pts[:, 1].min()) - 6))
            fs = max(0.45, max(im.shape[:2]) / 2200.0)
            cv2.putText(im, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
            cv2.putText(im, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, col, th, cv2.LINE_AA)
    return im


# --------------------------------------------------------------------------
# Command line interface
# --------------------------------------------------------------------------
def _default_out_dir() -> Path:
    return Path(__file__).resolve().parent / "out"


def _check_out_path(out: Path, inputs: Sequence[Path]) -> None:
    """Refuse to write into the folder that holds the input images."""
    o = out.resolve()
    for p in inputs:
        d = (p if p.is_dir() else p.parent).resolve()
        if o == d or d in o.parents:
            raise SystemExit("refusing to write output into the input image folder: %s" % o)


def _format_roi(i: int, r: DMRoi) -> str:
    poly = ", ".join("(%.0f, %.0f)" % (x, y) for x, y in r.polygon)
    mod = "n/a" if r.module_px is None else "%.2f px/module (x %.2f, y %.2f), N=%s, %s conf %.2f" % (
        r.module_px, r.module_px_xy[0], r.module_px_xy[1], r.grid_n if r.grid_n else "?",
        r.module_method, r.module_conf)
    return ("ROI #%d  score %.3f  %s\n  bbox (x, y, w, h) = %s\n  polygon TL,TR,BR,BL = [%s]\n"
            "  rotation = %s deg, tilt = %+.1f deg, angle = %.1f deg, side = %.0f px\n"
            "  module: %s" % (i + 1, r.score, r.polarity, tuple(r.bbox), poly,
                              r.rotation if r.rotation is not None else "?", r.tilt_deg,
                              r.angle, r.side_px, mod))


def _list_images(folder: Path) -> List[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file())


def _run_batch(folder: Path, out_dir: Path, max_results: int, params: Dict[str, Any],
               fields: Optional[str] = None) -> int:
    files = _list_images(folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs = out_dir / "thumbs"
    thumbs.mkdir(exist_ok=True)
    rows = []
    records = []
    for p in files:
        t0 = time.perf_counter()
        img = load_image(p)
        t1 = time.perf_counter()
        rois = find_dm_rois(img, max_results=max_results, **params)
        t2 = time.perf_counter()
        cv2.imwrite(str(thumbs / (p.stem + "_roi.jpg")), draw_rois(img, rois, max_long_side=1000),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        best = rois[0] if rois else None
        if fields:
            sel = _resolve_fields(fields)
            if rois:
                for k, r in enumerate(rois):
                    rr = r.to_row(sel, image=p.name, rank=k + 1)
                    rows.append({kk: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                                 for kk, v in rr.items()})
            else:
                rows.append({f: (p.name if f == "image" else "") for f in sel})
            records.append({"file": p.name, "load_s": t1 - t0, "detect_s": t2 - t1,
                            "rois": [r.to_dict() for r in rois]})
            print("%-20s %d ROI(s)  (load %.2fs, detect %.2fs)" % (p.name, len(rois), t1 - t0, t2 - t1))
            continue
        rows.append({
            "file": p.name, "n_rois": len(rois),
            "score": "" if best is None else round(best.score, 3),
            "bbox": "" if best is None else " ".join(str(v) for v in best.bbox),
            "rotation": "" if best is None or best.rotation is None else best.rotation,
            "tilt_deg": "" if best is None else round(best.tilt_deg, 1),
            "polarity": "" if best is None else best.polarity,
            "module_px": "" if best is None or best.module_px is None else round(best.module_px, 2),
            "grid_n": "" if best is None or not best.grid_n else best.grid_n,
            "load_s": round(t1 - t0, 3), "detect_s": round(t2 - t1, 3),
        })
        records.append({"file": p.name, "load_s": t1 - t0, "detect_s": t2 - t1,
                        "rois": [r.to_dict() for r in rois]})
        print("%-20s %d ROI(s)  best score %s  module %s px  (load %.2fs, detect %.2fs)" % (
            p.name, len(rois), rows[-1]["score"], rows[-1]["module_px"], t1 - t0, t2 - t1))
    if rows:
        with open(out_dir / "summary.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(records, fh, indent=1)
    print("wrote %s, %s and %s" % (out_dir / "summary.csv", out_dir / "summary.json", thumbs))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line entry point (see module docstring)."""
    ap = argparse.ArgumentParser(description="Locate Data Matrix code ROIs in a photo.")
    ap.add_argument("image", nargs="?", help="image file (HEIC/JPEG/PNG)")
    ap.add_argument("--batch", metavar="FOLDER", help="process every image in FOLDER")
    ap.add_argument("--out", help="annotated PNG path (single image) or output folder (--batch); "
                                  "default: out/ next to dm_roi.py")
    ap.add_argument("--crop", help="save the deskewed crop of the best ROI to this path")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    ap.add_argument("--all", action="store_true", help="report all candidates, not only the best")
    ap.add_argument("--max-results", type=int, default=5)
    ap.add_argument("--max-tilt", type=float, default=None, help="max tilt in degrees (default 10)")
    ap.add_argument("--min-score", type=float, default=None)
    ap.add_argument("--fields", default=None,
                    help="comma-separated output fields for --json and the --batch CSV "
                         "(see ROI_OUTPUT_FIELDS; 'all' = every field)")
    args = ap.parse_args(argv)
    params: Dict[str, Any] = {}
    if args.max_tilt is not None:
        params["max_tilt_deg"] = args.max_tilt
    if args.min_score is not None:
        params["min_score"] = args.min_score

    if args.batch:
        folder = Path(args.batch)
        out_dir = Path(args.out) if args.out else _default_out_dir() / "batch"
        _check_out_path(out_dir, [folder])
        return _run_batch(folder, out_dir, args.max_results, params, args.fields)
    if not args.image:
        ap.error("give an image path or --batch FOLDER")
    path = Path(args.image)
    t0 = time.perf_counter()
    img = load_image(path)
    t1 = time.perf_counter()
    rois = find_dm_rois(img, max_results=args.max_results if args.all else max(1, args.max_results),
                        **params)
    t2 = time.perf_counter()
    shown = rois if args.all else rois[:1]
    if args.json:
        print(json.dumps({"file": str(path), "image_size": [img.shape[1], img.shape[0]],
                          "load_s": round(t1 - t0, 3), "detect_s": round(t2 - t1, 3),
                          "rois": [r.to_dict() if not args.fields else
                                   r.to_row(args.fields, image=path.name, rank=i + 1)
                                   for i, r in enumerate(shown)]}, indent=1))
    else:
        print("%s  (%dx%d, load %.2fs, detect %.2fs): %d candidate(s)" % (
            path, img.shape[1], img.shape[0], t1 - t0, t2 - t1, len(rois)))
        if not shown:
            print("  no Data Matrix found")
        for i, r in enumerate(shown):
            print(_format_roi(i, r))
    out_png = Path(args.out) if args.out else _default_out_dir() / (path.stem + "_roi.png")
    _check_out_path(out_png.parent, [path])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), draw_rois(img, shown))
    if not args.json:
        print("annotated image: %s" % out_png)
    if args.crop and rois:
        cp = Path(args.crop)
        _check_out_path(cp.parent, [path])
        cp.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cp), crop_roi(img, rois[0], pad=0.2, annotate=True))
        if not args.json:
            print("crop: %s" % cp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
