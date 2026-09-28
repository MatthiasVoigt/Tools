"""dm_detector -- small public API for Data Matrix localisation (wraps dm_roi).

This module only re-exposes :mod:`dm_roi` (same folder) under clear names, so
detection behaviour is identical to ``dm_roi.find_dm_rois``. The algorithm
itself lives in ``dm_roi.py``; see ``DMDetectorAlgorithm.md``.

Assumptions: square codes, coarse rotation in 90-degree steps, up to about
+/-10 degrees residual tilt (``max_tilt_deg``). All sizes are in ORIGINAL
full-resolution pixels. Decoding is out of scope.

Example::

    import dm_detector as d

    img = d.load_photo("images/IMG_0066.HEIC")
    results = d.detect(img)                   # ranked list of DMRoi, best first
    top = d.best(img)                         # DMRoi or None
    if top is not None:
        print(top.score, top.rotation, top.tilt_deg, top.module_px, top.grid_n,
              top.dotted_fallback)
        crop = d.crop(img, top)               # deskewed crop with "px/mod" text
    df = d.results_table("images", fields=["image", "score", "rotation", "tilt_deg",
                                            "module_px", "grid_n", "dotted_fallback"])

Per-result attributes (see :class:`dm_roi.DMRoi`): ``score``, ``rotation``,
``tilt_deg``, ``angle``, ``module_px`` (pixels per module), ``module_px_xy``,
``grid_n``, ``polarity``, ``side_px``, ``module_conf``, ``module_method``,
``polygon``, ``bbox``, ``details`` and the booleans ``dotted_located`` (ROI
located by the dotted-code fallback), ``dotted_pitch`` (module size from the
dot-pitch estimate) and ``dotted_fallback`` (either of the two).
"""
from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence, Union

import numpy as np

import dm_roi as _dm
from dm_roi import (
    DEFAULT_TABLE_FIELDS,
    ROI_OUTPUT_FIELDS,
    VALID_SQUARE_SIZES,
    DMConfig,
    DMRoi,
    annotate_crop,
    draw_rois,
    estimate_module_size,
    rois_to_dataframe,
)

__all__ = [
    "DMConfig",
    "DMRoi",
    "ROI_OUTPUT_FIELDS",
    "DEFAULT_TABLE_FIELDS",
    "PRESENTATION_FIELDS",
    "VALID_SQUARE_SIZES",
    "load_photo",
    "detect",
    "best",
    "crop",
    "draw",
    "module_size",
    "results_table",
    "rois_table",
    "annotate_crop",
]

#: Columns used by ``results_presentation.ipynb`` by default.
PRESENTATION_FIELDS = ("image", "rank", "score", "rotation", "tilt_deg", "module_px", "grid_n",
                       "dotted_fallback")

ImageLike = Union[str, "os.PathLike[str]", np.ndarray]


def load_photo(path: ImageLike, max_long_side: Optional[int] = None) -> np.ndarray:
    """Load a photo (HEIC via pillow-heif, JPEG, PNG or ndarray) as BGR uint8, EXIF-rotated."""
    return _dm.load_image(path, max_long_side=max_long_side)


def detect(image_or_path: ImageLike, *, max_results: int = 5, **params: Any) -> List[DMRoi]:
    """Detect Data Matrix codes; ranked list of :class:`DMRoi` (best first).

    ``params`` are :class:`DMConfig` fields, e.g. ``max_tilt_deg=8``, ``min_score=0.6``.
    """
    return _dm.find_dm_rois(image_or_path, max_results=max_results, **params)


def best(image_or_path: ImageLike, **params: Any) -> Optional[DMRoi]:
    """Best single result or ``None`` (same parameters as :func:`detect`)."""
    rois = detect(image_or_path, **params)
    return rois[0] if rois else None


def crop(image: ImageLike, roi: DMRoi, annotate: bool = True, pad: float = 0.2,
         out_size: Optional[int] = None, normalize_rotation: bool = True) -> np.ndarray:
    """Deskewed, padded crop of one code; with ``annotate`` the px/module text is drawn on it."""
    return _dm.crop_roi(image, roi, pad=pad, deskew=True, normalize_rotation=normalize_rotation,
                        out_size=out_size, annotate=annotate)


def draw(image: ImageLike, rois: Sequence[DMRoi], max_long_side: Optional[int] = 1600) -> np.ndarray:
    """Annotated (downscaled) copy of the photo with ROI outlines."""
    return draw_rois(image, rois, max_long_side=max_long_side)


def module_size(image: ImageLike, roi: DMRoi) -> dict:
    """Pixels-per-module estimate for one ROI (full-res pixels); see dm_roi.estimate_module_size."""
    return estimate_module_size(image, roi)


def rois_table(rois: Sequence[DMRoi], fields: Optional[Union[str, Sequence[str]]] = None,
               image: Optional[str] = None) -> Any:
    """Table (pandas DataFrame, or list[dict] without pandas) for the ROIs of one photo."""
    return rois_to_dataframe(rois, fields=fields, image=image)


def results_table(images: Any, fields: Optional[Union[str, Sequence[str]]] = None, *,
                  include_empty: bool = True, max_results: int = 5, **params: Any) -> Any:
    """Detect over a folder / list of photos and return one table (one row per code).

    ``fields`` selects columns from :data:`ROI_OUTPUT_FIELDS` (incl. ``dotted_fallback``)
    or ``details_<key>``; ``None`` = :data:`DEFAULT_TABLE_FIELDS`.
    """
    return _dm.batch_results_to_dataframe(images, fields=fields, include_empty=include_empty,
                                          max_results=max_results, **params)
