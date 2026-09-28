#!/usr/bin/env python3
"""crop_images.py -- detect and crop Data Matrix codes locally, reusing dm_detector.

Everything runs on this machine: photos are read from an input folder, the
existing detector in this project (``dm_detector`` -> ``dm_roi``) finds the
codes, and each code is written as a lossless PNG crop to a local output folder.
Nothing is uploaded or copied anywhere else. Input photos are never modified.

Naming: ``<photo stem>_<NN>.png`` with a two-digit, 1-based index in detector
rank order (best first), e.g. ``IMG_0051_01.png``, ``IMG_0051_02.png``. The
suffix is used even when a photo has one code, so the source photo is always
the part of the name before the last underscore.

Crops are made with ``dm_detector.crop(img, roi, annotate=False, pad=PAD)``:
deskewed, upright, full resolution, padded by ``PAD`` of the code side on each
side (default 0.2, same as dm_crops). A ``crops.csv`` manifest with the source
photo, index and detector values for every crop is written next to the PNGs.

Usage (from the DM_detector folder, with the project venv)::

    .venv/bin/python crop_images.py                       # images/ -> dm_crops_local/
    .venv/bin/python crop_images.py --images images --out my_crops --max-codes 3
    .venv/bin/python crop_images.py --overwrite           # replace existing crops

By default it refuses to overwrite existing crop files, and it will not write
into the input folder.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # dm_detector.py and dm_roi.py live next to this script

import dm_detector as d  # noqa: E402
from dm_roi import IMAGE_EXTENSIONS  # noqa: E402

MANIFEST_FIELDS = ["crop_file", "source_image", "roi_index", "score", "rotation", "tilt_deg",
                   "angle", "module_px", "grid_n", "polarity", "dotted_fallback",
                   "bbox_x", "bbox_y", "bbox_w", "bbox_h", "crop_w", "crop_h"]


def list_photos(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and not p.name.startswith(".")
                  and p.suffix.lower() in IMAGE_EXTENSIONS)


def crop_folder(images_dir: Path, out_dir: Path, *, max_codes: int = 5, pad: float = 0.2,
                overwrite: bool = False, verbose: bool = True) -> list[dict]:
    """Detect and crop every photo in ``images_dir``; return the manifest rows."""
    images_dir, out_dir = images_dir.resolve(), out_dir.resolve()
    if not images_dir.is_dir():
        raise SystemExit(f"input folder not found: {images_dir}")
    if out_dir == images_dir or images_dir in out_dir.parents:
        raise SystemExit("refusing to write crops inside the input folder")
    photos = list_photos(images_dir)
    if not photos:
        raise SystemExit(f"no images with extensions {IMAGE_EXTENSIONS} in {images_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    n_none = n_skip = 0
    t_all = time.perf_counter()
    for i, photo in enumerate(photos, 1):
        t0 = time.perf_counter()
        try:
            img = d.load_photo(photo)
            rois = d.detect(img, max_results=max_codes)
        except Exception as exc:  # keep going on a bad file
            print(f"[{i}/{len(photos)}] {photo.name}: ERROR {exc}", file=sys.stderr)
            continue
        if not rois:
            n_none += 1
        for k, roi in enumerate(rois, 1):
            name = f"{photo.stem}_{k:02d}.png"
            dst = out_dir / name
            if dst.exists() and not overwrite:
                n_skip += 1
                continue
            crop = d.crop(img, roi, annotate=False, pad=pad)
            if not cv2.imwrite(str(dst), crop):
                print(f"  could not write {dst}", file=sys.stderr)
                continue
            x, y, w, h = roi.bbox
            rows.append({
                "crop_file": name, "source_image": photo.name, "roi_index": k,
                "score": round(float(roi.score), 4),
                "rotation": roi.rotation if roi.rotation is not None else "",
                "tilt_deg": round(float(roi.tilt_deg), 2), "angle": round(roi.angle, 2),
                "module_px": round(float(roi.module_px), 2) if roi.module_px else "",
                "grid_n": roi.grid_n or "", "polarity": roi.polarity,
                "dotted_fallback": bool(roi.dotted_fallback),
                "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
                "crop_w": crop.shape[1], "crop_h": crop.shape[0],
            })
        if verbose:
            print(f"[{i}/{len(photos)}] {photo.name}: {len(rois)} code(s) "
                  f"in {time.perf_counter() - t0:.1f} s")

    manifest = out_dir / "crops.csv"
    if rows:
        new = not manifest.exists() or overwrite
        with manifest.open("w" if new else "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
    if verbose:
        print(f"\n{len(rows)} crop(s) from {len(photos)} photo(s) written to {out_dir} "
              f"in {time.perf_counter() - t_all:.0f} s; {n_none} photo(s) without a code"
              + (f"; {n_skip} existing crop(s) kept (use --overwrite to replace)" if n_skip else ""))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Detect and crop Data Matrix codes locally "
                                             "using dm_detector.")
    ap.add_argument("--images", default=str(HERE / "images"),
                    help="input folder with photos (default: images/ next to this script)")
    ap.add_argument("--out", default=str(HERE / "dm_crops_local"),
                    help="output folder for the PNG crops (default: dm_crops_local/)")
    ap.add_argument("--max-codes", type=int, default=5,
                    help="maximum codes kept per photo, best first (default 5)")
    ap.add_argument("--pad", type=float, default=0.2,
                    help="padding around the code as a fraction of its side (default 0.2)")
    ap.add_argument("--overwrite", action="store_true", help="replace existing crop files")
    ap.add_argument("-q", "--quiet", action="store_true", help="only print errors")
    a = ap.parse_args(argv)
    crop_folder(Path(a.images), Path(a.out), max_codes=a.max_codes, pad=a.pad,
                overwrite=a.overwrite, verbose=not a.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
