#!/usr/bin/env python3
"""dm_demo_app -- single-page GUI demo for the DM detector release files.

Imports ONLY a release file ``release/build<N>/dm_detector_build<N_>.py`` (via importlib; no project
modules). A drop-down at the top lists every release build found in ``release/`` (newest first);
the selected build is imported under its own module name, cached, and used for "Process".

    # from the repository root (the folder that contains release/)
    python release/demo/dm_demo_app.py

Other modes (no clicking needed):
    --list-builds                 print the discovered builds and exit
    --headless IMG [IMG ...]      score images with the selected build, print a summary, no window
    --build 0.01                  pre-select a build (default: newest valid)
    --release-dir DIR             scan another folder instead of release/ (testing)
    --smoke-ms N                  open the window, close it automatically after N ms
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import os
import queue
import re
import sys
import threading
import traceback
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
DEFAULT_RELEASE_DIR = HERE.parent                      # .../DM_detector/release
BUILD_RE = re.compile(r"^build(\d+(?:\.\d+)*)$")       # build0.01, build7 (not build0.01.bak-*, not demo)
IMAGE_TYPES = (("Images", "*.heic *.HEIC *.heif *.HEIF *.jpg *.JPG *.jpeg *.JPEG *.png *.PNG"),
               ("All files", "*"))
THUMB = 150            # px, input thumbnail (longest side)
CROP = 150             # px, code crop display size (longest side)


# --------------------------------------------------------------------------
# Build discovery and loading (no GUI)
# --------------------------------------------------------------------------
@dataclass
class BuildEntry:
    folder: Path
    number: str                       # from the folder name, e.g. "0.01"
    py: Optional[Path]                # release file, None if missing
    reason: str = ""                  # why unavailable

    @property
    def valid(self) -> bool:
        return self.py is not None

    @property
    def sort_key(self):
        try:
            return (Decimal(self.number), tuple(int(x) for x in self.number.split(".")))
        except InvalidOperation:
            return (Decimal(0), tuple(int(x) for x in self.number.split(".")))


def discover_builds(release_dir: Path = DEFAULT_RELEASE_DIR) -> List[BuildEntry]:
    """All ``build<N>`` subfolders of ``release_dir``, newest (highest number) first. A folder is
    valid only if it contains ``dm_detector_build<N with dots as underscores>.py``."""
    out: List[BuildEntry] = []
    if not release_dir.is_dir():
        return out
    for d in release_dir.iterdir():
        if not d.is_dir():
            continue
        m = BUILD_RE.match(d.name)
        if not m:                               # skips demo/, *.bak*, anything else
            continue
        n = m.group(1)
        py = d / ("dm_detector_build%s.py" % n.replace(".", "_"))
        if py.is_file():
            out.append(BuildEntry(d, n, py))
        else:
            out.append(BuildEntry(d, n, None, "no %s" % py.name))
    out.sort(key=lambda e: e.sort_key, reverse=True)
    return out


_MODULE_CACHE: Dict[str, Any] = {}


def load_build(entry: BuildEntry):
    """Import (once) and return the release module of ``entry`` under a unique module name."""
    if not entry.valid:
        raise ImportError("build %s unavailable: %s" % (entry.number, entry.reason))
    key = str(entry.py.resolve())
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    name = "dm_release_%s_%s" % (entry.number.replace(".", "_"), hashlib.sha1(key.encode()).hexdigest()[:8])
    spec = importlib.util.spec_from_file_location(name, key)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                    # needed by dataclasses inside the release file
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    _MODULE_CACHE[key] = mod
    return mod


def supports_crop(mod) -> bool:
    try:
        return "return_crop" in inspect.signature(mod.score_file).parameters
    except (TypeError, ValueError, AttributeError):
        return False


def module_classes(mod) -> List[str]:
    return [str(c) for c in getattr(mod, "CLASSES", ())]


def build_number_of(mod) -> str:
    f = getattr(mod, "get_build_number", None) or getattr(mod, "get_version", None)
    return str(f()) if f else "?"


# --------------------------------------------------------------------------
# Processing one image (no GUI; Pillow only)
# --------------------------------------------------------------------------
def load_thumbnail(path: str, size: int = THUMB):
    from PIL import Image, ImageOps
    if path.lower().endswith((".heic", ".heif")):
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except Exception:
            pass
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            im.draft("RGB", (size * 2, size * 2))
            im = im.convert("RGB")
            im.thumbnail((size, size))
            return im
    except Exception:
        return None


@dataclass
class RowResult:
    path: str
    status: str = ""
    error: Optional[str] = None
    classes: List[str] = field(default_factory=list)
    scores: Optional[Dict[str, float]] = None
    predicted: Optional[str] = None
    thumb: Any = None                 # PIL image or None
    crop: Any = None                  # PIL image or None
    crop_text: str = ""               # shown instead of a crop
    n_codes: int = 0
    n_valid: int = 0


def process_image(mod, path: str) -> RowResult:
    """Score one file with the release module; returns display-ready data (PIL images)."""
    from PIL import Image
    classes = module_classes(mod)
    rr = RowResult(path=path, classes=classes, thumb=load_thumbnail(path))
    crop_ok = supports_crop(mod)
    try:
        r = mod.score_file(path, return_crop=True) if crop_ok else mod.score_file(path)
    except Exception as exc:                  # release files never raise, but be safe
        rr.status, rr.error = "EXCEPTION", "%s: %s" % (type(exc).__name__, exc)
        rr.crop_text = rr.status
        return rr
    rr.status, rr.error = r.get("status", "?"), r.get("error")
    rr.scores, rr.predicted = r.get("scores"), r.get("predicted_class")
    rr.n_codes, rr.n_valid = int(r.get("n_codes") or 0), int(r.get("n_valid_codes") or 0)
    if rr.status != "ok":
        rr.crop_text = rr.status
    elif not crop_ok:
        rr.crop_text = "crop not supported by this build"
    else:
        c = r.get("crop")
        if c is None:
            rr.crop_text = "no crop returned"
        else:
            im = Image.fromarray(c)
            im.thumbnail((CROP, CROP), Image.Resampling.LANCZOS)
            rr.crop = im
    return rr


# --------------------------------------------------------------------------
# GUI (tkinter + Pillow ImageTk)
# --------------------------------------------------------------------------
BAR_COLORS = ("#4c78a8", "#e45756", "#f58518", "#54a24b", "#72b7b2", "#b279a2", "#9d755d", "#bab0ac")


class DemoApp:
    def __init__(self, root, release_dir: Path, preselect: Optional[str] = None):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk, self.root = tk, ttk, root
        self.release_dir = release_dir
        self.files: List[str] = []
        self.entries: List[BuildEntry] = []
        self.mod = None
        self.q: "queue.Queue" = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self.photos: List[Any] = []            # keep ImageTk references alive
        root.title("DM detector demo")
        root.geometry("900x700")

        top = ttk.Frame(root, padding=6)
        top.pack(side="top", fill="x")
        ttk.Label(top, text="Release build:").pack(side="left")
        self.build_var = tk.StringVar()
        self.combo = ttk.Combobox(top, textvariable=self.build_var, state="readonly", width=42)
        self.combo.pack(side="left", padx=4)
        self.combo.bind("<<ComboboxSelected>>", lambda e: self.select_build(self.combo.current()))
        ttk.Button(top, text="Refresh", command=self.refresh_builds).pack(side="left", padx=2)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(top, text="Open images…", command=self.open_files).pack(side="left", padx=2)
        self.process_btn = ttk.Button(top, text="Process", command=self.process)
        self.process_btn.pack(side="left", padx=2)

        mid = ttk.Frame(root, padding=(6, 0))
        mid.pack(side="top", fill="x")
        self.progress = ttk.Progressbar(mid, mode="determinate", length=260)
        self.progress.pack(side="left", pady=4)
        self.files_var = tk.StringVar(value="no images selected")
        ttk.Label(mid, textvariable=self.files_var).pack(side="left", padx=8)

        self.status_var = tk.StringVar(value="")
        ttk.Label(root, textvariable=self.status_var, anchor="w", relief="sunken", padding=3).pack(side="bottom", fill="x")

        # scrollable table
        wrap = ttk.Frame(root)
        wrap.pack(side="top", fill="both", expand=True)
        self.canvas = tk.Canvas(wrap, highlightthickness=0, background="white")
        vs = ttk.Scrollbar(wrap, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vs.set)
        vs.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.table = tk.Frame(self.canvas, background="white")
        self.canvas.create_window((0, 0), window=self.table, anchor="nw")
        self.table.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        self._header()
        self.refresh_builds(preselect)
        root.after(100, self._poll)

    # ---- helpers
    def _on_wheel(self, e):
        d = e.delta if abs(e.delta) < 120 else e.delta // 120
        self.canvas.yview_scroll(int(-d), "units")

    def _header(self):
        for w in self.table.winfo_children():
            w.destroy()
        self.photos.clear()
        self.next_row = 1
        for j, t in enumerate(("Input image", "Code crop (25 px/module)", "Class scores")):
            self.tk.Label(self.table, text=t, font=("TkDefaultFont", 12, "bold"), background="white").grid(
                row=0, column=j, padx=8, pady=4, sticky="w")

    def set_status(self, s: str):
        self.status_var.set(s)

    # ---- builds
    def refresh_builds(self, preselect: Optional[str] = None):
        self.entries = discover_builds(self.release_dir)
        labels = []
        for e in self.entries:
            labels.append("build %s  —  %s" % (e.number, e.py.name if e.valid else "UNAVAILABLE (%s)" % e.reason))
        self.combo["values"] = labels
        if not self.entries:
            self.build_var.set("")
            self.mod = None
            self.set_status("no release builds found in %s" % self.release_dir)
            return
        idx = 0
        if preselect:
            idx = next((i for i, e in enumerate(self.entries) if e.number == preselect), 0)
        elif any(e.valid for e in self.entries):
            idx = next(i for i, e in enumerate(self.entries) if e.valid)   # newest valid
        self.combo.current(idx)
        self.select_build(idx)

    def select_build(self, idx: int):
        if idx < 0 or idx >= len(self.entries):
            return
        e = self.entries[idx]
        self.mod = None
        if not e.valid:
            self.set_status("build %s unavailable: %s" % (e.number, e.reason))
            return
        try:
            self.root.config(cursor="watch"); self.root.update_idletasks()
            mod = load_build(e)
        except BaseException as exc:
            self.set_status("build %s: import error: %s: %s" % (e.number, type(exc).__name__, exc))
            return
        finally:
            self.root.config(cursor="")
        self.mod = mod
        bn = build_number_of(mod)
        vals = list(self.combo["values"])
        vals[idx] = "build %s  —  get_build_number() = %s" % (e.number, bn)
        self.combo["values"] = vals
        self.combo.current(idx)
        self.set_status("loaded %s: get_build_number() = %s; classes %s; crop %s" % (
            e.py.name, bn, ", ".join(module_classes(mod)), "supported" if supports_crop(mod) else "not supported"))

    # ---- files and processing
    def open_files(self):
        from tkinter import filedialog
        fs = filedialog.askopenfilenames(title="Select images", filetypes=IMAGE_TYPES)
        if fs:
            self.files = list(fs)
            self.files_var.set("%d image(s) selected" % len(self.files))

    def process(self, files: Optional[List[str]] = None):
        if files is not None:
            self.files = list(files)
        if self.worker and self.worker.is_alive():
            return
        if self.mod is None:
            self.set_status("select a valid build first")
            return
        if not self.files:
            self.set_status("open some images first")
            return
        self._header()
        mod, files = self.mod, list(self.files)
        self.progress.configure(maximum=len(files), value=0)
        self.process_btn.state(["disabled"]); self.combo.state(["disabled"])
        self.set_status("processing %d image(s) with build %s …" % (len(files), build_number_of(mod)))

        def run():
            for i, p in enumerate(files, 1):
                try:
                    rr = process_image(mod, p)
                except Exception as exc:
                    rr = RowResult(path=p, status="EXCEPTION", error=str(exc), crop_text="EXCEPTION")
                self.q.put(("row", i, len(files), rr))
            self.q.put(("done", len(files)))
        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "row":
                    _, i, n, rr = msg
                    self._add_row(rr)
                    self.progress.configure(value=i)
                    self.set_status("processed %d / %d: %s → %s" % (i, n, os.path.basename(rr.path),
                                                                   rr.predicted or rr.status))
                else:
                    self.process_btn.state(["!disabled"]); self.combo.state(["!disabled", "readonly"])
                    self.set_status("done: %d image(s) with build %s" % (msg[1], build_number_of(self.mod)))
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _add_row(self, rr: RowResult):
        from PIL import ImageTk
        tk, r = self.tk, self.next_row
        self.next_row += 1
        bg = "white" if r % 2 else "#f4f4f4"
        f0 = tk.Frame(self.table, background=bg)
        f0.grid(row=r, column=0, sticky="nsew", padx=4, pady=3)
        if rr.thumb is not None:
            ph = ImageTk.PhotoImage(rr.thumb); self.photos.append(ph)
            tk.Label(f0, image=ph, background=bg).pack()
        else:
            tk.Label(f0, text="(no preview)", width=18, height=6, background=bg).pack()
        tk.Label(f0, text=os.path.basename(rr.path), background=bg, font=("TkDefaultFont", 10)).pack()
        f1 = tk.Frame(self.table, background=bg, width=CROP + 20, height=THUMB + 20)
        f1.grid(row=r, column=1, sticky="nsew", padx=4, pady=3)
        if rr.crop is not None:
            ph = ImageTk.PhotoImage(rr.crop); self.photos.append(ph)
            tk.Label(f1, image=ph, background=bg).pack(expand=True)
        else:
            tk.Label(f1, text=rr.crop_text or rr.status, fg="#b00020" if rr.status != "ok" else "#555",
                     background=bg, wraplength=CROP + 10, justify="center",
                     font=("TkDefaultFont", 12, "bold")).pack(expand=True, pady=40)
        self._bars(r, rr, bg)

    def _bars(self, r: int, rr: RowResult, bg: str):
        tk = self.tk
        classes = rr.classes
        W, rowh, lab, valw = 330, 26, 60, 60
        H = max(1, len(classes)) * rowh + 16
        c = tk.Canvas(self.table, width=W, height=H, background=bg, highlightthickness=0)
        c.grid(row=r, column=2, sticky="w", padx=4, pady=3)
        if not rr.scores:
            c.create_text(W // 2, H // 2, text="no scores (%s)" % rr.status, fill="#b00020")
            return
        top = max(classes, key=lambda k: rr.scores.get(k, 0.0))
        span = W - lab - valw
        for i, k in enumerate(classes):
            v = float(rr.scores.get(k, 0.0))
            y0 = 8 + i * rowh
            is_top = k == top
            c.create_text(lab - 6, y0 + rowh / 2 - 2, text=k, anchor="e",
                          font=("TkDefaultFont", 11, "bold" if is_top else "normal"))
            c.create_rectangle(lab, y0 + 3, lab + span, y0 + rowh - 7, outline="#ddd", fill="#eee")
            col = "#d62728" if is_top else BAR_COLORS[i % len(BAR_COLORS)]
            c.create_rectangle(lab, y0 + 3, lab + max(1, span * v), y0 + rowh - 7,
                               outline="black" if is_top else "", width=2 if is_top else 1, fill=col)
            c.create_text(lab + span + 6, y0 + rowh / 2 - 2, anchor="w", text="%.3f" % v,
                          font=("TkDefaultFont", 11, "bold" if is_top else "normal"))


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-dir", default=str(DEFAULT_RELEASE_DIR))
    ap.add_argument("--build", default=None, help="build number to pre-select, e.g. 0.01")
    ap.add_argument("--list-builds", action="store_true")
    ap.add_argument("--headless", nargs="+", metavar="IMG")
    ap.add_argument("--smoke-ms", type=int, default=0)
    ap.add_argument("--smoke-process", nargs="*", metavar="IMG", help="with --smoke-ms: process these images in the window")
    a = ap.parse_args(argv)
    rel = Path(a.release_dir).expanduser()
    entries = discover_builds(rel)
    if a.list_builds or a.headless:
        for e in entries:
            print("build %-8s %-12s %s" % (e.number, "valid" if e.valid else "UNAVAILABLE", e.py or e.reason))
    if a.list_builds and not a.headless:
        return 0
    if a.headless:
        cand = [e for e in entries if e.valid and (a.build is None or e.number == a.build)]
        if not cand:
            print("no valid build"); return 2
        mod = load_build(cand[0])
        print("using build %s (get_build_number() = %s), classes %s, crop %s" % (
            cand[0].number, build_number_of(mod), module_classes(mod), supports_crop(mod)))
        for p in a.headless:
            rr = process_image(mod, p)
            print("%s: status %s, predicted %s, scores %s, codes %d/%d valid, thumb %s, crop %s" % (
                os.path.basename(p), rr.status, rr.predicted,
                {k: round(v, 4) for k, v in (rr.scores or {}).items()}, rr.n_valid, rr.n_codes,
                None if rr.thumb is None else rr.thumb.size,
                rr.crop.size if rr.crop is not None else repr(rr.crop_text)))
        return 0
    import tkinter as tk
    root = tk.Tk()
    app = DemoApp(root, rel, a.build)
    if a.smoke_ms:
        if a.smoke_process:
            root.after(300, lambda: app.process(a.smoke_process))

        def finish():
            if app.worker and app.worker.is_alive():
                root.after(500, finish); return
            print("smoke: status line:", app.status_var.get())
            print("smoke: rows:", app.next_row - 1, "builds:", list(app.combo["values"]))
            root.destroy()
        root.after(a.smoke_ms, finish)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
