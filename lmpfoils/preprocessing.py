"""Raw-TIFF-to-interim-data preprocessing: constant-background removal,
detector-circle finding and angular-alignment, for a single irradiation
dataset directory.

Ported from the Snakemake rules in ``ifj_lmp_foils/rules/raw.smk`` and
``ifj_lmp_foils/rules/find_det.smk``. The original pipeline cached each
step's output as a separate file tracked by Snakemake; here the same steps
are exposed as a single function per foil (:func:`preprocess_foil`) plus a
driver (:func:`preprocess_dataset`) that iterates over every foil id in a
raw dataset directory and writes the ``raw.npy`` / ``det-circle.json`` /
``angle.npy`` sidecar files that :func:`lmpfoils.dataset.read_df` expects,
alongside a copy of the raw dataset's ``Pos0/metadata.txt``.

Two raw sub-folders exist per foil position:

- ``<dataset_dir>/<det_id>/`` -- the full (long) proton-irradiation exposure.
- ``<dataset_dir>/<det_id>lv/`` -- a short "live view" exposure used only to
  locate the foil (circle finding + rotation alignment); it is *not* part of
  the dose signal.
"""
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import numpy as np

from lmpfoils.alignment import get_angle_with_min_value, get_line_circle
from lmpfoils.detector import find_circle_hough_method, img_for_circle_detection
from lmpfoils.geometry import Circle
from lmpfoils.image_io import read_tiff_img, subtract_background


def _tiff_path(dataset_dir: Path, det_id: str) -> Path:
    return dataset_dir / det_id / "Pos0" / "img_000000000_Default_000.tif"


def preprocess_foil(dataset_dir: Path, det_id: str, out_dir: Path,
                     max_angle_deg: float = 360.0) -> None:
    """Process one foil position (``det_id``) of a raw dataset directory,
    writing ``out_dir/<det_id>/{raw.npy,metadata.txt}`` and
    ``out_dir/<det_id>lv/{det-circle.json,angle.npy}``.

    Mirrors, in order: ``read_tiff`` -> ``copy_metadata`` (full exposure);
    ``read_tiff`` -> ``background_constant_subtraction`` ->
    ``image_contour`` -> ``detector_circle`` -> ``signal_on_circle``
    (live-view exposure).
    """
    dataset_dir = Path(dataset_dir)
    out_dir = Path(out_dir)

    # --- full exposure: just cache the raw array + metadata ---
    full_out = out_dir / det_id
    full_out.mkdir(parents=True, exist_ok=True)
    raw = read_tiff_img(_tiff_path(dataset_dir, det_id), border_px=0)
    np.save(full_out / "raw.npy", raw)
    metadata_src = dataset_dir / det_id / "Pos0" / "metadata.txt"
    if metadata_src.exists():
        shutil.copy(metadata_src, full_out / "metadata.txt")

    # --- live-view exposure: locate + align the foil circle ---
    lv_id = f"{det_id}lv"
    lv_tiff = _tiff_path(dataset_dir, lv_id)
    if not lv_tiff.exists():
        logging.warning(f"No live-view exposure found for foil {det_id} at {lv_tiff}; skipping alignment")
        return
    lv_out = out_dir / lv_id
    lv_out.mkdir(parents=True, exist_ok=True)

    raw_lv = read_tiff_img(lv_tiff, border_px=0)
    const_bg = float(np.nanmin(raw_lv))
    bg_removed = subtract_background(input=raw_lv, const_bg=const_bg)
    thresholded = img_for_circle_detection(bg_removed)
    det_circle = find_circle_hough_method(thresholded)
    det_circle.save_json(lv_out / "det-circle.json")

    if det_circle.proper:
        meas_radius = max(0.0, det_circle.r - 60)
        meas_circle = Circle(det_circle.x, det_circle.y, meas_radius)
        angles, values = get_line_circle(raw_lv, circ=meas_circle, step_angle_deg=0.1,
                                          max_angle_deg=max_angle_deg)
        values[values == 0.0] = np.nan
        angle_with_min_value, _ = get_angle_with_min_value(angles, values, median_filter_size=10)
        np.save(lv_out / "angle.npy", np.array(angle_with_min_value))


def preprocess_dataset(dataset_dir: Path, out_dir: Path, det_ids: Optional[List[str]] = None,
                        max_angle_deg: float = 360.0, max_workers: int = 8) -> None:
    """Preprocess every foil position in ``dataset_dir`` (all numeric
    sub-folder names, or the explicit ``det_ids`` list), writing results
    under ``out_dir``. Runs foils in parallel via a thread pool since each
    foil's Hough-circle detection is independent I/O + OpenCV work."""
    dataset_dir = Path(dataset_dir)
    out_dir = Path(out_dir)
    if det_ids is None:
        det_ids = sorted((entry.name for entry in dataset_dir.iterdir() if entry.name.isdigit()),
                          key=lambda s: int(s))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(lambda det_id: preprocess_foil(dataset_dir, det_id, out_dir, max_angle_deg), det_ids))
