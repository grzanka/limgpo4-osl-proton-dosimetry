"""Build per-foil analysis dataframes from raw micro-manager image folders.

Ported from the ``read_df`` helper duplicated across the exploratory
notebooks (``3-flat_field.ipynb`` through ``8-article-plots.ipynb``).

The expected raw-data layout, per irradiation dataset, is::

    <dataset_dir>/<det_id>/raw.npy          # background-corrected raw image
    <dataset_dir>/<det_id>/metadata.txt     # Micro-Manager JSON metadata
    <dataset_dir>/<det_id>lv/det-circle.json  # detected foil circle (px)
    <dataset_dir>/<det_id>lv/angle.npy         # alignment angle (deg), optional

where ``det_id`` is the numeric foil/detector index.
"""
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from lmpfoils.geometry import Circle, get_mean_std
from lmpfoils.image_io import get_timestamp


def list_detector_ids(path: Path) -> List[str]:
    """List numeric detector-id subfolder names in a raw dataset directory,
    sorted numerically."""
    return sorted((entry.name for entry in path.iterdir() if entry.name.isdigit()), key=lambda s: int(s))


def read_df(path: Path, det_names: List[str], analysis_radius: Optional[float] = None) -> pd.DataFrame:
    """Read raw per-foil data and detector geometry into a DataFrame.

    Parameters
    ----------
    path:
        Directory holding one subfolder per detector id.
    det_names:
        Detector id strings (as produced by :func:`list_detector_ids`).
    analysis_radius:
        If given, an additional ``analysis_circle`` (same center as the
        detected foil circle but with this radius) is used to compute
        ``raw_mean`` / ``raw_std`` over a possibly smaller/larger region
        than the detected foil. If omitted, only the detected-circle
        (``raw_mean_center`` / ``raw_std_center``) statistics are computed.
    """
    df = pd.DataFrame()
    df["det_id"] = det_names
    df.det_id = df.det_id.astype('uint8')
    df["raw_data"] = df.det_id.apply(lambda id: np.load(path / f"{id}" / "raw.npy"))
    df["timestamp"] = df.det_id.apply(lambda id: get_timestamp(path / f"{id}" / "metadata.txt"))
    df["det_circle"] = df.det_id.apply(lambda x: Circle.from_json(path / f"{x}lv" / "det-circle.json"))

    angle_path_exists = all((path / f"{d}lv" / "angle.npy").exists() for d in det_names) if det_names else False
    if angle_path_exists:
        df["det_angle"] = df.det_id.apply(lambda id: np.load(path / f"{id}lv" / "angle.npy"))

    df["raw_mean_center"] = df.apply(lambda tmpdf: get_mean_std(tmpdf.raw_data, tmpdf.det_circle)[0], axis=1)
    df["raw_std_center"] = df.apply(lambda tmpdf: get_mean_std(tmpdf.raw_data, tmpdf.det_circle)[1], axis=1)

    if analysis_radius is not None:
        df['analysis_circle'] = df.det_circle.apply(lambda c: Circle(c.x, c.y, analysis_radius))
        df["raw_mean"] = df.apply(lambda tmpdf: get_mean_std(tmpdf.raw_data, tmpdf.analysis_circle)[0], axis=1)
        df["raw_std"] = df.apply(lambda tmpdf: get_mean_std(tmpdf.raw_data, tmpdf.analysis_circle)[1], axis=1)

    return df


def load_dataset(dataset_dir: Path, analysis_radius: Optional[float] = None) -> pd.DataFrame:
    """Convenience wrapper: list detector ids under ``dataset_dir`` and read
    the resulting per-foil DataFrame."""
    dataset_dir = Path(dataset_dir)
    det_ids = list_detector_ids(dataset_dir)
    return read_df(dataset_dir, det_ids, analysis_radius=analysis_radius)
