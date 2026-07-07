"""Signal-processing pipeline: background subtraction, flat-field
correction, re-centering and de-rotation of per-foil images.

Ported from the ``singal_df`` helper duplicated across the exploratory
notebooks (``4-from_raw_to_signal.ipynb`` through ``8-article-plots.ipynb``).
This is the core step that turns raw per-foil TIFF-derived arrays (already
loaded via :func:`lmpfoils.dataset.read_df`) into dose-comparable, aligned
images.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import ndimage

from lmpfoils.geometry import Circle, create_circular_mask


def _shift(data: npt.NDArray, x: float, y: float) -> npt.NDArray:
    return ndimage.shift(data, (data.shape[1] / 2 - y, data.shape[0] / 2 - x), cval=np.nan, prefilter=False)


def _rotate(data: npt.NDArray, angle_deg: float) -> npt.NDArray:
    return ndimage.rotate(data, -angle_deg, cval=np.nan, reshape=False, prefilter=False)


def stabilize_sensor_drift(df_bg: pd.DataFrame, df_signal: pd.DataFrame) -> pd.DataFrame:
    """Fit a linear sensor-stability drift ``raw_mean_center(t)`` from a
    background-only dataset (``df_bg``) and use it to detrend
    ``df_signal.raw_data`` in time, returning ``df_signal`` with a new
    ``raw_data_stabilised`` column.

    Mirrors the ``a, b = np.polyfit(...)`` / ``raw_data_stabilised`` cells
    repeated across the exploratory notebooks. Pass ``df_signal is df_bg``
    (or an equivalent background dataframe) to stabilize the background set
    itself.
    """
    x = df_bg.timestamp.apply(lambda t: (t - df_bg.timestamp.min()).total_seconds()).values
    y = df_bg.raw_mean_center.values
    a, b = np.polyfit(x, y, 1)

    df_out = df_signal.copy()
    t0 = df_signal.timestamp.min()
    df_out["raw_data_stabilised"] = df_out.apply(
        lambda row: row.raw_data - a * (row.timestamp - t0).total_seconds(), axis=1)
    return df_out, a, b


def flat_field_gain(ff_array: npt.NDArray, margin_px: float = 10) -> npt.NDArray:
    """Per-pixel flat-field gain map, normalised to 1 over a large circular
    region centered on the flat-field image (excluding a ``margin_px``
    border)."""
    big_circle = Circle(x=ff_array.shape[1] / 2, y=ff_array.shape[0] / 2, r=ff_array.shape[1] / 2 - margin_px)
    mask = create_circular_mask(img=ff_array, circle_px=big_circle)
    gain = ff_array / np.mean(ff_array[mask])
    return gain


def signal_df(df_data: pd.DataFrame,
              df_bg: pd.DataFrame,
              ff_array: Optional[npt.NDArray] = None,
              analysis_radius: float = 300,
              use_stabilised: bool = True) -> pd.DataFrame:
    """Background-subtract, flat-field correct, re-center and de-rotate the
    per-foil images in ``df_data``.

    Parameters
    ----------
    df_data:
        Output of :func:`lmpfoils.dataset.read_df` for the irradiation of
        interest; must include ``raw_data``, ``det_circle``, ``det_angle``
        and (if ``use_stabilised``) ``raw_data_stabilised``.
    df_bg:
        Matching background dataset (same detector ids), with
        ``raw_data``/``raw_data_stabilised``.
    ff_array:
        Flat-field-corrected reference image (e.g. LED flood image), or
        ``None`` to skip flat-field correction.
    analysis_radius:
        Radius (px) of the circular ROI used for the *centered* image
        analysis circle.
    use_stabilised:
        If True, use ``raw_data_stabilised`` (post sensor-drift correction)
        for background subtraction; otherwise use raw ``raw_data``.

    Returns
    -------
    DataFrame indexed by ``det_id`` with new columns ``bg_sub``,
    ``bg_sub_stabilized``, ``sensor_corr``, ``centered``, ``rotated`` and the
    per-image ``analysis_circle_centered`` / ``detector_circle_centered``.
    """
    df_signal = df_data.copy()
    df_signal.set_index("det_id", inplace=True)

    df_bg = df_bg.copy()
    df_bg.set_index("det_id", inplace=True)

    # remove background
    df_signal["bg_sub"] = df_signal.raw_data - df_bg.raw_data
    if "raw_data_stabilised" in df_signal.columns and "raw_data_stabilised" in df_bg.columns:
        df_signal["bg_sub_stabilized"] = df_signal.raw_data_stabilised - df_bg.raw_data_stabilised

    # apply flat-field correction
    corrected_col = "bg_sub_stabilized" if (use_stabilised and "bg_sub_stabilized" in df_signal.columns) else "bg_sub"
    df_signal["sensor_corr"] = df_signal[corrected_col]
    if ff_array is not None:
        gain = flat_field_gain(ff_array)
        df_signal["sensor_corr"] = df_signal[corrected_col].apply(lambda x: x / gain)

    # move to center, using multiple threads (one per available CPU)
    no_of_cpus = os.cpu_count() or 1

    def apply_shift(df: pd.DataFrame) -> pd.Series:
        return pd.Series(
            [_shift(row.sensor_corr, row.det_circle.x, row.det_circle.y) for row in df.itertuples()],
            index=df.index, dtype=object)

    with ThreadPoolExecutor() as executor:
        chunks_of_shifted_df = executor.map(apply_shift, np.array_split(df_signal, no_of_cpus))
    df_signal["centered"] = pd.concat(chunks_of_shifted_df)

    df_signal["analysis_circle_centered"] = df_signal.apply(
        lambda x: Circle(x.centered.shape[0] // 2, x.centered.shape[1] // 2, analysis_radius), axis=1)
    df_signal["detector_circle_centered"] = df_signal.apply(
        lambda x: Circle(x.centered.shape[0] // 2, x.centered.shape[1] // 2, x.det_circle.r), axis=1)

    # rotate to align foil orientation, if per-foil angle is available
    if "det_angle" in df_signal.columns:
        def apply_rotate(df: pd.DataFrame) -> pd.Series:
            return pd.Series(
                [_rotate(row.centered, row.det_angle) for row in df.itertuples()],
                index=df.index, dtype=object)

        with ThreadPoolExecutor() as executor:
            chunks_of_rotated_df = executor.map(apply_rotate, np.array_split(df_signal, no_of_cpus))
        df_signal["rotated"] = pd.concat(chunks_of_rotated_df)
    else:
        df_signal["rotated"] = df_signal["centered"]

    return df_signal
