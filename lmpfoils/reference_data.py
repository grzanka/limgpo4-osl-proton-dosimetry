"""Reading external reference curves (Markus ionization-chamber
measurements, PHITS Monte Carlo simulations) bundled in
``data/reference/article-plots.h5``.

See ``data/reference/README.md`` for the full table-by-table description
and provenance notes.
"""
from typing import NamedTuple

import pandas as pd

from lmpfoils.image_io import PathLike

_KEYS = (
    "bp_exp", "bp_mc", "bp_kin_en", "bp_kin_en_spectrum",
    "sobp_exp", "sobp_mc", "sobp_kin_en", "sobp_kin_en_spectrum",
)


class ReferenceData(NamedTuple):
    """Container for all tables in ``article-plots.h5``."""
    bp_exp: pd.DataFrame
    bp_mc: pd.DataFrame
    bp_kin_en: pd.DataFrame
    bp_kin_en_spectrum: pd.DataFrame
    sobp_exp: pd.DataFrame
    sobp_mc: pd.DataFrame
    sobp_kin_en: pd.DataFrame
    sobp_kin_en_spectrum: pd.DataFrame


def load_reference_data(h5_path: PathLike) -> ReferenceData:
    """Load all reference tables from ``article-plots.h5`` into a
    :class:`ReferenceData` namedtuple of DataFrames."""
    with pd.HDFStore(h5_path, mode="r") as store:
        tables = {key: store[f"/{key}"] for key in _KEYS}
    return ReferenceData(**tables)


def bp_experimental_dose_gy(bp_exp: pd.DataFrame, ref_dose_gy: float = 12.0) -> pd.DataFrame:
    """Rescale the raw Markus-chamber BP signal to absolute dose [Gy],
    normalised so the first (surface) point equals ``ref_dose_gy``.

    Mirrors ``df_bp_exp['dose_Gy'] = 12 * df_bp_exp.Signal / df_bp_exp.Signal.iloc[0]``
    from ``6-efficiency.ipynb`` / ``7-efficiency_fitting.ipynb``.
    """
    out = bp_exp.copy()
    out["dose_Gy"] = ref_dose_gy * out.Signal / out.Signal.iloc[0]
    return out


def sobp_experimental_dose_gy(sobp_exp: pd.DataFrame, ref_dose_gy: float = 60.0,
                               smoothing_window: int = 9) -> pd.DataFrame:
    """Rescale the raw Markus-chamber SOBP signal to absolute dose [Gy],
    smoothed with a centered rolling-mean window (to suppress the finer
    ionization-chamber structure not resolved in the LMP-foil measurement).

    Mirrors ``df_sobp_exp['dose_Gy'] = 60 * df_sobp_exp.Signal.rolling(window=9,
    center=True).mean()``.
    """
    out = sobp_exp.copy()
    out["dose_Gy"] = ref_dose_gy * out.Signal.rolling(window=smoothing_window, center=True).mean()
    return out
