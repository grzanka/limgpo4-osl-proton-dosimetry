"""Helpers for comparing LMP-foil results against reference curves
(Markus ionisation-chamber measurements, PHITS Monte Carlo simulations).

Ported from ``ifj_lmp_foils/src/data/analysis.py`` (``perc_error_to_ref_df``).
"""
import numpy as np
import pandas as pd


def perc_error_to_ref_df(df: pd.DataFrame,
                          xcolname: str,
                          ycolname: str,
                          ref_df: pd.DataFrame,
                          ref_xcolname: str,
                          ref_ycolname: str,
                          threshold: float = 0.01) -> pd.DataFrame:
    """Percentage difference of ``df[ycolname]`` relative to ``ref_df``,
    linearly interpolated onto ``df[xcolname]``.

    Points where ``df[ycolname] <= threshold`` are dropped to avoid huge
    relative errors from near-zero reference values (e.g. far outside the
    Bragg peak / SOBP range).
    """
    keep = df[ycolname] > threshold
    rel_error = df[ycolname][keep] / np.interp(
        x=df[xcolname][keep], xp=ref_df[ref_xcolname], fp=ref_df[ref_ycolname])
    result = pd.DataFrame({'z': df[xcolname][keep], 'perc_err': 100 * (rel_error - 1.0)})
    return result
