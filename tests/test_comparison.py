import numpy as np
import pandas as pd

from lmpfoils.comparison import perc_error_to_ref_df


def test_perc_error_zero_when_matching_reference():
    z = np.linspace(0, 10, 11)
    y = z**2 + 1.0  # keep above threshold everywhere
    df = pd.DataFrame({"depth": z, "dose": y})
    ref_df = pd.DataFrame({"depth": z, "dose": y})

    result = perc_error_to_ref_df(df, "depth", "dose", ref_df, "depth", "dose", threshold=0.01)
    np.testing.assert_allclose(result["perc_err"].values, 0.0, atol=1e-8)


def test_perc_error_detects_offset_and_drops_low_values():
    z = np.linspace(0, 10, 11)
    y = np.full_like(z, 2.0)
    y[0] = 0.0  # below threshold, should be dropped
    ref_y = np.full_like(z, 1.0)

    df = pd.DataFrame({"depth": z, "dose": y})
    ref_df = pd.DataFrame({"depth": z, "dose": ref_y})

    result = perc_error_to_ref_df(df, "depth", "dose", ref_df, "depth", "dose", threshold=0.5)
    assert len(result) == len(z) - 1
    np.testing.assert_allclose(result["perc_err"].values, 100.0)
