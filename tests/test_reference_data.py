import os

import pandas as pd
import pytest

from lmpfoils.reference_data import (bp_experimental_dose_gy, load_reference_data,
                                      sobp_experimental_dose_gy)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H5_PATH = os.path.join(REPO_ROOT, "data", "reference", "article-plots.h5")

pytestmark = pytest.mark.skipif(not os.path.exists(H5_PATH), reason="article-plots.h5 not present")


def test_load_reference_data_all_tables_present():
    ref = load_reference_data(H5_PATH)
    for name in ref._fields:
        df = getattr(ref, name)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


def test_bp_experimental_dose_gy_normalisation():
    ref = load_reference_data(H5_PATH)
    out = bp_experimental_dose_gy(ref.bp_exp, ref_dose_gy=12.0)
    assert out.dose_Gy.iloc[0] == pytest.approx(12.0)


def test_sobp_experimental_dose_gy_normalisation():
    ref = load_reference_data(H5_PATH)
    out = sobp_experimental_dose_gy(ref.sobp_exp, ref_dose_gy=60.0)
    assert "dose_Gy" in out.columns
    assert out.dose_Gy.notna().sum() > 0
