import numpy as np
import pandas as pd
import pytest

from lmpfoils.geometry import Circle
from lmpfoils.pipeline import flat_field_gain, signal_df


def test_flat_field_gain_normalises_to_one_at_mean():
    ff = np.full((40, 40), 4.0)
    gain = flat_field_gain(ff, margin_px=2)
    np.testing.assert_allclose(gain, 1.0)


def test_flat_field_gain_highlights_nonuniformity():
    ff = np.full((40, 40), 4.0)
    ff[20, 20] = 8.0  # local hotspot
    gain = flat_field_gain(ff, margin_px=2)
    assert gain[20, 20] > gain[0, 20]


def _make_df(det_ids, value, angle=0.0, size=40, r=10.0):
    rows = []
    for det_id in det_ids:
        rows.append({
            "det_id": det_id,
            "raw_data": np.full((size, size), value),
            "det_circle": Circle(x=size / 2, y=size / 2, r=r),
            "det_angle": angle,
        })
    return pd.DataFrame(rows)


def test_signal_df_background_subtraction_and_centering():
    df_data = _make_df([1, 2], value=10.0)
    df_bg = _make_df([1, 2], value=2.0)

    result = signal_df(df_data, df_bg, ff_array=None, analysis_radius=5.0, use_stabilised=False)

    assert "bg_sub" in result.columns
    assert "centered" in result.columns
    assert "rotated" in result.columns
    # background-subtracted value should be close to 10 - 2 = 8 away from NaN border
    center = result.loc[1, "centered"]
    mid = center.shape[0] // 2
    assert center[mid, mid] == pytest.approx(8.0)
