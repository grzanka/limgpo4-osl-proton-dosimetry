import json

import numpy as np
import pytest

from lmpfoils.image_io import median_filter, subtract_background, get_timestamp


def test_median_filter_smooths_spike():
    img = np.zeros((20, 20))
    img[10, 10] = 100.0
    filtered = median_filter(img, size=3)
    assert filtered[10, 10] < img[10, 10]


def test_subtract_background_removes_offset_and_clips_negatives():
    img = np.array([[5.0, 2.0], [10.0, 0.0]])
    bg = np.array([[1.0, 1.0], [1.0, 1.0]])
    out = subtract_background(img, img_bg=bg, const_bg=1.0)
    # (5-1-1, 2-1-1, 10-1-1, 0-1-1) -> clipped at 0
    expected = np.array([[3.0, 0.0], [8.0, 0.0]])
    np.testing.assert_allclose(out, expected)


def test_subtract_background_no_bg_image():
    img = np.array([[5.0, -2.0]])
    out = subtract_background(img, img_bg=None, const_bg=0.0)
    np.testing.assert_allclose(out, [[5.0, 0.0]])


def test_get_timestamp_parses_metadata(tmp_path):
    metadata = {"Summary": {"Time": "2022-11-17 10:30:00 +0100"}}
    p = tmp_path / "metadata.txt"
    p.write_text(json.dumps(metadata), encoding="ISO-8859-1")
    ts = get_timestamp(p)
    assert ts.year == 2022
    assert ts.month == 11
    assert ts.day == 17
    assert ts.hour == 10
