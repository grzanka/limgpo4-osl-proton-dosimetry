import json

import numpy as np
import pytest

from lmpfoils.dataset import list_detector_ids, read_df, load_dataset
from lmpfoils.geometry import Circle


def _make_fake_detector_dir(base, det_id, value=5.0, r=10.0, with_angle=True):
    det_dir = base / str(det_id)
    det_dir.mkdir()
    img = np.full((40, 40), value)
    np.save(det_dir / "raw.npy", img)
    metadata = {"Summary": {"Time": "2022-11-17 10:00:00 +0100"}}
    (det_dir / "metadata.txt").write_text(json.dumps(metadata), encoding="ISO-8859-1")

    lv_dir = base / f"{det_id}lv"
    lv_dir.mkdir()
    Circle(x=20, y=20, r=r).save_json(lv_dir / "det-circle.json")
    if with_angle:
        np.save(lv_dir / "angle.npy", np.array(0.0))


def test_list_detector_ids_sorted_numerically(tmp_path):
    for det_id in [10, 2, 1]:
        _make_fake_detector_dir(tmp_path, det_id)
    ids = list_detector_ids(tmp_path)
    assert ids == ["1", "2", "10"]


def test_read_df_basic_columns(tmp_path):
    for det_id in [1, 2]:
        _make_fake_detector_dir(tmp_path, det_id, value=float(det_id) * 10)
    ids = list_detector_ids(tmp_path)
    df = read_df(tmp_path, ids)

    assert list(df.det_id) == [1, 2]
    assert "raw_mean_center" in df.columns
    assert "det_angle" in df.columns
    np.testing.assert_allclose(df.raw_mean_center.values, [10.0, 20.0])


def test_read_df_with_analysis_radius(tmp_path):
    _make_fake_detector_dir(tmp_path, 1, value=7.0)
    df = read_df(tmp_path, ["1"], analysis_radius=5.0)
    assert "raw_mean" in df.columns
    assert df.raw_mean.iloc[0] == pytest.approx(7.0)


def test_load_dataset_convenience(tmp_path):
    _make_fake_detector_dir(tmp_path, 1, value=3.0)
    df = load_dataset(tmp_path)
    assert len(df) == 1
    assert df.raw_mean_center.iloc[0] == pytest.approx(3.0)
