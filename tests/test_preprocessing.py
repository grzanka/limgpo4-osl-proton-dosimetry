"""Tests for lmpfoils.preprocessing using a synthetic single-foil dataset
(no dependency on the real raw-data TIFFs, so this always runs in CI)."""
import json

import numpy as np
import pytest
from PIL import Image

from lmpfoils.dataset import read_df
from lmpfoils.preprocessing import preprocess_dataset, preprocess_foil


def _write_foil(dataset_dir, det_id, value, size=750, disc_radius=305):
    """Write a synthetic <det_id>/Pos0/{img...tif,metadata.txt} pair with a
    dark disc (the foil) of the given pixel value on a brighter, mildly
    noisy background, plus an identical 'lv' (live-view) exposure used for
    circle finding. Image size and disc radius are chosen to sit inside the
    hardcoded Hough-detection bounds (minRadius=300, maxRadius=600)."""
    rng = np.random.default_rng(0)
    for suffix in ("", "lv"):
        folder = dataset_dir / f"{det_id}{suffix}" / "Pos0"
        folder.mkdir(parents=True, exist_ok=True)

        arr = 1000 + rng.normal(0, 5, size=(size, size))
        yy, xx = np.ogrid[:size, :size]
        mask = (xx - size / 2) ** 2 + (yy - size / 2) ** 2 <= disc_radius ** 2
        arr[mask] = value + rng.normal(0, 5, size=int(mask.sum()))
        arr = np.clip(arr, 0, 65535).astype(np.uint16)
        Image.fromarray(arr).save(folder / "img_000000000_Default_000.tif")

        metadata = {"Summary": {"Time": "2022-11-17 14:38:43 +0100"}}
        (folder / "metadata.txt").write_text(json.dumps(metadata))


@pytest.fixture
def synthetic_dataset(tmp_path):
    dataset_dir = tmp_path / "raw" / "demo"
    _write_foil(dataset_dir, "1", value=1000)
    return dataset_dir


def test_preprocess_foil_writes_expected_files(synthetic_dataset, tmp_path):
    out_dir = tmp_path / "interim" / "demo"
    preprocess_foil(synthetic_dataset, "1", out_dir, max_angle_deg=30.0)

    assert (out_dir / "1" / "raw.npy").exists()
    assert (out_dir / "1" / "metadata.txt").exists()
    assert (out_dir / "1lv" / "det-circle.json").exists()
    # a tiny (30 deg) sweep still produces a finite alignment angle
    assert (out_dir / "1lv" / "angle.npy").exists()


def test_preprocess_dataset_then_read_df(synthetic_dataset, tmp_path):
    out_dir = tmp_path / "interim" / "demo"
    preprocess_dataset(synthetic_dataset, out_dir, max_angle_deg=30.0, max_workers=2)

    df = read_df(out_dir, ["1"], analysis_radius=50)
    assert len(df) == 1
    assert df.det_circle.iloc[0].proper
    assert df.raw_mean_center.iloc[0] > 0
