from pathlib import Path

import pytest

from lmpfoils.paths import (data_interim_dir, data_processed_dir, data_raw_dir,
                             data_reference_dir, find_repo_root, reference_h5_path)


def test_find_repo_root_from_nested_start(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    assert find_repo_root(start=nested) == tmp_path


def test_find_repo_root_missing_marker_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_repo_root(start=tmp_path, marker="does-not-exist.marker")


def test_data_dir_helpers_relative_to_repo_root(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    assert data_raw_dir(tmp_path) == tmp_path / "data" / "raw"
    assert data_interim_dir(tmp_path) == tmp_path / "data" / "interim"
    assert data_processed_dir(tmp_path) == tmp_path / "data" / "processed"
    assert data_reference_dir(tmp_path) == tmp_path / "data" / "reference"
    assert reference_h5_path(tmp_path) == tmp_path / "data" / "reference" / "article-plots.h5"


def test_find_repo_root_default_uses_real_repo():
    # sanity check against the actual checked-out repository
    root = find_repo_root()
    assert (root / "lmpfoils").is_dir()
    assert (root / "pyproject.toml").exists()
