"""Repository-root discovery and standard data-directory paths.

Notebooks in ``notebooks/`` are executed from varying working directories
depending on the runner (Jupyter, papermill, nbclient/CI), so path
resolution walks upward from the current working directory (or an
explicit ``start``) looking for the repository marker file
(``pyproject.toml``), rather than assuming a fixed relative path.
"""
from pathlib import Path
from typing import Optional


def find_repo_root(start: Optional[Path] = None, marker: str = "pyproject.toml") -> Path:
    """Walk upward from ``start`` (default: current working directory)
    until a directory containing ``marker`` is found, and return it.

    Raises
    ------
    FileNotFoundError
        If no ancestor directory contains ``marker``.
    """
    here = Path(start if start is not None else Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate repository root (no ancestor of {here} contains '{marker}')")


def data_raw_dir(repo_root: Optional[Path] = None) -> Path:
    return (repo_root or find_repo_root()) / "data" / "raw"


def data_interim_dir(repo_root: Optional[Path] = None) -> Path:
    return (repo_root or find_repo_root()) / "data" / "interim"


def data_processed_dir(repo_root: Optional[Path] = None) -> Path:
    return (repo_root or find_repo_root()) / "data" / "processed"


def data_reference_dir(repo_root: Optional[Path] = None) -> Path:
    return (repo_root or find_repo_root()) / "data" / "reference"


def reference_h5_path(repo_root: Optional[Path] = None) -> Path:
    return data_reference_dir(repo_root) / "article-plots.h5"
