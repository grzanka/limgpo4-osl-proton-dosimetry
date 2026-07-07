"""TIFF reading, median filtering and background subtraction.

Ported from ``ifj_lmp_foils/src/data/analysis.py``
(``read_tiff_img``, ``median_filter``, ``subtract_background``) and
notebook helpers (``get_timestamp``).
"""
import datetime
import json
import logging
import os
from typing import Optional, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy import ndimage

PathLike = TypeVar("PathLike", str, bytes, os.PathLike)


def read_tiff_img(file_path: PathLike, border_px: int = 50) -> npt.NDArray:
    """Read a tif image and add a border filled with NaN.

    The new image will have bigger size (by 2 x border_px in each direction)
    than the original tiff. Note: input file contains 16-bit integers, but
    we intentionally cast the output to the 32-bit float array, the reason
    is that we want to pad the array with NaNs which are transparent to
    methods like min/max/mean; there is no NaN for integers, therefore we
    are forced to use floats despite int-like values.
    """
    logging.info(f'Reading file {file_path}')
    raw_img = plt.imread(str(file_path)).astype('float32')
    logging.info(f'Original image shape: {raw_img.shape}, min value {raw_img.min()}, max value {raw_img.max()}')
    padded_img = np.pad(raw_img, pad_width=border_px, constant_values=np.nan)
    logging.info(f'Padded image shape: {padded_img.shape}')
    return padded_img


def median_filter(input: npt.NDArray, size: int = 10, gpu: bool = False) -> npt.NDArray:
    """Apply median filter (optionally on GPU via cupy, if available)."""
    logging.info('Before median filter ' +
                 f'min {np.nanmin(input)}, mean max {np.nanmean(input):3.3f}, max {np.nanmax(input)}')
    output = np.empty(shape=1)
    if gpu:
        try:
            import cupy as cp
            from cupyx.scipy.ndimage import median_filter as median_filter_gpu
            output = median_filter_gpu(cp.asarray(input), size=size).get()
        except (ModuleNotFoundError, ImportError):
            logging.warning('GPU mode selected and no `cupy` library installed')
            return np.full_like(input, np.nan, dtype=np.double)
    else:
        output = ndimage.median_filter(input, size=size)
    logging.info('After median filter ' +
                 f'min {np.nanmin(output)}, mean max {np.nanmean(output):3.3f}, max {np.nanmax(output)}')
    return output


def subtract_background(input: npt.NDArray,
                         img_bg: Optional[npt.NDArray] = None,
                         const_bg: float = 0,
                         gpu: bool = False) -> npt.NDArray:
    """Background remove (constant BG and image BG); assume zero background
    if no ``img_bg`` option is provided."""
    output = input.copy()

    if img_bg is not None:
        output -= img_bg
    logging.info('After background image subtraction ' +
                 f'min {np.nanmin(output)}, mean max {np.nanmean(output):3.3f}, max {np.nanmax(output)}')

    output -= const_bg
    logging.info('After constant background factor subtraction ' +
                 f'min {np.nanmin(output)}, mean max {np.nanmean(output):3.3f}, max {np.nanmax(output)}')

    # set all pixels with negative values to zero
    output[output < 0] = 0
    logging.info('After removing pixels with negative value ' +
                 f'min {np.nanmin(output)}, mean max {np.nanmean(output):3.3f}, max {np.nanmax(output)}')

    return output


def get_timestamp(filepath: PathLike) -> datetime.datetime:
    """Parse the acquisition timestamp out of a Micro-Manager ``metadata.txt`` file."""
    metadata_contents = ''
    with open(filepath, 'r', encoding="ISO-8859-1") as metadata_file:
        metadata_contents = metadata_file.read()

    parsed_json = json.loads(metadata_contents)
    time_str = parsed_json['Summary']['Time']  # or 'StartTime'
    result = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S %z')
    return result
