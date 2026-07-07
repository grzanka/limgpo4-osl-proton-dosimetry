"""Angular alignment of detector (foil) circles.

Ported from ``ifj_lmp_foils/src/data/alignment.py``.
"""
import logging
from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from lmpfoils.geometry import Circle


def get_line_circle(img: npt.NDArray, circ: Circle, step_angle_deg: float = 1.0, max_angle_deg: float = 360.0):
    """Get a pixel-value profile along the circle defined by ``circ``."""
    img_line = img.copy()
    img_line[np.isnan(img_line)] = 0
    logging.info(f'Input image shape {img_line.shape} min {img_line.min():3.3f}, max {img_line.max():3.3f}')

    array_of_angles_deg = np.arange(start=0, stop=max_angle_deg, step=step_angle_deg)
    logging.info(f'Array of angles, step {step_angle_deg}, max angle {max_angle_deg}')
    points_on_circle_px = np.array([(np.sin(np.deg2rad(-angle_deg)) * circ.r + circ.x,
                                      -np.cos(np.deg2rad(-angle_deg)) * circ.r + circ.y)
                                     for angle_deg in array_of_angles_deg])
    logging.info(f'Points on circle # {points_on_circle_px.size}, shape {points_on_circle_px.shape}')
    logging.info(f'Points on circle X from {points_on_circle_px[:,0].min()} to {points_on_circle_px[:,0].max()}')
    logging.info(f'Points on circle Y from {points_on_circle_px[:,1].min()} to {points_on_circle_px[:,1].max()}')

    values_on_circle = ndimage.map_coordinates(img_line.T, points_on_circle_px.T)
    logging.info(f'Values on circle min {values_on_circle.min():3.3f}, max {values_on_circle.max():3.3f}')
    return array_of_angles_deg, values_on_circle


def get_angle_with_min_value(array_of_angles_deg: npt.NDArray,
                              values_on_circle: npt.NDArray,
                              median_filter_size: int = 10) -> Tuple[float, npt.NDArray]:
    min_value = ndimage.median_filter(input=values_on_circle, size=median_filter_size)
    min_value_angle_deg = array_of_angles_deg[np.nanargmin(min_value)]
    return min_value_angle_deg, min_value
