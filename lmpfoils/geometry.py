"""Circle geometry and circular-mask utilities.

Ported from ``ifj_lmp_foils/src/data/analysis.py`` (``Circle``,
``create_circular_mask``, ``default_circular_mask``, ``get_mean_std``).
"""
import dataclasses
import json
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Circle:
    """Read-only data class for storing center position and radius of a circle."""
    x: float = float('nan')
    y: float = float('nan')
    r: float = float('nan')

    @property
    def proper(self):
        return self.x > -np.inf and self.y > -np.inf and self.r >= 0.

    def save_json(self, path):
        with open(path, 'w') as json_file:
            json.dump(dataclasses.asdict(self), json_file)

    @classmethod
    def from_json(cls, path: str):
        circle = cls()
        with open(path, 'r') as openfile:
            json_data = json.load(openfile)
            circle = cls(**json_data)
        return circle

    def section_x(self, x: float) -> Tuple[float, float]:
        under_sqrt = self.r**2 - (x - self.x)**2
        result = (float('nan'), float('nan'))
        if under_sqrt >= 0:
            result = (self.y - math.sqrt(under_sqrt), self.y + math.sqrt(under_sqrt))
        return result

    def section_y(self, y: float) -> Tuple[float, float]:
        under_sqrt = self.r**2 - (y - self.y)**2
        result = (float('nan'), float('nan'))
        if under_sqrt >= 0:
            result = (self.x - math.sqrt(under_sqrt), self.x + math.sqrt(under_sqrt))
        return result


def create_circular_mask(img: npt.NDArray, circle_px: Circle) -> npt.NDArray:
    """Create a circular mask of the same resolution as the image."""
    y_grid, x_grid = np.ogrid[:img.shape[0], :img.shape[1]]
    dist_from_center_squared = (x_grid - circle_px.x)**2 + (y_grid - circle_px.y)**2
    circ_mask = dist_from_center_squared <= circle_px.r**2
    return circ_mask


def default_circular_mask(img: npt.NDArray) -> npt.NDArray:
    """Create a circular mask centered on the image with the maximum radius
    that stays fully enclosed in the image."""
    center_x = img.shape[0] / 2
    center_y = img.shape[1] / 2
    radius = min(center_x, center_y)
    return create_circular_mask(img, Circle(x=center_x, y=center_y, r=radius))


def get_mean_std(data: npt.NDArray, circle: Circle) -> Tuple[float, float]:
    """Mean and standard deviation of pixel values within a circular ROI."""
    mask = create_circular_mask(img=data, circle_px=circle)
    mean = np.nanmean(data[mask]).astype(float)
    std = np.nanstd(data[mask]).astype(float)
    return mean, std
