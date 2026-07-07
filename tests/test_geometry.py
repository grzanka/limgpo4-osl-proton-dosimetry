import json

import numpy as np
import pytest

from lmpfoils.geometry import Circle, create_circular_mask, default_circular_mask, get_mean_std


def test_circle_proper():
    assert Circle(x=0, y=0, r=5).proper
    assert not Circle().proper  # nan defaults


def test_circle_json_roundtrip(tmp_path):
    c = Circle(x=1.5, y=2.5, r=10.0)
    p = tmp_path / "circle.json"
    c.save_json(p)
    loaded = Circle.from_json(p)
    assert loaded == c
    with open(p) as f:
        data = json.load(f)
    assert data == {"x": 1.5, "y": 2.5, "r": 10.0}


def test_circle_section_x_y():
    c = Circle(x=0, y=0, r=5)
    lower, upper = c.section_x(x=0)
    assert lower == pytest.approx(-5)
    assert upper == pytest.approx(5)

    lower, upper = c.section_y(y=0)
    assert lower == pytest.approx(-5)
    assert upper == pytest.approx(5)

    # outside the circle -> nan
    lower, upper = c.section_x(x=100)
    assert np.isnan(lower) and np.isnan(upper)


def test_create_circular_mask_shape_and_content():
    img = np.zeros((11, 11))
    circle = Circle(x=5, y=5, r=2)
    mask = create_circular_mask(img, circle)
    assert mask.shape == img.shape
    assert mask[5, 5]  # center is inside
    assert not mask[0, 0]  # corner is outside


def test_default_circular_mask_centered_square():
    img = np.zeros((10, 10))
    mask = default_circular_mask(img)
    assert mask[5, 5]
    assert not mask[0, 0]


def test_get_mean_std_uniform_image():
    img = np.full((21, 21), 3.0)
    circle = Circle(x=10, y=10, r=5)
    mean, std = get_mean_std(img, circle)
    assert mean == pytest.approx(3.0)
    assert std == pytest.approx(0.0)


def test_get_mean_std_with_nan_outside_roi():
    img = np.zeros((21, 21))
    img[10, 10] = 10.0
    circle = Circle(x=10, y=10, r=1)
    mean, _ = get_mean_std(img, circle)
    # small circle around the hot pixel should push the mean up noticeably
    assert mean > 0.5
