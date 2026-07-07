import numpy as np

from lmpfoils.experiment_config import (CO60_REFERENCE_DOSE_GY, DATASET_DIRS,
                                          foil_positions_mm)


def test_foil_positions_count_and_order():
    pos = foil_positions_mm()
    assert len(pos) == 40
    # increasing detector id => decreasing depth (foil 1 is deepest)
    assert np.all(np.diff(pos) <= 0)
    assert pos[0] == 30.11
    assert pos[-1] == 0.28


def test_dataset_dirs_and_reference_dose():
    assert set(DATASET_DIRS) == {"background", "co60", "bp", "sobp", "flat_field"}
    assert CO60_REFERENCE_DOSE_GY == 60.0
