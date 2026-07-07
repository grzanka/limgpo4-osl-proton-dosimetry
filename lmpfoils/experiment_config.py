"""Fixed experiment-configuration constants for the 2022-11 measurement
campaign (BP/SOBP proton irradiations, Co-60 reference, background,
flat-field), ported from the exploratory notebooks
(``5-proton_sobp.ipynb`` / ``8-article-plots.ipynb``).

These are properties of the *physical setup* (foil holder geometry, beam
monitor readout), not of the analysis code, so they are kept as data here
rather than re-derived in every figure notebook.
"""
import numpy as np
import numpy.typing as npt

#: Raw-data subfolder names (relative to ``data/raw/``), one per irradiation.
DATASET_DIRS = {
    "background": "2022_11_23_background",
    "co60": "2022_11_23_Co60",
    "bp": "2022_11_17_bp",
    "sobp": "2022_11_18_sobp",
    "flat_field": "2022_08_22_flat_field/FF_2sLED_U340/FF_1",
}

#: Reference dose [Gy] delivered to the Co-60 calibration foil set, used as
#: the denominator for the ``dose_Gy = ref_dose_gy * signal / signal_ref``
#: conversion applied to the BP/SOBP foil signals.
CO60_REFERENCE_DOSE_GY = 60.0

#: Foil-holder positions, decreasing depth-in-water [mm], measured for the
#: 43-slot holder (foils 1..40 plus 3 spacer slots removed below) used in
#: both the BP and SOBP irradiations. ``positions[::-1]`` gives increasing
#: depth ordered by detector id 1..40 (foil 1 = deepest / last in beam path).
_RAW_POSITIONS_MM = """
0.28
0.85
1.43
2.00
2.53
4.18
6.97
9.73
11.37
11.93
12.47
13.00
13.55
14.09
14.62
15.17
15.71
16.25
16.81
17.35
17.90
18.46
19.01
19.55
20.12
20.67
21.24
21.81
22.38
22.96
23.51
24.06
24.59
25.16
25.72
26.28
26.84
27.38
27.92
28.46
29.00
29.56
30.11
""".split()


def foil_positions_mm() -> npt.NDArray:
    """40 foil-holder depth positions [mm], ordered by increasing detector
    id (1..40), matching ``df_signal['pos_mm'] = pos_mm[::-1]`` in the
    original notebooks.

    Three of the raw holder's 43 slots are spacers with no foil and are
    dropped (matching the notebooks' ``positions.pop(5)`` x3).
    """
    positions = list(_RAW_POSITIONS_MM)
    for _ in range(3):
        positions.pop(5)
    pos_mm = np.array(positions, dtype=float)
    return pos_mm[::-1]
