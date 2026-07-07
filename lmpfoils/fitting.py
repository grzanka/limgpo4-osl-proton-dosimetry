"""Relative-efficiency curve fitting (LMP-foil response vs. proton kinetic
energy), ported from ``7-efficiency_fitting.ipynb`` / ``8-article-plots.ipynb``
(figure 8).

Requires the optional ``lmfit`` dependency (``pip install lmpfoils[fit]``).
"""
from typing import Optional

import numpy as np
import numpy.typing as npt

try:
    import lmfit
    from lmfit.models import LinearModel, StepModel
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "lmpfoils.fitting requires the optional 'lmfit' dependency; "
        "install with `pip install lmpfoils[fit]`.") from exc


def fit_efficiency_curve(ekin: npt.NDArray, efficiency: npt.NDArray,
                          efficiency_err: Optional[npt.NDArray] = None) -> "lmfit.model.ModelResult":
    """Fit the LMP-foil relative-efficiency vs. proton kinetic energy curve
    with a logistic step + linear background model (``StepModel(form='logistic')
    + LinearModel()``, intercept fixed at 0), matching the fit used for the
    article's Figure 8.

    The fitted curve has the closed form::

        eta(E) = A / (1 + exp((C - E) / S)) + B * E

    where ``A`` = ``amplitude``, ``B`` = ``slope``, ``C`` = ``center``,
    ``S`` = ``sigma`` (see ``result.best_values``).

    Parameters
    ----------
    ekin:
        Proton kinetic energy [MeV] for each data point (typically the
        median energy from the PHITS kinetic-energy spectrum, interpolated
        onto the foil's depth position).
    efficiency:
        Measured relative efficiency (dimensionless) at each ``ekin``.
    efficiency_err:
        Optional 1-sigma uncertainty on ``efficiency``, used as
        ``weights = 1 / efficiency_err`` in the least-squares fit (as in the
        article). If omitted, an unweighted fit is performed.

    Returns
    -------
    lmfit ``ModelResult`` -- use ``.eval(x=...)`` to evaluate the fitted
    curve, and ``.fit_report()`` for a text summary.
    """
    step_mod = StepModel(form='logistic')
    lin_mod = LinearModel()
    pars = step_mod.guess(efficiency, x=ekin)
    pars += lin_mod.guess(efficiency, x=ekin)
    pars['intercept'].value = 0
    pars['intercept'].vary = False
    model = step_mod + lin_mod
    weights = None if efficiency_err is None else 1.0 / np.asarray(efficiency_err)
    return model.fit(efficiency, pars, x=ekin, weights=weights)
