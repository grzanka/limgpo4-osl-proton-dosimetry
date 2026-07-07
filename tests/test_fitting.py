import numpy as np
import pytest

lmfit = pytest.importorskip("lmfit")

from lmpfoils.fitting import fit_efficiency_curve


def test_fit_efficiency_curve_recovers_step_shape():
    rng = np.random.default_rng(0)
    ekin = np.linspace(0, 60, 40)
    true_eff = 0.1 + 0.8 / (1 + np.exp(-(ekin - 20) / 3))
    noisy_eff = true_eff + rng.normal(scale=0.01, size=ekin.shape)

    result = fit_efficiency_curve(ekin, noisy_eff)
    fitted = result.eval(x=ekin)

    assert np.corrcoef(fitted, true_eff)[0, 1] > 0.95
    # plateau at high energy should be recovered within reason
    assert fitted[-1] == pytest.approx(true_eff[-1], abs=0.15)
    # intercept is fixed at 0, as in the article fit
    assert result.params['intercept'].value == 0
    assert not result.params['intercept'].vary


def test_fit_efficiency_curve_accepts_weights():
    ekin = np.linspace(0, 60, 40)
    eff = 0.1 + 0.8 / (1 + np.exp(-(ekin - 20) / 3))
    err = np.full_like(eff, 0.02)
    result = fit_efficiency_curve(ekin, eff, efficiency_err=err)
    assert result.success
