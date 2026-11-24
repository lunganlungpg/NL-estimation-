"""Accuracy against NIST certified values, and solver behaviour."""

import numpy as np
import pytest
from scipy.optimize import least_squares

import nlsfit
from nlsfit.datasets import load, log_relative_error
from nlsfit.solvers import numerical_jacobian, relative_offset

NIST = ["misra1a", "boxbod", "danwood", "rat42", "rat43", "mgh09"]


@pytest.mark.parametrize("name", NIST)
def test_lm_matches_nist_from_start2(name):
    d = load(name)
    res = nlsfit.fit(d.model, d.x, d.y, p0=d.starts[1], method="lm")
    assert res.converged
    assert log_relative_error(res.params, d.certified).min() >= 6  # ≥ 6 correct digits
    assert res.sse == pytest.approx(d.certified_sse, rel=1e-7)


@pytest.mark.parametrize("name", ["misra1a", "danwood", "rat42", "rat43", "mgh09"])
def test_lm_matches_nist_from_hard_start1(name):
    d = load(name)
    res = nlsfit.fit(d.model, d.x, d.y, p0=d.starts[0], method="lm")
    assert res.converged, res.solver.message
    assert log_relative_error(res.params, d.certified).min() >= 6


@pytest.mark.parametrize("name", ["misra1a", "boxbod", "danwood", "rat42", "rat43"])
def test_gauss_newton_with_step_halving_matches_nist(name):
    d = load(name)
    res = nlsfit.fit(d.model, d.x, d.y, p0=d.starts[1], method="gn")
    assert res.converged
    assert log_relative_error(res.params, d.certified).min() >= 6


def test_step_halving_rescues_gauss_newton_on_boxbod():
    d = load("boxbod")
    pure = nlsfit.fit(d.model, d.x, d.y, p0=d.starts[0], method="gn-pure")
    safe = nlsfit.fit(d.model, d.x, d.y, p0=d.starts[0], method="gn")
    assert not pure.converged
    assert safe.converged and log_relative_error(safe.params, d.certified).min() >= 6


def test_failure_is_reported_not_hidden():
    # LM (like MINPACK) stalls on a plateau from BoxBOD start 1; it must say so.
    d = load("boxbod")
    res = nlsfit.fit(d.model, d.x, d.y, p0=d.starts[0], method="lm")
    scipy_res = least_squares(lambda p: d.model(d.x, *p) - d.y, d.starts[0], method="lm")
    assert not res.converged
    assert "stalled" in res.solver.message or "λ" in res.solver.message
    assert 2 * scipy_res.cost > 5 * d.certified_sse  # MINPACK also fails here


def test_matches_scipy_least_squares_on_simulated_data():
    for name in ["crop_yield_nitrogen", "plant_growth", "dose_response", "drug_elimination"]:
        d = load(name)
        ours = nlsfit.fit(d.model, d.x, d.y)
        ref = least_squares(
            lambda p, d=d: d.model(d.x, *p) - d.y, ours.p0, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15
        )
        np.testing.assert_allclose(ours.params, ref.x, rtol=1e-6)


@pytest.mark.parametrize("name", list(nlsfit.MODELS))
def test_analytic_jacobians_match_finite_differences(name):
    m = nlsfit.MODELS[name]
    rng = np.random.default_rng(1)
    x = np.linspace(0.3, 3.0, 25)
    theta = {
        "gompertz": [10, 2, 0.6],
        "four_pl": [1, 9, 1.2, 1.5],
        "richards": [10, 2, 1, 1.3],
        "kowalik_osborne": [0.2, 0.2, 0.12, 0.14],
        "logistic": [10, 1.5, 0.4],
    }.get(name, list(rng.uniform(0.5, 2, m.n_params)))
    J = m.jac(x, *theta)
    Jn = numerical_jacobian(lambda t: m(x, *t), np.array(theta, float))
    np.testing.assert_allclose(J, Jn, rtol=1e-5, atol=1e-7)


def test_numerical_jacobian_forward_vs_central():
    f = lambda t: np.array([np.sin(t[0]) * t[1], t[1] ** 3])  # noqa: E731
    t = np.array([0.7, 1.3])
    exact = np.array([[np.cos(0.7) * 1.3, np.sin(0.7)], [0, 3 * 1.3**2]])
    assert np.abs(numerical_jacobian(f, t, "central") - exact).max() < 1e-8
    assert np.abs(numerical_jacobian(f, t, "forward") - exact).max() < 1e-6


def test_relative_offset_zero_at_linear_least_squares_solution():
    rng = np.random.default_rng(0)
    J = rng.normal(size=(30, 3))
    y = rng.normal(size=30)
    beta, *_ = np.linalg.lstsq(J, y, rcond=None)
    assert relative_offset(J, J @ beta - y) < 1e-12
    assert relative_offset(J, J @ (beta + 0.1) - y) > 1e-2


def test_lm_options_and_history():
    d = load("puromycin")
    for scaling in ("marquardt", "levenberg"):
        for update in ("nielsen", "classic"):
            r = nlsfit.fit(d.model, d.x, d.y, method="lm", scaling=scaling, update=update)
            assert r.converged and r["Vmax"] == pytest.approx(212.6837, rel=1e-5)
    h = r.history
    assert h[0].iteration == 0 and h[-1].sse <= h[0].sse
    assert all(it.lam is not None for it in h)


def test_already_optimal_start():
    d = load("danwood")
    r = nlsfit.fit(d.model, d.x, d.y, p0=d.certified, method="gn")
    assert r.converged and r.solver.n_iter <= 1


def test_bad_arguments():
    d = load("puromycin")
    with pytest.raises(ValueError, match="method"):
        nlsfit.fit(d.model, d.x, d.y, method="newton")
    with pytest.raises(ValueError, match="p0 has"):
        nlsfit.fit(d.model, d.x, d.y, p0=[1, 2, 3])
    with pytest.raises(ValueError, match="different lengths"):
        nlsfit.fit(d.model, d.x, d.y[:-1])
    with pytest.raises(ValueError, match="NaN"):
        nlsfit.fit(d.model, d.x, np.r_[d.y[:-1], np.nan])
    with pytest.raises(ValueError, match="more observations"):
        nlsfit.fit(d.model, d.x[:2], d.y[:2])
    with pytest.raises(ValueError, match="no automatic starting"):
        nlsfit.fit(nlsfit.models.kowalik_osborne, d.x, d.y)
    with pytest.raises(KeyError, match="Available"):
        nlsfit.fit("not_a_model", d.x, d.y)
