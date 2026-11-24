"""Standard errors, intervals, bands and model comparison."""

import numpy as np
import pytest
from scipy.optimize import curve_fit

import nlsfit
from nlsfit.datasets import load


@pytest.fixture(scope="module")
def puro():
    d = load("puromycin")
    return nlsfit.fit(d.model, d.x, d.y)


def test_puromycin_matches_published_values(puro):
    # Bates & Watts (1988): Vmax = 212.7 (SE 6.947), Km = 0.06412 (SE 0.008281), s = 10.93 on 10 df
    assert puro["Vmax"] == pytest.approx(212.68, abs=0.01)
    assert puro["Km"] == pytest.approx(0.06412, abs=1e-5)
    np.testing.assert_allclose(puro.se, [6.947, 0.008281], rtol=1e-3)
    assert puro.sigma == pytest.approx(10.93, abs=0.01)
    assert puro.df == 10


def test_covariance_matches_scipy_curve_fit(puro):
    d = load("puromycin")
    popt, pcov = curve_fit(
        lambda x, v, k: v * x / (k + x), d.x, d.y, p0=[200, 0.1], xtol=1e-15, ftol=1e-15, gtol=1e-15
    )
    np.testing.assert_allclose(puro.params, popt, rtol=1e-6)
    np.testing.assert_allclose(puro.cov, pcov, rtol=1e-4)


def test_wald_profile_bootstrap_intervals(puro):
    wald = puro.conf_int(method="wald")
    prof = puro.conf_int(method="profile")
    boot = puro.conf_int(method="bootstrap", n_boot=300)
    for ci in (wald, prof, boot):
        assert np.all(ci[:, 0] < puro.params) and np.all(puro.params < ci[:, 1])
    # Profile interval for Km is asymmetric (longer on the right), as Bates & Watts show
    km_lo, km_hi = prof[1]
    assert (km_hi - puro["Km"]) > (puro["Km"] - km_lo)
    # Profile limits are where the profile t statistic equals the t quantile
    from scipy import stats

    q = stats.t.ppf(0.975, puro.df)
    for bound in prof[1]:
        s = puro.profile(1, np.array([bound]))[0]
        assert np.sqrt(s - puro.sse) / puro.sigma == pytest.approx(q, rel=1e-4)


def test_bootstrap_is_reproducible(puro):
    a = puro.bootstrap(n_boot=50, random_state=3)
    b = puro.bootstrap(n_boot=50, random_state=3)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (50, 2)


def test_bands(puro):
    xg = np.linspace(0.02, 1.1, 20)
    yhat, lo, hi = puro.predict(xg, interval="confidence")
    _, plo, phi = puro.predict(xg, interval="prediction")
    assert np.all(lo < yhat) and np.all(yhat < hi)
    assert np.all(plo < lo) and np.all(hi < phi)  # prediction band is wider
    with pytest.raises(ValueError):
        puro.predict(xg, interval="tolerance")


def test_summary_and_frame(puro):
    text = puro.summary()
    for token in ("Vmax", "Km", "Residual standard error", "10 degrees of freedom", "AIC"):
        assert token in text
    df = puro.to_frame()
    assert list(df.index) == ["Vmax", "Km"]
    assert "p_value" in df.columns
    assert "converged=True" in repr(puro)


def test_simulated_data_recovers_true_parameters():
    for name in ["crop_yield_nitrogen", "plant_growth", "dose_response", "drug_elimination"]:
        d = load(name)
        r = nlsfit.fit(d.model, d.x, d.y)
        ci = r.conf_int(level=0.999)
        assert r.converged
        assert np.all((ci[:, 0] < d.true_params) & (np.array(d.true_params) < ci[:, 1])), name


def test_model_comparison_prefers_true_model():
    d = load("plant_growth")  # generated from a Gompertz curve
    gomp = nlsfit.fit(nlsfit.models.gompertz, d.x, d.y)
    logi = nlsfit.fit(nlsfit.models.logistic, d.x, d.y)
    assert gomp.aic < logi.aic
    table = nlsfit.compare(gomp, logi)
    assert table.splitlines()[1].startswith("gompertz")


def test_extra_ss_test_nested():
    d = load("crop_yield_nitrogen")
    full = nlsfit.fit(nlsfit.models.asymptotic_regression, d.x, d.y)
    reduced = nlsfit.fit(lambda x, a, c: a * (1 - np.exp(-c * x)), d.x, d.y, p0=[10, 0.01])
    F, p = nlsfit.extra_ss_test(reduced, full)
    assert F > 0 and p < 0.05  # yield at zero N is clearly above zero
    with pytest.raises(ValueError):
        nlsfit.extra_ss_test(full, reduced)


def test_weighted_fit():
    d = load("drug_elimination")  # multiplicative noise → weights 1/ŷ² are appropriate
    unweighted = nlsfit.fit(d.model, d.x, d.y)
    weighted = nlsfit.fit(d.model, d.x, d.y, weights=1 / unweighted.fitted**2)
    assert weighted.converged
    assert abs(weighted["k"] - 0.35) < 3 * weighted.se[1]


def test_user_function_and_numerical_jacobian():
    d = load("puromycin")
    r = nlsfit.fit(lambda x, vmax, km: vmax * x / (km + x), d.x, d.y, p0=[200, 0.1])
    assert r.param_names == ("vmax", "km")
    assert r["vmax"] == pytest.approx(212.6837, rel=1e-5)
    r2 = nlsfit.fit(d.model, d.x, d.y, jacobian="forward")
    np.testing.assert_allclose(r2.params, r.params, rtol=1e-6)
    r3 = nlsfit.fit("michaelis_menten", d.x, d.y)
    np.testing.assert_allclose(r3.params, r.params, rtol=1e-6)


def test_multivariate_x():
    rng = np.random.default_rng(0)
    X = rng.uniform(0.5, 2, size=(40, 2))
    y = 3 * X[:, 0] ** 0.5 * np.exp(-0.4 * X[:, 1]) + rng.normal(0, 0.01, 40)
    r = nlsfit.fit(lambda X, a, b, c: a * X[:, 0] ** b * np.exp(-c * X[:, 1]), X, y, p0=[1, 1, 1])
    np.testing.assert_allclose(r.params, [3, 0.5, 0.4], atol=0.05)


def test_datasets_api():
    names = nlsfit.datasets.list_datasets()
    assert {"boxbod", "puromycin", "plant_growth"} <= set(names)
    with pytest.raises(KeyError):
        nlsfit.datasets.load("nope")
    assert nlsfit.datasets.load("BoxBOD").name == "boxbod"
