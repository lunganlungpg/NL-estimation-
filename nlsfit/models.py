"""A library of common nonlinear models.

Each :class:`Model` bundles the function, its analytic Jacobian (exact
derivatives make Gauss–Newton and Levenberg–Marquardt faster and more
reliable than finite differences), parameter names, a formula for display and
a data-driven guess of starting values.

You can also pass any plain Python function ``f(x, a, b, ...)`` to
:func:`nlsfit.fit`; its Jacobian is then computed numerically.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Model:
    name: str
    formula: str
    param_names: tuple[str, ...]
    func: Callable[..., np.ndarray]
    jac: Callable[..., np.ndarray] | None = None
    guess: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
    description: str = ""

    def __call__(self, x, *params):
        return self.func(np.asarray(x, dtype=float), *params)

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    def __repr__(self) -> str:
        return f"Model({self.name!r}: {self.formula})"


def _stack(*cols):
    return np.column_stack([np.broadcast_to(c, np.shape(cols[0])) for c in cols])


def _slope_through_origin_half(x, y):
    """x where y first reaches half of its range (a robust 'midpoint' guess)."""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    half = ys.min() + 0.5 * (ys.max() - ys.min())
    idx = int(np.argmax(ys >= half)) if ys[-1] >= ys[0] else int(np.argmax(ys <= half))
    return xs[idx]


# ---------------------------------------------------------------- exponential


def _exp_decay(x, a, k):
    return a * np.exp(-k * x)


def _exp_decay_jac(x, a, k):
    e = np.exp(-k * x)
    return _stack(e, -a * x * e)


def _exp_decay_guess(x, y):
    pos = y > 0
    if pos.sum() >= 2:
        slope, intercept = np.polyfit(x[pos], np.log(y[pos]), 1)
        return np.array([np.exp(intercept), max(-slope, 1e-6)])
    return np.array([y.max(), 1.0 / (np.ptp(x) or 1.0)])


exponential_decay = Model(
    "exponential_decay",
    "y = a·exp(−k·x)",
    ("a", "k"),
    _exp_decay,
    _exp_decay_jac,
    _exp_decay_guess,
    "First-order decay: drug elimination, radioactive decay, cooling.",
)


def _exp_rise(x, a, k):
    return a * (1 - np.exp(-k * x))


def _exp_rise_jac(x, a, k):
    e = np.exp(-k * x)
    return _stack(1 - e, a * x * e)


def _exp_rise_guess(x, y):
    a = 1.05 * np.max(y)
    frac = np.clip(y / a, 1e-6, 1 - 1e-6)
    k = np.median(-np.log(1 - frac[x > 0]) / x[x > 0]) if np.any(x > 0) else 1.0
    return np.array([a, k])


exponential_rise = Model(
    "exponential_rise",
    "y = a·(1 − exp(−k·x))",
    ("a", "k"),
    _exp_rise,
    _exp_rise_jac,
    _exp_rise_guess,
    "Monomolecular / first-order approach to an asymptote (NIST BoxBOD, Misra1a).",
)


def _asym(x, a, b, c):
    return a - b * np.exp(-c * x)


def _asym_jac(x, a, b, c):
    e = np.exp(-c * x)
    return _stack(np.ones_like(x), -e, b * x * e)


def _asym_guess(x, y):
    a = np.max(y) + 0.05 * np.ptp(y)
    b = a - y[np.argmin(x)]
    frac = np.clip((a - y) / max(b, 1e-12), 1e-6, None)
    c = np.median(-np.log(frac[x > 0]) / x[x > 0]) if np.any(x > 0) else 1.0
    return np.array([a, b, max(c, 1e-6)])


asymptotic_regression = Model(
    "asymptotic_regression",
    "y = a − b·exp(−c·x)",
    ("a", "b", "c"),
    _asym,
    _asym_jac,
    _asym_guess,
    "Mitscherlich law of diminishing returns: crop yield response to fertiliser.",
)

# ---------------------------------------------------------------- kinetics


def _mm(x, vmax, km):
    return vmax * x / (km + x)


def _mm_jac(x, vmax, km):
    d = km + x
    return _stack(x / d, -vmax * x / d**2)


def _mm_guess(x, y):
    # Lineweaver–Burk regression gives a decent (if biased) start
    pos = (x > 0) & (y > 0)
    if pos.sum() >= 2:
        slope, intercept = np.polyfit(1 / x[pos], 1 / y[pos], 1)
        if intercept > 0 and slope > 0:
            return np.array([1 / intercept, slope / intercept])
    return np.array([np.max(y), _slope_through_origin_half(x, y)])


michaelis_menten = Model(
    "michaelis_menten",
    "y = Vmax·x / (Km + x)",
    ("Vmax", "Km"),
    _mm,
    _mm_jac,
    _mm_guess,
    "Enzyme kinetics; also Monod growth and Langmuir adsorption.",
)

# ---------------------------------------------------------------- growth curves


def _logistic(x, asym, xmid, scal):
    return asym / (1 + np.exp((xmid - x) / scal))


def _logistic_jac(x, asym, xmid, scal):
    e = np.exp((xmid - x) / scal)
    d = (1 + e) ** 2
    return _stack(1 / (1 + e), -asym * e / (scal * d), asym * e * (xmid - x) / (scal**2 * d))


def _logistic_guess(x, y):
    asym = 1.05 * np.max(y)
    xmid = _slope_through_origin_half(x, y)
    scal = np.ptp(x) / 8 or 1.0
    return np.array([asym, xmid, scal])


logistic = Model(
    "logistic",
    "y = Asym / (1 + exp((xmid − x)/scal))",
    ("Asym", "xmid", "scal"),
    _logistic,
    _logistic_jac,
    _logistic_guess,
    "Symmetric S-shaped growth (R's SSlogis parameterisation).",
)


def _gompertz(x, asym, b2, b3):
    return asym * np.exp(-b2 * b3**x)


def _gompertz_jac(x, asym, b2, b3):
    p = b3**x
    e = np.exp(-b2 * p)
    return _stack(e, -asym * p * e, -asym * b2 * x * b3 ** (x - 1) * e)


def _gompertz_guess(x, y):
    asym = 1.05 * np.max(y)
    frac = np.clip(y / asym, 1e-6, 1 - 1e-6)
    z = np.log(-np.log(frac))  # = log(b2) + x·log(b3)
    slope, intercept = np.polyfit(x, z, 1)
    return np.array([asym, np.exp(intercept), np.exp(slope)])


gompertz = Model(
    "gompertz",
    "y = Asym·exp(−b2·b3^x)",
    ("Asym", "b2", "b3"),
    _gompertz,
    _gompertz_jac,
    _gompertz_guess,
    "Asymmetric S-shaped growth (R's SSgompertz parameterisation, 0 < b3 < 1).",
)


def _fpl(x, bottom, top, ec50, hill):
    return bottom + (top - bottom) / (1 + (x / ec50) ** hill)


def _fpl_jac(x, bottom, top, ec50, hill):
    xr = np.where(x > 0, x / ec50, 1e-300)
    u = xr**hill
    d = 1 + u
    return _stack(
        1 - 1 / d,
        1 / d,
        (top - bottom) * hill * u / (ec50 * d**2),
        -(top - bottom) * u * np.log(xr) / d**2,
    )


def _fpl_guess(x, y):
    order = np.argsort(x)
    ys = y[order]
    top, bottom = ys[0], ys[-1]
    pos = x > 0
    mid = 0.5 * (top + bottom)
    ec50 = float(np.exp(np.median(np.log(x[pos][np.argsort(np.abs(y[pos] - mid))[:3]]))))
    return np.array([bottom, top, ec50, 1.0])


four_pl = Model(
    "four_pl",
    "y = bottom + (top − bottom)/(1 + (x/EC50)^hill)",
    ("bottom", "top", "EC50", "hill"),
    _fpl,
    _fpl_jac,
    _fpl_guess,
    "Four-parameter logistic dose–response curve (x must be > 0).",
)

# ---------------------------------------------------------------- other shapes


def _power(x, a, b):
    return a * x**b


def _power_jac(x, a, b):
    p = x**b
    return _stack(p, a * p * np.log(x))


def _power_guess(x, y):
    pos = (x > 0) & (y > 0)
    b, log_a = np.polyfit(np.log(x[pos]), np.log(y[pos]), 1)
    return np.array([np.exp(log_a), b])


power_law = Model(
    "power_law",
    "y = a·x^b",
    ("a", "b"),
    _power,
    _power_jac,
    _power_guess,
    "Allometric scaling and other power laws (x must be > 0; NIST DanWood).",
)


def _gauss(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _gauss_jac(x, amp, mu, sigma):
    z = (x - mu) / sigma
    e = np.exp(-0.5 * z**2)
    return _stack(e, amp * e * z / sigma, amp * e * z**2 / sigma)


def _gauss_guess(x, y):
    i = int(np.argmax(y))
    w = np.clip(y, 0, None)
    sd = np.sqrt(np.sum(w * (x - x[i]) ** 2) / max(np.sum(w), 1e-12)) or (np.ptp(x) / 6)
    return np.array([y[i], x[i], sd])


gaussian_peak = Model(
    "gaussian_peak",
    "y = amp·exp(−½((x − μ)/σ)²)",
    ("amp", "mu", "sigma"),
    _gauss,
    _gauss_jac,
    _gauss_guess,
    "A single peak: spectroscopy, chromatography.",
)


def _rational(x, b1, b2, b3, b4):
    return b1 * (x**2 + x * b2) / (x**2 + x * b3 + b4)


def _rational_jac(x, b1, b2, b3, b4):
    num = x**2 + x * b2
    den = x**2 + x * b3 + b4
    return _stack(num / den, b1 * x / den, -b1 * num * x / den**2, -b1 * num / den**2)


kowalik_osborne = Model(
    "kowalik_osborne",
    "y = b1·(x² + b2·x)/(x² + b3·x + b4)",
    ("b1", "b2", "b3", "b4"),
    _rational,
    _rational_jac,
    None,
    "Rational model from enzyme kinetics (NIST MGH09, a hard test problem).",
)


def _rat43(x, b1, b2, b3, b4):
    return b1 / (1 + np.exp(b2 - b3 * x)) ** (1 / b4)


def _rat43_jac(x, b1, b2, b3, b4):
    e = np.exp(b2 - b3 * x)
    base = 1 + e
    f = base ** (-1 / b4)
    dfdbase = -(1 / b4) * base ** (-1 / b4 - 1)
    return _stack(f, b1 * dfdbase * e, -b1 * dfdbase * e * x, b1 * f * np.log(base) / b4**2)


richards = Model(
    "richards",
    "y = b1 / (1 + exp(b2 − b3·x))^(1/b4)",
    ("b1", "b2", "b3", "b4"),
    _rat43,
    _rat43_jac,
    None,
    "Richards (generalised logistic) growth curve (NIST Rat43).",
)


def _rat42(x, b1, b2, b3):
    return b1 / (1 + np.exp(b2 - b3 * x))


def _rat42_jac(x, b1, b2, b3):
    e = np.exp(b2 - b3 * x)
    d = (1 + e) ** 2
    return _stack(1 / (1 + e), -b1 * e / d, b1 * e * x / d)


def _rat42_guess(x, y):
    b1 = 1.05 * np.max(y)
    frac = np.clip(y / b1, 1e-6, 1 - 1e-6)
    slope, intercept = np.polyfit(x, np.log(1 / frac - 1), 1)  # = b2 − b3·x
    return np.array([b1, intercept, -slope])


logistic3 = Model(
    "logistic3",
    "y = b1 / (1 + exp(b2 − b3·x))",
    ("b1", "b2", "b3"),
    _rat42,
    _rat42_jac,
    _rat42_guess,
    "Three-parameter logistic in rate form (NIST Rat42).",
)

MODELS: dict[str, Model] = {
    m.name: m
    for m in [
        exponential_decay,
        exponential_rise,
        asymptotic_regression,
        michaelis_menten,
        logistic,
        logistic3,
        gompertz,
        richards,
        four_pl,
        power_law,
        gaussian_peak,
        kowalik_osborne,
    ]
}


def get_model(name: str) -> Model:
    """Look up a built-in model by name (see ``nlsfit.MODELS``)."""
    try:
        return MODELS[name]
    except KeyError:
        raise KeyError(f"Unknown model {name!r}. Available: {', '.join(MODELS)}") from None
