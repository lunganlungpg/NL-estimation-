"""High-level fitting API and statistical inference."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

from .models import Model
from .solvers import SolverResult, gauss_newton, levenberg_marquardt, numerical_jacobian

ModelLike = Model | Callable[..., np.ndarray]


def _as_model(model: ModelLike, p0) -> Model:
    if isinstance(model, Model):
        return model
    if not callable(model):
        raise TypeError("model must be an nlsfit.Model or a function f(x, *params)")
    params = list(inspect.signature(model).parameters)[1:]
    if not params or any(
        inspect.signature(model).parameters[p].kind == inspect.Parameter.VAR_POSITIONAL for p in params
    ):
        if p0 is None:
            raise ValueError("p0 is required when the model signature does not name its parameters")
        params = [f"b{i + 1}" for i in range(len(p0))]
    return Model(getattr(model, "__name__", "model"), "user-defined", tuple(params), model)


def fit(
    model: ModelLike,
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    p0: Sequence[float] | None = None,
    *,
    method: str = "lm",
    weights: Sequence[float] | np.ndarray | None = None,
    jacobian: str = "auto",
    **solver_options,
) -> FitResult:
    """Fit a nonlinear model by least squares.

    Parameters
    ----------
    model
        A built-in :class:`~nlsfit.Model` (e.g. ``nlsfit.models.michaelis_menten``),
        a model name such as ``"logistic"``, or any function ``f(x, a, b, ...)``.
    x, y
        Data. ``x`` may be 1-D or an (n, k) array for models with several inputs.
    p0
        Starting values. Optional for built-in models that provide a guess.
    method
        ``"lm"`` (Levenberg–Marquardt, the robust default), ``"gn"``
        (Gauss–Newton with step halving) or ``"gn-pure"`` (textbook
        Gauss–Newton with no safeguards, for teaching).
    weights
        Optional weights ``wᵢ``; minimises ``Σ wᵢ rᵢ²``. Use ``1/variance``.
    jacobian
        ``"auto"`` uses the model's analytic Jacobian when available,
        otherwise ``"central"`` or ``"forward"`` finite differences.
    **solver_options
        Passed to the solver, e.g. ``max_iter``, ``xtol``, ``lam0``.

    Returns
    -------
    FitResult
        Estimates with standard errors, confidence intervals, diagnostics and
        the full iteration history.
    """
    if isinstance(model, str):
        from .models import get_model

        model = get_model(model)
    mdl = _as_model(model, p0)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x and y have different lengths ({x.shape[0]} and {y.shape[0]})")
    mask = np.isfinite(y) & (np.all(np.isfinite(x), axis=1) if x.ndim == 2 else np.isfinite(x))
    if not np.all(mask):
        raise ValueError("x and y must not contain NaN or infinite values")

    if y.size <= mdl.n_params:
        raise ValueError(f"Need more observations ({y.size}) than parameters ({mdl.n_params})")
    if p0 is None:
        if mdl.guess is None:
            raise ValueError(f"Model {mdl.name!r} has no automatic starting values; pass p0")
        p0 = mdl.guess(x, y)
    p0 = np.asarray(p0, dtype=float)
    if p0.size != mdl.n_params:
        raise ValueError(f"p0 has {p0.size} values but the model has {mdl.n_params} parameters")
    n = y.size

    sw = np.ones(n) if weights is None else np.sqrt(np.asarray(weights, dtype=float))
    if np.any(sw < 0) or sw.shape != (n,):
        raise ValueError("weights must be non-negative with one value per observation")

    def predict(theta):
        return np.asarray(mdl.func(x, *theta), dtype=float)

    def residual(theta):
        return sw * (predict(theta) - y)

    if jacobian == "auto" and mdl.jac is not None:

        def jac(theta):
            return sw[:, None] * np.asarray(mdl.jac(x, *theta), dtype=float)
    else:
        fd = "central" if jacobian == "auto" else jacobian

        def jac(theta):
            return sw[:, None] * numerical_jacobian(predict, theta, fd)

    method_key = method.lower().replace("_", "-")
    if method_key in ("lm", "levenberg-marquardt"):
        raw = levenberg_marquardt(residual, jac, p0, **solver_options)
    elif method_key in ("gn", "gauss-newton"):
        raw = gauss_newton(residual, jac, p0, step_halving=True, **solver_options)
    elif method_key == "gn-pure":
        raw = gauss_newton(residual, jac, p0, step_halving=False, **solver_options)
    else:
        raise ValueError("method must be 'lm', 'gn' or 'gn-pure'")

    return FitResult(mdl, x, y, sw, p0, raw)


@dataclass
class FitResult:
    """Estimates, inference and diagnostics for a nonlinear least-squares fit."""

    model: Model
    x: np.ndarray
    y: np.ndarray
    sqrt_weights: np.ndarray
    p0: np.ndarray
    solver: SolverResult

    # ------------------------------------------------------------ basics
    @property
    def params(self) -> np.ndarray:
        return self.solver.params

    @property
    def param_names(self) -> tuple[str, ...]:
        return self.model.param_names

    def __getitem__(self, name: str) -> float:
        return float(self.params[self.param_names.index(name)])

    @property
    def converged(self) -> bool:
        return self.solver.converged

    @property
    def history(self):
        return self.solver.history

    @property
    def n(self) -> int:
        return self.y.size

    @property
    def df(self) -> int:
        """Residual degrees of freedom, n − p."""
        return self.n - self.params.size

    @property
    def fitted(self) -> np.ndarray:
        return np.asarray(self.model.func(self.x, *self.params), dtype=float)

    @property
    def residuals(self) -> np.ndarray:
        """Raw residuals y − ŷ (unweighted)."""
        return self.y - self.fitted

    @property
    def sse(self) -> float:
        """(Weighted) residual sum of squares at the solution."""
        return float(self.solver.sse)

    @property
    def sigma(self) -> float:
        """Residual standard error s = √(SSE / (n − p))."""
        return float(np.sqrt(self.sse / self.df))

    # ------------------------------------------------------------ inference
    @property
    def cov(self) -> np.ndarray:
        """Asymptotic covariance s²(JᵀJ)⁻¹, computed stably from the SVD of J."""
        J = self.solver.jacobian
        _, sv, vt = np.linalg.svd(J, full_matrices=False)
        tol = sv.max() * max(J.shape) * np.finfo(float).eps
        if np.any(sv <= tol):
            return np.full((self.params.size,) * 2, np.nan)
        return (vt.T / sv**2) @ vt * self.sigma**2

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    @property
    def corr(self) -> np.ndarray:
        """Correlation matrix of the estimates. |ρ| near 1 signals poor identifiability."""
        se = self.se
        return self.cov / np.outer(se, se)

    @property
    def tvalues(self) -> np.ndarray:
        return self.params / self.se

    @property
    def pvalues(self) -> np.ndarray:
        return 2 * stats.t.sf(np.abs(self.tvalues), self.df)

    def conf_int(self, level: float = 0.95, method: str = "wald", **kwargs) -> np.ndarray:
        """Confidence intervals, shape (p, 2).

        ``"wald"``: estimate ± t·SE, fast but symmetric and can be poor when
        the model is strongly nonlinear in a parameter.
        ``"profile"``: profile-likelihood (F-test inversion), the method used
        by R's ``confint.nls``; respects curvature and asymmetry.
        ``"bootstrap"``: residual-bootstrap percentile intervals.
        """
        if method == "wald":
            q = stats.t.ppf(0.5 + level / 2, self.df)
            return np.column_stack([self.params - q * self.se, self.params + q * self.se])
        if method == "profile":
            return np.array([self._profile_ci(j, level) for j in range(self.params.size)])
        if method == "bootstrap":
            boot = self.bootstrap(**kwargs)
            a = (1 - level) / 2
            return np.quantile(boot, [a, 1 - a], axis=0).T
        raise ValueError("method must be 'wald', 'profile' or 'bootstrap'")

    def profile(self, j: int, values: np.ndarray) -> np.ndarray:
        """Minimum SSE with parameter ``j`` held fixed at each of ``values``."""
        free = [k for k in range(self.params.size) if k != j]
        out = np.empty(len(values))
        start = self.params[free].copy()
        for i, v in enumerate(values):
            res = self._fit_fixed(j, v, start)
            out[i] = res.sse
            if res.converged:
                start = res.params
        return out

    def _fit_fixed(self, j: int, value: float, start: np.ndarray) -> SolverResult:
        free = [k for k in range(self.params.size) if k != j]
        x, y, sw, mdl = self.x, self.y, self.sqrt_weights, self.model

        def full(theta_free):
            th = np.empty(self.params.size)
            th[free] = theta_free
            th[j] = value
            return th

        def residual(tf):
            return sw * (np.asarray(mdl.func(x, *full(tf)), dtype=float) - y)

        if mdl.jac is not None:

            def jac(tf):
                return sw[:, None] * np.asarray(mdl.jac(x, *full(tf)), dtype=float)[:, free]
        else:

            def jac(tf):
                return numerical_jacobian(residual, tf)

        return levenberg_marquardt(residual, jac, start, max_iter=200)

    def _profile_ci(self, j: int, level: float) -> tuple[float, float]:
        q = stats.t.ppf(0.5 + level / 2, self.df)
        theta_hat, se = self.params[j], self.se[j]
        free = [k for k in range(self.params.size) if k != j]
        start = self.params[free]

        def tau(v):
            s = self._fit_fixed(j, v, start).sse
            return np.sign(v - theta_hat) * np.sqrt(max(s - self.sse, 0.0)) / self.sigma

        bounds = []
        for direction in (-1, 1):
            step = se if np.isfinite(se) and se > 0 else abs(theta_hat) * 0.1 or 0.1
            inner, outer = theta_hat, theta_hat + direction * step
            for _ in range(60):
                try:
                    val = abs(tau(outer))
                except (FloatingPointError, ValueError):
                    val = np.nan
                if not np.isfinite(val):
                    outer = np.nan
                    break
                if val >= q:
                    break
                inner, outer = outer, theta_hat + (outer - theta_hat) * 1.6
            else:
                outer = np.nan
            if np.isnan(outer):
                bounds.append(np.nan)  # interval is open on this side
                continue
            lo, hi = sorted((inner, outer))
            bounds.append(
                optimize.brentq(lambda v: abs(tau(v)) - q, lo, hi, xtol=1e-10 * (abs(theta_hat) + 1))
            )
        return tuple(bounds)

    def bootstrap(self, n_boot: int = 1000, random_state: int | None = 0) -> np.ndarray:
        """Residual bootstrap: refit to ŷ + resampled residuals. Returns (n_boot, p)."""
        rng = np.random.default_rng(random_state)
        fitted, res = self.fitted, self.residuals * self.sqrt_weights
        # Inflate residuals for the degrees of freedom used by the fit
        res = (res - res.mean()) * np.sqrt(self.n / self.df)
        mdl = self.model
        x, sw = self.x, self.sqrt_weights
        jac_fn = None
        out = []
        for _ in range(n_boot):
            y_star = fitted + rng.choice(res, size=self.n, replace=True) / np.where(sw > 0, sw, 1)

            def residual(theta, y_star=y_star):
                return sw * (np.asarray(mdl.func(x, *theta), dtype=float) - y_star)

            if mdl.jac is not None:

                def jac_fn(theta):
                    return sw[:, None] * np.asarray(mdl.jac(x, *theta), dtype=float)
            else:

                def jac_fn(theta, residual=residual):
                    return numerical_jacobian(residual, theta)

            r = levenberg_marquardt(residual, jac_fn, self.params, max_iter=200)
            if r.converged:
                out.append(r.params)
        return np.array(out)

    # ------------------------------------------------------------ prediction
    def predict(
        self, x_new=None, interval: str | None = None, level: float = 0.95
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predictions, optionally with a delta-method band.

        ``interval="confidence"`` gives the band for the mean curve;
        ``interval="prediction"`` also includes observation noise.
        Returns ``yhat`` or ``(yhat, lower, upper)``.
        """
        x_new = self.x if x_new is None else np.asarray(x_new, dtype=float)
        yhat = np.asarray(self.model.func(x_new, *self.params), dtype=float)
        if interval is None:
            return yhat
        if self.model.jac is not None:
            G = np.asarray(self.model.jac(x_new, *self.params), dtype=float)
        else:
            G = numerical_jacobian(lambda th: self.model.func(x_new, *th), self.params)
        var = np.einsum("ij,jk,ik->i", G, self.cov, G)
        if interval == "prediction":
            var = var + self.sigma**2
        elif interval != "confidence":
            raise ValueError("interval must be None, 'confidence' or 'prediction'")
        q = stats.t.ppf(0.5 + level / 2, self.df)
        half = q * np.sqrt(var)
        return yhat, yhat - half, yhat + half

    # ------------------------------------------------------------ goodness of fit
    @property
    def r_squared(self) -> float:
        """1 − SSE/SST. Use with care: it is not a reliable guide for nonlinear models."""
        sst = float(
            np.sum(self.sqrt_weights**2 * (self.y - np.average(self.y, weights=self.sqrt_weights**2)) ** 2)
        )
        return 1 - self.sse / sst

    @property
    def loglik(self) -> float:
        """Gaussian log-likelihood at the ML estimate of σ²."""
        n = self.n
        logw = np.sum(np.log(self.sqrt_weights[self.sqrt_weights > 0] ** 2))
        return -0.5 * n * (np.log(2 * np.pi * self.sse / n) + 1) + 0.5 * logw

    @property
    def aic(self) -> float:
        return -2 * self.loglik + 2 * (self.params.size + 1)

    @property
    def bic(self) -> float:
        return -2 * self.loglik + np.log(self.n) * (self.params.size + 1)

    # ------------------------------------------------------------ reporting
    def summary(self, level: float = 0.95) -> str:
        """An R ``summary.nls``-style report."""
        ci = self.conf_int(level)

        def stars(p):
            return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else ""

        w = max(8, max(len(n) for n in self.param_names))
        lines = [
            f"Model:   {self.model.formula}"
            + ("" if self.model.formula != "user-defined" else f"  ({self.model.name})"),
            f"Method:  {self.solver.method}",
            f"Status:  {self.solver.message} after {self.solver.n_iter} iterations "
            f"({self.solver.n_fev} function evaluations)",
            "",
            f"{'':<{w}} {'Estimate':>12} {'Std.Error':>11} {'t value':>9} {'Pr(>|t|)':>10}   "
            f"{int(level * 100)}% CI (Wald)",
        ]
        for i, name in enumerate(self.param_names):
            pv = self.pvalues[i]
            lines.append(
                f"{name:<{w}} {self.params[i]:>12.6g} {self.se[i]:>11.4g} {self.tvalues[i]:>9.3f} "
                f"{pv:>10.3g} {stars(pv):<3} [{ci[i, 0]:.5g}, {ci[i, 1]:.5g}]"
            )
        lines += [
            "---",
            "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1",
            "",
            f"Residual standard error: {self.sigma:.5g} on {self.df} degrees of freedom",
            f"SSE: {self.sse:.6g}   R²: {self.r_squared:.4f}   AIC: {self.aic:.2f}   BIC: {self.bic:.2f}",
        ]
        if self.params.size > 1:
            c = self.corr
            off = np.abs(c[np.triu_indices_from(c, 1)])
            if np.nanmax(off) > 0.95:
                lines.append(
                    f"Note: max |correlation| between estimates is {np.nanmax(off):.3f}; "
                    "consider reparameterising."
                )
        if not self.converged:
            lines.append("WARNING: the solver did not converge; estimates are unreliable.")
        return "\n".join(lines)

    def __repr__(self) -> str:
        est = ", ".join(f"{n}={v:.5g}" for n, v in zip(self.param_names, self.params, strict=True))
        return f"<FitResult {self.model.name}: {est}; SSE={self.sse:.5g}; converged={self.converged}>"

    def to_frame(self, level: float = 0.95):
        """Parameter table as a pandas DataFrame (requires pandas)."""
        import pandas as pd

        ci = self.conf_int(level)
        return pd.DataFrame(
            {
                "estimate": self.params,
                "std_error": self.se,
                "t_value": self.tvalues,
                "p_value": self.pvalues,
                f"ci_{int(level * 100)}_lower": ci[:, 0],
                f"ci_{int(level * 100)}_upper": ci[:, 1],
            },
            index=list(self.param_names),
        )


def compare(*fits: FitResult) -> str:
    """Side-by-side AIC / BIC / σ table for competing models fitted to the same data."""
    aics = np.array([f.aic for f in fits])
    best = aics.min()
    weights = np.exp(-0.5 * (aics - best))
    weights /= weights.sum()
    rows = [
        f"{'model':<24}{'params':>7}{'SSE':>12}{'sigma':>10}{'AIC':>10}"
        f"{'ΔAIC':>8}{'BIC':>10}{'Akaike wt':>11}"
    ]
    for f, w in sorted(zip(fits, weights, strict=True), key=lambda t: t[0].aic):
        rows.append(
            f"{f.model.name:<24}{f.params.size:>7}{f.sse:>12.5g}{f.sigma:>10.4g}"
            f"{f.aic:>10.2f}{f.aic - best:>8.2f}{f.bic:>10.2f}{w:>11.3f}"
        )
    return "\n".join(rows)


def extra_ss_test(reduced: FitResult, full: FitResult) -> tuple[float, float]:
    """F-test for nested models (extra sum of squares). Returns (F, p-value)."""
    df_diff = reduced.df - full.df
    if df_diff <= 0:
        raise ValueError("'reduced' must have fewer parameters than 'full'")
    F = ((reduced.sse - full.sse) / df_diff) / (full.sse / full.df)
    return float(F), float(stats.f.sf(F, df_diff, full.df))
