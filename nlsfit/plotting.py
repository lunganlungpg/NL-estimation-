"""Matplotlib helpers (install with ``pip install nlsfit[plot]``)."""

from __future__ import annotations

import numpy as np

COLORS = {"data": "#1f2937", "fit": "#0e7490", "gn": "#d9480f", "lm": "#1c64d1", "band": "#0e7490"}


def _plt():
    import matplotlib.pyplot as plt

    return plt


def plot_fit(
    result,
    ax=None,
    band: str | None = "confidence",
    level: float = 0.95,
    n_points: int = 300,
    title: str | None = None,
    xlabel="x",
    ylabel="y",
    logx=False,
):
    """Data, fitted curve and an optional confidence or prediction band."""
    plt = _plt()
    ax = ax or plt.gca()
    x = result.x
    if logx:
        grid = np.geomspace(x[x > 0].min(), x.max(), n_points)
    else:
        grid = np.linspace(x.min(), x.max(), n_points)
    ax.scatter(x, result.y, s=28, color=COLORS["data"], zorder=3, label="data")
    if band:
        yhat, lo, hi = result.predict(grid, interval=band, level=level)
        ax.fill_between(
            grid, lo, hi, color=COLORS["band"], alpha=0.18, lw=0, label=f"{int(level * 100)}% {band} band"
        )
        if band == "prediction":
            _, clo, chi = result.predict(grid, interval="confidence", level=level)
            ax.fill_between(
                grid,
                clo,
                chi,
                color=COLORS["band"],
                alpha=0.25,
                lw=0,
                label=f"{int(level * 100)}% confidence band",
            )
    else:
        yhat = result.predict(grid)
    ax.plot(grid, yhat, color=COLORS["fit"], lw=2.2, label="fitted curve")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or result.model.formula, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    return ax


def plot_residuals(result, fig=None):
    """Four diagnostic panels: residuals vs fitted, vs x, normal Q–Q and histogram."""
    from scipy import stats

    plt = _plt()
    fig = fig or plt.figure(figsize=(10, 7))
    axes = fig.subplots(2, 2)
    std_res = result.residuals * result.sqrt_weights / result.sigma
    a = axes[0, 0]
    a.scatter(result.fitted, std_res, s=22, color=COLORS["fit"])
    a.axhline(0, color="k", lw=0.8)
    for v in (-2, 2):
        a.axhline(v, color="k", lw=0.6, ls=":")
    a.set(title="Standardised residuals vs fitted", xlabel="fitted", ylabel="residual / σ")
    a = axes[0, 1]
    xv = result.x if result.x.ndim == 1 else np.arange(result.n)
    a.scatter(xv, std_res, s=22, color=COLORS["fit"])
    a.axhline(0, color="k", lw=0.8)
    a.set(title="Residuals vs x (look for patterns)", xlabel="x", ylabel="residual / σ")
    a = axes[1, 0]
    (osm, osr), (slope, intercept, _) = stats.probplot(std_res)
    a.scatter(osm, osr, s=22, color=COLORS["fit"])
    a.plot(osm, slope * np.asarray(osm) + intercept, color="k", lw=0.8)
    a.set(title="Normal Q–Q", xlabel="theoretical quantile", ylabel="sample quantile")
    a = axes[1, 1]
    a.hist(std_res, bins=min(15, max(5, result.n // 3)), color=COLORS["fit"], alpha=0.7)
    a.set(title="Histogram of residuals", xlabel="residual / σ", ylabel="count")
    for a in axes.ravel():
        a.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_convergence(results: dict, ax=None, log=True):
    """SSE against iteration for one or more fits, e.g. ``{"GN": fit_gn, "LM": fit_lm}``."""
    plt = _plt()
    ax = ax or plt.gca()
    for i, (label, res) in enumerate(results.items()):
        hist = res.history if hasattr(res, "history") else res
        it = [h.iteration for h in hist if h.accepted]
        sse = [h.sse for h in hist if h.accepted]
        color = COLORS.get(label.lower(), f"C{i}")
        ax.plot(it, sse, "o-", ms=4, color=color, label=label)
    if log:
        ax.set_yscale("log")
    ax.set(xlabel="iteration", ylabel="SSE", title="Convergence")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, which="both")
    return ax


def sse_grid(model, x, y, p1_range, p2_range, n=200, fixed=None, idx=(0, 1)):
    """SSE evaluated on a grid over two parameters (others held at ``fixed``)."""
    P1, P2 = np.meshgrid(np.linspace(*p1_range, n), np.linspace(*p2_range, n))
    S = np.empty_like(P1)
    base = np.array(fixed if fixed is not None else [0.0] * model.n_params, dtype=float)
    with np.errstate(all="ignore"):
        for i in range(n):
            for j in range(n):
                th = base.copy()
                th[idx[0]], th[idx[1]] = P1[i, j], P2[i, j]
                r = model(x, *th) - y
                S[i, j] = r @ r
    return P1, P2, S


def plot_sse_surface(
    model,
    x,
    y,
    p1_range,
    p2_range,
    paths: dict | None = None,
    ax=None,
    n=200,
    levels=30,
    title="SSE landscape and solver paths",
):
    """Log-SSE contour map for a two-parameter model with optimisation paths overlaid."""
    plt = _plt()
    ax = ax or plt.gca()
    P1, P2, S = sse_grid(model, x, y, p1_range, p2_range, n=n)
    logS = np.log10(np.where(np.isfinite(S) & (S > 0), S, np.nan))
    cs = ax.contourf(P1, P2, logS, levels=levels, cmap="Greys_r", alpha=0.9)
    ax.contour(P1, P2, logS, levels=levels, colors="white", linewidths=0.3, alpha=0.6)
    plt.colorbar(cs, ax=ax, label="log₁₀ SSE")
    for i, (label, res) in enumerate((paths or {}).items()):
        hist = [h for h in (res.history if hasattr(res, "history") else res) if h.accepted]
        pts = np.array([h.params[:2] for h in hist])
        color = COLORS.get(label.lower().split()[0], f"C{i}")
        ax.plot(pts[:, 0], pts[:, 1], "o-", color=color, ms=4, lw=1.8, label=label)
        ax.plot(*pts[0], "s", color=color, ms=8, mec="white")
        ax.plot(*pts[-1], "*", color=color, ms=14, mec="white")
    ax.set_xlim(p1_range)
    ax.set_ylim(p2_range)
    ax.set_xlabel(model.param_names[0])
    ax.set_ylabel(model.param_names[1])
    ax.set_title(title, fontsize=10)
    if paths:
        ax.legend(frameon=True, fontsize=8, loc="best")
    return ax
