"""Gauss–Newton vs Levenberg–Marquardt on the NIST BoxBOD problem.

From NIST's 'start 1' (b1 = 1, b2 = 1) the textbook Gauss–Newton step overshoots
into a region where exp(−b2·x) underflows, and it never recovers. Step halving
fixes it. Levenberg–Marquardt, despite its reputation, stalls on a plateau here
(SciPy's MINPACK does too). From 'start 2' every method converges.

Run: python examples/02_gauss_newton_vs_lm.py
"""

import matplotlib.pyplot as plt
import numpy as np
from _common import save

import nlsfit
from nlsfit.datasets import load, log_relative_error
from nlsfit.plotting import plot_convergence, plot_sse_surface

d = load("boxbod")
print("Certified:", d.certified, "SSE", d.certified_sse, "\n")

for label, start in (("start 1 (hard)", d.starts[0]), ("start 2 (easy)", d.starts[1])):
    print(f"--- {label}: {start}")
    for method in ("gn-pure", "gn", "lm"):
        r = nlsfit.fit(d.model, d.x, d.y, p0=start, method=method)
        lre = log_relative_error(r.params, d.certified).min()
        print(f"{method:8} {r.solver.message[:62]:<62} iters={r.solver.n_iter:3}  correct digits={lre:4.1f}")
    print()

# Visualise both solvers on the SSE landscape from a moderately hard start
start = (120.0, 2.0)
gn = nlsfit.fit(d.model, d.x, d.y, p0=start, method="gn")
lm = nlsfit.fit(d.model, d.x, d.y, p0=start, method="lm")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
plot_sse_surface(
    d.model,
    d.x,
    d.y,
    (80, 280),
    (0.05, 2.5),
    {"GN (step halving)": gn, "LM": lm},
    ax=axes[0],
    title=f"BoxBOD SSE landscape, start {start}",
)
plot_convergence({"GN": gn, "LM": lm}, ax=axes[1])
axes[1].axhline(d.certified_sse, color="k", ls=":", lw=1, label="certified minimum")
axes[1].legend(frameon=False)
save(fig, "gn_vs_lm_boxbod.png")

# Puromycin: the same pure Gauss–Newton that is fast from a good start fails from a poor one
p = load("puromycin")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, start, note in (
    (axes[0], (160.0, 0.02), "good start: both converge"),
    (axes[1], (240.0, 0.25), "poor start: pure GN wanders off, LM converges"),
):
    with np.errstate(all="ignore"):
        gn = nlsfit.fit(p.model, p.x, p.y, p0=start, method="gn-pure")
        lm = nlsfit.fit(p.model, p.x, p.y, p0=start, method="lm", lam0=50.0)
    plot_sse_surface(
        p.model,
        p.x,
        p.y,
        (140, 260),
        (-0.05, 0.3),
        {"GN (pure)": gn, "LM (λ₀=50)": lm},
        ax=ax,
        title=f"Puromycin, {note}",
    )
    print(
        f"start {start}: pure GN converged={gn.converged} ({gn.solver.n_iter} it, SSE {gn.sse:.4g}); "
        f"LM converged={lm.converged} ({lm.solver.n_iter} it, SSE {lm.sse:.4g})"
    )
save(fig, "gn_vs_lm_puromycin.png")
