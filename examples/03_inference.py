"""Confidence intervals done right: Wald vs profile likelihood vs bootstrap.

Wald intervals (estimate ± t·SE) assume the model is nearly linear in each
parameter. When it is not, profile-likelihood intervals (what R's confint()
uses for nls) are more accurate and can be asymmetric.

Run: python examples/03_inference.py
"""

import matplotlib.pyplot as plt
import numpy as np
from _common import save
from scipy import stats

import nlsfit
from nlsfit.plotting import plot_residuals

d = nlsfit.datasets.load("puromycin")
res = nlsfit.fit(d.model, d.x, d.y)

methods = {
    "Wald": res.conf_int(method="wald"),
    "Profile": res.conf_int(method="profile"),
    "Bootstrap": res.conf_int(method="bootstrap", n_boot=2000),
}
print(f"{'':10}" + "".join(f"{m:>28}" for m in methods))
for j, name in enumerate(res.param_names):
    print(f"{name:10}" + "".join(f"   [{ci[j, 0]:>10.5g}, {ci[j, 1]:>10.5g}]" for ci in methods.values()))
print("\nCorrelation between estimates:\n", np.round(res.corr, 3))

# Profile t plot: a straight line means Wald intervals are fine; curvature means they are not
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
q = stats.t.ppf(0.975, res.df)
for j, ax in enumerate(axes):
    grid = res.params[j] + np.linspace(-4, 4, 41) * res.se[j]
    S = res.profile(j, grid)
    tau = np.sign(grid - res.params[j]) * np.sqrt(np.maximum(S - res.sse, 0)) / res.sigma
    ax.plot(grid, tau, color="#0e7490", lw=2, label="profile t (exact)")
    ax.plot(grid, (grid - res.params[j]) / res.se[j], "--", color="#9ca3af", label="Wald (linear approx.)")
    for v in (-q, q):
        ax.axhline(v, color="k", lw=0.7, ls=":")
    ax.set(xlabel=res.param_names[j], ylabel="τ", title=f"Profile t for {res.param_names[j]}")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
fig.tight_layout()
save(fig, "profile_t.png")

save(plot_residuals(res), "residual_diagnostics.png")

boot = res.bootstrap(n_boot=2000)
fig, ax = plt.subplots(figsize=(5.5, 4.2))
ax.scatter(boot[:, 0], boot[:, 1], s=4, alpha=0.3, color="#1c64d1")
ax.plot(*res.params, "*", ms=14, color="#d9480f", mec="white")
ax.set(xlabel="Vmax", ylabel="Km", title="2,000 bootstrap estimates (note the correlation)")
ax.grid(alpha=0.25)
save(fig, "bootstrap_cloud.png")
