"""Which growth curve? Compare logistic, Gompertz and Richards with AIC/BIC.

The plant-growth data were simulated from a Gompertz curve, so AIC should
prefer it, and its residuals should show no pattern.

Run: python examples/04_model_selection.py
"""

import matplotlib.pyplot as plt
import numpy as np
from _common import save

import nlsfit

d = nlsfit.datasets.load("plant_growth")
fits = {
    "logistic": nlsfit.fit(nlsfit.models.logistic, d.x, d.y),
    "gompertz": nlsfit.fit(nlsfit.models.gompertz, d.x, d.y),
}
# Richards has 4 parameters; start it from the logistic solution (b4 = 1 is logistic)
lg = fits["logistic"]
fits["richards"] = nlsfit.fit(
    nlsfit.models.richards, d.x, d.y, p0=[lg["Asym"], lg["xmid"] / lg["scal"], 1 / lg["scal"], 1.0]
)

print(nlsfit.compare(*fits.values()))
print("\nTrue (simulation) parameters for Gompertz:", d.true_params)
print(fits["gompertz"].summary())

grid = np.linspace(d.x.min(), d.x.max(), 300)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
axes[0].scatter(d.x, d.y, color="#1f2937", s=26, zorder=3, label="data")
colors = {"logistic": "#d9480f", "gompertz": "#0e7490", "richards": "#7c3aed"}
for name, f in fits.items():
    axes[0].plot(grid, f.predict(grid), color=colors[name], lw=2, label=f"{name} (AIC {f.aic:.1f})")
    axes[1].plot(d.x, f.residuals, "o-", color=colors[name], ms=4, label=name)
axes[0].set(xlabel=d.x_label, ylabel=d.y_label, title="Three growth curves")
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set(xlabel=d.x_label, ylabel="residual", title="Residuals: look for systematic waves")
for a in axes:
    a.legend(frameon=False, fontsize=8)
    a.grid(alpha=0.25)
save(fig, "model_selection.png")
