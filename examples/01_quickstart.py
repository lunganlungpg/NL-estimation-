"""Fit an enzyme-kinetics model in four lines and report it properly.

Run: python examples/01_quickstart.py
"""

import matplotlib.pyplot as plt
from _common import save

import nlsfit
from nlsfit.plotting import plot_fit

data = nlsfit.datasets.load("puromycin")

result = nlsfit.fit(nlsfit.models.michaelis_menten, data.x, data.y)  # starting values are automatic
print(result.summary())

fig, ax = plt.subplots(figsize=(6.5, 4.2))
plot_fit(
    result,
    ax=ax,
    band="prediction",
    xlabel=data.x_label,
    ylabel=data.y_label,
    title="Puromycin: Michaelis–Menten fit with 95% bands",
)
save(fig, "quickstart_fit.png")
