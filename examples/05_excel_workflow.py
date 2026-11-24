"""Read data from the Excel workbook, fit, and write a results workbook.

This is the workflow most applied users want: data in a spreadsheet in,
publication-ready tables and a figure out. Needs: pip install "nlsfit[excel]"

Run: python examples/05_excel_workflow.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import save

import nlsfit
from nlsfit.plotting import plot_fit

ROOT = Path(__file__).resolve().parents[1]
sheet = pd.read_excel(ROOT / "data" / "nonlinear_datasets.xlsx", sheet_name="crop_yield_nitrogen")
x_col, y_col = sheet.columns[:2]
x, y = sheet[x_col].to_numpy(), sheet[y_col].to_numpy()

res = nlsfit.fit(nlsfit.models.asymptotic_regression, x, y)
print(res.summary())

# Agronomic quantities derived from the fitted Mitscherlich curve
a, b, c = res.params
yield_zero_n = a - b
econ_optimum = np.log(b * c / 0.006) / c  # where marginal yield = price ratio (0.006 t grain per kg N)
print(f"\nPredicted yield without N: {yield_zero_n:.2f} t/ha;  plateau: {a:.2f} t/ha")
print(f"Economic optimum N rate (price ratio 6 kg grain per kg N): {econ_optimum:.0f} kg N/ha")

grid = np.linspace(0, 240, 49)
yhat, lo, hi = res.predict(grid, interval="confidence")
out = ROOT / "examples" / "output"
out.mkdir(exist_ok=True)
with pd.ExcelWriter(out / "crop_yield_results.xlsx") as xl:
    res.to_frame().to_excel(xl, sheet_name="parameters")
    pd.DataFrame({x_col: x, y_col: y, "fitted": res.fitted, "residual": res.residuals}).to_excel(
        xl, sheet_name="fitted values", index=False
    )
    pd.DataFrame({x_col: grid, "prediction": yhat, "lower 95%": lo, "upper 95%": hi}).to_excel(
        xl, sheet_name="prediction curve", index=False
    )
    pd.DataFrame(
        {
            "statistic": ["n", "df", "residual SE", "SSE", "R²", "AIC", "BIC", "converged"],
            "value": [res.n, res.df, res.sigma, res.sse, res.r_squared, res.aic, res.bic, res.converged],
        }
    ).to_excel(xl, sheet_name="fit statistics", index=False)
print(f"Wrote {out / 'crop_yield_results.xlsx'}")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
plot_fit(res, ax=ax, xlabel=x_col, ylabel=y_col, title="Wheat yield response to nitrogen (Mitscherlich)")
ax.axvline(econ_optimum, color="#d9480f", ls="--", lw=1.2, label="economic optimum")
ax.legend(frameon=False, fontsize=8)
save(fig, "crop_yield_fit.png")
