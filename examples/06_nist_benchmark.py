"""Benchmark the solvers on NIST certified problems, from both official starts.

Accuracy is reported as the LRE (log relative error): the number of
significant digits that agree with NIST's certified values. NIST treats an
LRE of 6 or more as a successful solve.

Run: python examples/06_nist_benchmark.py
"""

import warnings

import numpy as np
from scipy.optimize import least_squares

import nlsfit
from nlsfit.datasets import load, log_relative_error

warnings.filterwarnings("ignore")
NIST = ["misra1a", "danwood", "boxbod", "rat42", "rat43", "mgh09"]
methods = {"Pure GN": "gn-pure", "GN + halving": "gn", "LM (nlsfit)": "lm"}


def cell(params, cert, ok):
    lre = log_relative_error(params, cert).min()
    return lre if ok else 0.0


rows = []
for name in NIST:
    d = load(name)
    for s, start in enumerate(d.starts, 1):
        row = [f"{name} ({d.difficulty})", f"start {s}"]
        for m in methods.values():
            r = nlsfit.fit(d.model, d.x, d.y, p0=start, method=m, max_iter=2000)
            row.append(cell(r.params, d.certified, r.converged))
        ref = least_squares(
            lambda p, d=d: d.model(d.x, *p) - d.y,
            start,
            method="lm",
            xtol=1e-15,
            ftol=1e-15,
            gtol=1e-15,
            max_nfev=100000,
        )
        row.append(cell(ref.x, d.certified, ref.success))
        rows.append(row)

header = ["problem", "start", *methods, "SciPy MINPACK LM"]
widths = [20, 8, 10, 14, 13, 18]
print("Correct significant digits (LRE). 0 = failed or wrong minimum; ≥ 6 = success.\n")
print("".join(f"{h:<{w}}" for h, w in zip(header, widths, strict=True)))
for row in rows:
    print(
        f"{row[0]:<20}{row[1]:<8}" + "".join(f"{v:<{w}.1f}" for v, w in zip(row[2:], widths[2:], strict=True))
    )
solved = np.array([[v >= 6 for v in r[2:]] for r in rows]).sum(axis=0)
print("\nSolved (of 12): " + ", ".join(f"{h} {n}" for h, n in zip(header[2:], solved, strict=True)))
