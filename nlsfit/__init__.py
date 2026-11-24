"""nlsfit — nonlinear least squares with Gauss–Newton and Levenberg–Marquardt.

Readable, tested implementations of the two classic algorithms, with the full
statistical toolkit you need to report a nonlinear regression: standard
errors, Wald / profile-likelihood / bootstrap confidence intervals,
confidence and prediction bands, AIC/BIC model comparison and diagnostics.

Quick start
-----------
>>> import nlsfit
>>> data = nlsfit.datasets.load("puromycin")
>>> res = nlsfit.fit(nlsfit.models.michaelis_menten, data.x, data.y)
>>> print(round(res["Vmax"], 2), round(res["Km"], 5))
212.68 0.06412
"""

from . import datasets, models
from .fit import FitResult, compare, extra_ss_test, fit
from .models import MODELS, Model, get_model
from .solvers import gauss_newton, levenberg_marquardt, numerical_jacobian

__all__ = [
    "MODELS",
    "FitResult",
    "Model",
    "compare",
    "datasets",
    "extra_ss_test",
    "fit",
    "gauss_newton",
    "get_model",
    "levenberg_marquardt",
    "models",
    "numerical_jacobian",
]

__version__ = "1.0.0"
