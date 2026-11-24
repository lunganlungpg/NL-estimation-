# Changelog

## 1.0.0

First release.

- `nlsfit` package: Gauss–Newton (pure and step-halving) and Levenberg–Marquardt (Marquardt/Levenberg scaling,
  Nielsen/classic damping), Bates–Watts relative-offset convergence test, numerical Jacobians.
- Inference: SE, t and p values, Wald / profile-likelihood / bootstrap intervals, delta-method confidence and
  prediction bands, AIC/BIC, model comparison, extra-sum-of-squares F-test, weighted fits.
- 12 built-in models with analytic Jacobians and automatic starting values.
- 11 datasets (6 NIST StRD, Puromycin, 4 simulated) as CSV and an Excel workbook.
- Interactive web lab: SSE landscape, step-by-step GN vs LM, λ-path, fit-your-own-data.
- Six examples, theory guide, 49 tests verified against NIST certified values and SciPy.
