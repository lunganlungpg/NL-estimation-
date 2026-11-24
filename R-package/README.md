# nlestim: nonlinear least squares for R

Gauss-Newton and Levenberg-Marquardt, implemented readably, with the inference
you need to report a nonlinear regression and a **one-call workflow that works
on any dataset**. The R companion to the
[`nlsfit` Python package and interactive lab](https://lunganlungpg.github.io/NL-estimation-/).

## Install

```r
# install.packages("remotes")
remotes::install_github("lunganlungpg/NL-estimation-", subdir = "R-package",
                        build_vignettes = TRUE)
```

Only base R is required. `readxl` is used to read Excel files if installed.

## Analyse a dataset in one call

```r
library(nlestim)
wf <- nl_workflow(plant_growth, x = "day", y = "biomass", output_dir = "growth-results")
wf          # model comparison (AICc), best model, profile CIs, diagnostic notes
plot(wf)    # all candidate curves, best fit with bands, residual diagnostics
```

`nl_workflow()` picks candidate models from the shape of the data (or takes
`models = c(...)`), fits each, ranks them by AICc with Akaike weights, and
writes CSV tables, a PNG figure and a text report when `output_dir` is set.

Your own data:

```r
my_data <- nl_read("my_experiment.xlsx", sheet = "Sheet1")   # or a .csv file
nl_workflow(my_data, x = "dose", y = "response")
```

## Fit one model

```r
fit <- nl_fit(rate ~ conc, puromycin, model = "michaelis_menten")     # automatic starts
fit <- nl_fit(rate ~ Vmax * conc / (Km + conc), puromycin,            # any formula
              start = c(Vmax = 200, Km = 0.1), method = "gn")
summary(fit); confint(fit, method = "profile"); AIC(fit)
predict(fit, data.frame(conc = 0.5), interval = "prediction")
plot(fit)
```

| Function | Purpose |
|---|---|
| `nl_workflow()` | Fit and compare candidate models on any dataset; export results |
| `nl_fit()` | Fit one model (`method = "lm"`, `"gn"` or `"gn-pure"`), optional weights |
| `summary()`, `coef()`, `vcov()`, `confint()`, `predict()`, `anova()`, `AIC()`, `plot()` | Standard methods; `confint()` offers Wald, profile and bootstrap |
| `nl_compare()` | AICc/AIC/BIC table for fits |
| `nl_compare_solvers()` | Run both algorithms from one start; draw the SSE landscape and paths |
| `nl_profile()`, `nl_bootstrap()` | Profile t statistics; residual bootstrap |
| `nl_models()`, `nl_suggest_models()` | Built-in model library |
| `nl_datasets()`, `nl_read()` | Bundled data index; read CSV / Excel |
| `nl_gauss_newton()`, `nl_levenberg_marquardt()`, `relative_offset()` | Low-level solvers |

## Datasets

`misra1a`, `boxbod`, `danwood`, `rat42`, `rat43`, `mgh09` (NIST, with
`nist_certified`), `puromycin` (real), and simulated `crop_yield_nitrogen`,
`plant_growth`, `dose_response` and `drug_elimination`. `nl_datasets()` lists
them with a ready-to-run workflow call for each.

## Verified

* Agrees with `stats::nls()` (estimates, standard errors, profile intervals).
* Reproduces NIST certified values: Levenberg-Marquardt solves 11 of 12 runs
  (both official starts of six problems) to at least 6 significant digits.
* `R CMD check --as-cran` passes; 156 testthat expectations.

See `vignette("nlestim")` and `vignette("algorithms")`.
