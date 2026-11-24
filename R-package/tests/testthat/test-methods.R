fit <- nl_fit(rate ~ conc, puromycin, model = "michaelis_menten")

test_that("estimates and standard errors match Bates & Watts (1988)", {
  expect_equal(unname(coef(fit)), c(212.68, 0.06412), tolerance = 1e-4)
  expect_equal(unname(sqrt(diag(vcov(fit)))), c(6.947, 0.008281), tolerance = 1e-3)
  expect_equal(fit$sigma, 10.93, tolerance = 1e-3)
  expect_equal(nobs(fit), 12L)
  expect_equal(fit$df.residual, 10L)
})

test_that("summary, print and logLik-based criteria work", {
  s <- summary(fit, correlation = TRUE)
  expect_s3_class(s, "summary.nlestim")
  expect_output(print(s), "Residual standard error: 10.93 on 10 degrees of freedom")
  expect_output(print(fit), "levenberg-marquardt")
  expect_equal(AIC(fit), 95.27, tolerance = 1e-3)
  expect_equal(attr(logLik(fit), "df"), 3L)
  expect_true(is.finite(BIC(fit)))
})

test_that("confidence intervals: profile is asymmetric and matches R's confint", {
  wald <- confint(fit)
  prof <- confint(fit, method = "profile")
  # Values from R's confint() for nls (MASS / stats profile method)
  expect_equal(unname(prof[, 1]), c(197.302, 0.046925), tolerance = 1e-4)
  expect_equal(unname(prof[, 2]), c(229.290, 0.086160), tolerance = 1e-4)
  km <- prof["Km", ]
  expect_gt(km[2] - coef(fit)["Km"], coef(fit)["Km"] - km[1])
  expect_true(all(wald[, 1] < coef(fit) & coef(fit) < wald[, 2]))
  boot <- confint(fit, method = "bootstrap", R = 200)
  expect_true(all(boot[, 1] < coef(fit) & coef(fit) < boot[, 2]))
  expect_equal(rownames(confint(fit, "Km")), "Km")
})

test_that("profile t statistic is zero at the estimate and hits t at the profile limits", {
  pr <- nl_profile(fit, "Km", values = c(coef(fit)["Km"], confint(fit, "Km", method = "profile")))
  expect_equal(pr$tau[1], 0, tolerance = 1e-6)
  expect_equal(abs(pr$tau[2:3]), rep(qt(0.975, 10), 2), tolerance = 1e-4)
})

test_that("bootstrap is reproducible", {
  expect_identical(nl_bootstrap(fit, R = 30, seed = 4), nl_bootstrap(fit, R = 30, seed = 4))
})

test_that("predict returns fits and nested bands", {
  nd <- data.frame(conc = c(0.05, 0.3, 1))
  p0 <- predict(fit, nd)
  pc <- predict(fit, nd, interval = "confidence")
  pp <- predict(fit, nd, interval = "prediction")
  expect_length(p0, 3)
  expect_equal(pc[, "fit"], p0)
  expect_true(all(pp[, "lwr"] < pc[, "lwr"] & pc[, "upr"] < pp[, "upr"]))
  expect_length(predict(fit), 12)
})

test_that("anova performs the extra sum of squares F test", {
  full <- nl_fit(yield ~ nitrogen, crop_yield_nitrogen, model = "asymptotic_regression")
  reduced <- nl_fit(yield ~ a * (1 - exp(-c * nitrogen)), crop_yield_nitrogen, start = c(a = 10, c = 0.01))
  a <- anova(reduced, full)
  expect_s3_class(a, "anova")
  expect_lt(a[2, "Pr(>F)"], 0.05)
  expect_error(anova(full, reduced), "fewer parameters")
})

test_that("weights, user functions and numeric derivatives", {
  f <- nl_fit(conc ~ time, drug_elimination, model = "exponential_decay")
  fw <- nl_fit(conc ~ time, drug_elimination, model = "exponential_decay", weights = 1 / fitted(f)^2)
  expect_true(fw$converged)
  expect_lt(abs(coef(fw)["k"] - 0.35), 3 * sqrt(vcov(fw)["k", "k"]))
  mm <- function(x, v, k) v * x / (k + x)  # deriv() cannot differentiate this: numeric fallback
  fu <- nl_fit(rate ~ mm(conc, Vmax, Km), puromycin, start = c(Vmax = 200, Km = 0.1))
  expect_equal(unname(coef(fu)), unname(coef(fit)), tolerance = 1e-6)
})

test_that("simulated datasets recover their true parameters", {
  cases <- list(
    list(crop_yield_nitrogen, yield ~ nitrogen, "asymptotic_regression", c(9.5, 6, 0.012)),
    list(plant_growth, biomass ~ day, "gompertz", c(180, 6, 0.92)),
    list(dose_response, viability ~ conc, "four_pl", c(5, 98, 1.5, 1.2)),
    list(drug_elimination, conc ~ time, "exponential_decay", c(12, 0.35))
  )
  for (cs in cases) {
    f <- nl_fit(cs[[2]], cs[[1]], model = cs[[3]])
    ci <- confint(f, level = 0.999)
    expect_true(all(ci[, 1] < cs[[4]] & cs[[4]] < ci[, 2]), info = cs[[3]])
  }
})

test_that("plot methods run", {
  pdf(NULL)
  on.exit(dev.off())
  expect_invisible(plot(fit, ask = FALSE))
})

test_that("input errors are clear", {
  expect_error(nl_fit(rate ~ Vmax * conc / (Km + conc), puromycin), "'start' is required")
  expect_error(nl_fit(rate ~ conc, puromycin, model = "nope"), "unknown model")
  expect_error(nl_fit(rate ~ conc + 1, puromycin, model = "michaelis_menten"), "response ~ predictor")
  expect_error(nl_fit(rate ~ conc, puromycin[1:2, ], model = "michaelis_menten"), "more observations")
  expect_error(nl_fit(rate ~ Vmax * conc / (Km + conc), puromycin, start = c(Vmax = 200)),
               "every parameter|object 'Km' not found")
})
