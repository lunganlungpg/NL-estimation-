lre <- function(est, cert) min(pmin(11, -log10(abs(est - cert) / abs(cert))))

nist_fit <- function(ds, start_col, method) {
  cc <- nist_certified[nist_certified$dataset == ds, ]
  st <- stats::setNames(cc[[start_col]], cc$parameter)
  fit <- suppressWarnings(nl_fit(stats::as.formula(cc$formula[1]), get(ds), start = st, method = method))
  list(fit = fit, cert = cc$certified, sse = cc$certified_sse[1])
}

test_that("Levenberg-Marquardt reproduces NIST certified values from start 2", {
  for (ds in unique(nist_certified$dataset)) {
    r <- nist_fit(ds, "start2", "lm")
    expect_true(r$fit$converged, info = ds)
    expect_gte(lre(coef(r$fit), r$cert), 6)
    expect_equal(deviance(r$fit), r$sse, tolerance = 1e-7)
  }
})

test_that("Levenberg-Marquardt solves the hard NIST start 1 problems", {
  for (ds in c("misra1a", "danwood", "rat42", "rat43", "mgh09")) {
    r <- nist_fit(ds, "start1", "lm")
    expect_true(r$fit$converged, info = ds)
    expect_gte(lre(coef(r$fit), r$cert), 6)
  }
})

test_that("step halving rescues Gauss-Newton on BoxBOD; failures are reported", {
  pure <- nist_fit("boxbod", "start1", "gn-pure")$fit
  safe <- nist_fit("boxbod", "start1", "gn")
  lm <- nist_fit("boxbod", "start1", "lm")$fit
  expect_false(pure$converged)
  expect_match(pure$message, "diverged")
  expect_true(safe$fit$converged)
  expect_gte(lre(coef(safe$fit), safe$cert), 6)
  expect_false(lm$converged)
  expect_match(lm$message, "stalled")
  expect_warning(nl_fit(bod ~ days, boxbod, model = "exponential_rise",
                        start = c(a = 1, k = 1), method = "gn-pure"), "did not converge")
})

test_that("results agree with stats::nls on Puromycin", {
  fit <- nl_fit(rate ~ conc, puromycin, model = "michaelis_menten")
  # nls() stops at a relative offset of 1e-5 by default (and cannot reach the
  # 1e-10 that nlestim uses), so agreement is to about 5 significant digits.
  ref <- stats::nls(rate ~ Vmax * conc / (Km + conc), puromycin, start = list(Vmax = 200, Km = 0.1))
  expect_equal(unname(coef(fit)), unname(coef(ref)), tolerance = 1e-5)
  expect_equal(unname(sqrt(diag(vcov(fit)))), unname(sqrt(diag(vcov(ref)))), tolerance = 1e-5)
  expect_equal(fit$sigma, summary(ref)$sigma, tolerance = 1e-6)
  expect_lte(deviance(fit), deviance(ref))
})

test_that("both LM scalings and damping rules converge to the same answer", {
  for (sc in c("marquardt", "levenberg")) for (up in c("nielsen", "classic")) {
    f <- nl_fit(rate ~ conc, puromycin, model = "michaelis_menten",
                control = nl_control(scaling = sc, update = up))
    expect_true(f$converged)
    expect_equal(unname(coef(f)["Vmax"]), 212.6837, tolerance = 1e-5)
  }
})

test_that("relative offset and numeric Jacobian behave as documented", {
  J <- cbind(1, 1:10)
  y <- c(2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1, 18.0, 20.2)
  beta <- qr.coef(qr(J), y)
  expect_lt(relative_offset(J, J %*% beta - y), 1e-10)
  expect_gt(relative_offset(J, J %*% (beta + 0.1) - y), 1e-2)
  f <- function(p) c(sin(p[1]) * p[2], p[2]^3)
  exact <- rbind(c(cos(0.7) * 1.3, sin(0.7)), c(0, 3 * 1.3^2))
  expect_lt(max(abs(numeric_jacobian(f, c(0.7, 1.3)) - exact)), 1e-8)
  expect_lt(max(abs(numeric_jacobian(f, c(0.7, 1.3), "forward") - exact)), 1e-6)
})

test_that("history records every iteration", {
  f <- nl_fit(rate ~ conc, puromycin, model = "michaelis_menten")
  expect_s3_class(f$history, "data.frame")
  expect_true(all(c("iteration", "Vmax", "Km", "sse", "lambda", "accepted") %in% names(f$history)))
  expect_lte(tail(f$history$sse, 1), f$history$sse[1])
})
