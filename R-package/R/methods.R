#' Methods for nonlinear fits
#'
#' Standard R model methods for objects returned by [nl_fit()].
#'
#' @param object,x An `"nlestim"` fit.
#' @param ... Unused.
#' @return `coef()` the estimates; `vcov()` their covariance matrix
#'   `s^2 (J'J)^-1`; `logLik()` the Gaussian log-likelihood (so [AIC()] and
#'   [BIC()] work); `nobs()`, `deviance()` (weighted residual sum of squares).
#' @name nlestim-methods
#' @examples
#' fit <- nl_fit(rate ~ conc, subset(Puromycin, state == "treated"),
#'               model = "michaelis_menten")
#' fit
#' vcov(fit)
#' AIC(fit)
NULL

#' @rdname nlestim-methods
#' @export
print.nlestim <- function(x, ...) {
  cat("Nonlinear regression model fitted by", x$method, "\n")
  cat("  model:", .model_label(x), "\n")
  cat("  data:  ", length(x$y), "observations\n\n")
  print(signif(x$coefficients, 6))
  cat(sprintf("\nResidual sum of squares: %.6g    residual standard error: %.5g on %d df\n",
              x$sse, x$sigma, x$df.residual))
  cat(sprintf("%s after %d iterations\n", x$message, x$iterations))
  invisible(x)
}

#' @rdname nlestim-methods
#' @export
vcov.nlestim <- function(object, ...) object$cov

#' @rdname nlestim-methods
#' @export
nobs.nlestim <- function(object, ...) length(object$y)

#' @rdname nlestim-methods
#' @export
deviance.nlestim <- function(object, ...) object$sse

#' @rdname nlestim-methods
#' @export
logLik.nlestim <- function(object, ...) {
  n <- length(object$y)
  w <- object$weights
  val <- -0.5 * n * (log(2 * pi * object$sse / n) + 1) + 0.5 * sum(log(w[w > 0]))
  structure(val, df = length(object$coefficients) + 1L, nobs = n, class = "logLik")
}

#' Summarise a nonlinear fit
#'
#' Coefficient table (estimate, standard error, t value, p value), residual
#' standard error, R-squared, AIC/BIC, convergence information and a warning
#' when estimates are almost perfectly correlated.
#'
#' @param object An `"nlestim"` fit.
#' @param correlation Logical; print the correlation matrix of the estimates.
#' @param x A `"summary.nlestim"` object.
#' @param digits Number of significant digits to print.
#' @param ... Unused.
#' @return An object of class `"summary.nlestim"`.
#' @examples
#' fit <- nl_fit(rate ~ conc, subset(Puromycin, state == "treated"),
#'               model = "michaelis_menten")
#' summary(fit, correlation = TRUE)
#' @export
summary.nlestim <- function(object, correlation = FALSE, ...) {
  est <- object$coefficients
  se <- sqrt(diag(object$cov))
  tval <- est / se
  pval <- 2 * stats::pt(abs(tval), object$df.residual, lower.tail = FALSE)
  coefs <- cbind(Estimate = est, `Std. Error` = se, `t value` = tval, `Pr(>|t|)` = pval)
  y <- object$y
  w <- object$weights
  sst <- sum(w * (y - stats::weighted.mean(y, w))^2)
  corr <- object$cov / outer(se, se)
  structure(list(
    model = .model_label(object), method = object$method, coefficients = coefs,
    sigma = object$sigma, df = object$df.residual, sse = object$sse,
    r_squared = 1 - object$sse / sst, aic = stats::AIC(object), bic = stats::BIC(object),
    converged = object$converged, message = object$message, iterations = object$iterations,
    n_fev = object$n_fev, correlation = corr, show_correlation = correlation
  ), class = "summary.nlestim")
}

#' @rdname summary.nlestim
#' @export
print.summary.nlestim <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat("Model:  ", x$model, "\n")
  cat("Method: ", x$method, "\n")
  cat(sprintf("Status:  %s after %d iterations (%d function evaluations)\n\n",
              x$message, x$iterations, x$n_fev))
  stats::printCoefmat(x$coefficients, digits = digits)
  cat(sprintf("\nResidual standard error: %s on %d degrees of freedom\n",
              format(signif(x$sigma, digits)), x$df))
  cat(sprintf("Residual sum of squares: %s   R-squared: %.4f   AIC: %.2f   BIC: %.2f\n",
              format(signif(x$sse, digits)), x$r_squared, x$aic, x$bic))
  if (x$show_correlation) {
    cat("\nCorrelation of parameter estimates:\n")
    print(round(x$correlation, 3))
  }
  off <- abs(x$correlation[upper.tri(x$correlation)])
  if (length(off) && any(off > 0.95, na.rm = TRUE)) {
    cat(sprintf("\nNote: max |correlation| between estimates is %.3f; consider reparameterising.\n",
                max(off, na.rm = TRUE)))
  }
  if (!x$converged) cat("\nWARNING: the solver did not converge; estimates are unreliable.\n")
  invisible(x)
}

#' Predictions with confidence or prediction bands
#'
#' Bands use the delta method: the variance of the fitted curve at a new point
#' is `g' V g`, where `g` is the gradient of the model with respect to the
#' parameters and `V = vcov(object)`. Prediction bands add the residual
#' variance.
#'
#' @param object An `"nlestim"` fit.
#' @param newdata Data frame of new predictor values. Defaults to the data used
#'   in the fit.
#' @param interval `"none"`, `"confidence"` or `"prediction"`.
#' @param level Confidence level.
#' @param ... Unused.
#' @return A numeric vector (`interval = "none"`) or a matrix with columns
#'   `fit`, `lwr` and `upr`.
#' @examples
#' fit <- nl_fit(rate ~ conc, subset(Puromycin, state == "treated"),
#'               model = "michaelis_menten")
#' predict(fit, data.frame(conc = c(0.05, 0.5, 1)), interval = "confidence")
#' @export
predict.nlestim <- function(object, newdata = NULL, interval = c("none", "confidence", "prediction"),
                            level = 0.95, ...) {
  interval <- match.arg(interval)
  nd <- if (is.null(newdata)) object$frame else as.list(newdata)
  fit <- object$predict_fun(object$coefficients, nd)
  if (interval == "none") return(fit)
  G <- object$jac_fun(object$coefficients, nd)
  v <- rowSums((G %*% object$cov) * G)
  if (interval == "prediction") v <- v + object$sigma^2
  q <- stats::qt(0.5 + level / 2, object$df.residual)
  half <- q * sqrt(pmax(v, 0))
  cbind(fit = fit, lwr = fit - half, upr = fit + half)
}

.fit_fixed <- function(object, j, value, start) {
  pn <- object$pnames
  free <- setdiff(seq_along(pn), j)
  sw <- sqrt(object$weights)
  full <- function(q) {
    th <- numeric(length(pn))
    th[free] <- q
    th[j] <- value
    th
  }
  res <- function(q) sw * (object$predict_fun(full(q)) - object$y)
  jac <- function(q) sw * object$jac_fun(full(q))[, free, drop = FALSE]
  nl_levenberg_marquardt(res, jac, start, max_iter = 200)
}

#' Profile the residual sum of squares
#'
#' Fixes parameter `which` at each of `values`, re-optimises the others, and
#' returns the profile t statistic
#' `tau = sign(value - estimate) * sqrt(S(value) - S(estimate)) / s`.
#' A straight-line profile means Wald intervals are accurate; curvature means
#' profile intervals should be preferred.
#'
#' @param fitted An `"nlestim"` fit.
#' @param which Parameter name or index.
#' @param values Values at which to profile. By default 41 points spanning
#'   +/- 4 standard errors.
#' @param ... Unused.
#' @return A data frame with columns `value`, `sse` and `tau`.
#' @examples
#' fit <- nl_fit(rate ~ conc, subset(Puromycin, state == "treated"),
#'               model = "michaelis_menten")
#' pr <- nl_profile(fit, "Km")
#' plot(pr$value, pr$tau, type = "l", xlab = "Km", ylab = "profile t")
#' @export
nl_profile <- function(fitted, which, values = NULL, ...) {
  j <- if (is.character(which)) match(which, fitted$pnames) else which
  if (is.na(j)) stop("unknown parameter", call. = FALSE)
  est <- fitted$coefficients[j]
  se <- sqrt(fitted$cov[j, j])
  if (is.null(values)) values <- est + seq(-4, 4, length.out = 41) * se
  start <- fitted$coefficients[-j]
  sse <- vapply(values, function(v) .fit_fixed(fitted, j, v, start)$sse, 0)
  data.frame(value = values, sse = sse,
             tau = sign(values - est) * sqrt(pmax(sse - fitted$sse, 0)) / fitted$sigma)
}

.profile_ci <- function(object, j, level) {
  q <- stats::qt(0.5 + level / 2, object$df.residual)
  est <- object$coefficients[[j]]
  se <- sqrt(object$cov[j, j])
  start <- object$coefficients[-j]
  tau <- function(v) {
    s <- .fit_fixed(object, j, v, start)$sse
    sign(v - est) * sqrt(max(s - object$sse, 0)) / object$sigma
  }
  vapply(c(-1, 1), function(direction) {
    step <- if (is.finite(se) && se > 0) se else max(abs(est) * 0.1, 0.1)
    inner <- est
    outer <- est + direction * step
    for (k in 1:60) {
      val <- tryCatch(abs(tau(outer)), error = function(e) NA_real_)
      if (!is.finite(val)) return(NA_real_)
      if (val >= q) break
      inner <- outer
      outer <- est + (outer - est) * 1.6
      if (k == 60) return(NA_real_)
    }
    stats::uniroot(function(v) abs(tau(v)) - q, sort(c(inner, outer)),
                   tol = 1e-10 * (abs(est) + 1))$root
  }, 0)
}

#' Residual bootstrap of a nonlinear fit
#'
#' Resamples centred residuals (inflated by `sqrt(n / (n - p))`), adds them to
#' the fitted values, and refits from the original estimates.
#'
#' @param object An `"nlestim"` fit.
#' @param R Number of bootstrap replicates.
#' @param seed Random seed for reproducibility (`NULL` to leave the RNG alone).
#' @return A matrix with one row per successful replicate and one column per
#'   parameter.
#' @examples
#' fit <- nl_fit(rate ~ conc, subset(Puromycin, state == "treated"),
#'               model = "michaelis_menten")
#' boot <- nl_bootstrap(fit, R = 200)
#' apply(boot, 2, quantile, c(0.025, 0.975))
#' @export
nl_bootstrap <- function(object, R = 1000, seed = 1) {
  if (!is.null(seed)) {
    old <- if (exists(".Random.seed", envir = globalenv())) get(".Random.seed", envir = globalenv())
    on.exit(if (!is.null(old)) assign(".Random.seed", old, envir = globalenv()), add = TRUE)
    set.seed(seed)
  }
  n <- length(object$y)
  df <- object$df.residual
  sw <- sqrt(object$weights)
  res <- object$residuals * sw
  res <- (res - mean(res)) * sqrt(n / df)
  jac <- function(par) sw * object$jac_fun(par)
  out <- matrix(NA_real_, R, length(object$pnames), dimnames = list(NULL, object$pnames))
  for (b in seq_len(R)) {
    y_star <- object$fitted.values + sample(res, n, replace = TRUE) / ifelse(sw > 0, sw, 1)
    fit <- nl_levenberg_marquardt(function(par) sw * (object$predict_fun(par) - y_star), jac,
                                  object$coefficients, max_iter = 200)
    if (fit$converged) out[b, ] <- fit$par
  }
  out[stats::complete.cases(out), , drop = FALSE]
}

#' Confidence intervals for nonlinear fits
#'
#' * `"wald"`: estimate +/- t * SE. Fast and symmetric; accurate only when the
#'   model is close to linear in the parameter.
#' * `"profile"`: inverts the profile t statistic (see [nl_profile()]), the
#'   method used by R's `confint()` for `nls` fits. Respects curvature and can
#'   be asymmetric.
#' * `"bootstrap"`: percentile intervals from [nl_bootstrap()].
#'
#' @param object An `"nlestim"` fit.
#' @param parm Parameters (names or indices). Defaults to all.
#' @param level Confidence level.
#' @param method `"wald"`, `"profile"` or `"bootstrap"`.
#' @param R Bootstrap replicates (for `method = "bootstrap"`).
#' @param ... Unused.
#' @return A matrix with one row per parameter.
#' @examples
#' fit <- nl_fit(rate ~ conc, subset(Puromycin, state == "treated"),
#'               model = "michaelis_menten")
#' confint(fit)
#' confint(fit, method = "profile")
#' @export
confint.nlestim <- function(object, parm = NULL, level = 0.95,
                            method = c("wald", "profile", "bootstrap"), R = 1000, ...) {
  method <- match.arg(method)
  pn <- object$pnames
  idx <- if (is.null(parm)) seq_along(pn) else if (is.character(parm)) match(parm, pn) else parm
  a <- (1 - level) / 2
  ci <- switch(method,
    wald = {
      se <- sqrt(diag(object$cov))[idx]
      q <- stats::qt(1 - a, object$df.residual)
      cbind(object$coefficients[idx] - q * se, object$coefficients[idx] + q * se)
    },
    profile = t(vapply(idx, function(j) .profile_ci(object, j, level), c(0, 0))),
    bootstrap = {
      boot <- nl_bootstrap(object, R = R)
      t(apply(boot[, idx, drop = FALSE], 2, stats::quantile, c(a, 1 - a)))
    }
  )
  dimnames(ci) <- list(pn[idx], sprintf("%.1f %%", 100 * c(a, 1 - a)))
  ci
}

#' Compare nested nonlinear models (extra sum of squares F test)
#'
#' @param object The reduced model (fewer parameters).
#' @param ... The full model.
#' @return An object of class `"anova"`.
#' @examples
#' d <- crop_yield_nitrogen
#' full <- nl_fit(yield ~ nitrogen, d, model = "asymptotic_regression")
#' reduced <- nl_fit(yield ~ a * (1 - exp(-c * nitrogen)), d, start = c(a = 10, c = 0.01))
#' anova(reduced, full)
#' @export
anova.nlestim <- function(object, ...) {
  full <- list(...)[[1L]]
  if (!inherits(full, "nlestim")) stop("supply two nlestim fits: reduced, full", call. = FALSE)
  df_diff <- object$df.residual - full$df.residual
  if (df_diff <= 0) stop("the first model must have fewer parameters", call. = FALSE)
  F <- ((object$sse - full$sse) / df_diff) / (full$sse / full$df.residual)
  p <- stats::pf(F, df_diff, full$df.residual, lower.tail = FALSE)
  tab <- data.frame(`Res.Df` = c(object$df.residual, full$df.residual),
                    `Res.Sum Sq` = c(object$sse, full$sse), Df = c(NA, df_diff),
                    `Sum Sq` = c(NA, object$sse - full$sse), `F value` = c(NA, F),
                    `Pr(>F)` = c(NA, p), check.names = FALSE)
  rownames(tab) <- c("reduced", "full")
  structure(tab, heading = c("Analysis of Variance Table (extra sum of squares)\n",
                             paste("Model 1:", .model_label(object)),
                             paste("Model 2:", .model_label(full))),
            class = c("anova", "data.frame"))
}

#' Diagnostic plots for a nonlinear fit
#'
#' @param x An `"nlestim"` fit.
#' @param which Which plots: 1 = data, fitted curve and bands; 2 = residuals
#'   vs fitted; 3 = normal Q-Q plot; 4 = convergence (SSE by iteration).
#' @param level Level for the bands in plot 1.
#' @param ask Ask before each new page (as in [plot.lm()]).
#' @param ... Passed to the first plot.
#' @return `x`, invisibly.
#' @examples
#' fit <- nl_fit(rate ~ conc, subset(Puromycin, state == "treated"),
#'               model = "michaelis_menten")
#' op <- par(mfrow = c(2, 2))
#' plot(fit)
#' par(op)
#' @export
plot.nlestim <- function(x, which = 1:4, level = 0.95,
                         ask = length(which) > 1 && prod(graphics::par("mfrow")) < length(which) &&
                           grDevices::dev.interactive(), ...) {
  if (ask) {
    old <- grDevices::devAskNewPage(TRUE)
    on.exit(grDevices::devAskNewPage(old))
  }
  xvar <- setdiff(all.vars(x$rhs), x$pnames)
  xvar <- xvar[vapply(xvar, function(v) !is.null(x$frame[[v]]), TRUE)]
  std_res <- x$residuals * sqrt(x$weights) / x$sigma
  if (1 %in% which && length(xvar) == 1L) {
    xv <- x$frame[[xvar]]
    grid <- seq(min(xv), max(xv), length.out = 200)
    nd <- stats::setNames(list(grid), xvar)
    band_c <- predict(x, nd, "confidence", level)
    band_p <- predict(x, nd, "prediction", level)
    graphics::plot(xv, x$y, pch = 19, xlab = xvar, ylab = deparse(x$formula[[2L]]),
                   ylim = range(x$y, band_p[, 2:3], finite = TRUE),
                   main = "Fitted curve with bands", ...)
    graphics::polygon(c(grid, rev(grid)), c(band_p[, 2], rev(band_p[, 3])),
                      col = grDevices::adjustcolor("#0e7490", 0.12), border = NA)
    graphics::polygon(c(grid, rev(grid)), c(band_c[, 2], rev(band_c[, 3])),
                      col = grDevices::adjustcolor("#0e7490", 0.25), border = NA)
    graphics::lines(grid, band_c[, 1], col = "#0e7490", lwd = 2)
    graphics::points(xv, x$y, pch = 19)
    graphics::legend("topleft", bty = "n", cex = 0.8,
                     fill = c(grDevices::adjustcolor("#0e7490", 0.25), grDevices::adjustcolor("#0e7490", 0.12)), border = NA,
                     legend = sprintf("%d%% %s band", round(100 * level), c("confidence", "prediction")))
  }
  if (2 %in% which) {
    graphics::plot(x$fitted.values, std_res, pch = 19, xlab = "fitted values",
                   ylab = "standardised residual", main = "Residuals vs fitted")
    graphics::abline(h = 0)
    graphics::abline(h = c(-2, 2), lty = 3)
  }
  if (3 %in% which) {
    stats::qqnorm(std_res, pch = 19, main = "Normal Q-Q")
    stats::qqline(std_res)
  }
  if (4 %in% which) {
    h <- x$history[x$history$accepted & is.finite(x$history$sse), ]
    graphics::plot(h$iteration, h$sse, type = "b", pch = 19, log = "y", xlab = "iteration",
                   ylab = "SSE (log scale)", main = paste("Convergence:", x$method))
  }
  invisible(x)
}
