#' Compare fitted models by AIC and BIC
#'
#' @param ... `"nlestim"` fits to the same data (optionally named).
#' @return A data frame sorted by AICc (AIC with the small-sample correction
#'   `2k(k + 1) / (n - k - 1)`, recommended when n / k < 40), with Akaike
#'   weights (the relative support for each model), AIC and BIC.
#' @examples
#' d <- plant_growth
#' nl_compare(logistic = nl_fit(biomass ~ day, d, model = "logistic"),
#'            gompertz = nl_fit(biomass ~ day, d, model = "gompertz"))
#' @export
nl_compare <- function(...) {
  fits <- list(...)
  nm <- names(fits)
  if (is.null(nm)) nm <- rep("", length(fits))
  nm[nm == ""] <- vapply(fits[nm == ""], function(f) if (is.null(f$model)) .model_label(f) else f$model, "")
  .comparison_table(stats::setNames(fits, nm))
}

.comparison_table <- function(fits) {
  rows <- lapply(names(fits), function(nm) {
    f <- fits[[nm]]
    y <- f$y
    w <- f$weights
    sst <- sum(w * (y - stats::weighted.mean(y, w))^2)
    k <- length(f$coefficients) + 1L
    n <- length(y)
    aic <- stats::AIC(f)
    data.frame(model = nm, parameters = length(f$coefficients), converged = f$converged,
               SSE = f$sse, sigma = f$sigma, R2 = 1 - f$sse / sst,
               AICc = if (n - k - 1 > 0) aic + 2 * k * (k + 1) / (n - k - 1) else Inf,
               AIC = aic, BIC = stats::BIC(f), stringsAsFactors = FALSE)
  })
  tab <- do.call(rbind, rows)
  ok <- tab$converged
  tab$delta_AICc <- NA_real_
  tab$weight <- NA_real_
  if (any(ok)) {
    tab$delta_AICc[ok] <- tab$AICc[ok] - min(tab$AICc[ok])
    wts <- exp(-0.5 * tab$delta_AICc[ok])
    tab$weight[ok] <- wts / sum(wts)
  }
  tab <- tab[order(!tab$converged, tab$AICc), c("model", "parameters", "converged", "SSE", "sigma",
                                               "R2", "AICc", "delta_AICc", "weight", "AIC", "BIC")]
  rownames(tab) <- NULL
  tab
}

.param_table <- function(fit, level, ci_method) {
  s <- summary(fit)$coefficients
  ci <- tryCatch(suppressWarnings(stats::confint(fit, level = level, method = ci_method)),
                 error = function(e) NULL)
  if (is.null(ci) || anyNA(ci)) {
    wald <- stats::confint(fit, level = level, method = "wald")
    if (is.null(ci)) ci <- wald else ci[is.na(ci)] <- wald[is.na(ci)]
  }
  data.frame(parameter = rownames(s), estimate = s[, 1], std_error = s[, 2], t_value = s[, 3],
             p_value = s[, 4], lower = ci[, 1], upper = ci[, 2], row.names = NULL)
}

#' Complete nonlinear regression workflow for any dataset
#'
#' One call that takes a data frame and two column names, then:
#'
#' 1. chooses candidate models from the shape of the data (or uses `models`);
#' 2. fits each one, recording any that fail to converge;
#' 3. ranks them by AICc (small-sample corrected AIC), with Akaike weights,
#'    AIC and BIC;
#' 4. reports the best model's estimates with standard errors and confidence
#'    intervals (profile likelihood by default);
#' 5. checks residuals and flags common problems;
#' 6. optionally writes CSV tables, a PNG figure and a text report to
#'    `output_dir`.
#'
#' @param data A data frame.
#' @param x,y Names of the predictor and response columns.
#' @param models Built-in model names (see [nl_models()]). `NULL` picks them
#'   with [nl_suggest_models()].
#' @param method Solver: `"lm"`, `"gn"` or `"gn-pure"`.
#' @param weights Optional weights (a numeric vector, or a column name).
#' @param level Confidence level.
#' @param ci_method `"profile"`, `"wald"` or `"bootstrap"`.
#' @param output_dir Directory to write results to, or `NULL` to write nothing.
#' @param plot Logical; draw the summary figure.
#' @return An object of class `"nl_workflow"` with elements `comparison`,
#'   `best` (the best `"nlestim"` fit), `fits` (all fits), `parameters`
#'   (table for the best model), `notes` and `files`.
#' @examples
#' wf <- nl_workflow(plant_growth, x = "day", y = "biomass", plot = FALSE)
#' wf
#' wf$comparison
#' coef(wf$best)
#'
#' # Write tables, a figure and a report
#' out <- file.path(tempdir(), "growth-analysis")
#' wf <- nl_workflow(crop_yield_nitrogen, "nitrogen", "yield", output_dir = out, plot = FALSE)
#' list.files(out)
#' @export
nl_workflow <- function(data, x, y, models = NULL, method = c("lm", "gn", "gn-pure"),
                        weights = NULL, level = 0.95,
                        ci_method = c("profile", "wald", "bootstrap"),
                        output_dir = NULL, plot = interactive()) {
  method <- match.arg(method)
  ci_method <- match.arg(ci_method)
  data <- as.data.frame(data)
  for (v in c(x, y)) {
    if (is.null(data[[v]])) stop(sprintf("column '%s' not found in data", v), call. = FALSE)
    if (!is.numeric(data[[v]])) stop(sprintf("column '%s' must be numeric", v), call. = FALSE)
  }
  if (is.character(weights) && length(weights) == 1L) weights <- data[[weights]]
  ok <- stats::complete.cases(data[c(x, y)])
  data <- data[ok, , drop = FALSE]
  if (!is.null(weights)) weights <- weights[ok]
  if (is.null(models)) models <- nl_suggest_models(data[[x]], data[[y]])
  if (!length(models)) stop("no suitable built-in models for these data", call. = FALSE)
  fml <- stats::as.formula(paste0("`", y, "` ~ `", x, "`"))

  fits <- list()
  failed <- character()
  for (m in models) {
    f <- tryCatch(suppressWarnings(nl_fit(fml, data, model = m, method = method, weights = weights)),
                  error = function(e) e)
    if (inherits(f, "error")) {
      failed[m] <- conditionMessage(f)
    } else {
      fits[[m]] <- f
    }
  }
  if (!length(fits)) stop("no model could be fitted:\n",
                          paste0("  ", names(failed), ": ", failed, collapse = "\n"), call. = FALSE)
  comparison <- .comparison_table(fits)
  if (!any(comparison$converged)) {
    warning("no model converged; results are unreliable", call. = FALSE)
  }
  best_name <- comparison$model[1]
  best <- fits[[best_name]]
  parameters <- .param_table(best, level, ci_method)

  notes <- character()
  if (nrow(data) < 10) notes <- c(notes, sprintf(
    "Only %d observations: model ranking is uncertain; let subject knowledge guide the choice.", nrow(data)))
  close <- comparison$model[comparison$converged & comparison$delta_AICc < 2 & comparison$model != best_name]
  if (length(close)) {
    notes <- c(notes, sprintf(
      "Models within 2 AICc units of '%s' (%s) are about equally supported; prefer the simpler or more interpretable one.",
      best_name, paste(close, collapse = ", ")))
  }
  nc <- comparison$model[!comparison$converged]
  if (length(nc)) notes <- c(notes, sprintf("Did not converge (excluded from ranking): %s.",
                                            paste(nc, collapse = ", ")))
  if (length(failed)) notes <- c(notes, sprintf("Could not be fitted: %s.",
                                                paste(names(failed), collapse = ", ")))
  corr <- best$cov / outer(sqrt(diag(best$cov)), sqrt(diag(best$cov)))
  off <- abs(corr[upper.tri(corr)])
  if (length(off) && any(off > 0.95, na.rm = TRUE)) {
    notes <- c(notes, sprintf("Estimates of '%s' are highly correlated (max |r| = %.3f): some parameters are poorly identified.",
                              best_name, max(off, na.rm = TRUE)))
  }
  std_res <- best$residuals * sqrt(best$weights) / best$sigma
  if (length(std_res) >= 3 && length(std_res) <= 5000) {
    sw <- stats::shapiro.test(std_res)$p.value
    if (sw < 0.05) notes <- c(notes, sprintf(
      "Residuals look non-normal (Shapiro-Wilk p = %.3f): check the residual plots; consider weights or a transformation.", sw))
  }
  if (any(abs(std_res) > 3)) notes <- c(notes, sprintf(
    "%d observation(s) have |standardised residual| > 3: possible outliers.", sum(abs(std_res) > 3)))

  out <- structure(list(comparison = comparison, best = best, best_model = best_name, fits = fits,
                        parameters = parameters, notes = notes, failed = failed, x = x, y = y,
                        data = data, level = level, ci_method = ci_method, files = character()),
                   class = "nl_workflow")
  if (plot) plot(out)
  if (!is.null(output_dir)) out$files <- .write_workflow(out, output_dir)
  out
}

.write_workflow <- function(wf, dir) {
  dir.create(dir, showWarnings = FALSE, recursive = TRUE)
  files <- c(comparison = file.path(dir, "model_comparison.csv"),
             parameters = file.path(dir, "best_model_parameters.csv"),
             all_parameters = file.path(dir, "all_model_parameters.csv"),
             fitted = file.path(dir, "fitted_values.csv"),
             predictions = file.path(dir, "prediction_curve.csv"),
             figure = file.path(dir, "workflow_plots.png"),
             report = file.path(dir, "report.txt"))
  utils::write.csv(wf$comparison, files[["comparison"]], row.names = FALSE)
  utils::write.csv(wf$parameters, files[["parameters"]], row.names = FALSE)
  allp <- do.call(rbind, lapply(names(wf$fits), function(m) {
    f <- wf$fits[[m]]
    s <- summary(f)$coefficients
    data.frame(model = m, parameter = rownames(s), estimate = s[, 1], std_error = s[, 2],
               converged = f$converged, row.names = NULL)
  }))
  utils::write.csv(allp, files[["all_parameters"]], row.names = FALSE)
  b <- wf$best
  utils::write.csv(data.frame(x = wf$data[[wf$x]], y = wf$data[[wf$y]], fitted = b$fitted.values,
                              residual = b$residuals,
                              standardised_residual = b$residuals * sqrt(b$weights) / b$sigma),
                   files[["fitted"]], row.names = FALSE)
  xv <- wf$data[[wf$x]]
  grid <- seq(min(xv), max(xv), length.out = 101)
  nd <- stats::setNames(list(grid), wf$x)
  pc <- predict(b, nd, "confidence", wf$level)
  pp <- predict(b, nd, "prediction", wf$level)
  utils::write.csv(data.frame(x = grid, fit = pc[, 1], conf_lower = pc[, 2], conf_upper = pc[, 3],
                              pred_lower = pp[, 2], pred_upper = pp[, 3]),
                   files[["predictions"]], row.names = FALSE)
  grDevices::png(files[["figure"]], width = 1600, height = 1200, res = 150)
  tryCatch(plot(wf), finally = grDevices::dev.off())
  writeLines(utils::capture.output(print(wf), print(summary(b))), files[["report"]])
  files
}

#' @rdname nl_workflow
#' @param ... Unused.
#' @export
print.nl_workflow <- function(x, ...) {
  cat(sprintf("Nonlinear regression workflow: %s ~ %s (%d observations)\n\n", x$y, x$x, nrow(x$data)))
  cat("Model comparison (best first):\n")
  tab <- x$comparison
  num <- vapply(tab, is.numeric, TRUE)
  tab[num] <- lapply(tab[num], function(v) signif(v, 4))
  print(tab, row.names = FALSE)
  cat(sprintf("\nBest model: %s   (%s)\n", x$best_model, .model_label(x$best)))
  cat(sprintf("%d%% confidence intervals: %s\n\n", round(100 * x$level), x$ci_method))
  p <- x$parameters
  p[-1] <- lapply(p[-1], signif, 5)
  print(p, row.names = FALSE)
  if (length(x$notes)) cat("\nNotes:\n", paste0("  - ", x$notes, "\n"), sep = "")
  if (length(x$files)) cat("\nFiles written:\n", paste0("  ", x$files, "\n"), sep = "")
  invisible(x)
}

#' @rdname nl_workflow
#' @export
plot.nl_workflow <- function(x, ...) {
  op <- graphics::par(mfrow = c(2, 2), mar = c(4.2, 4.2, 2.5, 1))
  on.exit(graphics::par(op))
  xv <- x$data[[x$x]]
  yv <- x$data[[x$y]]
  grid <- seq(min(xv), max(xv), length.out = 200)
  nd <- stats::setNames(list(grid), x$x)
  cols <- grDevices::hcl.colors(max(3, length(x$fits)), "Dark 3")
  graphics::plot(xv, yv, pch = 19, xlab = x$x, ylab = x$y, main = "Candidate models")
  for (i in seq_along(x$fits)) {
    f <- x$fits[[i]]
    graphics::lines(grid, predict(f, nd), col = cols[i], lwd = if (names(x$fits)[i] == x$best_model) 3 else 1.5,
                    lty = if (f$converged) 1 else 3)
  }
  graphics::legend("topleft", legend = names(x$fits), col = cols[seq_along(x$fits)], lwd = 2,
                   bty = "n", cex = 0.75)
  plot(x$best, which = 1, level = x$level, ask = FALSE)
  graphics::title(sub = paste("best:", x$best_model), cex.sub = 0.8)
  plot(x$best, which = 2:3, ask = FALSE)
  invisible(x)
}

#' Compare Gauss-Newton and Levenberg-Marquardt on one problem
#'
#' Fits the same model from the same start with pure Gauss-Newton,
#' Gauss-Newton with step halving and Levenberg-Marquardt. For two-parameter
#' models it can draw the sum-of-squares landscape with each solver's path.
#'
#' @inheritParams nl_fit
#' @param plot Logical; draw the landscape (two-parameter models) and the
#'   convergence curves.
#' @param xlim,ylim Ranges of the two parameters for the landscape. Chosen
#'   automatically when `NULL`.
#' @return Invisibly, a list with `table` (one row per method) and `fits`.
#' @examples
#' nist <- boxbod
#' cmp <- nl_compare_solvers(bod ~ days, nist, model = "exponential_rise",
#'                           start = c(a = 1, k = 1), plot = FALSE)
#' cmp$table
#' @export
nl_compare_solvers <- function(formula, data = NULL, start = NULL, model = NULL, plot = interactive(),
                               xlim = NULL, ylim = NULL) {
  methods <- c("gn-pure", "gn", "lm")
  fits <- lapply(stats::setNames(methods, methods), function(m) {
    tryCatch(suppressWarnings(nl_fit(formula, data, start = start, model = model, method = m)),
             error = function(e) e)
  })
  table <- do.call(rbind, lapply(methods, function(m) {
    f <- fits[[m]]
    if (inherits(f, "error")) {
      return(data.frame(method = m, converged = FALSE, iterations = NA, SSE = NA,
                        message = conditionMessage(f), stringsAsFactors = FALSE))
    }
    data.frame(method = m, converged = f$converged, iterations = f$iterations, SSE = f$sse,
               t(signif(f$coefficients, 7)), message = f$message, check.names = FALSE,
               stringsAsFactors = FALSE)
  }))
  ok <- fits[!vapply(fits, inherits, TRUE, "error")]
  if (plot && length(ok)) {
    cols <- c(`gn-pure` = "#c92a2a", gn = "#f59f00", lm = "#1c64d1")
    two <- length(ok[[1]]$pnames) == 2L
    op <- graphics::par(mfrow = c(1, if (two) 2 else 1))
    on.exit(graphics::par(op))
    if (two) {
      ref <- ok[[length(ok)]]
      paths <- lapply(ok, function(f) as.matrix(f$history[f$history$accepted, f$pnames]))
      allp <- do.call(rbind, paths)
      allp <- allp[apply(is.finite(allp), 1, all), , drop = FALSE]
      if (is.null(xlim)) xlim <- range(allp[, 1]) + c(-0.1, 0.1) * diff(range(allp[, 1]))
      if (is.null(ylim)) ylim <- range(allp[, 2]) + c(-0.1, 0.1) * diff(range(allp[, 2]))
      g1 <- seq(xlim[1], xlim[2], length.out = 120)
      g2 <- seq(ylim[1], ylim[2], length.out = 120)
      S <- outer(g1, g2, Vectorize(function(a, b) {
        v <- suppressWarnings(ref$predict_fun(c(a, b)))
        s <- sum(ref$weights * (v - ref$y)^2)
        if (is.finite(s)) log10(s) else NA_real_
      }))
      graphics::image(g1, g2, S, col = grDevices::hcl.colors(60, "Teal", rev = FALSE),
                      xlab = ref$pnames[1], ylab = ref$pnames[2], main = "log10 SSE and solver paths")
      graphics::contour(g1, g2, S, add = TRUE, drawlabels = FALSE, col = "grey40", nlevels = 15)
      for (m in names(paths)) {
        graphics::lines(paths[[m]], type = "o", pch = 19, cex = 0.7, lwd = 2, col = cols[[m]])
      }
      graphics::legend("topright", legend = names(paths), col = cols[names(paths)], lwd = 2,
                       bg = "white", cex = 0.8)
    }
    hs <- lapply(ok, function(f) f$history[f$history$accepted & is.finite(f$history$sse), ])
    graphics::plot(NULL, xlim = c(0, max(vapply(hs, function(h) max(h$iteration), 0))),
                   ylim = range(unlist(lapply(hs, `[[`, "sse"))), log = "y",
                   xlab = "iteration", ylab = "SSE (log scale)", main = "Convergence")
    for (m in names(hs)) graphics::lines(hs[[m]]$iteration, hs[[m]]$sse, type = "o", pch = 19,
                                         cex = 0.6, col = cols[[m]], lwd = 2)
    graphics::legend("topright", legend = names(hs), col = cols[names(hs)], lwd = 2, bty = "n", cex = 0.8)
  }
  invisible(list(table = table, fits = fits))
}

#' Read a dataset from CSV or Excel
#'
#' @param path Path to a `.csv`, `.xlsx` or `.xls` file.
#' @param sheet Sheet name or number for Excel files (needs the readxl package).
#' @return A data frame.
#' @examples
#' path <- system.file("extdata", "nonlinear_datasets.xlsx", package = "nlestim")
#' if (requireNamespace("readxl", quietly = TRUE)) {
#'   d <- nl_read(path, sheet = "plant_growth")
#'   head(d)
#' }
#' head(nl_read(system.file("extdata", "puromycin.csv", package = "nlestim")))
#' @export
nl_read <- function(path, sheet = 1) {
  ext <- tolower(tools::file_ext(path))
  if (ext %in% c("xlsx", "xls")) {
    if (!requireNamespace("readxl", quietly = TRUE)) {
      stop("reading Excel files needs the 'readxl' package: install.packages('readxl')", call. = FALSE)
    }
    return(as.data.frame(readxl::read_excel(path, sheet = sheet)))
  }
  utils::read.csv(path, check.names = FALSE)
}
