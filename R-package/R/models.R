# Library of common nonlinear models. Each entry has a right-hand side written
# in terms of the predictor `x`, the parameter names, and a function that
# guesses starting values from the data.

.median_pos <- function(v) {
  v <- v[is.finite(v)]
  if (length(v)) stats::median(v) else NA_real_
}

.half_x <- function(x, y) {
  o <- order(x)
  xs <- x[o]
  ys <- y[o]
  half <- min(ys) + 0.5 * (max(ys) - min(ys))
  idx <- if (ys[length(ys)] >= ys[1]) which(ys >= half)[1] else which(ys <= half)[1]
  xs[idx]
}

.slope <- function(x, y) unname(stats::coef(stats::lm(y ~ x)))

.models <- list(
  exponential_decay = list(
    rhs = quote(a * exp(-k * x)), params = c("a", "k"),
    label = "a * exp(-k * x)",
    use = "First-order decay: drug elimination, radioactive decay, cooling.",
    start = function(x, y) {
      ok <- y > 0
      if (sum(ok) >= 2) {
        b <- .slope(x[ok], log(y[ok]))
        return(c(a = exp(b[1]), k = max(-b[2], 1e-6)))
      }
      c(a = max(y), k = 1 / diff(range(x)))
    }
  ),
  exponential_rise = list(
    rhs = quote(a * (1 - exp(-k * x))), params = c("a", "k"),
    label = "a * (1 - exp(-k * x))",
    use = "Monomolecular approach to an asymptote through the origin.",
    start = function(x, y) {
      a <- 1.05 * max(y)
      frac <- pmin(pmax(y / a, 1e-6), 1 - 1e-6)
      k <- .median_pos((-log(1 - frac) / x)[x > 0])
      c(a = a, k = if (is.na(k)) 1 else k)
    }
  ),
  asymptotic_regression = list(
    rhs = quote(a - b * exp(-c * x)), params = c("a", "b", "c"),
    label = "a - b * exp(-c * x)",
    use = "Mitscherlich law of diminishing returns (yield response to fertiliser).",
    start = function(x, y) {
      a <- max(y) + 0.05 * diff(range(y))
      b <- a - y[which.min(x)]
      frac <- pmax((a - y) / b, 1e-6)
      cc <- .median_pos((-log(frac) / x)[x > 0])
      c(a = a, b = b, c = if (is.na(cc)) 1 / diff(range(x)) else max(cc, 1e-6))
    }
  ),
  michaelis_menten = list(
    rhs = quote(Vmax * x / (Km + x)), params = c("Vmax", "Km"),
    label = "Vmax * x / (Km + x)",
    use = "Enzyme kinetics; also Monod growth and Langmuir adsorption.",
    start = function(x, y) {
      ok <- x > 0 & y > 0
      if (sum(ok) >= 2) {
        b <- .slope(1 / x[ok], 1 / y[ok])  # Lineweaver-Burk
        if (b[1] > 0 && b[2] > 0) return(c(Vmax = 1 / b[1], Km = b[2] / b[1]))
      }
      c(Vmax = max(y), Km = .half_x(x, y))
    }
  ),
  logistic = list(
    rhs = quote(Asym / (1 + exp((xmid - x) / scal))), params = c("Asym", "xmid", "scal"),
    label = "Asym / (1 + exp((xmid - x) / scal))",
    use = "Symmetric S-shaped growth (same parameterisation as stats::SSlogis).",
    start = function(x, y) {
      c(Asym = 1.05 * max(y), xmid = .half_x(x, y), scal = diff(range(x)) / 8)
    }
  ),
  gompertz = list(
    rhs = quote(Asym * exp(-b2 * b3^x)), params = c("Asym", "b2", "b3"),
    label = "Asym * exp(-b2 * b3^x)",
    use = "Asymmetric S-shaped growth (same parameterisation as stats::SSgompertz).",
    start = function(x, y) {
      asym <- 1.05 * max(y)
      frac <- pmin(pmax(y / asym, 1e-6), 1 - 1e-6)
      b <- .slope(x, log(-log(frac)))
      c(Asym = asym, b2 = exp(b[1]), b3 = exp(b[2]))
    }
  ),
  richards = list(
    rhs = quote(b1 / (1 + exp(b2 - b3 * x))^(1 / b4)), params = c("b1", "b2", "b3", "b4"),
    label = "b1 / (1 + exp(b2 - b3 * x))^(1 / b4)",
    use = "Generalised logistic growth with a shape parameter (NIST Rat43).",
    start = function(x, y) {
      b1 <- 1.05 * max(y)
      frac <- pmin(pmax(y / b1, 1e-6), 1 - 1e-6)
      b <- .slope(x, log(1 / frac - 1))
      c(b1 = b1, b2 = b[1], b3 = -b[2], b4 = 1)
    }
  ),
  four_pl = list(
    rhs = quote(bottom + (top - bottom) / (1 + (x / EC50)^hill)),
    params = c("bottom", "top", "EC50", "hill"),
    label = "bottom + (top - bottom) / (1 + (x / EC50)^hill)",
    use = "Four-parameter logistic dose-response curve (x must be > 0).",
    start = function(x, y) {
      o <- order(x)
      top <- y[o[1]]
      bottom <- y[o[length(o)]]
      mid <- (top + bottom) / 2
      pos <- which(x > 0)
      near <- pos[order(abs(y[pos] - mid))][seq_len(min(3, length(pos)))]
      c(bottom = bottom, top = top, EC50 = exp(stats::median(log(x[near]))), hill = 1)
    }
  ),
  power_law = list(
    rhs = quote(a * x^b), params = c("a", "b"),
    label = "a * x^b",
    use = "Allometric scaling and other power laws (x must be > 0).",
    start = function(x, y) {
      ok <- x > 0 & y > 0
      b <- .slope(log(x[ok]), log(y[ok]))
      c(a = exp(b[1]), b = b[2])
    }
  ),
  gaussian_peak = list(
    rhs = quote(amp * exp(-0.5 * ((x - mu) / sigma)^2)), params = c("amp", "mu", "sigma"),
    label = "amp * exp(-0.5 * ((x - mu) / sigma)^2)",
    use = "A single symmetric peak: spectroscopy, chromatography, optimum curves.",
    start = function(x, y) {
      i <- which.max(y)
      w <- pmax(y, 0)
      s <- sqrt(sum(w * (x - x[i])^2) / max(sum(w), 1e-12))
      c(amp = y[i], mu = x[i], sigma = if (s > 0) s else diff(range(x)) / 6)
    }
  )
)

#' Built-in nonlinear models
#'
#' `nl_models()` lists the model library. Pass a model's name to [nl_fit()] or
#' [nl_workflow()] and starting values are found automatically.
#'
#' @return A data frame with the model name, formula, parameters and typical use.
#' @examples
#' nl_models()
#' @export
nl_models <- function() {
  data.frame(
    model = names(.models),
    formula = vapply(.models, function(m) paste("y ~", m$label), ""),
    parameters = vapply(.models, function(m) paste(m$params, collapse = ", "), ""),
    use = vapply(.models, function(m) m$use, ""),
    row.names = NULL, stringsAsFactors = FALSE
  )
}

.get_model <- function(name) {
  m <- .models[[name]]
  if (is.null(m)) {
    stop(sprintf("unknown model '%s'. Available: %s", name,
                 paste(names(.models), collapse = ", ")), call. = FALSE)
  }
  m
}

#' Suggest candidate models from the shape of the data
#'
#' Looks at whether `y` rises, falls or peaks along `x` and whether the values
#' are positive, and returns the built-in models that suit that shape. Used by
#' [nl_workflow()] when no models are given.
#'
#' @param x,y Numeric vectors.
#' @return A character vector of model names.
#' @examples
#' nl_suggest_models(Puromycin$conc, Puromycin$rate)
#' @export
nl_suggest_models <- function(x, y) {
  ok <- is.finite(x) & is.finite(y)
  x <- x[ok]
  y <- y[ok]
  o <- order(x)
  xs <- x[o]
  ys <- y[o]
  n <- length(unique(xs))
  rho <- suppressWarnings(stats::cor(x, y, method = "spearman"))
  imax <- which.max(ys)
  rng <- diff(range(ys))
  peaked <- imax > 1 && imax < length(ys) &&
    (ys[imax] - ys[1]) > 0.3 * rng && (ys[imax] - ys[length(ys)]) > 0.3 * rng
  positive_x <- all(x > 0)
  positive_y <- all(y > 0)
  out <- if (peaked) {
    "gaussian_peak"
  } else if (is.finite(rho) && rho < 0) {
    c("exponential_decay", "asymptotic_regression",
      if (positive_x) c("four_pl", if (positive_y) "power_law"))
  } else {
    c("michaelis_menten", "exponential_rise", "asymptotic_regression", "logistic",
      "gompertz", if (n >= 8) "richards", if (positive_x && positive_y) "power_law")
  }
  if (!all(x >= 0)) out <- setdiff(out, c("michaelis_menten", "exponential_rise"))
  out[vapply(out, function(m) length(.models[[m]]$params) < n, TRUE)]
}
