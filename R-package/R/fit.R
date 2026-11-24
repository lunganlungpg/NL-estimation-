#' Control settings for the solvers
#'
#' @param max_iter Maximum iterations (default 200 for Gauss-Newton, 500 for
#'   Levenberg-Marquardt).
#' @param rtol Relative-offset tolerance for convergence.
#' @param xtol Component-wise relative step tolerance.
#' @param ftol Relative SSE-reduction tolerance.
#' @param lambda0,tau,scaling,update Levenberg-Marquardt settings; see
#'   [nl_levenberg_marquardt()].
#' @return A list of settings for [nl_fit()].
#' @examples
#' nl_control(max_iter = 1000, update = "classic")
#' @export
nl_control <- function(max_iter = NULL, rtol = 1e-10, xtol = 1e-14, ftol = 1e-15,
                       lambda0 = NULL, tau = 1e-3, scaling = c("marquardt", "levenberg"),
                       update = c("nielsen", "classic")) {
  list(max_iter = max_iter, rtol = rtol, xtol = xtol, ftol = ftol, lambda0 = lambda0,
       tau = tau, scaling = match.arg(scaling), update = match.arg(update))
}

#' Fit a nonlinear regression model
#'
#' Fits a nonlinear model by least squares with Levenberg-Marquardt
#' (`method = "lm"`, the robust default), Gauss-Newton with step halving
#' (`"gn"`, the algorithm of [stats::nls()]) or textbook Gauss-Newton
#' (`"gn-pure"`, for teaching). Derivatives are computed exactly from the
#' formula with [stats::deriv()] when possible, and by central differences
#' otherwise.
#'
#' There are two ways to specify the model:
#'
#' * **Your own formula**, as in [stats::nls()]:
#'   `nl_fit(rate ~ Vmax * conc / (Km + conc), data, start = c(Vmax = 200, Km = 0.1))`.
#' * **A built-in model** (see [nl_models()]) with a simple `y ~ x` formula.
#'   Starting values are then found automatically:
#'   `nl_fit(rate ~ conc, data, model = "michaelis_menten")`.
#'
#' @param formula A model formula. With `model`, use `response ~ predictor`.
#' @param data A data frame (or list) containing the variables.
#' @param start Named numeric vector or list of starting values. Optional when
#'   `model` is given.
#' @param model Name of a built-in model, or `NULL`.
#' @param method `"lm"`, `"gn"` or `"gn-pure"`.
#' @param weights Optional non-negative weights; minimises `sum(w * r^2)`. Use
#'   `1 / variance` of each observation.
#' @param control Settings from [nl_control()].
#' @return An object of class `"nlestim"`. Use [summary()], [coef()],
#'   [confint()], [predict()], [plot()], [anova()], [AIC()], [residuals()],
#'   [fitted()] and [vcov()] on it. `$history` holds every iteration.
#' @seealso [nl_workflow()] to fit and compare several models in one call.
#' @examples
#' # Built-in model with automatic starting values
#' treated <- subset(Puromycin, state == "treated")
#' fit <- nl_fit(rate ~ conc, treated, model = "michaelis_menten")
#' summary(fit)
#'
#' # Your own formula, fitted by Gauss-Newton
#' fit2 <- nl_fit(rate ~ Vmax * conc / (Km + conc), treated,
#'                start = c(Vmax = 200, Km = 0.1), method = "gn")
#' coef(fit2)
#' confint(fit2, method = "profile")
#' @export
nl_fit <- function(formula, data = NULL, start = NULL, model = NULL,
                   method = c("lm", "gn", "gn-pure"), weights = NULL,
                   control = nl_control()) {
  cl <- match.call()
  method <- match.arg(method)
  if (!inherits(formula, "formula") || length(formula) != 3L) {
    stop("'formula' must be a two-sided formula such as y ~ f(x, params)", call. = FALSE)
  }
  env <- environment(formula)
  if (is.null(env)) env <- parent.frame()
  if (is.null(data)) data <- list()
  data <- as.list(data)
  rhs <- formula[[3L]]
  model_info <- NULL

  if (!is.null(model)) {
    model_info <- .get_model(model)
    if (!is.name(rhs)) {
      stop("with 'model', the formula must be 'response ~ predictor', e.g. rate ~ conc",
           call. = FALSE)
    }
    rhs <- do.call("substitute", list(model_info$rhs, list(x = rhs)))
    pnames <- model_info$params
  } else {
    if (is.null(start)) {
      stop("'start' is required for a user-supplied formula (or use a built-in 'model')",
           call. = FALSE)
    }
    pnames <- names(start)
    if (is.null(pnames) || any(pnames == "")) stop("'start' must be named", call. = FALSE)
  }

  vars <- setdiff(all.vars(rhs), pnames)
  vars <- unique(c(all.vars(formula[[2L]]), vars))
  frame <- lapply(stats::setNames(vars, vars), function(v) {
    if (!is.null(data[[v]])) data[[v]] else get(v, envir = env)
  })
  lens <- vapply(frame, length, 1L)
  n_all <- max(lens)
  frame <- frame[lens == n_all]
  keep <- Reduce(`&`, lapply(frame, function(v) !is.na(v)), TRUE)
  if (!is.null(weights)) keep <- keep & !is.na(weights)
  frame <- lapply(frame, function(v) v[keep])
  if (!is.null(weights)) weights <- weights[keep]
  y <- as.numeric(eval(formula[[2L]], frame, env))
  n <- length(y)

  if (n <= length(pnames)) {
    stop(sprintf("need more observations (%d) than parameters (%d)", n, length(pnames)), call. = FALSE)
  }
  if (!is.null(model_info)) {
    xvar <- all.vars(formula[[3L]])
    if (is.null(start)) {
      start <- tryCatch(model_info$start(as.numeric(frame[[xvar]]), y), error = function(e) NULL)
      if (is.null(start) || !all(is.finite(start))) {
        stop(sprintf("could not find starting values for model '%s'; supply 'start'", model),
             call. = FALSE)
      }
    }
  }
  start <- unlist(start)[pnames]
  if (anyNA(start)) stop("'start' must give a value for every parameter: ",
                         paste(pnames, collapse = ", "), call. = FALSE)
  p <- length(pnames)

  w <- if (is.null(weights)) rep(1, n) else as.numeric(weights)
  if (length(w) != n || any(w < 0)) stop("'weights' must be non-negative, one per observation",
                                         call. = FALSE)
  sw <- sqrt(w)

  predict_fun <- function(par, newdata = frame) {
    val <- eval(rhs, c(as.list(newdata), as.list(stats::setNames(par, pnames))), env)
    nn <- length(newdata[[1L]])
    if (length(val) == 1L && nn > 1L) val <- rep(val, nn)
    as.numeric(val)
  }
  dexpr <- tryCatch(stats::deriv(rhs, pnames), error = function(e) NULL)
  jac_fun <- function(par, newdata = frame) {
    nn <- length(newdata[[1L]])
    if (!is.null(dexpr)) {
      g <- attr(eval(dexpr, c(as.list(newdata), as.list(stats::setNames(par, pnames))), env),
                "gradient")
      if (!is.null(g) && nrow(g) == nn) return(unname(g))
    }
    unname(numeric_jacobian(function(q) predict_fun(q, newdata), par))
  }
  residual <- function(par) sw * (predict_fun(par) - y)
  jacobian <- function(par) sw * jac_fun(par)

  ctl <- control
  args <- list(residual = residual, jacobian = jacobian, start = start, rtol = ctl$rtol,
               xtol = ctl$xtol, ftol = ctl$ftol)
  if (!is.null(ctl$max_iter)) args$max_iter <- ctl$max_iter
  raw <- if (method == "lm") {
    do.call(nl_levenberg_marquardt,
            c(args, list(lambda0 = ctl$lambda0, tau = ctl$tau, scaling = ctl$scaling,
                         update = ctl$update)))
  } else {
    do.call(nl_gauss_newton, c(args, list(step_halving = method == "gn")))
  }
  if (!raw$converged) {
    warning("the solver did not converge: ", raw$message, call. = FALSE)
  }

  par <- stats::setNames(raw$par, pnames)
  fitted <- predict_fun(par)
  df <- n - p
  sigma <- sqrt(raw$sse / df)
  J <- raw$jacobian
  sv <- svd(J)
  cov <- if (all(is.finite(J)) && min(sv$d) > max(sv$d) * max(dim(J)) * .Machine$double.eps) {
    sv$v %*% (t(sv$v) / sv$d^2) * sigma^2
  } else {
    matrix(NA_real_, p, p)
  }
  dimnames(cov) <- list(pnames, pnames)

  structure(list(
    coefficients = par, residuals = y - fitted, fitted.values = fitted, weights = w,
    y = y, sse = raw$sse, df.residual = df, sigma = sigma, cov = cov, jacobian = J,
    converged = raw$converged, message = raw$message, iterations = raw$iterations,
    n_fev = raw$n_fev, history = raw$history, method = raw$method, start = start,
    call = cl, formula = formula, rhs = rhs, pnames = pnames, model = model,
    frame = frame, env = env, predict_fun = predict_fun, jac_fun = jac_fun,
    control = ctl, fit_method = method
  ), class = "nlestim")
}

.model_label <- function(object) {
  paste(deparse(object$formula[[2L]]), "~", paste(deparse(object$rhs), collapse = " "))
}
