# Gauss-Newton and Levenberg-Marquardt for nonlinear least squares.
#
# Both minimise S(theta) = sum r_i(theta)^2 with r = f(x, theta) - y by
# linearising the model, f(x, theta + delta) ~ f(x, theta) + J delta, and
# solving a linear least-squares problem for the step delta. Linear systems
# are solved through a QR decomposition of J, never by forming t(J) %*% J,
# which would square the condition number.

#' Relative-offset convergence criterion
#'
#' The Bates and Watts (1981) relative offset compares the part of the
#' residual vector lying in the tangent plane of the model surface (which a
#' further step could remove) with the part orthogonal to it (which no
#' parameter change can remove). It is zero at an exact least-squares solution
#' and does not depend on how the parameters are scaled. R's [stats::nls()]
#' uses the same criterion.
#'
#' @param J Jacobian matrix (n x p) of the model with respect to the parameters.
#' @param r Residual vector of length n.
#' @return A non-negative number; values below about 1e-8 indicate convergence.
#' @references Bates, D. M. and Watts, D. G. (1981). A relative offset
#'   orthogonality convergence criterion for nonlinear least squares.
#'   *Technometrics*, 23, 179-183.
#' @examples
#' J <- cbind(1, 1:10)
#' y <- c(2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1, 18.0, 20.2)
#' beta <- qr.coef(qr(J), y)
#' relative_offset(J, J %*% beta - y)          # ~ 0 at the least-squares solution
#' relative_offset(J, J %*% (beta + 0.1) - y)  # clearly > 0 elsewhere
#' @export
relative_offset <- function(J, r) {
  J <- as.matrix(J)
  r <- as.numeric(r)
  n <- nrow(J)
  p <- ncol(J)
  rr <- sum(r^2)
  if (rr <= .Machine$double.xmin) return(0)
  Q <- qr.Q(qr(J))
  tangential <- sum(crossprod(Q, r)^2)
  orthogonal <- max(rr - tangential, 1e-300 * rr)
  if (n == p) return(sqrt(tangential / rr))
  sqrt((tangential / p) / (orthogonal / (n - p)))
}

#' Finite-difference Jacobian
#'
#' @param f A function of the parameter vector returning a numeric vector.
#' @param par Numeric parameter vector.
#' @param method `"central"` (accurate to O(h^2)) or `"forward"` (cheaper,
#'   O(h)).
#' @return An n x p matrix of partial derivatives.
#' @examples
#' f <- function(p) c(sin(p[1]) * p[2], p[2]^3)
#' numeric_jacobian(f, c(0.7, 1.3))
#' @export
numeric_jacobian <- function(f, par, method = c("central", "forward")) {
  method <- match.arg(method)
  f0 <- as.numeric(f(par))
  J <- matrix(NA_real_, length(f0), length(par))
  eps <- .Machine$double.eps
  for (j in seq_along(par)) {
    if (method == "central") {
      h <- eps^(1 / 3) * max(abs(par[j]), 1)
      up <- par
      dn <- par
      up[j] <- up[j] + h
      dn[j] <- dn[j] - h
      J[, j] <- (as.numeric(f(up)) - as.numeric(f(dn))) / (up[j] - dn[j])
    } else {
      h <- sqrt(eps) * max(abs(par[j]), 1)
      up <- par
      up[j] <- up[j] + h
      J[, j] <- (as.numeric(f(up)) - f0) / (up[j] - par[j])
    }
  }
  colnames(J) <- names(par)
  J
}

.safe_sse <- function(r) {
  s <- sum(r^2)
  if (is.finite(s)) s else Inf
}

.stop_check <- function(J, r, sse_old, sse_new, step, par, rtol, xtol, ftol) {
  offset <- relative_offset(J, r)
  if (offset <= rtol) {
    return(list(converged = TRUE, message = sprintf(
      "converged: relative offset %.1e <= rtol", offset)))
  }
  small_step <- all(abs(step) <= xtol * (abs(par) + xtol))
  small_change <- (sse_old - sse_new) >= 0 &&
    (sse_old - sse_new) <= ftol * max(sse_old, .Machine$double.xmin)
  if (small_step || small_change) {
    why <- if (small_step) "step" else "SSE reduction"
    if (offset <= 1e-5) {
      return(list(converged = TRUE, message = sprintf(
        "converged: %s negligible, relative offset %.1e", why, offset)))
    }
    return(list(converged = FALSE, message = sprintf(
      "stalled: %s negligible but relative offset is %.1e (not at a minimum)", why, offset)))
  }
  NULL
}

.history_row <- function(iter, par, sse, step_norm, lambda, accepted) {
  data.frame(iteration = iter, t(par), sse = sse, step_norm = step_norm,
             lambda = lambda, accepted = accepted, check.names = FALSE)
}

.solver_result <- function(par, r, J, sse, converged, message, n_fev, method, history) {
  history <- do.call(rbind, history)
  rownames(history) <- NULL
  list(par = par, residuals = r, jacobian = J, sse = sse, converged = converged,
       message = message, iterations = nrow(history) - 1L, n_fev = n_fev,
       method = method, history = history)
}

#' Gauss-Newton algorithm
#'
#' Minimises `sum(residual(par)^2)` with Gauss-Newton steps
#' `(J'J) delta = -J'r`, computed by QR. With `step_halving = TRUE`
#' (Hartley, 1961; the default algorithm of [stats::nls()]) a step that
#' increases the sum of squares is halved until it decreases. With
#' `step_halving = FALSE` you get the textbook method, which can diverge from
#' a poor start.
#'
#' Most users should call [nl_fit()]; these low-level solvers are exported for
#' teaching and for custom problems.
#'
#' @param residual Function of the parameter vector returning `f(x, par) - y`.
#' @param jacobian Function returning the n x p Jacobian of the model.
#' @param start Named numeric vector of starting values.
#' @param step_halving Logical; safeguard steps by halving.
#' @param max_iter Maximum number of iterations.
#' @param rtol Relative-offset tolerance (see [relative_offset()]).
#' @param xtol Component-wise relative step tolerance.
#' @param ftol Relative SSE-reduction tolerance.
#' @param max_halvings Maximum number of step halvings per iteration.
#' @return A list with `par`, `residuals`, `jacobian`, `sse`, `converged`,
#'   `message`, `iterations`, `n_fev`, `method` and `history` (a data frame
#'   with one row per iteration).
#' @references Hartley, H. O. (1961). The modified Gauss-Newton method for the
#'   fitting of non-linear regression functions by least squares.
#'   *Technometrics*, 3, 269-280.
#' @seealso [nl_levenberg_marquardt()], [nl_fit()]
#' @examples
#' x <- c(1, 2, 3, 5, 7, 10); y <- c(109, 149, 149, 191, 213, 224)
#' res <- function(p) p[1] * (1 - exp(-p[2] * x)) - y
#' jac <- function(p) cbind(1 - exp(-p[2] * x), p[1] * x * exp(-p[2] * x))
#' nl_gauss_newton(res, jac, c(a = 100, k = 0.75))$par
#' @export
nl_gauss_newton <- function(residual, jacobian, start, step_halving = TRUE,
                            max_iter = 200, rtol = 1e-10, xtol = 1e-14,
                            ftol = 1e-15, max_halvings = 30) {
  par <- start
  r <- residual(par)
  sse <- .safe_sse(r)
  if (!is.finite(sse)) stop("the model is not finite at the starting values", call. = FALSE)
  J <- jacobian(par)
  n_fev <- 1L
  history <- list(.history_row(0L, par, sse, 0, NA_real_, TRUE))
  converged <- FALSE
  message <- sprintf("reached max_iter = %d", max_iter)
  if (relative_offset(J, r) <= rtol) {
    return(.solver_result(par, r, J, sse, TRUE, "converged: starting values are already optimal",
                          n_fev, "gauss-newton", history))
  }
  for (it in seq_len(max_iter)) {
    delta <- qr.coef(qr(J), -r)
    if (anyNA(delta) || !all(is.finite(delta))) {
      message <- "failed: the Jacobian is singular (parameters not identifiable here)"
      break
    }
    factor <- 1
    repeat {
      trial <- par + factor * delta
      r_new <- suppressWarnings(residual(trial))
      n_fev <- n_fev + 1L
      sse_new <- if (all(is.finite(r_new))) .safe_sse(r_new) else Inf
      if (!step_halving || sse_new <= sse) break
      factor <- factor / 2
      if (factor < 2^-max_halvings) break
    }
    step <- factor * delta
    if (!is.finite(sse_new)) {
      history[[length(history) + 1L]] <- .history_row(it, trial, Inf, sqrt(sum(step^2)), NA_real_, TRUE)
      message <- "diverged: the model produced non-finite values"
      par <- trial
      sse <- Inf
      break
    }
    if (step_halving && sse_new > sse) {
      message <- "stopped: step halving could not reduce the SSE"
      break
    }
    sse_old <- sse
    par <- trial
    r <- r_new
    sse <- sse_new
    J <- suppressWarnings(jacobian(par))
    history[[length(history) + 1L]] <- .history_row(it, par, sse, sqrt(sum(step^2)), NA_real_, TRUE)
    if (!all(is.finite(J))) {
      message <- "diverged: the Jacobian is not finite"
      break
    }
    chk <- .stop_check(J, r, sse_old, sse, step, par, rtol, xtol, ftol)
    if (!is.null(chk)) {
      converged <- chk$converged
      message <- chk$message
      break
    }
  }
  .solver_result(par, r, J, sse, converged, message, n_fev,
                 if (step_halving) "gauss-newton" else "gauss-newton (pure)", history)
}

#' Levenberg-Marquardt algorithm
#'
#' Solves the damped system `(J'J + lambda D) delta = -J'r` as the augmented
#' least-squares problem `[J; sqrt(lambda D)] delta = [-r; 0]`. Small lambda
#' gives the Gauss-Newton step; large lambda gives a short steepest-descent
#' step. After each trial step the gain ratio (actual over predicted SSE
#' reduction) decides whether to accept the step and how to update lambda.
#'
#' @inheritParams nl_gauss_newton
#' @param lambda0 Initial damping. By default `tau * max(diag(J'J))`.
#' @param tau Scale factor for the default `lambda0`.
#' @param scaling `"marquardt"` uses `D = diag(J'J)` (invariant to rescaling
#'   parameters); `"levenberg"` uses the identity matrix.
#' @param update `"nielsen"` (smooth gain-ratio update; Madsen, Nielsen and
#'   Tingleff, 2004) or `"classic"` (divide or multiply lambda by 10).
#' @inherit nl_gauss_newton return
#' @references Levenberg, K. (1944). *Quarterly of Applied Mathematics*, 2,
#'   164-168. Marquardt, D. W. (1963). *Journal of the Society for Industrial
#'   and Applied Mathematics*, 11, 431-441.
#' @seealso [nl_gauss_newton()], [nl_fit()]
#' @examples
#' x <- c(1, 2, 3, 5, 7, 10); y <- c(109, 149, 149, 191, 213, 224)
#' res <- function(p) p[1] * (1 - exp(-p[2] * x)) - y
#' jac <- function(p) cbind(1 - exp(-p[2] * x), p[1] * x * exp(-p[2] * x))
#' fit <- nl_levenberg_marquardt(res, jac, c(a = 100, k = 0.75))
#' fit$par
#' head(fit$history)
#' @export
nl_levenberg_marquardt <- function(residual, jacobian, start, lambda0 = NULL,
                                   tau = 1e-3, scaling = c("marquardt", "levenberg"),
                                   update = c("nielsen", "classic"), max_iter = 500,
                                   rtol = 1e-10, xtol = 1e-14, ftol = 1e-15) {
  scaling <- match.arg(scaling)
  update <- match.arg(update)
  par <- start
  p <- length(par)
  r <- residual(par)
  sse <- .safe_sse(r)
  if (!is.finite(sse)) stop("the model is not finite at the starting values", call. = FALSE)
  J <- jacobian(par)
  n_fev <- 1L
  diag_jtj <- function(J) colSums(J^2)
  damping <- function(J) {
    d <- if (scaling == "marquardt") diag_jtj(J) else rep(1, p)
    pmax(d, 1e-12 * max(d, 1))
  }
  D <- damping(J)
  lambda <- if (is.null(lambda0)) tau * max(diag_jtj(J)) else lambda0
  lambda <- max(lambda, 1e-12)
  nu <- 2
  history <- list(.history_row(0L, par, sse, 0, lambda, TRUE))
  converged <- FALSE
  message <- sprintf("reached max_iter = %d", max_iter)
  if (relative_offset(J, r) <= rtol) {
    return(.solver_result(par, r, J, sse, TRUE, "converged: starting values are already optimal",
                          n_fev, paste0("levenberg-marquardt (", scaling, ")"), history))
  }
  for (it in seq_len(max_iter)) {
    grad <- crossprod(J, r)
    A <- rbind(J, diag(sqrt(lambda * D), p))
    delta <- qr.coef(qr(A), c(-r, rep(0, p)))
    delta[is.na(delta)] <- 0
    trial <- par + delta
    r_new <- suppressWarnings(residual(trial))
    n_fev <- n_fev + 1L
    sse_new <- if (all(is.finite(r_new))) .safe_sse(r_new) else Inf
    Jd <- J %*% delta
    predicted <- -2 * sum(delta * grad) - sum(Jd^2)
    rho <- if (predicted > 0 && is.finite(sse_new)) (sse - sse_new) / predicted else -1
    if (rho > 0) {
      sse_old <- sse
      par <- trial
      r <- r_new
      sse <- sse_new
      J <- suppressWarnings(jacobian(par))
      if (!all(is.finite(J))) {
        message <- "diverged: the Jacobian is not finite"
        break
      }
      if (scaling == "marquardt") D <- pmax(D, damping(J))
      if (update == "nielsen") {
        lambda <- lambda * max(1 / 3, 1 - (2 * rho - 1)^3)
        nu <- 2
      } else {
        lambda <- lambda / 10
      }
      lambda <- max(lambda, 1e-15)
      history[[length(history) + 1L]] <- .history_row(it, par, sse, sqrt(sum(delta^2)), lambda, TRUE)
      chk <- .stop_check(J, r, sse_old, sse, delta, par, rtol, xtol, ftol)
      if (!is.null(chk)) {
        converged <- chk$converged
        message <- chk$message
        break
      }
    } else {
      if (update == "nielsen") {
        lambda <- lambda * nu
        nu <- 2 * nu
      } else {
        lambda <- lambda * 10
      }
      history[[length(history) + 1L]] <- .history_row(it, par, sse, sqrt(sum(delta^2)), lambda, FALSE)
      if (all(abs(delta) <= xtol * (abs(par) + xtol))) {
        offset <- relative_offset(J, r)
        converged <- offset <= 1e-5
        message <- sprintf("%s: no downhill step left, relative offset %.1e",
                           if (converged) "converged" else "stalled", offset)
        break
      }
      if (lambda > 1e16) {
        message <- "stalled: damping lambda became too large (no downhill step found)"
        break
      }
    }
  }
  .solver_result(par, r, J, sse, converged, message, n_fev,
                 paste0("levenberg-marquardt (", scaling, ")"), history)
}
