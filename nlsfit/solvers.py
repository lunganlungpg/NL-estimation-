"""Gauss–Newton and Levenberg–Marquardt, written to be read.

Both solve the nonlinear least-squares problem

    minimise  S(θ) = Σ rᵢ(θ)²,   rᵢ(θ) = f(xᵢ, θ) − yᵢ

by repeatedly linearising the model around the current estimate,
``f(x, θ + δ) ≈ f(x, θ) + J δ``, where ``J = ∂f/∂θ`` is the n × p Jacobian.

* **Gauss–Newton** takes the step that minimises the linearised problem:
  ``(JᵀJ) δ = −Jᵀr``. Fast (near-quadratic convergence) when started close
  to the solution, but it can overshoot or diverge from a poor start.
* **Levenberg–Marquardt** adds a damping term:
  ``(JᵀJ + λD) δ = −Jᵀr``. Small λ gives the Gauss–Newton step; large λ gives
  a short step along the steepest-descent direction. λ is adapted every
  iteration, so LM behaves like gradient descent far from the solution and
  like Gauss–Newton close to it.

Every linear system here is solved as a least-squares problem with
``numpy.linalg.lstsq`` instead of forming ``JᵀJ`` explicitly, which squares
the condition number and loses accuracy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

ResidualFn = Callable[[np.ndarray], np.ndarray]
JacobianFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class Iteration:
    """One row of the optimisation history."""

    iteration: int
    params: np.ndarray
    sse: float
    grad_norm: float
    step_norm: float = 0.0
    lam: float | None = None
    accepted: bool = True


@dataclass
class SolverResult:
    """Raw output of a solver (statistics are added by :class:`nlsfit.FitResult`)."""

    params: np.ndarray
    residuals: np.ndarray
    jacobian: np.ndarray
    sse: float
    converged: bool
    message: str
    n_iter: int
    n_fev: int
    n_jev: int
    method: str
    history: list[Iteration] = field(default_factory=list)


def relative_offset(J: np.ndarray, r: np.ndarray) -> float:
    """Bates & Watts (1981) relative-offset convergence measure.

    It compares the part of the residual vector lying in the tangent plane
    of the model surface (which the next step could still remove) with the
    part orthogonal to it (which no parameter change can remove). It is zero
    at an exact least-squares solution and does not depend on how the
    parameters are scaled, unlike step-size or SSE-change tests. R's ``nls``
    uses the same criterion.
    """
    n, p = J.shape
    rr = float(r @ r)
    if rr <= np.finfo(float).tiny:
        return 0.0
    Q, _ = np.linalg.qr(J)
    qr = Q.T @ r
    tangential = float(qr @ qr)
    orthogonal = max(rr - tangential, 1e-300 * rr)
    if n == p:
        return np.sqrt(tangential / rr)
    return float(np.sqrt((tangential / p) / (orthogonal / (n - p))))


def _check(J, r, sse_old, sse_new, step, params, rtol, xtol, ftol) -> tuple[bool, str] | None:
    """Return (converged, message) if the iteration should stop, else None."""
    offset = relative_offset(J, r)
    if offset <= rtol:
        return True, f"converged: relative offset {offset:.1e} ≤ rtol={rtol:g}"
    # Component-wise, so parameters on very different scales are each judged on their own
    small_step = bool(np.all(np.abs(step) <= xtol * (np.abs(params) + xtol)))
    small_change = 0 <= sse_old - sse_new <= ftol * max(sse_old, np.finfo(float).tiny)
    if small_step or small_change:
        why = "step" if small_step else "SSE reduction"
        if offset <= 1e-5:  # R's nls default tolerance: accept, but say so
            return True, f"converged: {why} negligible, relative offset {offset:.1e}"
        return False, f"stalled: {why} negligible but relative offset is {offset:.1e} (not at a minimum)"
    return None


def gauss_newton(
    residual: ResidualFn,
    jacobian: JacobianFn,
    p0: np.ndarray,
    *,
    step_halving: bool = True,
    max_halvings: int = 30,
    max_iter: int = 200,
    rtol: float = 1e-10,
    xtol: float = 1e-14,
    ftol: float = 1e-15,
) -> SolverResult:
    """Gauss–Newton with optional step halving (Hartley, 1961).

    Parameters
    ----------
    residual, jacobian
        ``residual(θ)`` returns ``f(x, θ) − y``; ``jacobian(θ)`` returns ``∂f/∂θ``.
    p0
        Starting values.
    step_halving
        If True (the default, as in R's ``nls``), a step that increases the
        SSE is halved until it decreases. Set to False for the textbook
        algorithm, which can diverge from a poor start.
    """
    theta = np.asarray(p0, dtype=float).copy()
    r = residual(theta)
    sse = float(r @ r)
    n_fev, n_jev = 1, 0
    if not np.isfinite(sse):
        raise ValueError("The model returns non-finite values at the starting parameters")
    J = jacobian(theta)
    n_jev += 1
    history = [Iteration(0, theta.copy(), sse, float(np.max(np.abs(J.T @ r))))]
    message, converged = f"reached max_iter={max_iter}", False
    if relative_offset(J, r) <= rtol:
        message, converged, max_iter = "converged: starting values are already optimal", True, 0

    for it in range(1, max_iter + 1):
        # Gauss–Newton step: least-squares solution of J δ = −r
        delta, *_ = np.linalg.lstsq(J, -r, rcond=None)
        if not np.all(np.isfinite(delta)):
            message = "Gauss–Newton step is not finite (singular Jacobian)"
            break

        factor = 1.0
        while True:
            trial = theta + factor * delta
            with np.errstate(all="ignore"):
                r_new = residual(trial)
                sse_new = float(r_new @ r_new) if np.all(np.isfinite(r_new)) else np.inf
            n_fev += 1
            if not step_halving or sse_new <= sse:
                break
            factor /= 2
            if factor < 2.0**-max_halvings:
                break

        step = factor * delta
        if not np.isfinite(sse_new):
            history.append(Iteration(it, trial.copy(), np.inf, np.nan, float(np.linalg.norm(step))))
            message = "diverged: the model produced non-finite values"
            theta = trial
            break
        if step_halving and sse_new > sse:
            message = "step halving failed to reduce the SSE"
            break

        sse_old, theta, r, sse = sse, trial, r_new, sse_new
        with np.errstate(all="ignore"):
            J = jacobian(theta)
        n_jev += 1
        grad = J.T @ r
        history.append(
            Iteration(it, theta.copy(), sse, float(np.max(np.abs(grad))), float(np.linalg.norm(step)))
        )
        if not np.all(np.isfinite(J)):
            message = "diverged: the Jacobian is not finite"
            break
        stop = _check(J, r, sse_old, sse, step, theta, rtol, xtol, ftol)
        if stop:
            converged, message = stop
            break
        if sse > 1e300:
            message = "diverged: SSE overflowed"
            break

    return SolverResult(
        params=theta,
        residuals=r,
        jacobian=J,
        sse=sse,
        converged=converged,
        message=message,
        n_iter=len(history) - 1,
        n_fev=n_fev,
        n_jev=n_jev,
        method="gauss-newton" + ("" if step_halving else " (pure)"),
        history=history,
    )


def levenberg_marquardt(
    residual: ResidualFn,
    jacobian: JacobianFn,
    p0: np.ndarray,
    *,
    lam0: float | None = None,
    tau: float = 1e-3,
    scaling: str = "marquardt",
    update: str = "nielsen",
    max_iter: int = 500,
    rtol: float = 1e-10,
    xtol: float = 1e-14,
    ftol: float = 1e-15,
) -> SolverResult:
    """Levenberg–Marquardt with gain-ratio damping control.

    Parameters
    ----------
    lam0
        Initial damping λ. Defaults to ``tau * max(diag(JᵀJ))`` (Madsen,
        Nielsen & Tingleff, 2004), which makes the first step scale-aware.
    scaling
        ``"marquardt"`` uses ``D = diag(JᵀJ)`` so the damping is invariant to
        rescaling the parameters (Marquardt, 1963). ``"levenberg"`` uses the
        identity matrix (Levenberg, 1944).
    update
        ``"nielsen"`` adapts λ smoothly from the gain ratio ρ (actual vs
        predicted SSE reduction). ``"classic"`` divides λ by 10 after a
        successful step and multiplies it by 10 after a failed one.
    """
    if scaling not in ("marquardt", "levenberg"):
        raise ValueError("scaling must be 'marquardt' or 'levenberg'")
    if update not in ("nielsen", "classic"):
        raise ValueError("update must be 'nielsen' or 'classic'")

    theta = np.asarray(p0, dtype=float).copy()
    r = residual(theta)
    sse = float(r @ r)
    if not np.isfinite(sse):
        raise ValueError("The model returns non-finite values at the starting parameters")
    J = jacobian(theta)
    n_fev, n_jev = 1, 1
    p = theta.size

    def damping_diag(J):
        d = np.einsum("ij,ij->j", J, J) if scaling == "marquardt" else np.ones(p)
        return np.maximum(d, 1e-12 * max(d.max(), 1.0))  # keep D positive definite

    D = damping_diag(J)
    lam = float(lam0) if lam0 is not None else tau * float(np.max(np.einsum("ij,ij->j", J, J)))
    lam = max(lam, 1e-12)
    nu = 2.0
    grad = J.T @ r
    history = [Iteration(0, theta.copy(), sse, float(np.max(np.abs(grad))), lam=lam)]
    message, converged = f"reached max_iter={max_iter}", False

    for it in range(1, max_iter + 1):
        if it == 1 and relative_offset(J, r) <= rtol:
            message, converged = "converged: starting values are already optimal", True
            break
        # Damped step: least-squares solution of [J; sqrt(λD)] δ = [−r; 0]
        A = np.vstack([J, np.diag(np.sqrt(lam * D))])
        b = np.concatenate([-r, np.zeros(p)])
        delta, *_ = np.linalg.lstsq(A, b, rcond=None)

        trial = theta + delta
        with np.errstate(all="ignore"):
            r_new = residual(trial)
            sse_new = float(r_new @ r_new) if np.all(np.isfinite(r_new)) else np.inf
        n_fev += 1
        # SSE reduction predicted by the linearised model: −2δᵀJᵀr − ‖Jδ‖²
        Jd = J @ delta
        predicted = -2.0 * float(delta @ grad) - float(Jd @ Jd)
        rho = (sse - sse_new) / predicted if predicted > 0 else -1.0

        if rho > 0 and np.isfinite(sse_new):
            sse_old = sse
            theta, r, sse = trial, r_new, sse_new
            J = jacobian(theta)
            n_jev += 1
            grad = J.T @ r
            D = np.maximum(D, damping_diag(J)) if scaling == "marquardt" else D
            if update == "nielsen":
                lam *= max(1 / 3, 1 - (2 * rho - 1) ** 3)
                nu = 2.0
            else:
                lam /= 10
            lam = max(lam, 1e-15)
            history.append(
                Iteration(
                    it, theta.copy(), sse, float(np.max(np.abs(grad))), float(np.linalg.norm(delta)), lam
                )
            )
            stop = _check(J, r, sse_old, sse, delta, theta, rtol, xtol, ftol)
            if stop:
                converged, message = stop
                break
        else:
            if update == "nielsen":
                lam *= nu
                nu *= 2
            else:
                lam *= 10
            history.append(
                Iteration(
                    it,
                    theta.copy(),
                    sse,
                    float(np.max(np.abs(grad))),
                    float(np.linalg.norm(delta)),
                    lam,
                    accepted=False,
                )
            )
            if np.all(np.abs(delta) <= xtol * (np.abs(theta) + xtol)):
                offset = relative_offset(J, r)
                converged = offset <= 1e-5
                message = (
                    f"{'converged' if converged else 'stalled'}: no downhill step "
                    f"left, relative offset {offset:.1e}"
                )
                break
            if lam > 1e16:
                message = "stopped: damping λ became too large (no downhill step found)"
                break

    return SolverResult(
        params=theta,
        residuals=r,
        jacobian=J,
        sse=sse,
        converged=converged,
        message=message,
        n_iter=len(history) - 1,
        n_fev=n_fev,
        n_jev=n_jev,
        method=f"levenberg-marquardt ({scaling})",
        history=history,
    )


def numerical_jacobian(
    func: Callable[[np.ndarray], np.ndarray], theta: np.ndarray, method: str = "central"
) -> np.ndarray:
    """Finite-difference Jacobian of ``func`` (returns an n × p array).

    Central differences are accurate to O(h²); forward differences need half
    the evaluations but are only O(h) accurate. Step sizes are scaled to each
    parameter's magnitude.
    """
    theta = np.asarray(theta, dtype=float)
    f0 = np.asarray(func(theta), dtype=float)
    J = np.empty((f0.size, theta.size))
    eps = np.finfo(float).eps
    for j in range(theta.size):
        if method == "central":
            h = eps ** (1 / 3) * max(abs(theta[j]), 1.0)
            tp, tm = theta.copy(), theta.copy()
            tp[j] += h
            tm[j] -= h
            J[:, j] = (np.asarray(func(tp)) - np.asarray(func(tm))) / (tp[j] - tm[j])
        elif method == "forward":
            h = np.sqrt(eps) * max(abs(theta[j]), 1.0)
            tp = theta.copy()
            tp[j] += h
            J[:, j] = (np.asarray(func(tp)) - f0) / (tp[j] - theta[j])
        else:
            raise ValueError("method must be 'central' or 'forward'")
    return J
