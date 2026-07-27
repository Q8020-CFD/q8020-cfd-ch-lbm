"""Cole-Hopf linearization + TEBD for Burgers equation (F10).

Transforms the nonlinear Burgers equation into the linear heat equation
via the Cole-Hopf substitution, then evolves the transformed variable
phi(x,t) with a FIXED, state-independent evolution operator.

    u_t + u u_x = nu u_xx   <-->   phi_t = nu phi_xx

Forward transform:  phi(x) = exp( -(1/(2 nu)) * integral_0^x u(s) ds )
Inverse transform:  u(x)   = -2 nu * phi_x(x) / phi(x)

Key properties (contrast with F2):
  - H_heat = nu * Laplacian is state-INdependent: build once, reuse.
  - phi > 0 everywhere (heat equation preserves positivity), so
    shots readout gives true amplitudes without sign recovery.
  - No classical mirror: the evolution operator does not depend on
    the current state.

The heat equation propagator exp(nu * L * dt) is a contraction (NOT
unitary): it damps high-frequency modes.  In the MPS framework this is
fine; quimb applies non-unitary MPOs and we re-normalize.  For Phase B
(quantum circuit), unitarization is a separate design concern.

Phase A (this module): builds the dense propagator, converts to MPO
via from_dense, and applies to the MPS.  The propagator MPO is
constructed ONCE at setup and reused for every time step.  Dense
construction limits this to q <= ~12; same ceiling as F2 Phase A.

References (the Cole-Hopf substitution):
  - J. D. Cole, "On a quasi-linear parabolic equation occurring in
    aerodynamics," Quart. Appl. Math. 9, 225-236 (1951).
    doi:10.1090/qam/42889
  - E. Hopf, "The partial differential equation u_t + u u_x = mu u_xx,"
    Commun. Pure Appl. Math. 3, 201-230 (1950).
    doi:10.1002/cpa.3160030302
"""

from __future__ import annotations

import numpy as np

# -------------------------------------------------------------------
# Cole-Hopf transforms
# -------------------------------------------------------------------


def cole_hopf_forward(
    u: np.ndarray,
    dx: float,
    nu: float,
) -> np.ndarray:
    """Forward Cole-Hopf transform: u -> phi.

    phi(x) = exp( -(1/(2 nu)) * integral_0^x u(s) ds )

    Uses cumulative trapezoidal integration on the uniform grid.
    Returns un-normalized phi (positive everywhere).
    """
    integral = np.zeros_like(u)
    integral[1:] = np.cumsum(0.5 * (u[:-1] + u[1:]) * dx)
    phi = np.exp(-integral / (2.0 * nu))
    return phi


def cole_hopf_forward_centered(
    u: np.ndarray,
    dx: float,
    nu: float,
) -> tuple[np.ndarray, float]:
    """Forward Cole-Hopf in log domain for small-nu stability.

    Computes e(x) = -(1/(2 nu)) * integral_0^x u(s) ds, shifts by
    e_mid = 0.5*(max(e)+min(e)).  Returns the centered exponent
    (log_phi_centered = e - e_mid) instead of exp(e - e_mid), because
    at nu < ~1e-3 the exp overflows float64.

    Returns (log_phi_centered, e_mid).  Use cole_hopf_inverse_centered
    for the inverse (operates in log domain, never exponentiates).
    """
    integral = np.zeros_like(u)
    integral[1:] = np.cumsum(0.5 * (u[:-1] + u[1:]) * dx)
    exponent = -integral / (2.0 * nu)
    e_mid = 0.5 * (np.max(exponent) + np.min(exponent))
    return exponent - e_mid, float(e_mid)


def log_phi_to_normalized_psi(log_phi: np.ndarray) -> np.ndarray:
    """Convert log-domain phi to unit-norm psi via log-sum-exp.

    psi(x) = exp(log_phi(x) - log_norm) where
    log_norm = 0.5 * logsumexp(2 * log_phi).

    At extreme dynamic ranges (nu << 1e-3) psi will be nearly a
    delta function; this is physically correct.
    """
    twice = 2.0 * log_phi
    max_val = np.max(twice)
    log_norm = 0.5 * (max_val + np.log(np.sum(np.exp(twice - max_val))))
    return np.exp(log_phi - log_norm)


def cole_hopf_inverse(
    phi: np.ndarray,
    dx: float,
    nu: float,
    eps_floor: float | None = None,
    bc: str = "periodic",
) -> np.ndarray:
    """Inverse Cole-Hopf transform: phi -> u.

    u(x) = -2 nu * phi_x(x) / phi(x) = -2 nu * d(log phi)/dx

    Uses the logarithmic form for numerical stability: FD is applied
    to log(phi) rather than phi, avoiding division by near-zero phi
    values that arise at small nu (large dynamic range).

    Central difference in the interior.  At the boundaries: periodic
    wrapping for bc='periodic'; one-sided (forward/backward) differences
    otherwise.  The wrap is wrong under Dirichlet -- it pulls the
    opposite wall's phi into the wall derivative, and when phi spans a
    large dynamic range that manufactures a spurious u spike at each
    wall.  One-sided differences keep the wall value local.
    """
    if eps_floor is None:
        eps_floor = 1e-30
    phi_safe = np.maximum(phi, eps_floor)
    log_phi = np.log(phi_safe)

    dlog = np.zeros_like(phi)
    dlog[1:-1] = (log_phi[2:] - log_phi[:-2]) / (2.0 * dx)
    if bc == "periodic":
        dlog[0] = (log_phi[1] - log_phi[-1]) / (2.0 * dx)
        dlog[-1] = (log_phi[0] - log_phi[-2]) / (2.0 * dx)
    else:
        dlog[0] = (log_phi[1] - log_phi[0]) / dx
        dlog[-1] = (log_phi[-1] - log_phi[-2]) / dx

    u = -2.0 * nu * dlog
    return u


def fourier_low_pass_phi(
    phi: np.ndarray,
    n_modes: int,
) -> np.ndarray:
    """Low-pass filter phi by keeping only the first n_modes Fourier modes.

    Zeroes out high-frequency components that are dominated by shot
    noise, before the log-derivative in cole_hopf_inverse amplifies
    them.  n_modes=0 means no filtering (return as-is).
    """
    if n_modes <= 0:
        return phi
    N = len(phi)
    n_modes = min(n_modes, N // 2)
    fk = np.fft.rfft(phi)
    fk[n_modes + 1:] = 0.0
    return np.fft.irfft(fk, n=N)


def _should_center(u: np.ndarray, dx: float, nu: float) -> bool:
    """Decide whether to use centered exponent for the CH transform.

    Returns True when |e_max - e_min| > 50 or nu < 1e-3.
    """
    integral = np.zeros_like(u)
    integral[1:] = np.cumsum(0.5 * (u[:-1] + u[1:]) * dx)
    exponent = -integral / (2.0 * nu)
    e_range = float(np.max(exponent) - np.min(exponent))
    return e_range > 50.0 or nu < 1e-3


# -------------------------------------------------------------------
# Closed-form analytic family (FUTURE-WORK #12)
#
# When phi(x, t=0) is a finite Neumann cosine sum on [0, L_box],
#   phi_0(x) = a_0 + sum_{n>=1} a_n * cos(n*pi*x/L_box),
# each mode evolves independently under the heat equation:
#   phi(x, t) = a_0 + sum_n a_n * cos(n*pi*x/L_box) * exp(-nu*(n*pi/L_box)^2 * t).
# The inverse Cole-Hopf transform u = -2*nu * phi_x / phi then gives a
# closed-form analytic reference for u(x, t).  This is restricted to
# --bc dirichlet (Neumann-on-phi) and unforced --source none.  Used by
# burgers_solver.py to replace the FTCS classical reference with an
# exact one in this narrow but rigorous test family.
# -------------------------------------------------------------------


def validate_cole_hopf_coeffs(coeffs: np.ndarray) -> None:
    """Reject coefficient lists that would make phi(x, t) <= 0 anywhere.

    Sufficient (not necessary) positivity condition: a_0 > sum(|a_n|)
    for n>=1.  Since modes only decay over time, satisfying this at
    t=0 guarantees phi > 0 for all t >= 0.
    """
    if coeffs.ndim != 1 or coeffs.size < 1:
        raise ValueError(
            "coeffs must be a 1-D array with at least one element (a_0)"
        )
    a0 = float(coeffs[0])
    tail = float(np.sum(np.abs(coeffs[1:])))
    if a0 <= tail:
        raise ValueError(
            f"Cole-Hopf coefficients fail positivity: a_0={a0:.6g} must "
            f"exceed sum(|a_n|)={tail:.6g} so that phi(x,t) > 0 for all "
            f"t >= 0."
        )


def _phi_and_phi_x(
    x: np.ndarray, t: float, coeffs: np.ndarray, nu: float, L_box: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate phi(x, t) and phi_x(x, t) on the grid."""
    n_modes = coeffs.size - 1
    phi = np.full_like(x, float(coeffs[0]), dtype=float)
    phi_x = np.zeros_like(x, dtype=float)
    for n in range(1, n_modes + 1):
        a_n = float(coeffs[n])
        if a_n == 0.0:
            continue
        kn = n * np.pi / L_box
        decay = np.exp(-nu * kn * kn * t)
        phi = phi + a_n * np.cos(kn * x) * decay
        phi_x = phi_x - a_n * kn * np.sin(kn * x) * decay
    return phi, phi_x


def initial_condition_cole_hopf_exact(
    x: np.ndarray, coeffs: np.ndarray, nu: float, L_box: float = 1.0,
) -> np.ndarray:
    """u_0(x) from a Neumann cosine sum phi_0(x).

    coeffs[0] = a_0; coeffs[n] = a_n for n >= 1.  Returns
    u_0(x) = -2*nu * phi_x(x, 0) / phi(x, 0).
    """
    validate_cole_hopf_coeffs(coeffs)
    phi, phi_x = _phi_and_phi_x(x, 0.0, coeffs, nu, L_box)
    return -2.0 * nu * phi_x / phi


def analytic_solution_cole_hopf(
    x: np.ndarray, t: float, coeffs: np.ndarray, nu: float,
    L_box: float = 1.0,
) -> np.ndarray:
    """u(x, t) under heat-equation evolution of phi_0 (no source).

    Each mode amplitude decays as a_n -> a_n * exp(-nu*(n*pi/L_box)^2 * t).
    """
    validate_cole_hopf_coeffs(coeffs)
    phi, phi_x = _phi_and_phi_x(x, t, coeffs, nu, L_box)
    return -2.0 * nu * phi_x / phi

