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

import sys
from typing import Any

import numpy as np
import quimb.tensor as qtn
from scipy.linalg import expm

from burgers_mpo import shift_matrix
from burgers_tebd import dense_to_mpo, mps_to_vector, state_to_mps


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


def cole_hopf_inverse_centered(
    log_phi_centered: np.ndarray,
    e_mid: float,
    dx: float,
    nu: float,
) -> np.ndarray:
    """Inverse Cole-Hopf from log-domain centered exponent.

    u = -2 nu * d(log phi)/dx = -2 nu * d(log_phi_centered + e_mid)/dx
      = -2 nu * d(log_phi_centered)/dx

    The e_mid constant drops out.  FD is applied directly to the
    log-domain array, so this works at any nu without overflow.
    """
    dlog = np.zeros_like(log_phi_centered)
    dlog[1:-1] = (
        (log_phi_centered[2:] - log_phi_centered[:-2]) / (2.0 * dx)
    )
    dlog[0] = (
        (log_phi_centered[1] - log_phi_centered[-1]) / (2.0 * dx)
    )
    dlog[-1] = (
        (log_phi_centered[0] - log_phi_centered[-2]) / (2.0 * dx)
    )
    return -2.0 * nu * dlog


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


# -------------------------------------------------------------------
# Heat-equation Laplacian and propagator
# -------------------------------------------------------------------


def build_laplacian_dense(
    N: int,
    dx: float,
    bc: str = "periodic",
) -> np.ndarray:
    """Build the N x N Laplacian matrix using shift operators.

    L = (S+ + S- - 2I) / dx^2

    bc='periodic': wrapping shift operators (default).
    bc='neumann' : zero-flux boundaries (phi_x = 0 at endpoints).
    """
    if bc == "periodic":
        sp = shift_matrix(N, +1, bc="periodic")
        sm = shift_matrix(N, -1, bc="periodic")
        L = (sp + sm - 2.0 * np.eye(N)) / dx**2
    elif bc == "neumann":
        # Half-cell mirror BC: ghost u[-1]=u[0], u[N]=u[N-1].
        # Boundary rows are [-1, 1, ...] and [..., 1, -1]; symmetric,
        # row sums = 0, all eigenvalues <= 0.  Diagonalised by DCT-II.
        L = np.zeros((N, N))
        for i in range(N):
            L[i, i] = -2.0
            if i > 0:
                L[i, i - 1] = 1.0
            if i < N - 1:
                L[i, i + 1] = 1.0
        L[0, 0] = -1.0
        L[N - 1, N - 1] = -1.0
        L /= dx**2
    else:
        raise ValueError(f"Unknown bc: {bc!r}; expected 'periodic' or 'neumann'")
    return L


def build_heat_propagator(
    N: int,
    dx: float,
    dt: float,
    nu: float,
    bc: str = "periodic",
) -> np.ndarray:
    """Build the dense heat-equation propagator exp(nu * L * dt).

    This is a contraction (NOT unitary): eigenvalues in (0, 1].
    Built once at setup and reused for every time step.
    """
    L = build_laplacian_dense(N, dx, bc=bc)
    P = expm(nu * L * dt)
    return P


def build_heat_propagator_mpo(
    N: int,
    dx: float,
    dt: float,
    nu: float,
    bc: str = "periodic",
    max_bond: int | None = None,
    cutoff: float = 1e-14,
) -> qtn.MatrixProductOperator:
    """Build the heat propagator as a quimb MPO.

    The propagator exp(nu * L * dt) is constructed once in dense form
    and converted to MPO via from_dense.

    Returns the propagator MPO (reusable across time steps).
    """
    q = int(np.log2(N))
    P = build_heat_propagator(N, dx, dt, nu, bc=bc)
    return dense_to_mpo(P, q, max_bond=max_bond, cutoff=cutoff)


# -------------------------------------------------------------------
# Single-step Cole-Hopf TEBD evolution
# -------------------------------------------------------------------


def cole_hopf_step_mps(
    phi_mps: qtn.MatrixProductState,
    propagator_mpo: qtn.MatrixProductOperator,
    phi_norm: float,
    max_bond: int | None = None,
    cutoff: float = 1e-14,
) -> tuple[qtn.MatrixProductState, float, dict]:
    """One Cole-Hopf heat-equation step in MPS form.

    Applies the FIXED propagator MPO to the normalized phi MPS.
    Returns (phi_mps_next, phi_norm_next, metrics).

    The propagator is a contraction, so the MPS norm changes.
    phi_norm tracks the cumulative un-normalized phi magnitude.
    """
    bonds_in = phi_mps.bond_sizes()

    result_mps = phi_mps.gate_with_mpo(
        propagator_mpo, max_bond=max_bond, cutoff=cutoff,
    )

    # The MPS norm after applying the contraction gives the ratio
    # ||P * psi|| where psi was unit-norm.  Track cumulative norm.
    mps_norm = result_mps.norm()
    phi_norm_next = phi_norm * float(np.real(mps_norm))

    # Re-normalize MPS for amplitude encoding
    result_mps /= mps_norm
    bonds_out = result_mps.bond_sizes()

    metrics: dict[str, Any] = {
        "method": "cole_hopf",
        "mps_bond_dims_in": bonds_in,
        "mps_bond_dims_out": bonds_out,
        "phi_norm": phi_norm_next,
        "mps_contraction_factor": float(np.real(mps_norm)),
    }

    return result_mps, phi_norm_next, metrics


# -------------------------------------------------------------------
# Multi-step simulation
# -------------------------------------------------------------------


def run_cole_hopf_simulation(
    u0: np.ndarray,
    x: np.ndarray,
    nu: float,
    dt: float,
    n_steps: int,
    bc: str = "periodic",
    max_bond: int | None = None,
    cutoff: float = 1e-14,
    snapshot_interval: int = 1,
) -> tuple[list[np.ndarray], list[dict]]:
    """Run multi-step Burgers simulation via Cole-Hopf + TEBD.

    1. Transform u0 -> phi0 (once, classical)
    2. Build heat propagator MPO (once, reused every step)
    3. Time-loop: apply propagator to phi MPS (no classical rebuild)
    4. Inverse transform phi -> u at snapshot steps

    Parameters
    ----------
    u0       : initial velocity field, length N = 2^q.
    x        : grid coordinates.
    nu       : kinematic viscosity (must be >= ~1e-3 for float64).
    dt       : time step.
    n_steps  : number of time steps.
    bc       : u-side BC: "periodic" or "dirichlet" (u=0 walls).
               "neumann" is accepted as the equivalent phi-side label
               for the Dirichlet-on-u case.  Internally the heat
               propagator is built in phi-space, so Dirichlet-on-u is
               mapped to Neumann-on-phi (OVERVIEW §4.1 BC mapping).
    max_bond : max MPS bond dimension (None = full rank).
    cutoff   : SVD cutoff for MPO/MPS construction.
    snapshot_interval : save u every this many steps (1 = every step).

    Returns
    -------
    (solutions, metrics_list) — u(x,t) at snapshot times and per-step
    diagnostics.
    """
    N = len(u0)
    q = int(np.log2(N))
    dx = x[1] - x[0]

    # u-side -> phi-side BC mapping (OVERVIEW §4.1).  Dirichlet on u
    # (u(0)=u(L)=0) maps to Neumann on phi (dphi/dx=0), which is what
    # build_heat_propagator_mpo expects.  Mirror the same translation
    # the circuit path does in burgers_cole_hopf_circuit.py:1911.
    phi_bc = "neumann" if bc == "dirichlet" else bc
    if phi_bc not in ("periodic", "neumann"):
        raise NotImplementedError(
            f"bc={bc!r} not supported by run_cole_hopf_simulation; "
            f"u-side supported: 'periodic', 'dirichlet' (or pass "
            f"'neumann' as the equivalent phi-side label)."
        )

    # --- Setup (once) ---

    # Forward transform — auto-center for small nu
    use_centering = _should_center(u0, dx, nu)
    e_mid = 0.0
    if use_centering:
        log_phi, e_mid = cole_hopf_forward_centered(u0, dx, nu)
        psi0 = log_phi_to_normalized_psi(log_phi)
        phi_norm = 1.0  # norm is absorbed into psi via log-sum-exp
        print(
            f"[cole_hopf] using centered exponent (e_mid={e_mid:.2f}, "
            f"nu={nu:.1e}, log_phi range="
            f"[{log_phi.min():.1f},{log_phi.max():.1f}])",
            file=sys.stderr, flush=True,
        )
    else:
        phi0 = cole_hopf_forward(u0, dx, nu)
        phi_norm = np.linalg.norm(phi0)
        if phi_norm < 1e-15:
            raise ValueError(
                "phi0 norm is near zero; check IC and nu"
            )
        psi0 = phi0 / phi_norm
        print(
            f"[cole_hopf] standard exponent (nu={nu:.1e})",
            file=sys.stderr, flush=True,
        )
    phi_mps = state_to_mps(psi0, q, cutoff=cutoff)

    # Build propagator MPO ONCE (phi-side BC).
    propagator_mpo = build_heat_propagator_mpo(
        N, dx, dt, nu, bc=phi_bc,
        max_bond=max_bond, cutoff=cutoff,
    )

    # --- Snapshot the initial condition ---
    solutions = [u0.copy()]
    metrics_list: list[dict] = []

    # --- Time loop (no classical rebuild) ---
    current_mps = phi_mps
    current_phi_norm = phi_norm

    for step in range(n_steps):
        current_mps, current_phi_norm, metrics = cole_hopf_step_mps(
            current_mps, propagator_mpo, current_phi_norm,
            max_bond=max_bond, cutoff=cutoff,
        )
        metrics["step"] = step + 1
        metrics["propagator_mpo_bonds"] = propagator_mpo.bond_sizes()
        metrics_list.append(metrics)

        # Snapshot: inverse transform phi -> u
        if (step + 1) % snapshot_interval == 0 or step == n_steps - 1:
            psi_vec = mps_to_vector(current_mps).real
            if use_centering:
                # log domain: scalars cancel under d/dx
                u_vec = cole_hopf_inverse(
                    psi_vec, dx, nu,
                )
            else:
                phi_vec = psi_vec * current_phi_norm
                u_vec = cole_hopf_inverse(phi_vec, dx, nu)
            solutions.append(u_vec)

            if not np.all(np.isfinite(u_vec)):
                print(
                    f"[cole_hopf] diverged at step {step + 1}/{n_steps}",
                    file=sys.stderr, flush=True,
                )
                remaining = (n_steps - step - 1) // snapshot_interval
                nan_fill = np.full_like(u0, np.nan)
                solutions.extend([nan_fill] * remaining)
                break

    return solutions, metrics_list


# -------------------------------------------------------------------
# Smoke test and validation
# -------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    print("=== F10 Cole-Hopf + TEBD — Smoke Test ===\n")

    # --- A.1: Round-trip test ---
    # Round-trip error is dominated by FD discretization in the
    # inverse (central-difference gradient).  Converges as O(dx^2).
    print("--- A.1: Cole-Hopf round-trip ---\n")
    for nu_test in (1e-1, 1e-2):
        print(f"  nu={nu_test:.0e}:")
        for q in (4, 5, 6, 8):
            N = 2**q
            dx = 1.0 / N
            x = np.linspace(0, 1, N, endpoint=False)

            u_orig = np.sin(2.0 * np.pi * x)
            phi = cole_hopf_forward(u_orig, dx, nu_test)
            u_roundtrip = cole_hopf_inverse(phi, dx, nu_test)
            err = np.max(np.abs(u_orig - u_roundtrip))
            ratio = phi.max() / max(phi.min(), 1e-300)
            print(
                f"    q={q} (N={N:>4d})  err={err:.2e}  "
                f"phi ratio={ratio:.1e}"
            )

    # --- A.2: Heat equation vs analytic Fourier decay ---
    print("\n--- A.2: Heat propagator vs analytic Fourier decay ---\n")
    q = 5
    N = 2**q
    dx = 1.0 / N
    x = np.linspace(0, 1, N, endpoint=False)
    nu_test = 1e-2
    dt = 1e-3
    n_steps = 50  # T = 0.05

    # Initial phi: 1 + 0.5*cos(2*pi*x) — smooth, positive, periodic
    phi_init = 1.0 + 0.5 * np.cos(2.0 * np.pi * x)

    # Analytic: heat equation preserves Fourier structure.
    # phi(x,t) = 1 + 0.5*cos(2*pi*x) * exp(-nu*(2*pi)^2*t)
    T_final = n_steps * dt
    decay = np.exp(-nu_test * (2.0 * np.pi) ** 2 * T_final)
    phi_analytic = 1.0 + 0.5 * decay * np.cos(2.0 * np.pi * x)

    # Numerical: apply propagator n_steps times
    L = build_laplacian_dense(N, dx, bc="periodic")
    P = build_heat_propagator(N, dx, dt, nu_test, bc="periodic")
    phi_numerical = phi_init.copy()
    for _ in range(n_steps):
        phi_numerical = P @ phi_numerical

    err_dense = np.max(np.abs(phi_numerical - phi_analytic))
    print(f"  Dense propagator error (q={q}, T={T_final}): {err_dense:.2e}")

    # Same via MPO path
    phi_norm = np.linalg.norm(phi_init)
    psi_mps = state_to_mps(phi_init / phi_norm, q, cutoff=1e-14)
    prop_mpo = build_heat_propagator_mpo(
        N, dx, dt, nu_test, bc="periodic", cutoff=1e-14,
    )
    print(f"  Propagator MPO bonds: {prop_mpo.bond_sizes()}")

    current_mps = psi_mps
    current_norm = phi_norm
    for _ in range(n_steps):
        current_mps, current_norm, _ = cole_hopf_step_mps(
            current_mps, prop_mpo, current_norm,
        )
    phi_mpo = mps_to_vector(current_mps).real * current_norm
    err_mpo = np.max(np.abs(phi_mpo - phi_analytic))
    print(f"  MPO path error (q={q}, T={T_final}): {err_mpo:.2e}")

    # --- A.3: Full Cole-Hopf pipeline vs classical Burgers ---
    print("\n--- A.3: Cole-Hopf pipeline vs classical shift-Euler ---\n")

    from burgers_trotter import shift_euler_step

    for q in (3, 4, 5):
        N = 2**q
        dx = 1.0 / N
        x = np.linspace(0, 1, N, endpoint=False)
        nu_ch = 1e-2  # Safe for Cole-Hopf
        dt_ch = 1e-4
        n_steps_ch = 100

        u0 = np.sin(2.0 * np.pi * x)

        # Cole-Hopf TEBD
        sols_ch, mets_ch = run_cole_hopf_simulation(
            u0, x, nu_ch, dt_ch, n_steps_ch,
        )

        # Classical shift-Euler
        sols_shift = [u0.copy()]
        u_s = u0.copy()
        for step in range(n_steps_ch):
            u_s, _ = shift_euler_step(u_s, dx, dt_ch, nu_ch)
            sols_shift.append(u_s.copy())

        # Compare at final time
        err = np.max(np.abs(sols_ch[-1] - sols_shift[-1]))
        print(
            f"  q={q} (N={N:>3d})  T={n_steps_ch*dt_ch:.4f}  "
            f"cole_hopf vs shift err={err:.2e}"
        )

    # --- Scalability ---
    print("\n--- Scalability (single step) ---\n")
    import time

    for q in (6, 7, 8, 10):
        N = 2**q
        dx = 1.0 / N
        x = np.linspace(0, 1, N, endpoint=False)
        nu_s = 1e-2
        dt_s = 1e-4
        u0 = np.sin(2.0 * np.pi * x)

        t0 = time.time()
        # Setup (one-time cost)
        phi0 = cole_hopf_forward(u0, dx, nu_s)
        pn = np.linalg.norm(phi0)
        psi = state_to_mps(phi0 / pn, q, cutoff=1e-14)
        pmpo = build_heat_propagator_mpo(
            N, dx, dt_s, nu_s, cutoff=1e-14,
        )
        t_setup = time.time() - t0

        t0 = time.time()
        # Single step
        psi2, pn2, m = cole_hopf_step_mps(psi, pmpo, pn)
        t_step = time.time() - t0

        print(
            f"  q={q:>2d} (N={N:>4d})  "
            f"setup={t_setup:.3f}s  step={t_step:.4f}s  "
            f"prop_bonds={pmpo.bond_sizes()}  "
            f"mps_out={m['mps_bond_dims_out']}"
        )

    print("\nDone.")
