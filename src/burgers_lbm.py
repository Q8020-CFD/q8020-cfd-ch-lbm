"""Classical D1Q3 Lattice Boltzmann solver for 1-D viscous Burgers (F11).

Encodes mesoscopic distribution functions f_i(x,t) on a D1Q3 stencil
with BGK collision and periodic/bounce-back streaming.  Recovers the
macroscopic velocity u(x,t) via moment reconstruction.

Reference: Quartey & Zhong, "Beyond the Simulator: A Practical
Demonstration of Quantum Lattice Boltzmann Methods on IBM Quantum,"
RPI / IBM, Nov 2025.
"""

from __future__ import annotations

import sys
from typing import Callable

import numpy as np


# ── D1Q3 constants ────────────────────────────────────────────────────

# Lattice velocities: c = (-1, 0, +1)
# Weights: equal 1/3 for athermal Burgers D1Q3
W = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])


# ── Equilibrium ───────────────────────────────────────────────────────


def equilibrium(u: np.ndarray, rho: np.ndarray | None = None) -> np.ndarray:
    """D1Q3 equilibrium distributions for Burgers (density-conserving).

    Returns shape (3, N) array: [f_{-1}^eq, f_0^eq, f_{+1}^eq].

    Uses the form that exactly conserves both density and momentum:
      f_0^eq   = rho * (1 - u^2)
      f_{+1}^eq = rho * (u^2 + u) / 2
      f_{-1}^eq = rho * (u^2 - u) / 2

    Satisfies: sum(f_i^eq) = rho, sum(c_i * f_i^eq) = rho*u.
    Chapman-Enskog recovers Burgers via Pi_eq = rho*u^2.
    Valid for |u| < 1 (lattice Mach < 1).
    """
    if rho is None:
        rho = np.ones_like(u)
    u2 = u * u
    f_eq = np.zeros((3, len(u)))
    f_eq[0] = rho * (u2 - u) / 2.0   # c = -1
    f_eq[1] = rho * (1.0 - u2)        # c = 0
    f_eq[2] = rho * (u2 + u) / 2.0   # c = +1
    return f_eq


def density(f: np.ndarray) -> np.ndarray:
    """Macroscopic density: rho = sum_i f_i."""
    return f[0] + f[1] + f[2]


def momentum(f: np.ndarray) -> np.ndarray:
    """Macroscopic momentum: rho*u = -f_{-1} + f_{+1}."""
    return -f[0] + f[2]


def velocity(f: np.ndarray) -> np.ndarray:
    """Macroscopic velocity: u = (-f_{-1} + f_{+1}) / rho."""
    rho = density(f)
    rho = np.where(np.abs(rho) < 1e-15, 1.0, rho)
    return momentum(f) / rho


# ── Collision ─────────────────────────────────────────────────────────


def tau_from_nu(nu: float, dx: float, dt: float) -> float:
    """BGK relaxation time from kinematic viscosity (Eq. 5).

    nu = (tau - 1/2) * dx^2 / dt  =>  tau = nu*dt/dx^2 + 1/2
    """
    return nu * dt / (dx * dx) + 0.5


def collide_bgk(
    f: np.ndarray, tau: float, u_local: np.ndarray | None = None
) -> np.ndarray:
    """BGK collision: f* = f - (1/tau)(f - f^eq).

    If u_local is None, compute from f via moment reconstruction.
    Returns post-collision distributions (3, N).
    """
    if u_local is None:
        u_local = velocity(f)
    rho_local = density(f)
    f_eq = equilibrium(u_local, rho_local)
    return f - (1.0 / tau) * (f - f_eq)


def collision_matrix(
    u_local: np.ndarray, tau: float
) -> np.ndarray:
    """Build the 3N x 3N block-diagonal collision matrix M.

    M acts on the flattened vector [f_{-1}(0..N-1), f_0(0..N-1),
    f_{+1}(0..N-1)].  Each 3x3 block at site j is:
      M_j = (1 - 1/tau)*I + (1/tau)*E_j
    where E_j maps any f_i(j) to f_i^eq(j) (rank-1 projection onto eq).

    Actually for the linearized collision about a given u(j), the 3x3
    matrix is:
      f_i*(j) = sum_k M_{ik}(j) * f_k(j)
    with M_{ik} = (1-1/tau)*delta_{ik} + (1/tau)*(df_i^eq/df_k).
    Since f_i^eq depends on u which depends on f via moments, the
    correct linearization is non-trivial.

    For Option A (spec §3.1), we use the full nonlinear collision
    classically pre-computed. This function builds the *affine* collision
    operator directly as a dense 3N x 3N matrix suitable for a unitary
    embedding (circuit path).

    Simplified: for fixed u(j), the collision is:
      f_i*(j) = (1-1/tau)*f_i(j) + (1/tau)*f_i^eq(u(j))
    This is affine in f (not linear) because f_i^eq depends on u(j)
    which is a ratio of f's. For the circuit (Option A), we just apply
    the full nonlinear step classically each timestep and embed the
    resulting 3N->3N map as a unitary via block encoding.

    Returns the 3Nx3N collision map for the CURRENT state.
    """
    N = len(u_local)
    omega = 1.0 / tau
    dim = 3 * N
    # Linear part of BGK: (1-omega)*I.  The equilibrium shift is
    # handled nonlinearly by the circuit path (see qlbm_circuit).
    M = (1.0 - omega) * np.eye(dim)
    return M


# ── Streaming ─────────────────────────────────────────────────────────


def stream(f: np.ndarray, bc: str = "periodic") -> np.ndarray:
    """Streaming step: shift each population by its lattice velocity.

    f[0] (c=-1): shift left by 1
    f[1] (c= 0): no shift
    f[2] (c=+1): shift right by 1

    bc='periodic': cyclic wrap.
    bc='dirichlet': bounce-back at walls (f_out -> f_in).
    """
    f_new = np.empty_like(f)

    if bc == "periodic":
        f_new[0] = np.roll(f[0], -1)  # shift left
        f_new[1] = f[1].copy()         # no shift
        f_new[2] = np.roll(f[2], +1)  # shift right
    elif bc == "dirichlet":
        # Shift with bounce-back at walls
        # f_{-1}: shift left (interior)
        f_new[0, :-1] = f[0, 1:]
        f_new[0, -1] = f[2, -1]  # bounce-back at right wall

        # f_0: no shift
        f_new[1] = f[1].copy()

        # f_{+1}: shift right (interior)
        f_new[2, 1:] = f[2, :-1]
        f_new[2, 0] = f[0, 0]  # bounce-back at left wall
    else:
        raise ValueError(f"Unknown bc: {bc}")

    return f_new


# ── Full LBM timestep ─────────────────────────────────────────────────


def lbm_step(
    f: np.ndarray, tau: float, bc: str = "periodic"
) -> np.ndarray:
    """One LBM timestep: collide then stream."""
    f_post = collide_bgk(f, tau)
    return stream(f_post, bc=bc)


# ── Multi-step simulation ─────────────────────────────────────────────


def run_lbm_simulation(
    u0: np.ndarray,
    x: np.ndarray,
    nu: float,
    dt: float,
    n_steps: int,
    bc: str = "periodic",
    source_fn: Callable | None = None,
) -> tuple[list[np.ndarray], list[dict]]:
    """Run classical LBM for n_steps, returning velocity at every step.

    In LBM, dt is locked to dx (streaming = one lattice site per step).
    The caller's dt*n_steps defines T_end; we recompute the LBM step
    count as n_steps_lbm = round(T_end / dx).  Returned solutions are
    then interpolated back to the caller's time grid so that
    solutions[i] corresponds to time i*dt.

    Returns
    -------
    (solutions, metrics) where solutions[i] is u at step i (length
    n_steps+1 including IC), metrics is list of per-step dicts.
    """
    N = len(u0)
    dx = x[1] - x[0]

    # LBM uses dt_lbm = dx (one lattice site per streaming step).
    dt_lbm = dx
    T_end = dt * n_steps
    n_steps_lbm = max(1, round(T_end / dt_lbm))
    tau = tau_from_nu(nu, dx, dt_lbm)

    if tau <= 0.5:
        print(
            f"[lbm] WARNING: tau={tau:.4f} <= 0.5 (unstable). "
            f"Increase dx or decrease nu.",
            file=sys.stderr, flush=True,
        )

    print(
        f"[lbm] N={N} tau={tau:.4f} nu={nu:.2e} "
        f"dx={dx:.4e} dt_lbm={dt_lbm:.4e} "
        f"n_steps_lbm={n_steps_lbm} (caller: dt={dt:.4e} n={n_steps})",
        file=sys.stderr, flush=True,
    )

    # Initialize from equilibrium at u0
    f = equilibrium(u0)

    # Run LBM at its native dt_lbm = dx, storing all LBM steps.
    lbm_solutions = [u0.copy()]
    metrics: list[dict] = []

    for step in range(1, n_steps_lbm + 1):
        f = lbm_step(f, tau, bc=bc)

        # Source term (additive forcing in collision)
        if source_fn is not None:
            t = step * dt_lbm
            g = source_fn(x, t)
            for i in range(3):
                f[i] += W[i] * g * dt_lbm

        u_cur = velocity(f)
        rho_cur = density(f)

        step_met = {
            "step": step,
            "rho_max": float(np.max(rho_cur)),
            "rho_min": float(np.min(rho_cur)),
            "u_max": float(np.max(np.abs(u_cur))),
            "mass": float(np.sum(rho_cur)),
        }
        metrics.append(step_met)
        lbm_solutions.append(u_cur.copy())

    # Map LBM solutions onto the caller's time grid (n_steps+1 points).
    # LBM time grid: t_lbm[k] = k * dt_lbm, k = 0..n_steps_lbm
    # Caller grid:   t_cal[j] = j * dt,      j = 0..n_steps
    # Use nearest-neighbor: for each caller time, pick the closest LBM
    # snapshot.
    solutions = [u0.copy()]
    for j in range(1, n_steps + 1):
        t_target = j * dt
        k = min(
            round(t_target / dt_lbm),
            n_steps_lbm,
        )
        solutions.append(lbm_solutions[k].copy())

    return solutions, metrics


# ── Helpers for circuit path ──────────────────────────────────────────


def flatten_distributions(f: np.ndarray) -> np.ndarray:
    """Flatten (3, N) distributions into 4N vector for circuit encoding.

    Layout: [f_{-1}(0..N-1), f_0(0..N-1), f_{+1}(0..N-1), zeros(N)]
    The 4th slot (|11> velocity index) is padded with zeros.
    """
    N = f.shape[1]
    vec = np.zeros(4 * N)
    vec[0:N] = f[0]        # |00> = f_{-1}
    vec[N:2*N] = f[1]      # |01> = f_0
    vec[2*N:3*N] = f[2]    # |10> = f_{+1}
    # vec[3*N:4*N] = 0     # |11> = unused
    return vec


def unflatten_distributions(vec: np.ndarray, N: int) -> np.ndarray:
    """Unflatten 4N vector back to (3, N) distributions."""
    f = np.zeros((3, N))
    f[0] = vec[0:N]
    f[1] = vec[N:2*N]
    f[2] = vec[2*N:3*N]
    return f


def normalize_for_circuit(f: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalize distribution vector for quantum state preparation.

    Returns (normalized_4N_vector, norm).
    """
    vec = flatten_distributions(f)
    norm = np.linalg.norm(vec)
    if norm < 1e-15:
        return vec, 0.0
    return vec / norm, norm
