"""Classical shift-operator finite-difference primitives for Burgers.

Consolidates the classical FD building blocks shared by every solver:
- `shift_matrix`      : S+/S- permutation matrices (the FD stencils)
- `compute_rhs_shift` : central-difference forward-Euler RHS
- `shift_euler_step`  : one explicit Euler step (the `shift` method)
- `compute_error`     : normalized-L2 error metric for comparisons

The stencils match the quantum circuit operators so classical and quantum
runs use the same discretization:
- Gradient:   du/dx   ~ (S+ - S-) / (2 dx)
- Laplacian:  d2u/dx2 ~ (S+ + S- - 2I) / dx^2

At each time step the classical update is (§V.C Eq. 15):
  u(t+dt) = u(t) + dt [nu*u_xx - u*u_x + g]

All routines operate on the physical (un-normalized) velocity field.
"""

import numpy as np


def shift_matrix(N: int, direction: int = 1, bc: str = "periodic") -> np.ndarray:
    """Classical NxN shift (permutation) matrix.

    direction=+1: S+ matrix, maps |i> -> |i+1 mod N>
    direction=-1: S- matrix, maps |i> -> |i-1 mod N>

    bc='periodic': wraps around (S[0, N-1] = 1 for S+, etc.)
    bc='dirichlet': no wrapping; boundary rows that would wrap are zeroed,
        equivalent to u=0 ghost nodes outside the domain.
    """
    S = np.zeros((N, N), dtype=float)
    for i in range(N):
        j = (i + direction) % N
        S[j, i] = 1.0
    if bc == "dirichlet":
        # Zero the wrapping entries
        if direction == 1:
            S[0, N - 1] = 0.0   # S+ would wrap |N-1> -> |0>
        else:
            S[N - 1, 0] = 0.0   # S- would wrap |0> -> |N-1>
    return S


def compute_rhs_shift(
    u: np.ndarray,
    dx: float,
    nu: float,
    g: np.ndarray | None = None,
    bc: str = "periodic",
) -> np.ndarray:
    """Classical Euler RHS using shift-operator FD.

    RHS = ν∇²u - u·∇u + g

    Consistent with the quantum circuit operators:
    - ∇u = -(S+ - S-)u / (2δx)   [central difference]
    - ∇²u = (S+ + S- - 2I)u / δx²

    bc='periodic': shift operators wrap mod N (default).
    bc='dirichlet': boundary wrapping zeroed (u=0 ghost nodes).
    """
    N = len(u)
    sp = shift_matrix(N, +1, bc=bc)
    sm = shift_matrix(N, -1, bc=bc)

    # Sign convention: the code's S+ maps |i⟩→|i+1⟩ column-wise, so
    # (S+ u)_j = u_{j-1}.  The standard central-difference gradient
    # (u_{j+1} - u_{j-1})/(2dx) is therefore -(S+ - S-)u/(2dx).
    # Paper Eq. 9 writes (S+ - S-)/(2dx) without the minus because it
    # uses the opposite S+ convention.  The leading minus here is correct.
    grad_u = -(sp - sm) @ u / (2 * dx)
    lap_u = (sp + sm - 2 * np.eye(N)) @ u / dx**2

    rhs = nu * lap_u - u * grad_u
    if g is not None:
        rhs += g

    # Dirichlet: u is prescribed at boundaries, so du/dt = 0 there.
    if bc == "dirichlet":
        rhs[0] = 0.0
        rhs[-1] = 0.0

    return rhs


def shift_euler_step(
    u: np.ndarray,
    dx: float,
    dt: float,
    nu: float,
    g: np.ndarray | None = None,
    bc: str = "periodic",
) -> tuple[np.ndarray, dict]:
    """One explicit Euler step using shift-operator FD.

    Baseline for comparison with quantum methods.
    """
    rhs = compute_rhs_shift(u, dx, nu, g, bc=bc)
    return u + dt * rhs, {}


def compute_error(
    u_quantum: np.ndarray,
    u_classical: np.ndarray,
) -> float:
    """Normalized L2 error: ||u_q - u_c||₂ / ||u_c||₂  (paper's ε metric)."""
    if not np.all(np.isfinite(u_classical)):
        return float("nan")
    norm_c = np.linalg.norm(u_classical)
    if norm_c < 1e-15:
        return 0.0
    return float(np.linalg.norm(u_quantum - u_classical) / norm_c)
