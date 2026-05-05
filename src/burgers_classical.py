"""Classical solver for the 1D viscous Burgers equation.

PDE:  du/dt + (1/2) d(u*u)/dx - nu * d2u/dx2 = g(x, t)

The production solver (solve_burgers) uses forward-time explicit Euler
with shift-operator finite differences (via compute_rhs_shift from
burgers_nonlinear), matching the verification approach in Murali et al.
AIAA 2026.

The functions gradient_central, laplacian_central, euler_step,
build_gradient_matrix, and build_laplacian_matrix are legacy
implementations retained for the module's __main__ smoke test and for
reference against quimb MPO validation (UCAN plan §3.2).  They are
not used by the solver.
"""

import numpy as np

from burgers_nonlinear import compute_rhs_shift


def initial_condition_sine(x: np.ndarray) -> np.ndarray:
    """Simple sine wave IC: u0 = sin(2*pi*x)."""
    return np.sin(2.0 * np.pi * x)


def initial_condition_multimode(
    x: np.ndarray,
    n_modes: int = 6,
    seed: int = 42,
    alpha: float = 1.0,
) -> np.ndarray:
    """Multi-mode Fourier IC with random phases.

    u0(x) = sum_{k=1}^{n_modes} c_k * sin(k*pi*x)
    with c_k = rng.normal() * k^{-alpha}.  The basis functions sin(k*pi*x)
    are the Dirichlet eigenfunctions on [0, 1] and vanish at both
    endpoints for any integer k >= 1, so u(0) = u(1) = 0 exactly.
    Randomness enters through the signed amplitudes c_k (Gaussian,
    seeded for reproducibility), not through phase offsets.

    This is NOT Burgers turbulence.  Burgers turbulence ("Burgulence")
    is a statistical object requiring an ensemble of realizations with
    prescribed IC statistics (or stochastic forcing), analyzed via
    ensemble-averaged spectra, structure functions, and PDFs.  This
    function returns one deterministic field -- a complex initial
    condition useful for demonstrating multi-shock dynamics, nothing
    more.  See F11 in the implementation plan for a genuine Burgulence
    study.

    The final profile is normalized so max|u0| = 1 for consistent CFL
    behavior.

    Parameters
    ----------
    x        : grid coordinates in [0, 1].
    n_modes  : number of Fourier modes (k = 1..n_modes).  Keep <= N/4 to
               avoid aliasing on a grid of N points.
    seed     : RNG seed for reproducibility.
    alpha    : velocity spectrum exponent A_k ~ k^{-alpha}.
    """
    N = len(x)
    nyquist = N // 2 - 1
    if n_modes > nyquist:
        raise ValueError(
            f"n_modes={n_modes} exceeds Nyquist limit {nyquist} for N={N} "
            f"grid points.  Reduce --ic-modes or increase --q."
        )

    rng = np.random.default_rng(seed)
    u = np.zeros_like(x)
    for k in range(1, n_modes + 1):
        c_k = rng.normal() * k ** (-alpha)
        u += c_k * np.sin(np.pi * k * x)
    m = np.max(np.abs(u))
    if m == 0:
        raise ValueError(
            f"Multimode IC collapsed to zero (seed={seed}, n_modes={n_modes}, "
            f"alpha={alpha}).  Pick a different seed."
        )
    u /= m
    return u


def source_term_sine(
    x: np.ndarray, t: float
) -> np.ndarray:
    """Source g = sin(2*pi*x) * cos(2*pi*t) (paper Sec. III.A)."""
    return np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * t)


def gradient_central(u: np.ndarray, dx: float) -> np.ndarray:
    """Central difference first derivative, one-sided at boundaries.

    NOTE: Legacy — not used by the solver (shift_matrix path supersedes it).
    Retained for the __main__ smoke test.
    """
    N = len(u)
    du = np.zeros(N)
    du[1:-1] = (u[2:] - u[:-2]) / (2.0 * dx)
    du[0] = (u[1] - u[0]) / dx
    du[-1] = (u[-1] - u[-2]) / dx
    return du


def laplacian_central(u: np.ndarray, dx: float) -> np.ndarray:
    """Central difference second derivative, one-sided at boundaries.

    NOTE: Legacy — not used by the solver (shift_matrix path supersedes it).
    Retained for the __main__ smoke test.
    """
    N = len(u)
    d2u = np.zeros(N)
    d2u[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx * dx)
    d2u[0] = (u[1] - 2.0 * u[0]) / (dx * dx)
    d2u[-1] = (u[-2] - 2.0 * u[-1]) / (dx * dx)
    return d2u


def euler_step(
    u: np.ndarray,
    dx: float,
    dt: float,
    nu: float,
    g: np.ndarray,
) -> np.ndarray:
    """One explicit Euler step (Eq. 15 in paper).

    u(t+dt) = u(t) + dt * [nu * laplacian(u) - u * gradient(u) + g]

    NOTE: Legacy — not used by the solver (solve_burgers uses
    compute_rhs_shift instead).  Retained for the __main__ smoke test.
    """
    grad_u = gradient_central(u, dx)
    lap_u = laplacian_central(u, dx)
    rhs = nu * lap_u - u * grad_u + g
    return u + dt * rhs


def solve_burgers(
    u0: np.ndarray,
    x: np.ndarray,
    nu: float,
    dt: float,
    n_steps: int,
    source_fn=None,
    bc: str = "periodic",
) -> list[np.ndarray]:
    """Run the full FTCS simulation, returning solution at each step.

    Uses the same shift-operator finite-difference stencil as the quantum
    path (via burgers_nonlinear.compute_rhs_shift) so the classical
    reference and quantum solution see identical boundary treatment.

    Parameters
    ----------
    u0 : initial velocity array (length N)
    x : grid coordinates (length N)
    nu : kinematic viscosity
    dt : time step
    n_steps : number of Euler steps
    source_fn : callable(x, t) -> np.ndarray, or None for no source
    bc : 'periodic' (shift wraps) or 'dirichlet' (u=0 ghost nodes)

    Returns
    -------
    List of solution arrays [u0, u1, ..., u_{n_steps}].
    """

    dx = x[1] - x[0]
    solutions = [u0.copy()]
    u = u0.copy()

    for step in range(n_steps):
        t = step * dt
        g = source_fn(x, t) if source_fn is not None else np.zeros_like(u)
        with np.errstate(over="ignore", invalid="ignore"):
            u = u + dt * compute_rhs_shift(u, dx, nu, g, bc=bc)
        if not np.all(np.isfinite(u)):
            import sys
            print(
                f"[burgers] FTCS blowup at step {step + 1}/{n_steps} "
                f"(t={t + dt:.4e}); padding remaining steps with NaN",
                file=sys.stderr, flush=True,
            )
            nan_fill = np.full_like(u, np.nan)
            solutions.extend([nan_fill] * (n_steps - step))
            break
        solutions.append(u.copy())

    return solutions


def _godunov_flux_burgers(
    u_L: np.ndarray, u_R: np.ndarray
) -> np.ndarray:
    """Vectorized exact Godunov flux for f(u) = u^2/2.

    Solves the Riemann problem at each interface:
      shock (u_L >= u_R): upwind by Rankine-Hugoniot speed
      rarefaction (u_L < u_R): upwind, with f=0 at sonic point
    """
    f_L = 0.5 * u_L**2
    f_R = 0.5 * u_R**2
    s = 0.5 * (u_L + u_R)

    shock = u_L >= u_R
    rar = ~shock

    flux = np.zeros_like(u_L)
    flux = np.where(shock & (s >= 0), f_L, flux)
    flux = np.where(shock & (s < 0), f_R, flux)
    flux = np.where(rar & (u_L >= 0), f_L, flux)
    flux = np.where(rar & (u_R <= 0), f_R, flux)
    # sonic rarefaction (u_L < 0 < u_R): f(0)=0, already 0
    return flux


def solve_burgers_godunov(
    u0: np.ndarray,
    x: np.ndarray,
    nu: float,
    dt: float,
    n_steps: int,
    source_fn=None,
    bc: str = "periodic",
) -> list[np.ndarray]:
    """Godunov + explicit diffusion for 1D viscous Burgers.

    Conservative shock-capturing scheme. Unconditionally stable for
    the advective part (exact Riemann solver). Diffusion is explicit
    central-difference (stable when nu*dt/dx^2 < 0.5).

    Same signature and return convention as solve_burgers (FTCS).
    """
    dx = x[1] - x[0]
    solutions = [u0.copy()]
    u = u0.copy()

    for step in range(n_steps):
        t = step * dt
        g = (
            source_fn(x, t)
            if source_fn is not None
            else None
        )

        # Ghost cells for flux computation
        if bc == "periodic":
            u_ext = np.concatenate(([u[-1]], u, [u[0]]))
        else:
            u_ext = np.concatenate(([0.0], u, [0.0]))

        # Godunov flux at N+1 interfaces
        F = _godunov_flux_burgers(u_ext[:-1], u_ext[1:])

        # Conservative advection update
        u_new = u - (dt / dx) * (F[1:] - F[:-1])

        # Explicit central-difference diffusion
        if bc == "periodic":
            lap = (
                np.roll(u, -1) + np.roll(u, 1) - 2 * u
            ) / dx**2
        else:
            u_ext_d = np.concatenate(([0.0], u, [0.0]))
            lap = (
                u_ext_d[2:] + u_ext_d[:-2] - 2 * u
            ) / dx**2
        u_new += dt * nu * lap

        if g is not None:
            u_new += dt * g

        if bc == "dirichlet":
            u_new[0] = 0.0
            u_new[-1] = 0.0

        u = u_new
        if not np.all(np.isfinite(u)):
            import sys

            print(
                f"[burgers] Godunov blowup at step {step + 1}"
                f"/{n_steps} (t={t + dt:.4e}); padding with NaN",
                file=sys.stderr,
                flush=True,
            )
            nan_fill = np.full_like(u, np.nan)
            solutions.extend([nan_fill] * (n_steps - step))
            break
        solutions.append(u.copy())

    return solutions


def build_gradient_matrix(N: int, dx: float) -> np.ndarray:
    """Full NxN Jacobian matrix for central-difference gradient.

    Interior: centered. Boundaries: one-sided.
    This is the dense matrix form used by quimb to construct MPOs.

    NOTE: Not used by the solver (shift_matrix path supersedes it).
    Kept for reference against quimb MPO validation (UCAN plan §3.2)
    and used by analysis/test_mpo.py.
    """
    J = np.zeros((N, N))
    for i in range(1, N - 1):
        J[i, i + 1] = 1.0 / (2.0 * dx)
        J[i, i - 1] = -1.0 / (2.0 * dx)
    J[0, 1] = 1.0 / dx
    J[0, 0] = -1.0 / dx
    J[-1, -1] = 1.0 / dx
    J[-1, -2] = -1.0 / dx
    return J


def build_laplacian_matrix(N: int, dx: float) -> np.ndarray:
    """Full NxN Hessian matrix for central-difference Laplacian.

    Interior: centered. Boundaries: one-sided (Dirichlet u=0 ghost).

    NOTE: Not used by the solver (shift_matrix path supersedes it).
    Kept for reference against quimb MPO validation (UCAN plan §3.2)
    and used by analysis/test_mpo.py.
    """
    H = np.zeros((N, N))
    dx2 = dx * dx
    for i in range(1, N - 1):
        H[i, i - 1] = 1.0 / dx2
        H[i, i] = -2.0 / dx2
        H[i, i + 1] = 1.0 / dx2
    H[0, 0] = -2.0 / dx2
    H[0, 1] = 1.0 / dx2
    H[-1, -2] = 1.0 / dx2
    H[-1, -1] = -2.0 / dx2
    return H


if __name__ == "__main__":  # smoke test only — not the solver entry point
    import matplotlib.pyplot as plt

    N = 200
    x = np.linspace(0, 1, N, endpoint=False)
    dx = x[1] - x[0]
    nu = 1e-4
    cfl = 0.1
    dt = cfl * dx
    n_steps = int(0.1 / dt)

    u0 = initial_condition_sine(x)
    solutions = solve_burgers(
        u0, x, nu, dt, n_steps, source_fn=source_term_sine
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    step_indices = np.linspace(0, n_steps, 5, dtype=int)
    for idx in step_indices:
        t_val = idx * dt
        ax.plot(x, solutions[idx], label=f"t = {t_val:.3e}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("u (m/s)")
    ax.set_title("1D Burgers FTCS — sine wave IC")
    ax.legend()
    plt.tight_layout()
    plt.savefig("/tmp/burgers_classical.png", dpi=150)
    print(f"Saved plot to /tmp/burgers_classical.png")
    print(f"Final max |u| = {np.max(np.abs(solutions[-1])):.6f}")
