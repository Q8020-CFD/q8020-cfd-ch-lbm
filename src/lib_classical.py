"""Classical solver for the 1D viscous Burgers equation.

PDE:  du/dt + (1/2) d(u*u)/dx - nu * d2u/dx2 = g(x, t)

The production solver (solve_burgers) uses forward-time explicit Euler
with shift-operator finite differences (via compute_rhs_shift from
lib_fd).  The form of the update matches Gopalakrishnan Meena
et al. AIAA-2026 §V.C Eq. 15, but here the spatial operators are plain
shift-matrix FD (not quimb MPS/MPO as in the paper).  This serves as
our internal classical reference; the paper's actual §V.C MPS/MPO
pipeline can be used as an external cross-check.
"""

import sys

import numpy as np

from lib_fd import compute_rhs_shift


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


def initial_condition_gaussian(
    x: np.ndarray,
    amplitude: float = 1.0,
    center: float = 0.5,
    sigma: float = 0.1,
) -> np.ndarray:
    """Localized Gaussian pulse: u0(x) = A * exp(-((x - x0) / sigma)^2).

    Useful for shock-formation demos from a single-lobe disturbance.
    No closed-form Cole-Hopf analytic reference (unlike
    ``cole_hopf_exact``) -- under Cole-Hopf the integral becomes an
    erf and phi_0 has no clean heat-equation evolution.  Pairs with
    FTCS as the classical reference.

    Parameters
    ----------
    x         : grid coordinates.
    amplitude : peak velocity.  LBM methods (qlbm*) typically need
                |u| < ~0.5 for stable D1Q3; --ic-amplitude is the
                canonical knob.
    center    : pulse centre x0 in the domain.
    sigma     : Gaussian width.  Pick small enough that u(boundary)
                is negligible under --bc dirichlet, else the
                discontinuity at x=0 / x=L radiates spurious shocks.
    """
    return amplitude * np.exp(-((x - center) / sigma) ** 2)


def source_term_sine(
    x: np.ndarray, t: float
) -> np.ndarray:
    """Source g = sin(2*pi*x) * cos(2*pi*t) (paper Sec. III.A)."""
    return np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * t)


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
    path (via lib_fd.compute_rhs_shift) so the classical
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


# ── Resolved (grid-independent) FTCS reference ────────────────────────
#
# The quantum methods run on N = 2^q nodes, which is far too coarse to
# serve as a converged classical "truth" at small nu.  These helpers run
# FTCS on a refined grid of >= min_points nodes, chosen so the quantum
# nodes are an *exact subset* of the refined grid (BC-aware), then return
# snapshots subsampled back to the quantum nodes.  Error scoring is thus
# pointwise on identical coordinates with no interpolation error, while
# the reference itself is resolution-independent of q.


def make_reference_grid(
    x: np.ndarray, bc: str = "periodic", min_points: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Refined classical-reference grid containing the q-grid `x` as an
    exact subset.  Returns (x_ref, take) with len(x_ref) >= min_points and
    x_ref[take] == x.  Refinement factor k = ceil(min_points / N)."""
    n = len(x)
    k = max(1, int(np.ceil(min_points / n))) if min_points else 1
    if bc == "dirichlet":
        x_ref = np.linspace(0.0, 1.0, (n - 1) * k + 1, endpoint=True)
    else:
        x_ref = np.linspace(0.0, 1.0, n * k, endpoint=False)
    take = np.arange(n) * k
    return x_ref, take


def solve_burgers_subsampled(
    u0_ref: np.ndarray,
    x_ref: np.ndarray,
    take: np.ndarray,
    nu: float,
    dt: float,
    n_steps: int,
    source_fn=None,
    bc: str = "periodic",
) -> list[np.ndarray]:
    """FTCS on the refined grid, sub-stepped for stability, then subsampled
    back to the q-grid.  Returns n_steps+1 snapshots on x_ref[take].

    The refined grid has a smaller dx, so the caller's macro `dt` is split
    into `sub` micro steps to satisfy the FTCS diffusion floor
    dt <= dx^2/(2 nu) (0.25 safety also covers advection)."""
    dx_ref = x_ref[1] - x_ref[0]
    if nu > 0.0:
        dt_stable = 0.25 * dx_ref * dx_ref / nu
        sub = max(1, int(np.ceil(dt / dt_stable)))
    else:
        sub = 1
    sols_fine = solve_burgers(
        u0_ref, x_ref, nu, dt / sub, n_steps * sub,
        source_fn=source_fn, bc=bc,
    )
    return [sols_fine[m * sub][take].copy() for m in range(n_steps + 1)]

