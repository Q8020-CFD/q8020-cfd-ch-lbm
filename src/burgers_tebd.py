"""TEBD-style time evolution for Burgers equation (F2).

Replaces the O(4^q) Pauli-Trotter path with an MPO-based evolution that
scales as O(q * chi^3) per step.  The state is maintained as a quimb
MatrixProductState; the evolution operator is an MPO applied with bond-
dimension truncation.

Phase A (this module): builds the dense evolution unitary via the same
Hamiltonian as quantum_exact_step, converts to MPO via from_dense, and
applies to the MPS.  Dense construction limits this to q <= ~12; Phase B
replaces the dense intermediate with analytical MPO construction for
arbitrary q.

Pipeline per step:
  1. Build H_n via build_evolution_hamiltonian (same as Pauli path)
  2. Compute U = expm(-i H_n dt) as dense matrix
  3. Convert U to MPO via quimb from_dense (with bond truncation)
  4. Apply MPO to MPS (with bond truncation)
  5. Rescale by classical norm prediction (same as quantum_exact_step)

The classical mirror for the state-dependent H_n is maintained via
shift_euler_step.  This is the F2 ceiling; F10 (Cole-Hopf) removes it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import quimb.tensor as qtn
from scipy.linalg import expm, polar

from burgers_mps import normalize_state
from burgers_nonlinear import compute_rhs_shift


# -------------------------------------------------------------------
# MPS / MPO conversion helpers
# -------------------------------------------------------------------


def state_to_mps(
    psi: np.ndarray,
    q: int,
    max_bond: int | None = None,
    cutoff: float = 1e-14,
) -> qtn.MatrixProductState:
    """Convert a state vector to a quimb MPS."""
    return qtn.MatrixProductState.from_dense(
        psi.astype(complex), dims=[2] * q,
        max_bond=max_bond, cutoff=cutoff,
    )


def mps_to_vector(mps: qtn.MatrixProductState) -> np.ndarray:
    """Contract an MPS back to a dense vector."""
    return mps.to_dense().ravel()


def dense_to_mpo(
    mat: np.ndarray,
    q: int,
    max_bond: int | None = None,
    cutoff: float = 1e-14,
) -> qtn.MatrixProductOperator:
    """Convert a dense NxN matrix to a quimb MPO."""
    return qtn.MatrixProductOperator.from_dense(
        mat, dims=[2] * q,
        max_bond=max_bond, cutoff=cutoff,
    )


# -------------------------------------------------------------------
# Dense Hamiltonian -> evolution MPO
# -------------------------------------------------------------------


def build_hamiltonian_dense(
    u: np.ndarray,
    dx: float,
    dt: float,
    nu: float,
    g: np.ndarray | None = None,
    bc: str = "periodic",
) -> np.ndarray:
    """Build the Hermitian evolution generator directly from shift operators.

    Bypasses the O(4^q) Pauli decomposition used by quantum_exact_step.
    Cost is O(N^2) where N = 2^q — works up to q ~= 12.

    Constructs A such that exp(-i dt A) |psi_n> reproduces the classical
    forward-Euler update to first order in dt.  Uses the same lstsq
    outer-product construction as build_evolution_hamiltonian, but skips
    the Pauli basis projection — A is built directly in the computational
    basis.

    Returns H : ndarray, shape (N, N), complex Hermitian.
    """
    N = len(u)
    norm_u = np.linalg.norm(u)
    if norm_u < 1e-15:
        return np.zeros((N, N), dtype=complex)
    psi = u / norm_u

    rhs = compute_rhs_shift(u, dx, nu, g, bc=bc)
    u_next = u + dt * rhs
    norm_next = np.linalg.norm(u_next)
    if norm_next < 1e-15:
        return np.zeros((N, N), dtype=complex)
    psi_next = u_next / norm_next

    delta = (psi_next - psi) / dt

    # Minimal Hermitian A satisfying A|psi> = i*delta:
    #   exp(-i dt A)|psi> = |psi> - i dt A|psi> + ... = |psi> + dt*delta + ...
    # which matches |psi_next> to first order.
    A_outer = np.outer(delta, psi.conj())
    A = 1j * A_outer - 1j * A_outer.conj().T

    return A


def build_evolution_mpo(
    u: np.ndarray,
    dx: float,
    dt: float,
    nu: float,
    g: np.ndarray | None = None,
    bc: str = "periodic",
    max_bond: int | None = None,
    cutoff: float = 1e-14,
) -> qtn.MatrixProductOperator:
    """Build the evolution operator as an MPO.

    Constructs the Hermitian generator directly from shift operators
    (O(N^2), no Pauli decomposition), computes expm, and converts to
    MPO form.

    Returns U_mpo.
    """
    q = int(np.log2(len(u)))
    A = build_hamiltonian_dense(u, dx, dt, nu, g, bc=bc)
    U_mat = expm(-1j * dt * A)
    U_mpo = dense_to_mpo(U_mat, q, max_bond=max_bond, cutoff=cutoff)
    return U_mpo


# -------------------------------------------------------------------
# Single-step TEBD evolution
# -------------------------------------------------------------------


def tebd_step(
    u: np.ndarray,
    dx: float,
    dt: float,
    nu: float,
    g: np.ndarray | None = None,
    bc: str = "periodic",
    max_bond: int | None = None,
    cutoff: float = 1e-14,
) -> tuple[np.ndarray, dict]:
    """One TEBD evolution step on the physical velocity field.

    Builds H_n from the current state, converts the evolution unitary
    to an MPO, applies it to the normalised state (as MPS), and rescales
    by the classical norm prediction.

    Parameters
    ----------
    u        : physical velocity field, length N = 2^q.
    dx       : grid spacing.
    dt       : time step.
    nu       : kinematic viscosity.
    g        : source term array or None.
    bc       : "periodic" or "dirichlet".
    max_bond : max MPS/MPO bond dimension (None = full rank).
    cutoff   : SVD cutoff for from_dense and gate_with_mpo.

    Returns
    -------
    (u_next, metrics) — evolved physical velocity and diagnostics.
    """
    u_norm, _ = normalize_state(u)
    q = int(np.log2(len(u)))

    U_mpo = build_evolution_mpo(
        u, dx, dt, nu, g, bc=bc,
        max_bond=max_bond, cutoff=cutoff,
    )

    psi_mps = state_to_mps(u_norm, q, cutoff=cutoff)
    psi_bonds_in = psi_mps.bond_sizes()

    result_mps = psi_mps.gate_with_mpo(
        U_mpo, max_bond=max_bond, cutoff=cutoff,
    )
    psi_next = mps_to_vector(result_mps).real
    psi_bonds_out = result_mps.bond_sizes()

    # Classical norm prediction (same as quantum_exact_step)
    rhs = compute_rhs_shift(u, dx, nu, g, bc=bc)
    norm_next = np.linalg.norm(u + dt * rhs)

    u_next = psi_next * norm_next

    metrics: dict[str, Any] = {
        "method": "tebd",
        "mpo_bond_dims": U_mpo.bond_sizes(),
        "mps_bond_dims_in": psi_bonds_in,
        "mps_bond_dims_out": psi_bonds_out,
        "max_bond": max_bond,
        "cutoff": cutoff,
    }

    return u_next, metrics


# -------------------------------------------------------------------
# Multi-step simulation
# -------------------------------------------------------------------


def run_tebd_simulation(
    u0: np.ndarray,
    x: np.ndarray,
    nu: float,
    dt: float,
    n_steps: int,
    source_fn: Any = None,
    bc: str = "periodic",
    max_bond: int | None = None,
    cutoff: float = 1e-14,
) -> tuple[list[np.ndarray], list[dict]]:
    """Run multi-step Burgers simulation using TEBD evolution.

    Parameters
    ----------
    u0       : initial velocity field.
    x        : grid coordinates.
    nu       : kinematic viscosity.
    dt       : time step.
    n_steps  : number of Euler steps.
    source_fn: callable(x, t) -> ndarray, or None.
    bc       : "periodic" or "dirichlet".
    max_bond : max MPS/MPO bond dimension (None = full rank).
    cutoff   : SVD cutoff for MPO/MPS construction.

    Returns
    -------
    (solutions, metrics_list).
    """
    dx = x[1] - x[0]
    solutions = [u0.copy()]
    metrics_list: list[dict] = []
    u = u0.copy()

    for step in range(n_steps):
        t = step * dt
        g = source_fn(x, t) if source_fn is not None else None

        u, metrics = tebd_step(
            u, dx, dt, nu, g, bc=bc,
            max_bond=max_bond, cutoff=cutoff,
        )
        metrics_list.append(metrics)

        if not np.all(np.isfinite(u)):
            import sys
            print(
                f"[tebd] diverged at step {step + 1}/{n_steps}; "
                f"padding remaining steps with NaN",
                file=sys.stderr, flush=True,
            )
            nan_fill = np.full_like(u, np.nan)
            solutions.extend([nan_fill] * (n_steps - step))
            break
        solutions.append(u.copy())

    return solutions, metrics_list


# -------------------------------------------------------------------
# Phase B.1: W-II gate extraction and polar unitarization
# -------------------------------------------------------------------


def build_wii_layer_bond2(
    H_mpo: qtn.MatrixProductOperator | np.ndarray,
    dt: float,
) -> list[np.ndarray]:
    """Legacy Phase B.1 stub: bond-2 truncated W-II surrogate.

    Kept for reference / comparison.  Caps accuracy at O(1) when H has
    MPO bond dim > 2 (true for Burgers), which is why the project
    switched to :func:`build_wii_layer` (exact-SVD X2).

    Returns a list of q 4x4 unitary gates; contract with
    :func:`wii_gates_to_dense_bond2` (the legacy contractor, embedded
    inline here for completeness) — NOT compatible with the new variable-
    bond :func:`wii_gates_to_dense`.
    """
    if isinstance(H_mpo, qtn.MatrixProductOperator):
        H_dense = H_mpo.to_dense()
        q = H_mpo.L
    else:
        H_dense = np.asarray(H_mpo, dtype=complex)
        N = H_dense.shape[0]
        q = int(np.log2(N))
        if 2**q != N:
            raise ValueError(f"H size {N} is not a power of 2")

    U = expm(-1j * dt * H_dense)
    U_mpo = qtn.MatrixProductOperator.from_dense(
        U, dims=[2] * q, max_bond=2, cutoff=0.0,
    )

    gates: list[np.ndarray] = []
    for site in range(q):
        t = U_mpo.tensors[site]
        inds = list(t.inds)
        k_ax = inds.index(f"k{site}")
        b_ax = inds.index(f"b{site}")
        bond_axes = [a for a in range(len(inds)) if a not in (k_ax, b_ax)]

        if site == 0:
            right_ax = bond_axes[0]
            block = np.transpose(
                t.data, (k_ax, b_ax, right_ax)
            ).reshape(1, 2, 2, -1)
        elif site == q - 1:
            left_ax = bond_axes[0]
            block = np.transpose(
                t.data, (left_ax, k_ax, b_ax)
            ).reshape(-1, 2, 2, 1)
        else:
            left_ax, right_ax = bond_axes
            block = np.transpose(
                t.data, (left_ax, k_ax, b_ax, right_ax)
            )

        dl = block.shape[0]
        dr = block.shape[3]
        gate = np.zeros((4, 4), dtype=complex)
        for a_l in range(dl):
            for s_o in range(2):
                for s_i in range(2):
                    for a_r in range(dr):
                        gate[a_l * 2 + s_o, a_r * 2 + s_i] = (
                            block[a_l, s_o, s_i, a_r]
                        )

        Ug, _ = polar(gate)
        gates.append(Ug)

    return gates


def _exact_mpo_from_dense(
    U: np.ndarray, q: int,
) -> list[np.ndarray]:
    """Factor an N x N matrix (N = 2^q) into an exact MPO via iterated SVD.

    No truncation.  The MPO represents U with bond dims determined by
    the rank profile of U (which for U = exp(-i H dt) grows with dt but
    is bounded by min(4^k, 4^(q-k)) at bond k).

    Returns a list of q tensors each with shape (D_L, 2, 2, D_R) under
    the index convention (left_bond, s_out, s_in, right_bond).  D_L=1
    at site 0 and D_R=1 at site q-1.
    """
    # Reshape U : (N, N) -> (2, ..., 2, 2, ..., 2) with q ket + q bra
    # axes, then interleave to site-local (s_out_k, s_in_k) pairs.
    arr = U.reshape([2] * (2 * q))
    # Current axes: k0, k1, ..., k_{q-1}, b0, b1, ..., b_{q-1}
    # Target axes interleaved: k0, b0, k1, b1, ..., k_{q-1}, b_{q-1}
    perm = []
    for site in range(q):
        perm.append(site)
        perm.append(q + site)
    arr = np.transpose(arr, perm)
    # Fuse each (k_site, b_site) pair into a dim-4 index.
    arr = arr.reshape([4] * q)

    tensors: list[np.ndarray] = []
    # Left-to-right exact SVD.  Carry R = remainder.
    R = arr.reshape(1, 4, -1)  # shape (D_L=1, 4, 4^(q-1))
    for site in range(q - 1):
        d_l = R.shape[0]
        rest_dim = R.shape[2]
        M = R.reshape(d_l * 4, rest_dim)
        # Exact SVD (no truncation)
        Uf, S, Vh = np.linalg.svd(M, full_matrices=False)
        # Drop strictly-zero singular values so we don't carry rank
        # inflation for a factorization that is rank-deficient.
        tol = max(M.shape) * np.finfo(M.dtype).eps * (S[0] if S.size else 0)
        keep = int(np.sum(S > tol))
        if keep == 0:
            keep = 1
        Uf = Uf[:, :keep]
        S = S[:keep]
        Vh = Vh[:keep, :]
        d_r = keep
        # Site tensor: (d_l, 4, d_r) -> (d_l, 2, 2, d_r)
        site_t = Uf.reshape(d_l, 2, 2, d_r)
        tensors.append(site_t)
        # Fold S into remainder on the right.
        R = (np.diag(S) @ Vh).reshape(d_r, 4, -1)

    # Last site: remainder R has shape (d_l, 4, 1) -> (d_l, 2, 2, 1)
    d_l = R.shape[0]
    last = R.reshape(d_l, 2, 2, 1)
    tensors.append(last)
    return tensors


def build_wii_layer(
    H_mpo: qtn.MatrixProductOperator | np.ndarray,
    dt: float,
) -> list[np.ndarray]:
    """Build a W-II-style layer of unitary gates for U = exp(-i H dt).

    X2 implementation: exact-SVD factorization of U into an MPO,
    followed by polar unitarization of each site tensor.

    Approach:

    1. Compute U = expm(-i H dt) densely from H.
    2. Factor U into an MPO via iterated SVD without truncation.  The
       resulting per-site tensor has shape (D_L, 2, 2, D_R).  For open
       boundaries D_L=1 at site 0 and D_R=1 at site q-1.
    3. Pad each tensor on both bonds to a uniform D = max(bond dims)
       so every gate has the same (2D, 2D) square shape.  The boundary
       bonds are embedded at index 0; all other padded entries are zero.
    4. Reshape each (D, 2, 2, D) tensor to a (2D, 2D) matrix under
           G[a_L * 2 + s_out, a_R * 2 + s_in] = T[a_L, s_out, s_in, a_R].
    5. Polar-decompose each matrix into its nearest unitary.

    Phase B.1 acceptance status:

    - Unitarity (||G G^dag - I||_F < 1e-12): PASSES at all sites.
    - O(dt^3) step scaling: DOES NOT HOLD with this construction.
      The pre-polar reconstruction from raw SVD tensors is exact to
      machine precision (~1e-15), but the Zaletel fusion
        G[a_L*2 + s_out, a_R*2 + s_in] = T[a_L, s_out, s_in, a_R]
      is not aligned with SVD left- or right-canonical isometry;
      per-site tensors have O(1) deviation from isometric in this
      fusion at any dt, so the polar correction moves each site by
      O(1) and produces a dt-independent reconstruction floor of
      ~sqrt(2).  Achieving true O(dt^3) per-step accuracy requires
      the algebraic Zaletel W-II ladder construction, which
      presupposes H's MPO is in lower-triangular ladder form — not
      the form produced by MatrixProductOperator.from_dense on our
      rank-2 state-dependent Hermitian H.  Documented as a Phase
      B.2+ blocker; Phase B.1's hard acceptance is unitarity alone
      (spec §10a line 523).

    Reconstruction via :func:`wii_gates_to_dense` clamps the left and
    right ancilla indices to 0, recovering a (2^q, 2^q) effective
    operator on the q data qubits.

    Returns: list of q complex numpy arrays each shaped (2D, 2D),
    unitary to ||G G^dag - I||_F < 1e-12.
    """
    if isinstance(H_mpo, qtn.MatrixProductOperator):
        H_dense = H_mpo.to_dense()
        q = H_mpo.L
    else:
        H_dense = np.asarray(H_mpo, dtype=complex)
        N = H_dense.shape[0]
        q = int(np.log2(N))
        if 2**q != N:
            raise ValueError(f"H size {N} is not a power of 2")

    U = expm(-1j * dt * H_dense)
    tensors = _exact_mpo_from_dense(U, q)

    # Uniform padding to common bond dim D.
    D = max(max(t.shape[0], t.shape[3]) for t in tensors)

    gates: list[np.ndarray] = []
    for t in tensors:
        d_l, _, _, d_r = t.shape
        padded = np.zeros((D, 2, 2, D), dtype=complex)
        padded[:d_l, :, :, :d_r] = t
        # Reshape (D, 2, 2, D) -> (2D, 2D) with the convention
        # G[a_L*2 + s_out, a_R*2 + s_in] = T[a_L, s_out, s_in, a_R].
        # Transpose (a_L, s_out, s_in, a_R) -> (a_L, s_out, a_R, s_in),
        # then fuse (a_L, s_out) x (a_R, s_in).
        mat = np.transpose(padded, (0, 1, 3, 2)).reshape(D * 2, D * 2)
        Ug, _ = polar(mat)
        gates.append(Ug)

    return gates


def wii_gates_to_dense(gates: list[np.ndarray]) -> np.ndarray:
    """Reconstruct the dense operator from a W-II gate layer (X2 form).

    Gates are uniform (2D, 2D) unitaries with the convention
        G[a_L * 2 + s_out, a_R * 2 + s_in] = T[a_L, s_out, s_in, a_R].
    Contract the resulting MPO with the left ancilla clamped to 0 on
    input and the right ancilla clamped to 0 on output.  Returns the
    effective (2^q, 2^q) operator on the q data qubits.
    """
    q = len(gates)

    # Unfuse each gate (2D, 2D) -> (D, 2, 2, D) site tensor.
    blocks: list[np.ndarray] = []
    for g in gates:
        two_D = g.shape[0]
        if g.shape != (two_D, two_D) or two_D % 2 != 0:
            raise ValueError(
                f"Gate must be square with even dim; got shape {g.shape}"
            )
        D = two_D // 2
        # Inverse of the fusion (a_L, s_out) x (a_R, s_in):
        # (2D, 2D) -> (D, 2, D, 2) -> transpose to (D, 2, 2, D).
        t = g.reshape(D, 2, D, 2).transpose(0, 1, 3, 2)
        blocks.append(t)

    # Contract sequentially.  Start at site 0 with left-bond pinned to 0.
    # blocks[0] shape: (D, 2, 2, D); pick a_L=0 -> shape (2, 2, D).
    acc = blocks[0][0]  # indices: (k0, b0, bond)

    for k in range(1, q):
        # acc indices after k sites: (k0..k_{k-1}, b0..b_{k-1}, bond)
        # blocks[k] indices: (bond_L, k_k, b_k, bond_R)
        bond_axis = acc.ndim - 1
        acc = np.tensordot(acc, blocks[k], axes=([bond_axis], [0]))
        # New axes: (k0..k_{k-1}, b0..b_{k-1}, k_k, b_k, bond_R)
        # Rearrange to: (k0..k_{k-1}, k_k, b0..b_{k-1}, b_k, bond_R)
        n_k = k
        n_b = k
        perm = (
            list(range(n_k))                               # k0..k_{k-1}
            + [n_k + n_b]                                  # k_k
            + list(range(n_k, n_k + n_b))                  # b0..b_{k-1}
            + [n_k + n_b + 1]                              # b_k
            + [n_k + n_b + 2]                              # bond_R
        )
        acc = np.transpose(acc, perm)

    # Final axes: (k0..k_{q-1}, b0..b_{q-1}, bond_R).  Pin bond_R=0.
    acc = acc[(slice(None),) * (2 * q) + (0,)]
    dim = 2**q
    # acc shape now: (2,)*q + (2,)*q  = (2^q, 2^q) when reshaped.
    # But the axis order is (k0..k_{q-1}, b0..b_{q-1}) — already the
    # natural ket/bra ordering.
    U_eff = acc.reshape(dim, dim)
    return U_eff


# -------------------------------------------------------------------
# Smoke test and validation
# -------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from burgers_trotter import quantum_exact_step, shift_euler_step

    print("=== F2 TEBD Classical Reference — Smoke Test ===\n")

    for q in (3, 4, 5):
        N = 2**q
        dx = 1.0 / N
        dt = 1e-4
        nu = 1e-4
        x = np.linspace(0, 1, N, endpoint=False)
        u = np.sin(2.0 * np.pi * x)

        # Single-step comparison: tebd vs quantum_exact vs shift
        u_tebd, m_tebd = tebd_step(u, dx, dt, nu)
        u_exact, m_exact = quantum_exact_step(u, dx, dt, nu)
        u_shift, _ = shift_euler_step(u, dx, dt, nu)

        err_tebd_exact = np.max(np.abs(u_tebd - u_exact))
        err_tebd_shift = np.max(np.abs(u_tebd - u_shift))
        err_exact_shift = np.max(np.abs(u_exact - u_shift))

        print(f"q={q} (N={N}):")
        print(f"  tebd vs exact:  {err_tebd_exact:.2e}")
        print(f"  tebd vs shift:  {err_tebd_shift:.2e}")
        print(f"  exact vs shift: {err_exact_shift:.2e}")
        print(f"  MPO bonds: {m_tebd['mpo_bond_dims']}")
        print(f"  MPS bonds in:  {m_tebd['mps_bond_dims_in']}")
        print(f"  MPS bonds out: {m_tebd['mps_bond_dims_out']}")
        print()

    # Bond-dim truncation sweep at q=5
    print("=== Bond-dim truncation sweep (q=5) ===\n")
    q = 5
    N = 2**q
    dx = 1.0 / N
    x = np.linspace(0, 1, N, endpoint=False)
    u = np.sin(2.0 * np.pi * x)

    u_ref, _ = quantum_exact_step(u, dx, dt, nu)
    for chi in (4, 8, 16, None):
        u_t, m_t = tebd_step(u, dx, dt, nu, max_bond=chi)
        err = np.max(np.abs(u_t - u_ref))
        label = "full" if chi is None else str(chi)
        print(
            f"  chi={label:>4s}  mpo={m_t['mpo_bond_dims']}  "
            f"mps_out={m_t['mps_bond_dims_out']}  err={err:.2e}"
        )

    # Multi-step trajectory at q=4
    print("\n=== Multi-step trajectory (q=4, 10 steps) ===\n")
    q = 4
    N = 2**q
    dx = 1.0 / N
    n_steps = 10
    x = np.linspace(0, 1, N, endpoint=False)
    u0 = np.sin(2.0 * np.pi * x)

    sols_tebd, mets = run_tebd_simulation(
        u0, x, nu, dt, n_steps,
    )
    sols_shift = [u0.copy()]
    u_s = u0.copy()
    for step in range(n_steps):
        u_s, _ = shift_euler_step(u_s, dx, dt, nu)
        sols_shift.append(u_s.copy())

    for step in range(n_steps + 1):
        err = np.max(np.abs(sols_tebd[step] - sols_shift[step]))
        print(f"  step {step:>2d}  err(tebd-shift)={err:.2e}")

    # Scalability test: q=6..10
    import time
    print("\n=== Scalability (single step, q=6..10) ===\n")
    for q in (6, 7, 8, 10):
        N = 2**q
        dx = 1.0 / N
        x = np.linspace(0, 1, N, endpoint=False)
        u = np.sin(2.0 * np.pi * x)
        t0 = time.time()
        u_next, m = tebd_step(u, dx, dt, nu)
        elapsed = time.time() - t0
        print(
            f"  q={q:>2d} (N={N:>4d})  {elapsed:.2f}s  "
            f"mpo={m['mpo_bond_dims']}  "
            f"mps_out={m['mps_bond_dims_out']}"
        )

    # Bond-dim sweep at q=8
    print("\n=== Bond-dim sweep (q=8) ===\n")
    q = 8
    N = 2**q
    dx = 1.0 / N
    x = np.linspace(0, 1, N, endpoint=False)
    u = np.sin(2.0 * np.pi * x)
    u_ref, _ = tebd_step(u, dx, dt, nu)
    for chi in (4, 8, 16, 32, None):
        u_t, m_t = tebd_step(u, dx, dt, nu, max_bond=chi)
        err = np.max(np.abs(u_t - u_ref))
        label = "full" if chi is None else str(chi)
        print(
            f"  chi={label:>4s}  mpo={m_t['mpo_bond_dims']}  "
            f"mps_out={m_t['mps_bond_dims_out']}  err={err:.2e}"
        )

    # ---------------------------------------------------------------
    # Phase B.1 validation: W-II gate extraction and polar unitarization
    # X2 implementation: exact-SVD factorization + polar.
    # ---------------------------------------------------------------
    print("\n=== Phase B.1: W-II gate extraction + polar (X2) ===\n")

    from burgers_tebd import (
        build_wii_layer, wii_gates_to_dense, _exact_mpo_from_dense,
    )

    # Self-check: exact-SVD reconstruction (no polar) for identity.
    print("  Self-check: exact-SVD round-trip for random unitaries")
    for q_c in (2, 3):
        rng = np.random.default_rng(0)
        A_rand = rng.normal(size=(2**q_c, 2**q_c)) + 1j * rng.normal(
            size=(2**q_c, 2**q_c)
        )
        U_test, _ = polar(A_rand)
        ts = _exact_mpo_from_dense(U_test, q_c)
        D_ = max(max(t.shape[0], t.shape[3]) for t in ts)
        raw_gates = []
        for t in ts:
            d_l, _, _, d_r = t.shape
            padded = np.zeros((D_, 2, 2, D_), dtype=complex)
            padded[:d_l, :, :, :d_r] = t
            mat = np.transpose(padded, (0, 1, 3, 2)).reshape(D_ * 2, D_ * 2)
            raw_gates.append(mat)
        U_rt = wii_gates_to_dense(raw_gates)
        err_rt = np.linalg.norm(U_rt - U_test, ord="fro")
        print(
            f"    q={q_c} D={D_}: round-trip err = {err_rt:.3e}"
        )


    unit_tol = 1e-12

    all_unit_pass = True
    for q_test in (3, 4):
        N = 2**q_test
        dx_t = 1.0 / N
        nu_t = 1e-4
        x_t = np.linspace(0, 1, N, endpoint=False)
        u_t = np.sin(2.0 * np.pi * x_t)
        for dt_t in (1e-3, 5e-4):
            H_t = build_hamiltonian_dense(u_t, dx_t, dt_t, nu_t)
            gates = build_wii_layer(H_t, dt_t)
            # Gates are uniform (2D, 2D); compute identity from actual dim.
            max_unit = 0.0
            for g in gates:
                I_n = np.eye(g.shape[0])
                max_unit = max(
                    max_unit,
                    np.linalg.norm(g @ g.conj().T - I_n),
                )
            status = "PASS" if max_unit < unit_tol else "FAIL"
            print(
                f"  q={q_test} dt={dt_t:.1e} "
                f"gate_dim={gates[0].shape[0]}: "
                f"max unitarity err = {max_unit:.2e}  [{status}]"
            )
            if max_unit >= unit_tol:
                all_unit_pass = False

    print(
        f"\n  Unitarity criterion (<{unit_tol:.0e}): "
        f"{'PASS' if all_unit_pass else 'FAIL'}"
    )
    assert all_unit_pass, "W-II gate unitarity check failed"

    # Criterion 2: step error vs dense expm scales as O(dt^3).
    # Halve dt across 5 values; fit exponent.  Use q=3 (scaling study).
    print("\n  O(dt^3) scaling (halve dt, 5 values):")
    q_s = 3
    N_s = 2**q_s
    dx_s = 1.0 / N_s
    nu_s = 1e-4
    x_s = np.linspace(0, 1, N_s, endpoint=False)
    u_s = np.sin(2.0 * np.pi * x_s)

    # Fix H at a reference dt (state-dependent; freeze to avoid confound).
    H_ref = build_hamiltonian_dense(u_s, dx_s, 1e-4, nu_s)

    dts = [1e-3, 5e-4, 2.5e-4, 1.25e-4, 6.25e-5]
    errs: list[float] = []
    for dt_s in dts:
        gates_s = build_wii_layer(H_ref, dt_s)
        U_rec = wii_gates_to_dense(gates_s)
        U_exact = expm(-1j * dt_s * H_ref)
        err = float(np.linalg.norm(U_rec - U_exact, ord="fro"))
        errs.append(err)
        print(f"    dt={dt_s:.2e}: err_F={err:.3e}")

    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    print(f"  Halve-dt ratios (target ~8): {[f'{r:.2f}' for r in ratios]}")

    # Fit scaling exponent p from log(err) ~ p*log(dt).
    log_dt = np.log(np.array(dts))
    log_err = np.log(np.array(errs))
    p_fit = float(np.polyfit(log_dt, log_err, 1)[0])
    print(f"  fitted scaling exponent p = {p_fit:.3f}  (target ~3)")

    # Acceptance: p in [2.5, 3.5].
    p_lo, p_hi = 2.5, 3.5
    scaling_pass = p_lo <= p_fit <= p_hi
    print(
        f"  O(dt^3) criterion (p in [{p_lo}, {p_hi}]): "
        f"{'PASS' if scaling_pass else 'FAIL'}"
    )
    if not scaling_pass:
        print(
            "  WARN: fitted exponent outside [2.5, 3.5].  The X2\n"
            "  exact-SVD construction gives a perfect reconstruction\n"
            "  of the exact MPO factorization (round-trip err ~1e-15\n"
            "  before polar), but the Zaletel fusion\n"
            "    G[a_L*2+s_out, a_R*2+s_in] = T[a_L, s_out, s_in, a_R]\n"
            "  is not aligned with SVD left/right-canonical isometry;\n"
            "  the per-site tensors have O(1) deviation from isometric\n"
            "  in this fusion at any dt.  Polar-unitarization therefore\n"
            "  moves each site by O(1), producing a dt-independent\n"
            "  reconstruction floor of ~sqrt(2).  True O(dt^3) accuracy\n"
            "  requires the algebraic Zaletel W-II ladder construction,\n"
            "  which presupposes H's MPO in lower-triangular ladder\n"
            "  form — not the native form produced by\n"
            "  MatrixProductOperator.from_dense on our rank-2 state-\n"
            "  dependent Hermitian H.  Phase B.2+ blocker; acceptance\n"
            "  for B.1 is unitarity only (passes above)."
        )

    print(
        "\n  Phase B.1 unitarity criterion: PASS\n"
        f"  Phase B.1 scaling criterion: "
        f"{'PASS' if scaling_pass else 'FAIL (see WARN above)'}"
    )

    print("\nDone.")
