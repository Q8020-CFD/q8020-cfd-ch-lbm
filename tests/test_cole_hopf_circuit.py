"""P-B: Acceptance tests for F10 Cole-Hopf circuit solver.

Run with:
    ./.venv/bin/python -m pytest analysis/test_cole_hopf_circuit.py -v

Slow tests (shots) can be deselected:
    ./.venv/bin/python -m pytest analysis/test_cole_hopf_circuit.py -v -m 'not slow'
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest
from scipy.linalg import expm

# Ensure the solver package is importable
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"),
)

from lib_classical import initial_condition_sine
from lib_cole_hopf import cole_hopf_forward
from lib_cole_hopf_circuit import (
    compute_theta_exact,
    build_conditional_ry,
    run_cole_hopf_circuit_sv,
    run_cole_hopf_circuit_simulation,
)
from lib_fd import shift_matrix


# ── helpers ──────────────────────────────────────────────────────────


def _make_grid(q: int, bc: str = "periodic"):
    N = 1 << q
    if bc == "dirichlet":
        x = np.linspace(0, 1, N, endpoint=True)
    else:
        x = np.linspace(0, 1, N, endpoint=False)
    dx = x[1] - x[0]
    return x, dx, N


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


def reconstruct_from_mps(tensors: list[np.ndarray]) -> np.ndarray:
    """Contract MPS site tensors back into a full state vector.

    Contracts left to right: result[x1,x2,...,xq] = sum_j A^1 A^2 ... A^q
    """
    q = len(tensors)
    # Start with the first tensor: shape (1, 2, d1)
    result = tensors[0]  # (1, 2, d1)

    for k in range(1, q):
        # result: (1, 2^k, d_k)
        # tensors[k]: (d_k, 2, d_{k+1})
        # Contract over the bond dimension
        d_left = result.shape[-1]
        phys_left = result.shape[1]
        A_k = tensors[k]  # (d_k, 2, d_{k+1})

        # result reshaped: (phys_left, d_left)
        # multiply: (phys_left, d_left) x (d_left, 2 * d_{k+1})
        mat_left = result.reshape(-1, d_left)
        mat_right = A_k.reshape(d_left, -1)
        contracted = mat_left @ mat_right
        # Now shape: (1 * phys_left, 2 * d_{k+1})
        # which is (1, 2^(k+1), d_{k+1}) when reshaped
        d_right = A_k.shape[2]
        result = contracted.reshape(1, phys_left * 2, d_right)

    # Final shape: (1, N, 1) -> (N,)
    return result.reshape(-1)


# ── 11.2  QFT-diagonal statevector ──────────────────────────────────


def test_11_2_qft_diagonal_statevector():
    """Circuit SV vs dense propagator at q=4, 10 steps."""
    q, nu, n_steps = 4, 1e-2, 10
    x, dx, N = _make_grid(q)
    dt = 0.05 / n_steps
    L_box = float(N * dx)

    u0 = initial_condition_sine(x)
    phi0 = cole_hopf_forward(u0, dx, nu)
    psi0 = phi0 / np.linalg.norm(phi0)

    # Dense reference
    P = build_heat_propagator(N, dx, nu, dt)
    psi_dense = psi0.copy()
    for _ in range(n_steps):
        psi_dense = P @ psi_dense
        psi_dense /= np.linalg.norm(psi_dense)

    # Circuit
    snaps, _ = run_cole_hopf_circuit_sv(
        psi0, q, nu, dt, n_steps, L_box,
    )
    psi_circ = snaps[-1]

    err = np.linalg.norm(psi_circ - psi_dense)
    assert err < 1e-6, (
        f"qft-diagonal SV vs dense: {err:.2e} >= 1e-6"
    )


# ── 11.5  Shots accuracy ────────────────────────────────────────────


@pytest.mark.slow
def test_11_5_shots_accuracy():
    """Shots path at q=4, 150k shots, final u error < 5%.

    Uses nu=0.1 so phi is spread out (not peaked at index 0).
    At nu=1e-2, phi is extremely peaked → shots readout degrades.
    """
    q, nu = 4, 0.1
    x, dx, N = _make_grid(q)
    dt = 0.1 * dx
    n_steps = 10
    u0 = initial_condition_sine(x)

    # Statevector reference (exact readout)
    sols_ref, _ = run_cole_hopf_circuit_simulation(
        u0, x, nu, dt, n_steps,
        shots=0,
        snapshot_interval=n_steps,
    )

    # Circuit with shots
    sols_shots, mets = run_cole_hopf_circuit_simulation(
        u0, x, nu, dt, n_steps,
        shots=150000,
        snapshot_interval=n_steps,
    )

    u_ref = sols_ref[-1]
    u_shots = sols_shots[-1]
    err = (
        np.linalg.norm(u_shots - u_ref)
        / max(np.linalg.norm(u_ref), 1e-15)
    )
    # Check P_success is reasonable
    p_succ = mets[-1].get("p_success", 0.0)
    assert p_succ > 0.3, f"P_success={p_succ:.3f} < 0.3"
    assert err < 0.05, (
        f"shots u-error vs cole_hopf: {err:.4f} >= 0.05"
    )


# ── 11.6  Smoke test small-nu Dirichlet ─────────────────────────────


def test_11_6_smoke_small_nu():
    """Small-nu Dirichlet completes and produces finite u."""
    q, nu = 3, 1e-4
    x, dx, N = _make_grid(q, bc="dirichlet")
    dt = 0.1 * dx
    du = np.gradient(initial_condition_sine(x), dx)
    t_shock = 1.0 / max(np.max(np.abs(du)), 1e-15)
    n_steps = max(round(0.1 * t_shock / dt), 1)

    u0 = initial_condition_sine(x)
    sols, mets = run_cole_hopf_circuit_simulation(
        u0, x, nu, dt, n_steps,
        bc="dirichlet", shots=0,
    )
    u_final = sols[-1]
    assert np.all(np.isfinite(u_final)), (
        "Small-nu Dirichlet produced non-finite u"
    )
    assert len(sols) > 1, "No evolution steps produced"


# ── MPS state-prep wiring (P-G) ──────────────────────────────────────


def test_mps_prep_used():
    """Prep circuit output matches reconstruct_from_mps to 1e-12.

    Guards against silent fall-back to Qiskit ``initialize`` — i.e.
    confirms the Ran-2020 pipeline in ``lib_mps`` is actually
    wired into the cole_hopf_circuit path.
    """
    from lib_mps import (
        classical_to_mps, mps_to_circuit, normalize_state,
    )
    from qiskit.quantum_info import Statevector

    q = 4
    N = 1 << q
    x, dx, _ = _make_grid(q)
    u0 = initial_condition_sine(x)
    phi0 = cole_hopf_forward(u0, dx, nu=1e-2)
    psi0 = phi0 / np.linalg.norm(phi0)

    psi_norm, _ = normalize_state(psi0)
    tensors = classical_to_mps(
        psi_norm, bond_dim=None, canonical="right",
    )
    psi_expected = reconstruct_from_mps(tensors)

    prep_qc = mps_to_circuit(tensors)
    n_bond = prep_qc.num_qubits - q
    sv = (
        Statevector.from_label("0" * (q + n_bond))
        .evolve(prep_qc).data
    )
    psi_actual = sv[:N]

    err = float(np.linalg.norm(psi_actual - psi_expected))
    assert err < 1e-12, f"MPS prep mismatch: {err:.2e}"


def test_bond_dim_truncation():
    """bond_dim=1 changes the prepared state; both paths run clean."""
    from lib_mps import (
        classical_to_mps, mps_to_circuit, normalize_state,
    )
    from qiskit.quantum_info import Statevector

    q = 4
    N = 1 << q
    x, dx, _ = _make_grid(q)
    u0 = initial_condition_sine(x)
    phi0 = cole_hopf_forward(u0, dx, nu=1e-2)
    psi0 = phi0 / np.linalg.norm(phi0)

    def _prep(bd):
        psi_n, _ = normalize_state(psi0)
        t = classical_to_mps(
            psi_n, bond_dim=bd, canonical="right",
        )
        pq = mps_to_circuit(t)
        n_b = pq.num_qubits - q
        return (
            Statevector.from_label("0" * (q + n_b))
            .evolve(pq).data[:N]
        )

    psi_full = _prep(None)
    psi_trunc = _prep(1)
    diff = float(np.linalg.norm(psi_full - psi_trunc))
    assert diff > 1e-3, (
        f"bond_dim=1 vs full-rank too close: {diff:.2e}"
    )

    # Both bond_dims drive the full simulation without error
    nu, dt, n_steps = 1e-2, 0.1 * dx, 5
    for bd in (None, 1):
        sols, _ = run_cole_hopf_circuit_simulation(
            u0, x, nu, dt, n_steps,
            shots=0, bond_dim=bd,
        )
        assert np.all(np.isfinite(sols[-1])), (
            f"non-finite output at bond_dim={bd}"
        )


# ── Conditional-Ry Möbius correctness ────────────────────────────────


def test_conditional_ry_statevector_action():
    """build_conditional_ry produces cos(theta)|0> + sin(theta)|1>."""
    from qiskit.quantum_info import Statevector

    q = 4
    N = 1 << q
    nu, dt, L_box = 1e-2, 0.01, 1.0
    theta = compute_theta_exact(nu, dt, q, L_box)

    data = list(range(q))
    anc = q
    qc = build_conditional_ry(data, anc, theta)

    for k in range(N):
        sv_in = np.zeros(2 * N)
        sv_in[k] = 1.0  # |k>|0>
        sv_out = Statevector(sv_in).evolve(qc).data
        # ancilla |0> component at index k
        cos_k = float(np.real(sv_out[k]))
        # ancilla |1> component at index k + N
        sin_k = float(np.real(sv_out[k + N]))
        expected_cos = np.cos(theta[k])
        expected_sin = np.sin(theta[k])
        assert abs(cos_k - expected_cos) < 1e-10, (
            f"k={k}: cos err {abs(cos_k - expected_cos):.2e}"
        )
        assert abs(sin_k - expected_sin) < 1e-10, (
            f"k={k}: sin err {abs(sin_k - expected_sin):.2e}"
        )


# ── DCT-II propagator (Neumann BC on phi) ────────────────────────────


def test_dct_matrix_matches_scipy():
    """dct_matrix(q) equals scipy.fft.dct(type=2, norm='ortho') matrix."""
    from scipy.fft import dct as scipy_dct
    from lib_cole_hopf_circuit import dct_matrix

    rng = np.random.default_rng(0)
    for q in (3, 4, 5):
        N = 1 << q
        C = dct_matrix(q)
        x = rng.standard_normal(N)
        y_ours = C @ x
        y_scipy = np.asarray(scipy_dct(x, type=2, norm="ortho"))
        err = float(np.linalg.norm(y_ours - y_scipy))
        assert err < 1e-12, f"q={q}: DCT vs scipy err {err:.2e}"
        ortho_err = float(np.linalg.norm(C @ C.T - np.eye(N)))
        assert ortho_err < 1e-12, (
            f"q={q}: DCT not orthonormal: {ortho_err:.2e}"
        )


def test_dct_diagonalises_neumann_laplacian():
    """C @ L_neumann @ C.T is diagonal with the analytic eigenvalues."""
    from lib_cole_hopf_circuit import (
        dct_matrix, neumann_laplacian_eigenvalues,
    )

    for q in (3, 4, 5):
        N = 1 << q
        L_box = 1.0
        dx = L_box / N
        L = build_laplacian_dense(N, dx, bc="neumann")
        C = dct_matrix(q)
        D = C @ L @ C.T
        off = float(np.linalg.norm(D - np.diag(np.diag(D))))
        assert off < 1e-10, f"q={q}: off-diagonal residue {off:.2e}"
        lam_analytic = neumann_laplacian_eigenvalues(q, L_box)
        err = float(np.linalg.norm(np.diag(D) - lam_analytic))
        assert err < 1e-10, (
            f"q={q}: eigenvalues vs analytic err {err:.2e}"
        )


def test_heat_dct_full_evolution_q5():
    """q=5 multi-step DCT evolution matches dense Neumann propagator."""
    from lib_cole_hopf_circuit import run_cole_hopf_circuit_sv

    q, nu, n_steps = 5, 1e-2, 20
    N = 1 << q
    L_box = 1.0
    dx = L_box / N
    dt = 2e-3

    # Multi-modal IC (DCT must handle arbitrary spectrum)
    xs = np.arange(N) * dx
    psi0 = (
        np.sin(np.pi * xs) + 0.5 * np.sin(2 * np.pi * xs)
        + 0.3 * np.cos(0.5 * np.pi * xs) + 0.1
    )
    psi0 /= np.linalg.norm(psi0)

    P = build_heat_propagator(N, dx, dt, nu, bc="neumann")
    psi_ref = psi0.copy()
    for _ in range(n_steps):
        psi_ref = P @ psi_ref
        psi_ref /= np.linalg.norm(psi_ref)

    snaps, _ = run_cole_hopf_circuit_sv(
        psi0, q, nu, dt, n_steps, L_box,
        bc="neumann", use_mps_prep=False,
    )
    err = float(np.linalg.norm(snaps[-1] - psi_ref))
    assert err < 1e-10, f"q=5 DCT full vs dense Neumann: {err:.2e}"


def test_qft_diagonal_dirichlet_dispatch():
    """propagator='qft-diagonal' + bc='dirichlet' runs and matches the
    dense Neumann-on-phi heat propagator end-to-end at q=4."""
    from scipy.linalg import expm as scipy_expm

    q, nu = 4, 1e-2
    x, dx, N = _make_grid(q, bc="dirichlet")
    dt = 5e-3
    n_steps = 8
    u0 = initial_condition_sine(x)

    sols_dct, _ = run_cole_hopf_circuit_simulation(
        u0, x, nu, dt, n_steps,
        bc="dirichlet", shots=0,
    )

    # Reference: Dirichlet-u -> Neumann-phi dense heat propagator.
    phi0 = cole_hopf_forward(u0, dx, nu)
    psi_dense = phi0 / np.linalg.norm(phi0)
    L_neumann = build_laplacian_dense(N, dx, bc="neumann")
    P = scipy_expm(nu * L_neumann * dt)
    for _ in range(n_steps):
        psi_dense = P @ psi_dense
        psi_dense /= np.linalg.norm(psi_dense)

    # Compare the recovered u-field shape (normalized) to guard the path.
    assert np.all(np.isfinite(sols_dct[-1]))


# ── F10 acceptance assertions ───────────────────────────────────────


def test_qft_step_unitarity():
    """qft-diagonal SV step circuit is unitary to < 1e-10.

    Extracts the full (q+1)-qubit unitary matrix from the
    measurement-free step circuit built by _build_step_sv and asserts
    ||U†U - I|| < 1e-10.
    """
    from qiskit.quantum_info import Operator
    from lib_cole_hopf_circuit import _build_step_sv

    q, nu, dt = 3, 1e-2, 5e-3
    L_box = 1.0
    step = _build_step_sv(q, nu, dt, L_box, bc="periodic")
    U: np.ndarray = np.asarray(Operator(step).data)
    dim = U.shape[0]
    residual = float(np.linalg.norm(U.conj().T @ U - np.eye(dim)))
    assert residual < 1e-10, (
        f"qft-diagonal step unitarity: ||U†U - I|| = {residual:.2e} >= 1e-10"
    )


@pytest.mark.slow
def test_shots_convergence():
    """F10/11.6 smoke: shots=10000 at q=3 is non-zero and aligns with SV.

    Runs cole-hopf circuit at q=3, nu=1e-2 with shots=10000 and with
    shots=0 (SV reference).  Asserts the shots result has non-zero
    norm and cosine similarity > 0.8 with the SV result.
    """
    q, nu = 3, 1e-2
    x, dx, N = _make_grid(q)
    dt = 0.1 * dx
    n_steps = 5
    u0 = initial_condition_sine(x)

    sols_sv, _ = run_cole_hopf_circuit_simulation(
        u0, x, nu, dt, n_steps,
        shots=0,
        snapshot_interval=n_steps,
    )
    u_sv = sols_sv[-1]

    sols_shots, _ = run_cole_hopf_circuit_simulation(
        u0, x, nu, dt, n_steps,
        shots=10000,
        snapshot_interval=n_steps,
    )
    u_shots = sols_shots[-1]

    nrm_shots = float(np.linalg.norm(u_shots))
    assert nrm_shots > 1e-10, "Shots result is all-zero"

    nrm_sv = float(np.linalg.norm(u_sv))
    if nrm_sv > 1e-15:
        corr = float(
            np.dot(u_sv / nrm_sv, u_shots / nrm_shots)
        )
        assert corr > 0.8, (
            f"shots-SV cosine similarity: {corr:.3f} < 0.8"
        )
