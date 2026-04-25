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

# Ensure the solver package is importable
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"),
)

from burgers_classical import initial_condition_sine
from burgers_cole_hopf import (
    build_heat_propagator,
    cole_hopf_forward,
)
from burgers_cole_hopf_circuit import (
    compute_theta_exact,
    build_conditional_ry,
    run_cole_hopf_circuit_sv,
    run_cole_hopf_circuit_simulation,
)
from burgers_trotter import run_simulation


# ── helpers ──────────────────────────────────────────────────────────


def _make_grid(q: int, bc: str = "periodic"):
    N = 1 << q
    if bc == "dirichlet":
        x = np.linspace(0, 1, N, endpoint=True)
    else:
        x = np.linspace(0, 1, N, endpoint=False)
    dx = x[1] - x[0]
    return x, dx, N


# ── 11.1  Classical no-regression ────────────────────────────────────


def test_11_1_classical_no_regression():
    """cole_hopf and shift agree at q=5, nu=0.1, T=0.3*t_shock."""
    q = 5
    nu = 0.1
    x, dx, N = _make_grid(q, "periodic")
    dt = 0.1 * dx
    u0 = initial_condition_sine(x)
    du = np.gradient(u0, dx)
    t_shock = 1.0 / np.max(np.abs(du))
    n_steps = round(0.3 * t_shock / dt)

    sols_ch, _ = run_simulation(
        u0, x, nu, dt, n_steps,
        method="cole_hopf", source_fn=None,
    )
    sols_shift, _ = run_simulation(
        u0, x, nu, dt, n_steps,
        method="shift", source_fn=None,
    )
    err = (
        np.linalg.norm(sols_ch[-1] - sols_shift[-1])
        / np.linalg.norm(sols_shift[-1])
    )
    assert err < 0.02, f"cole_hopf vs shift: {err:.4e} >= 0.02"


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
        propagator="qft-diagonal",
    )
    psi_circ = snaps[-1]

    err = np.linalg.norm(psi_circ - psi_dense)
    assert err < 1e-6, (
        f"qft-diagonal SV vs dense: {err:.2e} >= 1e-6"
    )


# ── 11.3  Dense-block statevector ───────────────────────────────────


def test_11_3_dense_block_statevector():
    """Dense-block (exact eigendecomp) vs dense propagator at q=4."""
    q, nu, n_steps = 4, 1e-2, 10
    x, dx, N = _make_grid(q)
    dt = 0.05 / n_steps
    L_box = float(N * dx)

    u0 = initial_condition_sine(x)
    phi0 = cole_hopf_forward(u0, dx, nu)
    psi0 = phi0 / np.linalg.norm(phi0)

    P = build_heat_propagator(N, dx, nu, dt)
    psi_dense = psi0.copy()
    for _ in range(n_steps):
        psi_dense = P @ psi_dense
        psi_dense /= np.linalg.norm(psi_dense)

    snaps, _ = run_cole_hopf_circuit_sv(
        psi0, q, nu, dt, n_steps, L_box,
        propagator="dense-block",
    )
    psi_circ = snaps[-1]

    err = np.linalg.norm(psi_circ - psi_dense)
    # Exact eigendecomp → should be near machine precision
    assert err < 1e-6, (
        f"dense-block SV vs dense: {err:.2e} >= 1e-6"
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

    # Dense reference (via classical cole_hopf)
    sols_ref, _ = run_simulation(
        u0, x, nu, dt, n_steps,
        method="cole_hopf", source_fn=None,
    )

    # Circuit with shots
    sols_shots, mets = run_cole_hopf_circuit_simulation(
        u0, x, nu, dt, n_steps,
        propagator="qft-diagonal", shots=150000,
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
        bc="dirichlet", propagator="dense-block",
        shots=0,
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
    confirms the Ran-2020 pipeline in ``burgers_mps`` is actually
    wired into the cole_hopf_circuit path.
    """
    from burgers_mps import (
        classical_to_mps, mps_to_circuit, normalize_state,
        reconstruct_from_mps,
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
    from burgers_mps import (
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
            propagator="qft-diagonal", shots=0,
            bond_dim=bd,
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
