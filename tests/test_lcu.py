"""Tests for LCU block-encoding primitives and CH-LCU propagator.

Covers SPEC-F3-LCU-method.md §5.5 tests #1, #2, #3, #6, #8.
LCU+source forcing tests in test_lcu_source.py.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"),
)

from burgers_lcu import (
    build_prepare_circuit,
    build_select_circuit,
    lcu_block_encoding,
    heat_lcu_step_circuit,
)
from burgers_mpo import (
    extract_block_encoded_operator,
    gradient_lcu_circuit,
    shift_matrix,
)


# ── #1  LCU block-encoding correctness ─────────────────────────────


def test_lcu_block_encoding_correctness():
    """Synthetic A = c0*X + c1*Z on 1 qubit, verify block-encoding."""
    X_qc = QuantumCircuit(1, name="X")
    X_qc.x(0)
    Z_qc = QuantumCircuit(1, name="Z")
    Z_qc.z(0)

    c0, c1 = 0.7, -0.3
    coeffs = np.array([c0, c1])
    qc, lam = lcu_block_encoding([X_qc, Z_qc], coeffs, 1)
    assert lam == pytest.approx(abs(c0) + abs(c1), rel=1e-12)

    U = Operator(qc).data
    n_anc = qc.num_qubits - 1
    A_enc = extract_block_encoded_operator(
        U, n_system=1, n_ancilla=n_anc, anc_in=0, anc_out=0,
    )

    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    A_ref = (c0 * X + c1 * Z) / lam

    np.testing.assert_allclose(A_enc, A_ref, atol=1e-12)


# ── #2  K not power-of-2 ───────────────────────────────────────────


def test_lcu_select_prepare_handles_K_not_pow2():
    """K=3 on 2 ancillas: one unused state, post-selection correct."""
    X_qc = QuantumCircuit(1, name="X")
    X_qc.x(0)
    Z_qc = QuantumCircuit(1, name="Z")
    Z_qc.z(0)
    I_qc = QuantumCircuit(1, name="I")
    I_qc.id(0)

    coeffs = np.array([0.5, 0.3, 0.2])
    qc, lam = lcu_block_encoding([I_qc, X_qc, Z_qc], coeffs, 1)
    n_anc = qc.num_qubits - 1
    assert n_anc == 2  # ceil(log2(3)) = 2

    U = Operator(qc).data
    A_enc = extract_block_encoded_operator(
        U, n_system=1, n_ancilla=n_anc, anc_in=0, anc_out=0,
    )

    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    A_ref = (0.5 * I + 0.3 * X + 0.2 * Z) / lam

    np.testing.assert_allclose(A_enc, A_ref, atol=1e-12)


# ── #2b  K=5 on 3 ancillas ─────────────────────────────────────────


def test_lcu_handles_K5_3_ancillas():
    """K=5 shift ops on 3 ancillas: verifies non-palindromic states."""
    from burgers_mpo import increment_circuit, decrement_circuit

    q = 3
    N = 2**q
    I_qc = QuantumCircuit(q, name="I")
    I_qc.id(list(range(q)))
    sp = increment_circuit(q)
    sm = decrement_circuit(q)
    sp2 = QuantumCircuit(q, name="S+2")
    sp2.compose(increment_circuit(q), inplace=True)
    sp2.compose(increment_circuit(q), inplace=True)
    sm2 = QuantumCircuit(q, name="S-2")
    sm2.compose(decrement_circuit(q), inplace=True)
    sm2.compose(decrement_circuit(q), inplace=True)

    coeffs = np.array([0.5, 0.2, 0.2, 0.05, 0.05])
    qc, lam = lcu_block_encoding(
        [I_qc, sp, sm, sp2, sm2], coeffs, q,
    )
    U = Operator(qc).data
    n_anc = qc.num_qubits - q

    A_enc = extract_block_encoded_operator(
        U, n_system=q, n_ancilla=n_anc, anc_in=0, anc_out=0,
    )
    A_ref = (
        0.5 * np.eye(N)
        + 0.2 * Operator(sp).data
        + 0.2 * Operator(sm).data
        + 0.05 * Operator(sp2).data
        + 0.05 * Operator(sm2).data
    ) / lam

    np.testing.assert_allclose(A_enc, A_ref, atol=1e-12)


# ── #3  gradient LCU vs compute_rhs_shift ──────────────────────────


def test_gradient_lcu_matches_compute_rhs_shift():
    """gradient_lcu_circuit post-selected matches classical shift grad."""
    from qiskit.quantum_info import Statevector

    q = 5
    N = 2**q
    dx = 1.0 / N
    x = np.linspace(0, 1, N, endpoint=False)
    u = np.sin(2 * np.pi * x)
    u_norm = u / np.linalg.norm(u)

    # LCU circuit
    grad_qc = gradient_lcu_circuit(q)
    sv_in = np.zeros(2 * N, dtype=complex)
    sv_in[:N] = u_norm
    sv_out = Statevector(sv_in).evolve(grad_qc).data

    # Post-select ancilla = |1> (antisymmetric block)
    result_lcu = sv_out[N:2 * N]

    # Classical reference: (S+ - S-)/2 applied to u_norm
    sp = shift_matrix(N, +1)
    sm = shift_matrix(N, -1)
    result_ref = (sp - sm) @ u_norm / 2.0

    # Norm-compare (block encoding introduces subnormalization)
    nrm_lcu = np.linalg.norm(result_lcu)
    nrm_ref = np.linalg.norm(result_ref)
    assert nrm_lcu > 1e-10, "post-selected state has zero norm"

    np.testing.assert_allclose(
        result_lcu / nrm_lcu,
        result_ref / nrm_ref,
        atol=1e-10,
    )


# ── #6  CH-LCU propagator matches dense-block at q=5 ──────────────


def test_ch_lcu_propagator_matches_dense_block_at_q5():
    """LCU SV vs dense-block SV at q=5, nu=0.1, 20 steps."""
    from burgers_classical import initial_condition_sine
    from burgers_cole_hopf_circuit import run_cole_hopf_circuit_simulation

    q, nu = 5, 0.1
    N = 1 << q
    x = np.linspace(0, 1, N, endpoint=False)
    dx = x[1] - x[0]
    dt = 0.1 * dx
    n_steps = 20
    u0 = initial_condition_sine(x)

    sols_lcu, _ = run_cole_hopf_circuit_simulation(
        u0, x, nu, dt, n_steps,
        propagator="lcu", shots=0,
        snapshot_interval=n_steps,
        taylor_order=4,
    )
    sols_db, _ = run_cole_hopf_circuit_simulation(
        u0, x, nu, dt, n_steps,
        propagator="dense-block", shots=0,
        snapshot_interval=n_steps,
    )

    u_lcu = sols_lcu[-1]
    u_db = sols_db[-1]
    rel_err = (
        np.linalg.norm(u_lcu - u_db)
        / max(np.linalg.norm(u_db), 1e-15)
    )
    assert rel_err < 0.02, (
        f"LCU vs dense-block: {rel_err:.4e} >= 0.02"
    )


# ── #8  P_success above threshold ─────────────────────────────────


def test_ch_lcu_p_success_above_threshold():
    """LCU SV per-step P_success > 0.3 at q=5, nu=0.1."""
    from burgers_classical import initial_condition_sine
    from burgers_cole_hopf import cole_hopf_forward
    from burgers_cole_hopf_circuit import run_cole_hopf_circuit_sv

    q, nu = 5, 0.1
    N = 1 << q
    x = np.linspace(0, 1, N, endpoint=False)
    dx = x[1] - x[0]
    dt = 0.1 * dx
    L_box = float(N * dx)
    n_steps = 10

    u0 = initial_condition_sine(x)
    phi0 = cole_hopf_forward(u0, dx, nu)
    psi0 = phi0 / np.linalg.norm(phi0)

    _, mets = run_cole_hopf_circuit_sv(
        psi0, q, nu, dt, n_steps, L_box,
        propagator="lcu", taylor_order=4,
    )

    for m in mets:
        p = m["p_success_step"]
        assert p > 0.3, (
            f"step {m['step']}: P_success={p:.3f} < 0.3"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
