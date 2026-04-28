"""Tests for LCU source forcing (Strang-split CH-LCU).

Covers SPEC-F3-LCU-source-forcing.md §5 tests #2 and #4 — the two
load-bearing correctness gates.  Other tests (#1 unit, #3
dt^2 scaling, #5 vs FTCS, #6 p_success) deferred for follow-up.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from qiskit.quantum_info import Operator
from scipy.linalg import expm

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "src",
    ),
)

from burgers_cole_hopf import build_laplacian_dense
from burgers_lcu import (
    diag_potential_block_encoding,
    heat_lcu_with_potential_step_circuit,
    heat_taylor_lcu_terms,
    lcu_block_encoding,
)
from burgers_mpo import extract_block_encoded_operator


# ── #2  Strang block-encoded operator matches Taylor-truncated math ──


def test_strang_step_correctness_unitary():
    """Block-encoded Strang LCU equals
        exp(-V*dt/2) @ P_M(nu*L*dt) @ exp(-V*dt/2) / lam_total
    where P_M is the truncated-Taylor heat (NOT exp(nu*L*dt)).

    Compares the *full* unitary (no measurement)'s |all-ancilla=0>
    block to the math.  This catches the V-ancilla
    composition bug — if the two V-halves share an ancilla
    coherently, the operator is wrong by O(1).
    """
    q = 4
    N = 1 << q
    L_box = 1.0
    dx = L_box / N
    nu = 0.1
    dt = 0.001
    taylor_order = 4

    # Paper-shape V at t=0: cos(2*pi*x)*1/(4*pi*nu), gauge mean-zero.
    x = np.linspace(0, 1, N, endpoint=False)
    V = -np.cos(2 * np.pi * x) / (4 * np.pi * nu)
    V = V - V.mean()

    qc, lam_total = heat_lcu_with_potential_step_circuit(
        q, nu, dt, L_box, V, taylor_order=taylor_order,
    )

    n_anc = qc.num_qubits - q
    U = Operator(qc).data

    # Block-encoded operator: project ancillas to |0>^n_anc on both
    # input and output.
    A_enc = extract_block_encoded_operator(
        U, n_system=q, n_ancilla=n_anc, anc_in=0, anc_out=0,
    )

    # Reference: classical Strang of Taylor-truncated heat
    # propagator with the *same* potential layer.
    L_dense = build_laplacian_dense(N, dx, bc="periodic")

    # Reconstruct truncated Taylor P_M from heat_taylor_lcu_terms
    # (this is the heat operator the LCU actually encodes).
    ops, coeffs = heat_taylor_lcu_terms(
        q, nu, dt, L_box, taylor_order=taylor_order,
    )
    P_M = np.zeros((N, N), dtype=complex)
    for op, c in zip(ops, coeffs):
        P_M += c * Operator(op).data
    # The LCU encodes P_M / lam_heat — recover lam_heat.
    _, lam_heat = lcu_block_encoding(ops, coeffs, q)

    # V-half block-encoded operator is B_V = diag(exp(-V*dt/2))/s_max.
    # Two of them around the heat block-encoding B_heat = P_M/lam_heat:
    #     B_V · B_heat · B_V = (diag/s_max) · (P_M/lam_heat) · (diag/s_max)
    #                       = (diag · P_M · diag) / (lam_heat · s_max^2)
    diag_V_half = np.exp(-V * dt / 2.0)
    s_max_V = float(np.max(np.abs(diag_V_half)))
    diag_M = np.diag(diag_V_half)
    A_ref = (diag_M @ P_M @ diag_M) / (lam_heat * (s_max_V ** 2))

    # Sanity: lam_total returned by builder should match
    assert lam_total == pytest.approx(
        lam_heat * (s_max_V ** 2), rel=1e-12,
    )

    np.testing.assert_allclose(
        A_enc, A_ref, atol=1e-10,
        err_msg=(
            "Strang block-encoded operator does not match math.  "
            "If this fails, the most likely cause is the V-anc "
            "composition bug (two halves sharing one ancilla "
            "coherently)."
        ),
    )


# ── #4  CH-LCU forced SV matches dense-block forced SV at q=5 ─────


def test_ch_lcu_with_source_matches_dense_block_sv():
    """Forced LCU (Strang) vs dense-block (exact per-step) SV at
    q=5, nu=0.1, source=sine, n_steps=10.  Both solve the same
    forced PDE.  LCU's Taylor truncation + Strang dt^2 error is
    bounded by ~0.05 at this case size."""
    from burgers_classical import (
        initial_condition_sine, source_term_sine,
    )
    from burgers_cole_hopf_circuit import (
        run_cole_hopf_circuit_simulation,
    )

    q, nu = 5, 0.1
    N = 1 << q
    x = np.linspace(0, 1, N, endpoint=False)
    dx = x[1] - x[0]
    dt = 0.1 * dx
    n_steps = 10
    u0 = initial_condition_sine(x)

    sols_lcu, _ = run_cole_hopf_circuit_simulation(
        u0, x, nu, dt, n_steps,
        propagator="lcu", shots=0,
        snapshot_interval=n_steps,
        taylor_order=4,
        source_fn=source_term_sine,
    )
    sols_db, _ = run_cole_hopf_circuit_simulation(
        u0, x, nu, dt, n_steps,
        propagator="dense-block", shots=0,
        snapshot_interval=n_steps,
        source_fn=source_term_sine,
    )

    u_lcu = sols_lcu[-1]
    u_db = sols_db[-1]
    rel_err = (
        np.linalg.norm(u_lcu - u_db)
        / max(np.linalg.norm(u_db), 1e-15)
    )
    assert rel_err < 0.05, (
        f"forced LCU vs dense-block: {rel_err:.4e} >= 0.05"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
