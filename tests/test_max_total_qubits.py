"""Tests for the --max-total-qubits budget plumbing (fast/slow Aer path).

The budget bounds the COMPOSED segment circuit width: data (q) + MPS
bond (n_bond) + heat ancillas.  heat_qft_full_circuit's internal
max_qubits counts only data + heat -- it is built before the prep
circuit is composed alongside it -- so build_segment_circuit must
subtract n_bond before passing the budget down.  These tests pin that
offset: a budget passed through un-shifted would produce a composed
circuit n_bond qubits wider than the cap.

Run with:
    pytest tests/test_max_total_qubits.py -v
Slow tests (shots execution) can be deselected with -m 'not slow'.
"""

from __future__ import annotations

import subprocess
import sys
import os

import numpy as np
import pytest

# Ensure the solver package is importable
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"),
)

from q8020_backend_utils.ibm.backend import get_backend

from lib_classical import initial_condition_sine
from lib_cole_hopf_circuit import (
    build_segment_circuit,
    run_cole_hopf_circuit_simulation,
)


# ── helpers ──────────────────────────────────────────────────────────


def _make_grid(q: int):
    N = 1 << q
    x = np.linspace(0, 1, N, endpoint=False)
    return x, x[1] - x[0], N


def _smooth_state(q: int) -> np.ndarray:
    """Unit-norm smooth positive state (Gaussian bump)."""
    N = 1 << q
    x = np.linspace(0, 1, N, endpoint=False)
    psi = np.exp(-0.5 * ((x - 0.5) / 0.2) ** 2)
    return psi / np.linalg.norm(psi)


def _build(q, k, max_total_qubits, bond_dim=4):
    """build_segment_circuit with study-typical physics scalars."""
    psi = _smooth_state(q)
    return build_segment_circuit(
        psi, q, nu=0.03, dt=1e-3, segment_size=k, L_box=1.0,
        bc="periodic", bond_dim=bond_dim,
        max_total_qubits=max_total_qubits,
    )


def _n_resets(qc) -> int:
    return qc.count_ops().get("reset", 0)


# ── budget semantics on the composed segment circuit ────────────────


def test_default_budget_unchanged():
    """max_total_qubits=None keeps the original 2*q heat default."""
    q, k = 4, 8
    raw_qc, total_q, n_bond, n_heat_anc = _build(q, k, None)
    # Default heat budget is 2*q (q ancillas); block = min(q, k).
    assert n_heat_anc == min(q, k)
    assert total_q == q + n_bond + n_heat_anc
    assert raw_qc.num_qubits == total_q


def test_fully_deferred_no_resets():
    """Budget covering q + n_bond + k gives a reset-free segment."""
    q, k = 4, 4
    _, _, n_bond, _ = _build(q, k, None)
    budget = q + n_bond + k
    raw_qc, total_q, n_bond2, n_heat_anc = _build(q, k, budget)
    assert n_bond2 == n_bond
    assert n_heat_anc == k, "fully deferred: one ancilla per step"
    assert _n_resets(raw_qc) == 0, "fast path must have no mid-circuit reset"
    assert total_q == budget
    assert raw_qc.num_qubits == budget


def test_capped_budget_inserts_resets_and_respects_cap():
    """One-under-fast budget: resets appear AND the cap holds.

    Regression for the bond-qubit offset: passing the total budget
    straight into heat_qft_full_circuit (which counts only data + heat)
    would build a q + n_bond + k circuit here -- n_bond qubits OVER
    budget -- because its internal max(q + budget-q, ...) leaves room
    for all k ancillas once n_bond is not subtracted.
    """
    q, k = 4, 4
    _, _, n_bond, _ = _build(q, k, None)
    assert n_bond > 0, "MPS prep at bond_dim=4 must use bond qubits"
    budget = q + n_bond + k - 1
    raw_qc, total_q, _, n_heat_anc = _build(q, k, budget)
    assert _n_resets(raw_qc) > 0, "under-budget segment must recycle ancillas"
    assert n_heat_anc == k - 1
    assert total_q == budget
    assert raw_qc.num_qubits <= budget, (
        f"composed circuit ({raw_qc.num_qubits}q) exceeds "
        f"--max-total-qubits={budget}: bond offset not applied"
    )


def test_budget_sweep_never_exceeded():
    """Every legal budget yields a composed circuit within the cap."""
    q, k = 3, 4
    _, _, n_bond, _ = _build(q, k, None)
    for budget in range(q + n_bond + 1, q + n_bond + k + 2):
        raw_qc, total_q, _, n_heat_anc = _build(q, k, budget)
        assert raw_qc.num_qubits <= budget
        assert raw_qc.num_qubits == total_q
        # Width = min(budget headroom, one ancilla per step).
        assert n_heat_anc == min(budget - q - n_bond, k)


def test_budget_too_small_raises():
    """A budget with no room for even one heat ancilla is an error."""
    q, k = 4, 4
    _, _, n_bond, _ = _build(q, k, None)
    with pytest.raises(ValueError, match="max_total_qubits"):
        _build(q, k, q + n_bond)


def test_initialize_prep_has_no_bond_offset():
    """use_mps_prep=False (initialize): n_bond=0, budget maps 1:1."""
    q, k = 3, 3
    psi = _smooth_state(q)
    budget = q + k
    raw_qc, total_q, n_bond, n_heat_anc = build_segment_circuit(
        psi, q, nu=0.03, dt=1e-3, segment_size=k, L_box=1.0,
        bc="periodic", use_mps_prep=False,
        max_total_qubits=budget,
    )
    assert n_bond == 0
    assert n_heat_anc == k
    assert _n_resets(raw_qc) == 0
    assert raw_qc.num_qubits == budget


# ── fast path / SLOW PATH diagnostic print ───────────────────────────


def _run_reprepare(q, segment_size, max_total_qubits, shots=256):
    x, dx, N = _make_grid(q)
    u0 = initial_condition_sine(x) * 0.3
    n_steps = 2 * segment_size
    return run_cole_hopf_circuit_simulation(
        u0, x, nu=0.1, dt=0.1 * dx, n_steps=n_steps,
        shots=shots, snapshot_interval=n_steps,
        evolution_mode="measure_reprepare", segment_size=segment_size,
        bond_dim=4, seed=7,
        backend=get_backend(backend_type="sim"),
        max_total_qubits=max_total_qubits,
    )


def test_fast_path_printed(capfd):
    q, k = 3, 2
    _, _, n_bond, _ = _build(q, k, None)
    _run_reprepare(q, k, max_total_qubits=q + n_bond + k)
    err = capfd.readouterr().err
    assert "fast path" in err
    assert "SLOW PATH" not in err


def test_slow_path_printed(capfd):
    q, k = 3, 3
    _, _, n_bond, _ = _build(q, k, None)
    _run_reprepare(q, k, max_total_qubits=q + n_bond + 1)
    err = capfd.readouterr().err
    assert "SLOW PATH" in err


# ── physics: the budget changes cost, not results ────────────────────


@pytest.mark.slow
def test_capped_vs_deferred_same_physics():
    """Capped (reset-recycling) and deferred segments agree with the
    SV reference to the same tolerance -- blocking is cost-only."""
    q, k = 4, 4
    x, dx, N = _make_grid(q)
    u0 = initial_condition_sine(x)
    nu, n_steps = 0.1, 8
    dt = 0.1 * dx

    sols_ref, _ = run_cole_hopf_circuit_simulation(
        u0, x, nu, dt, n_steps, shots=0, snapshot_interval=n_steps,
    )
    u_ref = sols_ref[-1]

    _, _, n_bond, _ = _build(q, k, None)
    errs = {}
    for label, budget in [
        ("deferred", q + n_bond + k),
        ("capped", q + n_bond + 2),
    ]:
        sols, _ = run_cole_hopf_circuit_simulation(
            u0, x, nu, dt, n_steps,
            shots=50000, snapshot_interval=n_steps,
            evolution_mode="measure_reprepare", segment_size=k,
            bond_dim=4, seed=1234,
            backend=get_backend(backend_type="sim"),
            max_total_qubits=budget,
        )
        u = sols[-1]
        assert np.all(np.isfinite(u)), f"{label}: non-finite u"
        errs[label] = (
            np.linalg.norm(u - u_ref) / max(np.linalg.norm(u_ref), 1e-15)
        )

    for label, err in errs.items():
        assert err < 0.15, f"{label}: relL2 vs SV {err:.4f} >= 0.15"


# ── CLI plumbing ─────────────────────────────────────────────────────


def test_cli_exposes_flag():
    """burgers_solver.py --help advertises --max-total-qubits."""
    solver = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "src", "burgers_solver.py",
    )
    res = subprocess.run(
        [sys.executable, solver, "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert res.returncode == 0, res.stderr
    assert "--max-total-qubits" in res.stdout


def test_config_carries_field():
    """BurgersConfig exposes max_total_qubits (default None)."""
    from lib_fw import BurgersConfig
    cfg = BurgersConfig()
    assert cfg.max_total_qubits is None
    cfg26 = BurgersConfig(max_total_qubits=26)
    assert cfg26.max_total_qubits == 26
