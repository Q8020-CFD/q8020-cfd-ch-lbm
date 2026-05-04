"""F10-11 MPS state-prep acceptance tests.

Two acceptance criteria:
  A. MPS state prep produces a normalized statevector  (||psi||^2 == 1).
  B. MPS prep with bond-dim truncation gives reasonable fidelity
     relative to the exact (full-rank) preparation.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from burgers_mps import (
    classical_to_mps,
    mps_to_circuit,
    normalize_state,
    mps_fidelity,
    mps_circuit_fidelity,
)
from burgers_cole_hopf import cole_hopf_forward
from burgers_classical import initial_condition_sine


def _make_phi(q: int, nu: float = 1e-2) -> np.ndarray:
    """Cole-Hopf forward transform of the sine IC."""
    N = 1 << q
    x = np.linspace(0, 1, N, endpoint=True)
    dx = x[1] - x[0]
    u0 = initial_condition_sine(x)
    return cole_hopf_forward(u0, dx, nu=nu)


# ── A. Normalization ────────────────────────────────────────────────────────


@pytest.mark.parametrize("q,bond_dim", [
    (4, None),   # full rank
    (4, 4),      # exact for q=4 (max bond = 4)
    (4, 2),      # mild truncation
    (5, None),   # full rank
    (5, 8),      # exact for q=5 (max bond = 8)
    (5, 4),      # mild truncation
])
def test_mps_prep_statevector_normalized(q, bond_dim):
    """Circuit-prepared statevector has unit norm to floating-point precision."""
    from qiskit.quantum_info import Statevector

    phi = _make_phi(q)
    psi_norm, _ = normalize_state(phi)
    tensors = classical_to_mps(psi_norm, bond_dim=bond_dim, canonical="right")
    qc = mps_to_circuit(tensors)

    n_bond = qc.num_qubits - q
    sv = Statevector.from_label("0" * (q + n_bond)).evolve(qc).data
    # Physical qubits are the low n_phys amplitudes (Qiskit little-endian)
    psi_phys = sv[: 1 << q]

    norm_sq = float(np.real(np.vdot(psi_phys, psi_phys)))
    assert abs(norm_sq - 1.0) < 1e-10, (
        f"q={q}, bond_dim={bond_dim}: ||psi||^2 = {norm_sq:.6e} (expected 1)"
    )


# ── B. Fidelity under truncation ────────────────────────────────────────────


@pytest.mark.parametrize("q,bond_dim,min_fidelity", [
    # q=4: max bond = 4, so chi>=4 is exact
    (4, 4, 0.9999),
    (4, 2, 0.90),
    # q=5: smooth sine IC is highly compressible; chi=4 still high-fidelity
    (5, 8, 0.9999),
    (5, 4, 0.99),
    (5, 2, 0.80),
])
def test_mps_fidelity_truncation(q, bond_dim, min_fidelity):
    """MPS reconstruction fidelity meets the per-chi floor."""
    phi = _make_phi(q)
    _, fidelity, bonds = mps_fidelity(phi, bond_dim=bond_dim)
    assert fidelity >= min_fidelity, (
        f"q={q}, bond_dim={bond_dim}: fidelity={fidelity:.6f} < {min_fidelity} "
        f"(bonds={bonds})"
    )


@pytest.mark.parametrize("q,chi_lo,chi_hi", [
    (4, 2, 4),
    (5, 2, 4),
    (5, 4, 8),
])
def test_mps_fidelity_monotone_in_bond_dim(q, chi_lo, chi_hi):
    """Higher bond dimension gives equal or better fidelity."""
    phi = _make_phi(q)
    _, fid_lo, _ = mps_fidelity(phi, bond_dim=chi_lo)
    _, fid_hi, _ = mps_fidelity(phi, bond_dim=chi_hi)
    assert fid_hi >= fid_lo - 1e-12, (
        f"q={q}: fidelity({chi_hi})={fid_hi:.6f} < fidelity({chi_lo})={fid_lo:.6f}"
    )


@pytest.mark.parametrize("q,bond_dim,min_fidelity", [
    (4, 4, 0.9999),
    (4, 2, 0.85),
    (5, 4, 0.95),
])
def test_mps_circuit_fidelity_truncation(q, bond_dim, min_fidelity):
    """Full circuit-level fidelity (simulate the prep circuit) meets floor."""
    phi = _make_phi(q)
    _, fidelity = mps_circuit_fidelity(phi, bond_dim=bond_dim)
    assert fidelity >= min_fidelity, (
        f"q={q}, bond_dim={bond_dim}: circuit fidelity={fidelity:.6f} < {min_fidelity}"
    )
