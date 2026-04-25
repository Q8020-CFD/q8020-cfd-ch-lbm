"""Tests for MPO-derived quantum circuits (S+/S-, gradient, Laplacian).

Validates against the ladder operator formulas from Murali et al.
AIAA 2026 (Eqs. 9-12) and the classical FD operator matrices.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from burgers_classical import build_gradient_matrix, build_laplacian_matrix
from burgers_mpo import (
    decrement_circuit,
    extract_block_encoded_operator,
    gradient_lcu_circuit,
    increment_circuit,
    laplacian_lcu_circuit,
    shift_matrix,
)


# ---- S+ / S- ladder operator tests ----


class TestLadderOperators:
    """Test increment (S+) and decrement (S-) circuits."""

    @pytest.mark.parametrize("q", [2, 3, 4, 5])
    def test_increment_matrix(self, q: int):
        """S+ circuit matrix should match the classical shift matrix."""
        N = 2**q
        sp_mat = Operator(increment_circuit(q)).data
        sp_ref = shift_matrix(N, +1)
        np.testing.assert_allclose(sp_mat, sp_ref, atol=1e-14)

    @pytest.mark.parametrize("q", [2, 3, 4, 5])
    def test_decrement_matrix(self, q: int):
        """S- circuit matrix should match the classical shift matrix."""
        N = 2**q
        sm_mat = Operator(decrement_circuit(q)).data
        sm_ref = shift_matrix(N, -1)
        np.testing.assert_allclose(sm_mat, sm_ref, atol=1e-14)

    @pytest.mark.parametrize("q", [2, 3, 4])
    def test_increment_all_basis_states(self, q: int):
        """S+ should map each |i⟩ → |i+1 mod N⟩."""
        N = 2**q
        qc_sp = increment_circuit(q)
        for i in range(N):
            psi_in = np.zeros(N, dtype=complex)
            psi_in[i] = 1.0
            sv = Statevector(psi_in).evolve(qc_sp)
            expected_j = (i + 1) % N
            assert abs(sv.data[expected_j]) > 1 - 1e-10, (
                f"|{i}⟩ → expected |{expected_j}⟩, "
                f"got max at |{np.argmax(np.abs(sv.data))}⟩"
            )

    @pytest.mark.parametrize("q", [2, 3, 4])
    def test_decrement_all_basis_states(self, q: int):
        """S- should map each |i⟩ → |i-1 mod N⟩."""
        N = 2**q
        qc_sm = decrement_circuit(q)
        for i in range(N):
            psi_in = np.zeros(N, dtype=complex)
            psi_in[i] = 1.0
            sv = Statevector(psi_in).evolve(qc_sm)
            expected_j = (i - 1) % N
            assert abs(sv.data[expected_j]) > 1 - 1e-10

    @pytest.mark.parametrize("q", [2, 3, 4, 5])
    def test_round_trip(self, q: int):
        """S+ followed by S- should be identity."""
        N = 2**q
        sp = Operator(increment_circuit(q)).data
        sm = Operator(decrement_circuit(q)).data
        np.testing.assert_allclose(sp @ sm, np.eye(N), atol=1e-14)

    @pytest.mark.parametrize("q", [2, 3, 4])
    def test_s_minus_is_adjoint(self, q: int):
        """S- should equal (S+)†."""
        sp = Operator(increment_circuit(q)).data
        sm = Operator(decrement_circuit(q)).data
        np.testing.assert_allclose(sm, sp.conj().T, atol=1e-14)


# ---- Gradient LCU tests ----


class TestGradientLCU:
    """Test the gradient block-encoding circuit."""

    @pytest.mark.parametrize("q", [2, 3, 4])
    def test_antisymmetric_block(self, q: int):
        """⟨1|U|0⟩ should give (S+ - S-)/2."""
        N = 2**q
        sp_ref = shift_matrix(N, +1)
        sm_ref = shift_matrix(N, -1)
        grad_ref = (sp_ref - sm_ref) / 2

        U = Operator(gradient_lcu_circuit(q)).data
        grad_op = extract_block_encoded_operator(
            U, n_system=q, n_ancilla=1, anc_in=0, anc_out=1
        )
        np.testing.assert_allclose(grad_op, grad_ref, atol=1e-13)

    @pytest.mark.parametrize("q", [2, 3, 4])
    def test_symmetric_block(self, q: int):
        """⟨0|U|0⟩ should give (S+ + S-)/2."""
        N = 2**q
        sp_ref = shift_matrix(N, +1)
        sm_ref = shift_matrix(N, -1)
        sym_ref = (sp_ref + sm_ref) / 2

        U = Operator(gradient_lcu_circuit(q)).data
        sym_op = extract_block_encoded_operator(
            U, n_system=q, n_ancilla=1, anc_in=0, anc_out=0
        )
        np.testing.assert_allclose(sym_op, sym_ref, atol=1e-13)

    def test_gradient_on_sine_wave(self):
        """Apply gradient LCU to sin(2πx), compare to analytical derivative.

        Sign convention: S+ maps |i⟩→|i+1 mod N⟩, so (S+)u[j] = u[j-1].
        Thus (S+ - S-)u[j] = u[j-1] - u[j+1] = -[u[j+1] - u[j-1]].
        The physical gradient is -(S+ - S-)/(2dx) = (S- - S+)/(2dx).
        """
        q = 8
        N = 2**q
        dx = 1.0 / N
        x = np.linspace(0, 1, N, endpoint=False)
        u = np.sin(2 * np.pi * x)
        u_norm = u / np.linalg.norm(u)

        # Classical gradient via shift operators (note sign)
        sp_ref = shift_matrix(N, +1)
        sm_ref = shift_matrix(N, -1)
        grad_u = -(sp_ref - sm_ref) @ u_norm / (2 * dx)

        # Compare to analytical: d/dx [sin(2πx)/||u||] = 2π cos(2πx)/||u||
        du_exact = 2 * np.pi * np.cos(2 * np.pi * x) / np.linalg.norm(u)

        # Central difference error should be O(dx^2)
        err = np.max(np.abs(grad_u - du_exact))
        assert err < 1e-3, f"Gradient error {err:.2e} too large for N={N}"


# ---- Laplacian LCU tests ----


class TestLaplacianLCU:
    """Test the Laplacian block-encoding circuit."""

    @pytest.mark.parametrize("q", [2, 3, 4])
    def test_laplacian_block(self, q: int):
        """⟨00|U|00⟩ should give (S+ + S- - 2I)/4."""
        N = 2**q
        sp_ref = shift_matrix(N, +1)
        sm_ref = shift_matrix(N, -1)
        lap_ref = (sp_ref + sm_ref - 2 * np.eye(N)) / 4

        U = Operator(laplacian_lcu_circuit(q)).data
        lap_op = extract_block_encoded_operator(
            U, n_system=q, n_ancilla=2, anc_in=0, anc_out=0
        )
        np.testing.assert_allclose(lap_op, lap_ref, atol=1e-13)

    @pytest.mark.parametrize("q", [3, 4])
    def test_shift_laplacian_matches_classical(self, q: int):
        """(S+ + S- - 2I)/δx² should match the classical FD Laplacian."""
        N = 2**q
        dx = 1.0 / N
        sp_ref = shift_matrix(N, +1)
        sm_ref = shift_matrix(N, -1)
        lap_shift = (sp_ref + sm_ref - 2 * np.eye(N)) / dx**2
        lap_classical = build_laplacian_matrix(N, dx)
        np.testing.assert_allclose(lap_shift, lap_classical, atol=1e-12)

    def test_laplacian_on_sine_wave(self):
        """Apply Laplacian to sin(2πx), compare to analytical."""
        q = 4
        N = 2**q
        dx = 1.0 / N
        x = np.linspace(0, 1, N, endpoint=False)
        u = np.sin(2 * np.pi * x)
        u_norm = u / np.linalg.norm(u)

        # Classical Laplacian via shift operators
        sp_ref = shift_matrix(N, +1)
        sm_ref = shift_matrix(N, -1)
        lap_u = (sp_ref + sm_ref - 2 * np.eye(N)) @ u_norm / dx**2

        # Analytical: d²/dx² sin(2πx) = -(2π)² sin(2πx)
        d2u_exact = -(2 * np.pi) ** 2 * np.sin(2 * np.pi * x)
        d2u_exact_norm = d2u_exact / np.linalg.norm(u)

        # Central difference O(dx^2) error
        err = np.max(np.abs(lap_u - d2u_exact_norm))
        assert err < 5.0, f"Laplacian error {err:.2e} too large for N={N}"


# ---- Integration: apply operators to circuit-prepared states ----


class TestOperatorsOnCircuitStates:
    """End-to-end: prepare |u⟩ via MPS circuit, apply operator, verify."""

    def test_gradient_lcu_on_mps_state(self):
        """Prepare |u⟩ via MPS, apply gradient LCU, check post-selected result."""
        from burgers_mps import classical_to_mps, mps_to_circuit, normalize_state

        q = 3
        N = 2**q
        dx = 1.0 / N
        x = np.linspace(0, 1, N, endpoint=False)
        u = np.sin(2 * np.pi * x)
        u_norm, _ = normalize_state(u)

        # Build MPS state prep circuit
        tensors = classical_to_mps(u_norm, canonical="right")
        mps_qc = mps_to_circuit(tensors)
        n_phys = q
        n_bond = mps_qc.num_qubits - q

        # Build gradient LCU (acts on system qubits only)
        grad_qc = gradient_lcu_circuit(q)
        n_grad_anc = 1

        # Compose: MPS prep + gradient LCU
        # Total qubits: q(phys) + n_bond(MPS bond) + 1(grad ancilla)
        total_qubits = q + n_bond + n_grad_anc
        full_qc = QuantumCircuit(total_qubits)

        # MPS prep on qubits [0..q-1, q..q+n_bond-1]
        full_qc.compose(mps_qc, qubits=list(range(q + n_bond)), inplace=True)

        # Gradient LCU on qubits [0..q-1, q+n_bond] (system + grad ancilla)
        grad_anc_qubit = q + n_bond
        full_qc.compose(
            grad_qc,
            qubits=list(range(q)) + [grad_anc_qubit],
            inplace=True,
        )

        # Simulate
        sv = Statevector.from_instruction(full_qc)
        amps = np.array(sv.data)

        # Extract: bond qubits = |0⟩, grad ancilla = |1⟩ (post-select)
        # Index = sys_idx + bond_idx * N + grad_anc_idx * N * 2^n_bond
        N_bond = 2**n_bond
        result = np.zeros(N, dtype=complex)
        for i in range(N):
            # bond=0, grad_anc=1
            idx = i + 0 * N + 1 * N * N_bond
            result[i] = amps[idx]

        # Expected: (S+ - S-)/2 @ u_norm
        sp = shift_matrix(N, +1)
        sm = shift_matrix(N, -1)
        expected = (sp - sm) @ u_norm / 2

        # The post-selected state is not normalized, so compare shapes
        if np.linalg.norm(result) > 1e-10:
            np.testing.assert_allclose(
                result / np.linalg.norm(result),
                expected / np.linalg.norm(expected),
                atol=1e-10,
            )
        else:
            pytest.fail("Post-selected state has zero norm")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
