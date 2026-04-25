"""Tests for classical Burgers solver and MPS state preparation.

Validates against results from Murali et al. AIAA 2026:
- Stage 1: Classical FTCS solver produces stable evolution (Fig. 1/11)
- Stage 2: MPS encoding matches ground truth at various bond dims (Fig. 7)
- Stage 2: MPS circuit produces correct quantum state
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

from burgers_classical import (
    build_gradient_matrix,
    build_laplacian_matrix,
    euler_step,
    gradient_central,
    initial_condition_sine,
    laplacian_central,
    solve_burgers,
    source_term_sine,
)
from burgers_mps import (
    classical_to_mps,
    extract_physical_state,
    mps_bond_dimensions,
    mps_circuit_fidelity,
    mps_fidelity,
    mps_to_circuit,
    normalize_state,
    reconstruct_from_mps,
)


# ---- Stage 1: Classical solver tests ----


class TestFiniteDifference:
    """Verify gradient and Laplacian against known analytical results."""

    def test_gradient_sine(self):
        """d/dx sin(2*pi*x) = 2*pi*cos(2*pi*x)."""
        N = 256
        x = np.linspace(0, 1, N, endpoint=False)
        dx = x[1] - x[0]
        u = np.sin(2 * np.pi * x)

        du_numerical = gradient_central(u, dx)
        du_exact = 2 * np.pi * np.cos(2 * np.pi * x)

        # Central difference is O(dx^2), so error ~ dx^2 ~ 1.5e-5
        max_err = np.max(np.abs(du_numerical[1:-1] - du_exact[1:-1]))
        assert max_err < 1e-3, f"Gradient error {max_err:.2e} too large"

    def test_laplacian_sine(self):
        """d2/dx2 sin(2*pi*x) = -(2*pi)^2 sin(2*pi*x)."""
        N = 256
        x = np.linspace(0, 1, N, endpoint=False)
        dx = x[1] - x[0]
        u = np.sin(2 * np.pi * x)

        d2u_numerical = laplacian_central(u, dx)
        d2u_exact = -(2 * np.pi) ** 2 * np.sin(2 * np.pi * x)

        max_err = np.max(np.abs(d2u_numerical[1:-1] - d2u_exact[1:-1]))
        assert max_err < 1.0, f"Laplacian error {max_err:.2e} too large"

    def test_gradient_matrix_matches_function(self):
        """Matrix form of gradient matches the direct computation."""
        N = 16
        dx = 1.0 / N
        u = np.sin(2 * np.pi * np.linspace(0, 1, N, endpoint=False))

        J = build_gradient_matrix(N, dx)
        du_mat = J @ u
        du_fn = gradient_central(u, dx)

        np.testing.assert_allclose(du_mat, du_fn, atol=1e-14)

    def test_laplacian_matrix_matches_function(self):
        """Matrix form of Laplacian matches the direct computation."""
        N = 16
        dx = 1.0 / N
        u = np.sin(2 * np.pi * np.linspace(0, 1, N, endpoint=False))

        H = build_laplacian_matrix(N, dx)
        d2u_mat = H @ u
        d2u_fn = laplacian_central(u, dx)

        np.testing.assert_allclose(d2u_mat, d2u_fn, atol=1e-14)


class TestEulerSolver:
    """Test the FTCS Burgers solver."""

    def test_single_step_small_change(self):
        """One Euler step should produce a small perturbation."""
        N = 64
        x = np.linspace(0, 1, N, endpoint=False)
        dx = x[1] - x[0]
        u0 = initial_condition_sine(x)
        g = source_term_sine(x, 0.0)
        nu = 1e-4
        dt = 0.1 * dx

        u1 = euler_step(u0, dx, dt, nu, g)
        delta = np.max(np.abs(u1 - u0))
        assert delta < 0.1, f"Single step delta {delta:.4f} too large"
        assert delta > 0, "Single step should change the state"

    def test_energy_bounded(self):
        """Solution should not blow up over short simulation."""
        N = 200
        x = np.linspace(0, 1, N, endpoint=False)
        nu = 1e-4
        dx = x[1] - x[0]
        dt = 0.1 * dx
        n_steps = 100

        u0 = initial_condition_sine(x)
        solutions = solve_burgers(
            u0, x, nu, dt, n_steps, source_fn=source_term_sine
        )

        max_u = max(np.max(np.abs(s)) for s in solutions)
        assert max_u < 10.0, f"Solution blew up: max |u| = {max_u:.2f}"

    def test_zero_source_diffusion(self):
        """With no source and small nu, solution should diffuse slowly."""
        N = 64
        x = np.linspace(0, 1, N, endpoint=False)
        nu = 0.01
        dx = x[1] - x[0]
        dt = 0.1 * dx
        n_steps = 50

        u0 = initial_condition_sine(x)
        solutions = solve_burgers(u0, x, nu, dt, n_steps, source_fn=None)

        # Energy should decrease due to diffusion
        energy_0 = np.sum(solutions[0] ** 2)
        energy_f = np.sum(solutions[-1] ** 2)
        assert energy_f < energy_0, "Diffusion should decrease energy"


# ---- Stage 2: MPS tests ----


class TestMPSDecomposition:
    """Test MPS state vector decomposition and reconstruction."""

    @pytest.mark.parametrize("q", [3, 4, 5])
    def test_full_rank_reconstruction(self, q: int):
        """Full-rank MPS should exactly reconstruct the state."""
        N = 2**q
        x = np.linspace(0, 1, N, endpoint=False)
        u = np.sin(2 * np.pi * x)
        u_norm, _ = normalize_state(u)

        tensors = classical_to_mps(u_norm)
        u_recon = reconstruct_from_mps(tensors)
        max_err = np.max(np.abs(u_norm - u_recon))
        assert max_err < 1e-14, f"Recon error {max_err:.2e}"

    def test_bond_dim_truncation(self):
        """Truncated MPS should have bounded bond dimensions."""
        N = 64
        x = np.linspace(0, 1, N, endpoint=False)
        u = np.sin(2 * np.pi * x)
        u_norm, _ = normalize_state(u)

        for bd in [2, 4, 8]:
            tensors = classical_to_mps(u_norm, bond_dim=bd)
            bonds = mps_bond_dimensions(tensors)
            assert all(
                d <= bd for d in bonds
            ), f"Bond dim exceeded {bd}: {bonds}"

    def test_sine_bond_dim_2_sufficient(self):
        """Sine wave should be well-represented with bond dim 2.

        Paper Fig. 7: threshold ~0.001 gives near-exact for sine.
        """
        N = 256
        x = np.linspace(0, 1, N, endpoint=False)
        u = np.sin(2 * np.pi * x)

        max_err, fid, bonds = mps_fidelity(u, bond_dim=2)
        assert max_err < 1e-14, f"Sine with bd=2: err={max_err:.2e}"
        assert fid > 1 - 1e-10

    def test_threshold_convergence(self):
        """Error should decrease as threshold decreases (Fig. 7)."""
        N = 256
        x = np.linspace(0, 1, N, endpoint=False)
        u = np.sin(2 * np.pi * x)

        thresholds = [0.5, 0.05, 0.001]
        errors = []
        for th in thresholds:
            err, _, _ = mps_fidelity(u, threshold=th)
            errors.append(err)

        # Errors should be monotonically decreasing
        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i + 1], (
                f"Error not decreasing: "
                f"th={thresholds[i]} err={errors[i]:.2e} vs "
                f"th={thresholds[i+1]} err={errors[i+1]:.2e}"
            )

    def test_right_canonical_reconstruction(self):
        """Right-canonical MPS should also reconstruct exactly."""
        N = 16
        x = np.linspace(0, 1, N, endpoint=False)
        u = np.sin(2 * np.pi * x)
        u_norm, _ = normalize_state(u)

        tensors = classical_to_mps(u_norm, canonical="right")
        u_recon = reconstruct_from_mps(tensors)
        max_err = np.max(np.abs(u_norm - u_recon))
        assert max_err < 1e-14, f"Right-canonical recon error {max_err:.2e}"

    def test_right_canonical_isometry(self):
        """Right-canonical tensors should satisfy sum_j A^j A^{j†} = I."""
        N = 16
        x = np.linspace(0, 1, N, endpoint=False)
        u = np.sin(2 * np.pi * x)
        u_norm, _ = normalize_state(u)

        tensors = classical_to_mps(u_norm, canonical="right")
        for k, A in enumerate(tensors):
            d_left, K, d_right = A.shape
            check = np.zeros((d_left, d_left), dtype=complex)
            for j in range(K):
                Aj = A[:, j, :]
                check += Aj @ Aj.conj().T
            np.testing.assert_allclose(
                check,
                np.eye(d_left),
                atol=1e-12,
                err_msg=f"Isometry failed at site {k}",
            )


class TestMPSCircuit:
    """Test MPS-to-quantum-circuit conversion."""

    @pytest.mark.parametrize("q", [3, 4])
    def test_circuit_fidelity_full_rank(self, q: int):
        """Full-rank MPS circuit should produce exact state."""
        N = 2**q
        x = np.linspace(0, 1, N, endpoint=False)
        u = np.sin(2 * np.pi * x)

        max_err, fid = mps_circuit_fidelity(u)
        assert fid > 1 - 1e-10, f"Fidelity {fid:.10f} too low"
        assert max_err < 1e-10, f"Max error {max_err:.2e} too high"

    def test_circuit_fidelity_truncated(self):
        """Truncated MPS circuit should still achieve high fidelity."""
        N = 8
        x = np.linspace(0, 1, N, endpoint=False)
        u = np.sin(2 * np.pi * x)

        max_err, fid = mps_circuit_fidelity(u, bond_dim=2)
        assert fid > 1 - 1e-10, f"Truncated fidelity {fid:.10f}"

    def test_circuit_qubit_count(self):
        """Circuit should use q + ceil(log2(max_bond)) qubits."""
        N = 8
        x = np.linspace(0, 1, N, endpoint=False)
        u = np.sin(2 * np.pi * x)
        u_norm, _ = normalize_state(u)

        tensors = classical_to_mps(u_norm, bond_dim=2, canonical="right")
        qc = mps_to_circuit(tensors)

        q = 3
        max_bond = max(
            max(t.shape[0], t.shape[2]) for t in tensors
        )
        expected_bond_qubits = int(np.ceil(np.log2(max(max_bond, 2))))
        assert qc.num_qubits == q + expected_bond_qubits

    def test_bond_register_disentangled(self):
        """After exact MPS circuit, bond register should be in |0>."""
        from qiskit.quantum_info import Statevector

        N = 8
        x = np.linspace(0, 1, N, endpoint=False)
        u = np.sin(2 * np.pi * x)
        u_norm, _ = normalize_state(u)

        tensors = classical_to_mps(u_norm, canonical="right")
        qc = mps_to_circuit(tensors)

        q = 3
        n_bond = qc.num_qubits - q

        sv = Statevector.from_instruction(qc)
        amps = np.array(sv.data)

        # All amplitudes with bond != 0 should be zero
        for i in range(len(amps)):
            bond_bits = i >> q  # upper bits = bond register
            if bond_bits != 0:
                assert abs(amps[i]) < 1e-12, (
                    f"Bond not disentangled: "
                    f"|{i:0{qc.num_qubits}b}> = {amps[i]:.6f}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
