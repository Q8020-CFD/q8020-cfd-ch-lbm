"""Tests for nonlinear evolution operator (Appendix A) and time stepping.

Validates the Pauli decomposition approach for the nonlinear term u·∇u
and the hybrid classical-quantum time evolution against the classical
shift-operator Euler solver.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest
from qiskit.quantum_info import Operator, Statevector

from burgers_nonlinear import (
    build_evolution_hamiltonian,
    compute_rhs_shift,
    evolution_circuit,
    exact_evolution_matrix,
    generate_pauli_labels,
    quantum_evolution_step,
    solve_pauli_coefficients,
)
from burgers_trotter import (
    compute_error,
    quantum_circuit_step,
    quantum_exact_step,
    run_simulation,
    shift_euler_step,
)


# ---- Pauli decomposition tests ----


class TestPauliDecomposition:
    """Test the Pauli coefficient solver (Eq. 16)."""

    @pytest.mark.parametrize("q", [2, 3])
    def test_hamiltonian_is_hermitian(self, q: int):
        """Â = Σ c_i P̂_i should be Hermitian."""
        N = 2**q
        dx = 1.0 / N
        dt = 0.001
        nu = 1e-4
        x = np.linspace(0, 1, N, endpoint=False)
        u = np.sin(2 * np.pi * x)
        g = np.sin(2 * np.pi * x)

        H = build_evolution_hamiltonian(u, dx, dt, nu, g)
        H_mat = H.to_matrix()
        if hasattr(H_mat, "toarray"):
            H_mat = H_mat.toarray()
        H_mat = np.asarray(H_mat)

        np.testing.assert_allclose(
            H_mat, H_mat.conj().T, atol=1e-14,
            err_msg="Hamiltonian is not Hermitian",
        )

    @pytest.mark.parametrize("q", [2, 3])
    def test_linearization_accuracy(self, q: int):
        """The first-order approximation -iÂu ≈ Δ₀ should be accurate."""
        N = 2**q
        dx = 1.0 / N
        dt = 0.001
        nu = 1e-4
        x = np.linspace(0, 1, N, endpoint=False)
        u = np.sin(2 * np.pi * x)
        g = np.sin(2 * np.pi * x)

        u_norm = u / np.linalg.norm(u)
        rhs = compute_rhs_shift(u, dx, nu, g)
        u_next = u + dt * rhs
        u_next_norm = u_next / np.linalg.norm(u_next)
        delta0 = (u_next_norm - u_norm) / dt

        labels, coeffs = solve_pauli_coefficients(u_norm, delta0, q)

        # Reconstruct Â and compute -iÂu
        H = build_evolution_hamiltonian(u, dx, dt, nu, g)
        H_mat = H.to_matrix()
        if hasattr(H_mat, "toarray"):
            H_mat = H_mat.toarray()
        H_mat = np.asarray(H_mat)

        delta_approx = -1j * H_mat @ u_norm

        # Should match delta0 (the real part, since delta0 is real)
        # Accuracy depends on q: the least-squares fit over 4^q Paulis
        # may not perfectly resolve delta0 for larger systems
        err = np.max(np.abs(delta_approx.real - delta0))
        assert err < 0.01, f"-iÂu vs Δ₀ error: {err:.2e}"

    def test_pauli_labels_count(self):
        """Should generate 4^q labels."""
        for q in [2, 3, 4]:
            labels = generate_pauli_labels(q)
            assert len(labels) == 4**q

    @pytest.mark.parametrize("q", [2, 3])
    def test_coefficients_are_real(self, q: int):
        """Pauli coefficients should be real (for Hermitian Â)."""
        N = 2**q
        dx = 1.0 / N
        dt = 0.001
        nu = 1e-4
        x = np.linspace(0, 1, N, endpoint=False)
        u = np.sin(2 * np.pi * x)

        u_norm = u / np.linalg.norm(u)
        rhs = compute_rhs_shift(u, dx, nu)
        u_next = u + dt * rhs
        u_next_norm = u_next / np.linalg.norm(u_next)
        delta0 = (u_next_norm - u_norm) / dt

        labels, coeffs = solve_pauli_coefficients(u_norm, delta0, q)
        assert np.all(np.isreal(coeffs)), "Coefficients should be real"


# ---- Evolution operator tests ----


class TestEvolutionOperator:
    """Test the exact evolution e^{-iδτÂ}."""

    @pytest.mark.parametrize("q", [2, 3])
    def test_evolution_is_unitary(self, q: int):
        """e^{-iδτÂ} should be unitary."""
        N = 2**q
        dx = 1.0 / N
        dt = 0.001
        nu = 1e-4
        x = np.linspace(0, 1, N, endpoint=False)
        u = np.sin(2 * np.pi * x)

        H = build_evolution_hamiltonian(u, dx, dt, nu)
        U = exact_evolution_matrix(H, dt)

        np.testing.assert_allclose(
            U @ U.conj().T, np.eye(N), atol=1e-12,
            err_msg="Evolution matrix is not unitary",
        )

    @pytest.mark.parametrize("q", [2, 3])
    def test_exact_evolution_matches_euler(self, q: int):
        """e^{-iδτÂ}|u⟩ should approximate the Euler-updated state."""
        N = 2**q
        dx = 1.0 / N
        dt = 0.001
        nu = 1e-4
        x = np.linspace(0, 1, N, endpoint=False)
        u = np.sin(2 * np.pi * x)
        g = np.sin(2 * np.pi * x)

        u_norm = u / np.linalg.norm(u)

        # Quantum evolution
        u_evolved = quantum_evolution_step(u, dx, dt, nu, g, use_exact=True)

        # Classical Euler target
        rhs = compute_rhs_shift(u, dx, nu, g)
        u_next = u + dt * rhs
        u_next_norm = u_next / np.linalg.norm(u_next)

        # Should match to O(dt²)
        err = np.max(np.abs(u_evolved - u_next_norm))
        assert err < 1e-6, f"Evolution error: {err:.2e}"


# ---- Circuit tests ----


class TestEvolutionCircuit:
    """Test Trotterized evolution circuit."""

    @pytest.mark.parametrize("q", [2, 3])
    def test_circuit_matches_exact(self, q: int):
        """Trotterized circuit should approximate exact evolution."""
        N = 2**q
        dx = 1.0 / N
        dt = 0.001
        nu = 1e-4
        x = np.linspace(0, 1, N, endpoint=False)
        u = np.sin(2 * np.pi * x)

        u_norm = u / np.linalg.norm(u)
        H = build_evolution_hamiltonian(u, dx, dt, nu)

        # Exact evolution
        U_exact = exact_evolution_matrix(H, dt)
        u_exact = (U_exact @ u_norm).real

        # Circuit evolution (high reps for accuracy)
        qc = evolution_circuit(H, dt, trotter_order=2, trotter_reps=3)
        sv = Statevector(u_norm).evolve(qc)
        u_circuit = np.array(sv.data).real

        err = np.max(np.abs(u_circuit - u_exact))
        assert err < 1e-4, f"Circuit vs exact error: {err:.2e}"

    @pytest.mark.parametrize("q", [2, 3])
    def test_circuit_step_matches_shift_euler(self, q: int):
        """quantum_circuit_step should approximate shift_euler_step."""
        N = 2**q
        dx = 1.0 / N
        dt = 0.001
        nu = 1e-4
        x = np.linspace(0, 1, N, endpoint=False)
        u = np.sin(2 * np.pi * x)
        g = np.sin(2 * np.pi * x)

        u_shift = shift_euler_step(u, dx, dt, nu, g)
        u_qc = quantum_circuit_step(
            u, dx, dt, nu, g, trotter_order=2, trotter_reps=3,
        )

        err = compute_error(u_qc, u_shift)
        assert err < 1e-3, f"Circuit step error: {err:.2e}"


# ---- Time stepping tests ----


class TestTimeEvolution:
    """Test multi-step time evolution."""

    def test_shift_euler_matches_classical(self):
        """Shift-operator Euler should match classical Euler for periodic BC."""
        from burgers_classical import solve_burgers, source_term_sine

        q = 3
        N = 2**q
        x = np.linspace(0, 1, N, endpoint=False)
        dx = x[1] - x[0]
        nu = 1e-4
        dt = 0.01 * dx
        n_steps = 5
        u0 = np.sin(2 * np.pi * x)

        sols_shift = run_simulation(
            u0, x, nu, dt, n_steps,
            source_fn=source_term_sine, method="shift",
        )
        sols_classical = solve_burgers(
            u0, x, nu, dt, n_steps, source_fn=source_term_sine,
        )

        # At interior points, should be very close (boundary treatment differs)
        # For periodic BC vs one-sided, check overall shape matches
        for step in [1, n_steps]:
            err = compute_error(sols_shift[step], sols_classical[step])
            assert err < 0.1, (
                f"Step {step}: shift vs classical error {err:.2e}"
            )

    def test_quantum_exact_multistep(self):
        """Quantum exact evolution should track classical over several steps."""
        q = 3
        N = 2**q
        x = np.linspace(0, 1, N, endpoint=False)
        dx = x[1] - x[0]
        nu = 1e-4
        dt = 0.001
        n_steps = 5

        u0 = np.sin(2 * np.pi * x)

        sols_shift = run_simulation(
            u0, x, nu, dt, n_steps, method="shift",
        )
        sols_quantum = run_simulation(
            u0, x, nu, dt, n_steps, method="quantum_exact",
        )

        # Error should stay small over short simulation
        for step in range(1, n_steps + 1):
            err = compute_error(sols_quantum[step], sols_shift[step])
            assert err < 1e-3, (
                f"Step {step}: quantum vs shift error {err:.2e}"
            )

    def test_error_bounded_initially(self):
        """Error should be small initially, potentially growing (Fig. 11)."""
        q = 3
        N = 2**q
        x = np.linspace(0, 1, N, endpoint=False)
        dx = x[1] - x[0]
        nu = 1e-4
        dt = 0.001
        n_steps = 10

        u0 = np.sin(2 * np.pi * x)
        g_fn = lambda x, t: np.sin(2 * np.pi * x) * np.cos(2 * np.pi * t)

        sols_shift = run_simulation(
            u0, x, nu, dt, n_steps, source_fn=g_fn, method="shift",
        )
        sols_quantum = run_simulation(
            u0, x, nu, dt, n_steps, source_fn=g_fn, method="quantum_exact",
        )

        errors = [
            compute_error(sols_quantum[i], sols_shift[i])
            for i in range(1, n_steps + 1)
        ]

        # First step error should be very small
        assert errors[0] < 1e-4, f"First step error: {errors[0]:.2e}"

        # All errors should be bounded
        assert max(errors) < 0.01, f"Max error: {max(errors):.2e}"

    def test_energy_bounded(self):
        """Solution should not blow up."""
        q = 3
        N = 2**q
        x = np.linspace(0, 1, N, endpoint=False)
        nu = 1e-4
        dt = 0.001
        n_steps = 10

        u0 = np.sin(2 * np.pi * x)
        sols = run_simulation(
            u0, x, nu, dt, n_steps, method="quantum_exact",
        )

        max_u = max(np.max(np.abs(s)) for s in sols)
        assert max_u < 10.0, f"Solution blew up: max|u| = {max_u:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
