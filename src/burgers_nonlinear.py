"""Nonlinear evolution operator via Pauli decomposition.

Implements Appendix A.A from Gopalakrishnan Meena et al. AIAA-2026
(the "Proposed algorithms" appendix that the paper itself flags as
future work — not the §V.C MPS/MPO classical pipeline).  The full
time-step evolution operator Â = Σ c_i P̂_i is constructed by solving
the linear system that matches the quantum unitary evolution e^{-iδτÂ}
to the classical Euler update.

IMPORTANT: This Hamiltonian is *fit per-step* to reproduce the classical
forward-Euler result.  It is NOT an independent quantum PDE solver —
build_evolution_hamiltonian uses the classical RHS internally to define
the target state.  As a consequence, quantum_exact and shift_euler must
agree to within the lstsq residual of the Pauli coefficient solve (Eq. 16),
not to machine precision.  This is intentional: the appendix validates
the quantum pipeline (state-prep → Hamiltonian simulation → measurement)
by comparing against a known classical trajectory.

The approach:
1. Compute classical RHS: Δ₀ = -u·∇u + ν∇²u + g  (Euler step, §V.C Eq. 15)
2. Normalize: delta0 = (u_next_norm - u_norm) / δτ
3. Decompose into Pauli strings: solve (S^T + S)c = b  (Appendix A.A Eq. 16)
4. Build Hamiltonian: Â = Σ c_i P̂_i
5. Evolution circuit: e^{-iδτÂ}|u⟩  (Appendix A.A Eq. 17)
"""

import numpy as np
from itertools import product as cartesian_product

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter

from burgers_mpo import shift_matrix


# ---------------------------------------------------------------------------
# Classical RHS using shift operators (periodic BC)
# ---------------------------------------------------------------------------


def compute_rhs_shift(
    u: np.ndarray,
    dx: float,
    nu: float,
    g: np.ndarray | None = None,
    bc: str = "periodic",
) -> np.ndarray:
    """Classical Euler RHS using shift-operator FD.

    RHS = ν∇²u - u·∇u + g

    Consistent with the quantum circuit operators:
    - ∇u = -(S+ - S-)u / (2δx)   [central difference]
    - ∇²u = (S+ + S- - 2I)u / δx²

    bc='periodic': shift operators wrap mod N (default).
    bc='dirichlet': boundary wrapping zeroed (u=0 ghost nodes).
    """
    N = len(u)
    sp = shift_matrix(N, +1, bc=bc)
    sm = shift_matrix(N, -1, bc=bc)

    # Sign convention: the code's S+ maps |i⟩→|i+1⟩ column-wise, so
    # (S+ u)_j = u_{j-1}.  The standard central-difference gradient
    # (u_{j+1} - u_{j-1})/(2dx) is therefore -(S+ - S-)u/(2dx).
    # Paper Eq. 9 writes (S+ - S-)/(2dx) without the minus because it
    # uses the opposite S+ convention.  The leading minus here is correct.
    grad_u = -(sp - sm) @ u / (2 * dx)
    lap_u = (sp + sm - 2 * np.eye(N)) @ u / dx**2

    rhs = nu * lap_u - u * grad_u
    if g is not None:
        rhs += g

    # Dirichlet: u is prescribed at boundaries, so du/dt = 0 there.
    if bc == "dirichlet":
        rhs[0] = 0.0
        rhs[-1] = 0.0

    return rhs


def diffusion_rhs(
    u: np.ndarray,
    dx: float,
    nu: float,
    bc: str = "periodic",
) -> np.ndarray:
    """Pure Laplacian RHS: ν·∂²u/∂x² (no advection, no source).

    Dissipative sub-step generator for the operator-split TEBD circuit
    (F2 Phase B.1a). Uses the same second-order central-difference
    Laplacian as ``compute_rhs_shift`` so that for any (u, dx, nu, bc):

        compute_rhs_shift(u, dx, nu, g=None, bc=bc)
            == diffusion_rhs(u, dx, nu, bc) - u·grad_u

    bc='periodic': wrap-around stencil.
    bc='dirichlet': zero ghost nodes; rhs at boundary indices forced to 0.
    """
    N = len(u)
    sp = shift_matrix(N, +1, bc=bc)
    sm = shift_matrix(N, -1, bc=bc)

    lap_u = (sp + sm - 2 * np.eye(N)) @ u / dx**2
    rhs = nu * lap_u

    if bc == "dirichlet":
        rhs[0] = 0.0
        rhs[-1] = 0.0

    return rhs


# ---------------------------------------------------------------------------
# Pauli decomposition (Appendix A.A, Eq. 16)
# ---------------------------------------------------------------------------


def generate_pauli_labels(q: int) -> list[str]:
    """Generate all 4^q Pauli string labels for q qubits."""
    return ["".join(p) for p in cartesian_product("IXYZ", repeat=q)]


def _pauli_matrix(label: str) -> np.ndarray:
    """Get the dense matrix for a Pauli string label."""
    mat = SparsePauliOp(label).to_matrix()
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    return np.asarray(mat)


def solve_pauli_coefficients(
    u_norm: np.ndarray,
    delta0: np.ndarray,
    q: int,
) -> tuple[list[str], np.ndarray]:
    """Solve for Pauli coefficients matching e^{-iδτÂ} to classical RHS.

    From Appendix A.A, Eq. 16:
      b_i = i⟨u|P̂_i†|Δ₀⟩ + h.c. = -2·Im(⟨u|P̂_i|Δ₀⟩)
      S_ij = ⟨u|P̂_i†P̂_j|u⟩ = ⟨P̂_i u|P̂_j u⟩  (Paulis are Hermitian)
      (S^T + S)c = b  →  2·Re(S) c = b

    Parameters
    ----------
    u_norm : normalized state vector (length 2^q)
    delta0 : target change vector Δ₀ = (u_next_norm - u_norm) / δτ
    q : number of qubits

    Returns
    -------
    (labels, coefficients) where Â = Σ coefficients[i] * P̂_{labels[i]}
    """
    labels = generate_pauli_labels(q)
    n_paulis = len(labels)
    N = len(u_norm)

    # Precompute P_i|u⟩ for all Pauli strings
    Pu = np.zeros((n_paulis, N), dtype=complex)
    for i, label in enumerate(labels):
        P = _pauli_matrix(label)
        Pu[i] = P @ u_norm

    # b_i = -2·Im(⟨P_i u|Δ₀⟩)  [since P_i is Hermitian: ⟨u|P_i|Δ₀⟩ = (P_i u)†Δ₀]
    b = -2.0 * np.imag(Pu.conj() @ delta0)

    # S_ij = ⟨P_i u|P_j u⟩  →  (S^T + S) = 2·Re(S)  [S is Hermitian]
    S = Pu.conj() @ Pu.T
    A_sys = 2.0 * S.real

    # Solve via least squares (A_sys may be rank-deficient)
    coeffs, _, _, _ = np.linalg.lstsq(A_sys, b, rcond=None)

    return labels, coeffs.real


def build_evolution_hamiltonian(
    u: np.ndarray,
    dx: float,
    dt: float,
    nu: float,
    g: np.ndarray | None = None,
    bc: str = "periodic",
) -> SparsePauliOp:
    """Build Â = Σ c_i P̂_i for one evolution step (Appendix A.A).

    Steps:
    1. Compute classical RHS from physical (un-normalized) u
    2. Compute normalized next state
    3. Form Δ₀ = (u_next_norm - u_norm) / δτ
    4. Solve Pauli linear system for coefficients

    Parameters
    ----------
    u : physical (un-normalized) velocity field, length 2^q
    dx : grid spacing
    dt : time step
    nu : kinematic viscosity
    g : source term (same length as u), or None

    Returns
    -------
    SparsePauliOp representing the Hermitian operator Â
    """
    N = len(u)
    q = int(np.log2(N))

    u_norm = u / np.linalg.norm(u)

    # Classical Euler step (on un-normalized u)
    rhs = compute_rhs_shift(u, dx, nu, g, bc=bc)
    u_next = u + dt * rhs
    u_next_norm = u_next / np.linalg.norm(u_next)

    # Target change in normalized state
    delta0 = (u_next_norm - u_norm) / dt

    # Solve for Pauli coefficients
    labels, coeffs = solve_pauli_coefficients(u_norm, delta0, q)

    # Filter near-zero coefficients for circuit efficiency
    terms = [
        (label, float(coeff))
        for label, coeff in zip(labels, coeffs)
        if abs(coeff) > 1e-15
    ]

    if not terms:
        terms = [("I" * q, 0.0)]

    H = SparsePauliOp.from_list(terms)

    H_mat = H.to_matrix()
    assert np.allclose(H_mat, H_mat.conj().T, atol=1e-12), (
        "build_evolution_hamiltonian produced a non-Hermitian operator"
    )

    return H


# ---------------------------------------------------------------------------
# Evolution circuit (Eq. 17)
# ---------------------------------------------------------------------------


def evolution_circuit(
    hamiltonian: SparsePauliOp,
    dt: float,
    trotter_order: int = 1,
    trotter_reps: int = 1,
) -> QuantumCircuit:
    """Build quantum circuit for e^{-iδτÂ}|u⟩ (Eq. 17).

    Uses Suzuki-Trotter decomposition to implement the Hamiltonian
    simulation as a product of Pauli rotation gates.

    Parameters
    ----------
    hamiltonian : Â as SparsePauliOp
    dt : evolution time δτ
    trotter_order : order of Suzuki-Trotter decomposition (1 or 2)
    trotter_reps : number of Trotter repetitions

    Returns
    -------
    QuantumCircuit implementing e^{-iδτÂ}
    """
    q = hamiltonian.num_qubits
    synthesis = SuzukiTrotter(order=trotter_order, reps=trotter_reps)
    evo_gate = PauliEvolutionGate(hamiltonian, time=dt, synthesis=synthesis)

    qc = QuantumCircuit(q, name="evo_step")
    qc.append(evo_gate, range(q))

    return qc


def quantum_evolution_step(
    u: np.ndarray,
    dx: float,
    dt: float,
    nu: float,
    g: np.ndarray | None = None,
    use_exact: bool = True,
) -> np.ndarray:
    """One-step quantum evolution of velocity field u.

    Normalizes u, builds the Pauli Hamiltonian via build_evolution_hamiltonian,
    and applies e^{-iδτÂ} either exactly (matrix exponentiation) or via
    statevector simulation of the Trotter circuit.

    Returns the real part of the normalized evolved state (length N).
    """
    from qiskit.quantum_info import Statevector

    u_norm = u / np.linalg.norm(u)
    hamiltonian = build_evolution_hamiltonian(u, dx, dt, nu, g)

    if use_exact:
        U_mat = exact_evolution_matrix(hamiltonian, dt)
        return (U_mat @ u_norm).real
    else:
        qc_evo = evolution_circuit(hamiltonian, dt)
        sv = Statevector(u_norm).evolve(qc_evo)
        return sv.data.real


def exact_evolution_matrix(
    hamiltonian: SparsePauliOp,
    dt: float,
) -> np.ndarray:
    """Compute exact e^{-iδτÂ} matrix (for testing, not for circuits).

    Uses matrix exponentiation via eigendecomposition.
    """
    from scipy.linalg import expm

    H_mat = hamiltonian.to_matrix()
    if hasattr(H_mat, "toarray"):
        H_mat = H_mat.toarray()
    H_mat = np.asarray(H_mat)

    return expm(-1j * dt * H_mat)


if __name__ == "__main__":  # smoke test only — not the solver entry point
    q = 3
    N = 2**q
    dx = 1.0 / N
    dt = 0.001
    nu = 1e-4
    x = np.linspace(0, 1, N, endpoint=False)
    u = np.sin(2 * np.pi * x)
    g = np.sin(2 * np.pi * x)

    print(f"=== Nonlinear Evolution (q={q}, N={N}) ===")
    print(f"  dt={dt}, nu={nu}")

    # Build Hamiltonian
    hamiltonian = build_evolution_hamiltonian(u, dx, dt, nu, g)
    n_terms = len(hamiltonian)
    print(f"  Hamiltonian: {n_terms} non-zero Pauli terms (of {4**q})")

    # Check Hermiticity
    H_mat = hamiltonian.to_matrix()
    if hasattr(H_mat, "toarray"):
        H_mat = H_mat.toarray()
    H_mat = np.asarray(H_mat)
    herm_err = np.max(np.abs(H_mat - H_mat.conj().T))
    print(f"  Hermiticity error: {herm_err:.2e}")

    # Compare exact evolution to classical Euler
    u_norm = u / np.linalg.norm(u)
    U_mat = exact_evolution_matrix(hamiltonian, dt)
    u_evolved = (U_mat @ u_norm).real

    rhs = compute_rhs_shift(u, dx, nu, g)
    u_next = u + dt * rhs
    u_next_norm = u_next / np.linalg.norm(u_next)

    shape_err = np.max(np.abs(u_evolved - u_next_norm))
    print(f"  Evolution vs Euler (normalized): {shape_err:.2e}")
