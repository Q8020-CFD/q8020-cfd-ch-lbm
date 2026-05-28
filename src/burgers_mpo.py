"""MPO-derived quantum circuits for spatial differential operators.

Implements the ladder operator approach from Gopalakrishnan Meena et al.
AIAA-2026 §III.B.2 (Eqs. 9-12) as quantum circuits:

- S+ (increment): |i⟩ → |i+1 mod N⟩  (Eq. 11)
- S- (decrement): |i⟩ → |i-1 mod N⟩  (Eq. 12, S- = S+†)

And the LCU (Linear Combination of Unitaries) circuits for:
- Gradient:   ∂u/∂x  ≃ (S+ - S-) / (2δx)       (Eq. 9)
- Laplacian:  ∂²u/∂x² ≃ (S+ + S- - 2I) / δx²   (Eq. 10)
"""

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector


# ---------------------------------------------------------------------------
# S+ and S- ladder operator circuits
# ---------------------------------------------------------------------------


def increment_circuit(q: int) -> QuantumCircuit:
    """Quantum circuit for S+ (binary increment mod 2^q).

    Maps |i⟩ → |i+1 mod 2^q⟩ on q qubits.

    Uses the standard ripple-increment pattern:
    - X on qubit 0 (always flip LSB)
    - CNOT on qubit 1 controlled by qubit 0
    - Toffoli on qubit 2 controlled by qubits 0,1
    - ... etc up to qubit q-1

    Qiskit convention: qubit 0 = LSB.
    """
    qc = QuantumCircuit(q, name="S+")

    # Work from MSB down to LSB.
    # Qubit k flips when all qubits 0..k-1 are 1 (carry propagation).
    for k in range(q - 1, 0, -1):
        controls = list(range(k))
        qc.mcx(controls, k)

    # LSB always flips
    qc.x(0)

    return qc


def decrement_circuit(q: int) -> QuantumCircuit:
    """Quantum circuit for S- (binary decrement mod 2^q).

    S- = (S+)†, so we just invert the increment circuit.
    """
    qc = increment_circuit(q).inverse()
    qc.name = "S-"
    return qc


def shift_matrix(N: int, direction: int = 1, bc: str = "periodic") -> np.ndarray:
    """Classical NxN shift (permutation) matrix.

    direction=+1: S+ matrix, maps |i⟩ → |i+1 mod N⟩
    direction=-1: S- matrix, maps |i⟩ → |i-1 mod N⟩

    bc='periodic': wraps around (S[0, N-1] = 1 for S+, etc.)
    bc='dirichlet': no wrapping; boundary rows that would wrap are zeroed,
        equivalent to u=0 ghost nodes outside the domain.
    """
    S = np.zeros((N, N), dtype=float)
    for i in range(N):
        j = (i + direction) % N
        S[j, i] = 1.0
    if bc == "dirichlet":
        # Zero the wrapping entries
        if direction == 1:
            S[0, N - 1] = 0.0   # S+ would wrap |N-1⟩ → |0⟩
        else:
            S[N - 1, 0] = 0.0   # S- would wrap |0⟩ → |N-1⟩
    return S


# ---------------------------------------------------------------------------
# LCU circuits for gradient and Laplacian
# ---------------------------------------------------------------------------


def gradient_lcu_circuit(q: int) -> QuantumCircuit:
    """LCU block-encoding of (S+ - S-)/2 on q system qubits.

    Uses 1 ancilla qubit. The block encoding satisfies:
      ⟨1|_anc U |0⟩_anc = (S+ - S-) / 2

    To get the physical gradient, divide by δx:
      ∂u/∂x ≃ (S+ - S-) / (2δx)

    Circuit layout: qubits 0..q-1 = system, qubit q = ancilla.
    Post-select ancilla = |1⟩ for the gradient operator.
    (Post-select ancilla = |0⟩ gives (S+ + S-)/2 as a bonus.)
    """
    n_total = q + 1
    anc = q

    qc = QuantumCircuit(n_total, name="grad_LCU")

    # Prepare: ancilla in |+⟩
    qc.h(anc)

    # Select: controlled-S+ (anc=|0⟩), controlled-S- (anc=|1⟩)
    # Qiskit .control() prepends the control qubit, so gate qubits = [ctrl, targets...]
    sp = increment_circuit(q)
    qc.append(sp.control(1, ctrl_state=0), [anc] + list(range(q)))

    sm = decrement_circuit(q)
    qc.append(sm.control(1, ctrl_state=1), [anc] + list(range(q)))

    # Unprepare: H on ancilla
    qc.h(anc)

    return qc


def laplacian_lcu_circuit(q: int) -> QuantumCircuit:
    """LCU block-encoding of (S+ + S- - 2I)/4 on q system qubits.

    The Laplacian stencil is (S+ + S- - 2I)/δx². We block-encode the
    numerator (S+ + S- - 2I) using a 3-term LCU.

    Standard LCU for A = Σ c_j U_j:
      Prepare: |0⟩ → |α⟩ = (1/√λ) Σ √|c_j| |j⟩,  λ = Σ|c_j|
      Select:  |j⟩|ψ⟩ → |j⟩ sign(c_j) U_j |ψ⟩
      Then: ⟨α| Select |α⟩ = A / λ

    For [c_0, c_1, c_2] = [1, 1, -2] with [U_0, U_1, U_2] = [S+, S-, I]:
      λ = |1| + |1| + |-2| = 4
      |α⟩ = (1/2)|0⟩ + (1/2)|1⟩ + (√2/2)|2⟩   (since √|c_j|/√λ)

    Wait — the standard formula is: |α⟩ = (1/√λ) Σ √|c_j| |j⟩
      √|c_0|/√4 = 1/2,  √|c_1|/√4 = 1/2,  √|c_2|/√4 = √2/2

    Select absorbs signs: U_0→S+, U_1→S-, U_2→(-I).

    Block encoding gives ⟨α| Select |α⟩ = (S+ + S- - 2I) / 4.

    Uses 2 ancilla qubits to address 3 unitaries (4 states, |11⟩ unused).

    Circuit layout: qubits 0..q-1 = system, qubits q,q+1 = ancillas.
    Post-select ancillas = |00⟩.
    """
    n_total = q + 2
    anc0 = q      # low bit of ancilla register
    anc1 = q + 1  # high bit of ancilla register

    qc = QuantumCircuit(n_total, name="lap_LCU")

    # --- Prepare oracle ---
    # Target state on [anc0, anc1]:
    #   |α⟩ = (1/2)|00⟩ + (1/2)|01⟩ + (√2/2)|10⟩
    #
    # Qiskit little-endian: |anc1, anc0⟩, so:
    #   |00⟩: anc0=0, anc1=0  → amplitude 1/2
    #   |01⟩: anc0=1, anc1=0  → amplitude 1/2
    #   |10⟩: anc0=0, anc1=1  → amplitude √2/2
    #
    # Step 1: Ry on anc1 to set prob of anc1=0 vs anc1=1
    #   P(anc1=0) = (1/2)² + (1/2)² = 1/2
    #   P(anc1=1) = (√2/2)² = 1/2
    #   So anc1 goes to |+⟩: Ry(π/2) or just H
    qc.h(anc1)

    # Step 2: When anc1=0, split anc0 equally → H on anc0
    #   When anc1=1, anc0 stays |0⟩ (amplitude √2/2 all on |10⟩)
    qc.ch(anc1, anc0, ctrl_state=0)

    # Verify: starting from |00⟩:
    #   After H on anc1: (1/√2)|00⟩ + (1/√2)|10⟩  (anc1 superposition)
    #   After CH(anc1=0 → H on anc0):
    #     anc1=0 branch: (1/√2) * (1/√2)(|00⟩+|01⟩) = (1/2)|00⟩ + (1/2)|01⟩
    #     anc1=1 branch: (1/√2)|10⟩
    #   Total: (1/2)|00⟩ + (1/2)|01⟩ + (1/√2)|10⟩  ✓

    # --- Select oracle ---
    # |00⟩ → S+ (positive sign)
    sp = increment_circuit(q)
    qc.append(sp.control(2, ctrl_state=0), [anc0, anc1] + list(range(q)))

    # |01⟩ → S- (positive sign)
    sm = decrement_circuit(q)
    qc.append(sm.control(2, ctrl_state=1), [anc0, anc1] + list(range(q)))

    # |10⟩ → -I (negative sign, identity on system)
    # Apply phase -1 to the |10⟩ component of the ancilla register.
    # |10⟩ means anc0=0, anc1=1. Phase-flip this state:
    #   X(anc0), CZ(anc0, anc1), X(anc0)
    qc.x(anc0)
    qc.cz(anc0, anc1)
    qc.x(anc0)

    # --- Unprepare oracle (inverse of prepare) ---
    qc.ch(anc1, anc0, ctrl_state=0)
    qc.h(anc1)

    # Post-select ancillas = |00⟩ → effective operator = (S+ + S- - 2I) / 4

    return qc


# ---------------------------------------------------------------------------
# Block-encoding extraction utilities
# ---------------------------------------------------------------------------


def extract_block_encoded_operator(
    U_full: np.ndarray,
    n_system: int,
    n_ancilla: int,
    anc_in: int = 0,
    anc_out: int | None = None,
) -> np.ndarray:
    """Extract the block-encoded operator from a full unitary.

    Given U on (n_system + n_ancilla) qubits, extracts:
      O = ⟨anc_out|_anc U |anc_in⟩_anc

    where anc_in/anc_out are computational basis indices for the ancilla.

    Qiskit convention: system qubits are low bits, ancilla are high bits.
    So full index = system_idx + ancilla_idx * 2^n_system.
    """
    N_sys = 2**n_system

    if anc_out is None:
        anc_out = anc_in

    O = np.zeros((N_sys, N_sys), dtype=complex)
    for i in range(N_sys):
        for j in range(N_sys):
            O[i, j] = U_full[i + anc_out * N_sys, j + anc_in * N_sys]

    return O


if __name__ == "__main__":  # smoke test only — not the solver entry point
    q = 3
    N = 2**q
    sp_ref = shift_matrix(N, +1)
    sm_ref = shift_matrix(N, -1)

    print(f"=== S+/S- on {q} qubits (N={N}) ===")
    sp_mat = Operator(increment_circuit(q)).data
    sm_mat = Operator(decrement_circuit(q)).data
    print(f"  S+ error: {np.max(np.abs(sp_mat - sp_ref)):.2e}")
    print(f"  S- error: {np.max(np.abs(sm_mat - sm_ref)):.2e}")
    print(f"  S+S-=I error: {np.max(np.abs(sp_mat @ sm_mat - np.eye(N))):.2e}")

    print(f"\n=== Gradient LCU ===")
    grad_full = Operator(gradient_lcu_circuit(q)).data

    # ⟨1|U|0⟩ should be (S+ - S-)/2
    grad_op = extract_block_encoded_operator(
        grad_full, n_system=q, n_ancilla=1, anc_in=0, anc_out=1
    )
    grad_ref = (sp_ref - sm_ref) / 2
    print(f"  ⟨1|U|0⟩ vs (S+-S-)/2: {np.max(np.abs(grad_op - grad_ref)):.2e}")

    # ⟨0|U|0⟩ should be (S+ + S-)/2
    sym_op = extract_block_encoded_operator(
        grad_full, n_system=q, n_ancilla=1, anc_in=0, anc_out=0
    )
    sym_ref = (sp_ref + sm_ref) / 2
    print(f"  ⟨0|U|0⟩ vs (S++S-)/2: {np.max(np.abs(sym_op - sym_ref)):.2e}")

    print(f"\n=== Laplacian LCU ===")
    lap_full = Operator(laplacian_lcu_circuit(q)).data

    # ⟨00|U|00⟩ should be (S+ + S- - 2I)/4
    lap_op = extract_block_encoded_operator(
        lap_full, n_system=q, n_ancilla=2, anc_in=0, anc_out=0
    )
    lap_ref = (sp_ref + sm_ref - 2 * np.eye(N)) / 4
    lap_err = np.max(np.abs(lap_op - lap_ref))
    print(f"  ⟨00|U|00⟩ vs (S++S--2I)/4: {lap_err:.2e}")

    # Verify against classical Laplacian matrix
    dx = 1.0 / N
    from burgers_classical import build_laplacian_matrix
    lap_classical = build_laplacian_matrix(N, dx)
    # (S+ + S- - 2I)/dx^2 should match the classical Laplacian
    lap_from_shift = (sp_ref + sm_ref - 2 * np.eye(N)) / dx**2
    lap_vs_classical = np.max(np.abs(lap_from_shift - lap_classical))
    print(f"  Shift Laplacian vs classical: {lap_vs_classical:.2e}")
