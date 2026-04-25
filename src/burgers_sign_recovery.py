"""Sign recovery strategies for measurement-based quantum Burgers solver.

Implements the four options from F9 (IMPLEMENTATION-PLAN.md):

  Option 1 — none (as-is)
    sqrt(counts/total); all amplitudes non-negative.  Demonstrates the
    sign-loss problem.  Already implemented in quantum_circuit_step /
    mps_step; nothing extra needed.

  Option 2 — hadamard_test / interferometric approach
    One ancilla qubit.  Prepare reference |φ⟩ = u_norm (current step,
    signs known), apply controlled-e^{-iHdt}, measure.  Interference
    between |φ⟩ and |ψ⟩ = e^{-iHdt}|φ⟩ reveals sign(ψ_k).

  Option 3 — classical_oracle
    The classical RHS is already computed for norm prediction; extract
    signs from u + dt*RHS.  Zero extra quantum cost.

  Option 4 — dual_rail
    Decompose u = u⁺ - u⁻ (both non-negative).  Evolve each rail on a
    separate quantum circuit; subtract classically.  No sign-loss on
    either rail.  Costs 2× circuit executions per step.

Reference: F9, IMPLEMENTATION-PLAN.md.
"""

from __future__ import annotations

import numpy as np

from burgers_nonlinear import compute_rhs_shift


# ---------------------------------------------------------------------------
# Option 3 — Classical sign oracle
# ---------------------------------------------------------------------------

def classical_oracle_signs(
    u: np.ndarray,
    dx: float,
    dt: float,
    nu: float,
    g: np.ndarray | None = None,
    bc: str = "periodic",
) -> np.ndarray:
    """Predict signs of the evolved state from the classical Euler update.

    The update u + dt*RHS is already computed for norm prediction, so
    this function costs one extra compute_rhs_shift call (O(N) classical).

    Near zero crossings (|u_next_k| < eps * max|u_next|) the sign is
    unreliable under discretization noise, so we default to +1 there
    to avoid speckle in the recovered wavefunction.

    Returns
    -------
    signs : ndarray of +1 or -1, same shape as u.
    """
    u_next = u + dt * compute_rhs_shift(u, dx, nu, g, bc=bc)
    eps = 1e-10 * max(np.max(np.abs(u_next)), 1.0)
    s = np.where(np.abs(u_next) < eps, 1.0, np.sign(u_next))
    return s


# ---------------------------------------------------------------------------
# Option 2 — Hadamard test / interferometric sign recovery
# ---------------------------------------------------------------------------

def hadamard_test_sign_circuit(
    phi_norm: np.ndarray,
    hamiltonian,
    dt: float,
    trotter_order: int = 1,
    trotter_reps: int = 1,
):
    """Build the interferometric sign-recovery circuit (Option 2).

    Layout: qubits [0..q-1] = data, qubit q = ancilla.

    Circuit:
      anc |0⟩ ── H ── ctrl[e^{-iHdt}] ── H ── M_anc
      reg |0⟩ ── U_φ ─────────────────────── M_reg

    For real states |φ⟩ and |ψ⟩ = e^{-iHdt}|φ⟩:
      P(anc=0, reg=k) - P(anc=1, reg=k) = φ_k · ψ_k
      sign(ψ_k) = sign(interference_k) · sign(φ_k)

    NOTE: This recovers sign(Re ψ_k)·|ψ_k|, not Re(ψ_k).  For a
    single Trotter step at small dt the evolved state is approximately
    real and the two coincide, but the systematic error grows with dt
    and trotter_reps.  This is consistent with the rest of the code,
    which takes .real of the evolved statevector (e.g.
    _statevector_branch), so comparison targets also discard Im(ψ).

    The reference |φ⟩ = u_norm (current step, signs known from previous
    step's recovery or from the analytical initial condition at t=0).

    Parameters
    ----------
    phi_norm      : signed normalized reference state (length 2^q).
    hamiltonian   : SparsePauliOp Hamiltonian for this time step.
    dt            : time step.
    trotter_order : Suzuki-Trotter order.
    trotter_reps  : Trotter repetitions.

    Returns
    -------
    QuantumCircuit ready to transpile and execute (q+1 qubits, measured).
    """
    from qiskit.circuit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.synthesis import SuzukiTrotter

    N = len(phi_norm)
    q = int(np.log2(N))
    anc = q

    qc = QuantumCircuit(q + 1, name="hadamard_sign_test")

    qc.initialize(phi_norm.tolist(), range(q))
    qc.h(anc)

    synthesis = SuzukiTrotter(order=trotter_order, reps=trotter_reps)
    evo_gate = PauliEvolutionGate(hamiltonian, time=dt, synthesis=synthesis)
    qc.append(evo_gate.control(1), [anc] + list(range(q)))

    qc.h(anc)
    qc.measure_all()
    return qc


def extract_signs_from_hadamard_counts(
    counts: dict,
    phi_norm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract sign vector and magnitude estimates from Hadamard test counts.

    Bitstring convention (Qiskit little-endian):
      qubit 0 = rightmost character
      qubit q (ancilla) = leftmost character

    Parameters
    ----------
    counts   : dict from Qiskit get_counts().
    phi_norm : signed normalized reference used to build the circuit.

    Returns
    -------
    signs      : ndarray of +1/-1, shape (N,).
    magnitudes : ndarray of |ψ_k| estimates, shape (N,).
    """
    N = len(phi_norm)
    q = int(np.log2(N))
    total = max(sum(counts.values()), 1)

    p0 = np.zeros(N)
    p1 = np.zeros(N)

    for bitstr, cnt in counts.items():
        bits = bitstr.replace(" ", "")
        if len(bits) < q + 1:
            continue
        anc_bit = int(bits[0])
        data_idx = int(bits[1:], 2)
        if data_idx < N:
            if anc_bit == 0:
                p0[data_idx] += cnt / total
            else:
                p1[data_idx] += cnt / total

    interference = p0 - p1  # = φ_k · ψ_k for real states

    signs = np.ones(N)
    for k in range(N):
        if abs(phi_norm[k]) > 1e-10 and abs(interference[k]) > 1e-8:
            signs[k] = float(np.sign(interference[k]) * np.sign(phi_norm[k]))

    # |ψ_k|² = 2*(p0[k]+p1[k]) - φ_k²  (from Hadamard test algebra)
    mag_sq = np.maximum(2.0 * (p0 + p1) - phi_norm ** 2, 0.0)
    magnitudes = np.sqrt(mag_sq)

    return signs, magnitudes


# ---------------------------------------------------------------------------
# Option 4 — Dual-rail encoding
# ---------------------------------------------------------------------------

def dual_rail_split(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose u into non-negative positive and negative rails.

    u = u_pos - u_neg  where u_pos, u_neg >= 0 elementwise.
    """
    return np.maximum(u, 0.0), np.maximum(-u, 0.0)


def dual_rail_recombine(
    u_pos_evolved: np.ndarray,
    u_neg_evolved: np.ndarray,
    norm_target: float,
) -> np.ndarray:
    """Subtract rails and rescale to the classical norm prediction.

    Parameters
    ----------
    u_pos_evolved : evolved positive rail (non-negative amplitudes).
    u_neg_evolved : evolved negative rail (non-negative amplitudes).
    norm_target   : ||u + dt*RHS|| from classical norm prediction.

    Returns
    -------
    Signed, rescaled solution vector.
    """
    u_combined = u_pos_evolved - u_neg_evolved
    norm_combined = np.linalg.norm(u_combined)
    if norm_combined > 1e-12:
        return u_combined * (norm_target / norm_combined)
    return u_combined
