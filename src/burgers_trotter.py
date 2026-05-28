"""Hybrid classical-quantum time evolution for Burgers equation.

The `quantum_circuit_step` here is the per-step driver for Gopalakrishnan
Meena et al. AIAA-2026 Appendix A.A (Eqs. 16-17): fit a Hermitian Pauli
operator Â per timestep so that e^{-iÂδτ}|u⟩ reproduces the classical
Euler update, then evolve via a Suzuki-Trotter circuit.  This is a
HYBRID pathway, not pure quantum: the Hamiltonian Â is fit per step to
a classical Euler trajectory, and classical Euler RHS, lstsq Pauli-coef
solve, norm tracking, and (for `bc='dirichlet'`) boundary projection
all run alongside the circuit each step.

NOTE: the paper's §V.C (classical Euler with `quimb` MPS/MPO spatial
operators) is NOT implemented in this module.  The classical reference
used here (`compute_rhs_shift` in burgers_nonlinear) matches the FORM of
§V.C Eq. 15 but uses plain shift-matrix FD, not MPS/MPO via quimb.

At each time step the underlying classical update is (§V.C Eq. 15):
  u(t+δτ) = u(t) + δτ [ν∇²u - u·∇u + g]

Four methods are provided:
1. shift_euler: classical Euler with shift-operator FD (periodic BC)
2. quantum_exact: Pauli decomposition + exact matrix exponential
3. quantum_circuit: Pauli decomposition + Trotterized circuit (Meena
   Appendix A.A)
4. mps: MPS state-prep circuit + exact Hamiltonian evolution

All methods operate on the physical (un-normalized) velocity field and
track norm changes classically.
"""

import time
from typing import Any

import numpy as np

from burgers_cole_hopf import run_cole_hopf_simulation
from burgers_cole_hopf_circuit import run_cole_hopf_circuit_simulation
from burgers_mps import (
    classical_to_mps,
    extract_physical_state,
    mps_to_circuit,
    normalize_state,
)
from burgers_nonlinear import (
    build_evolution_hamiltonian,
    compute_rhs_shift,
    diffusion_rhs,
    evolution_circuit,
    exact_evolution_matrix,
)
from burgers_sign_recovery import (
    classical_oracle_signs,
    dual_rail_recombine,
    dual_rail_split,
    extract_signs_from_hadamard_counts,
    hadamard_test_sign_circuit,
)
from burgers_tebd import (
    build_hamiltonian_dense,
    build_hamiltonian_mpo_ladder,
    build_wii_layer,
    build_wii_layer_ladder,
    run_tebd_simulation,
)


# ---------------------------------------------------------------------------
# Single-step methods
# ---------------------------------------------------------------------------


def _project_dirichlet(u_norm: np.ndarray) -> np.ndarray:
    """Zero boundary amplitudes and renormalize (Dirichlet BC enforcement).

    The Pauli lstsq fit targets delta0[0]=delta0[-1]=0, but the unitary
    e^{-iHdt} mixes all amplitudes, causing O(dt) boundary leakage each step.
    Project back onto the Dirichlet subspace to prevent accumulation.
    """
    v = u_norm.copy()
    v[0] = 0.0
    v[-1] = 0.0
    mag = np.linalg.norm(v)
    return v / mag if mag > 1e-15 else v


def shift_euler_step(
    u: np.ndarray,
    dx: float,
    dt: float,
    nu: float,
    g: np.ndarray | None = None,
    bc: str = "periodic",
) -> tuple[np.ndarray, dict]:
    """One explicit Euler step using shift-operator FD.

    Baseline for comparison with quantum methods.
    """
    rhs = compute_rhs_shift(u, dx, nu, g, bc=bc)
    return u + dt * rhs, {}


def _build_step_context(
    u: np.ndarray,
    dx: float,
    dt: float,
    nu: float,
    g: np.ndarray | None,
    bc: str,
) -> tuple:
    """Shared setup for quantum evolution steps.

    Returns (u_norm, hamiltonian, norm_next) where norm_next is the
    classical norm prediction used to recover physical velocity.
    """
    u_norm = u / np.linalg.norm(u)
    hamiltonian = build_evolution_hamiltonian(u, dx, dt, nu, g, bc=bc)
    with np.errstate(over="ignore", invalid="ignore"):
        rhs = compute_rhs_shift(u, dx, nu, g, bc=bc)
        norm_next = np.linalg.norm(u + dt * rhs)
    return u_norm, hamiltonian, norm_next


def quantum_exact_step(
    u: np.ndarray,
    dx: float,
    dt: float,
    nu: float,
    g: np.ndarray | None = None,
    bc: str = "periodic",
) -> tuple[np.ndarray, dict]:
    """One evolution step via Pauli decomposition + exact expm.

    Uses exact matrix exponential e^{-iδτÂ} (not Trotterized).
    Returns (physical velocity, metrics dict).
    """
    u_norm, hamiltonian, norm_next = _build_step_context(
        u, dx, dt, nu, g, bc,
    )
    U = exact_evolution_matrix(hamiltonian, dt)
    u_evolved_norm = (U @ u_norm).real
    if bc == "dirichlet":
        u_evolved_norm = _project_dirichlet(u_evolved_norm)
    return u_evolved_norm * norm_next, {"n_pauli_terms": len(hamiltonian)}


def _statevector_branch(
    u_norm: np.ndarray,
    qc: Any,
    backend: Any,
) -> tuple[np.ndarray, Any, float]:
    """Statevector mode: exact amplitudes, no measurement.

    Uses the supplied backend (which may include a noise model from
    T1/T2 configuration).  Falls back to a plain statevector sim
    only if no backend is provided.
    """
    from qiskit import QuantumCircuit as QC
    from qiskit.compiler import transpile
    from qiskit_aer import AerSimulator

    qc_full = QC(qc.num_qubits)
    qc_full.initialize(u_norm)
    qc_full.compose(qc, inplace=True)
    qc_full.save_statevector()  # pylint: disable=no-member
    sim = backend if backend is not None else AerSimulator(method="statevector")

    t0 = time.perf_counter()
    qc_t = transpile(qc_full, sim, optimization_level=1)
    transpile_time = time.perf_counter() - t0
    result = sim.run(qc_t, shots=1).result()
    u_evolved_norm = np.array(result.get_statevector().data).real
    return u_evolved_norm, qc_t, transpile_time


def _sampling_hadamard_branch(
    u_norm: np.ndarray,
    hamiltonian: Any,
    dt: float,
    trotter_order: int,
    trotter_reps: int,
    shots: int,
    backend: Any,
) -> tuple[np.ndarray, Any, float]:
    """Hadamard-test sign recovery: interferometric circuit."""
    from qiskit.compiler import transpile

    qc_ht = hadamard_test_sign_circuit(
        u_norm, hamiltonian, dt, trotter_order, trotter_reps,
    )
    t0 = time.perf_counter()
    qc_t = transpile(qc_ht, backend, optimization_level=1)
    transpile_time = time.perf_counter() - t0
    result = backend.run(qc_t, shots=shots).result()
    signs, magnitudes = extract_signs_from_hadamard_counts(
        result.get_counts(), u_norm,
    )
    return signs * magnitudes, qc_t, transpile_time


def _sampling_basic_branch(
    u: np.ndarray,
    u_norm: np.ndarray,
    qc: Any,
    dx: float,
    dt: float,
    nu: float,
    g: np.ndarray | None,
    shots: int,
    backend: Any,
    bc: str,
    sign_recovery: str,
) -> tuple[np.ndarray, Any, float]:
    """Standard sampling: reconstruct amplitudes from counts."""
    from qiskit import QuantumCircuit as QC
    from qiskit.compiler import transpile

    qc_full = QC(qc.num_qubits)
    qc_full.initialize(u_norm)
    qc_full.compose(qc, inplace=True)
    qc_full.measure_all()

    t0 = time.perf_counter()
    qc_t = transpile(qc_full, backend, optimization_level=1)
    transpile_time = time.perf_counter() - t0
    result = backend.run(qc_t, shots=shots).result()
    counts = result.get_counts()
    n_qubits = qc.num_qubits
    total = sum(counts.values())
    amps = np.zeros(2**n_qubits)
    for bitstr, cnt in counts.items():
        idx = int(bitstr.replace(" ", ""), 2)
        amps[idx] = np.sqrt(cnt / total)
    u_evolved_norm = amps
    if sign_recovery == "classical_oracle":
        signs = classical_oracle_signs(u, dx, dt, nu, g, bc)
        u_evolved_norm = signs * np.abs(u_evolved_norm)
    return u_evolved_norm, qc_t, transpile_time


def quantum_circuit_step(
    u: np.ndarray,
    dx: float,
    dt: float,
    nu: float,
    g: np.ndarray | None = None,
    trotter_order: int = 1,
    trotter_reps: int = 1,
    shots: int = 0,
    backend: Any = None,
    t1: float | None = None,
    t2: float | None = None,
    backend_name: str | None = None,
    bc: str = "periodic",
    sign_recovery: str = "none",
) -> tuple[np.ndarray, dict]:
    """One evolution step via Pauli decomposition + Trotterized circuit.

    Uses Suzuki-Trotter decomposition for Hamiltonian simulation.
    Returns (physical velocity, per-step metrics dict).

    shots=0 (default): statevector mode — exact amplitudes, no measurement.
    shots>0: sampling mode — measurements added, state reconstructed from
      counts as amp[i] = sqrt(counts[i] / total_shots).  Sign recovery is
      controlled by the sign_recovery parameter:
        "none"             — abs(amplitude) only; sign is lost.
        "classical_oracle" — signs copied from classical reference.
        "hadamard_test"    — ancilla-based phase estimation per amplitude.
        "dual_rail"        — dispatched to dual_rail_quantum_step instead.

    Metrics captured:
      pauli_decomp_time_s  — time in build_evolution_hamiltonian()
      circuit_construction_time_s — time in evolution_circuit()
      transpilation_time_s  — time in transpile()
      execution_time_s     — time in backend.run()
      circuit_depth        — qc_t.depth() after transpilation
      gate_counts          — qc_t.count_ops() after transpilation
      n_pauli_terms        — len(hamiltonian)
    """
    t0 = time.perf_counter()
    u_norm, hamiltonian, norm_next = _build_step_context(
        u, dx, dt, nu, g, bc,
    )
    pauli_decomp_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    qc = evolution_circuit(hamiltonian, dt, trotter_order, trotter_reps)
    circuit_build_time = time.perf_counter() - t0

    # Ensure backend exists (covers direct callers outside run_simulation)
    if backend is None:
        from q8020_cfd_qutil.backend import get_backend
        backend = get_backend(
            name=backend_name, backend_type="sim", t1=t1, t2=t2,
        )

    t0 = time.perf_counter()
    if not shots:
        u_evolved_norm, qc_t, transpile_time = _statevector_branch(
            u_norm, qc, backend,
        )
    elif sign_recovery == "hadamard_test":
        u_evolved_norm, qc_t, transpile_time = _sampling_hadamard_branch(
            u_norm, hamiltonian, dt, trotter_order, trotter_reps,
            shots, backend,
        )
    else:
        u_evolved_norm, qc_t, transpile_time = _sampling_basic_branch(
            u, u_norm, qc, dx, dt, nu, g, shots, backend, bc,
            sign_recovery,
        )
    execute_time = time.perf_counter() - t0 - transpile_time

    if bc == "dirichlet":
        u_evolved_norm = _project_dirichlet(u_evolved_norm)

    metrics = {
        "pauli_decomp_time_s": pauli_decomp_time,
        "circuit_construction_time_s": circuit_build_time,
        "transpilation_time_s": transpile_time,
        "execution_time_s": execute_time,
        "circuit_depth": qc_t.depth(),
        "gate_counts": dict(qc_t.count_ops()),
        "n_qubits": qc_t.num_qubits,
        "n_pauli_terms": len(hamiltonian),
        "sign_recovery": sign_recovery,
    }

    return u_evolved_norm * norm_next, metrics


def mps_step(
    u: np.ndarray,
    dx: float,
    dt: float,
    nu: float,
    g: np.ndarray | None = None,
    bond_dim: int | None = None,
    threshold: float = 0.0,
    shots: int = 0,
    bc: str = "periodic",
    sign_recovery: str = "none",
    backend: Any = None,
) -> tuple[np.ndarray, dict]:
    """One evolution step: MPS state-prep circuit + exact Hamiltonian evolution.

    Encodes the current field into an MPS circuit (with optional bond-dim
    truncation), simulates it to obtain the (approximate) quantum state, then
    applies the exact matrix exponential for time evolution.

    shots=0 (default): statevector mode — exact amplitudes.
    shots>0: measurement mode — measure all qubits, post-select on bond
        qubits being |0>, reconstruct amplitudes from counts.  Sign
        information is lost (same limitation as quantum_circuit_step).

        Post-selection normalizes by the post-selected count (not total
        shots), giving a clean state in the physical subspace as if the
        MPS truncation were exact.  Bond leakage (fraction of shots where
        bond qubits != |0>) is reported in the metrics dict as
        ``bond_leakage`` so callers can assess truncation quality without
        it polluting the evolved wavefunction.

    Returns (physical velocity, metrics dict).
    """
    from qiskit.quantum_info import Statevector

    u_norm, norm_u = normalize_state(u)
    bond_leakage: float | None = None

    tensors = classical_to_mps(
        u_norm, bond_dim=bond_dim, threshold=threshold, canonical="right"
    )
    qc = mps_to_circuit(tensors)
    q = len(tensors)
    n_bond = qc.num_qubits - q

    if not shots:
        sv = Statevector.from_instruction(qc)
        u_prepared = extract_physical_state(sv, q, n_bond).real
    else:
        from qiskit.compiler import transpile

        if backend is None:
            from qiskit_aer import AerSimulator
            backend = AerSimulator()

        qc_m = qc.copy()
        qc_m.measure_all()
        qc_t = transpile(qc_m, backend, optimization_level=1)
        result = backend.run(qc_t, shots=shots).result()
        counts = result.get_counts()

        # Post-select: keep only bitstrings where bond qubits are all 0.
        # Qiskit little-endian: physical qubits are low bits (0..q-1),
        # bond qubits are high bits (q..q+n_bond-1).
        N = 2**q
        amps = np.zeros(N)
        total = 0
        for bitstr, cnt in counts.items():
            idx = int(bitstr.replace(" ", ""), 2)
            if idx < N:  # bond qubits all zero
                amps[idx] = cnt
                total += cnt
        bond_leakage = 1.0 - total / shots if shots else 0.0
        if total > 0:
            amps = np.sqrt(amps / total)
        u_prepared = amps

    # Apply sign recovery to u_prepared before matrix evolution (shots path).
    _MPS_SIGN_OPTIONS = ("none", "classical_oracle")
    if sign_recovery not in _MPS_SIGN_OPTIONS:
        raise ValueError(
            f"mps_step does not support sign_recovery={sign_recovery!r}; "
            f"valid options: {_MPS_SIGN_OPTIONS}"
        )
    if shots and sign_recovery == "classical_oracle":
        signs = classical_oracle_signs(u, dx, dt, nu, g, bc)
        u_prepared = signs * np.abs(u_prepared)

    hamiltonian = build_evolution_hamiltonian(u, dx, dt, nu, g, bc=bc)
    U = exact_evolution_matrix(hamiltonian, dt)
    u_evolved_norm = (U @ u_prepared).real
    if bc == "dirichlet":
        u_evolved_norm = _project_dirichlet(u_evolved_norm)

    rhs = compute_rhs_shift(u, dx, nu, g, bc=bc)
    norm_next = np.linalg.norm(u + dt * rhs)

    return u_evolved_norm * norm_next, {
        "n_pauli_terms": len(hamiltonian),
        "bond_dim": bond_dim,
        "n_bond_qubits": n_bond,
        "sign_recovery": sign_recovery,
        "bond_leakage": bond_leakage,
    }


# ---------------------------------------------------------------------------
# Option 4 — Dual-rail quantum step
# ---------------------------------------------------------------------------


def dual_rail_quantum_step(
    u: np.ndarray,
    dx: float,
    dt: float,
    nu: float,
    g: np.ndarray | None = None,
    trotter_order: int = 1,
    trotter_reps: int = 1,
    shots: int = 0,
    backend: Any = None,
    t1: float | None = None,
    t2: float | None = None,
    backend_name: str | None = None,
    bc: str = "periodic",
) -> tuple[np.ndarray, dict]:
    """Option 4: decompose u = u⁺ - u⁻, evolve each rail, recombine.

    Both u⁺ and u⁻ are non-negative, so the shots path has no sign-loss
    on either rail.  Costs 2× circuit executions per step.

    The Hamiltonian for each rail is built from that rail's own state
    (freeze-coefficient approximation).  The final result is rescaled
    to the classical norm prediction from the full u.
    """
    u_pos, u_neg = dual_rail_split(u)

    rhs_full = compute_rhs_shift(u, dx, nu, g, bc=bc)
    norm_target = float(np.linalg.norm(u + dt * rhs_full))

    def _evolve_rail(u_rail: np.ndarray) -> np.ndarray:
        if np.linalg.norm(u_rail) < 1e-12:
            return np.zeros_like(u_rail)
        evolved, _ = quantum_circuit_step(
            u_rail, dx, dt, nu, g, trotter_order, trotter_reps,
            shots=shots, backend=backend, t1=t1, t2=t2,
            backend_name=backend_name, bc=bc, sign_recovery="none",
        )
        return evolved

    u_pos_evolved = _evolve_rail(u_pos)
    u_neg_evolved = _evolve_rail(u_neg)

    u_combined = dual_rail_recombine(u_pos_evolved, u_neg_evolved, norm_target)

    metrics = {"sign_recovery": "dual_rail"}
    return u_combined, metrics


# ---------------------------------------------------------------------------
# F2 Phase B.2 — TEBD circuit step (MPS state-prep + W-II gate layer)
# ---------------------------------------------------------------------------


def _pad_to_qubit_unitary(g: np.ndarray) -> tuple[np.ndarray, int]:
    """Embed a (D, D) unitary into a ``2^n_g`` x ``2^n_g`` unitary.

    Returns (padded_gate, n_g) where ``n_g = ceil(log2(D))``.  The
    original block occupies the top-left; the remaining diagonal is
    identity, so the padded matrix is still unitary.  No-op if D is
    already a power of 2.
    """
    d = g.shape[0]
    n_g = int(np.ceil(np.log2(d)))
    target_dim = 1 << n_g
    if target_dim == d:
        return g, n_g
    padded = np.eye(target_dim, dtype=g.dtype)
    padded[:d, :d] = g
    return padded, n_g


def _tile_wii_gates(
    circuit: Any,
    gates: list[np.ndarray],
    n_phys: int,
) -> None:
    """Append W-II gates to `circuit` using MPS site-to-qubit mapping.

    Gate list convention (matches `burgers_tebd.build_wii_layer` X2 form):
      - One gate per physical site.  Each gate is square with dim
        ``2 * D`` where ``D`` is the uniform ancilla bond dim.  ``D`` is
        not required to be a power of 2; non-power-of-two dims are
        padded to the next power of 2 by identity embedding so the
        gate becomes a proper n-qubit unitary.
      - Each gate acts on one physical qubit plus the ancilla bond
        register.
      - Physical qubit layout follows `mps_to_circuit`: MPS site 0 maps
        to qubit ``n_phys - 1``, site k to qubit ``n_phys - 1 - k``.

    The gate list reconstructs the dense U via `wii_gates_to_dense`,
    which clamps the left boundary ancilla to |0> on input and the
    right boundary ancilla to |0> on output.  Wrapping each gate as a
    UnitaryGate on [phys_site_k] + [ancilla bond qubits] reproduces
    that contraction when the ancilla register starts in |0>.

    Assumption (flagged for reviewer): B.1 currently emits one gate per
    site with uniform (2D, 2D) shape; every gate shares the same
    ancilla register.  When B.1's upgrade lands with per-site variable-
    size gates (variable D), this helper already consumes exactly
    ``n_g = ceil(log2(shape[0]))`` qubits per gate, but the ancilla-
    sharing convention may need revisiting (e.g. disjoint ancilla
    registers per gate).
    """
    from qiskit.circuit.library import UnitaryGate

    if not gates:
        return

    dim = gates[0].shape[0]
    for g in gates:
        if g.shape != (dim, dim):
            raise NotImplementedError(
                "Non-uniform W-II gate widths are not yet supported; "
                "Phase B.1 upgrade required.  See _tile_wii_gates docstring."
            )

    # MPS site k -> physical qubit (n_phys - 1 - k).  Apply gates in MPS
    # site order so the ancilla contraction matches `wii_gates_to_dense`.
    for k, g in enumerate(gates):
        padded, n_g = _pad_to_qubit_unitary(g)
        n_bond = n_g - 1
        if circuit.num_qubits < n_phys + n_bond:
            raise ValueError(
                f"Circuit has {circuit.num_qubits} qubits but needs "
                f"{n_phys + n_bond} for W-II gate (n_phys={n_phys}, "
                f"n_bond={n_bond})."
            )
        ancilla_qubits = list(range(n_phys, n_phys + n_bond))
        phys_qubit = n_phys - 1 - k
        target = [phys_qubit] + ancilla_qubits
        circuit.append(UnitaryGate(padded, label=f"W_{k}"), target)


def tebd_circuit_step(
    u: np.ndarray,
    x: np.ndarray,
    dt: float,
    nu: float,
    g: np.ndarray | None = None,
    bond_dim: int | None = None,
    threshold: float = 0.0,
    shots: int = 0,
    backend: Any = None,
    sign_recovery: str = "none",
    bc: str = "periodic",
    splitting: str = "lie",
    readout: str = "direct",
) -> tuple[np.ndarray, dict]:
    """One TEBD circuit step: MPS state prep -> W-II layer -> measure.

    Lie–Trotter splitting: ladder-form MPO advection on the quantum
    circuit, then a classical diffusion sub-step via ``diffusion_rhs``.

    ``g`` is accepted for API compatibility but ignored: the ladder MPO
    encodes only the symmetrized advection generator and has no forcing
    term. Pass forcing through a separate classical sub-step if needed.

    readout: 'direct' post-selects bond ancillas and estimates amplitudes
    from sqrt(counts).  'hadamard_per_bin' wraps the full circuit in a
    Hadamard test per basis bin to recover signed Re(psi_k) — use with
    shots > 0 to avoid sign loss (F2-10).
    """
    from qiskit import QuantumCircuit as QC
    from qiskit.quantum_info import Statevector

    if sign_recovery != "none":
        raise NotImplementedError(
            f"tebd_circuit_step does not support sign_recovery="
            f"{sign_recovery!r} in Phase B; see Phase C (Hadamard test) "
            "in F2-IMPLEMENTATION-SPEC.md §9."
        )

    if readout not in ("direct", "hadamard_per_bin"):
        raise ValueError(
            f"tebd_circuit_step: unknown readout={readout!r}; "
            "expected 'direct' or 'hadamard_per_bin'"
        )

    dx = float(x[1] - x[0])

    # Strang splitting: apply half diffusion step to u before advection circuit,
    # then apply another half after.  Lie-Trotter uses u directly and applies
    # the full diffusion step after.
    if splitting == "strang":
        u_in = u + (dt / 2.0) * diffusion_rhs(u, dx, nu, bc)
    else:
        u_in = u

    # 1. Ladder-form advection MPO built from the circuit input state.
    H_mpo = build_hamiltonian_mpo_ladder(u_in, dx, bc=bc)

    # 2. Amplitude encoding: normalize and save norm for decode.
    u_norm_vec, norm_u = normalize_state(u_in)

    # 3. MPS state-prep circuit (right-canonical — required by Ran 2020
    #    unitary completion used in burgers_mps.mps_to_circuit).
    tensors = classical_to_mps(
        u_norm_vec, bond_dim=bond_dim, threshold=threshold,
        canonical="right",
    )
    prep_qc = mps_to_circuit(tensors)
    n_phys = len(tensors)
    n_prep_total = prep_qc.num_qubits
    n_prep_bond = n_prep_total - n_phys

    # 4. Build W-II gate layer and determine ancilla width required.
    #    Gate dim may not be a power of 2; pad-plan to next power of 2.
    gates = build_wii_layer_ladder(H_mpo, dt)
    if not gates:
        raise RuntimeError("build_wii_layer returned no gates")
    gate_dim = gates[0].shape[0]
    n_g = int(np.ceil(np.log2(gate_dim)))
    n_wii_bond = max(n_g - 1, 0)

    # Ancilla registers: MPS prep uses the first n_prep_bond ancilla
    # qubits; the W-II layer needs n_wii_bond ancillas.  Re-use the
    # MPS bond qubits (they return to |0> at end of prep for exact MPS)
    # and extend if W-II needs more.
    n_total = n_phys + max(n_prep_bond, n_wii_bond)

    qc = QC(n_total)
    qc.compose(prep_qc, qubits=list(range(n_prep_total)), inplace=True)
    _tile_wii_gates(qc, gates, n_phys)

    # 5. Execute.
    readout_meta: dict[str, Any] = {}
    if not shots:
        sv = Statevector.from_instruction(qc)
        # extract_physical_state takes (sv, n_phys, n_bond).
        u_evolved_norm = extract_physical_state(
            sv, n_phys, n_total - n_phys,
        ).real
    else:
        if backend is None:
            from qiskit_aer import AerSimulator
            backend = AerSimulator()

        if readout == "hadamard_per_bin":
            from qiskit.compiler import transpile as _transpile
            from burgers_cole_hopf_circuit import (
                hadamard_per_bin_circuit,
                extract_hadamard_per_bin_amplitudes,
            )
            N_bins = 2**n_phys
            n_anc_evo = n_total - n_phys
            trivial_prep = QC(n_phys)
            per_k_counts: list[dict[str, int]] = []
            for k in range(N_bins):
                qc_k = hadamard_per_bin_circuit(
                    n_phys, trivial_prep, 0, qc, n_anc_evo, k,
                )
                qc_t = _transpile(qc_k, backend, optimization_level=1)
                result = backend.run(qc_t, shots=shots).result()
                per_k_counts.append(result.get_counts())
            u_evolved_norm, agg = extract_hadamard_per_bin_amplitudes(
                per_k_counts, n_phys, 0, n_anc_evo, shots, phi_norm=1.0,
            )
            readout_meta = {"readout": "hadamard_per_bin", **agg}
        else:
            from qiskit.compiler import transpile
            qc_m = qc.copy()
            qc_m.measure_all()
            qc_t = transpile(qc_m, backend, optimization_level=1)
            result = backend.run(qc_t, shots=shots).result()
            counts = result.get_counts()
            # Post-select: bond qubits == 0 (high bits of Qiskit bitstring).
            N = 2**n_phys
            amps = np.zeros(N)
            total = 0
            for bitstr, cnt in counts.items():
                idx = int(bitstr.replace(" ", ""), 2)
                if idx < N:
                    amps[idx] = cnt
                    total += cnt
            if total > 0:
                amps = np.sqrt(amps / total)
            u_evolved_norm = amps
            readout_meta = {"readout": "direct"}

    u_adv = u_evolved_norm * norm_u

    if splitting == "strang":
        u_next = u_adv + (dt / 2.0) * diffusion_rhs(u_adv, dx, nu, bc)
        splitting_label = "strang"
    else:
        u_next = u_adv + dt * diffusion_rhs(u_adv, dx, nu, bc)
        splitting_label = "lie"

    metrics = {
        "method": "tebd_circuit",
        "n_qubits": qc.num_qubits,
        "n_phys_qubits": n_phys,
        "n_bond_qubits": n_total - n_phys,
        "circuit_depth": qc.depth(),
        "gate_counts": dict(qc.count_ops()),
        "n_gates": int(sum(qc.count_ops().values())),
        "wii_gate_dim": gate_dim,
        "sign_recovery": sign_recovery,
        "splitting": splitting_label,
        **readout_meta,
    }

    return u_next, metrics


# ---------------------------------------------------------------------------
# Multi-step simulation
# ---------------------------------------------------------------------------


def run_simulation(
    u0: np.ndarray,
    x: np.ndarray,
    nu: float,
    dt: float,
    n_steps: int,
    source_fn=None,
    method: str = "shift",
    trotter_order: int = 1,
    trotter_reps: int = 1,
    bond_dim: int | None = None,
    mps_threshold: float = 0.0,
    shots: int = 0,
    backend_name: str | None = None,
    t1: float | None = None,
    t2: float | None = None,
    bc: str = "periodic",
    sign_recovery: str = "none",
    propagator: str = "qft-diagonal",
    snapshot_interval: int = 1,
    encoding: str = "binary",
    backend_type: str = "sim",
    coupling_map: str = "default",
    optimization_level: int = 1,
    seed: int | None = None,
    evolution_mode: str = "single",
    chunk_size: int = 10,
    phi_modes: int = 0,
    readout: str = "direct",
) -> tuple[list[np.ndarray], list[dict] | None]:
    """Run multi-step Burgers simulation.

    Parameters
    ----------
    u0 : initial velocity field
    x : grid coordinates
    nu : kinematic viscosity
    dt : time step
    n_steps : number of Euler steps
    source_fn : callable(x, t) -> np.ndarray, or None
    method : "shift" (classical), "quantum_exact", or "quantum_circuit"
    trotter_order : Suzuki-Trotter order (for quantum_circuit)
    trotter_reps : Trotter repetitions (for quantum_circuit)
    shots : shots for quantum_circuit (0=statevector, >0=sampling)
    backend_name : fake backend name for noise modeling (None=ideal)
    t1, t2 : T1/T2 relaxation times in µs (None=no thermal noise)

    Returns
    -------
    (solutions, circuit_metrics) where solutions is [u0, u1, ..., u_{n_steps}]
    and circuit_metrics is a list of per-step metric dicts for quantum methods
    (quantum_circuit, mps, quantum_exact) or None for shift.
    """
    # TEBD / Cole-Hopf: delegate to the multi-step drivers that own their
    # own loop and return (solutions, metrics_list) directly.
    if method == "tebd":
        sols, mets = run_tebd_simulation(
            u0, x, nu, dt, n_steps,
            source_fn=source_fn, bc=bc,
            max_bond=bond_dim, cutoff=mps_threshold or 1e-14,
        )
        return sols, mets
    if method == "cole_hopf":
        sols, mets = run_cole_hopf_simulation(
            u0, x, nu, dt, n_steps, bc=bc,
            max_bond=bond_dim, cutoff=mps_threshold or 1e-14,
        )
        return sols, mets
    if method == "cole_hopf_circuit":
        backend = None
        if shots > 0:
            from q8020_cfd_qutil.backend import get_backend
            backend = get_backend(
                name=backend_name,
                backend_type=backend_type,
                t1=t1, t2=t2,
                coupling_map=coupling_map,
            )
        sols, mets = run_cole_hopf_circuit_simulation(
            u0, x, nu, dt, n_steps, bc=bc,
            propagator=propagator, shots=shots,
            snapshot_interval=snapshot_interval,
            bond_dim=bond_dim, encoding=encoding,
            backend=backend,
            backend_type=backend_type,
            backend_name=backend_name,
            optimization_level=optimization_level,
            seed=seed,
            source_fn=source_fn,
            evolution_mode=evolution_mode,
            chunk_size=chunk_size,
            phi_modes=phi_modes,
            readout=readout,
        )
        return sols, mets

    dx = x[1] - x[0]
    solutions = [u0.copy()]
    u = u0.copy()
    circuit_metrics: list[dict] | None = (
        [] if method in (
            "quantum_circuit", "mps", "quantum_exact", "tebd_circuit",
        ) else None
    )

    # Pre-build backend once to avoid per-step overhead.  Built for all
    # quantum methods (including shots=0 statevector) so that T1/T2 noise
    # configuration is respected even in statevector mode.
    backend: Any = None
    if method in ("quantum_circuit", "mps", "tebd_circuit"):
        from q8020_cfd_qutil.backend import get_backend
        backend = get_backend(name=backend_name, backend_type="sim", t1=t1, t2=t2)

    for step in range(n_steps):
        t = step * dt
        g = source_fn(x, t) if source_fn is not None else None

        if method == "quantum_circuit" and sign_recovery == "dual_rail":
            u, metrics = dual_rail_quantum_step(
                u, dx, dt, nu, g, trotter_order, trotter_reps,
                shots=shots, backend=backend, t1=t1, t2=t2,
                backend_name=backend_name, bc=bc,
            )
        elif method == "quantum_circuit":
            u, metrics = quantum_circuit_step(
                u, dx, dt, nu, g, trotter_order, trotter_reps,
                shots=shots, backend=backend, t1=t1, t2=t2,
                backend_name=backend_name, bc=bc, sign_recovery=sign_recovery,
            )
        elif method == "mps":
            u, metrics = mps_step(
                u, dx, dt, nu, g, bond_dim, mps_threshold, shots, bc,
                sign_recovery=sign_recovery, backend=backend,
            )
        elif method == "tebd_circuit":
            u, metrics = tebd_circuit_step(
                u, x, dt, nu, g, bond_dim=bond_dim,
                threshold=mps_threshold, shots=shots, backend=backend,
                sign_recovery=sign_recovery, bc=bc,
            )
        elif method == "quantum_exact":
            u, metrics = quantum_exact_step(u, dx, dt, nu, g, bc=bc)
        else:
            u, metrics = shift_euler_step(u, dx, dt, nu, g, bc=bc)
        if metrics:
            circuit_metrics.append(metrics)  # type: ignore[union-attr]

        if not np.all(np.isfinite(u)):
            import sys
            print(
                f"[burgers] {method} diverged at step {step + 1}/{n_steps}; "
                f"padding remaining steps with NaN",
                file=sys.stderr, flush=True,
            )
            nan_fill = np.full_like(u, np.nan)
            solutions.extend([nan_fill] * (n_steps - step))
            break
        solutions.append(u.copy())

    return solutions, circuit_metrics


def compute_error(
    u_quantum: np.ndarray,
    u_classical: np.ndarray,
) -> float:
    """Normalized L2 error: ||u_q - u_c||₂ / ||u_c||₂  (paper's ε metric)."""
    if not np.all(np.isfinite(u_classical)):
        return float("nan")
    norm_c = np.linalg.norm(u_classical)
    if norm_c < 1e-15:
        return 0.0
    return float(np.linalg.norm(u_quantum - u_classical) / norm_c)


if __name__ == "__main__":  # smoke test only — not the solver entry point
    from burgers_classical import initial_condition_sine, source_term_sine

    # Small test: N=8, short simulation
    q = 3
    N = 2**q
    x = np.linspace(0, 1, N, endpoint=False)
    dx = x[1] - x[0]
    nu = 1e-4
    dt = 0.1 * dx  # CFL ~ 0.1
    n_steps = 10

    u0 = initial_condition_sine(x)

    print(f"=== Time Evolution Test (N={N}, {n_steps} steps) ===")
    print(f"  dx={dx:.4f}, dt={dt:.4e}, CFL={dt/dx:.2f}")

    # Classical baseline
    sols_shift, _ = run_simulation(
        u0, x, nu, dt, n_steps,
        source_fn=source_term_sine, method="shift",
    )

    # Quantum exact
    sols_qexact, _ = run_simulation(
        u0, x, nu, dt, n_steps,
        source_fn=source_term_sine, method="quantum_exact",
    )

    print(f"\n  Step  |  Shift max|u|  |  Q-exact max|u|  |  Error ε")
    print(f"  {'─'*55}")
    for step in [0, 1, 5, n_steps]:
        u_s = sols_shift[step]
        u_q = sols_qexact[step]
        err = compute_error(u_q, u_s)
        print(
            f"  {step:4d}  |  {np.max(np.abs(u_s)):12.6f}  |  "
            f"{np.max(np.abs(u_q)):14.6f}  |  {err:.2e}"
        )

    # F2 Phase B.2 smoke: tebd_circuit vs classical tebd at matched params.
    q_sm = 3
    N_sm = 2**q_sm
    x_sm = np.linspace(0, 1, N_sm, endpoint=False)
    dx_sm = x_sm[1] - x_sm[0]
    dt_sm = 1e-4
    nu_sm = 1e-2
    n_steps_sm = 5
    u0_sm = initial_condition_sine(x_sm)

    sols_tc, _ = run_simulation(
        u0_sm, x_sm, nu_sm, dt_sm, n_steps_sm, method="tebd_circuit",
    )
    sols_tebd, _ = run_simulation(
        u0_sm, x_sm, nu_sm, dt_sm, n_steps_sm, method="tebd",
    )
    max_err = float(np.max(np.abs(sols_tc[-1] - sols_tebd[-1])))
    print(
        f"\n  tebd_circuit vs classical tebd at q={q_sm} "
        f"dt={dt_sm:.0e} n_steps={n_steps_sm}: max|err| = {max_err:.3e}"
    )
