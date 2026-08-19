"""Quantum circuit Cole-Hopf Burgers solver (F10 Phase B).

qft-diagonal propagator for the heat equation exp(nu*L*dt):
QFT + diagonal phase-damping + QFT^-1 (periodic BC), or DCT-II +
phase-damping + DCT^T (Neumann-on-phi = u-Dirichlet).

A single ancilla qubit carries the non-unitary block encoding.
Post-selection on ancilla |0> recovers the contractive propagator.
"""

from __future__ import annotations

import datetime
import sys
import time
from typing import Any

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit import transpile as _qiskit_transpile
from qiskit.circuit.library import QFTGate, RYGate, UnitaryGate
from qiskit.quantum_info import Statevector
from q8020_backend_utils.ibm.circuit import (
    DEFAULT_METRIC_BASIS,
    execute_circuit_counts,
    get_circuit_info,
    safe_circuit_stats_in_basis,
    transpile_circuit,
)
from q8020_backend_utils.ibm.job import submit_job

from lib_cole_hopf import (
    _should_center,
    cole_hopf_forward,
    cole_hopf_forward_centered,
    cole_hopf_inverse,
    fourier_low_pass_phi,
    log_phi_to_normalized_psi,
)
from lib_mps import (
    classical_to_mps,
    mps_to_circuit,
    normalize_state,
)

# ── Laplacian eigenvalues ────────────────────────────────────────────


def laplacian_eigenvalues(q: int, L_box: float) -> np.ndarray:
    """Discrete periodic Laplacian eigenvalues for N = 2^q grid.

    Returns array of length N with
    lambda_j = -(4/dx^2) * sin^2(pi*j/N), j = 0..N-1.
    All values are <= 0.
    """
    N = 1 << q
    dx = L_box / N
    j = np.arange(N)
    return -(4.0 / dx**2) * np.sin(np.pi * j / N) ** 2


# ── P2: Exact theta(k) ──────────────────────────────────────────────


def compute_theta_exact(
    nu: float,
    dt: float,
    q: int,
    L_box: float,
) -> np.ndarray:
    """Exact rotation angles theta(k) = arccos(exp(nu*lambda_k*dt)).

    Returns array of length N = 2^q.
    """
    evals = laplacian_eigenvalues(q, L_box)
    damping = np.exp(nu * evals * dt)
    damping = np.clip(damping, 0.0, 1.0)
    return np.arccos(damping)


# ── P2: Multilinear (Mobius) expansion ──────────────────────────────


def _mobius_transform(f_values: np.ndarray, q: int) -> np.ndarray:
    """Mobius/inclusion-exclusion: f(k) -> multilinear coefficients a[S].

    f(k) = sum_S a[S] * prod_{i in S} b_i  where k = sum 2^i b_i.
    S is encoded as an integer bitmask.
    """
    N = 1 << q
    a = f_values.astype(float).copy()
    for i in range(q):
        mask = 1 << i
        for j in range(N):
            if j & mask:
                a[j] -= a[j ^ mask]
    return a


# ── P2: Conditional Möbius-Ry circuit ───────────────────────────────


def build_conditional_ry(
    data_qubits: list[int],
    ancilla: int,
    theta_exact: np.ndarray,
) -> QuantumCircuit:
    """Build conditional Ry(2*theta(k)) on ancilla, controlled by data.

    theta_exact: array of length 2^q with exact rotation angles.

    After application: |k>|0> -> |k>(cos theta(k)|0> + sin theta(k)|1>).

    Decomposition: Mobius (inclusion-exclusion) expansion of
    2*theta(k) into multilinear terms over bit variables, then one
    (multi-)controlled Ry per nonzero Mobius coefficient.  Gate count
    is O(2^q) — honest about the cost at the q values we run.
    """
    q = len(data_qubits)
    N = 1 << q

    ry_angles = 2.0 * theta_exact
    ml = _mobius_transform(ry_angles, q)

    n_total = max(*data_qubits, ancilla) + 1
    qc = QuantumCircuit(n_total, name="mobius_ry")

    for bitmask in range(N):
        angle = float(ml[bitmask])
        if abs(angle) < 1e-15:
            continue

        ctrl_idx = [i for i in range(q) if bitmask & (1 << i)]
        ctrl_qb = [data_qubits[i] for i in ctrl_idx]

        if not ctrl_qb:
            qc.ry(angle, ancilla)
        elif len(ctrl_qb) == 1:
            qc.cry(angle, ctrl_qb[0], ancilla)
        else:
            gate = RYGate(angle).control(len(ctrl_qb))
            qc.append(gate, ctrl_qb + [ancilla])

    return qc


# ── DCT-II propagator (Neumann BC on phi) ──────────────────────────
#
# Historical note: F10-IMPLEMENTATION-SPEC §8 and FUTURE-WORK §4 call
# this the "DST-based" path.  The actual transform is DCT-II — the CH
# transform maps u-Dirichlet → phi-Neumann, and Neumann is diagonalised
# by cosines (DCT), not sines (DST).  Eigenvalue formula is the same
# for both; the basis differs.


def dct_matrix(q: int) -> np.ndarray:
    """Orthonormal DCT-II matrix on N = 2^q points.

    C[k, n] = alpha_k * cos(pi * k * (2n + 1) / (2N)),
    with alpha_0 = 1/sqrt(N), alpha_{k>0} = sqrt(2/N).
    Real, orthogonal: C @ C.T = I.  C diagonalises the half-cell
    mirror Neumann Laplacian to lambda_k = -(4/dx^2) sin^2(pi k/(2N)).
    """
    N = 1 << q
    n = np.arange(N)
    k = np.arange(N).reshape(-1, 1)
    C = np.cos(np.pi * k * (2 * n + 1) / (2 * N))
    C[0, :] *= 1.0 / np.sqrt(N)
    C[1:, :] *= np.sqrt(2.0 / N)
    return C


def neumann_laplacian_eigenvalues(q: int, L_box: float) -> np.ndarray:
    """Neumann (half-cell mirror) Laplacian eigenvalues on N = 2^q.

    lambda_k = -(4/dx^2) sin^2(pi k / (2N)) for k = 0..N-1.
    The k=0 mode is the constant-mode null space (lambda_0 = 0).
    """
    N = 1 << q
    dx = L_box / N
    k = np.arange(N)
    return -(4.0 / dx**2) * np.sin(np.pi * k / (2 * N)) ** 2


def compute_theta_dct(
    nu: float,
    dt: float,
    q: int,
    L_box: float,
) -> np.ndarray:
    """Rotation angles theta_k = arccos(exp(nu*lambda_k*dt)) for DCT.

    Uses Neumann Laplacian eigenvalues.  k=0 mode has lambda=0, so
    damping=1 and theta_0 = 0 (no rotation, mode preserved exactly).
    """
    evals = neumann_laplacian_eigenvalues(q, L_box)
    damping = np.exp(nu * evals * dt)
    damping = np.clip(damping, 0.0, 1.0)
    return np.arccos(damping)


def heat_dct_full_circuit(
    q: int,
    nu: float,
    T: float,
    N_steps: int,
    L_box: float,
) -> QuantumCircuit:
    """Full evolution circuit: N_steps DCT-diagonal Trotter layers."""
    dt = T / N_steps

    data = list(range(q))
    anc = q

    anc_cr = ClassicalRegister(N_steps, "anc_hist")
    data_cr = ClassicalRegister(q, "data")
    qc = QuantumCircuit(q + 1, name="heat_dct_full")
    qc.add_register(anc_cr)
    qc.add_register(data_cr)

    theta = compute_theta_dct(nu, dt, q, L_box)
    C = dct_matrix(q)
    dct_gate = UnitaryGate(C, label="DCT")
    dct_dag_gate = UnitaryGate(C.T, label="DCT_dag")
    cond_qc = build_conditional_ry(data, anc, theta)

    for step in range(N_steps):
        qc.append(dct_gate, data)
        qc.compose(cond_qc, inplace=True)
        qc.append(dct_dag_gate, data)
        qc.measure(anc, anc_cr[step])
        if step < N_steps - 1:
            qc.reset(anc)

    for i in range(q):
        qc.measure(data[i], data_cr[i])

    return qc


# ── P3: QFT-diagonal propagator step ────────────────────────────────


def heat_qft_full_circuit(
    q: int,
    nu: float,
    T: float,
    N_steps: int,
    L_box: float,
    bc: str = "periodic",
    max_qubits: int | None = None,
) -> QuantumCircuit:
    """Full evolution circuit: N_steps QFT-diagonal Trotter layers.

    Blocked deferred measurement, sized to the qubit budget.  The
    max_qubits budget leaves (max_qubits - q) ancilla qubits; the
    evolution is split into blocks of that many steps.  Within a block
    each step gets its own post-selection ancilla (deferred — no
    mid-circuit measurement); at the block boundary the whole ancilla
    register is measured into anc_hist, then reset and reused for the
    next block.  Every step still writes a distinct anc_hist bit, so
    post-selection (anc register all-zero) is unchanged.

    This interpolates the two extremes:
      * block >= N_steps  -> fully deferred, one block, no reset:
        q+N_steps qubits, zero mid-circuit measurement (Aer's cheap
        sample-once-from-statevector path).
      * block == 1        -> single reused ancilla, measure+reset every
        step: q+1 qubits, N_steps measurement rounds.
    The default sits between them: it uses every qubit the budget allows
    so the circuit fits while minimising measurement rounds
    (ceil(N_steps / block)), which keeps Aer closer to the fast path
    than the one-ancilla fallback would.

    max_qubits: total qubit budget.  Defaults to 2*q, i.e. q ancilla,
        so the deferred block is q steps wide and the circuit never
        exceeds twice the data register.

    bc='neumann' dispatches to the DCT-II propagator path.
    """
    if bc == "neumann":
        return heat_dct_full_circuit(q, nu, T, N_steps, L_box)
    if bc != "periodic":
        raise NotImplementedError(
            f"heat_qft_full_circuit: unsupported bc={bc!r}; "
            "expected 'periodic' or 'neumann'"
        )

    if max_qubits is None:
        max_qubits = 2 * q

    dt = T / N_steps

    data = list(range(q))
    # Ancilla qubits the budget allows, and the deferred block width.
    n_anc = max(1, max_qubits - q)
    block = min(n_anc, N_steps)

    theta = compute_theta_exact(nu, dt, q, L_box)
    qft = QFTGate(q)
    qft_inv = QFTGate(q).inverse()
    cond = [build_conditional_ry(data, q + j, theta) for j in range(block)]

    anc_cr = ClassicalRegister(N_steps, "anc_hist")
    data_cr = ClassicalRegister(q, "data")
    qc = QuantumCircuit(q + block, name="heat_qft_full")
    qc.add_register(anc_cr)
    qc.add_register(data_cr)

    for step in range(N_steps):
        j = step % block            # which ancilla within the block
        qc.append(qft, data)
        qc.compose(cond[j], inplace=True)
        qc.append(qft_inv, data)
        # Block boundary (or final step): flush ancilla -> anc_hist.
        if j == block - 1 or step == N_steps - 1:
            blk_start = step - j
            for k in range(j + 1):
                qc.measure(q + k, anc_cr[blk_start + k])
            if step < N_steps - 1:
                for k in range(j + 1):
                    qc.reset(q + k)

    # Final data measurement
    for i in range(q):
        qc.measure(data[i], data_cr[i])

    return qc


# ── Statevector simulation driver ───────────────────────────────────


def _build_step_sv(
    q: int,
    nu: float,
    dt: float,
    L_box: float,
    bc: str,
) -> QuantumCircuit:
    """Build a measurement-free qft-diagonal step circuit for SV sim."""
    data = list(range(q))
    anc = q
    qc = QuantumCircuit(q + 1)
    if bc == "neumann":
        theta = compute_theta_dct(nu, dt, q, L_box)
        C = dct_matrix(q)
        qc.append(UnitaryGate(C, label="DCT"), data)
        cond_qc = build_conditional_ry(data, anc, theta)
        qc.compose(cond_qc, inplace=True)
        qc.append(UnitaryGate(C.T, label="DCT_dag"), data)
        return qc
    theta = compute_theta_exact(nu, dt, q, L_box)
    qc.append(QFTGate(q), data)
    cond_qc = build_conditional_ry(data, anc, theta)
    qc.compose(cond_qc, inplace=True)
    qc.append(QFTGate(q).inverse(), data)
    return qc


def run_cole_hopf_circuit_sv(
    psi0: np.ndarray,
    q: int,
    nu: float,
    dt: float,
    n_steps: int,
    L_box: float,
    bc: str = "periodic",
    snapshot_interval: int = 1,
    use_mps_prep: bool = True,
    bond_dim: int | None = None,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Run Cole-Hopf circuit evolution via statevector simulation.

    psi0: unit-norm initial state (length N = 2^q).
    use_mps_prep: if True (default), seed the statevector by running
        psi0 through the Ran 2020 MPS-to-circuit prep pipeline so that
        bond_dim truncation is visible in the SV path — matches the
        shots path for fair bond-dim sweeps.  If False, seed directly
        from psi0 (original behaviour).
    bond_dim: MPS truncation bond dimension (None = full rank).
        Only used when use_mps_prep is True.

    Returns (psi_snapshots, metrics_list) where each psi snapshot
    is the unit-norm data-register state after post-selection.
    """
    N = 1 << q

    # qft-diagonal uses one post-selection ancilla per step.
    shared_step = _build_step_sv(q, nu, dt, L_box, bc)
    n_anc = shared_step.num_qubits - q
    sv_dim = N * (1 << n_anc)
    sv = np.zeros(sv_dim, dtype=complex)
    if use_mps_prep:
        psi_in, _ = normalize_state(psi0)
        tensors = classical_to_mps(
            psi_in, bond_dim=bond_dim, canonical="right",
        )
        prep_qc = mps_to_circuit(tensors)
        n_bond = prep_qc.num_qubits - q
        sv_prep = (
            Statevector.from_label("0" * (q + n_bond))
            .evolve(prep_qc).data
        )
        # Physical state when bond = |0>: first N amplitudes.
        psi_prepared = sv_prep[:N]
        nrm = float(np.linalg.norm(psi_prepared))
        if nrm > 1e-15:
            psi_prepared = psi_prepared / nrm
        sv[:N] = psi_prepared
    else:
        sv[:N] = psi0

    snapshots: list[np.ndarray] = []
    metrics: list[dict[str, Any]] = []
    p_success_total = 1.0

    for step in range(n_steps):
        sv = Statevector(sv).evolve(shared_step).data
        p0 = float(np.sum(np.abs(sv[:N]) ** 2))
        p_success_total *= p0
        sv_proj = np.zeros_like(sv)
        if p0 > 1e-30:
            sv_proj[:N] = sv[:N] / np.sqrt(p0)
        sv = sv_proj

        metrics.append({
            "step": step + 1,
            "p_success_step": p0,
            "p_success_total": p_success_total,
        })

        if (step + 1) % snapshot_interval == 0 or step == n_steps - 1:
            snapshots.append(np.real(sv[:N]).copy())

    return snapshots, metrics


# ── Shared shots helpers ─────────────────────────────────────────────


def post_select_counts(
    counts: dict[str, int],
    q: int,
) -> tuple[int, dict[str, int]]:
    """Post-select on ancilla = all-zero; return (n_kept, data_counts)."""
    n_kept = 0
    data_counts: dict[str, int] = {}
    for bitstring, count in counts.items():
        data_bits = bitstring[:q]
        anc_bits = bitstring[q:]
        if all(c == "0" for c in anc_bits):
            n_kept += count
            data_counts[data_bits] = (
                data_counts.get(data_bits, 0) + count
            )
    return n_kept, data_counts


def reconstruct_phi_from_counts(
    data_counts: dict[str, int],
    n_kept: int,
    N: int,
    phi_norm: float,
    p_success: float,
) -> np.ndarray:
    """Reconstruct phi amplitudes from post-selected data counts."""
    phi_hat = np.zeros(N)
    if n_kept > 0:
        for data_bits_k, cnt in data_counts.items():
            idx = int(data_bits_k, 2)
            phi_hat[idx] = np.sqrt(cnt / n_kept)
        phi_hat *= np.sqrt(p_success) * phi_norm
    return phi_hat


# ── Measure-and-reprepare (segmented) shots driver ─────────────────────────────────────────────


def build_segment_circuit(
    psi_current: np.ndarray,
    q: int,
    nu: float,
    dt: float,
    segment_size: int,
    L_box: float,
    bc: str,
    bond_dim: int | None = None,
    use_mps_prep: bool = True,
    max_total_qubits: int | None = None,
) -> tuple[QuantumCircuit, int, int, int]:
    """Build one measure-reprepare segment circuit (prep + segment_size
    Trotter steps), composed and ready to transpile.

    Single source of truth for segment construction: used by the
    measure-reprepare loop AND the F12 hardware-runner dry-run so the
    reported CX/depth always matches what actually executes.

    max_total_qubits: absolute cap on the segment width q + n_bond +
        n_heat_anc.  None (default) leaves the propagator at its own 2*q
        default (i.e. min(segment_size, q) heat ancillas).  When set, the
        heat-ancilla budget is (max_total_qubits - q - n_bond), computed
        AFTER n_bond is known so the MPS bond qubits are accounted for.  If
        that budget is >= segment_size the segment is fully deferred (no
        mid-circuit reset -> Aer's fast sample-once path); if smaller, the
        propagator tiles it into ceil(segment_size / budget) reset rounds
        (the per-shot trajectory path -- slow on Aer, avoid for sim).

    Returns (raw_qc, total_q, n_bond, n_heat_anc).
    """
    # 1. Prep circuit from current amplitudes
    if use_mps_prep:
        tensors = classical_to_mps(
            psi_current, bond_dim=bond_dim, canonical="right",
        )
        prep_qc = mps_to_circuit(tensors)
        n_bond = prep_qc.num_qubits - q
    else:
        prep_qc = QuantumCircuit(q, name="initialize_prep")
        prep_qc.initialize(psi_current.tolist(), range(q))
        n_bond = 0

    # 2. segment_size-step evolution circuit (qft-diagonal).  With a total
    #    cap set, hand the propagator a bond-aware qubit budget: heat
    #    ancillas get whatever the cap leaves after data + bond.  Fail fast
    #    (before the expensive transpile) if the cap can't seat even one.
    heat_max_qubits = None
    if max_total_qubits is not None:
        heat_budget = max_total_qubits - n_bond
        if heat_budget < q + 1:
            raise ValueError(
                f"max_total_qubits={max_total_qubits} too small: q={q} + "
                f"n_bond={n_bond} leaves no room for a heat ancilla "
                f"(need >= {q + n_bond + 1})"
            )
        heat_max_qubits = heat_budget
    T_segment = dt * segment_size
    full_qc = heat_qft_full_circuit(
        q, nu, T_segment, segment_size, L_box, bc=bc,
        max_qubits=heat_max_qubits,
    )

    n_heat_anc = full_qc.num_qubits - q
    total_q = q + n_bond + n_heat_anc
    heat_qubit_map = list(range(q)) + list(
        range(q + n_bond, q + n_bond + n_heat_anc),
    )

    # 3. Compose
    init_qc = QuantumCircuit(total_q)
    init_qc.compose(prep_qc, qubits=list(range(q + n_bond)), inplace=True)
    init_qc.compose(full_qc, qubits=heat_qubit_map, inplace=True)
    return init_qc, total_q, n_bond, n_heat_anc


def _run_shots_measure_reprepare(
    psi0: np.ndarray,
    q: int,
    nu: float,
    dt: float,
    snap_steps: list[int],
    L_box: float,
    phi_norm: float,
    bc: str,
    shots: int,
    segment_size: int,
    bond_dim: int | None = None,
    backend: Any = None,
    backend_type: str = "sim",
    backend_name: str | None = None,
    optimization_level: int = 1,
    seed: int | None = None,
    use_mps_prep: bool = True,
    metric_transpile_timeout: float = 60.0,
    allow_hardware: bool = False,
    session: Any = None,
    sampler_options: dict | None = None,
    initial_layout: list[int] | None = None,
    max_total_qubits: int | None = None,
) -> list[tuple[np.ndarray, dict[str, Any]]]:
    """Measure-and-reprepare (segmented) evolution: K segments of segment_size steps each.

    Between segments, classically read out post-selected amplitudes
    and re-prep as fresh IC.  No classical PDE physics — only
    amplitude IO (decode counts -> re-encode via MPS prep).

    use_mps_prep: True (default) uses Ran 2020 MPS-to-circuit prep
        (O(q*chi^2) gates).  False falls back to QuantumCircuit.
        initialize (O(2^q) gates) for comparison.

    allow_hardware (default False): the segmented loop is sim-only by
        default.  The F12 hardware runner opts in by passing True (with
        a real IBM backend + held Session); existing callers leave it
        False and keep the original guard behavior.
    session / sampler_options (default None): forwarded verbatim to
        execute_circuit_counts so each serial segment's SamplerV2 is
        session-bound and mitigated (TREX / dynamical decoupling).
    """
    if backend_type == "hardware":
        if not allow_hardware:
            raise NotImplementedError(
                "segmented evolution v1: sim only; hardware "
                "segmenting deferred to v2 (set allow_hardware=True to "
                "opt in)"
            )
        # A held Session lets the N serial QPU submissions share one queue
        # slot, avoiding N independent queue waits on a paid device. It is an
        # optimization, not a correctness requirement: each segment is
        # re-prepared classically from the prior segment's measured counts,
        # so session-less job mode yields identical results. The IBM open
        # (free) plan forbids Sessions, so session=None is the supported way
        # to run hardware there -- warn rather than block.
        if session is None:
            print(
                "[ch] WARNING: allow_hardware=True without a held Session: "
                f"the {max(snap_steps) // segment_size} serial segments will "
                "each open a separate queued job (required on the open plan).",
                file=sys.stderr, flush=True,
            )

    N = 1 << q
    n_steps_total = max(snap_steps)
    n_segments = n_steps_total // segment_size

    psi_current, init_norm = normalize_state(psi0)
    cumulative_norm = init_norm * phi_norm
    snapshots: dict[int, tuple[np.ndarray, dict[str, Any]]] = {}
    # Per-segment audit trail, recorded for EVERY segment (not just the
    # snapshot steps).  Under measure-reprepare each segment is a separate
    # QPU job; gating the record on snap_steps silently drops the job_id /
    # billed-usage provenance of the non-snapshot segments.  Stamped onto
    # the last snapshot's metrics below so the return signature is unchanged.
    seg_audit: list[dict[str, Any]] = []
    # Cache for out-of-process hardware-cost stats.  Each segment's prep
    # block differs (rebuilt from that segment's measured amplitudes), so
    # the cached stats are exact only for the segment that produced them;
    # later segments reuse them as an estimate and are labelled as such
    # (metric_stats_from_segment).  When a Session is held (hardware) the
    # inline metric transpile is skipped entirely -- it would burn QPU
    # reservation wall-clock on a classical decomposition, and the F12
    # dry-run already reports the authoritative segment CX/depth.
    metric_stats_cache = None
    metric_stats_from_segment: int | None = None
    skip_metric_transpile = session is not None

    print(
        f"[_run_shots_measure_reprepare] n_segments={n_segments} "
        f"segment_size={segment_size} shots={shots}",
        file=sys.stderr, flush=True,
    )
    t_total_start = time.time()

    for segment_idx in range(n_segments):
        t_segment = time.time()
        global_step_start = segment_idx * segment_size

        # 1-3. Build prep + evolution + compose (shared with the F12
        # dry-run so reported CX/depth matches what actually runs).
        raw_qc, total_q, n_bond, n_heat_anc = build_segment_circuit(
            psi_current, q, nu, dt, segment_size, L_box, bc,
            bond_dim=bond_dim, use_mps_prep=use_mps_prep,
            max_total_qubits=max_total_qubits,
        )
        # Construction wall time for this segment (prep + propagator
        # build + compose), measured from the segment start.
        construct_t = time.time() - t_segment

        # One-time path diagnostic: n_heat_anc < segment_size means the
        # propagator tiled the segment with mid-circuit reset, which forces
        # Aer's per-shot trajectory path (slow, ~linear in shots).  Warn so
        # the operator can raise --max-total-qubits or lower --segment-size.
        if segment_idx == 0 and backend_type != "hardware":
            reset_rounds = -(-segment_size // n_heat_anc)  # ceil
            sv_gib = (1 << total_q) * 16 / (1 << 30)
            if n_heat_anc < segment_size:
                print(
                    f"[_run_shots_measure_reprepare] SLOW PATH: "
                    f"segment_size={segment_size} > heat_anc={n_heat_anc} "
                    f"-> {reset_rounds} reset rounds/segment; Aer runs "
                    f"per-shot (~linear in shots). Raise --max-total-qubits "
                    f"to >= {q + n_bond + segment_size} (SV "
                    f"{(1 << (q + n_bond + segment_size)) * 16 / (1 << 30):.2f} "
                    f"GiB) for the fast deferred path, or lower "
                    f"--segment-size.",
                    file=sys.stderr, flush=True,
                )
            else:
                print(
                    f"[_run_shots_measure_reprepare] fast path: fully "
                    f"deferred (no reset), width={total_q} "
                    f"(SV {sv_gib:.2f} GiB), sample-once.",
                    file=sys.stderr, flush=True,
                )

        # 4. Transpile + execute
        #    AerSimulator handles arbitrary gates (ControlledGate,
        #    StatePreparation, UnitaryGate) natively.  Skipping
        #    transpile avoids a Qiskit qs_decomposition segfault
        print(
            f"[_run_shots_measure_reprepare] segment {segment_idx + 1}/"
            f"{n_segments} steps "
            f"{global_step_start+1}-"
            f"{global_step_start+segment_size} "
            f"depth={raw_qc.depth()} transpiling ...",
            file=sys.stderr, flush=True,
        )
        skip_stats = None
        metric_t = 0.0
        qc_t, t_info = transpile_circuit(
            raw_qc, backend,
            optimization_level=optimization_level,
            seed_transpiler=seed,
            initial_layout=initial_layout,
        )
        # Hardware-cost stats: decompose raw_qc to DEFAULT_METRIC_BASIS so
        # CH reports honest cx-depth/gate counts comparable to QLBM.  The
        # Aer execution path lowers to the backend basis, which does not
        # yield a faithful hardware cx-depth, so we measure it explicitly.
        # Run out-of-process so a qs_decomposition segfault/hang degrades
        # gracefully instead of killing the run.  Identical segments are
        # computed once and cached.
        if skip_metric_transpile:
            # Held Session: never run a classical decomposition on QPU
            # reservation time.  Authoritative CX/depth comes from the
            # F12 dry-run pre-flight (same build_segment_circuit).
            skip_stats = None
        else:
            if metric_stats_cache is None:
                _t_m = time.time()
                metric_stats_cache = safe_circuit_stats_in_basis(
                    raw_qc, DEFAULT_METRIC_BASIS,
                    optimization_level=optimization_level,
                    seed_transpiler=seed,
                    timeout=metric_transpile_timeout, try_decompose=False,
                )
                metric_stats_from_segment = segment_idx
                metric_t = time.time() - _t_m
            skip_stats = metric_stats_cache
        counts, exec_info = execute_circuit_counts(
            qc_t, backend, shots=shots, seed=seed,
            session=session, sampler_options=sampler_options,
        )

        # 5. Post-select and reconstruct
        n_kept, data_counts = post_select_counts(counts, q)
        p_success = n_kept / shots if shots > 0 else 0.0

        step_at_end = (segment_idx + 1) * segment_size

        if n_kept == 0:
            print(
                f"[_run_shots_measure_reprepare] segment {segment_idx + 1}: "
                f"ZERO post-selected counts; propagating NaN",
                file=sys.stderr, flush=True,
            )
            psi_current = np.full(N, np.nan)
            cumulative_norm = float("nan")
        else:
            # Reconstruct unit-norm amplitudes for re-prep
            psi_new = np.zeros(N)
            for data_bits_k, cnt in data_counts.items():
                idx = int(data_bits_k, 2)
                psi_new[idx] = np.sqrt(cnt / n_kept)
            psi_norm_new = np.linalg.norm(psi_new)
            if psi_norm_new > 1e-15:
                psi_new /= psi_norm_new
            cumulative_norm *= np.sqrt(p_success)
            psi_current = psi_new

        elapsed = time.time() - t_segment
        print(
            f"[_run_shots_measure_reprepare] segment {segment_idx + 1}/"
            f"{n_segments} done in {elapsed:.1f}s "
            f"p_success={p_success:.4f} "
            f"cumulative_norm={cumulative_norm:.4e}",
            file=sys.stderr, flush=True,
        )

        # Audit record for THIS segment (every segment, snapshot or not).
        # Mirrors the fields the runner reads off each metrics entry so the
        # full 6-job list can be written/harvested uniformly.
        seg_rec: dict[str, Any] = {
            "segment_idx": segment_idx,
            "step": step_at_end,
            "is_snapshot": step_at_end in snap_steps,
            "shots": shots,
            "n_kept": n_kept,
            "p_success": p_success,
            "cumulative_norm": cumulative_norm,
            "execution_time_s": exec_info.get("wall_time", 0.0),
            "execute": exec_info,
        }
        if skip_stats is not None and skip_stats.get("available"):
            seg_rec["circuit_depth"] = skip_stats["depth"]
            seg_rec["gate_counts"] = skip_stats["gate_counts"]
            seg_rec["n_qubits"] = skip_stats["num_qubits"]
            seg_rec["circuit_metrics_available"] = True
            seg_rec["circuit_metrics_from_segment"] = metric_stats_from_segment
            seg_rec["circuit_metrics_exact"] = (
                metric_stats_from_segment == segment_idx
            )
        else:
            seg_rec["circuit_metrics_available"] = False
        seg_audit.append(seg_rec)

        if step_at_end in snap_steps:
            # Scale for output: phi = psi * cumulative_norm
            phi_out = psi_current * cumulative_norm
            met: dict[str, Any] = {
                "shots": shots,
                "n_kept": n_kept,
                "p_success": p_success,
                "n_steps": step_at_end,
                "step": step_at_end,
                "segment_idx": segment_idx,
                "segment_size": segment_size,
                "cumulative_norm": cumulative_norm,
                "transpile": t_info,
                "execute": exec_info,
                # Wall-time split for rollup (construct / transpile /
                # execute) for this segment.
                "circuit_construction_time_s": construct_t,
                "transpilation_time_s": t_info.get("wall_time", 0.0),
                "metric_transpile_time_s": metric_t,
                "execution_time_s": exec_info.get("wall_time", 0.0),
                "n_circuits": 1,
            }
            # Hardware-cost stats (DEFAULT_METRIC_BASIS decomposition of
            # raw_qc), flat keys the postproc rolls up.  Computed on both
            # branches now so CH always reports honest cx-depth/gate counts.
            if skip_stats is not None and skip_stats.get("available"):
                met["circuit_depth"] = skip_stats["depth"]
                met["gate_counts"] = skip_stats["gate_counts"]
                met["n_qubits"] = skip_stats["num_qubits"]
                met["circuit_metrics_available"] = True
                # Exact only for the segment that produced the cache;
                # other segments reuse it as an estimate (prep differs).
                met["circuit_metrics_from_segment"] = metric_stats_from_segment
                met["circuit_metrics_exact"] = (
                    metric_stats_from_segment == segment_idx
                )
            elif skip_stats is not None:
                met["circuit_metrics_available"] = False
                met["circuit_metrics_reason"] = skip_stats.get("reason")
            elif skip_metric_transpile:
                met["circuit_metrics_available"] = False
                met["circuit_metrics_reason"] = (
                    "skipped on held Session; see dry-run pre-flight"
                )
            snapshots[step_at_end] = (phi_out, met)

    total_elapsed = time.time() - t_total_start
    print(
        f"[_run_shots_measure_reprepare] all {n_segments} segments "
        f"done in {total_elapsed:.1f}s",
        file=sys.stderr, flush=True,
    )
    # Stamp the end-to-end wall clock on the last snapshot's metrics so the
    # runtime panel can derive 'other classical' = wall - transpile - exec
    # (MPS prep, encode/decode, post-selection, Python overhead).  Per-segment
    # transpile/execute are already recorded; without this anchor the panel
    # can only show those two and the classical remainder collapses to zero.
    if snap_steps:
        snapshots[snap_steps[-1]][1]["method_wall_time_s"] = total_elapsed
        # Authoritative every-segment list (all QPU jobs), carried on the
        # final snapshot so callers get complete job provenance without a
        # return-signature change.  seg_audit holds independent dicts (not
        # the snapshot met objects), so no self-referential cycle is created.
        snapshots[snap_steps[-1]][1]["segments_full"] = seg_audit
    return [snapshots[s] for s in snap_steps]


# ── P5: Full u0 -> u(t) simulation pipeline ─────────────────────────


def _run_shots_batch(
    psi0: np.ndarray,
    q: int,
    nu: float,
    dt: float,
    snap_steps: list[int],
    L_box: float,
    phi_norm: float,
    bc: str,
    shots: int,
    bond_dim: int | None = None,
    backend: Any = None,
    backend_type: str = "sim",
    backend_name: str | None = None,
    optimization_level: int = 1,
    seed: int | None = None,
    use_mps_prep: bool = True,
    metric_transpile_timeout: float = 60.0,
    max_total_qubits: int | None = None,
) -> list[tuple[np.ndarray, dict[str, Any]]]:
    """Batch shots readout per spec §7 (P-C optimised).

    Builds one circuit per snap_step, transpiles all in a single
    ``transpile([...], backend)`` call, then runs each.  One
    AerSimulator instantiation and one transpile invocation total.

    State preparation uses the Ran 2020 MPS-to-circuit pipeline
    (``lib_mps.classical_to_mps`` + ``mps_to_circuit``), matching
    Gopalakrishnan Meena AIAA-2026 Eqs. 5-6 plus Ran 2020 (Ref [27]).
    ``bond_dim=None`` means full-rank (no truncation); finite values
    truncate the MPS.

    use_mps_prep: True (default) uses the Ran 2020 MPS-to-circuit
        path; False falls back to QuantumCircuit.initialize for
        comparison with the dense O(2^q) state-prep baseline.

    Returns list of (phi_reconstructed, metrics) per snap_step.
    """
    N = 1 << q

    # --- State prep (built once; shared across snap_steps) ------------
    psi_norm, _ = normalize_state(psi0)
    if use_mps_prep:
        tensors = classical_to_mps(
            psi_norm, bond_dim=bond_dim, canonical="right",
        )
        prep_qc = mps_to_circuit(tensors)
        n_bond = prep_qc.num_qubits - q
    else:
        prep_qc = QuantumCircuit(q, name="initialize_prep")
        prep_qc.initialize(psi_norm.tolist(), range(q))
        n_bond = 0

    # --- Build all circuits -------------------------------------------
    print(
        f"[_run_shots_batch] building {len(snap_steps)} circuit(s) "
        f"for snap_steps={list(snap_steps)} shots={shots}",
        file=sys.stderr, flush=True,
    )
    # Bond-aware heat-ancilla budget (see build_segment_circuit).  Here the
    # "segment" is the whole s-step circuit, so the cap governs how many of
    # the s steps defer vs reset-and-reuse.
    heat_max_qubits = None
    if max_total_qubits is not None:
        heat_budget = max_total_qubits - n_bond
        if heat_budget < q + 1:
            raise ValueError(
                f"max_total_qubits={max_total_qubits} too small: q={q} + "
                f"n_bond={n_bond} leaves no room for a heat ancilla "
                f"(need >= {q + n_bond + 1})"
            )
        heat_max_qubits = heat_budget
    t_build_start = time.time()
    raw_circs: list[QuantumCircuit] = []
    construct_times: list[float] = []
    total_q = None
    for s in snap_steps:
        t_cc = time.time()
        full_qc = heat_qft_full_circuit(
            q, nu, dt * s, s, L_box, bc=bc,
            max_qubits=heat_max_qubits,
        )
        n_heat_anc = full_qc.num_qubits - q
        total_q = q + n_bond + n_heat_anc
        heat_qubit_map = list(range(q)) + list(
            range(q + n_bond, q + n_bond + n_heat_anc),
        )
        init_qc = QuantumCircuit(total_q)
        init_qc.compose(
            prep_qc, qubits=list(range(q + n_bond)), inplace=True,
        )
        init_qc.compose(full_qc, qubits=heat_qubit_map, inplace=True)
        raw_circs.append(init_qc)
        construct_times.append(time.time() - t_cc)
    print(
        f"[_run_shots_batch] built {len(raw_circs)} circuit(s) in "
        f"{time.time() - t_build_start:.1f}s "
        f"(total qubits={total_q})",
        file=sys.stderr, flush=True,
    )

    # --- Hardware-async: fire-and-record, return placeholders ----------
    if backend_type == "hardware":
        if backend_name is None:
            raise ValueError("backend_name is required for hardware runs")
        job_id = submit_job(
            raw_circs,
            backend_name=backend_name,
            shots=shots,
            optimization_level=optimization_level,
        )
        print(
            f"[cole_hopf_circuit] hardware job submitted: {job_id}",
            file=sys.stderr, flush=True,
        )
        results: list[tuple[np.ndarray, dict[str, Any]]] = []
        for s in snap_steps:
            phi_hat = np.full(N, np.nan)
            met: dict[str, Any] = {
                "shots": shots,
                "n_kept": 0,
                "p_success": float("nan"),
                "n_steps": s,
                "step": s,
                "job_id": job_id,
                "backend_name": backend_name,
                "submitted_at": datetime.datetime.now(
                    datetime.timezone.utc,
                ).isoformat(),
                "shots_per_circuit": shots,
            }
            results.append((phi_hat, met))
        return results

    # --- Single batched transpile of all circuits (P-C: F10-5) --------
    # Pattern 1 from F10-REVIEW-PATCH §P-C: build the cumulative family
    # {circuit_1 .. circuit_n_steps} once, transpile in one
    # transpile([...], backend) call, then run each.  One AerSimulator()
    # and one transpile() per run_cole_hopf_circuit_simulation invocation.
    print(
        f"[_run_shots_batch] batched transpile of "
        f"{len(raw_circs)} circuit(s) "
        f"(opt_level={optimization_level}) ...",
        file=sys.stderr, flush=True,
    )
    t_tr = time.time()
    kwargs: dict[str, Any] = {
        "backend": backend,
        "optimization_level": optimization_level,
    }
    if seed is not None:
        kwargs["seed_transpiler"] = seed
    qc_t_out = _qiskit_transpile(list(raw_circs), **kwargs)
    # qiskit.transpile returns a list when given a list, but coerce
    # defensively in case of a single-element edge case.
    if isinstance(qc_t_out, QuantumCircuit):
        qc_t_list = [qc_t_out]
    else:
        qc_t_list = list(qc_t_out)
    transpile_wall = time.time() - t_tr
    t_info_list = [
        {
            "wall_time": transpile_wall,
            "optimization_level": optimization_level,
            "before": get_circuit_info(raw),
            "after": get_circuit_info(qc_t),
        }
        for raw, qc_t in zip(raw_circs, qc_t_list)
    ]
    print(
        f"[_run_shots_batch] batched transpile done in "
        f"{transpile_wall:.1f}s",
        file=sys.stderr, flush=True,
    )

    # --- Execute each transpiled circuit -------------------------------
    # The batched transpile is a single call for all circuits, so split
    # its wall time evenly across them for an honest per-iteration rollup
    # (summing the nested transpile.wall_time would N-times overcount).
    transpile_per = transpile_wall / max(1, len(raw_circs))
    results: list[tuple[np.ndarray, dict[str, Any]]] = []
    for idx, (raw_qc, qc_t, t_info, s) in enumerate(zip(
        raw_circs, qc_t_list, t_info_list, snap_steps,
    )):
        print(
            f"[_run_shots_batch] circuit {idx + 1}/{len(raw_circs)} "
            f"snap_step={s} depth={qc_t.depth()} size={qc_t.size()}; "
            f"running shots={shots} ...",
            file=sys.stderr, flush=True,
        )
        t_ex = time.time()
        counts, exec_info = execute_circuit_counts(
            qc_t, backend, shots=shots, seed=seed,
        )
        print(
            f"[_run_shots_batch] execute done in "
            f"{time.time() - t_ex:.1f}s; "
            f"unique outcomes={len(counts)}",
            file=sys.stderr, flush=True,
        )

        # Post-select and reconstruct
        n_kept, data_counts = post_select_counts(counts, q)
        p_success = n_kept / shots if shots > 0 else 0.0
        if p_success < 0.3:
            print(
                f"[cole_hopf_circuit] WARNING: P_success="
                f"{p_success:.3f} < 0.3; increase shots or "
                f"reduce n_steps",
                file=sys.stderr, flush=True,
            )
        phi_hat = reconstruct_phi_from_counts(
            data_counts, n_kept, N, phi_norm, p_success,
        )

        met: dict[str, Any] = {
            "shots": shots,
            "n_kept": n_kept,
            "p_success": p_success,
            "n_steps": s,
            "step": s,
            "transpile": t_info,
            "execute": exec_info,
            # Wall-time split for rollup.  transpile is the batched call
            # amortized per circuit (see transpile_per above).
            "circuit_construction_time_s": construct_times[idx],
            "transpilation_time_s": transpile_per,
            "execution_time_s": exec_info.get("wall_time", 0.0),
            "n_circuits": 1,
        }
        # Hardware-cost stats: decompose raw_qc to DEFAULT_METRIC_BASIS for
        # honest cx-depth/gate counts comparable to QLBM, on both branches.
        # Each batched circuit covers a different snap_step (different
        # depth), so stats are computed per-circuit, not cached across them.
        # Out-of-process for crash-safety on the LCU SELECT synthesis.
        _t_m = time.time()
        bs = safe_circuit_stats_in_basis(
            raw_qc, DEFAULT_METRIC_BASIS,
            optimization_level=optimization_level,
            seed_transpiler=seed,
            timeout=metric_transpile_timeout, try_decompose=False,
        )
        met["metric_transpile_time_s"] = time.time() - _t_m
        if bs.get("available"):
            met["circuit_depth"] = bs["depth"]
            met["gate_counts"] = bs["gate_counts"]
            met["n_qubits"] = bs["num_qubits"]
            met["circuit_metrics_available"] = True
        else:
            met["circuit_metrics_available"] = False
            met["circuit_metrics_reason"] = bs.get("reason")
        results.append((phi_hat, met))

    return results


def forward_ch_psi0(
    u0: np.ndarray,
    dx: float,
    nu: float,
) -> tuple[np.ndarray, float, bool]:
    """Forward Cole-Hopf transform u0 -> unit-norm psi0.

    Single source of truth for the transform + centering policy, shared
    by run_cole_hopf_circuit_simulation and the F12 hardware dry-run so
    the dry-run's segment 0 is identical to what actually executes.

    Returns (psi0, phi_norm, use_centering).
    """
    use_centering = _should_center(u0, dx, nu)
    if use_centering:
        log_phi, e_mid = cole_hopf_forward_centered(u0, dx, nu)
        psi0 = log_phi_to_normalized_psi(log_phi)
        phi_norm = 1.0
        print(
            f"[cole_hopf_circuit] centered exponent "
            f"(e_mid={e_mid:.2f}, nu={nu:.1e})",
            file=sys.stderr, flush=True,
        )
    else:
        phi0 = cole_hopf_forward(u0, dx, nu)
        phi_norm = float(np.linalg.norm(phi0))
        if phi_norm < 1e-15:
            raise ValueError(
                "phi0 norm near zero; check IC and nu"
            )
        psi0 = phi0 / phi_norm
    return psi0, phi_norm, use_centering


def dry_run_segment_transpile(
    u0: np.ndarray,
    x: np.ndarray,
    nu: float,
    dt: float,
    n_steps: int,
    segment_size: int,
    bc: str = "periodic",
    bond_dim: int | None = None,
    use_mps_prep: bool = True,
    max_total_qubits: int | None = None,
    backend: Any = None,
    optimization_level: int = 1,
    seed: int | None = None,
    initial_layout: list[int] | None = None,
) -> dict[str, Any]:
    """Build segment 0 and transpile it against `backend`; no shots.

    F12 pre-flight: reuses forward_ch_psi0 + build_segment_circuit (the
    SAME construction the measure-reprepare loop runs), so the reported
    CX/depth is the authoritative per-segment hardware cost.  Under a
    held Session the in-loop metric transpile is skipped in favour of
    this report.
    """
    dx = float(x[1] - x[0])
    N = len(u0)
    q = int(np.log2(N))
    L_box = float(N * dx)
    phi_bc = "neumann" if bc == "dirichlet" else bc

    psi0, _, _ = forward_ch_psi0(u0, dx, nu)
    psi_current, _ = normalize_state(psi0)

    raw_qc, total_q, n_bond, n_heat_anc = build_segment_circuit(
        psi_current, q, nu, dt, segment_size, L_box, phi_bc,
        bond_dim=bond_dim, use_mps_prep=use_mps_prep,
        max_total_qubits=max_total_qubits,
    )
    qc_t, info = transpile_circuit(
        raw_qc, backend,
        optimization_level=optimization_level,
        seed_transpiler=seed,
        initial_layout=initial_layout,
    )
    ops = qc_t.count_ops()
    two_q = ops.get("cx", 0) + ops.get("ecr", 0) + ops.get("cz", 0)
    return {
        "segment_qubits": total_q,
        "n_bond": n_bond,
        "n_heat_anc": n_heat_anc,
        "raw_depth": raw_qc.depth(),
        "transpiled_depth": qc_t.depth(),
        "transpiled_2q_gates": int(two_q),
        "transpiled_ops": {k: int(v) for k, v in ops.items()},
        "n_segments": n_steps // segment_size,
        "transpile_info": info,
    }


def run_cole_hopf_circuit_simulation(
    u0: np.ndarray,
    x: np.ndarray,
    nu: float,
    dt: float,
    n_steps: int,
    bc: str = "periodic",
    snapshot_interval: int = 1,
    shots: int = 0,
    bond_dim: int | None = None,
    backend: Any = None,
    backend_type: str = "sim",
    backend_name: str | None = None,
    optimization_level: int = 1,
    seed: int | None = None,
    source_fn=None,
    evolution_mode: str = "single",
    segment_size: int = 10,
    phi_modes: int = 0,
    use_mps_prep: bool = True,
    metric_transpile_timeout: float = 60.0,
    allow_hardware: bool = False,
    session: Any = None,
    sampler_options: dict | None = None,
    initial_layout: list[int] | None = None,
    max_total_qubits: int | None = None,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Full Cole-Hopf circuit Burgers solver.

    1. Forward CH transform: u0 -> phi0 -> psi0 (unit norm)
    2. Circuit evolution of the heat equation on psi
    3. Inverse CH transform: psi(t) -> u(t) at snapshot times

    shots=0: statevector simulation (exact, with per-step snapshots).
    shots>0: full circuit with measurements, post-select on ancilla
             history = all |0>, reconstruct phi from counts (§7).
             Only final-time output (no intermediate snapshots).

    bond_dim: MPS truncation bond dimension; None = full rank.  Used
        only when use_mps_prep=True.
    use_mps_prep: True (default) prepares the initial state via the
        Ran 2020 MPS-to-circuit pipeline (O(q*chi^2) gates).  False
        falls back to QuantumCircuit.initialize (O(2^q) gates) so
        the two prep paths can be A/B compared.

    Returns (solutions, metrics) matching the run_simulation interface.
    """
    N = len(u0)
    q = int(np.log2(N))
    dx = x[1] - x[0]
    L_box = float(N * dx)

    # §8: Dirichlet u=0 maps to Neumann dphi/dx=0 on phi.
    phi_bc = "neumann" if bc == "dirichlet" else bc
    # qft-diagonal + neumann uses the DCT-II propagator (see
    # heat_dct_step_circuit); periodic uses QFT.  No NotImplementedError.
    if phi_bc not in ("periodic", "neumann"):
        raise NotImplementedError(
            f"qft-diagonal supports periodic and dirichlet (Neumann-on-phi);"
            f" got bc={bc!r}"
        )

    # Source guard: V is position-diagonal, L is Fourier-diagonal; they
    # don't share an eigenbasis.  qft-diagonal has no source-forcing path.
    if source_fn is not None:
        raise NotImplementedError(
            "qft-diagonal propagator does not support source forcing; "
            "source_fn must be None."
        )

    # Forward CH transform (reuse P1 centering logic)
    psi0, phi_norm, use_centering = forward_ch_psi0(u0, dx, nu)

    print(
        f"[cole_hopf_circuit] propagator=qft-diagonal q={q} "
        f"nu={nu:.1e} dt={dt:.4e} "
        f"n_steps={n_steps} shots={shots}",
        file=sys.stderr, flush=True,
    )

    if shots > 0:
        nan_fill = np.full_like(u0, np.nan)
        solutions: list[np.ndarray] = (
            [u0.copy()] + [nan_fill] * n_steps
        )
        all_metrics: list[dict[str, Any]] = []
        snap_steps = list(range(
            snapshot_interval, n_steps + 1, snapshot_interval,
        ))
        if n_steps not in snap_steps:
            snap_steps.append(n_steps)

        if evolution_mode == "measure_reprepare":
            # Validate alignment
            if n_steps % segment_size != 0:
                raise ValueError(
                    f"segmented mode requires n_steps ({n_steps}) "
                    f"divisible by segment_size ({segment_size})"
                )
            bad = [s for s in snap_steps
                   if s % segment_size != 0]
            if bad:
                raise ValueError(
                    f"segmented mode: snap_steps {bad} not "
                    f"aligned to segment_size={segment_size}; "
                    f"set --save-every to a multiple of "
                    f"--segment-size"
                )
            batch_results = _run_shots_measure_reprepare(
                psi0, q, nu, dt, snap_steps, L_box,
                phi_norm,
                bc=phi_bc,
                shots=shots, segment_size=segment_size,
                bond_dim=bond_dim,
                backend=backend, backend_type=backend_type,
                backend_name=backend_name,
                optimization_level=optimization_level,
                seed=seed,
                use_mps_prep=use_mps_prep,
                metric_transpile_timeout=metric_transpile_timeout,
                allow_hardware=allow_hardware,
                session=session,
                sampler_options=sampler_options,
                initial_layout=initial_layout,
                max_total_qubits=max_total_qubits,
            )
        else:
            batch_results = _run_shots_batch(
                psi0, q, nu, dt, snap_steps, L_box,
                phi_norm,
                bc=phi_bc,
                shots=shots,
                bond_dim=bond_dim,
                backend=backend, backend_type=backend_type,
                backend_name=backend_name,
                optimization_level=optimization_level,
                seed=seed,
                use_mps_prep=use_mps_prep,
                metric_transpile_timeout=metric_transpile_timeout,
                max_total_qubits=max_total_qubits,
            )

        for (phi_hat, met), s in zip(batch_results, snap_steps):
            if phi_modes > 0:
                phi_hat = fourier_low_pass_phi(
                    phi_hat, phi_modes,
                )
            u_snap = cole_hopf_inverse(phi_hat, dx, nu, bc=bc)
            solutions[s] = u_snap
            all_metrics.append(met)
            if s % max(1, n_steps // 5) == 0 or s == n_steps:
                print(
                    f"[cole_hopf_circuit] shots step "
                    f"{s}/{n_steps} "
                    f"P_success={met['p_success']:.3f}",
                    file=sys.stderr, flush=True,
                )
        return solutions, all_metrics

    # Statevector path: per-step projection with snapshots
    psi_snaps, circ_metrics = run_cole_hopf_circuit_sv(
        psi0, q, nu, dt, n_steps, L_box,
        bc=phi_bc,
        snapshot_interval=snapshot_interval,
        use_mps_prep=use_mps_prep,
        bond_dim=bond_dim,
    )

    # Build snapshot step indices (mirrors run_cole_hopf_circuit_sv)
    snap_steps: list[int] = []
    for s in range(1, n_steps + 1):
        if s % snapshot_interval == 0 or s == n_steps:
            snap_steps.append(s)

    # Full-length solutions list indexed by step number (NaN-fill
    # for non-snapshot steps, matching the shots-path convention).
    nan_fill = np.full_like(u0, np.nan)
    solutions: list[np.ndarray] = (
        [u0.copy()] + [nan_fill] * n_steps
    )
    for psi_snap, s in zip(psi_snaps, snap_steps):
        if use_centering:
            u_snap = cole_hopf_inverse(psi_snap, dx, nu, bc=bc)
        else:
            phi_snap = psi_snap * phi_norm
            u_snap = cole_hopf_inverse(phi_snap, dx, nu, bc=bc)
        solutions[s] = u_snap

    return solutions, circ_metrics
