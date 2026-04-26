"""Quantum circuit Cole-Hopf Burgers solver (F10 Phase B).

Two propagator variants for the heat equation exp(nu*L*dt):
1. qft-diagonal: QFT + diagonal phase-damping + QFT^-1 (default)
2. dense-block: exact eigendecomposition + block-encoding (q <= 5)

Both use a single ancilla qubit for non-unitary block encoding.
Post-selection on ancilla |0> recovers the contractive propagator.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit.library import QFTGate, RYGate, UnitaryGate
from scipy.linalg import expm

from burgers_cole_hopf import build_laplacian_dense
from burgers_encoding import permute_operator


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


# ── P2: Polynomial theta fit ────────────────────────────────────────


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


# ── P2: Conditional polynomial-Ry circuit ───────────────────────────


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
    qc = QuantumCircuit(n_total, name="poly_ry")

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


# ── P3: QFT-diagonal propagator step ────────────────────────────────


def heat_qft_step_circuit(
    q: int,
    nu: float,
    dt: float,
    L_box: float,
    bc: str = "periodic",
) -> QuantumCircuit:
    """One Trotter layer: QFT + conditional-Ry(theta) + QFT^-1 + meas.

    Returns a circuit on q+1 qubits (data[0..q-1] + ancilla[q])
    with one classical bit for the ancilla measurement.

    bc must be 'periodic'; raises NotImplementedError for 'dirichlet'.
    """
    if bc != "periodic":
        raise NotImplementedError(
            "qft-diagonal requires periodic BC; Dirichlet needs a "
            "DST-based propagator (see spec section 13 future work). "
            "Use --propagator dense-block for Dirichlet."
        )

    theta = compute_theta_exact(nu, dt, q, L_box)

    data = list(range(q))
    anc = q

    qc = QuantumCircuit(q + 1, 1, name="heat_qft_step")

    # QFT on data register
    qc.append(QFTGate(q), data)

    # Conditional rotation on ancilla (exact Mobius, no polynomial fit)
    cond_qc = build_conditional_ry(data, anc, theta)
    qc.compose(cond_qc, inplace=True)

    # Inverse QFT
    qc.append(QFTGate(q).inverse(), data)

    # Measure ancilla -> classical bit 0, then reset
    qc.measure(anc, 0)
    qc.reset(anc)

    return qc


def heat_qft_full_circuit(
    q: int,
    nu: float,
    T: float,
    N_steps: int,
    L_box: float,
    bc: str = "periodic",
) -> QuantumCircuit:
    """Full evolution circuit: N_steps QFT-diagonal Trotter layers.

    Returns circuit on q+1 qubits with N_steps classical bits
    (one per ancilla measurement) plus q bits for final data readout.
    """
    if bc != "periodic":
        raise NotImplementedError(
            "qft-diagonal requires periodic BC; "
            "see heat_qft_step_circuit."
        )

    dt = T / N_steps

    data = list(range(q))
    anc = q

    anc_cr = ClassicalRegister(N_steps, "anc_hist")
    data_cr = ClassicalRegister(q, "data")
    qc = QuantumCircuit(q + 1, name="heat_qft_full")
    qc.add_register(anc_cr)
    qc.add_register(data_cr)

    theta = compute_theta_exact(nu, dt, q, L_box)

    qft = QFTGate(q)
    qft_inv = QFTGate(q).inverse()
    cond_qc = build_conditional_ry(data, anc, theta)

    for step in range(N_steps):
        qc.append(qft, data)
        qc.compose(cond_qc, inplace=True)
        qc.append(qft_inv, data)
        qc.measure(anc, anc_cr[step])
        if step < N_steps - 1:
            qc.reset(anc)

    # Final data measurement
    for i in range(q):
        qc.measure(data[i], data_cr[i])

    return qc


# ── P4: Dense-block propagator step ────────────────────────────────


def heat_dense_block_step_circuit(
    q: int,
    nu: float,
    dt: float,
    L_box: float,
    bc: str = "periodic",
    encoding: str = "binary",
) -> QuantumCircuit:
    """One propagator layer via dense eigendecomposition + block encoding.

    Builds exp(nu*L*dt), eigendecomposes, and block-encodes via
    V^H + conditional Ry + V with ancilla post-selection.
    Exact per step (no Trotter error). Tractable for q <= 5.
    """
    N = 1 << q
    dx = L_box / N
    L_dense = build_laplacian_dense(N, dx, bc=bc)
    L_dense = permute_operator(L_dense, q, encoding)

    M = nu * L_dense * dt
    data = list(range(q))
    anc = q
    qc = QuantumCircuit(q + 1, 1, name="heat_dense_step")

    # Block encoding of exp(M) via eigendecomposition + ancilla.
    # P = exp(M) = V D V^H with D = diag(d_k), d_k in (0,1].
    # Block encode: V^H (to eigenbasis), conditional Ry(theta_k)
    # on ancilla, V (back).  Post-select ancilla |0> gives P/alpha.
    P_mat = expm(M)
    s_max = float(np.linalg.svd(P_mat, compute_uv=False)[0])
    eigvals, eigvecs = np.linalg.eigh(P_mat)
    theta_vals = np.arccos(np.clip(eigvals / s_max, -1.0, 1.0))

    # V^H maps comp -> eigenbasis; V maps eigen -> comp
    V_gate = UnitaryGate(eigvecs, label="V")
    V_dag_gate = UnitaryGate(eigvecs.conj().T, label="V_dag")

    qc.append(V_dag_gate, data)  # to eigenbasis
    for k in range(N):
        if abs(theta_vals[k]) < 1e-15:
            continue
        ctrl_state = format(k, f"0{q}b")
        gate = RYGate(2.0 * theta_vals[k]).control(
            q, ctrl_state=ctrl_state,
        )
        qc.append(gate, data + [anc])
    qc.append(V_gate, data)  # back to computational basis

    qc.measure(anc, 0)
    qc.reset(anc)

    return qc


def heat_dense_block_full_circuit(
    q: int,
    nu: float,
    T: float,
    N_steps: int,
    L_box: float,
    bc: str = "periodic",
    encoding: str = "binary",
) -> QuantumCircuit:
    """Full evolution: N_steps dense-block layers."""
    dt = T / N_steps

    data = list(range(q))
    anc = q

    anc_cr = ClassicalRegister(N_steps, "anc_hist")
    data_cr = ClassicalRegister(q, "data")
    qc = QuantumCircuit(q + 1, name="heat_dense_full")
    qc.add_register(anc_cr)
    qc.add_register(data_cr)

    step_qc = heat_dense_block_step_circuit(
        q, nu, dt, L_box, bc=bc, encoding=encoding,
    )

    for step_idx in range(N_steps):
        # Inline the step circuit but remap ancilla measurement
        # to the correct classical bit
        qc_layer = step_qc.copy()
        qc_layer.remove_final_measurements()
        qc.compose(qc_layer, inplace=True)
        qc.measure(anc, anc_cr[step_idx])
        if step_idx < N_steps - 1:
            qc.reset(anc)

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
    propagator: str,
    encoding: str = "binary",
) -> QuantumCircuit:
    """Build a measurement-free step circuit for SV simulation."""
    if propagator == "qft-diagonal":
        theta = compute_theta_exact(nu, dt, q, L_box)
        data = list(range(q))
        anc = q
        qc = QuantumCircuit(q + 1)
        qc.append(QFTGate(q), data)
        cond_qc = build_conditional_ry(data, anc, theta)
        qc.compose(cond_qc, inplace=True)
        qc.append(QFTGate(q).inverse(), data)
        return qc
    elif propagator == "dense-block":
        N = 1 << q
        dx = L_box / N
        L_dense = build_laplacian_dense(N, dx, bc=bc)
        L_dense = permute_operator(L_dense, q, encoding)
        M = nu * L_dense * dt
        P_mat = expm(M)
        s_max = float(np.linalg.svd(P_mat, compute_uv=False)[0])
        eigvals, eigvecs = np.linalg.eigh(P_mat)
        theta_vals = np.arccos(
            np.clip(eigvals / s_max, -1.0, 1.0),
        )
        data = list(range(q))
        anc = q
        qc = QuantumCircuit(q + 1)
        qc.append(
            UnitaryGate(eigvecs.conj().T, label="Vd"), data,
        )
        for k in range(N):
            if abs(theta_vals[k]) < 1e-15:
                continue
            ctrl_state = format(k, f"0{q}b")
            gate = RYGate(2.0 * theta_vals[k]).control(
                q, ctrl_state=ctrl_state,
            )
            qc.append(gate, data + [anc])
        qc.append(UnitaryGate(eigvecs, label="V"), data)
        return qc
    else:
        raise ValueError(f"Unknown propagator: {propagator}")


def run_cole_hopf_circuit_sv(
    psi0: np.ndarray,
    q: int,
    nu: float,
    dt: float,
    n_steps: int,
    L_box: float,
    bc: str = "periodic",
    propagator: str = "qft-diagonal",
    snapshot_interval: int = 1,
    use_mps_prep: bool = True,
    bond_dim: int | None = None,
    encoding: str = "binary",
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
    from qiskit.quantum_info import Statevector

    N = 1 << q
    step_qc = _build_step_sv(
        q, nu, dt, L_box, bc, propagator,
        encoding=encoding,
    )

    sv = np.zeros(2 * N, dtype=complex)
    if use_mps_prep:
        from burgers_mps import (
            classical_to_mps, mps_to_circuit, normalize_state,
        )

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
        sv = Statevector(sv).evolve(step_qc).data
        p0 = float(np.sum(np.abs(sv[:N]) ** 2))
        p_success_total *= p0
        sv_proj = np.zeros_like(sv)
        if p0 > 1e-30:
            sv_proj[:N] = sv[:N] / np.sqrt(p0)
        sv = sv_proj

        step_metrics: dict[str, Any] = {
            "step": step + 1,
            "p_success_step": p0,
            "p_success_total": p_success_total,
        }
        metrics.append(step_metrics)

        if (step + 1) % snapshot_interval == 0 or step == n_steps - 1:
            snapshots.append(np.real(sv[:N]).copy())

    return snapshots, metrics


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
    propagator: str,
    shots: int,
    bond_dim: int | None = None,
    encoding: str = "binary",
    backend: Any = None,
    backend_type: str = "sim",
    backend_name: str | None = None,
    optimization_level: int = 1,
    seed: int | None = None,
) -> list[tuple[np.ndarray, dict[str, Any]]]:
    """Batch shots readout per spec §7 (P-C optimised).

    Builds one circuit per snap_step, transpiles all in a single
    ``transpile([...], backend)`` call, then runs each.  One
    AerSimulator instantiation and one transpile invocation total.

    State preparation uses the Ran 2020 MPS-to-circuit pipeline
    (``burgers_mps.classical_to_mps`` + ``mps_to_circuit``), matching
    Murali/Meena AIAA-2026 Eq. 5-6 + Ref [27].  ``bond_dim=None``
    means full-rank (no truncation); finite values truncate the MPS.

    Returns list of (phi_reconstructed, metrics) per snap_step.
    """
    from q8020_cfd_qutil.circuit import (
        transpile_circuit,
        execute_circuit_counts,
    )

    from burgers_mps import (
        classical_to_mps, mps_to_circuit, normalize_state,
    )

    N = 1 << q

    # --- MPS state prep (built once; shared across snap_steps) --------
    psi_norm, _ = normalize_state(psi0)
    tensors = classical_to_mps(
        psi_norm, bond_dim=bond_dim, canonical="right",
    )
    prep_qc = mps_to_circuit(tensors)
    n_bond = prep_qc.num_qubits - q

    # Register layout: data [0..q-1] | bond [q..q+n_bond-1] | anc [q+n_bond]
    total_q = q + n_bond + 1
    heat_qubit_map = list(range(q)) + [q + n_bond]

    # --- Build all circuits -------------------------------------------
    raw_circs: list[QuantumCircuit] = []
    for s in snap_steps:
        if propagator == "qft-diagonal":
            full_qc = heat_qft_full_circuit(
                q, nu, dt * s, s, L_box, bc=bc,
            )
        else:
            full_qc = heat_dense_block_full_circuit(
                q, nu, dt * s, s, L_box, bc=bc,
                encoding=encoding,
            )
        init_qc = QuantumCircuit(total_q)
        init_qc.compose(
            prep_qc, qubits=list(range(q + n_bond)), inplace=True,
        )
        raw_circs.append(
            init_qc.compose(full_qc, qubits=heat_qubit_map),
        )

    # --- Hardware-async: fire-and-record, return placeholders ----------
    if backend_type == "hardware":
        from q8020_cfd_qutil.job import submit_job
        import datetime
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

    # --- Transpile + execute each circuit via qutil --------------------
    results: list[tuple[np.ndarray, dict[str, Any]]] = []
    for raw_qc, s in zip(raw_circs, snap_steps):
        qc_t, t_info = transpile_circuit(
            raw_qc, backend,
            optimization_level=optimization_level,
            seed_transpiler=seed,
        )
        counts, exec_info = execute_circuit_counts(
            qc_t, backend, shots=shots, seed=seed,
        )

        # Post-select: slice by position (no .split())
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

        p_success = n_kept / shots if shots > 0 else 0.0
        if p_success < 0.3:
            print(
                f"[cole_hopf_circuit] WARNING: P_success="
                f"{p_success:.3f} < 0.3; increase shots or "
                f"reduce n_steps",
                file=sys.stderr, flush=True,
            )

        phi_hat = np.zeros(N)
        if n_kept > 0:
            for data_bits_k, cnt in data_counts.items():
                idx = int(data_bits_k, 2)
                phi_hat[idx] = np.sqrt(cnt / n_kept)
            phi_hat *= np.sqrt(p_success) * phi_norm

        met: dict[str, Any] = {
            "shots": shots,
            "n_kept": n_kept,
            "p_success": p_success,
            "n_steps": s,
            "step": s,
            "transpile": t_info,
            "execute": exec_info,
        }
        results.append((phi_hat, met))

    return results


def run_cole_hopf_circuit_simulation(
    u0: np.ndarray,
    x: np.ndarray,
    nu: float,
    dt: float,
    n_steps: int,
    bc: str = "periodic",
    propagator: str = "qft-diagonal",
    snapshot_interval: int = 1,
    shots: int = 0,
    bond_dim: int | None = None,
    encoding: str = "binary",
    backend: Any = None,
    backend_type: str = "sim",
    backend_name: str | None = None,
    optimization_level: int = 1,
    seed: int | None = None,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Full Cole-Hopf circuit Burgers solver.

    1. Forward CH transform: u0 -> phi0 -> psi0 (unit norm)
    2. Circuit evolution of the heat equation on psi
    3. Inverse CH transform: psi(t) -> u(t) at snapshot times

    shots=0: statevector simulation (exact, with per-step snapshots).
    shots>0: full circuit with measurements, post-select on ancilla
             history = all |0>, reconstruct phi from counts (§7).
             Only final-time output (no intermediate snapshots).

    Returns (solutions, metrics) matching the run_simulation interface.
    """
    from burgers_cole_hopf import (
        cole_hopf_forward,
        cole_hopf_forward_centered,
        cole_hopf_inverse,
        log_phi_to_normalized_psi,
        _should_center,
    )

    N = len(u0)
    q = int(np.log2(N))
    dx = x[1] - x[0]
    L_box = float(N * dx)

    # §8: Dirichlet u=0 maps to Neumann dphi/dx=0 on phi.
    phi_bc = "neumann" if bc == "dirichlet" else bc
    if phi_bc != "periodic" and propagator == "qft-diagonal":
        raise NotImplementedError(
            "qft-diagonal requires periodic BC; "
            "use --propagator dense-block for --bc dirichlet"
        )

    # Encoding guard: QFT diagonalises L only in binary basis.
    if encoding != "binary" and propagator == "qft-diagonal":
        raise NotImplementedError(
            f"qft-diagonal propagator requires encoding=binary; "
            f"got encoding={encoding!r}. Use "
            f"--propagator dense-block with --encoding {encoding}."
        )

    # Forward CH transform (reuse P1 centering logic)
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

    # Permute psi0 from grid order to encoded order for circuit prep.
    from burgers_encoding import permute_to_encoding
    psi0 = permute_to_encoding(psi0, q, encoding)

    print(
        f"[cole_hopf_circuit] propagator={propagator} q={q} "
        f"encoding={encoding} nu={nu:.1e} dt={dt:.4e} "
        f"n_steps={n_steps} shots={shots}",
        file=sys.stderr, flush=True,
    )

    if shots > 0:
        # Shots path: batch build + single transpile (P-C).
        nan_fill = np.full_like(u0, np.nan)
        solutions: list[np.ndarray] = [u0.copy()] + [nan_fill] * n_steps
        all_metrics: list[dict[str, Any]] = []
        snap_steps = list(range(
            snapshot_interval, n_steps + 1, snapshot_interval,
        ))
        if n_steps not in snap_steps:
            snap_steps.append(n_steps)
        batch_results = _run_shots_batch(
            psi0, q, nu, dt, snap_steps, L_box, phi_norm,
            bc=phi_bc, propagator=propagator, shots=shots,
            bond_dim=bond_dim, encoding=encoding,
            backend=backend, backend_type=backend_type,
            backend_name=backend_name,
            optimization_level=optimization_level, seed=seed,
        )
        from burgers_encoding import permute_from_encoding
        for (phi_hat, met), s in zip(batch_results, snap_steps):
            phi_hat = permute_from_encoding(
                phi_hat, q, encoding,
            )
            u_snap = cole_hopf_inverse(phi_hat, dx, nu)
            solutions[s] = u_snap
            all_metrics.append(met)
            if s % max(1, n_steps // 5) == 0 or s == n_steps:
                print(
                    f"[cole_hopf_circuit] shots step {s}/{n_steps} "
                    f"P_success={met['p_success']:.3f}",
                    file=sys.stderr, flush=True,
                )
        return solutions, all_metrics

    # Statevector path: per-step projection with snapshots
    psi_snaps, circ_metrics = run_cole_hopf_circuit_sv(
        psi0, q, nu, dt, n_steps, L_box,
        bc=phi_bc, propagator=propagator,
        snapshot_interval=snapshot_interval,
        bond_dim=bond_dim, encoding=encoding,
    )

    # Build snapshot step indices (mirrors run_cole_hopf_circuit_sv)
    snap_steps: list[int] = []
    for s in range(1, n_steps + 1):
        if s % snapshot_interval == 0 or s == n_steps:
            snap_steps.append(s)

    # Full-length solutions list indexed by step number (NaN-fill
    # for non-snapshot steps, matching the shots-path convention).
    from burgers_encoding import permute_from_encoding
    nan_fill = np.full_like(u0, np.nan)
    solutions: list[np.ndarray] = (
        [u0.copy()] + [nan_fill] * n_steps
    )
    for psi_snap, s in zip(psi_snaps, snap_steps):
        psi_snap = permute_from_encoding(
            psi_snap, q, encoding,
        )
        if use_centering:
            u_snap = cole_hopf_inverse(psi_snap, dx, nu)
        else:
            phi_snap = psi_snap * phi_norm
            u_snap = cole_hopf_inverse(phi_snap, dx, nu)
        solutions[s] = u_snap

    return solutions, circ_metrics
