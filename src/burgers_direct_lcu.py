"""Direct-u nonlinear Burgers via conservative LCU + measure-reprepare.

SPEC-direct-u-nonlinear-lcu.md.  Solves Burgers in conservative
(divergence) form by evolving the velocity field u directly under the
frozen-coefficient generator A_seg = nu*L - (1/2)*G*diag(u_seg), with
the advection coefficient refreshed every segment from the (measured)
state.  This is a measure-reprepare iterative solve, not a parallel
classical shadow: the only classical data is the field read out of the
quantum state at each segment boundary.

M2: statevector path.  The block-encoded operator exp(A_seg*dt) is
applied as its dense top-left block (statevector semantics, verified
identical to the LCU circuit to ~1e-15).

M3: shots / ancilla post-selection path.  One segment-spanning
block-encoding of exp(A_seg*T_segment) per segment, post-select
ancilla=|0>, reconstruct, renormalise by lambda*sqrt(p_success).
Reports the meaningful scaling indicators (n_qubits, n_ancilla,
lcu_lambda, p_success); honest gate-level depth/cx await the structured
encoder (M4) -- decomposing the flattened dense SELECT would give a
misleading ~4^(q+m) synthesis cost, so it is not force-synthesised.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Callable

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import Statevector

from burgers_lcu import (
    conservative_burgers_lcu_operator,
    conservative_burgers_lcu_step_circuit,
)


def _mps_truncate(
    u: np.ndarray, q: int, bond_dim: int | None,
) -> np.ndarray:
    """Reprepare u through the Ran-2020 MPS pipeline at bond_dim.

    Returns the bond-truncated field at the original physical norm.
    bond_dim=None -> identity (no truncation, full-rank reprep).
    """
    if bond_dim is None:
        return u
    from burgers_mps import (
        classical_to_mps, mps_to_circuit, normalize_state,
    )

    N = 1 << q
    psi_in, norm = normalize_state(u)
    tensors = classical_to_mps(
        psi_in, bond_dim=bond_dim, canonical="right",
    )
    prep_qc = mps_to_circuit(tensors)
    n_bond = prep_qc.num_qubits - q
    sv = (
        Statevector.from_label("0" * (q + n_bond))
        .evolve(prep_qc).data
    )
    psi = np.asarray(sv[:N], dtype=complex).real
    nrm = float(np.linalg.norm(psi))
    if nrm > 1e-15:
        psi = psi / nrm
    return norm * psi


def run_direct_lcu_simulation(
    u0: np.ndarray,
    x: np.ndarray,
    nu: float,
    dt: float,
    n_steps: int,
    bc: str = "periodic",
    source_fn: Callable | None = None,
    shots: int = 0,
    bond_dim: int | None = None,
    taylor_order: int = 4,
    segment_size: int = 1,
    use_mps_prep: bool = True,
    seed: int | None = None,
    backend: Any = None,
    optimization_level: int = 1,
    sign_recovery: str = "none",
    metric_transpile_timeout: float = 60.0,
) -> tuple[list[np.ndarray], list[dict]]:
    """Conservative direct-u LCU evolution with measure-reprepare.

    Frozen coefficient: A_seg is built from u at each segment start and
    held fixed for `segment_size` steps, then refreshed from the
    evolved state.  segment_size=1 refreshes every step (most accurate;
    most measurements).  Per-step truncated-Taylor of exp(A_seg*dt).

    Returns (solutions, metrics) with solutions[i] = u at step i
    (length n_steps + 1).
    """
    if shots > 0:
        return _run_direct_lcu_shots(
            u0, x, nu, dt, n_steps, bc=bc, source_fn=source_fn,
            shots=shots, bond_dim=bond_dim, taylor_order=taylor_order,
            segment_size=segment_size, use_mps_prep=use_mps_prep,
            backend=backend, optimization_level=optimization_level,
            seed=seed, sign_recovery=sign_recovery,
            metric_transpile_timeout=metric_transpile_timeout,
        )

    N = len(u0)
    q = int(round(np.log2(N)))
    if N != (1 << q):
        raise ValueError(f"N={N} must be a power of 2")
    L_box = N * float(x[1] - x[0])

    u = np.asarray(u0, dtype=float).copy()
    solutions = [u.copy()]
    metrics: list[dict[str, Any]] = []

    step = 0
    seg_start = 0
    while seg_start < n_steps:
        k = min(segment_size, n_steps - seg_start)

        # Measure-reprepare boundary: optionally MPS-truncate the
        # current field, then freeze the advection coefficient.
        t0 = time.time()
        if use_mps_prep:
            u = _mps_truncate(u, q, bond_dim)
        u_seg = u.copy()
        op = conservative_burgers_lcu_operator(
            q, nu, dt, L_box, u_seg, taylor_order=taylor_order, bc=bc,
        ).real
        seg_build_t = time.time() - t0

        for s in range(k):
            step += 1
            u = (op @ u).real
            if source_fn is not None:
                u = u + dt * source_fn(x, step * dt)
            solutions.append(u.copy())
            metrics.append({
                "step": step,
                "segment_start": seg_start,
                "frozen_steps": k,
                "u_max": float(np.max(np.abs(u))),
                "mass": float(np.sum(u)),
                "n_qubits": q,
                "taylor_order": taylor_order,
                "bond_dim": bond_dim,
                "segment_build_time_s": (
                    seg_build_t if s == 0 else 0.0
                ),
            })

        seg_start += k

    return solutions, metrics


def _direct_lcu_signs(
    sign_recovery: str,
    q: int,
    nu: float,
    t_seg: float,
    L_box: float,
    u_seg: np.ndarray,
    taylor_order: int,
    bc: str,
) -> np.ndarray | None:
    """Signs for magnitude reconstruction.

    "none"            -> None (caller keeps non-negative magnitudes).
    "classical_oracle"-> sign(exp(A_seg*t_seg) u_seg) from the dense
                         block operator.  DIAGNOSTIC / hybrid (mirrors
                         the quantum step classically) -- used to
                         validate the shots path against statevector.
    The shadow-free Hadamard / dual-rail recovery is a follow-up.
    """
    if sign_recovery in ("none", None):
        return None
    if sign_recovery == "classical_oracle":
        op = conservative_burgers_lcu_operator(
            q, nu, t_seg, L_box, u_seg, taylor_order=taylor_order, bc=bc,
        ).real
        u_next = op @ u_seg
        eps = 1e-10 * max(np.max(np.abs(u_next)), 1.0)
        return np.where(np.abs(u_next) < eps, 1.0, np.sign(u_next))
    raise NotImplementedError(
        f"direct_lcu sign_recovery={sign_recovery!r} not implemented "
        "(dual-rail is a follow-up); use 'none', 'classical_oracle', "
        "or 'hadamard_test'",
    )


def _direct_lcu_hadamard_signs(
    psi_ref: np.ndarray,
    block_qc: QuantumCircuit,
    q: int,
    m: int,
    shots: int,
    backend: Any,
    seed: int | None = None,
) -> np.ndarray:
    """Interferometric (Hadamard-test) signs for exp(A_seg*T) psi_ref.

    F9 Option 2 generalised to a block-encoding -- ONE extra circuit per
    segment (not per-bin).  Prepare the signed reference psi_ref on the
    data register and a test ancilla in |+>; controlled-apply the
    block-encoding U_BE on test=1; interfere with the final H.
    Post-select the block ancilla on |0>; the per-bin interference
    p0[k]-p1[k] ~ psi_ref[k] * psi_out[k] (both real), so
    sign(psi_out[k]) = sign(interference[k]) * sign(psi_ref[k]).

    Magnitudes come from the direct run; this recovers signs only.
    Shadow-free: psi_ref's signs propagate from the known IC, refreshed
    by interference each segment (no classical RHS).
    """
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Operator
    from scipy.linalg import block_diag

    from q8020_cfd_qutil.circuit import execute_circuit_counts

    N = 1 << q
    test = q + m
    banc_cr = ClassicalRegister(m, "banc")
    test_cr = ClassicalRegister(1, "test")
    data_cr = ClassicalRegister(q, "data")
    qc = QuantumCircuit(q + m + 1)
    qc.add_register(banc_cr)
    qc.add_register(test_cr)
    qc.add_register(data_cr)
    qc.initialize(psi_ref.tolist(), range(q))
    qc.h(test)
    # Controlled block-encoding as one dense UnitaryGate block_diag(I, U)
    # (control = test as MSB).  Aer can't assemble a controlled composite
    # gate, but runs a dense UnitaryGate natively -- same trick as QLBM.
    u_be = Operator(block_qc).data
    cu = block_diag(np.eye(u_be.shape[0], dtype=complex), u_be)
    qc.append(
        UnitaryGate(cu, label="cU_BE"), list(range(q + m)) + [test],
    )
    qc.h(test)
    for i in range(q):
        qc.measure(i, data_cr[i])
    qc.measure(test, test_cr[0])
    for j in range(m):
        qc.measure(q + j, banc_cr[j])

    counts, _ = execute_circuit_counts(qc, backend, shots=shots, seed=seed)
    p0 = np.zeros(N)
    p1 = np.zeros(N)
    tot = 0
    for bits, cnt in counts.items():
        b = bits.replace(" ", "")
        if any(c != "0" for c in b[q + 1:]):  # post-select block anc = 0
            continue
        tot += cnt
        idx = int(b[:q], 2)
        if b[q] == "0":
            p0[idx] += cnt
        else:
            p1[idx] += cnt
    if tot > 0:
        p0 /= tot
        p1 /= tot
    interference = p0 - p1
    thr = 1e-3 * max(np.max(np.abs(interference)), 1e-30)
    signs = np.ones(N)
    for k in range(N):
        if abs(psi_ref[k]) > 1e-6 and abs(interference[k]) > thr:
            signs[k] = np.sign(interference[k]) * np.sign(psi_ref[k])
    return signs


def _run_direct_lcu_shots(
    u0: np.ndarray,
    x: np.ndarray,
    nu: float,
    dt: float,
    n_steps: int,
    bc: str = "periodic",
    source_fn: Callable | None = None,
    shots: int = 0,
    bond_dim: int | None = None,
    taylor_order: int = 4,
    segment_size: int = 1,
    use_mps_prep: bool = True,
    backend: Any = None,
    optimization_level: int = 1,
    seed: int | None = None,
    sign_recovery: str = "none",
    metric_transpile_timeout: float = 60.0,
) -> tuple[list[np.ndarray], list[dict]]:
    """Shots / ancilla post-selection path (M3).

    One segment-spanning block-encoding of exp(A_seg*T_segment) per
    segment (A frozen over the segment), post-select ancilla=|0>,
    reconstruct magnitudes from counts, recover signs, renormalise by
    lambda*sqrt(p_success).  Reports per-segment p_success and circuit
    cost.  Sim only (AerSimulator); the SELECT UnitaryGate runs natively
    so transpile is skipped (matches the CH LCU path).
    """
    from burgers_cole_hopf_circuit import post_select_counts
    from burgers_mps import (
        classical_to_mps, mps_to_circuit, normalize_state,
    )
    from q8020_cfd_qutil.circuit import (
        DEFAULT_METRIC_BASIS,
        execute_circuit_counts,
        safe_circuit_stats_in_basis,
    )

    if n_steps % segment_size != 0:
        raise ValueError(
            f"n_steps={n_steps} must be divisible by "
            f"segment_size={segment_size}",
        )
    if backend is None:
        from qiskit_aer import AerSimulator
        backend = AerSimulator()

    N = len(u0)
    q = int(round(np.log2(N)))
    if N != (1 << q):
        raise ValueError(f"N={N} must be a power of 2")
    L_box = N * float(x[1] - x[0])
    t_seg = segment_size * dt
    n_segments = n_steps // segment_size

    u = np.asarray(u0, dtype=float).copy()
    psi_current, cumulative_norm = normalize_state(u)
    solutions = [u.copy()]
    metrics: list[dict[str, Any]] = []
    metric_cache: dict | None = None

    for seg in range(n_segments):
        t0 = time.time()
        u_seg = psi_current * cumulative_norm  # physical field, frozen

        if use_mps_prep:
            tensors = classical_to_mps(
                psi_current, bond_dim=bond_dim, canonical="right",
            )
            prep_qc = mps_to_circuit(tensors)
            n_bond = prep_qc.num_qubits - q
        else:
            prep_qc = QuantumCircuit(q, name="prep")
            prep_qc.initialize(psi_current.tolist(), range(q))
            n_bond = 0

        block_qc, lam = conservative_burgers_lcu_step_circuit(
            q, nu, t_seg, L_box, u_seg, taylor_order=taylor_order, bc=bc,
        )
        m = block_qc.num_qubits - q
        total = q + n_bond + m

        anc_cr = ClassicalRegister(n_bond + m, "anc")
        data_cr = ClassicalRegister(q, "data")
        qc = QuantumCircuit(total)
        qc.add_register(anc_cr)
        qc.add_register(data_cr)
        qc.compose(prep_qc, qubits=list(range(q + n_bond)), inplace=True)
        block_map = list(range(q)) + list(
            range(q + n_bond, q + n_bond + m),
        )
        qc.compose(block_qc, qubits=block_map, inplace=True)
        for i in range(q):
            qc.measure(i, data_cr[i])
        for j, aq in enumerate(range(q, total)):
            qc.measure(aq, anc_cr[j])
        construct_t = time.time() - t0

        counts, exec_info = execute_circuit_counts(
            qc, backend, shots=shots, seed=seed,
        )
        n_kept, data_counts = post_select_counts(counts, q)
        p_success = n_kept / shots if shots > 0 else 0.0
        n_circuits = 1

        if n_kept == 0:
            print(
                f"[direct_lcu shots] segment {seg + 1}: zero "
                "post-selected counts; propagating NaN",
                file=sys.stderr, flush=True,
            )
            psi_current = np.full(N, np.nan)
            cumulative_norm = float("nan")
        else:
            mags = np.zeros(N)
            for bits, cnt in data_counts.items():
                mags[int(bits, 2)] = np.sqrt(cnt / n_kept)
            if sign_recovery == "hadamard_test":
                # psi_current is still the segment-start signed reference.
                signs = _direct_lcu_hadamard_signs(
                    psi_current, block_qc, q, m, shots, backend, seed,
                )
                n_circuits = 2
            else:
                signs = _direct_lcu_signs(
                    sign_recovery, q, nu, t_seg, L_box, u_seg,
                    taylor_order, bc,
                )
            psi_new = mags if signs is None else signs * mags
            nrm = float(np.linalg.norm(psi_new))
            if nrm > 1e-15:
                psi_new = psi_new / nrm
            cumulative_norm = cumulative_norm * lam * np.sqrt(p_success)
            psi_current = psi_new

        u = psi_current * cumulative_norm
        if source_fn is not None and np.all(np.isfinite(u)):
            u = u + t_seg * source_fn(x, (seg + 1) * t_seg)
            cumulative_norm = float(np.linalg.norm(u))
            if cumulative_norm > 1e-15:
                psi_current = u / cumulative_norm

        # Honest depth/cx requires lowering the SELECT to a hardware
        # basis.  For the flattened dense SELECT UnitaryGate that is a
        # generic (q+m)-qubit unitary synthesis (~4^(q+m) cx) -- both
        # crash-prone and physically meaningless, so try_decompose stays
        # False: metrics degrade to "unavailable" rather than reporting a
        # bogus cost.  Meaningful gate counts come from the structured
        # encoder (M4); the scaling story here is n_qubits / lambda.
        if metric_cache is None or source_fn is not None:
            metric_cache = safe_circuit_stats_in_basis(
                qc, DEFAULT_METRIC_BASIS,
                optimization_level=optimization_level,
                seed_transpiler=seed,
                timeout=metric_transpile_timeout, try_decompose=False,
            )

        met: dict[str, Any] = {
            "segment": seg,
            "step": (seg + 1) * segment_size,
            "segment_size": segment_size,
            "shots": shots,
            "n_kept": n_kept,
            "p_success": p_success,
            "lcu_lambda": lam,
            "cumulative_norm": cumulative_norm,
            "n_qubits": total,
            "n_ancilla": n_bond + m,
            "taylor_order": taylor_order,
            "bond_dim": bond_dim,
            "sign_recovery": sign_recovery,
            "circuit_construction_time_s": construct_t,
            "execution_time_s": exec_info.get("wall_time", 0.0),
            "n_circuits": n_circuits,
        }
        if metric_cache is not None and metric_cache.get("available"):
            met["circuit_depth"] = metric_cache["depth"]
            met["gate_counts"] = metric_cache["gate_counts"]
            met["circuit_metrics_available"] = True
        elif metric_cache is not None:
            met["circuit_metrics_available"] = False
            met["circuit_metrics_reason"] = metric_cache.get("reason")
        metrics.append(met)

        for _ in range(segment_size):
            solutions.append(u.copy())

        print(
            f"[direct_lcu shots] segment {seg + 1}/{n_segments} "
            f"p_success={p_success:.4f} lam={lam:.3f} "
            f"norm={cumulative_norm:.4e}",
            file=sys.stderr, flush=True,
        )

    return solutions, metrics
