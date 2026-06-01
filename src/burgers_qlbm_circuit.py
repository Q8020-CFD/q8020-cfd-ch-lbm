"""Quantum circuit QLBM solver for 1-D Burgers (F11 Phase 2).

Implements the D1Q3 lattice Boltzmann algorithm as a quantum circuit:
- Collision: Option A (dense unitary per step, classically pre-computed)
- Streaming: controlled increment/decrement on position register

Register layout (interleaved encoding):
  |v1 v0> |p_{q-1} ... p_0>
  velocity (2 qubits) | position (q qubits)
  |00> = f_{-1}, |01> = f_0, |10> = f_{+1}, |11> = unused
"""

from __future__ import annotations

import sys
import time
from typing import Any, Callable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector
from scipy.linalg import block_diag

from burgers_lbm import (
    W,
    collide_bgk,
    density,
    equilibrium,
    flatten_distributions,
    stream,
    tau_from_nu,
    unflatten_distributions,
    velocity,
)


# Gate basis used only for reporting qlbm circuit depth / gate counts
# (the canonical IBM superconducting basis; cx is the dominant-cost
# metric and is directly comparable to the cole_hopf_circuit cx counts).
# Reporting only -- execution still runs the exact dense step unitary.
QLBM_METRIC_BASIS = ["cx", "rz", "sx", "x"]


# ── Streaming circuit ─────────────────────────────────────────────────


def _increment_gate(q: int) -> np.ndarray:
    """q-qubit increment (cyclic +1 mod 2^q) as a unitary matrix.

    Maps |j> -> |j+1 mod N> where N = 2^q.
    """
    N = 1 << q
    U = np.zeros((N, N))
    for j in range(N):
        U[(j + 1) % N, j] = 1.0
    return U


def _decrement_gate(q: int) -> np.ndarray:
    """q-qubit decrement (cyclic -1 mod 2^q) as a unitary matrix.

    Maps |j> -> |j-1 mod N> where N = 2^q.
    """
    N = 1 << q
    U = np.zeros((N, N))
    for j in range(N):
        U[(j - 1) % N, j] = 1.0
    return U


def build_streaming_unitary(q: int) -> np.ndarray:
    """Build the full (q+2)-qubit streaming unitary.

    Velocity register: 2 qubits (top); Position register: q qubits.
    Total dimension: 4 * 2^q.

    Action:
      |00>|j> -> |00>|j-1>  (f_{-1} shifts left = decrement)
      |01>|j> -> |01>|j>    (f_0 no shift)
      |10>|j> -> |10>|j+1>  (f_{+1} shifts right = increment)
      |11>|j> -> |11>|j>    (unused, identity)
    """
    N = 1 << q

    # Build as block-diagonal in velocity subspaces
    dec = _decrement_gate(q)   # |00>: shift left
    eye = np.eye(N)            # |01>: identity
    inc = _increment_gate(q)   # |10>: shift right
    eye2 = np.eye(N)           # |11>: identity

    U = block_diag(dec, eye, inc, eye2)
    return U


def build_streaming_circuit(q: int) -> QuantumCircuit:
    """Build streaming as a UnitaryGate on q+2 qubits.

    For small q this is practical. For larger q, a decomposed version
    using controlled-increment cascades would be more efficient.
    """
    n_qubits = q + 2
    U = build_streaming_unitary(q)
    qc = QuantumCircuit(n_qubits, name="stream")
    qc.append(UnitaryGate(U, label="S_stream"), range(n_qubits))
    return qc


# ── Collision circuit (Option A: dense unitary) ───────────────────────


def build_collision_unitary(
    f_pre: np.ndarray, tau: float, q: int
) -> np.ndarray:
    """Build the collision as a unitary embedding (Option A).

    Given pre-collision distributions f_pre (3, N), compute
    post-collision f_post, then find the unitary that maps the
    normalized f_pre state to the normalized f_post state.

    Since collision is non-unitary (contractive for tau > 0.5), we
    embed it in a unitary via the standard dilation: given |psi_in>
    and |psi_out>, find U such that U|psi_in> = |psi_out>.

    For statevector simulation, we construct the full 4N x 4N unitary
    that maps the input amplitudes to the output amplitudes. This is
    done via a Householder reflection approach.
    """
    N = 1 << q
    dim = 4 * N

    # Compute post-collision distributions
    f_post = collide_bgk(f_pre, tau)

    # Flatten to 4N vectors
    vec_in = flatten_distributions(f_pre)
    vec_out = flatten_distributions(f_post)

    # Normalize both
    norm_in = np.linalg.norm(vec_in)
    norm_out = np.linalg.norm(vec_out)

    if norm_in < 1e-15 or norm_out < 1e-15:
        return np.eye(dim)

    psi_in = vec_in / norm_in
    psi_out = vec_out / norm_out

    # Build unitary mapping psi_in -> psi_out using two Householder
    # reflections: U = R2 @ R1 where R1 maps psi_in to e_0 and
    # R2 maps e_0 to psi_out.
    U = _unitary_from_state_map(psi_in, psi_out, dim)
    return U, norm_out / norm_in


def _unitary_from_state_map(
    psi_in: np.ndarray, psi_out: np.ndarray, dim: int
) -> np.ndarray:
    """Build unitary U such that U @ psi_in = psi_out.

    Uses the construction: if v = psi_in - psi_out, then the
    Householder reflection H = I - 2 vv^H / (v^H v) maps psi_in
    to psi_out (when ||psi_in|| = ||psi_out|| = 1).

    If psi_in ~= psi_out, return identity.
    """
    v = psi_in - psi_out
    vdotv = np.dot(v.conj(), v)
    if np.abs(vdotv) < 1e-14:
        return np.eye(dim)
    H = np.eye(dim) - 2.0 * np.outer(v, v.conj()) / vdotv
    return H


# ── Full timestep circuit ─────────────────────────────────────────────


def build_qlbm_step_circuit(
    f_pre: np.ndarray, tau: float, q: int
) -> tuple[QuantumCircuit, float]:
    """Build one LBM timestep circuit: collision + streaming.

    Returns (circuit, contraction_factor) where contraction_factor
    is norm_out/norm_in from the collision (needed for amplitude
    rescaling).
    """
    n_qubits = q + 2

    # Collision unitary
    U_coll, contraction = build_collision_unitary(f_pre, tau, q)

    # Streaming unitary
    U_stream = build_streaming_unitary(q)

    # Combined: stream after collide
    U_step = U_stream @ U_coll

    qc = QuantumCircuit(n_qubits, name="qlbm_step")
    qc.append(
        UnitaryGate(U_step, label="QLBM_step"), range(n_qubits)
    )
    return qc, contraction


# ── Hadamard-test sign recovery (FUTURE-WORK #26) ─────────────────────


def _qlbm_hadamard_signs(
    psi_in: np.ndarray,
    qc_step: QuantumCircuit,
    n_qubits: int,
    shots: int,
    backend: Any,
    q: int,
    optimization_level: int = 1,
    seed: int | None = None,
    mag: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Per-bin Hadamard-test sign recovery for the QLBM shots path.

    For each basis index k in [0, 4N) it runs a Hadamard test that
    estimates Re(<k| U_step |psi_in>).  Every QLBM operator (the
    collision Householder reflection and the real streaming
    permutation) is real, so psi_out[k] is real and its sign is
    exactly sign(Re(<k| U_step |psi_in>)).  This is a stand-alone
    (non-hybrid) signal, unlike classical_oracle.

    The controlled gate is materialised as a single dense
    ``block_diag(I, X_k . U_step . P)`` ``UnitaryGate`` so Aer can
    simulate it natively (no per-bin transpilation of a controlled
    composite, which is ~10x faster).  ``P`` is the Householder prep
    with ``P|0> = psi_in``; ``X_k`` is the bit-flip selector, applied
    as a cheap XOR row permutation of the base matrix.  Sim-only by
    construction (dense 2^(n_qubits+1) gate).

    If ``mag`` (the direct ``|psi_out|`` estimate) is given, bins whose
    magnitude is negligible are skipped (sign there is irrelevant) and
    left at +1, roughly halving the circuit count.

    Returns a sign vector in {-1, +1}^{4N} (zeros mapped to +1) plus
    aggregate diagnostics.
    """
    from qiskit.quantum_info import Operator

    from q8020_cfd_qutil.circuit import (
        circuit_stats_in_basis,
        execute_circuit_counts,
    )

    dim = 1 << n_qubits  # 4N
    test = n_qubits  # ancilla is the last (most significant) qubit

    # base|0> = U_step P|0> = U_step psi_in; row-permuting base by XOR k
    # realises X_k, so (X_k base)[0, 0] = (U_step psi_in)[k] = psi_out[k].
    e0 = np.zeros(dim)
    e0[0] = 1.0
    P = _unitary_from_state_map(e0, psi_in, dim)
    U_step = Operator(qc_step).data
    base = U_step @ P
    eye = np.eye(dim)

    if mag is not None:
        thresh = 1e-6 * float(np.max(mag)) if np.max(mag) > 0 else 0.0
        run_bin = mag > thresh
    else:
        run_bin = np.ones(dim, dtype=bool)

    re_psi = np.zeros(dim)
    n_kept_total = 0
    n_run = 0
    stat_depth: int | None = None
    stat_gates: dict | None = None

    for k in range(dim):
        if not run_bin[k]:
            continue
        n_run += 1
        inner_k = base[[i ^ k for i in range(dim)], :]
        M = block_diag(eye, inner_k)

        qc_k = QuantumCircuit(n_qubits + 1, n_qubits + 1)
        qc_k.h(test)
        qc_k.append(UnitaryGate(M), list(range(n_qubits)) + [test])
        qc_k.h(test)
        qc_k.measure(range(n_qubits + 1), range(n_qubits + 1))

        # One-time metric-only lowering of the per-bin Hadamard circuit to
        # a hardware basis (dense UnitaryGate -> cx + 1-qubit gates), for
        # honest depth/gate reporting.  Every bin shares this structure --
        # the dense gate's decomposition cost is fixed by its dimension,
        # not its entries -- so one representative covers all n_bins_run.
        # Does not affect execution (Aer simulates the dense gate directly).
        if stat_depth is None:
            stats = circuit_stats_in_basis(
                qc_k, QLBM_METRIC_BASIS,
                optimization_level=optimization_level,
                seed_transpiler=seed,
            )
            stat_depth = stats["depth"]
            stat_gates = stats["gate_counts"]

        # Aer simulates the dense UnitaryGate directly; no transpile.
        counts, _ = execute_circuit_counts(
            qc_k, backend, shots=shots, seed=seed,
        )

        n0 = 0
        n1 = 0
        for bitstr, cnt in counts.items():
            # MSB-first: bits[0] is the test (highest) qubit; the
            # remaining n_qubits chars are the data register.
            bits = bitstr.replace(" ", "")
            if any(c != "0" for c in bits[-n_qubits:]):
                continue
            if bits[0] == "0":
                n0 += cnt
            else:
                n1 += cnt
        re_psi[k] = (n0 - n1) / max(shots, 1)
        n_kept_total += n0 + n1

    signs = np.sign(re_psi)
    signs[signs == 0] = 1.0
    info = {
        "p_kept_per_bin_mean": (
            n_kept_total / (n_run * max(shots, 1)) if n_run else 0.0
        ),
        "n_bins_run": n_run,
        "depth_per_circuit": stat_depth if stat_depth is not None else 0,
        "gate_counts_per_circuit": stat_gates if stat_gates is not None
        else {},
    }
    return signs, info


# ── Statevector simulation driver ─────────────────────────────────────


def run_qlbm_circuit_simulation(
    u0: np.ndarray,
    x: np.ndarray,
    nu: float,
    dt: float,
    n_steps: int,
    bc: str = "periodic",
    source_fn: Callable | None = None,
    shots: int = 0,
    backend: Any = None,
    sign_recovery: str = "none",
    optimization_level: int = 1,
    seed: int | None = None,
    metric_transpile_timeout: float = 60.0,
) -> tuple[list[np.ndarray], list[dict], list[int]]:
    """Run QLBM circuit simulation (statevector or shots).

    shots=0:   statevector path -- exact amplitudes via Aer/Statevector.
    shots>0:   real circuit + measurement on ``backend``, then magnitude
               reconstruction (sqrt(counts/S)) and sign recovery.

    sign_recovery (shots>0 only):
      "none"             -- non-negative magnitudes only; correct when
                            f stays >= 0 (typical smooth-flow regime).
      "classical_oracle" -- copy signs from a parallel classical
                            collide_bgk + stream step; diagnostic-grade
                            (run is hybrid in the recovery branch).
      "hadamard_test"    -- stand-alone interferometric sign recovery:
                            per-bin Hadamard test estimates
                            sign(Re(<k|U_step|psi_in>)).  Costs 4N extra
                            circuits per step; diagnostic-grade.

    Returns all steps (solutions[i] = u at step i, length n_steps+1).
    """
    N = len(u0)
    q = int(np.log2(N))
    assert N == (1 << q), f"N={N} must be a power of 2"
    dx = x[1] - x[0]

    # LBM native timestep: dt_lbm = dx (one lattice site per step).
    dt_lbm = dx
    T_end = dt * n_steps
    n_steps_lbm = max(1, round(T_end / dt_lbm))
    tau = tau_from_nu(nu, dx, dt_lbm)
    n_qubits = q + 2
    dim = 4 * N

    print(
        f"[qlbm_circuit] q={q} N={N} tau={tau:.4f} "
        f"n_qubits={n_qubits} n_steps_lbm={n_steps_lbm} "
        f"(caller: dt={dt:.4e} n={n_steps}) shots={shots}",
        file=sys.stderr, flush=True,
    )

    # Initialize distributions from equilibrium
    f = equilibrium(u0)
    lbm_solutions = [u0.copy()]
    metrics: list[dict] = []
    cumulative_norm = np.linalg.norm(flatten_distributions(f))

    for step in range(1, n_steps_lbm + 1):
        t0 = time.time()
        leakage: float | None = None
        neg_mass: float | None = None
        had_p_kept: float | None = None
        had_depth: int | None = None
        had_gates: dict | None = None
        circ_after: dict | None = None
        construct_t = transpile_t = execute_t = 0.0
        sign_t = metric_t = 0.0
        n_circ = 0

        if shots == 0:
            # Statevector path: build circuit, simulate
            qc, contraction = build_qlbm_step_circuit(f, tau, q)
            cumulative_norm *= contraction

            # Prepare input state
            vec_in = flatten_distributions(f)
            norm_in = np.linalg.norm(vec_in)
            if norm_in < 1e-15:
                psi_in = np.zeros(dim)
                psi_in[0] = 1.0
            else:
                psi_in = vec_in / norm_in

            # Simulate
            sv = Statevector(psi_in)
            sv = sv.evolve(qc)
            psi_out = np.array(sv.data, dtype=complex).real

            # Reconstruct distributions
            f_out_flat = psi_out * cumulative_norm
            f = unflatten_distributions(f_out_flat, N)

            if bc == "dirichlet":
                f[2, 0] = f[0, 0]    # left wall
                f[0, -1] = f[2, -1]  # right wall

        else:
            # Shots path: real circuit + measurement + reconstruction.
            # See SPEC-qlbm-shots-and-sign-recovery.md §2-§3.
            vec_in = flatten_distributions(f)
            norm_in = float(np.linalg.norm(vec_in))
            if norm_in < 1e-15:
                # Zero state -- skip the circuit entirely; collision
                # unitary is ill-defined for a zero input.
                f = np.zeros_like(f)
                leakage = 0.0
            else:
                t_c = time.time()
                qc, contraction = build_qlbm_step_circuit(f, tau, q)
                construct_t = time.time() - t_c
                cumulative_norm *= contraction
                psi_in = vec_in / norm_in

                # Pre-collision oracle for classical_oracle sign
                # recovery (run BEFORE we mutate f).
                if sign_recovery == "classical_oracle":
                    f_ref = collide_bgk(f, tau)
                    f_ref = stream(f_ref, bc=bc)
                    vec_ref = flatten_distributions(f_ref)
                    neg_mass = float(
                        np.sum(np.abs(vec_ref[vec_ref < 0]))
                        / max(np.sum(np.abs(vec_ref)), 1e-30)
                    )
                else:
                    vec_ref = None

                backend_eff = backend
                if backend_eff is None:
                    from qiskit_aer import AerSimulator
                    backend_eff = AerSimulator()

                # Use the shared qutil transpile + execute helpers so
                # QLBM and cole_hopf shots paths honour the same
                # --optimization-level / --seed contract, and so
                # hardware backends flow through SamplerV2 consistently.
                from q8020_cfd_qutil.circuit import (
                    execute_circuit_counts,
                    safe_circuit_stats_in_basis,
                    transpile_circuit,
                )

                # No explicit classical register: measure_all() adds its
                # own n_qubits-bit register.  Declaring one here too would
                # produce a 2*n_qubits-bit string ([measured | empty]), so
                # int(bitstr, 2) = measured << n_qubits >= dim and every
                # count gets dropped -> zero mass.
                qc_full = QuantumCircuit(n_qubits)
                qc_full.initialize(psi_in.tolist(), range(n_qubits))
                qc_full.compose(qc, inplace=True)
                qc_full.measure_all()
                qc_t, t_info = transpile_circuit(
                    qc_full, backend_eff,
                    optimization_level=optimization_level,
                    seed_transpiler=seed,
                )
                transpile_t = t_info["wall_time"]
                # Aer keeps the dense collide+stream step as one
                # un-lowered UnitaryGate, so its transpiled depth/cx are
                # meaningless.  Decompose to a hardware basis purely for
                # honest depth/gate reporting (does not affect execution).
                # Tracked as metric overhead, separate from the pipeline
                # transpile time above.
                # Hardware-basis decomposition for honest depth/cx, run
                # out-of-process so a synthesis crash/hang can't kill the
                # run.  metric_transpile_timeout caps it (0 = uncapped).
                t_m = time.time()
                circ_after = safe_circuit_stats_in_basis(
                    qc_full, QLBM_METRIC_BASIS,
                    optimization_level=optimization_level,
                    seed_transpiler=seed,
                    timeout=metric_transpile_timeout,
                    try_decompose=False,
                )
                metric_t = time.time() - t_m
                counts, exec_info = execute_circuit_counts(
                    qc_t, backend_eff, shots=shots, seed=seed,
                )
                execute_t = exec_info["wall_time"]
                n_circ = 1  # the main step circuit

                psi_out_mag = np.zeros(dim)
                for bitstr, cnt in counts.items():
                    # execute_circuit_counts strips inter-register
                    # spaces, so the bitstring is a single MSB-first
                    # token; int(bitstr, 2) matches Statevector
                    # indexing used by the statevector branch above.
                    idx = int(bitstr, 2)
                    if idx < dim:
                        psi_out_mag[idx] = np.sqrt(cnt / shots)

                # Leakage: probability mass landing in the |11>
                # velocity block (indices 3N..4N-1).  Structurally zero
                # on noise-free sim; nonzero flags noise / transpilation
                # error.
                unused_block = psi_out_mag[3 * N:4 * N]
                leakage = float(np.sum(unused_block ** 2))

                if sign_recovery == "classical_oracle":
                    signs = np.sign(vec_ref)
                    signs[signs == 0] = 1.0
                    psi_out = signs * psi_out_mag
                elif sign_recovery == "hadamard_test":
                    t_s = time.time()
                    signs, had_info = _qlbm_hadamard_signs(
                        psi_in, qc, n_qubits, shots, backend_eff, q,
                        optimization_level=optimization_level,
                        seed=seed, mag=psi_out_mag,
                    )
                    sign_t = time.time() - t_s
                    had_p_kept = had_info["p_kept_per_bin_mean"]
                    had_depth = had_info["depth_per_circuit"]
                    had_gates = had_info["gate_counts_per_circuit"]
                    n_circ += had_info["n_bins_run"]  # per-bin recovery
                    psi_out = signs * psi_out_mag
                else:
                    psi_out = psi_out_mag

                f_out_flat = psi_out * cumulative_norm
                f = unflatten_distributions(f_out_flat, N)

                if bc == "dirichlet":
                    f[2, 0] = f[0, 0]
                    f[0, -1] = f[2, -1]

        # Source term
        if source_fn is not None:
            t = step * dt_lbm
            g = source_fn(x, t)
            for i in range(3):
                f[i] += W[i] * g * dt_lbm

        u_cur = velocity(f)
        rho_cur = density(f)
        elapsed = time.time() - t0

        step_met = {
            "step": step,
            "rho_max": float(np.max(rho_cur)),
            "rho_min": float(np.min(rho_cur)),
            "u_max": float(np.max(np.abs(u_cur))),
            "mass": float(np.sum(rho_cur)),
            "norm": float(cumulative_norm),
            "step_time_s": elapsed,
            "n_qubits": n_qubits,
        }
        if shots > 0:
            step_met["leakage"] = leakage
            step_met["negative_mass"] = neg_mass
            step_met["sign_recovery"] = sign_recovery
            if had_p_kept is not None:
                step_met["hadamard_p_kept"] = had_p_kept
            # Sign-recovery circuit size, per Hadamard-test bin circuit
            # (all n_bins_run circuits share this depth/gate profile).
            if had_depth is not None:
                step_met["sign_recovery_depth_per_circuit"] = had_depth
                step_met["sign_recovery_gate_counts_per_circuit"] = had_gates
            # Transpiled-circuit stats for this step (basis-gate depth /
            # gate counts), captured out-of-process so a synthesis crash
            # can't kill the run.  Only meaningful on the shots path.
            if circ_after is not None and circ_after.get("available"):
                step_met["circuit_depth"] = circ_after["depth"]
                step_met["gate_counts"] = circ_after["gate_counts"]
                step_met["n_qubits"] = circ_after["num_qubits"]
                step_met["circuit_metrics_available"] = True
            else:
                step_met["circuit_metrics_available"] = False
                step_met["circuit_metrics_reason"] = (
                    circ_after.get("reason") if circ_after else "skipped"
                )
            # Wall-time split (construct / transpile / execute), plus the
            # qlbm-specific sign-recovery and metric-only transpile costs
            # broken out so they don't contaminate the pipeline splits.
            step_met["circuit_construction_time_s"] = construct_t
            step_met["transpilation_time_s"] = transpile_t
            step_met["execution_time_s"] = execute_t
            step_met["sign_recovery_time_s"] = sign_t
            step_met["metric_transpile_time_s"] = metric_t
            step_met["n_circuits"] = n_circ
        metrics.append(step_met)
        lbm_solutions.append(u_cur.copy())

    # Remap to caller's time grid via nearest-neighbor.  This fills every
    # caller step for contract safety (solutions[-1], arbitrary indexing),
    # but only the genuine_steps below correspond to a real native lattice
    # state -- the rest are duplicates and must not be persisted as data.
    solutions = [u0.copy()]
    for j in range(1, n_steps + 1):
        t_target = j * dt
        k = min(round(t_target / dt_lbm), n_steps_lbm)
        solutions.append(lbm_solutions[k].copy())

    # Caller-step indices that land on a genuinely-computed native state.
    genuine_steps = sorted({
        min(round(k * dt_lbm / dt), n_steps)
        for k in range(n_steps_lbm + 1)
    })

    return solutions, metrics, genuine_steps
