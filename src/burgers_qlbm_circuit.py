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
) -> tuple[list[np.ndarray], list[dict]]:
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
      "hadamard_test"    -- NotImplementedError (deferred to
                            FUTURE-WORK #26: per-bin Hadamard test).

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
            if sign_recovery == "hadamard_test":
                raise NotImplementedError(
                    "--sign-recovery hadamard_test for qlbm_circuit is "
                    "deferred to FUTURE-WORK #26.  v1 supports "
                    "{none, classical_oracle}."
                )

            vec_in = flatten_distributions(f)
            norm_in = float(np.linalg.norm(vec_in))
            if norm_in < 1e-15:
                # Zero state -- skip the circuit entirely; collision
                # unitary is ill-defined for a zero input.
                f = np.zeros_like(f)
                leakage = 0.0
            else:
                qc, contraction = build_qlbm_step_circuit(f, tau, q)
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

                from qiskit.compiler import transpile as _transpile

                qc_full = QuantumCircuit(n_qubits, n_qubits)
                qc_full.initialize(psi_in.tolist(), range(n_qubits))
                qc_full.compose(qc, inplace=True)
                qc_full.measure_all()
                qc_t = _transpile(
                    qc_full, backend_eff, optimization_level=1,
                )
                counts = (
                    backend_eff.run(qc_t, shots=shots)
                    .result()
                    .get_counts()
                )

                psi_out_mag = np.zeros(dim)
                for bitstr, cnt in counts.items():
                    # Qiskit measure_all bitstrings: low-index qubit on
                    # the right.  int(bitstr, 2) matches Statevector
                    # indexing used by the statevector branch above.
                    idx = int(bitstr.replace(" ", ""), 2)
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
        metrics.append(step_met)
        lbm_solutions.append(u_cur.copy())

    # Remap to caller's time grid via nearest-neighbor.
    solutions = [u0.copy()]
    for j in range(1, n_steps + 1):
        t_target = j * dt
        k = min(round(t_target / dt_lbm), n_steps_lbm)
        solutions.append(lbm_solutions[k].copy())

    return solutions, metrics
