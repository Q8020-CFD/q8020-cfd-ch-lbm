"""Phase-1 precursor to the pure-quantum QALB: linearised-BGK D1Q3.

Spec: docs/future/SPEC-qlbm-pure-quantum-qalb.md (Phase 1, future-work
#29).  Registered as `--method qlbm_circuit_linear`.

Idea (spec §5).  The hybrid `qlbm_circuit_hybrid` rebuilds the collision
unitary every step from a classically-known `f`, because full BGK is
nonlinear in the amplitudes -> state dependent -> forced to
measure-reprepare every step (k=1).  Linearising BGK about the rest
equilibrium (u_ref = 0, rho = 1) makes the collision a *fixed* linear
contraction `M3` on `delta_f = f - f_eq0` (f_eq0 = (0,1,0)).  Being
state independent it is block-encoded ONCE and applied across a k-step
measure-reprepare segment (k > 1).

  delta_f* = stream( M3 @ delta_f )        (collision then streaming)
  f       = delta_f + f_eq0,  u = moments(f)

`M3` is a symmetric contraction with eigenvalues {1, 1-omega, 1-omega}
(omega = 1/tau), block-encoded via the dilation
  U = [[M, sqrt(I - M^2)], [sqrt(I - M^2), -M]]
with one ancilla; post-selecting ancilla |0> applies M (spec §3.4 has
the Phase-2 Hermitization route which avoids post-selection; Phase 1
keeps the amplitude encoding and pays the post-selection, exactly the
plumbing this scaffold exists to exercise).

Valid only for smooth, low-Mach, near-equilibrium flow (drops the
O(delta_f^2) term -> loses shocks).  Scaffold for the
measure-reprepare(k) + readout machinery, not the production solver.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Callable

import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import UnitaryGate

from burgers_lbm import (
    collide_bgk,
    density,
    equilibrium,
    flatten_distributions,
    stream,
    tau_from_nu,
    unflatten_distributions,
    velocity,
)
from burgers_qlbm_circuit import build_streaming_circuit


# Rest equilibrium f_eq0 = (f_{-1}, f_0, f_{+1}) at u=0, rho=1.
F_EQ0 = np.array([0.0, 1.0, 0.0])


# ── Fixed linearised collision operator ───────────────────────────────


def linearized_collision_3x3(tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Fixed per-site affine BGK collision, linearised about u = 0.

    Returns (M3, b3) with f* = M3 @ f + b3.  In the delta_f = f - f_eq0
    variable the offset cancels (M3 @ f_eq0 + b3 = f_eq0), giving the
    purely linear map delta_f* = M3 @ delta_f.  omega = 1/tau.
    """
    omega = 1.0 / tau
    h = omega / 2.0
    M3 = np.array(
        [[1.0 - h, 0.0, -h], [0.0, 1.0 - omega, 0.0], [-h, 0.0, 1.0 - h]]
    )
    b3 = np.array([0.0, omega, 0.0])
    return M3, b3


def _collision_4x4(tau: float) -> np.ndarray:
    """M3 embedded in the 4-state velocity space (|11> slot = identity)."""
    M3, _ = linearized_collision_3x3(tau)
    M4 = np.eye(4)
    M4[:3, :3] = M3
    return M4


def _sym_sqrt(A: np.ndarray) -> np.ndarray:
    """Principal square root of a symmetric PSD matrix (eig-based, exact
    on the zero eigenvalues that make I - M^2 rank-deficient)."""
    w, V = np.linalg.eigh(0.5 * (A + A.T))
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def build_collision_block_unitary(tau: float) -> np.ndarray:
    """Block-encode the 4x4 collision M4 with one ancilla.

    Returns an 8x8 unitary on [vel(2) low, anc(1) high].  Post-selecting
    anc=|0> yields M4 on the velocity register.  Valid for tau > 1/2
    (omega < 2) so that I - M4^2 is PSD.
    """
    M4 = _collision_4x4(tau)
    S = _sym_sqrt(np.eye(4) - M4 @ M4)
    U = np.block([[M4, S], [S, -M4]])
    return U


def apply_linearized_collision(df: np.ndarray, tau: float) -> np.ndarray:
    """Apply the fixed linear collision M3 to delta_f at all sites."""
    M3, _ = linearized_collision_3x3(tau)
    return M3 @ df


# ── Circuit builders ──────────────────────────────────────────────────


def build_collision_circuit(tau: float, q: int) -> QuantumCircuit:
    """Collision block-encoding as a 3-qubit gate (vel0, vel1, anc).

    Returned circuit acts on q+3 qubits: position 0..q-1, velocity q,q+1,
    ancilla q+2.  Identity on position.
    """
    U = build_collision_block_unitary(tau)
    qc = QuantumCircuit(q + 3, name="collide_lin")
    qc.append(UnitaryGate(U, label="M_lin"), [q, q + 1, q + 2])
    return qc


def _f_to_delta_flat(f: np.ndarray) -> np.ndarray:
    """Flatten f -> 4N vector of delta_f = f - f_eq0 (broadcast)."""
    df = f - F_EQ0[:, None]
    return flatten_distributions(df)


def _delta_flat_to_f(vec: np.ndarray, N: int) -> np.ndarray:
    """Inverse of _f_to_delta_flat: 4N delta vector -> f (3,N)."""
    df = unflatten_distributions(vec, N)
    return df + F_EQ0[:, None]


# ── Main run ──────────────────────────────────────────────────────────


def run_qlbm_linear_simulation(
    u0: np.ndarray,
    x: np.ndarray,
    nu: float,
    dt: float,
    n_steps: int,
    bc: str = "periodic",
    source_fn: Callable | None = None,
    shots: int = 0,
    backend: Any = None,
    sign_recovery: str = "classical_oracle",
    segment_size: int = 1,
    optimization_level: int = 1,
    seed: int | None = None,
    metric_transpile_timeout: float = 60.0,
) -> tuple[list[np.ndarray], list[dict], list[int]]:
    """Linearised-BGK QLBM with measure-reprepare(k=segment_size).

    shots=0   -> exact path (post-selected block-encoding == M3 applied
                 directly); equals ideal linearised LBM.
    shots>0   -> segment circuits of k steps, ancilla post-selection per
                 collision, full delta_f readout + reprepare per segment.

    Returns (solutions, metrics, genuine_steps), same contract as the
    hybrid method.
    """
    N = len(u0)
    q = int(np.log2(N))
    assert N == (1 << q), f"N={N} must be a power of 2"
    dx = x[1] - x[0]

    dt_lbm = dx
    T_end = dt * n_steps
    n_steps_lbm = max(1, round(T_end / dt_lbm))
    tau = tau_from_nu(nu, dx, dt_lbm)
    k = max(1, min(int(segment_size), n_steps_lbm))
    if n_steps_lbm % k != 0:
        # keep snapshot alignment: fall back to k=1 if k does not divide
        print(f"[qlbm_circuit_linear] segment_size={k} does not divide "
              f"n_steps_lbm={n_steps_lbm}; using k=1", file=sys.stderr)
        k = 1

    print(
        f"[qlbm_circuit_linear] q={q} N={N} tau={tau:.4f} omega={1/tau:.4f} "
        f"n_steps_lbm={n_steps_lbm} k={k} (caller dt={dt:.3e} n={n_steps}) "
        f"shots={shots}",
        file=sys.stderr, flush=True,
    )

    M3, _ = linearized_collision_3x3(tau)
    f = equilibrium(u0)               # start from the local equilibrium
    lbm_solutions = [u0.copy()]
    metrics: list[dict] = []

    if shots == 0:
        # Exact path: iterate stream(M3 @ delta_f).
        for step in range(1, n_steps_lbm + 1):
            df = f - F_EQ0[:, None]
            df = stream(apply_linearized_collision(df, tau), bc=bc)
            f = df + F_EQ0[:, None]
            u_cur = velocity(f)
            lbm_solutions.append(u_cur.copy())
            metrics.append({
                "step": step, "u_max": float(np.max(np.abs(u_cur))),
                "rho_mean": float(np.mean(density(f))), "path": "statevector",
            })
    else:
        f = _run_shots_segments(
            f, M3, tau, q, N, n_steps_lbm, k, bc, shots, backend,
            sign_recovery, optimization_level, seed,
            metric_transpile_timeout, lbm_solutions, metrics,
        )

    # Remap to caller grid (nearest native lattice time), same as hybrid.
    solutions = [u0.copy()]
    for j in range(1, n_steps + 1):
        t_target = j * dt
        kk = min(round(t_target / dt_lbm), n_steps_lbm)
        solutions.append(lbm_solutions[kk].copy())
    genuine_steps = sorted({
        min(round(s * dt_lbm / dt), n_steps) for s in range(n_steps_lbm + 1)
    })
    return solutions, metrics, genuine_steps


def _run_shots_segments(
    f, M3, tau, q, N, n_steps_lbm, k, bc, shots, backend, sign_recovery,
    optimization_level, seed, metric_transpile_timeout,
    lbm_solutions, metrics,
):
    """Segmented shots evolution with per-collision ancilla post-select."""
    from q8020_cfd_qutil.circuit import (
        execute_circuit_counts,
        transpile_circuit,
    )
    if backend is None:
        from qiskit_aer import AerSimulator
        backend = AerSimulator()

    dim = 4 * N
    n_qubits = q + 2
    coll = build_collision_circuit(tau, q)        # q+3 qubits
    strm = build_streaming_circuit(q)             # q+2 qubits
    n_seg = n_steps_lbm // k

    for seg in range(n_seg):
        df_flat = _f_to_delta_flat(f)
        norm_in = float(np.linalg.norm(df_flat))
        if norm_in < 1e-15:
            for _ in range(k):
                lbm_solutions.append(velocity(f).copy())
            continue
        psi_in = df_flat / norm_in
        cumulative_norm = norm_in

        # Build segment: init delta_f, then k x [collide, postselect, stream].
        qc = QuantumCircuit(q + 3, q + 2)          # 1 anc + (q+2) data read
        qc.initialize(psi_in.tolist(), range(n_qubits))
        anc_cr = ClassicalRegister(k, "anc_hist")
        qc.add_register(anc_cr)
        for s in range(k):
            qc.compose(coll, range(q + 3), inplace=True)
            qc.measure(q + 2, anc_cr[s])
            qc.reset(q + 2)
            qc.compose(strm, range(n_qubits), inplace=True)
        qc.measure(range(n_qubits), range(n_qubits))

        t_c = time.time()
        qc_t, t_info = transpile_circuit(
            qc, backend, optimization_level=optimization_level,
            seed_transpiler=seed,
        )
        counts, exec_info = execute_circuit_counts(
            qc_t, backend, shots=shots, seed=seed,
        )

        # Post-select all-zero ancilla history; reconstruct |delta_f|.
        psi_mag = np.zeros(dim)
        kept = 0
        for bitstr, cnt in counts.items():
            tok = bitstr.replace(" ", "")
            anc_bits = tok[:k]                      # MSB-first: anc_hist high
            data_bits = tok[k:]
            if "1" in anc_bits:
                continue
            idx = int(data_bits, 2)
            if idx < dim:
                psi_mag[idx] += cnt
                kept += cnt
        if kept == 0:
            raise RuntimeError("all shots rejected by post-selection")
        psi_mag = np.sqrt(psi_mag / kept)          # unit-norm shape from shots

        # Magnitude scale + signs from the classical k-step linearised
        # oracle (the deflation/contraction is tracked classically, as in
        # the hybrid).  Quantum magnitude recovery + hadamard_test signs
        # are the next refinement (spec §4.3).
        oracle_flat = _segment_oracle(f, M3, k, bc, N)
        final_norm = float(np.linalg.norm(oracle_flat))
        signs = np.sign(oracle_flat)
        signs[signs == 0] = 1.0
        psi_signed = signs * psi_mag * final_norm

        f = _delta_flat_to_f(psi_signed, N)
        p_kept = kept / shots
        contraction = final_norm / max(norm_in, 1e-30)
        cumulative_norm = final_norm
        for s in range(k):
            lbm_solutions.append(velocity(f).copy())  # snapshot at boundary
        metrics.append({
            "segment": seg, "k": k, "p_success": float(p_kept),
            "norm": float(cumulative_norm * contraction),
            "u_max": float(np.max(np.abs(velocity(f)))),
            "transpile_s": t_info["wall_time"],
            "execute_s": exec_info["wall_time"], "path": "shots",
        })
    return f


def _segment_oracle(f, M3, k, bc, N) -> np.ndarray:
    """Classical k-step linearised trajectory (delta_f), for sign oracle."""
    df = (f - F_EQ0[:, None])
    for _ in range(k):
        df = stream(M3 @ df, bc=bc)
    return flatten_distributions(df)


# ── Offline / statevector gates (spec §6, gates 1-2) ──────────────────


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    tau = 2.42
    q = 4
    N = 1 << q

    # Gate 1: operator exactness (rest-eq fixed point + O(u^2) vs BGK).
    M3, b3 = linearized_collision_3x3(tau)
    f0 = np.array([[0.0], [1.0], [0.0]])
    fp = float(np.max(np.abs(M3 @ f0 + b3[:, None] - f0)))
    print(f"[gate1] rest-eq fixed point max|err|={fp:.2e} "
          f"({'PASS' if fp < 1e-14 else 'FAIL'})")
    errs = []
    for u in (0.2, 0.1, 0.05):
        fa = equilibrium(np.array([u]))
        errs.append(np.max(np.abs((M3 @ (fa - F_EQ0[:, None])) + F_EQ0[:, None]
                                  - collide_bgk(fa, tau))))
    slope = np.log2(errs[0] / errs[1])
    print(f"[gate1] O(u^2) slope={slope:.2f} "
          f"({'PASS' if 1.9 < slope < 2.1 else 'FAIL'})")

    # Gate 2: block-encoding extracts M4.
    from burgers_mpo import extract_block_encoded_operator
    U = build_collision_block_unitary(tau)
    M4 = _collision_4x4(tau)
    M4_ext = extract_block_encoded_operator(U, n_system=2, n_ancilla=1)
    be = float(np.max(np.abs(M4_ext - M4)))
    print(f"[gate2] block-encoding extract max|err|={be:.2e} "
          f"({'PASS' if be < 1e-12 else 'FAIL'})")

    # Gates 3-4 use run params giving n_steps_lbm = 10 (cfl=1) and a
    # gentle relaxation (tau ~ 2.4, like the aligned case).
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    dx = x[1] - x[0]
    u0 = 0.3 * np.sin(2 * np.pi * x)
    nu_t, dt_t, nsteps_t = 0.119, dx, 10          # tau = nu/dx + 0.5 ~= 2.4
    tau_run = tau_from_nu(nu_t, dx, dx)
    M3r, _ = linearized_collision_3x3(tau_run)

    def classical_lin(n_lbm: int) -> np.ndarray:
        ff = equilibrium(u0)
        for _ in range(n_lbm):
            ff = stream(M3r @ (ff - F_EQ0[:, None]), bc="periodic") \
                + F_EQ0[:, None]
        return velocity(ff)

    # Gate 3: statevector trajectory == classical linearised LBM.
    sols, _, _ = run_qlbm_linear_simulation(
        u0, x, nu_t, dt_t, nsteps_t, bc="periodic", shots=0,
    )
    g3 = float(np.max(np.abs(sols[-1] - classical_lin(10))))
    print(f"[gate3] statevector vs classical max|err|={g3:.2e} "
          f"({'PASS' if g3 < 1e-12 else 'FAIL'})")

    # Gate 4: shots pipeline with k=2 (segments chain; post-selection;
    # full-delta_f readout + reprepare) tracks the statevector shape.
    sols_s, mets_s, _ = run_qlbm_linear_simulation(
        u0, x, nu_t, dt_t, nsteps_t, bc="periodic", shots=60000,
        segment_size=2,
    )
    g4 = float(np.max(np.abs(sols_s[-1] - sols[-1])))
    p = mets_s[-1]["p_success"]
    print(f"[gate4] shots(k=2) vs statevector max|err|={g4:.2e}  "
          f"p_success={p:.3f}  ({'PASS' if g4 < 0.1 else 'FAIL'})")
