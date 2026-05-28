"""Quantum circuit Cole-Hopf Burgers solver (F10 Phase B).

Two propagator variants for the heat equation exp(nu*L*dt):
1. qft-diagonal: QFT + diagonal phase-damping + QFT^-1 (default)
2. dense-block: exact eigendecomposition + block-encoding (q <= 5)

Both use a single ancilla qubit for non-unitary block encoding.
Post-selection on ancilla |0> recovers the contractive propagator.
"""

from __future__ import annotations

import sys
import time
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit.library import QFTGate, RYGate, UnitaryGate, XGate
from scipy.linalg import expm

from burgers_cole_hopf import build_laplacian_dense
from burgers_encoding import (
    permute_from_encoding,
    permute_operator,
    permute_to_encoding,
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


def dct_circuit(q: int, inverse: bool = False) -> QuantumCircuit:
    """DCT-II (or its inverse) as a (q+1)-qubit quantum circuit.

    Klappenecker-Rotteler 2002 / Makhoul-style construction: the
    DCT-II of length N=2^q is realised as the upper-left N×N block of
    a (q+1)-qubit unitary, with an ancilla qubit (qubit q) that starts
    and ends in |0> deterministically.  Total gate count is O(q^2),
    versus O(4^q) for a UnitaryGate-wrapped dense matrix.

    Construction (forward DCT, ancilla a = qubit q, data d = qubits 0..q-1):
      1. H on a.
      2. Conditional bit-complement on data (CNOTs from a to each d).
      3. QFT^{-1} on all q+1 qubits.
      4. Diagonal phase exp(-i pi k/(2N)) on the (q+1)-bit index k —
         decomposes as one P gate per qubit.
      5. Conditional negation P_neg|d> = |(N-d) mod N> on data
         (CNOTs from a + a-controlled incrementer on d).
      6. Conditional Ry(pi/2) on a (active when d != 0): apply Ry(pi/2)
         then a (q-controlled, ctrl_state=0...0) Ry(-pi/2) to undo at d=0.

    Returns a (q+1)-qubit circuit; bottom q qubits are data, top qubit
    is the DCT ancilla.  ``Operator(dct_circuit(q)).data[:N, :N]`` equals
    ``dct_matrix(q)`` (or its transpose if ``inverse=True``).
    """
    label = "DCT_dag" if inverse else "DCT"

    qc = QuantumCircuit(q + 1, name=label)
    data = list(range(q))
    anc = q
    N = 1 << q
    M = 2 * N

    # 1. Hadamard on ancilla.
    qc.h(anc)

    # 2. Conditional bit-complement on data (anc=1 → flip).
    for d in data:
        qc.cx(anc, d)

    # 3. QFT^{-1} on all q+1 qubits.
    qc.append(QFTGate(q + 1).inverse(), data + [anc])

    # 4. Diagonal phase: per-qubit P(-pi * 2^j / M).
    for j in range(q + 1):
        qc.p(-np.pi * (1 << j) / M, j)

    # 5. Conditional P_neg = increment ∘ bit_complement, controlled by anc=1.
    for d in data:
        qc.cx(anc, d)
    # Conditional incrementer on data: cascade of MCX with anc as extra ctrl.
    for i in reversed(range(q)):
        controls = [anc] + data[:i]
        target = data[i]
        if len(controls) == 1:
            qc.cx(controls[0], target)
        else:
            qc.append(XGate().control(len(controls)), controls + [target])

    # 6. Conditional Ry(pi/2) on ancilla, active when data != 0.
    qc.ry(np.pi / 2, anc)
    qc.append(
        RYGate(-np.pi / 2).control(q, ctrl_state="0" * q),
        data + [anc],
    )

    if inverse:
        return qc.inverse()
    return qc


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


def heat_dct_step_circuit(
    q: int,
    nu: float,
    dt: float,
    L_box: float,
) -> QuantumCircuit:
    """One Trotter layer for Neumann heat eq: DCT + cond-Ry + DCT^T.

    Returns a circuit on q+1 qubits (data[0..q-1] + ancilla[q]) with
    one classical bit for the ancilla measurement.  Mirrors
    heat_qft_step_circuit but with DCT-II in place of QFT.
    """
    theta = compute_theta_dct(nu, dt, q, L_box)

    data = list(range(q))
    anc = q

    qc = QuantumCircuit(q + 1, 1, name="heat_dct_step")
    C = dct_matrix(q)
    qc.append(UnitaryGate(C, label="DCT"), data)
    cond_qc = build_conditional_ry(data, anc, theta)
    qc.compose(cond_qc, inplace=True)
    qc.append(UnitaryGate(C.T, label="DCT_dag"), data)
    qc.measure(anc, 0)
    qc.reset(anc)
    return qc


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

    bc='periodic' uses QFT (Fourier eigenbasis).
    bc='neumann' (= phi-side of u-Dirichlet under CH) dispatches to
    the DCT-II propagator path; see heat_dct_step_circuit.
    """
    if bc == "neumann":
        return heat_dct_step_circuit(q, nu, dt, L_box)
    if bc != "periodic":
        raise NotImplementedError(
            f"heat_qft_step_circuit: unsupported bc={bc!r}; "
            "expected 'periodic' or 'neumann'"
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

    bc='neumann' dispatches to the DCT-II propagator path.
    """
    if bc == "neumann":
        return heat_dct_full_circuit(q, nu, T, N_steps, L_box)
    if bc != "periodic":
        raise NotImplementedError(
            f"heat_qft_full_circuit: unsupported bc={bc!r}; "
            "expected 'periodic' or 'neumann'"
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
    V: np.ndarray | None = None,
) -> QuantumCircuit:
    """One propagator layer via dense eigendecomposition + block encoding.

    Builds exp(nu*L*dt), eigendecomposes, and block-encodes via
    V^H + conditional Ry + V with ancilla post-selection.
    Exact per step (no Trotter error). Tractable for q <= 5.

    V: optional diagonal potential (grid order, length N).  When
    provided the propagator becomes exp(dt*(nu*L - diag(V))).
    """
    N = 1 << q
    dx = L_box / N
    L_dense = build_laplacian_dense(N, dx, bc=bc)
    L_dense = permute_operator(L_dense, q, encoding)

    M = nu * L_dense * dt
    if V is not None:
        V_enc = permute_to_encoding(V, q, encoding) \
            if encoding != "binary" else V
        M = M - np.diag(V_enc) * dt
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
    source_fn=None,
    x: np.ndarray | None = None,
    t_start: float = 0.0,
) -> QuantumCircuit:
    """Full evolution: N_steps dense-block layers.

    If source_fn is given (with grid coords x), each step builds a
    fresh propagator with V(x, t_mid) at the Strang midpoint.
    t_start offsets the global time for chunked evolution.
    """
    from burgers_potential import potential_from_source

    dt = T / N_steps
    data = list(range(q))
    anc = q

    anc_cr = ClassicalRegister(N_steps, "anc_hist")
    data_cr = ClassicalRegister(q, "data")
    qc = QuantumCircuit(q + 1, name="heat_dense_full")
    qc.add_register(anc_cr)
    qc.add_register(data_cr)

    # Pre-build single step_qc when V is constant (unforced).
    if source_fn is None:
        shared_step = heat_dense_block_step_circuit(
            q, nu, dt, L_box, bc=bc, encoding=encoding,
        )
    else:
        shared_step = None

    forced = source_fn is not None
    log_every = max(1, N_steps // 20)
    t0 = time.time()
    print(
        f"[heat_dense_block_full_circuit] building N_steps={N_steps} "
        f"q={q} forced={forced}",
        file=sys.stderr, flush=True,
    )

    for step_idx in range(N_steps):
        if shared_step is not None:
            step_qc = shared_step
        else:
            t_mid = t_start + (step_idx + 0.5) * dt
            V_n = potential_from_source(
                source_fn, x, t_mid, nu, bc=bc,
            )
            step_qc = heat_dense_block_step_circuit(
                q, nu, dt, L_box, bc=bc,
                encoding=encoding, V=V_n,
            )

        qc_layer = step_qc.copy()
        qc_layer.remove_final_measurements()
        qc.compose(qc_layer, inplace=True)
        qc.measure(anc, anc_cr[step_idx])
        if step_idx < N_steps - 1:
            qc.reset(anc)

        if (step_idx + 1) % log_every == 0 or step_idx == N_steps - 1:
            elapsed = time.time() - t0
            rate = (step_idx + 1) / elapsed if elapsed > 0 else 0
            eta = (N_steps - step_idx - 1) / rate if rate > 0 else 0
            print(
                f"[heat_dense_block_full_circuit] step "
                f"{step_idx + 1}/{N_steps} "
                f"depth={qc.depth()} elapsed={elapsed:.1f}s "
                f"rate={rate:.1f}/s eta={eta:.0f}s",
                file=sys.stderr, flush=True,
            )

    for i in range(q):
        qc.measure(data[i], data_cr[i])

    print(
        f"[heat_dense_block_full_circuit] done: depth={qc.depth()} "
        f"size={qc.size()} build_time={time.time() - t0:.1f}s",
        file=sys.stderr, flush=True,
    )
    return qc


# ── LCU full circuit for shots path ─────────────────────────────────


def heat_lcu_full_circuit(
    q: int,
    nu: float,
    T: float,
    N_steps: int,
    L_box: float,
    bc: str = "periodic",
    taylor_order: int = 4,
    source_fn=None,
    x: np.ndarray | None = None,
    t_start: float = 0.0,
) -> QuantumCircuit:
    """Full evolution: N_steps LCU Taylor propagator layers.

    Each layer block-encodes one timestep of the heat propagator
    via truncated Taylor LCU.  Ancilla qubits are measured and
    reset between steps; post-select on all-zero ancilla history.

    If source_fn is given (with grid coords x), each step builds a
    fresh Strang-split propagator with V(x, t_mid).  t_start
    offsets the global time for chunked evolution.

    Returns circuit on q + n_anc qubits with the appropriate
    number of classical bits.
    """
    from burgers_lcu import (
        heat_lcu_step_circuit,
        heat_lcu_with_potential_step_circuit,
    )
    from burgers_potential import potential_from_source

    dt = T / N_steps
    forced = source_fn is not None

    # Build first step to determine ancilla count and layout.
    if not forced:
        step_qc, lam = heat_lcu_step_circuit(
            q, nu, dt, L_box, bc=bc,
            taylor_order=taylor_order,
        )
        shared_step = step_qc
    else:
        t_mid = t_start + 0.5 * dt
        V_0 = potential_from_source(
            source_fn, x, t_mid, nu, bc=bc,
        )
        step_qc, lam = heat_lcu_with_potential_step_circuit(
            q, nu, dt, L_box, V_0,
            taylor_order=taylor_order,
        )
        shared_step = None

    n_anc = step_qc.num_qubits - q
    data = list(range(q))
    anc_qubits = list(range(q, q + n_anc))

    anc_cr = ClassicalRegister(N_steps * n_anc, "anc_hist")
    data_cr = ClassicalRegister(q, "data")
    qc = QuantumCircuit(q + n_anc, name="heat_lcu_full")
    qc.add_register(anc_cr)
    qc.add_register(data_cr)
    # Track per-step lambdas so consumers can see the full series,
    # not just the last one.  In the unforced case lam is constant
    # and lcu_lambda_history has length 1.
    lambda_history: list[float] = [lam]
    qc.metadata = {
        "lcu_lambda": lam,
        "lcu_lambda_history": lambda_history,
    }

    log_every = max(1, N_steps // 20)
    t0 = time.time()
    print(
        f"[heat_lcu_full_circuit] building N_steps={N_steps} "
        f"q={q} forced={forced}",
        file=sys.stderr, flush=True,
    )

    for step_idx in range(N_steps):
        if shared_step is not None:
            cur_step = shared_step
        else:
            t_mid = t_start + (step_idx + 0.5) * dt
            V_n = potential_from_source(
                source_fn, x, t_mid, nu, bc=bc,
            )
            cur_step, lam_n = (
                heat_lcu_with_potential_step_circuit(
                    q, nu, dt, L_box, V_n,
                    taylor_order=taylor_order,
                )
            )
            # First step's lam was already pushed at construction
            # above (using V_0); subsequent steps append.
            if step_idx > 0:
                lambda_history.append(lam_n)
            qc.metadata["lcu_lambda_max"] = max(lambda_history)
            qc.metadata["lcu_lambda_mean"] = (
                sum(lambda_history) / len(lambda_history)
            )

        qc.compose(cur_step, inplace=True)
        for ai, aq in enumerate(anc_qubits):
            qc.measure(aq, anc_cr[step_idx * n_anc + ai])
        if step_idx < N_steps - 1:
            for aq in anc_qubits:
                qc.reset(aq)

        if (
            (step_idx + 1) % log_every == 0
            or step_idx == N_steps - 1
        ):
            elapsed = time.time() - t0
            rate = (step_idx + 1) / elapsed if elapsed > 0 else 0
            print(
                f"[heat_lcu_full_circuit] step "
                f"{step_idx + 1}/{N_steps} "
                f"elapsed={elapsed:.1f}s rate={rate:.1f}/s",
                file=sys.stderr, flush=True,
            )

    for i in range(q):
        qc.measure(data[i], data_cr[i])

    print(
        f"[heat_lcu_full_circuit] done: depth={qc.depth()} "
        f"size={qc.size()} build_time={time.time() - t0:.1f}s",
        file=sys.stderr, flush=True,
    )
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
    V: np.ndarray | None = None,
    taylor_order: int = 4,
) -> QuantumCircuit:
    """Build a measurement-free step circuit for SV simulation.

    V: optional diagonal potential (grid order, length N) for
    forced Burgers.  Only supported by dense-block propagator.
    taylor_order: truncation order for LCU Taylor propagator.
    """
    if propagator == "qft-diagonal":
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
    elif propagator == "dense-block":
        N = 1 << q
        dx = L_box / N
        L_dense = build_laplacian_dense(N, dx, bc=bc)
        L_dense = permute_operator(L_dense, q, encoding)
        M = nu * L_dense * dt
        if V is not None:
            V_enc = permute_to_encoding(V, q, encoding) \
                if encoding != "binary" else V
            M = M - np.diag(V_enc) * dt
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
    elif propagator == "lcu":
        from burgers_lcu import (
            heat_lcu_step_circuit,
            heat_lcu_with_potential_step_circuit,
        )
        if V is None:
            lcu_qc, lam = heat_lcu_step_circuit(
                q, nu, dt, L_box, bc=bc,
                taylor_order=taylor_order,
            )
            lcu_qc.metadata = {"lcu_lambda": lam}
        else:
            lcu_qc, lam = heat_lcu_with_potential_step_circuit(
                q, nu, dt, L_box, V,
                taylor_order=taylor_order,
            )
            # Metadata already set by builder
        return lcu_qc
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
    source_fn=None,
    x: np.ndarray | None = None,
    taylor_order: int = 4,
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
    source_fn: callable(x, t) -> g(x,t) or None for unforced.
    x: grid coordinates (required when source_fn is not None).

    Returns (psi_snapshots, metrics_list) where each psi snapshot
    is the unit-norm data-register state after post-selection.
    """
    from qiskit.quantum_info import Operator, Statevector
    from burgers_potential import potential_from_source

    N = 1 << q

    # Pre-build step circuit when V is constant (unforced).
    if source_fn is None:
        shared_step = _build_step_sv(
            q, nu, dt, L_box, bc, propagator,
            encoding=encoding, taylor_order=taylor_order,
        )
    else:
        shared_step = None

    # Ancilla count depends on propagator (1 for qft/dense, m for LCU).
    if shared_step is not None:
        n_anc = shared_step.num_qubits - q
    else:
        # Will be determined from first step circuit.
        n_anc = 1
    sv_dim = N * (1 << n_anc)
    sv = np.zeros(sv_dim, dtype=complex)
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

    # For LCU the circuit contains multi-controlled custom gates;
    # Statevector.evolve(qc) decomposes them one-by-one which is
    # extremely slow.  Convert to a dense unitary matrix once, then
    # use fast matrix-vector multiply.  At q=5 + ancillas the matrix
    # is only 2048x2048 — negligible memory.
    _use_dense = propagator == "lcu"

    if shared_step is not None and _use_dense:
        t0_op = time.time()
        shared_mat = Operator(shared_step).data
        print(
            f"[cole_hopf_circuit_sv] pre-built shared unitary "
            f"{shared_mat.shape} in {time.time() - t0_op:.1f}s",
            file=sys.stderr, flush=True,
        )
    else:
        shared_mat = None

    _step_metadata = (
        shared_step.metadata if shared_step is not None else None
    )
    _sv_log_every = max(1, n_steps // 10)
    _sv_t0 = time.time()

    for step in range(n_steps):
        if shared_step is not None:
            step_qc = shared_step
            step_mat = shared_mat
        else:
            t_mid = (step + 0.5) * dt
            V_n = potential_from_source(
                source_fn, x, t_mid, nu, bc=bc,
            )
            step_qc = _build_step_sv(
                q, nu, dt, L_box, bc, propagator,
                encoding=encoding, V=V_n,
                taylor_order=taylor_order,
            )
            _step_metadata = step_qc.metadata
            if step == 0:
                n_anc = step_qc.num_qubits - q
                sv_dim_new = N * (1 << n_anc)
                if sv_dim_new != len(sv):
                    sv_new = np.zeros(sv_dim_new, dtype=complex)
                    sv_new[:N] = sv[:N]
                    sv = sv_new
            if _use_dense:
                step_mat = Operator(step_qc).data
            else:
                step_mat = None

        if step_mat is not None:
            sv = step_mat @ sv
        else:
            sv = Statevector(sv).evolve(step_qc).data
        p0 = float(np.sum(np.abs(sv[:N]) ** 2))
        p_success_total *= p0
        sv_proj = np.zeros_like(sv)
        if p0 > 1e-30:
            sv_proj[:N] = sv[:N] / np.sqrt(p0)
        sv = sv_proj

        if (
            _use_dense
            and ((step + 1) % _sv_log_every == 0
                 or step == n_steps - 1)
        ):
            elapsed = time.time() - _sv_t0
            print(
                f"[cole_hopf_circuit_sv] step {step + 1}/{n_steps} "
                f"p_success={p0:.4f} elapsed={elapsed:.1f}s",
                file=sys.stderr, flush=True,
            )

        step_metrics: dict[str, Any] = {
            "step": step + 1,
            "p_success_step": p0,
            "p_success_total": p_success_total,
        }
        # Surface LCU normalization when present; informational for
        # noise-budget tuning across (q, nu, dt, taylor_order).
        if _step_metadata and "lcu_lambda" in _step_metadata:
            step_metrics["lcu_lambda"] = (
                _step_metadata["lcu_lambda"]
            )
            for k in ("lcu_lambda_heat", "lcu_s_max_V"):
                if k in _step_metadata:
                    step_metrics[k] = _step_metadata[k]
        metrics.append(step_metrics)

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


# ── Chunked shots driver ─────────────────────────────────────────────


def _run_shots_chunked(
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
    chunk_size: int,
    bond_dim: int | None = None,
    encoding: str = "binary",
    backend: Any = None,
    backend_type: str = "sim",
    backend_name: str | None = None,
    optimization_level: int = 1,
    seed: int | None = None,
    source_fn=None,
    x: np.ndarray | None = None,
    taylor_order: int = 4,
    use_mps_prep: bool = True,
) -> list[tuple[np.ndarray, dict[str, Any]]]:
    """Chunked evolution: K chunks of chunk_size steps each.

    Between chunks, classically read out post-selected amplitudes
    and re-prep as fresh IC.  No classical PDE physics — only
    amplitude IO (decode counts -> re-encode via MPS prep).

    use_mps_prep: True (default) uses Ran 2020 MPS-to-circuit prep
        (O(q*chi^2) gates).  False falls back to QuantumCircuit.
        initialize (O(2^q) gates) for comparison.
    """
    if backend_type == "hardware":
        raise NotImplementedError(
            "chunked evolution v1: sim only; hardware "
            "chunking deferred to v2"
        )

    from q8020_cfd_qutil.circuit import (
        transpile_circuit,
        execute_circuit_counts,
    )
    from burgers_mps import (
        classical_to_mps, mps_to_circuit, normalize_state,
    )

    N = 1 << q
    n_steps_total = max(snap_steps)
    n_chunks = n_steps_total // chunk_size

    psi_current, init_norm = normalize_state(psi0)
    cumulative_norm = init_norm * phi_norm
    snapshots: dict[int, tuple[np.ndarray, dict[str, Any]]] = {}

    print(
        f"[_run_shots_chunked] n_chunks={n_chunks} "
        f"chunk_size={chunk_size} shots={shots}",
        file=sys.stderr, flush=True,
    )
    t_total_start = time.time()

    for chunk_idx in range(n_chunks):
        t_chunk = time.time()
        global_step_start = chunk_idx * chunk_size
        t_start_chunk = global_step_start * dt

        # 1. Build prep circuit from current amplitudes
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

        # 2. Build chunk_size-step evolution circuit
        T_chunk = dt * chunk_size
        if propagator == "qft-diagonal":
            full_qc = heat_qft_full_circuit(
                q, nu, T_chunk, chunk_size, L_box, bc=bc,
            )
        elif propagator == "lcu":
            full_qc = heat_lcu_full_circuit(
                q, nu, T_chunk, chunk_size, L_box, bc=bc,
                taylor_order=taylor_order,
                source_fn=source_fn, x=x,
                t_start=t_start_chunk,
            )
        else:
            full_qc = heat_dense_block_full_circuit(
                q, nu, T_chunk, chunk_size, L_box, bc=bc,
                encoding=encoding,
                source_fn=source_fn, x=x,
                t_start=t_start_chunk,
            )

        n_heat_anc = full_qc.num_qubits - q
        total_q = q + n_bond + n_heat_anc
        heat_qubit_map = list(range(q)) + list(
            range(q + n_bond, q + n_bond + n_heat_anc),
        )

        # 3. Compose
        init_qc = QuantumCircuit(total_q)
        init_qc.compose(
            prep_qc, qubits=list(range(q + n_bond)),
            inplace=True,
        )
        raw_qc = init_qc.compose(
            full_qc, qubits=heat_qubit_map,
        )
        # Compose drops source metadata; carry forward (LCU lambda).
        if full_qc.metadata:
            raw_qc.metadata = dict(full_qc.metadata)

        # 4. Transpile + execute
        #    AerSimulator handles arbitrary gates (ControlledGate,
        #    StatePreparation, UnitaryGate) natively.  Skipping
        #    transpile avoids a Qiskit qs_decomposition segfault
        #    on the multi-controlled SELECT gates in the LCU circuit.
        try:
            from qiskit_aer import AerSimulator as _Aer
            _skip_transpile = (
                isinstance(backend, _Aer)
                and propagator == "lcu"
            )
        except ImportError:
            _skip_transpile = False

        print(
            f"[_run_shots_chunked] chunk {chunk_idx + 1}/"
            f"{n_chunks} steps "
            f"{global_step_start+1}-"
            f"{global_step_start+chunk_size} "
            f"depth={raw_qc.depth()}"
            f"{' (skip transpile — Aer+LCU)' if _skip_transpile else ' transpiling'} ...",
            file=sys.stderr, flush=True,
        )
        if _skip_transpile:
            qc_t = raw_qc
            t_info = {
                "wall_time": 0.0,
                "optimization_level": optimization_level,
                "skipped": "AerSimulator+LCU",
            }
        else:
            qc_t, t_info = transpile_circuit(
                raw_qc, backend,
                optimization_level=optimization_level,
                seed_transpiler=seed,
            )
        counts, exec_info = execute_circuit_counts(
            qc_t, backend, shots=shots, seed=seed,
        )

        # 5. Post-select and reconstruct
        n_kept, data_counts = post_select_counts(counts, q)
        p_success = n_kept / shots if shots > 0 else 0.0

        step_at_end = (chunk_idx + 1) * chunk_size

        if n_kept == 0:
            print(
                f"[_run_shots_chunked] chunk {chunk_idx + 1}: "
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

        elapsed = time.time() - t_chunk
        print(
            f"[_run_shots_chunked] chunk {chunk_idx + 1}/"
            f"{n_chunks} done in {elapsed:.1f}s "
            f"p_success={p_success:.4f} "
            f"cumulative_norm={cumulative_norm:.4e}",
            file=sys.stderr, flush=True,
        )

        if step_at_end in snap_steps:
            # Scale for output: phi = psi * cumulative_norm
            phi_out = psi_current * cumulative_norm
            met: dict[str, Any] = {
                "shots": shots,
                "n_kept": n_kept,
                "p_success": p_success,
                "n_steps": step_at_end,
                "step": step_at_end,
                "chunk_idx": chunk_idx,
                "chunk_size": chunk_size,
                "cumulative_norm": cumulative_norm,
                "transpile": t_info,
                "execute": exec_info,
            }
            if raw_qc.metadata and "lcu_lambda" in raw_qc.metadata:
                met["lcu_lambda"] = raw_qc.metadata["lcu_lambda"]
            snapshots[step_at_end] = (phi_out, met)

    total_elapsed = time.time() - t_total_start
    print(
        f"[_run_shots_chunked] all {n_chunks} chunks "
        f"done in {total_elapsed:.1f}s",
        file=sys.stderr, flush=True,
    )
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
    propagator: str,
    shots: int,
    bond_dim: int | None = None,
    encoding: str = "binary",
    backend: Any = None,
    backend_type: str = "sim",
    backend_name: str | None = None,
    optimization_level: int = 1,
    seed: int | None = None,
    source_fn=None,
    x: np.ndarray | None = None,
    taylor_order: int = 4,
    use_mps_prep: bool = True,
) -> list[tuple[np.ndarray, dict[str, Any]]]:
    """Batch shots readout per spec §7 (P-C optimised).

    Builds one circuit per snap_step, transpiles all in a single
    ``transpile([...], backend)`` call, then runs each.  One
    AerSimulator instantiation and one transpile invocation total.

    State preparation uses the Ran 2020 MPS-to-circuit pipeline
    (``burgers_mps.classical_to_mps`` + ``mps_to_circuit``), matching
    Gopalakrishnan Meena AIAA-2026 Eqs. 5-6 plus Ran 2020 (Ref [27]).
    ``bond_dim=None`` means full-rank (no truncation); finite values
    truncate the MPS.

    use_mps_prep: True (default) uses the Ran 2020 MPS-to-circuit
        path; False falls back to QuantumCircuit.initialize for
        comparison with the dense O(2^q) state-prep baseline.

    Returns list of (phi_reconstructed, metrics) per snap_step.
    """
    from q8020_cfd_qutil.circuit import execute_circuit_counts

    from burgers_mps import (
        classical_to_mps, mps_to_circuit, normalize_state,
    )

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
        f"for snap_steps={list(snap_steps)} "
        f"propagator={propagator} shots={shots}",
        file=sys.stderr, flush=True,
    )
    t_build_start = time.time()
    raw_circs: list[QuantumCircuit] = []
    total_q = None
    for s in snap_steps:
        if propagator == "qft-diagonal":
            full_qc = heat_qft_full_circuit(
                q, nu, dt * s, s, L_box, bc=bc,
            )
        elif propagator == "lcu":
            full_qc = heat_lcu_full_circuit(
                q, nu, dt * s, s, L_box, bc=bc,
                taylor_order=taylor_order,
                source_fn=source_fn, x=x,
            )
        else:
            full_qc = heat_dense_block_full_circuit(
                q, nu, dt * s, s, L_box, bc=bc,
                encoding=encoding,
                source_fn=source_fn, x=x,
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
        raw_qc = init_qc.compose(full_qc, qubits=heat_qubit_map)
        # Compose drops source metadata; carry it forward explicitly.
        if full_qc.metadata:
            raw_qc.metadata = dict(full_qc.metadata)
        raw_circs.append(raw_qc)
    print(
        f"[_run_shots_batch] built {len(raw_circs)} circuit(s) in "
        f"{time.time() - t_build_start:.1f}s "
        f"(total qubits={total_q})",
        file=sys.stderr, flush=True,
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

    # --- Single batched transpile of all circuits (P-C: F10-5) --------
    # Pattern 1 from F10-REVIEW-PATCH §P-C: build the cumulative family
    # {circuit_1 .. circuit_n_steps} once, transpile in one
    # transpile([...], backend) call, then run each.  One AerSimulator()
    # and one transpile() per run_cole_hopf_circuit_simulation invocation.
    try:
        from qiskit_aer import AerSimulator as _Aer
        _skip_transpile = (
            isinstance(backend, _Aer)
            and propagator == "lcu"
        )
    except ImportError:
        _skip_transpile = False

    if _skip_transpile:
        qc_t_list: list[QuantumCircuit] = list(raw_circs)
        t_info_list: list[dict[str, Any]] = [
            {
                "wall_time": 0.0,
                "optimization_level": optimization_level,
                "skipped": "AerSimulator+LCU",
            }
            for _ in raw_circs
        ]
        print(
            f"[_run_shots_batch] transpile skipped (Aer+LCU); "
            f"running {len(raw_circs)} circuit(s) ...",
            file=sys.stderr, flush=True,
        )
    else:
        from qiskit import transpile as _qiskit_transpile
        from q8020_cfd_qutil.circuit import get_circuit_info

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
        }
        # Surface LCU normalization when the propagator is LCU.
        if raw_qc.metadata and "lcu_lambda" in raw_qc.metadata:
            met["lcu_lambda"] = raw_qc.metadata["lcu_lambda"]
        results.append((phi_hat, met))

    return results


# ── F10-12: Hadamard-test-per-bin readout ───────────────────────────


def _build_evolution_qc_unitary(
    q: int,
    nu: float,
    dt: float,
    n_steps: int,
    L_box: float,
    bc: str,
    propagator: str,
    encoding: str,
    source_fn,
    x: np.ndarray | None,
    taylor_order: int,
) -> tuple[QuantumCircuit, int]:
    """Build measurement-free n_steps evolution on q + n_steps*a qubits.

    Each step uses a fresh ancilla register (no mid-circuit measure /
    reset) so the resulting circuit is a unitary suitable for use
    inside a Hadamard test.  Post-selecting all ancillas to |0> at
    the end is mathematically equivalent to the per-step reset path.

    Returns (qc, n_anc_per_step).
    """
    from burgers_potential import potential_from_source

    forced = source_fn is not None

    if not forced:
        step0 = _build_step_sv(
            q, nu, dt, L_box, bc, propagator,
            encoding=encoding, taylor_order=taylor_order,
        )
    else:
        t_mid0 = 0.5 * dt
        V_0 = potential_from_source(
            source_fn, x, t_mid0, nu, bc=bc,
        )
        step0 = _build_step_sv(
            q, nu, dt, L_box, bc, propagator,
            encoding=encoding, V=V_0,
            taylor_order=taylor_order,
        )

    n_anc_per_step = step0.num_qubits - q
    n_total = q + n_steps * n_anc_per_step
    qc = QuantumCircuit(n_total, name="evolution_unitary")

    for s in range(n_steps):
        if s == 0:
            step_qc = step0
        elif forced:
            t_mid = (s + 0.5) * dt
            V_n = potential_from_source(
                source_fn, x, t_mid, nu, bc=bc,
            )
            step_qc = _build_step_sv(
                q, nu, dt, L_box, bc, propagator,
                encoding=encoding, V=V_n,
                taylor_order=taylor_order,
            )
        else:
            step_qc = step0

        anc_start = q + s * n_anc_per_step
        qubit_map = (
            list(range(q))
            + list(range(anc_start, anc_start + n_anc_per_step))
        )
        qc.compose(step_qc, qubits=qubit_map, inplace=True)

    return qc, n_anc_per_step


def hadamard_per_bin_circuit(
    q: int,
    prep_qc: QuantumCircuit,
    n_bond: int,
    evo_qc: QuantumCircuit,
    n_anc_evo: int,
    k: int,
) -> QuantumCircuit:
    """Hadamard test for Re(<k|U|psi0>) on basis index k.

    Layout (qubit indices):
      data:     0 .. q-1
      bond:     q .. q + n_bond - 1            (state-prep ancillas)
      evo_anc:  q + n_bond .. q + n_bond + n_anc_evo - 1
      test:     last qubit

    The inner unitary V = X_k * evo * prep is wrapped in a single
    controlled gate whose control is the test qubit.  After two
    Hadamards on the test qubit, the joint outcome statistics give

        Re(psi_k) = P(test=0, data=0, bond=0, evo=0)
                  - P(test=1, data=0, bond=0, evo=0)

    where psi is the (unnormalised) post-selected amplitude vector.
    The post-selection factor is absorbed naturally — no extra
    sqrt(p_success) rescaling is needed.
    """
    n_inner = q + n_bond + n_anc_evo
    n_total = n_inner + 1
    test = n_total - 1

    inner = QuantumCircuit(n_inner, name=f"W_k{k}")
    inner.compose(
        prep_qc, qubits=list(range(q + n_bond)), inplace=True,
    )
    evo_qubits = (
        list(range(q))
        + list(range(q + n_bond, q + n_bond + n_anc_evo))
    )
    inner.compose(evo_qc, qubits=evo_qubits, inplace=True)
    for i in range(q):
        if (k >> i) & 1:
            inner.x(i)

    qc = QuantumCircuit(n_total, n_total)
    qc.h(test)
    controlled = inner.to_gate().control(1)
    qc.append(controlled, [test] + list(range(n_inner)))
    qc.h(test)
    qc.measure(list(range(n_total)), list(range(n_total)))
    return qc


def extract_hadamard_per_bin_amplitudes(
    counts_per_k: list[dict[str, int]],
    q: int,
    n_bond: int,
    n_anc_evo: int,
    shots: int,
    phi_norm: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reconstruct phi from per-bin Hadamard-test counts.

    counts_per_k[k]: counts dict from running the k-th circuit.
    Bitstring layout (Qiskit little-endian): bit 0 (rightmost char)
    is data qubit 0; the leftmost char is the test qubit.  Returns
    (phi_hat, info) where phi_hat is real-valued (phi_norm * Re psi_k)
    and info carries aggregate diagnostics.
    """
    N = 1 << q
    n_inner = q + n_bond + n_anc_evo
    re_psi = np.zeros(N)
    n_kept_total = 0
    n_test_zero = 0
    n_test_one = 0

    for k, counts in enumerate(counts_per_k):
        n0 = 0
        n1 = 0
        for bitstr, cnt in counts.items():
            bits = bitstr.replace(" ", "")
            if len(bits) < n_inner + 1:
                continue
            test_bit = int(bits[0])
            data_bits = bits[-q:] if q > 0 else ""
            anc_bits = bits[1:-q] if q > 0 else bits[1:]
            data_idx = int(data_bits, 2) if data_bits else 0
            if data_idx != 0:
                continue
            if not all(c == "0" for c in anc_bits):
                continue
            if test_bit == 0:
                n0 += cnt
            else:
                n1 += cnt

        denom = max(shots, 1)
        re_psi[k] = (n0 - n1) / denom
        n_kept_total += n0 + n1
        n_test_zero += n0
        n_test_one += n1

    phi_hat = phi_norm * re_psi
    info: dict[str, Any] = {
        "n_kept_total": n_kept_total,
        "n_test_zero": n_test_zero,
        "n_test_one": n_test_one,
        "p_kept_per_bin_mean": (
            n_kept_total / (len(counts_per_k) * shots)
            if counts_per_k and shots > 0 else 0.0
        ),
    }
    return phi_hat, info


def _run_shots_hadamard_per_bin(
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
    source_fn=None,
    x: np.ndarray | None = None,
    taylor_order: int = 4,
    use_mps_prep: bool = True,
) -> list[tuple[np.ndarray, dict[str, Any]]]:
    """Hadamard-test-per-bin readout driver (F10-12).

    For each snap_step s, builds N Hadamard-test circuits (one per
    basis index k) wrapping the full state-prep + s-step evolution,
    runs all of them, and reconstructs a real-valued phi_hat by
    differencing test-ancilla outcomes per bin.

    Aimed at the low-nu shots regime where direct amplitude readout
    has poor signal-to-noise: the Hadamard test recovers Re(psi_k)
    (signed) in linear shots per bin, at the cost of N circuits per
    snap_step.
    """
    if backend_type == "hardware":
        raise NotImplementedError(
            "hadamard_per_bin readout: hardware backends not "
            "supported yet (would need O(N) job submissions)."
        )

    from q8020_cfd_qutil.circuit import (
        transpile_circuit,
        execute_circuit_counts,
    )
    from burgers_mps import (
        classical_to_mps, mps_to_circuit, normalize_state,
    )

    N = 1 << q

    psi_normed, _ = normalize_state(psi0)
    if use_mps_prep:
        tensors = classical_to_mps(
            psi_normed, bond_dim=bond_dim, canonical="right",
        )
        prep_qc = mps_to_circuit(tensors)
        n_bond = prep_qc.num_qubits - q
    else:
        prep_qc = QuantumCircuit(q, name="initialize_prep")
        prep_qc.initialize(psi_normed.tolist(), range(q))
        n_bond = 0

    results: list[tuple[np.ndarray, dict[str, Any]]] = []

    for s in snap_steps:
        t_build = time.time()
        evo_qc, n_anc_evo = _build_evolution_qc_unitary(
            q, nu, dt, s, L_box, bc, propagator,
            encoding, source_fn, x, taylor_order,
        )
        print(
            f"[_run_shots_hadamard_per_bin] snap_step={s} "
            f"q={q} n_bond={n_bond} n_anc_evo={n_anc_evo} "
            f"building {N} Hadamard circuits ...",
            file=sys.stderr, flush=True,
        )

        per_k_circs: list[QuantumCircuit] = []
        for k in range(N):
            qc_k = hadamard_per_bin_circuit(
                q, prep_qc, n_bond, evo_qc, n_anc_evo, k,
            )
            per_k_circs.append(qc_k)
        print(
            f"[_run_shots_hadamard_per_bin] built {N} circuits in "
            f"{time.time() - t_build:.1f}s",
            file=sys.stderr, flush=True,
        )

        counts_per_k: list[dict[str, int]] = []
        t_info_list: list[dict[str, Any]] = []
        exec_info_list: list[dict[str, Any]] = []
        t_run = time.time()
        for k, qc_k in enumerate(per_k_circs):
            qc_t, t_info = transpile_circuit(
                qc_k, backend,
                optimization_level=optimization_level,
                seed_transpiler=seed,
            )
            counts, exec_info = execute_circuit_counts(
                qc_t, backend, shots=shots, seed=seed,
            )
            counts_per_k.append(counts)
            t_info_list.append(t_info)
            exec_info_list.append(exec_info)
            if (k + 1) % max(1, N // 8) == 0 or k == N - 1:
                print(
                    f"[_run_shots_hadamard_per_bin] snap_step={s} "
                    f"bin {k + 1}/{N} elapsed="
                    f"{time.time() - t_run:.1f}s",
                    file=sys.stderr, flush=True,
                )

        phi_hat, agg = extract_hadamard_per_bin_amplitudes(
            counts_per_k, q, n_bond, n_anc_evo, shots, phi_norm,
        )

        met: dict[str, Any] = {
            "shots": shots,
            "shots_per_bin": shots,
            "n_bins": N,
            "n_steps": s,
            "step": s,
            "readout": "hadamard_per_bin",
            "n_kept_total": agg["n_kept_total"],
            "n_test_zero": agg["n_test_zero"],
            "n_test_one": agg["n_test_one"],
            "p_kept_per_bin_mean": agg["p_kept_per_bin_mean"],
            "p_success": agg["p_kept_per_bin_mean"],
            "transpile_wall_total": sum(
                ti.get("wall_time", 0.0) for ti in t_info_list
            ),
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
    source_fn=None,
    evolution_mode: str = "single",
    chunk_size: int = 10,
    phi_modes: int = 0,
    taylor_order: int = 4,
    use_mps_prep: bool = True,
    readout: str = "direct",
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
    from burgers_cole_hopf import (
        cole_hopf_forward,
        cole_hopf_forward_centered,
        cole_hopf_inverse,
        fourier_low_pass_phi,
        log_phi_to_normalized_psi,
        _should_center,
    )

    N = len(u0)
    q = int(np.log2(N))
    dx = x[1] - x[0]
    L_box = float(N * dx)

    # §8: Dirichlet u=0 maps to Neumann dphi/dx=0 on phi.
    phi_bc = "neumann" if bc == "dirichlet" else bc
    # qft-diagonal + neumann uses the DCT-II propagator (see
    # heat_dct_step_circuit); periodic uses QFT.  No NotImplementedError.
    if phi_bc not in ("periodic", "neumann") and propagator == "qft-diagonal":
        raise NotImplementedError(
            f"qft-diagonal supports periodic and dirichlet (Neumann-on-phi);"
            f" got bc={bc!r}"
        )

    # Source guard: V is position-diagonal, L is Fourier-diagonal;
    # they don't share an eigenbasis.
    if source_fn is not None and propagator not in (
        "dense-block", "lcu",
    ):
        raise NotImplementedError(
            "Source forcing requires --propagator dense-block "
            "or lcu in this release. qft-diagonal+source is "
            "on the roadmap."
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
    psi0 = permute_to_encoding(psi0, q, encoding)

    print(
        f"[cole_hopf_circuit] propagator={propagator} q={q} "
        f"encoding={encoding} nu={nu:.1e} dt={dt:.4e} "
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

        if readout not in ("direct", "hadamard_per_bin"):
            raise ValueError(
                f"unknown readout={readout!r}; expected "
                f"'direct' or 'hadamard_per_bin'"
            )
        if readout == "hadamard_per_bin" and (
            evolution_mode == "chunked"
        ):
            raise NotImplementedError(
                "hadamard_per_bin readout requires "
                "evolution_mode=single (chunked v1 unsupported)"
            )

        if readout == "hadamard_per_bin":
            batch_results = _run_shots_hadamard_per_bin(
                psi0, q, nu, dt, snap_steps, L_box,
                phi_norm,
                bc=phi_bc, propagator=propagator,
                shots=shots,
                bond_dim=bond_dim, encoding=encoding,
                backend=backend, backend_type=backend_type,
                backend_name=backend_name,
                optimization_level=optimization_level,
                seed=seed,
                source_fn=source_fn, x=x,
                taylor_order=taylor_order,
                use_mps_prep=use_mps_prep,
            )
        elif evolution_mode == "chunked":
            # Validate alignment
            if n_steps % chunk_size != 0:
                raise ValueError(
                    f"chunked mode requires n_steps ({n_steps}) "
                    f"divisible by chunk_size ({chunk_size})"
                )
            bad = [s for s in snap_steps
                   if s % chunk_size != 0]
            if bad:
                raise ValueError(
                    f"chunked mode: snap_steps {bad} not "
                    f"aligned to chunk_size={chunk_size}; "
                    f"set --save-every to a multiple of "
                    f"--chunk-size"
                )
            batch_results = _run_shots_chunked(
                psi0, q, nu, dt, snap_steps, L_box,
                phi_norm,
                bc=phi_bc, propagator=propagator,
                shots=shots, chunk_size=chunk_size,
                bond_dim=bond_dim, encoding=encoding,
                backend=backend, backend_type=backend_type,
                backend_name=backend_name,
                optimization_level=optimization_level,
                seed=seed,
                source_fn=source_fn, x=x,
                taylor_order=taylor_order,
                use_mps_prep=use_mps_prep,
            )
        else:
            batch_results = _run_shots_batch(
                psi0, q, nu, dt, snap_steps, L_box,
                phi_norm,
                bc=phi_bc, propagator=propagator,
                shots=shots,
                bond_dim=bond_dim, encoding=encoding,
                backend=backend, backend_type=backend_type,
                backend_name=backend_name,
                optimization_level=optimization_level,
                seed=seed,
                source_fn=source_fn, x=x,
                taylor_order=taylor_order,
                use_mps_prep=use_mps_prep,
            )

        for (phi_hat, met), s in zip(batch_results, snap_steps):
            phi_hat = permute_from_encoding(
                phi_hat, q, encoding,
            )
            if phi_modes > 0:
                phi_hat = fourier_low_pass_phi(
                    phi_hat, phi_modes,
                )
            u_snap = cole_hopf_inverse(phi_hat, dx, nu)
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
        bc=phi_bc, propagator=propagator,
        snapshot_interval=snapshot_interval,
        use_mps_prep=use_mps_prep,
        bond_dim=bond_dim, encoding=encoding,
        source_fn=source_fn, x=x,
        taylor_order=taylor_order,
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
