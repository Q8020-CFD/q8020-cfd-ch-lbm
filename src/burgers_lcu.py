"""LCU (Linear Combination of Unitaries) SELECT/PREPARE primitives.

Implements the generic LCU block-encoding machinery and a truncated
Taylor LCU for the heat propagator exp(nu*L*dt), used as a propagator
option for the Cole-Hopf circuit path (--propagator lcu).

References:
  - Berry, Childs, Cleve, Kothari, Somma (2015) for LCU framework
  - SPEC-F3-LCU-method.md §4, §5.1, §5.3
"""

from __future__ import annotations

import math
from itertools import product
from typing import Any

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RYGate
from qiskit.quantum_info import Operator

from burgers_mpo import (
    extract_block_encoded_operator,
    increment_circuit,
    decrement_circuit,
    shift_matrix,
)


# ── Generic LCU primitives ──────────────────────────────────────────


def build_prepare_circuit(
    coefficients: np.ndarray,
    n_ancilla: int,
) -> QuantumCircuit:
    """PREPARE: |0>^m -> sum_k alpha_k |k> on m ancilla qubits.

    alpha_k = sqrt(|c_k| / lambda) where lambda = sum |c_k|.
    Signs of c_k are NOT absorbed here; the caller must wrap the
    corresponding U_k with a phase flip for negative c_k.

    Unused ancilla states (K < 2^m) get zero amplitude.
    Uses Qiskit StatePreparation for the ancilla register.
    """
    from qiskit.circuit.library import StatePreparation

    K = len(coefficients)
    M = 1 << n_ancilla
    lam = float(np.sum(np.abs(coefficients)))
    if lam < 1e-30:
        raise ValueError("All LCU coefficients are zero")

    alphas = np.zeros(M)
    for k in range(K):
        alphas[k] = math.sqrt(abs(coefficients[k]) / lam)
    # By construction sum(alphas^2) = sum(|c_k|)/lam = 1 exactly
    # (modulo floating point).  Assert rather than silently
    # renormalize: if this ever drifts, the block-encoded operator
    # would be A/(lam*norm^2) instead of A/lam and lam returned to
    # the caller would be inconsistent with the actual encoding.
    nrm_sq = float(np.sum(alphas ** 2))
    assert abs(nrm_sq - 1.0) < 1e-10, (
        f"PREPARE amplitudes not normalized: |alpha|^2 = {nrm_sq}"
    )

    qc = QuantumCircuit(n_ancilla, name="PREPARE")
    qc.append(StatePreparation(alphas), list(range(n_ancilla)))
    return qc


def build_select_circuit(
    operators: list[QuantumCircuit],
    coefficients: np.ndarray,
    n_system: int,
    n_ancilla: int,
) -> QuantumCircuit:
    """SELECT: |k>|psi> -> |k> sign(c_k) U_k |psi>.

    Applies controlled-U_k when ancilla register equals k.
    Negative c_k are handled by wrapping U_k with a global Z phase.
    """
    K = len(operators)
    n_total = n_system + n_ancilla
    qc = QuantumCircuit(n_total, name="SELECT")

    sys_qubits = list(range(n_system))
    anc_qubits = list(range(n_system, n_total))

    for k in range(K):
        if abs(coefficients[k]) < 1e-30:
            continue

        op_k = operators[k].copy()

        # Absorb sign: if c_k < 0, prepend a global phase of -1.
        if coefficients[k] < 0:
            # Apply Z to first qubit, then X-Z-X to flip sign globally
            # Simpler: wrap in a phase gate.  For block-encoding the
            # sign just needs to flip the operator, so we prepend a
            # diagonal -I via two X gates sandwiching a Z on qubit 0.
            wrapper = QuantumCircuit(op_k.num_qubits, name=f"neg_U{k}")
            wrapper.compose(op_k, inplace=True)
            wrapper.global_phase += math.pi  # multiply by -1
            op_k = wrapper

        ctrl_state = format(k, f"0{n_ancilla}b")
        c_gate = op_k.to_gate().control(
            n_ancilla, ctrl_state=ctrl_state,
        )
        qc.append(c_gate, anc_qubits + sys_qubits)

    return qc


def lcu_block_encoding(
    operators: list[QuantumCircuit],
    coefficients: np.ndarray,
    n_system: int,
) -> tuple[QuantumCircuit, float]:
    """Return (circuit, lambda) block-encoding A/lambda.

    A = sum_k c_k U_k.  Circuit layout: system [0..n_sys-1],
    ancilla [n_sys..n_sys+m-1].

    PREPARE and SELECT are materialized as UnitaryGate instances
    so the resulting circuit contains only ``unitary`` gates —
    no ControlledGate or StatePreparation.  This avoids:
      - Qiskit transpiler segfault (qs_decomposition on
        multi-controlled SELECT gates), and
      - AerSimulator ``unknown instruction: state_preparation``.
    """
    coefficients = np.asarray(coefficients, dtype=float)
    K = len(operators)
    n_ancilla = max(1, int(math.ceil(math.log2(max(K, 2)))))
    lam = float(np.sum(np.abs(coefficients)))

    n_total = n_system + n_ancilla
    anc_qubits = list(range(n_system, n_total))

    from qiskit.circuit.library import UnitaryGate

    # PREPARE: small (n_ancilla qubits), fast to convert.
    prep = build_prepare_circuit(coefficients, n_ancilla)
    prep_u = Operator(prep).data

    # SELECT: build block-diagonal unitary directly — O(K*N^2)
    # instead of Operator(circuit) which is very slow for the
    # multi-controlled gates at q>=5.
    N = 1 << n_system
    M = 1 << n_ancilla
    dim = N * M
    sel_u = np.eye(dim, dtype=complex)
    for k in range(K):
        if abs(coefficients[k]) < 1e-30:
            continue
        U_k = Operator(operators[k]).data
        if coefficients[k] < 0:
            U_k = -U_k
        sel_u[k * N:(k + 1) * N, k * N:(k + 1) * N] = U_k

    qc = QuantumCircuit(n_total, name="LCU")
    qc.append(
        UnitaryGate(prep_u, label="PREPARE"), anc_qubits,
    )
    qc.append(
        UnitaryGate(sel_u, label="SELECT"),
        list(range(n_total)),
    )
    qc.append(
        UnitaryGate(prep_u.conj().T, label="PREPARE_dg"),
        anc_qubits,
    )

    return qc, lam


# ── Heat propagator Taylor LCU ──────────────────────────────────────


def _build_net_shift_circuit(q: int, net_shift: int) -> QuantumCircuit:
    """Build a circuit applying S^net_shift to q qubits (periodic).

    net_shift > 0  -> compose increment_circuit |net_shift| times
    net_shift < 0  -> compose decrement_circuit |net_shift| times
    net_shift == 0 -> identity
    """
    if net_shift == 0:
        qc = QuantumCircuit(q, name="I")
        qc.id(list(range(q)))
        return qc
    if net_shift > 0:
        qc = QuantumCircuit(q, name=f"S+^{net_shift}")
        sp = increment_circuit(q)
        for _ in range(net_shift):
            qc.compose(sp, inplace=True)
        return qc
    qc = QuantumCircuit(q, name=f"S-^{abs(net_shift)}")
    sm = decrement_circuit(q)
    for _ in range(abs(net_shift)):
        qc.compose(sm, inplace=True)
    return qc


def _laplacian_power_net_shift_coeffs(
    dx: float,
    power: int,
) -> dict[int, float]:
    """Expand L^power as {net_shift: coeff}.

    L = (S+ + S- - 2I) / dx^2.  Multinomial expansion:
        L^k = (1/dx^2)^k * sum_{a+b+c=k} k!/(a!b!c!) * (-2)^c * S+^a * S-^b
    Net shift of each term is (a-b); aggregate by net shift.
    """
    if power == 0:
        return {0: 1.0}

    inv_dx2 = 1.0 / dx**2
    shift_coeffs: dict[int, float] = {}
    for a, b in product(range(power + 1), repeat=2):
        c = power - a - b
        if c < 0:
            continue
        mc = (
            math.factorial(power)
            / (math.factorial(a) * math.factorial(b) * math.factorial(c))
        )
        coeff = mc * ((-2.0) ** c) * (inv_dx2 ** power)
        net = a - b
        shift_coeffs[net] = shift_coeffs.get(net, 0.0) + coeff
    return shift_coeffs


def heat_taylor_lcu_terms(
    q: int,
    nu: float,
    dt: float,
    L_box: float,
    taylor_order: int = 4,
) -> tuple[list[QuantumCircuit], np.ndarray]:
    """Build operators and coefficients for the Taylor LCU.

    P_M = sum_{k=0}^{M} (nu*dt)^k * L^k / k!

    Aggregates by integer net_shift across all Taylor powers, then
    builds one circuit per unique net_shift.  Keying by integer
    avoids the brittleness of name-string merges.

    Returns (operators, coefficients) ready for lcu_block_encoding.
    """
    N = 1 << q
    dx = L_box / N

    # Aggregate by net_shift across all Taylor powers (integer key,
    # not name-string).
    merged: dict[int, float] = {}
    for k in range(taylor_order + 1):
        scalar = (nu * dt) ** k / math.factorial(k)
        for net_shift, c in _laplacian_power_net_shift_coeffs(
            dx, k,
        ).items():
            merged[net_shift] = merged.get(net_shift, 0.0) + scalar * c

    # Build one circuit per unique net_shift (drop near-zero terms).
    final_ops: list[QuantumCircuit] = []
    final_coeffs: list[float] = []
    for net_shift in sorted(merged):
        coeff = merged[net_shift]
        if abs(coeff) < 1e-30:
            continue
        final_ops.append(_build_net_shift_circuit(q, net_shift))
        final_coeffs.append(coeff)

    return final_ops, np.array(final_coeffs)


def _ucry_decompose(
    qc: QuantumCircuit,
    angles: np.ndarray,
    ctrl_qubits: list[int],
    tgt: int,
) -> None:
    """Recursively decompose uniformly-controlled Ry into CX+Ry.

    Applies Ry(angles[k]) on *tgt* when *ctrl_qubits* encode |k>.
    Uses the Mottonen/Shende recursive split: O(2^n) CX gates,
    zero multi-controlled gates, fully transpiler-safe.
    """
    n = len(ctrl_qubits)
    if n == 0:
        if abs(angles[0]) > 1e-15:
            qc.ry(float(angles[0]), tgt)
        return

    half = len(angles) // 2
    alpha = (angles[:half] + angles[half:]) / 2.0
    beta = (angles[:half] - angles[half:]) / 2.0

    _ucry_decompose(qc, alpha, ctrl_qubits[:-1], tgt)
    qc.cx(ctrl_qubits[-1], tgt)
    _ucry_decompose(qc, beta, ctrl_qubits[:-1], tgt)
    qc.cx(ctrl_qubits[-1], tgt)


def diag_potential_block_encoding(
    q: int,
    V: np.ndarray,
    dt_half: float,
) -> tuple[QuantumCircuit, float]:
    """Block-encode diag(exp(-V*dt_half)) on q+1 qubits.

    Post-selecting qubit q on |0> leaves data in
    diag(exp(-V*dt_half))/s_max applied to the input state.
    s_max = max_k |exp(-V_k*dt_half)| is the operator norm.

    Uses a recursively-decomposed uniformly-controlled Ry
    (Mottonen et al.) so that the resulting circuit contains
    only CX and Ry gates — avoids the Qiskit qs_decomposition
    segfault on multi-controlled gates.
    """
    N = 1 << q
    if V.shape != (N,):
        raise ValueError(f"V must have length {N}, got {V.shape}")
    diag_vals = np.exp(-V * dt_half)
    s_max = float(np.max(np.abs(diag_vals)))
    if s_max < 1e-300:
        raise ValueError(
            "diag(exp(-V*dt_half)) is identically zero"
        )

    thetas = 2.0 * np.arccos(
        np.clip(diag_vals / s_max, -1.0, 1.0),
    )

    qc = QuantumCircuit(q + 1, name="V_half")
    _ucry_decompose(qc, thetas, list(range(q)), q)
    return qc, s_max


def heat_lcu_with_potential_step_circuit(
    q: int,
    nu: float,
    dt: float,
    L_box: float,
    V: np.ndarray,
    taylor_order: int = 4,
) -> tuple[QuantumCircuit, float]:
    """Strang-split LCU step: V/2 -> heat LCU -> V/2.

    Returns (qc, lam_total) on q + m_heat + 2 qubits where:
      - q data qubits (low indices)
      - m_heat heat ancilla qubits (LCU PREPARE register)
      - 2 V ancillas (one per Strang half-step)

    lam_total = lam_heat * s_max_V^2.

    Post-selecting heat-ancilla=|0^m_heat> AND BOTH V-ancillas
    on |0> simultaneously at the end of the step gives:
        (1/lam_total) * exp(-V*dt/2) * P_M(nu*L*dt) * exp(-V*dt/2)
        |psi>

    Two V-ancillas (one per half-step) are used instead of a
    shared one because the two `v_qc` applications would compose
    *coherently* on a single V-ancilla (Ry(theta_k) twice = Ry(2
    theta_k)), producing cos(theta_k) on |0> instead of
    cos(theta_k/2)^2 = exp(-V_k*dt/2)^2 / s_max^2 — the wrong
    operator.  Independent ancillas restore the intended
    block-encoding semantics: each half-step's "rejected" branch
    goes to its own ancilla qubit and is post-selected away.
    """
    N = 1 << q
    if V.shape != (N,):
        raise ValueError(f"V must have length {N}, got {V.shape}")

    v_qc, s_max_V = diag_potential_block_encoding(q, V, dt / 2.0)
    heat_qc, lam_heat = heat_lcu_step_circuit(
        q, nu, dt, L_box, bc="periodic",
        taylor_order=taylor_order,
    )
    m_heat = heat_qc.num_qubits - q

    # Layout: data (q) | heat ancillas (m_heat) | V_a | V_b
    total_q = q + m_heat + 2
    v_anc_a = q + m_heat
    v_anc_b = q + m_heat + 1

    qc = QuantumCircuit(total_q, name="V_half_heat_V_half")

    # First V half — independent ancilla v_anc_a
    qc.compose(
        v_qc,
        qubits=list(range(q)) + [v_anc_a],
        inplace=True,
    )
    # Heat LCU — only acts on data + heat ancillas
    qc.compose(
        heat_qc,
        qubits=list(range(q + m_heat)),
        inplace=True,
    )
    # Second V half — independent ancilla v_anc_b
    qc.compose(
        v_qc,
        qubits=list(range(q)) + [v_anc_b],
        inplace=True,
    )

    lam_total = lam_heat * (s_max_V ** 2)
    qc.metadata = {
        "lcu_lambda": lam_total,
        "lcu_lambda_heat": lam_heat,
        "lcu_s_max_V": s_max_V,
    }
    return qc, lam_total


def heat_lcu_step_circuit(
    q: int,
    nu: float,
    dt: float,
    L_box: float,
    bc: str = "periodic",
    taylor_order: int = 4,
) -> tuple[QuantumCircuit, float]:
    """LCU block-encoding of the heat propagator for one timestep.

    Builds P_M = sum_{k=0}^{taylor_order} (nu*dt)^k L^k / k!
    as an LCU using S+/S- ladder primitives.

    Returns (circuit, lambda) where circuit has q system + m ancilla
    qubits, and post-selecting ancilla=|0>^m gives P_M / lambda.

    bc must be 'periodic' for v1.
    """
    if bc != "periodic":
        raise NotImplementedError(
            "LCU propagator requires periodic BC in v1"
        )

    ops, coeffs = heat_taylor_lcu_terms(
        q, nu, dt, L_box, taylor_order=taylor_order,
    )
    return lcu_block_encoding(ops, coeffs, q)
