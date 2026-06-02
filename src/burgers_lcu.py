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


def _dct_matrix_q(q: int) -> np.ndarray:
    """Orthonormal DCT-II matrix on N=2^q points.

    C[k,n] = alpha_k * cos(pi*k*(2n+1)/(2N)); C @ C.T = I.
    Private copy to avoid circular import with burgers_cole_hopf_circuit.
    """
    N = 1 << q
    n = np.arange(N)
    k = np.arange(N).reshape(-1, 1)
    C = np.cos(np.pi * k * (2 * n + 1) / (2 * N))
    C[0, :] *= 1.0 / np.sqrt(N)
    C[1:, :] *= np.sqrt(2.0 / N)
    return C


def heat_lcu_neumann_step_circuit(
    q: int,
    nu: float,
    dt: float,
    L_box: float,
    taylor_order: int = 8,
) -> tuple[QuantumCircuit, float]:
    """LCU block-encoding of the heat propagator for Neumann BC.

    Implements the Fourier-Bessel LCU decomposition of the diagonal
    heat propagator in the DCT-II eigenbasis.

    Neumann Laplacian eigenvalues: lam_k = -(4/dx^2)*sin^2(pi*k/(2N))
    Heat kernel: d_k = exp(nu*lam_k*dt) = exp(-A/2) * exp((A/2)*cos(pi*k/N))
    where A = 4*nu*dt/dx^2.

    Fourier-Bessel expansion (Jacobi-Anger identity):
      exp(x*cos(phi)) = I_0(x) + 2*sum_{j=1}^M I_j(x)*cos(j*phi)
    where I_j is the modified Bessel function of the first kind.

    Each term cos(j*pi*k/N) = Re[exp(i*j*pi*k/N)] is split into two
    diagonal unitaries V_j^+ and V_j^-:
      V_j^+[k,k] = exp(+i*j*pi*k/N)  -- product of q P-gates (LSB=qubit 0)
      V_j^-[k,k] = exp(-i*j*pi*k/N)  -- product of q P-gates

    Circuit layout: [DCT | PREPARE+SELECT+PREPARE_dag | DCT_dag]
    on q data qubits + m ancilla qubits.

    Gate count: O(M*q) for SELECT + O(q^2) for DCT = O(M*q) total
    where M = taylor_order.  All coefficients are positive (no sign
    absorption needed), so the LCU is a pure probability mixture.

    Returns (circuit, lambda) where post-selecting ancilla=|0>^m gives
    P_heat / lambda on the data register.
    """
    from scipy.special import iv as bessel_iv
    from qiskit.circuit.library import UnitaryGate

    N = 1 << q
    dx = L_box / N
    # A = 4*nu*dt/dx^2; x = A/2 is the Bessel argument
    A = 4.0 * nu * dt / (dx ** 2)
    x_bess = A / 2.0
    s = math.exp(-A / 2.0)

    M = taylor_order

    # Build LCU operators and coefficients.
    # Term 0: s*I_0(x)*I  (identity)
    # Term 2j-1: s*I_j(x)*V_j^+    for j=1..M
    # Term 2j:   s*I_j(x)*V_j^-    for j=1..M
    ops: list[QuantumCircuit] = []
    coeffs: list[float] = []

    i0 = float(bessel_iv(0, x_bess))
    qc_id = QuantumCircuit(q, name="I")
    if q > 0:
        qc_id.id(0)
    ops.append(qc_id)
    coeffs.append(s * i0)

    for j in range(1, M + 1):
        ij = float(bessel_iv(j, x_bess))
        c_j = s * ij
        if abs(c_j) < 1e-30:
            break  # Bessel series has converged; remaining terms negligible

        # V_j^+: P(j*pi*2^l/N) on qubit l  -->  phase exp(+i*j*pi*k/N) on |k>
        qc_p = QuantumCircuit(q, name=f"Vp{j}")
        for l in range(q):
            angle = j * math.pi * (1 << l) / N
            qc_p.p(angle, l)
        ops.append(qc_p)
        coeffs.append(c_j)

        # V_j^-: P(-j*pi*2^l/N) on qubit l -->  phase exp(-i*j*pi*k/N) on |k>
        qc_m = QuantumCircuit(q, name=f"Vm{j}")
        for l in range(q):
            angle = -j * math.pi * (1 << l) / N
            qc_m.p(angle, l)
        ops.append(qc_m)
        coeffs.append(c_j)

    coeffs_arr = np.array(coeffs)
    # All Bessel coefficients I_j(x) > 0 for x > 0, so all LCU weights are
    # positive -- no sign absorption needed in SELECT.
    lcu_qc, lam = lcu_block_encoding(ops, coeffs_arr, q)
    m_anc = lcu_qc.num_qubits - q

    # Sandwich the LCU with DCT / DCT^T to implement the full propagator
    # P_heat = C^T * D * C  (where C is the DCT-II matrix and D = diag(d_k)).
    C = _dct_matrix_q(q)
    C_gate = UnitaryGate(C, label="DCT")
    C_dag_gate = UnitaryGate(C.T, label="DCT_dag")

    n_total = q + m_anc
    data = list(range(q))
    qc = QuantumCircuit(n_total, name="heat_lcu_neumann")
    qc.append(C_gate, data)
    qc.compose(lcu_qc, inplace=True)
    qc.append(C_dag_gate, data)

    return qc, lam


def heat_lcu_step_circuit(
    q: int,
    nu: float,
    dt: float,
    L_box: float,
    bc: str = "periodic",
    taylor_order: int = 4,
) -> tuple[QuantumCircuit, float]:
    """LCU block-encoding of the heat propagator for one timestep.

    For bc='periodic': builds P_M = sum_{k=0}^{taylor_order} (nu*dt)^k L^k / k!
    as an LCU using S+/S- ladder primitives (shift-operator Taylor expansion).

    For bc='neumann': builds the Fourier-Bessel LCU in the DCT-II eigenbasis.
    taylor_order controls the Bessel truncation order M (default 8 for neumann,
    4 for periodic -- pass explicitly to override).

    Returns (circuit, lambda) where circuit has q system + m ancilla
    qubits, and post-selecting ancilla=|0>^m gives P_M / lambda.
    """
    if bc == "neumann":
        # Neumann BC: use Fourier-Bessel LCU in DCT-II eigenbasis.
        # Default to higher order for Neumann since Bessel converges faster.
        neumann_order = max(taylor_order, 8)
        return heat_lcu_neumann_step_circuit(
            q, nu, dt, L_box, taylor_order=neumann_order,
        )
    if bc != "periodic":
        raise NotImplementedError(
            f"LCU propagator: unsupported bc={bc!r}; "
            "expected 'periodic' or 'neumann'"
        )

    ops, coeffs = heat_taylor_lcu_terms(
        q, nu, dt, L_box, taylor_order=taylor_order,
    )
    return lcu_block_encoding(ops, coeffs, q)


# ── Conservative Burgers generator: A = nu*L - (1/2)*G*diag(u) ───────
#
# Direct-u nonlinear path (SPEC-direct-u-nonlinear-lcu.md).  The
# advection coefficient diag(u_seg) is classically known from the
# measure-reprepare boundary, so it is decomposed into two diagonal
# phase unitaries rather than block-encoded from a coherent oracle.
# Dense-SELECT path -- intended for small q (M1 correctness); the
# scalable nested encoder is M4.


def _diag_phase_pair(
    u_seg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Write diag(u) = (s/2)*(W + W_dag), s = ||u||_inf.

    W = diag(exp(i*arccos(u_i/s))) so (W + W_dag)/2 = diag(u_i/s),
    i.e. diag(u) = s * diag(u_i/s).  Both W and W_dag are unitary.
    """
    s = float(np.max(np.abs(u_seg))) if u_seg.size else 0.0
    N = u_seg.size
    if s < 1e-300:
        eye = np.eye(N, dtype=complex)
        return eye, eye, 0.0
    theta = np.arccos(np.clip(u_seg / s, -1.0, 1.0))
    w = np.exp(1j * theta)
    return np.diag(w), np.diag(np.conj(w)), s


def _conservative_base_terms(
    q: int,
    nu: float,
    L_box: float,
    u_seg: np.ndarray,
    bc: str = "periodic",
) -> list[tuple[float, np.ndarray]]:
    """Base LCU terms (coeff, unitary) for A = nu*L - (1/2)*G*diag(u).

    L = (S+ + S- - 2I)/dx^2,  G = (S+ - S-)/(2 dx),
    diag(u) = (s/2)(W + W_dag).  Conservative flux form:
    (1/2) d/dx(u^2) = (1/2) G diag(u) u.

    -(1/2) G diag(u) = -(s/(8 dx)) (S+ - S-)(W + W_dag).
    """
    N = 1 << q
    dx = L_box / N
    Sp = shift_matrix(N, +1, bc).astype(complex)
    Sm = shift_matrix(N, -1, bc).astype(complex)
    eye = np.eye(N, dtype=complex)
    W, W_dag, s = _diag_phase_pair(u_seg)

    c_lap = nu / dx**2
    c_adv = -s / (8.0 * dx)
    return [
        (c_lap, Sp),
        (c_lap, Sm),
        (-2.0 * c_lap, eye),
        (c_adv, Sp @ W),
        (c_adv, Sp @ W_dag),
        (-c_adv, Sm @ W),
        (-c_adv, Sm @ W_dag),
    ]


def advection_diffusion_taylor_lcu_terms(
    q: int,
    nu: float,
    dt: float,
    L_box: float,
    u_seg: np.ndarray,
    taylor_order: int = 4,
    bc: str = "periodic",
) -> tuple[list[QuantumCircuit], np.ndarray]:
    """Taylor-LCU terms for exp(A*dt), A = nu*L - (1/2)*G*diag(u_seg).

    P_M = sum_{m=0}^{M} (dt^m/m!) A^m, with A = sum_k c_k U_k.  Words
    are expanded iteratively as a dense-matrix dict (auto-dedup by the
    resulting unitary), then wrapped as UnitaryGate circuits for
    lcu_block_encoding.  Mirrors heat_taylor_lcu_terms but cannot
    aggregate to pure net-shifts (diag(u) does not commute with S+/-).

    Returns (operators, coefficients) ready for lcu_block_encoding.
    """
    from qiskit.circuit.library import UnitaryGate

    N = 1 << q
    base = _conservative_base_terms(q, nu, L_box, u_seg, bc)
    eye = np.eye(N, dtype=complex)

    def key(mat: np.ndarray) -> bytes:
        return np.round(mat, 9).astype(complex).tobytes()

    # result accumulates P_M; cur tracks A^m.  Values: [coeff, matrix].
    result: dict[bytes, list] = {key(eye): [1.0, eye]}
    cur: dict[bytes, list] = {key(eye): [1.0, eye]}

    for m in range(1, taylor_order + 1):
        nxt: dict[bytes, list] = {}
        for c_prev, m_prev in cur.values():
            for c_b, u_b in base:
                m_new = m_prev @ u_b
                c_new = c_prev * c_b
                k = key(m_new)
                if k in nxt:
                    nxt[k][0] += c_new
                else:
                    nxt[k] = [c_new, m_new]
        cur = nxt
        scal = dt**m / math.factorial(m)
        for k, (c, mat) in cur.items():
            if k in result:
                result[k][0] += scal * c
            else:
                result[k] = [scal * c, mat]

    ops: list[QuantumCircuit] = []
    coeffs: list[float] = []
    for c, mat in result.values():
        if abs(c) < 1e-30:
            continue
        qc = QuantumCircuit(q)
        qc.append(UnitaryGate(mat), list(range(q)))
        ops.append(qc)
        coeffs.append(float(np.real(c)))
    return ops, np.array(coeffs)


def conservative_burgers_lcu_step_circuit(
    q: int,
    nu: float,
    dt: float,
    L_box: float,
    u_seg: np.ndarray,
    taylor_order: int = 4,
    bc: str = "periodic",
) -> tuple[QuantumCircuit, float]:
    """Block-encode exp(A*dt)/lambda for the conservative Burgers
    generator A = nu*L - (1/2)*G*diag(u_seg).

    Returns (circuit, lambda); post-select ancilla=|0>^m for
    P_M / lambda applied to the system register.
    """
    ops, coeffs = advection_diffusion_taylor_lcu_terms(
        q, nu, dt, L_box, u_seg, taylor_order=taylor_order, bc=bc,
    )
    return lcu_block_encoding(ops, coeffs, q)


def conservative_burgers_lcu_operator(
    q: int,
    nu: float,
    dt: float,
    L_box: float,
    u_seg: np.ndarray,
    taylor_order: int = 4,
    bc: str = "periodic",
) -> np.ndarray:
    """Dense block-encoded operator P_M = sum_k c_k U_k ~= exp(A*dt).

    Equals the top-left block of
    `conservative_burgers_lcu_step_circuit` times lambda (verified to
    ~1e-15), assembled directly from the LCU terms.  Used by the
    statevector driver to avoid materialising the full (q+m)-qubit
    unitary, whose size blows up with the term count at high order.
    """
    N = 1 << q
    ops, coeffs = advection_diffusion_taylor_lcu_terms(
        q, nu, dt, L_box, u_seg, taylor_order=taylor_order, bc=bc,
    )
    out = np.zeros((N, N), dtype=complex)
    for c, op in zip(coeffs, ops):
        out += c * Operator(op).data
    return out
