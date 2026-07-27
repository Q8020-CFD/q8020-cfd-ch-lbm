"""Phase-2 pure-quantum QALB for 1-D D1Q3 Burgers (Itani et al.).

Phase 2, future-work #27.  Registered as `--method qlbm_circuit`; this
module holds the validated single-cell collision core.

Reference: W. Itani, K. R. Sreenivasan, S. Succi, "Quantum Algorithm for
Lattice Boltzmann (QALB)...", Phys. Fluids 36, 017112 (2024);
arXiv:2304.05915.  Tags "Eq. (n)" / "Eq. (Cn)" refer to the preprint.

Construction (validated to machine precision, see
tests/test_qalb_circuit.py):
  - Value/Fock encoding via the App C *finite-position embedding*: a
    value x in [-1,1] is encoded as |x> = sum_n P_n(x)|n>, where P_n are
    the monic polynomials orthogonal to the truncated-Gaussian linear
    functional (Eq C1/C4).  The finite-position operator q_hat_C
    (Eq C39) satisfies q_hat_C|x> = x|x> exactly, and the linear readout
    is exact: <1|x>/<0|x> = P_1(x)/P_0(x) = x  (Eq C20).  Earlier App B
    physicists' eigenstates only approximately diagonalise -> do not use
    them for the encoding.
  - Collision: the D1Q3 BGK functional Omega rederived against THIS
    repo's density-conserving equilibrium (rest (0,1,0)), in the
    delta_f variable, as operators in q_hat_C.  The single-cell
    collision is the Liouville flow generator G = sum_i Omega_i(q) D_i
    (D_i = d/dx on register i), exact on the truncated polynomial space.
    exp(T*G)|delta_f> decodes (via the scale-invariant linear readout,
    which cancels the constant-divergence deflation) to the classical
    collision flow to ~1e-13.

Hardware-honest collision synthesis (future-work #27.1, DONE): the dense
per-site UnitaryGate of e^{-iΔt Ĥ'} (Quantum-Shannon synthesis, ~4^(3qc)
depth) is replaceable by a Suzuki-Trotter circuit of the Pauli
decomposition of Ĥ' (`cell_collision_gate(..., trotter_reps>0)`) -- a
single position-free unitary on exactly 3*qc qubits, NO ancilla, exactly
unitary (no post-selection, unlike an LCU block-encoding).  Order-2
Trotter error ∝ 1/reps²; reps≈4 sits below the qc=2 Fock-truncation
floor.  The 3*qc / kron(reg_-1,reg_0,reg_+1) interface is frozen -- the
#27.2 transducer + streaming compose against exactly that.

Open (spec §7): log-depth streaming on the position register,
measure-reprepare(k) across lattice steps, and assembly into
`--method qlbm_circuit`.
"""

from __future__ import annotations

import sys
import time
from functools import lru_cache
from typing import Any, Callable

import numpy as np
from numpy.polynomial import polynomial as _poly
from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    PauliEvolutionGate,
    UnitaryGate,
)
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit_aer import AerSimulator
from q8020_backend_utils.ibm.circuit import (
    circuit_stats_in_basis,
    execute_circuit_counts,
    transpile_circuit,
)
from scipy.linalg import expm

from lib_lbm import (
    density,
    equilibrium,
    stream,
    tau_from_nu,
    velocity,
)

F_EQ0 = np.array([0.0, 1.0, 0.0])     # rest equilibrium (this repo's form)

# Hardware-cost reporting basis (cx-dominant), matches burgers_qlbm_circuit.
_METRIC_BASIS = ["cx", "rz", "sx", "x"]


# ── App C finite-position embedding (Eq C1-C43) ───────────────────────


def gamma_coeffs(N: int, z: float = 1.0, nq: int = 400) -> np.ndarray:
    """Recurrence coefficients gamma_n (Eq C4) of the monic polynomials
    orthogonal to L[p] = int_{-z}^{z} p(x) e^{-x^2} dx (Eq C1), via
    discretized Stieltjes.  Returns gamma_0..gamma_N (gamma_0 unused)."""
    t, w = np.polynomial.legendre.leggauss(nq)
    t = t * z
    w = w * z * np.exp(-t * t)
    pim1 = np.zeros_like(t)
    pi = np.ones_like(t)
    norms = [float(np.sum(w))]
    gam = [0.0]
    for _ in range(1, N + 1):
        pinext = t * pi - gam[-1] * pim1
        pim1, pi = pi, pinext
        nn = float(np.sum(w * pi * pi))
        gam.append(nn / norms[-1])
        norms.append(nn)
    return np.array(gam)


def _P_polys(N: int, gam: np.ndarray) -> list[np.ndarray]:
    """Monic polynomials P_n as monomial-coefficient arrays (Eq C4)."""
    polys = [np.array([1.0]), np.array([0.0, 1.0])]
    for n in range(1, N):
        xPn = np.concatenate([[0.0], polys[n]])
        prev = np.zeros(len(xPn))
        prev[:len(polys[n - 1])] = polys[n - 1]
        polys.append(xPn - gam[n] * prev)
    return polys


def finite_position_ops(qc: int) -> tuple[np.ndarray, np.ndarray, list]:
    """Return (q_hat_C, D, P_polys) for a qc-qubit register.

    q_hat_C = a^- + gamma_hat a^+  (Eq C39), with q_hat_C|x> = x|x>.
    D = d/dx as a matrix on |n> (exact on the truncated polynomial space):
    D[n,m] = coefficient of P_m in P_n'.
    """
    d = 1 << qc
    N = d - 1
    gam = gamma_coeffs(N)
    am = np.zeros((d, d))
    ap = np.zeros((d, d))
    for n in range(1, d):
        am[n - 1, n] = 1.0          # a^-|n> = |n-1>
    for n in range(d - 1):
        ap[n + 1, n] = 1.0          # a^+|n> = |n+1>
    q = am + np.diag(gam[:d]) @ ap
    polys = _P_polys(N, gam)
    D = np.zeros((d, d))
    for n in range(d):
        deriv = _poly.polyder(polys[n]) if len(polys[n]) > 1 else np.array([0.0])
        work = np.zeros(d)
        work[:len(deriv)] = deriv
        for m in range(d - 1, -1, -1):
            c = work[m]
            if c == 0.0:
                continue
            D[n, m] = c
            pm = np.zeros(d)
            pm[:len(polys[m])] = polys[m]
            work -= c * pm
    return q, D, polys


def encode_value(x: float, polys: list) -> np.ndarray:
    """Encode x in [-1,1] as |x> = sum_n P_n(x)|n>  (Eq C20).  Not
    normalised; the scale-invariant linear readout handles that."""
    return np.array([_poly.polyval(x, polys[n]) for n in range(len(polys))])


# ── D1Q3 collision (this repo's equilibrium) as flow generator ────────


def _embed(op: np.ndarray, slot: int, d: int) -> np.ndarray:
    mats = [np.eye(d)] * 3
    mats[slot] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def omega_operators(q: np.ndarray, tau: float) -> list[np.ndarray]:
    """Omega_i(q_hat_C) for the 3 densities, in delta_f, on the joint
    3-register space.  Rederived vs lib_lbm.equilibrium (rest
    (0,1,0)): u = -s, s = q_{-1}-q_{+1}."""
    d = q.shape[0]
    qm1, q0, qp1 = _embed(q, 0, d), _embed(q, 1, d), _embed(q, 2, d)
    s = qm1 - qp1
    s2 = s @ s
    inv = -1.0 / tau
    return [
        inv * (qm1 - 0.5 * (s2 + s)),
        inv * (q0 + s2),
        inv * (qp1 - 0.5 * (s2 - s)),
    ]


def collision_flow_generator(q: np.ndarray, D: np.ndarray,
                             tau: float) -> np.ndarray:
    """Liouville flow generator G = sum_i Omega_i(q) D_i for the single
    cell.  exp(T*G) evolves the encoded delta_f along the collision flow.
    """
    d = q.shape[0]
    om = omega_operators(q, tau)
    Ds = [_embed(D, 0, d), _embed(D, 1, d), _embed(D, 2, d)]
    return om[0] @ Ds[0] + om[1] @ Ds[1] + om[2] @ Ds[2]


def encode_cell(df3: np.ndarray, polys: list) -> np.ndarray:
    s = encode_value(float(df3[0]), polys)
    for v in df3[1:]:
        s = np.kron(s, encode_value(float(v), polys))
    return s


def decode_cell(psi: np.ndarray, qc: int) -> np.ndarray:
    d = 1 << qc
    p = psi.reshape(d, d, d)
    z = p[0, 0, 0]
    return np.real(np.array([p[1, 0, 0] / z, p[0, 1, 0] / z, p[0, 0, 1] / z]))


# ── App B bosonic encoding + Hermitised UNITARY collision (shots path) ─
#
# The App C generator above reproduces the flow exactly but is non-normal:
# forcing it unitary leaves an O(1) error (its momentum d/dx is not skew-
# adjoint).  Itani's App B fixes this with proper bosonic q,p ([q,p]=iI),
# at the price of normal-ordering the quadratic collision term.  This is
# the path that runs on shots without post-selection.


def osc_ops(qc: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Truncated bosonic mode (Itani Eq 60-63): q=(a+a†)/√2,
    p=i(a†-a)/√2 with [q,p]=iI (exact but for the truncated top corner).
    Returns (q, p, vacuum=|q=0>)."""
    d = 1 << qc
    a = np.zeros((d, d))
    for n in range(1, d):
        a[n - 1, n] = np.sqrt(n)
    ad = a.T
    q = (a + ad) / np.sqrt(2.0)
    p = 1j * (ad - a) / np.sqrt(2.0)
    vac = np.zeros(d)
    vac[0] = 1.0
    return q, p, vac


def omega_operators_B(q: np.ndarray, tau: float) -> list[np.ndarray]:
    """Normal-ordered D1Q3 collision Omega_i in the bosonic position op
    (this repo's rest eq (0,1,0), in delta_f).  On the vacuum/coherent
    encoding <s²>=s_cl²+Var(q_-1)+Var(q_+1)=s_cl²+1, so the bare s² over-
    counts by the vacuum variance; subtracting it (s@s - I) is what makes
    the Hermitised collision reproduce the classical flow (convergent in
    qc)."""
    d = q.shape[0]
    qm1, q0, qp1 = _embed(q, 0, d), _embed(q, 1, d), _embed(q, 2, d)
    s = qm1 - qp1
    s2 = s @ s - np.eye(d ** 3)               # normal ordering
    inv = -1.0 / tau
    return [
        inv * (qm1 - 0.5 * (s2 + s)),
        inv * (q0 + s2),
        inv * (qp1 - 0.5 * (s2 - s)),
    ]


def hermitised_collision_hamiltonian(q: np.ndarray, p: np.ndarray,
                                     tau: float) -> np.ndarray:
    """Itani Eq 85: H' = ½ Σ_i (p_i Ω_i(q) + Ω_i(q) p_i), Hermitian by
    construction so e^{-iΔt H'} is exactly unitary (no post-selection).
    The anti-Hermitian part is the constant divergence -2/τ (Eq 83),
    recovered as a scalar deflation that cancels in the <q> ratio
    readout."""
    d = q.shape[0]
    om = omega_operators_B(q, tau)
    pe = [_embed(p, i, d) for i in range(3)]
    return 0.5 * np.asarray(
        sum(pe[i] @ om[i] + om[i] @ pe[i] for i in range(3)))


def encode_cell_B(df3: np.ndarray, p: np.ndarray,
                  vac: np.ndarray) -> np.ndarray:
    """Encode the 3 densities as position eigenstates |δf_i>=e^{-iδf_i p}
    |0> (Itani Eq 66), tensored."""
    regs = [expm(-1j * float(v) * p) @ vac for v in df3]
    out = regs[0]
    for r in regs[1:]:
        out = np.kron(out, r)
    return out


@lru_cache(maxsize=None)
def cell_collision_unitary_B(tau: float, qc: int,
                             collision_time: float) -> np.ndarray:
    """Per-site Hermitised collision unitary U=e^{-i T H'} on 3*qc qubits
    (Itani Eq 85).  Exactly unitary -> shots without post-selection.
    State-independent, so cached and reused across every site/step."""
    q, p, _ = osc_ops(qc)
    H = hermitised_collision_hamiltonian(q, p, tau)
    return expm(-1j * collision_time * H)


# ── #27.1: hardware-honest Trotter synthesis of the collision unitary ─
#
# The dense UnitaryGate of e^{-iT H'} above is exact but Quantum-Shannon
# synthesises to ~4^(3qc) CX (~10⁴ depth at qc=3).  Ĥ' is a fixed sparse
# Hermitian operator, so decompose it into Pauli words and synthesise
# e^{-iT H'} by Suzuki-Trotter (Itani §IX / SPEC §3.7).  This keeps the
# App B virtue -- a single 3*qc-qubit unitary, NO ancilla, exactly
# unitary (no post-selection) -- unlike an LCU block-encoding, which
# would reintroduce an ancilla and post-selection.  Depth/accuracy trade
# off via (trotter_order, trotter_reps); order-2 error ∝ 1/reps².


@lru_cache(maxsize=None)
def collision_hamiltonian_pauli(tau: float, qc: int):
    """Pauli decomposition (SparsePauliOp) of the Hermitised collision
    Hamiltonian Ĥ' (Itani Eq 85) on 3*qc qubits.  Built once; the
    coefficients are real since Ĥ' is Hermitian."""
    q, p, _ = osc_ops(qc)
    H = hermitised_collision_hamiltonian(q, p, tau)
    return SparsePauliOp.from_operator(Operator(H)).simplify()


@lru_cache(maxsize=None)
def cell_collision_gate(tau: float, qc: int, collision_time: float,
                        trotter_reps: int = 0, trotter_order: int = 2):
    """Per-site collision as a Qiskit gate on exactly 3*qc qubits -- a
    position-free, ancilla-free, exactly-unitary block (the frozen
    interface the #27.2 transducer + streaming compose against).

      trotter_reps == 0 : dense UnitaryGate of e^{-iT H'} (Quantum-Shannon
                          synthesis; exact, but ~4^(3qc) CX).
      trotter_reps  > 0 : Suzuki/Lie-Trotter synthesis of the Pauli Ĥ'
                          (#27.1); tunable depth vs accuracy.

    Cached: state-independent, so built once and reused per site/step."""
    n = 3 * qc
    if trotter_reps <= 0:
        return UnitaryGate(cell_collision_unitary_B(tau, qc, collision_time),
                           label="W")
    spo = collision_hamiltonian_pauli(tau, qc)
    synth = (LieTrotter(reps=trotter_reps) if trotter_order == 1
             else SuzukiTrotter(order=trotter_order, reps=trotter_reps))
    evo = PauliEvolutionGate(spo, time=collision_time, synthesis=synth)
    circ = QuantumCircuit(n, name="qalb_collide_trotter")
    circ.append(evo, range(n))
    return circ.to_gate(label=f"Wtrot(r={trotter_reps})")


def cell_collision_shots(
    df3: np.ndarray, tau: float, qc: int, collision_time: float,
    shots: int, backend: Any = None, seed: int | None = None,
    trotter_reps: int = 0, trotter_order: int = 2,
    timing: dict | None = None,
) -> np.ndarray:
    """Per-site collision on SHOTS, no post-selection (the collision is
    unitary): prepare |δf>, apply the collision gate, rotate each density
    register into the q̂ eigenbasis, measure 3*qc qubits, and estimate
    δf_i'=<q̂_i> from the marginal counts.  RAM-safe (3*qc qubits).
    trotter_reps>0 selects the #27.1 Trotter synthesis of e^{-iT H'}
    (hardware-honest depth) over the dense UnitaryGate.

    If ``timing`` is given, the per-circuit transpile and execute wall times
    are ADDED into timing['transpile'] / timing['execute'] (so a caller can
    accumulate the split across all sites/steps)."""
    if backend is None:
        backend = AerSimulator()
    q, p, vac = osc_ops(qc)
    gate = cell_collision_gate(tau, qc, collision_time,
                               trotter_reps, trotter_order)
    lam, V = np.linalg.eigh(q)                # q̂ = V diag(lam) V†
    Vd = V.conj().T                           # eigenbasis -> computational
    n = 3 * qc
    psi = encode_cell_B(df3, p, vac)
    circ = QuantumCircuit(n, n)
    circ.initialize(psi.tolist(), range(n))
    circ.append(gate, range(n))
    for i in range(3):
        circ.append(UnitaryGate(Vd, label="Vd"), range(i * qc, (i + 1) * qc))
    circ.measure(range(n), range(n))
    circ_t, tr_info = transpile_circuit(circ, backend, seed_transpiler=seed)
    counts, ex_info = execute_circuit_counts(
        circ_t, backend, shots=shots, seed=seed,
    )
    if timing is not None:
        timing['transpile'] = timing.get('transpile', 0.0) + tr_info['wall_time']
        timing['execute'] = timing.get('execute', 0.0) + ex_info['wall_time']
    exp = np.zeros(3)
    tot = 0
    for bitstr, cnt in counts.items():
        rev = bitstr.replace(" ", "")[::-1]   # rev[j] = qubit j (0 = LSB)
        tot += cnt
        for i in range(3):
            idx = sum(int(rev[i * qc + b]) << b for b in range(qc))
            exp[i] += lam[idx] * cnt
    # encode_cell_B tensors reg0 as the high np.kron factor, but qiskit
    # qubit 0 is the LSB, so measured qubit-group j is density (2-j).
    return (exp / max(tot, 1))[::-1]


# ── Full lattice: pure-quantum collision (state-independent) + stream ─


def _qalb_cell_metric(
    tau: float, qc: int, collision_time: float,
    trotter_reps: int, trotter_order: int, seed: int | None,
) -> dict | None:
    """Hardware-honest cost of one per-site shots collision circuit
    (representative equilibrium prep + collision gate + q̂-basis rotations),
    decomposed to the cx-dominant metric basis.  Returns
    {n_qubits, circuit_depth, gate_counts} or None if transpile fails."""
    try:
        qosc, posc, vac = osc_ops(qc)
        gate = cell_collision_gate(
            tau, qc, collision_time, trotter_reps, trotter_order,
        )
        _, Vq = np.linalg.eigh(qosc)
        Vd = Vq.conj().T
        n = 3 * qc
        mc = QuantumCircuit(n)
        mc.initialize(
            encode_cell_B(np.zeros(3), posc, vac).tolist(), range(n),
        )
        mc.append(gate, range(n))
        for i in range(3):
            mc.append(UnitaryGate(Vd, label="Vd"), range(i * qc, (i + 1) * qc))
        info = circuit_stats_in_basis(mc, _METRIC_BASIS, seed_transpiler=seed)
        return {
            "n_qubits": info["num_qubits"],
            "circuit_depth": info["depth"],
            "gate_counts": info["gate_counts"],
        }
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"[qlbm_circuit] cell metric transpile failed: {e}",
              file=sys.stderr)
        return None


def run_qalb_simulation(
    u0: np.ndarray,
    x: np.ndarray,
    nu: float,
    dt: float,
    n_steps: int,
    bc: str = "periodic",
    source_fn: Callable | None = None,
    shots: int = 0,
    backend: Any = None,
    qc: int = 3,
    collision_time: float | None = None,
    seed: int | None = None,
    trotter_reps: int = 0,
    trotter_order: int = 2,
    **_ignored: Any,
) -> tuple[list[np.ndarray], list[dict], list[int]]:
    """Pure-quantum QALB: per-site quantum collision via the App C flow
    operator (built ONCE, state independent) + exact streaming.

    The collision is local, so the single-cell operator is applied
    site-by-site -- no classical collision mirror in the loop (contrast
    `qlbm_circuit_hybrid`).  Two collision paths:
      shots=0 : App C flow operator (exact, statevector-faithful).
      shots>0 : App B Hermitised UNITARY collision on shots, no post-
                selection (`cell_collision_shots`, Itani Eq 85), read out
                as <q̂> per site.  measure-reprepare(k=1): stream
                classically and reload the full measured f each step
                (never re-equilibrate from moments).

    Returns (solutions, metrics, genuine_steps).
    """
    N = len(u0)
    q = int(np.log2(N))
    assert N == (1 << q), f"N={N} must be a power of 2"
    dx = x[1] - x[0]
    dt_lbm = dx
    T_end = dt * n_steps
    n_steps_lbm = max(1, round(T_end / dt_lbm))
    tau = tau_from_nu(nu, dx, dt_lbm)

    # The QALB collision is the continuous BGK flow; the classical LBM
    # step relaxes the off-equilibrium part by 1/tau (Euler).  Match the
    # linear relaxation per step: 1 - e^{-T/tau} = 1/tau.
    if collision_time is None:
        collision_time = (
            float(-tau * np.log(1.0 - 1.0 / tau)) if tau > 1.0 else 1.0
        )

    # tau<1 (dt/tau>1) is Itani's divergent regime (App A: no time-
    # independent error bound); the Fock-truncation error blows up over a
    # multi-step run.  Warn -- results there are unreliable, not a bug.
    if tau <= 1.0:
        print(f"[qlbm_circuit] WARNING: tau={tau:.3f}<=1 (dt/tau={1.0 / tau:.2f}"
              ">1) is the QALB divergent regime (Itani App A); multi-step "
              "results are unreliable -- raise nu, q, or qc.", file=sys.stderr)

    if shots:
        synth = (f"trotter(o={trotter_order},r={trotter_reps})"
                 if trotter_reps else "dense")
        path = f"shots/{synth}"
    else:
        path = "operator"
    print(
        f"[qlbm_circuit] (QALB) q={q} N={N} tau={tau:.4f} qc={qc} "
        f"n_steps_lbm={n_steps_lbm} collision_time={collision_time} "
        f"path={path} shots={shots} (caller dt={dt:.3e} n={n_steps})",
        file=sys.stderr, flush=True,
    )

    if shots and backend is None:
        backend = AerSimulator()
    qop, D, polys = finite_position_ops(qc)
    U = None if shots else expm(collision_time
                                * collision_flow_generator(qop, D, tau))

    # Per-site collision circuit cost (state-independent -> measured once,
    # reported per step).  Only the shots path builds/executes real circuits;
    # the operator path is a statevector idealisation with no circuit cost.
    cell_metric = _qalb_cell_metric(
        tau, qc, collision_time, trotter_reps, trotter_order, seed,
    ) if shots else None

    f = equilibrium(u0)
    lbm_solutions = [u0.copy()]
    metrics: list[dict] = []
    t0 = time.time()
    for step in range(1, n_steps_lbm + 1):
        t_step = time.time()
        # Per-step transpile/execute accumulator (summed over all N sites);
        # the remainder of the step wall time is classical (encode/decode,
        # streaming, equilibrium, Python overhead).
        step_timing: dict = {"transpile": 0.0, "execute": 0.0}
        df = f - F_EQ0[:, None]
        for site in range(N):
            if shots:
                df[:, site] = cell_collision_shots(
                    df[:, site], tau, qc, collision_time, shots,
                    backend=backend, seed=seed,
                    trotter_reps=trotter_reps, trotter_order=trotter_order,
                    timing=step_timing,
                )
            else:
                df[:, site] = decode_cell(U @ encode_cell(df[:, site], polys), qc)
        f = df + F_EQ0[:, None]
        f = stream(f, bc=bc)
        u_cur = velocity(f)
        lbm_solutions.append(u_cur.copy())
        # Report step in the CALLER'S fine-step frame (lattice step lands on
        # fine step round(step*dt_lbm/dt)) so metrics align with the stored
        # snapshot timeline; otherwise lattice steps 1..n_steps_lbm collapse
        # into the first few % of a 0..n_steps axis.
        caller_step = min(int(round(step * dt_lbm / dt)), n_steps)
        rec = {
            "step": caller_step, "lattice_step": step,
            "u_max": float(np.max(np.abs(u_cur))),
            "rho_mean": float(np.mean(density(f))), "qc": qc, "path": path,
        }
        if shots:
            # One collision circuit per lattice site, executed this step.
            # Split the step wall time into transpile / quantum-execution /
            # classical-other so the runtime panel can stack them honestly.
            step_wall = time.time() - t_step
            rec["n_circuits"] = N
            rec["transpilation_time_s"] = step_timing["transpile"]
            rec["execution_time_s"] = step_timing["execute"]
            rec["circuit_construction_time_s"] = max(
                step_wall - step_timing["transpile"] - step_timing["execute"],
                0.0,
            )
            if cell_metric:
                rec.update(cell_metric)
        metrics.append(rec)
    metrics_total_s = time.time() - t0

    solutions = [u0.copy()]
    for j in range(1, n_steps + 1):
        kk = min(round(j * dt / dt_lbm), n_steps_lbm)
        solutions.append(lbm_solutions[kk].copy())
    genuine_steps = sorted({
        min(round(s * dt_lbm / dt), n_steps) for s in range(n_steps_lbm + 1)
    })
    if metrics:
        metrics[-1]["method_wall_time_s"] = metrics_total_s
    return solutions, metrics, genuine_steps
