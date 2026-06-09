"""Phase-2 pure-quantum QALB for 1-D D1Q3 Burgers (Itani et al.).

Spec: docs/future/SPEC-qlbm-pure-quantum-qalb.md (Phase 2, future-work
#27).  Will register as `--method qlbm_circuit` once the lattice
(streaming + measure-reprepare) is assembled; today this module holds
the validated single-cell collision core.

Reference: W. Itani, K. R. Sreenivasan, S. Succi, "Quantum Algorithm for
Lattice Boltzmann (QALB)...", Phys. Fluids 36, 017112 (2024);
arXiv:2304.05915.  Tags "Eq. (n)" / "Eq. (Cn)" refer to the preprint.

Construction (validated to machine precision, see __main__):
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

Open (spec §7): log-depth streaming on the position register,
measure-reprepare(k) across lattice steps, the unitary/Hermitised
circuit synthesis of the flow (Trotter/LCU of Eq 85 with the explicit
deflation for the norm), and assembly into `--method qlbm_circuit`.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Callable

import numpy as np
from numpy.polynomial import polynomial as _poly
from scipy.linalg import expm

from burgers_lbm import (
    density,
    equilibrium,
    stream,
    tau_from_nu,
    velocity,
)


F_EQ0 = np.array([0.0, 1.0, 0.0])     # rest equilibrium (this repo's form)


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
    for n in range(1, N + 1):
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


def decode_value(psi_reg: np.ndarray) -> float:
    """Exact linear readout x = <1|psi>/<0|psi>  (Eq C20, P_1/P_0 = x)."""
    return float((psi_reg[1] / psi_reg[0]).real)


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
    3-register space.  Rederived vs burgers_lbm.equilibrium (rest
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
    return np.array([p[1, 0, 0] / z, p[0, 1, 0] / z, p[0, 0, 1] / z]).real


# ── Transpilable per-site collision circuit (block-encoded) ───────────


def _sym_sqrt(A: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(0.5 * (A + A.conj().T))
    w = np.clip(w.real, 0.0, None)
    return (V * np.sqrt(w)) @ V.conj().T


def cell_collision_block_unitary(tau: float, qc: int,
                                 collision_time: float) -> tuple[np.ndarray, float]:
    """Sz-Nagy dilation of the per-site collision A = U_cell/alpha
    (U_cell = exp(T*G), state independent, built once).  Returns (dilation
    unitary on 3*qc+1 qubits, alpha).  Post-selecting the ancilla |0>
    applies A; the scale 1/alpha cancels in the ratio readout."""
    qop, D, _ = finite_position_ops(qc)
    U_cell = expm(collision_time * collision_flow_generator(qop, D, tau))
    alpha = float(np.linalg.norm(U_cell, 2)) * (1.0 + 1e-9)
    A = U_cell / alpha
    d3 = A.shape[0]
    S = _sym_sqrt(np.eye(d3) - A @ A.conj().T)
    T = _sym_sqrt(np.eye(d3) - A.conj().T @ A)
    return np.block([[A, S], [T, -A.conj().T]]), alpha


def cell_collision_circuit(tau: float, qc: int, collision_time: float):
    """Per-site collision as a Qiskit circuit on 3*qc system + 1 ancilla.
    System qubits 0..3qc-1 (the 3 density registers), ancilla = top qubit.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    n_sys = 3 * qc
    U, alpha = cell_collision_block_unitary(tau, qc, collision_time)
    qc_circ = QuantumCircuit(n_sys + 1, name="qalb_collide")
    qc_circ.append(UnitaryGate(U, label="A_blk"), range(n_sys + 1))
    return qc_circ, alpha


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
    return 0.5 * sum(pe[i] @ om[i] + om[i] @ pe[i] for i in range(3))


def encode_cell_B(df3: np.ndarray, p: np.ndarray,
                  vac: np.ndarray) -> np.ndarray:
    """Encode the 3 densities as position eigenstates |δf_i>=e^{-iδf_i p}
    |0> (Itani Eq 66), tensored."""
    regs = [expm(-1j * float(v) * p) @ vac for v in df3]
    out = regs[0]
    for r in regs[1:]:
        out = np.kron(out, r)
    return out


def decode_cell_B(psi: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Read each density as the normalised <q_i> (scale invariant, so the
    deflation scalar cancels)."""
    d = q.shape[0]
    n = (psi.conj() @ psi).real
    return np.array([(psi.conj() @ _embed(q, i, d) @ psi).real / n
                     for i in range(3)])


def cell_collision_unitary_B(tau: float, qc: int,
                             collision_time: float) -> np.ndarray:
    """Per-site Hermitised collision unitary U=e^{-i T H'} on 3*qc qubits
    (Itani Eq 85).  Exactly unitary -> shots without post-selection."""
    q, p, _ = osc_ops(qc)
    H = hermitised_collision_hamiltonian(q, p, tau)
    return expm(-1j * collision_time * H)


def cell_collision_shots(
    df3: np.ndarray, tau: float, qc: int, collision_time: float,
    shots: int, backend: Any = None, seed: int | None = None,
) -> np.ndarray:
    """Per-site collision on SHOTS, no post-selection (W is unitary):
    prepare |δf>, apply W=e^{-iT H'}, rotate each density register into
    the q̂ eigenbasis, measure 3*qc qubits, and estimate δf_i'=<q̂_i> from
    the marginal counts.  RAM-safe (3*qc qubits)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from q8020_cfd_qutil.circuit import (
        execute_circuit_counts,
        transpile_circuit,
    )
    if backend is None:
        from qiskit_aer import AerSimulator
        backend = AerSimulator()
    q, p, vac = osc_ops(qc)
    W = cell_collision_unitary_B(tau, qc, collision_time)
    lam, V = np.linalg.eigh(q)                # q̂ = V diag(lam) V†
    Vd = V.conj().T                           # eigenbasis -> computational
    n = 3 * qc
    psi = encode_cell_B(df3, p, vac)
    circ = QuantumCircuit(n, n)
    circ.initialize(psi.tolist(), range(n))
    circ.append(UnitaryGate(W, label="W"), range(n))
    for i in range(3):
        circ.append(UnitaryGate(Vd, label="Vd"), range(i * qc, (i + 1) * qc))
    circ.measure(range(n), range(n))
    circ_t, _ = transpile_circuit(circ, backend, seed_transpiler=seed)
    counts, _ = execute_circuit_counts(circ_t, backend, shots=shots, seed=seed)
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
        collision_time = -tau * np.log(1.0 - 1.0 / tau) if tau > 1.0 else 1.0

    # tau<1 (dt/tau>1) is Itani's divergent regime (App A: no time-
    # independent error bound); the Fock-truncation error blows up over a
    # multi-step run.  Warn -- results there are unreliable, not a bug.
    if tau <= 1.0:
        print(f"[qlbm_circuit] WARNING: tau={tau:.3f}<=1 (dt/tau={1.0 / tau:.2f}"
              ">1) is the QALB divergent regime (Itani App A); multi-step "
              "results are unreliable -- raise nu, q, or qc.", file=sys.stderr)

    path = "shots" if shots else "operator"
    print(
        f"[qlbm_circuit] (QALB) q={q} N={N} tau={tau:.4f} qc={qc} "
        f"n_steps_lbm={n_steps_lbm} collision_time={collision_time} "
        f"path={path} shots={shots} (caller dt={dt:.3e} n={n_steps})",
        file=sys.stderr, flush=True,
    )

    if shots and backend is None:
        from qiskit_aer import AerSimulator
        backend = AerSimulator()
    qop, D, polys = finite_position_ops(qc)
    U = None if shots else expm(collision_time
                                * collision_flow_generator(qop, D, tau))

    f = equilibrium(u0)
    lbm_solutions = [u0.copy()]
    metrics: list[dict] = []
    t0 = time.time()
    for step in range(1, n_steps_lbm + 1):
        df = f - F_EQ0[:, None]
        for site in range(N):
            if shots:
                df[:, site] = cell_collision_shots(
                    df[:, site], tau, qc, collision_time, shots,
                    backend=backend, seed=seed,
                )
            else:
                df[:, site] = decode_cell(U @ encode_cell(df[:, site], polys), qc)
        f = df + F_EQ0[:, None]
        f = stream(f, bc=bc)
        u_cur = velocity(f)
        lbm_solutions.append(u_cur.copy())
        metrics.append({
            "step": step, "u_max": float(np.max(np.abs(u_cur))),
            "rho_mean": float(np.mean(density(f))), "qc": qc, "path": path,
        })
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


if __name__ == "__main__":
    from scipy.linalg import expm
    from scipy.integrate import solve_ivp
    np.set_printoptions(precision=4, suppress=True)
    tau = 2.42

    print("[gate1] App C embedding (q|x>=x|x>, exact linear readout):")
    for qc in (2, 3, 4):
        q, D, polys = finite_position_ops(qc)
        e_eig = max(np.max(np.abs(q @ encode_value(x, polys)
                                  - x * encode_value(x, polys)))
                    for x in (-0.3, -0.1, 0.15, 0.3))
        e_dec = max(abs(decode_value(encode_value(x, polys)) - x)
                    for x in (-0.3, -0.1, 0.15, 0.3))
        e_tr = max(abs(decode_value(expm(dl * D) @ encode_value(x, polys))
                       - (x + dl))
                   for x in (-0.1, 0.0, 0.1) for dl in (0.05, 0.1))
        print(f"   qc={qc}: q|x>=x|x> {e_eig:.1e}   decode {e_dec:.1e}   "
              f"translate {e_tr:.1e}")

    print("[gate2] single-cell collision: flow generator vs classical flow")
    inv = -1.0 / tau

    def classical_flow(df, T):
        def rhs(t, y):
            sm = y[0] - y[2]
            return np.array([inv * (y[0] - 0.5 * (sm * sm + sm)),
                             inv * (y[1] + sm * sm),
                             inv * (y[2] - 0.5 * (sm * sm - sm))])
        return solve_ivp(rhs, [0, T], df, rtol=1e-11, atol=1e-13).y[:, -1]

    for qc in (2, 3, 4):
        q, D, polys = finite_position_ops(qc)
        G = collision_flow_generator(q, D, tau)
        U = expm(1.0 * G)
        worst = 0.0
        for df in (np.array([0.03, -0.05, 0.02]),
                   np.array([0.06, -0.10, 0.04]),
                   equilibrium(np.array([0.1])).ravel() - F_EQ0):
            dq = decode_cell(U @ encode_cell(df, polys), qc)
            worst = max(worst, float(np.max(np.abs(dq - classical_flow(df, 1.0)))))
        tag = "converged" if worst < 1e-9 else "truncation-limited (raise qc)"
        print(f"   qc={qc}: max|err| vs classical flow = {worst:.2e}  [{tag}]")

    print("[gate3] full lattice QALB run (pure-quantum collision + stream)")
    N = 32
    xg = np.linspace(0.0, 1.0, N, endpoint=False)
    dxg = xg[1] - xg[0]
    u0 = 0.3 * np.sin(2 * np.pi * xg)
    sols, mets, gs = run_qalb_simulation(
        u0, xg, nu=3e-2, dt=dxg, n_steps=10, bc="periodic", qc=3,
    )
    uf = sols[-1]
    print(f"   N={N} steps=10  u_max: {np.max(np.abs(u0)):.3f} -> "
          f"{np.max(np.abs(uf)):.3f}  (smooth decay expected)")
    smooth = np.max(np.abs(np.diff(uf, append=uf[:1]))) < 0.2
    decayed = np.max(np.abs(uf)) < np.max(np.abs(u0))
    print(f"   smooth={smooth}  decayed={decayed}  "
          f"({'PASS' if smooth and decayed else 'FAIL'})")

    print("[gate4] transpilable per-site collision circuit (block-encoded)")
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Statevector
    qcf = 2
    tcol = -tau * np.log(1.0 - 1.0 / tau)
    qop, D, polys = finite_position_ops(qcf)
    U_cell = expm(tcol * collision_flow_generator(qop, D, tau))
    circ, alpha = cell_collision_circuit(tau, qcf, tcol)
    n_sys = 3 * qcf
    d = 1 << qcf
    df = np.array([0.04, -0.06, 0.03])
    psi = encode_cell(df, polys)
    psi = psi / np.linalg.norm(psi)
    # circuit (Statevector) post-selected vs the operator
    full = QuantumCircuit(n_sys + 1)
    full.initialize(psi.tolist(), range(n_sys))
    full.compose(circ, range(n_sys + 1), inplace=True)
    sv = np.asarray(Statevector(full).data)
    block = sv[:1 << n_sys]                       # ancilla |0> block
    dq_circ = decode_cell(block, qcf)
    dq_op = decode_cell(U_cell @ encode_cell(df, polys), qcf)
    e_circ = float(np.max(np.abs(dq_circ - dq_op)))
    tp = transpile(circ, basis_gates=["u", "cx"], optimization_level=1)
    p_keep = float(np.sum(np.abs(block) ** 2))
    print(f"   circuit vs operator decode  max|err|={e_circ:.2e} "
          f"({'PASS' if e_circ < 1e-10 else 'FAIL'})")
    print(f"   transpiled depth={tp.depth()}  cx={tp.count_ops().get('cx', 0)}  "
          f"qubits={circ.num_qubits}  post-select p_keep={p_keep:.3f}")

    print("[gate5] App B Hermitised UNITARY collision (shots path: no "
          "post-selection, normal-ordered + vacuum encoding)")
    prev = None
    for qc in (2, 3):
        q, p, vac = osc_ops(qc)
        W = cell_collision_unitary_B(tau, qc, 1.0)
        u_err = float(np.linalg.norm(W.conj().T @ W - np.eye((1 << qc) ** 3)))
        rt = worst = 0.0
        for df in (np.array([0.03, -0.05, 0.02]),
                   np.array([0.06, -0.10, 0.04]),
                   np.array([0.15, -0.07, 0.10])):
            psi = encode_cell_B(df, p, vac)
            rt = max(rt, float(np.max(np.abs(decode_cell_B(psi, q) - df))))
            dq = decode_cell_B(W @ psi, q)
            worst = max(worst, float(np.max(np.abs(dq - classical_flow(df, 1.0)))))
        trend = "" if prev is None else f"  ({prev / worst:.1f}x better vs qc-1)"
        prev = worst
        print(f"   qc={qc}: unitary_err={u_err:.1e}  encode/decode={rt:.1e}  "
              f"collision-flow err={worst:.3e}{trend}")

    print("[gate6] shots collision (<q̂> from counts) vs statevector decode")
    qcs = 3
    qsv, psv, vsv = osc_ops(qcs)
    Wsv = cell_collision_unitary_B(tau, qcs, 1.0)
    nshots = 200_000
    worst_sv = worst_cl = 0.0
    for df in (np.array([0.04, -0.06, 0.03]), np.array([0.12, -0.05, 0.08])):
        sv_dec = decode_cell_B(Wsv @ encode_cell_B(df, psv, vsv), qsv)
        sh_dec = cell_collision_shots(df, tau, qcs, 1.0, nshots, seed=7)
        worst_sv = max(worst_sv, float(np.max(np.abs(sh_dec - sv_dec))))
        worst_cl = max(worst_cl, float(np.max(np.abs(sh_dec - classical_flow(df, 1.0)))))
    tol = 5.0 / np.sqrt(nshots)
    print(f"   qc={qcs} shots={nshots}: |shots-statevector|={worst_sv:.3e} "
          f"(tol~{tol:.1e}, {'PASS' if worst_sv < tol else 'CHECK'})  "
          f"|shots-classical|={worst_cl:.3e}")
