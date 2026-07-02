"""QALB (pure-quantum D1Q3, Phase 2) validation gates.

Migrated from the lib_qalb_circuit.py __main__ smoke block.  Each
"gate" checks one property of the App C / App B construction to the
precision the paper claims:

- gate1: finite-position embedding q|x>=x|x>, exact linear readout
- gate2: single-cell collision flow generator vs classical BGK flow
- gate3: full-lattice QALB run stays smooth and decays
- gate4: block-encoded per-site collision circuit vs the operator
- gate5: App B Hermitised unitary collision (shots path, no ancilla)
- gate6: shots <q-hat> readout vs statevector decode
- gate7: #27.1 Trotter synthesis converges to the dense unitary

Transpile-heavy gates (4, 7) and the full-lattice run (3) are marked
``slow``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.integrate import solve_ivp

from lib_qalb_circuit import (
    F_EQ0,
    cell_collision_circuit,
    cell_collision_gate,
    cell_collision_shots,
    cell_collision_unitary_B,
    collision_flow_generator,
    collision_hamiltonian_pauli,
    decode_cell,
    decode_cell_B,
    decode_value,
    encode_cell,
    encode_cell_B,
    encode_value,
    equilibrium,
    finite_position_ops,
    osc_ops,
    run_qalb_simulation,
)

TAU = 2.42


def classical_flow(df: np.ndarray, T: float) -> np.ndarray:
    """Reference single-cell D1Q3 BGK relaxation, integrated to time T."""
    inv = -1.0 / TAU

    def rhs(t, y):
        sm = y[0] - y[2]
        return np.array([
            inv * (y[0] - 0.5 * (sm * sm + sm)),
            inv * (y[1] + sm * sm),
            inv * (y[2] - 0.5 * (sm * sm - sm)),
        ])

    return solve_ivp(rhs, [0, T], df, rtol=1e-11, atol=1e-13).y[:, -1]


def test_gate1_finite_position_embedding():
    """q|x>=x|x> eigen-error shrinks with qc; decode/translate exact."""
    eig_errs = []
    for qc in (2, 3, 4):
        q, D, polys = finite_position_ops(qc)
        e_eig = max(
            np.max(np.abs(q @ encode_value(x, polys)
                          - x * encode_value(x, polys)))
            for x in (-0.3, -0.1, 0.15, 0.3)
        )
        e_dec = max(abs(decode_value(encode_value(x, polys)) - x)
                    for x in (-0.3, -0.1, 0.15, 0.3))
        e_tr = max(
            abs(decode_value(expm(dl * D) @ encode_value(x, polys)) - (x + dl))
            for x in (-0.1, 0.0, 0.1) for dl in (0.05, 0.1)
        )
        eig_errs.append(e_eig)
        assert e_dec < 1e-12       # linear readout is exact
        assert e_tr < 1e-12        # translation generator is exact

    # eigen-error decreases monotonically as the Fock space grows
    assert eig_errs[0] > eig_errs[1] > eig_errs[2]
    assert eig_errs[2] < 1e-4      # qc=4 is essentially converged


def test_gate2_collision_flow_generator():
    """Flow generator matches classical BGK flow; qc>=3 is converged."""
    worst_by_qc = {}
    for qc in (2, 3, 4):
        q, D, polys = finite_position_ops(qc)
        G = collision_flow_generator(q, D, TAU)
        U = expm(1.0 * G)
        worst = 0.0
        for df in (np.array([0.03, -0.05, 0.02]),
                   np.array([0.06, -0.10, 0.04]),
                   equilibrium(np.array([0.1])).ravel() - F_EQ0):
            dq = decode_cell(U @ encode_cell(df, polys), qc)
            worst = max(worst, float(np.max(np.abs(dq - classical_flow(df, 1.0)))))
        worst_by_qc[qc] = worst

    assert worst_by_qc[3] < 1e-9   # converged to machine precision
    assert worst_by_qc[4] < 1e-9
    assert worst_by_qc[2] > worst_by_qc[3]   # qc=2 truncation-limited


@pytest.mark.slow
def test_gate3_full_lattice_run_smooth_and_decays():
    """Full pure-quantum QALB run stays smooth and loses amplitude."""
    N = 32
    xg = np.linspace(0.0, 1.0, N, endpoint=False)
    dxg = xg[1] - xg[0]
    u0 = 0.3 * np.sin(2 * np.pi * xg)
    sols, _mets, _gs = run_qalb_simulation(
        u0, xg, nu=3e-2, dt=dxg, n_steps=10, bc="periodic", qc=3,
    )
    uf = sols[-1]
    assert np.all(np.isfinite(uf))
    assert np.max(np.abs(np.diff(uf, append=uf[:1]))) < 0.2   # smooth
    assert np.max(np.abs(uf)) < np.max(np.abs(u0))            # decayed


@pytest.mark.slow
def test_gate4_block_encoded_circuit_matches_operator():
    """Post-selected block-encoded circuit reproduces the cell operator."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    qcf = 2
    tcol = -TAU * np.log(1.0 - 1.0 / TAU)
    qop, D, polys = finite_position_ops(qcf)
    U_cell = expm(tcol * collision_flow_generator(qop, D, TAU))
    circ, _alpha = cell_collision_circuit(TAU, qcf, tcol)
    n_sys = 3 * qcf

    df = np.array([0.04, -0.06, 0.03])
    psi = encode_cell(df, polys)
    psi = psi / np.linalg.norm(psi)

    full = QuantumCircuit(n_sys + 1)
    full.initialize(psi.tolist(), range(n_sys))
    full.compose(circ, range(n_sys + 1), inplace=True)
    sv = np.asarray(Statevector(full).data)
    block = sv[:1 << n_sys]                    # ancilla |0> block

    dq_circ = decode_cell(block, qcf)
    dq_op = decode_cell(U_cell @ encode_cell(df, polys), qcf)
    assert float(np.max(np.abs(dq_circ - dq_op))) < 1e-10
    assert float(np.sum(np.abs(block) ** 2)) > 0.0    # non-zero post-select


def test_gate5_hermitised_unitary_collision():
    """App B unitary is unitary; collision-flow error improves with qc."""
    prev = None
    for qc in (2, 3):
        q, p, vac = osc_ops(qc)
        W = cell_collision_unitary_B(TAU, qc, 1.0)
        u_err = float(np.linalg.norm(W.conj().T @ W - np.eye((1 << qc) ** 3)))
        assert u_err < 1e-10       # genuinely unitary

        rt = worst = 0.0
        for df in (np.array([0.03, -0.05, 0.02]),
                   np.array([0.06, -0.10, 0.04]),
                   np.array([0.15, -0.07, 0.10])):
            psi = encode_cell_B(df, p, vac)
            rt = max(rt, float(np.max(np.abs(decode_cell_B(psi, q) - df))))
            dq = decode_cell_B(W @ psi, q)
            worst = max(worst, float(np.max(np.abs(dq - classical_flow(df, 1.0)))))

        assert rt < 1e-6           # encode/decode round-trip
        if prev is not None:
            assert worst < prev    # higher qc reduces truncation error
        prev = worst


def test_gate6_shots_readout_matches_statevector():
    """<q-hat> from counts agrees with the statevector decode within tol."""
    qcs = 3
    qsv, psv, vsv = osc_ops(qcs)
    Wsv = cell_collision_unitary_B(TAU, qcs, 1.0)
    nshots = 200_000
    worst_sv = 0.0
    for df in (np.array([0.04, -0.06, 0.03]), np.array([0.12, -0.05, 0.08])):
        sv_dec = decode_cell_B(Wsv @ encode_cell_B(df, psv, vsv), qsv)
        sh_dec = cell_collision_shots(df, TAU, qcs, 1.0, nshots, seed=7)
        worst_sv = max(worst_sv, float(np.max(np.abs(sh_dec - sv_dec))))

    tol = 5.0 / np.sqrt(nshots)
    assert worst_sv < tol


@pytest.mark.slow
def test_gate7_trotter_synthesis_converges():
    """Trotter synthesis of e^{-iT H'} converges to the dense unitary."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    qc7 = 2
    n7 = 3 * qc7
    q7, p7, vac7 = osc_ops(qc7)
    spo = collision_hamiltonian_pauli(TAU, qc7)
    assert len(spo) > 0            # H' decomposes into Pauli terms

    W_dense = cell_collision_unitary_B(TAU, qc7, 1.0)
    dfs7 = (np.array([0.03, -0.05, 0.02]), np.array([0.06, -0.10, 0.04]),
            np.array([0.15, -0.07, 0.10]))

    def decode_dense(df):
        return decode_cell_B(W_dense @ encode_cell_B(df, p7, vac7), q7)

    # qc=2 Fock-truncation floor: the best any exact operator can do
    floor = max(
        float(np.max(np.abs(decode_dense(df) - classical_flow(df, 1.0))))
        for df in dfs7
    )

    errs = {}
    for order, reps in ((2, 1), (2, 2), (2, 4), (2, 8)):
        gate = cell_collision_gate(TAU, qc7, 1.0, reps, order)
        tc = QuantumCircuit(n7)
        tc.append(gate, range(n7))
        Wt = Operator(tc.decompose(reps=4)).data
        errs[reps] = max(
            float(np.max(np.abs(
                decode_cell_B(Wt @ encode_cell_B(df, p7, vac7), q7)
                - decode_dense(df))))
            for df in dfs7
        )

    # more reps -> lower error, converging below the truncation floor
    assert errs[1] > errs[2] > errs[4] > errs[8]
    assert errs[2] < floor
    assert errs[8] < floor
