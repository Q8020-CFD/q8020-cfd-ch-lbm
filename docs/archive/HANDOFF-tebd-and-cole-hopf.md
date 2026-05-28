# Handoff Spec — TEBD and Cole-Hopf Quantum Burgers

**Scope.** This document consolidates the work remaining on two related
quantum-circuit Burgers solvers:

- **F2** — TEBD-style circuit evolution direct on Burgers via operator
  splitting (`--method tebd_circuit`). Sub-tickets B.1a, B.2, B.3, Phase C,
  Phase D.
- **F10** — Cole-Hopf-linearized quantum-circuit evolution
  (`--method cole_hopf_circuit`). Parcels P1–P9 plus review patches P-A
  through P-F (Patch 01) and P-G, P-H (Patch 02).

**Authoritative source documents** (this handoff supersedes nothing; it
indexes and sequences the existing specs):

- `docs/F2-IMPLEMENTATION-SPEC.md`
- `docs/F2-PHASE-B1a-SPEC.md`
- `docs/F10-IMPLEMENTATION-SPEC.md`
- `docs/F10-REVIEW-PATCH.md`
- `docs/F10-REVIEW-PATCH-02.md`

**Audience.** A developer (or agent team) picking up this work cold. Read
this doc in full, then drop into the per-spec doc for any item you intend
to implement.

---

## 0. Context: two complementary tracks

| | F2 (B route) | F10 (A route) |
|---|---|---|
| Equation evolved | Burgers `u` directly | Heat equation in `φ` (Cole-Hopf) |
| Per-step classical work | O(q · χ³) (build H from `u_n`) | O(1) — fixed gates after setup |
| Per-step quantum work | W-II layer + classical diffusion | One propagator layer (QFT-diag or dense-block) |
| Classical mirror needed? | Yes — H depends on current `u_n` | No — `φ` evolves linearly |
| Quantum-advantage story | No (replaces exp classical with poly classical) | Yes (no per-step classical) |
| Status | B.1 wired but non-physical; B.1a designed | Substantially implemented; patches outstanding |
| Murali consistency | Direct route they describe (route B) | Linearization route they reference (route A) |

The two tracks are **independent** in code (`burgers_tebd.py` /
`burgers_trotter.py:tebd_circuit_step` vs `burgers_cole_hopf_circuit.py`);
F10 has no dependency on F2 ([F10-IMPLEMENTATION-SPEC.md:54](F10-IMPLEMENTATION-SPEC.md:54)).
They can be taken on in parallel.

---

## 1. F2 — TEBD-style circuit evolution

### 1.1 Where we are

| Phase | Scope | Status |
|---|---|---|
| **A** — Classical reference (`tebd`) | Dense H → `expm` → MPO → MPS apply | **Done.** `--method tebd` works, scales to q=12, gif animations exist. |
| **B.1** — Circuit W-II from rank-2 H | Build W-II gates from existing `build_hamiltonian_dense`, polar-unitarize, wrap as `UnitaryGate` | **Wired but non-physical.** Unitarity passes; underlying H is rank-2 + globally dense → W-II fusion produces unitaries that don't approximate `exp(−iH·dt)`. ~25% per-step damping; field collapses in ~5 steps. Phase B.1's docstring acknowledges this: *"B.2+ blocker; Phase B.1's hard acceptance is unitarity alone."* |
| **B.1a** — Physical-Hamiltonian via operator splitting | Replace rank-2 H with ladder-form `H_adv` MPO; classical diffusion sub-step | **Designed, not implemented.** Spec at `F2-PHASE-B1a-SPEC.md`. |
| **B.2** — Wire `tebd_circuit` through framework | `--method tebd_circuit` parallel to other methods, shots/noise/backend reuse | **Mostly done as plumbing** (`tebd_circuit_step` exists, TOML cases run). Gated on B.1a producing meaningful output. |
| **B.3** — Validation and resource metrics | Statevector agreement with Phase A to machine precision; CX/depth metrics for q=3..6 | **Not started.** Cannot start until B.1a lands. |
| **Phase C** — Sign recovery (Hadamard test) | Option-2 phase discernment for shots path | **Not started.** Independent of B.1a/B.2 in design but downstream in time. |
| **Phase D** — Sweep + plotting integration | TOML groups (q=3..6, shots, noise), plot scripts, q8020 metadata fields | **Not started.** Last in sequence. |

### 1.2 B.1a — the unblock

Detailed spec: `F2-PHASE-B1a-SPEC.md`. Summary:

**Why.** The current `build_hamiltonian_dense(u, dx, dt, nu, ...)` returns the
rank-2 minimal Hermitian rotation `A = i·|δ⟩⟨ψ| − i·|ψ⟩⟨δ|`. It is
mathematically correct for the rotation it generates but globally dense, has
no spatial locality, and no identity-pass-through MPO structure. Zaletel
W-II fusion of that operator yields a unitary that bears no resemblance to
`exp(−iA·dt)`.

**What.** Operator-split viscous Burgers into:

```
advection:  ∂_t u + u·∂_x u = 0           (unitary, on circuit)
diffusion:  ∂_t u = ν·∂_xx u              (dissipative, classical mirror)
```

The advection generator

```
H_adv = −(i/2)·(diag(u_n)·D_x + D_x·diag(u_n))             (Eq. 2)
```

is Hermitian and **spatially local** — built as a ladder-form MPO with bond
dim ≤ 4·χ_u (where χ_u is the MPS bond dim of `u_n`). W-II fusion of a
local ladder-form MPO does deliver O(dt³) unitary evolution. Lie-Trotter
shipping first, Strang as follow-up.

**Code surface** (per `F2-PHASE-B1a-SPEC.md` §8):

- *New in `burgers_tebd.py`*: `build_hamiltonian_mpo_ladder`,
  `build_wii_layer_ladder`, `build_shift_mpo`, optional
  `build_laplacian_mpo`.
- *Modified in `burgers_tebd.py`*: B.1 validation block (L~659–776) — swap
  the rank-2 `A` for `build_hamiltonian_mpo_ladder`; re-run unitarity +
  O(dt³) checks.
- *Modified in `burgers_trotter.py`*: `tebd_circuit_step` (L538–644) calls
  the MPO-taking W-II layer and appends classical diffusion update.
- *New in `burgers_nonlinear.py`*: `diffusion_rhs(u, dx, nu, bc)`
  (Laplacian only).

**Acceptance** (per `F2-PHASE-B1a-SPEC.md` §6, all at q=4, dt=1e-4,
ν=1e-2, periodic, sine IC):

1. **Unitarity (hard):** `‖U·U† − I‖_F < 1e-10` for the per-step circuit
   unitary (advection only). Inherited from B.1.
2. **Advection-only accuracy (hard):** at ν=0 with traveling sine,
   fitted slope of `‖err‖_F` vs dt over [1e-4, 1e-3] in [2.8, 3.2].
3. **Full-step accuracy with Lie-Trotter (soft):** vs `shift-Euler`
   over 200 steps — `max|Δu| < 5e-2`, max relative error `< 5%`,
   ‖u‖ stable.
4. **Stability through shock (soft):** run to t=0.2 (past t_shock≈0.16);
   no NaN, no zero-collapse.
5. **Animation reproduction:** regenerate
   `tebd_circuit_comparison_q4_shock.gif`. Green line tracks `shift`/`tebd`
   through the shock.

### 1.3 B.2 — wiring

Mostly already in place ([burgers_trotter.py:539](../../src/burgers_trotter.py:539)
`tebd_circuit_step`; TOML cases `tebd_circuit_q3`, `tebd_circuit_q4` in
`input/`). What B.2 still needs once B.1a is in:

- `tebd_circuit_step` consumes `build_hamiltonian_mpo_ladder` (not the
  rank-2 dense H).
- Classical diffusion sub-step appended after the quantum advection
  decode (Lie-Trotter; Strang as a follow-up).
- Plumbing (shots, noise, backend) inherits unchanged from existing path.

### 1.4 B.3 — validation

Per `F2-IMPLEMENTATION-SPEC.md` §9 Phase B.3:

- `tebd_circuit` (statevector, shots=0) matches `tebd` (Phase A) to
  machine precision at q=3..6.
- Resource metrics report transpiled CX count, total gate count, depth
  for q=3..6 against `quantum_circuit` baseline at the same q.

### 1.5 Phase C — sign recovery (Hadamard test, shots-path)

Per `F2-IMPLEMENTATION-SPEC.md` §8 and §9 Phase C:

- Option-2 readout: maintain a reference state `|φ_n⟩` (the previous
  step's circuit), prepare `|ψ_{n+1}⟩` and `|φ_n⟩` on registers A and
  B with ancilla in `|+⟩`, controlled-SWAP, measure.
- Bootstrapping: `|φ_0⟩` signs known from IC; each snapshot updates the
  reference one step behind.
- Acceptance: shots-mode `tebd_circuit` produces signed profiles tracking
  the classical reference across the shock at x>0.5 (the F9 failure mode).

### 1.6 Phase D — integration + sweep

Per `F2-IMPLEMENTATION-SPEC.md` §9 Phase D:

- TOML groups: `tebd_q{3..6}` (statevector), `tebd_shots_q5`,
  `tebd_noise_q5` (T1/T2).
- Plot scripts: extend `plot_paper_aligned.py` and
  `plot_shots_study.py` with `method=tebd` rows; new
  `plot_tebd_depth.py` (depth and CX-count vs q).
- q8020 metadata: add `chi_max`, `W-II order`, `sign_recovery_method`,
  `ancilla_count` to the analysis fragment.

### 1.7 F2 out of scope

Per `F2-PHASE-B1a-SPEC.md` §7:

- Full quantum diffusion via LCU / block-encoding (that is F10 territory).
- QITE.
- Dirichlet BC for the ladder MPO (TBD; needs boundary bond-closure
  logic).
- Non-power-of-2 N (inherits the existing `q = log₂ N` constraint).

---

## 2. F10 — Cole-Hopf quantum-circuit Burgers

### 2.1 Where we are

The authoritative spec is `F10-IMPLEMENTATION-SPEC.md`. Substantial code
exists in [src/burgers_cole_hopf_circuit.py](../../src/burgers_cole_hopf_circuit.py)
(~1.3k lines) and a test file at
[tests/test_cole_hopf_circuit.py](../../tests/test_cole_hopf_circuit.py).

| Parcel (`F10-IMPLEMENTATION-SPEC.md` §12) | Scope | Status |
|---|---|---|
| **P1** | CH classical hardening + small-ν centering | Implemented (`burgers_cole_hopf.py`). |
| **P2** | Ancilla-conditional Möbius-Ry primitive | Implemented (`build_conditional_ry`). |
| **P3** | `qft-diagonal` propagator per Trotter step | Implemented (`heat_qft_step_circuit`, `heat_qft_full_circuit`). |
| **P4** | `dense-block` propagator (exact eigendecomp + block-encoding) | Implemented (`heat_dense_block_step_circuit`, `heat_dense_block_full_circuit`). |
| **P5** | CLI wiring + `run_cole_hopf_circuit_simulation` | Implemented (statevector path; `--method cole_hopf_circuit`). |
| **P6** | Shots + post-selection + noise | Implemented (`_run_shots_chunked`, `_run_shots_batch`). |
| **P7** | Neumann/Dirichlet adaptation | Implemented (`dense-block + dirichlet`); `qft-diagonal + dirichlet` raises `NotImplementedError`. |
| **P8** | TOML groups, sweep integration, animation | Partially in place — paper-scale small-ν Dirichlet groups still missing (see P-D). |
| **P9** | Tests + acceptance artifacts | Partial — file exists; needs the assertions enumerated in P-B. |

So the algorithm and most of the parcels are landed. The remaining work is
the **two patch documents**, which together close out F10.

### 2.2 Patch 01 (`F10-REVIEW-PATCH.md`) — six items

| Patch | Severity | Title | Status |
|---|---|---|---|
| **P-A** | BLOCKER | Rename or re-implement `pauli-trotter` | Open — pick Fork A (real Pauli-Trotter via SparsePauliOp + commuting-group LCU-of-2) **or** Fork B (rename to `dense-block` and delete acceptance 11.4). Spec recommends Fork A. |
| **P-B** | BLOCKER | Add `test_cole_hopf_circuit.py` assertions | Open — tests for 11.1, 11.2, 11.3, 11.4 (Fork A only), 11.5, 11.6 smoke. |
| **P-C** | SERIOUS | Collapse the shots path to one circuit | Open — current path rebuilds and re-transpiles per snapshot (~50× cost at `save_every=1, n_steps=50`). Pattern 1: transpile once, run N circuits batched. |
| **P-D** | BLOCKER for 11.6 | Paper-scale small-ν Dirichlet TOML groups | Open — add `[paper_cole_hopf_circuit_q{3,4,5}_shots150k]` groups at ν=1e-4, BC=dirichlet, shots=150k. Run q=5 → `paper_cole_hopf_circuit_q5_shots150k.png`. |
| **P-E** | SERIOUS | Resolve unused polynomial fit | Open — Fork E2: drop `fit_theta_polynomial`, build Möbius coefficients directly from exact `θ(k)`. Spec edit: §4 "O(q²)" → `O(2^q)`. |
| **P-F** | BLOCKER (trivial) | Repo hygiene | Open — remove stray `.claude/`, `.DS_Store`, scratch JSONs; add to `.gitignore`. |

Patch-01 dependency chain: P-A → P-B → run/verify; P-C, P-D, P-E, P-F can
run in parallel.

### 2.3 Patch 02 (`F10-REVIEW-PATCH-02.md`) — two parcels

| Patch | Severity | Title | Status |
|---|---|---|---|
| **P-G** | Load-bearing | Wire MPS / Ran 2020 state prep into `cole_hopf_circuit` | Open — current code uses Qiskit's generic `QuantumCircuit.initialize`. Replace with `classical_to_mps + mps_to_circuit` pipeline from `burgers_mps.py`. Thread `--bond-dim` through `run_cole_hopf_circuit_simulation`. Add `use_mps_prep=True` to the SV driver. New TOML group `[paper_cole_hopf_circuit_q5_shots150k_mps]` with bond-dim ∈ {1, 2, 4}. |
| **P-H** | Load-bearing | Peaked-φ shots readout (low-ν regime) | Open — at ν=1e-4 φ concentrates on ~one bin; direct `√counts` scaling explodes in tail. P-H.1 (recommended): Hadamard-test per bin (one ancilla, `2^q` circuits at ~1k shots each). Reuse the F9 `--sign-recovery hadamard_test` machinery; here the test returns amplitude only because φ>0. New CLI flag `--readout {direct, hadamard_per_bin}`. |

Patch-02 parcels are independent of each other; both converge on the same
paper-comparison artifact.

### 2.4 F10 acceptance criteria (consolidated from spec §11)

| # | Criterion | Acceptance artifact |
|---|---|---|
| **11.1** | Classical `--method cole_hopf` matches `--method shift` to 2% L2 at q=5, ν=1e-2, T=0.5·t_shock | `test_cole_hopf_classical.py` PASS |
| **11.2** | `qft-diagonal` SV: `‖φ_circuit − φ_dense‖₂/‖φ_dense‖₂ < 1e-6` at q=4, ν=1e-2, T=0.05, N_steps=10 | `cole_hopf_qft_q4_verify.npz` + plot |
| **11.3** | `dense-block` SV: same case, same tolerance | pytest in `test_cole_hopf_circuit.py` |
| **11.4** | Trotter-error convergence (Fork A only) | log-log slope < −0.9 in N_steps sweep |
| **11.5** | Shots: `‖u_circuit − u_dense‖₂/‖u_dense‖₂ < 0.05` and `P_success > 0.3` at shots=150k | `cole_hopf_shots_q4.png` |
| **11.6** | Small-ν endurance: q ∈ {3,4,5}, ν=1e-4, BC=dirichlet, T=0.8·t_shock, shots=150k | `paper_cole_hopf_circuit_q{3,4,5}_shots150k.png` |
| **11.7** | Sweep + animation | MP4 |
| **11.8** | Noise: `depolarizing_error(p=1e-3)` on 2-qubit gates at 11.6 q=5; recognizable at T=0.5·t_shock | `noise_sensitivity.png` |

### 2.5 F10 §14 reviewer checklist

Repeated here for handoff convenience; gate F10 close-out on this passing:

- [ ] No branch of `cole_hopf_circuit` reads `u` state inside a timestep.
- [ ] `φ̂` from readout is positive everywhere before inverse-CH.
- [ ] Post-selection `P_success` is logged and surfaced in the NPZ.
- [ ] Classical `cole_hopf` pipeline is unchanged by this work.
- [ ] `method=tebd_circuit` is not touched (F10 has no F2 dep).
- [ ] `--bc dirichlet` works via `dense-block` or fails with a clear
      `NotImplementedError` in `qft-diagonal`.
- [ ] Small-ν centering (§9) is on-by-default when applicable, with a
      log line explaining the choice.
- [ ] Prepared ψ matches `reconstruct_from_mps(classical_to_mps(ψ₀,
      canonical="right"))` to 1e-12 at full rank (P-G).
- [ ] `--bond-dim` visibly truncates in both shots and SV paths (P-G).
- [ ] No new `.claude/` dir in repo; PEP 8 + 88-char lines + venv use.

### 2.6 F10 out of scope

Per `F10-IMPLEMENTATION-SPEC.md` §13:

- Encoding change (binary amplitude is fixed for F10).
- Direct u-space evolution (Carleman or equivalent).
- DST-based `qft-diagonal + dirichlet`.
- Hardware execution.
- QSVT-polynomial alternative.
- F11 Burgulence (depends on F10 being stable at small ν).

---

## 3. Recommended sequencing

Two independent tracks; pick the one that matches the team's time budget,
or run both in parallel with separate owners.

### Track 1 — Close out F10 (fast path; mostly merge work)

```
P-F  ──┐
P-A  ──┤
P-E  ──┼──▶ P-B ──┐
P-C  ──┤          │
P-D  ──┘          │
                  ├──▶ P-G ──┐
                  │          ├──▶ run paper_cole_hopf_circuit_q5_shots150k_mps
                  │          │    & paper_q5_shots150k → F10 closed
                  └──▶ P-H ──┘
```

P-F + P-A + P-C + P-D + P-E in parallel. P-A's fork choice (A vs B)
constrains P-B's flag names and acceptance set; once an owner decides,
P-B starts. P-G + P-H run after Patch-01 settles.

**Estimated effort**: 1–2 weeks for an experienced agent if Fork B is taken
on P-A; 3–4 weeks if Fork A (real Pauli-Trotter LCU-of-2) is taken.

### Track 2 — Land F2 B.1a + B.2 + B.3

```
B.1a (build_hamiltonian_mpo_ladder + build_wii_layer_ladder + diffusion_rhs)
   ──▶ B.2 (wire tebd_circuit_step to ladder MPO + classical diffusion)
       ──▶ B.3 (validation: SV match Phase A; CX/depth metrics q=3..6)
           ──▶ Phase C (sign recovery) ──▶ Phase D (sweep + plots + metadata)
```

Single-track sequential. No safe parallelism inside F2 until B.3 lands.

**Estimated effort**: 2–3 weeks for B.1a + B.2 + B.3. Phase C and D add
another 1–2 weeks.

### Cross-track gating

- **F10 is the quantum-advantage story.** If competing for resources,
  prioritize Track 1.
- **F2 is the route Murali specifically requested.** It is also the
  cheaper sequel for hardware-relevance studies (W-II layers are
  shallower than `qft-diagonal` × N_steps).
- The two tracks share **no code or test surface**; landing one cannot
  break the other.

---

## 4. Work breakdown

Tasks are atomic units of work — one PR each, owned by one person.
Effort estimates assume an experienced developer working with prior
context (e.g. an agent briefed on the relevant per-spec doc).

### 4.1 F2 track

| ID | Task | Depends on | Effort | Deliverable |
|---|---|---|---|---|
| F2-1 | Implement `build_shift_mpo(q, direction)` (bond-2 increment/decrement on periodic BC) | — | 0.5d | New function in `burgers_tebd.py` + unit test (matches dense shift to 1e-12 at q=3,4,5). |
| F2-2 | Implement `build_hamiltonian_mpo_ladder(u, dx, bc)` (Eq. 2 in B.1a spec §3.4) | F2-1 | 1d | New function + Hermiticity test + bond-dim ≤ 4·χ_u sanity. |
| F2-3 | Implement `build_wii_layer_ladder(H_mpo, dt)` (Zaletel W-II on ladder MPO) | F2-2 | 1d | Per-site gates + unit test reconstructing `exp(−iHdt)` to err < 10·dt³ on toy bond-2 H. |
| F2-4 | Implement `diffusion_rhs(u, dx, nu, bc)` in `burgers_nonlinear.py` (pure Laplacian) | — | 0.5d | New helper + parity test vs existing `compute_rhs_shift` minus advection. |
| F2-5 | Refactor `tebd_circuit_step` to call `build_hamiltonian_mpo_ladder` + W-II ladder + classical diffusion (Lie-Trotter) | F2-3, F2-4 | 1d | `burgers_trotter.py:539` updated; existing TOML cases run end-to-end. |
| F2-6 | B.1a acceptance harness — five tests in B.1a spec §6 (unitarity, advection-only O(dt³), full-step accuracy, shock stability, animation) | F2-5 | 1.5d | New `tests/test_tebd_phase_b1a.py` + regenerated gif. |
| F2-7 | Strang variant of F2-5 (half-step diffusion + full advection + half-step diffusion) | F2-5 | 0.5d | Optional kwarg `splitting={"lie","strang"}`; default `lie`. |
| F2-8 | B.3 validation — SV match Phase A to machine precision at q=3..6 | F2-5 | 0.5d | Test in `test_tebd_phase_b1a.py`. |
| F2-9 | B.3 resource metrics — transpiled CX, total gates, depth at q=3..6 vs `quantum_circuit` baseline | F2-5 | 0.5d | New plot script `plot_tebd_depth.py` (depth/CX vs q). |
| F2-10 | Phase C — Hadamard-test sign recovery for `tebd_circuit` shots path | F2-5 | 2d | Reuse F9 `--sign-recovery hadamard_test` machinery; new test that signed profiles track classical reference at x>0.5. |
| F2-11 | Phase D — TOML groups `tebd_q{3..6}`, `tebd_shots_q5`, `tebd_noise_q5` | F2-5, F2-10 | 0.5d | Updates to `input/burgers_quantum.toml`. |
| F2-12 | Phase D — extend `plot_paper_aligned.py`, `plot_shots_study.py` with `method=tebd` rows | F2-11 | 0.5d | Updated plot scripts; sample outputs. |
| F2-13 | Phase D — q8020 metadata: add `chi_max`, `wii_order`, `sign_recovery_method`, `ancilla_count` to analysis fragment | F2-11 | 0.5d | Schema update in `q8020-cfd-metautil` + harvester wiring. |

**F2 total:** ≈10 days sequential, ≈6 days with F2-1/F2-4 parallel and
F2-7/F2-9 deferred.

### 4.2 F10 track

| ID | Task | Depends on | Effort | Deliverable |
|---|---|---|---|---|
| F10-1 | P-F repo hygiene — remove stray `.claude/`, `.DS_Store`, scratch JSONs; update `.gitignore` | — | 0.25d | Clean `git status`. |
| F10-2 | P-A decision — pick Fork A (real Pauli-Trotter via SparsePauliOp + commuting-group LCU-of-2) or Fork B (rename to `dense-block`) | — | — | Decision + rationale documented in PR description. |
| F10-3a | P-A Fork B — rename `pauli-trotter` → `dense-block` across CLI, function names, TOML, spec; delete acceptance 11.4 | F10-2 (B chosen) | 0.5d | Rename diff + spec edits to §2/§4/§6/§11. |
| F10-3b | P-A Fork A — implement Pauli-Trotter via SparsePauliOp + `group_commuting` + LCU-of-2 per group | F10-2 (A chosen) | 4d | New propagator path + first-order convergence test (slope < −0.9 in log-log N_steps sweep). |
| F10-4 | P-E — drop `fit_theta_polynomial`, build Möbius coefficients directly from `compute_theta_exact`; spec edit §4 "O(q²)" → `O(2^q)` | — | 0.5d | Smaller `burgers_cole_hopf_circuit.py`; no behavior change at full precision. |
| F10-5 | P-C — collapse shots path to one transpile + N_steps runs (Pattern 1) | — | 1d | One `AerSimulator()` and one `transpile()` per `run_cole_hopf_circuit_simulation` invocation. |
| F10-6 | P-D — paper-scale TOML groups `paper_cole_hopf_circuit_q{3,4,5}_shots150k` (ν=1e-4, BC=dirichlet, shots=150k) | F10-3a or F10-3b | 0.25d | TOML additions + `_group_postproc`. |
| F10-7 | P-D q=5 sweep run + artifact | F10-6 | 0.5d | `paper_cole_hopf_circuit_q5_shots150k.png` showing forming shock. |
| F10-8 | P-B test harness — assertions for 11.1, 11.2, 11.3, 11.4 (Fork A only), 11.5 (slow), 11.6 smoke | F10-3a or F10-3b, F10-4 | 1.5d | `tests/test_cole_hopf_circuit.py` PASS in <2 min (non-slow), <30 min (slow). |
| F10-9 | P-G — replace `QuantumCircuit.initialize` with `classical_to_mps + mps_to_circuit` Ran-2020 prep | F10-5 | 1.5d | Updated `run_cole_hopf_circuit_simulation`, `_run_shots_batch`; `--bond-dim` plumbed; `use_mps_prep=True` flag on SV driver. |
| F10-10 | P-G TOML group `paper_cole_hopf_circuit_q5_shots150k_mps` with bond-dim ∈ {1,2,4} sweep | F10-9 | 0.25d | TOML additions. |
| F10-11 | P-G acceptance tests — `test_mps_prep_used` (psi matches Ran-2020 reconstruction to 1e-12) and `test_bond_dim_truncation` (bond_dim=1 differs from full rank) | F10-9 | 0.5d | Two new tests in `test_cole_hopf_circuit.py`. |
| F10-12 | P-H Hadamard-test-per-bin readout for low-ν shots regime; reuse F9 sign-recovery machinery | F10-5 | 2d | New `--readout {direct, hadamard_per_bin}` flag; default `direct`. Updated 11.5 test at ν=1e-2 (no nu=0.1 deviation). |
| F10-13 | P-H spec edits §7.A (Hadamard-test readout) + §10 CLI `--readout` flag | F10-12 | 0.25d | Spec text. |
| F10-14 | F10 §14 reviewer checklist run — all ten items PASS | F10-3*, F10-5, F10-7, F10-8, F10-9, F10-11, F10-12 | 0.5d | Checklist with ticks; F10 close-out. |

**F10 total:** ≈9 days with Fork B (P-A); ≈12.5 days with Fork A.
Parallelizable to ≈5 days wall-clock if owner can split P-C/P-D/P-E/P-F
across agents and run F10-9 + F10-12 concurrently after P-A settles.

### 4.3 Critical path and parallelism

**Track 1 (F10):**
- Earliest start: F10-1, F10-2, F10-4, F10-5 in parallel (day 1).
- F10-3 starts when F10-2 decision is made.
- F10-6 needs F10-3 (TOML propagator name depends on Fork A/B choice).
- F10-8 needs F10-3 + F10-4 (test acceptance set depends on Fork; Möbius
  edit is needed before tests can pin numbers).
- F10-9 and F10-12 can run in parallel after F10-5 lands.
- F10-14 is the last gate.

**Track 2 (F2):**
- F2-1 and F2-4 in parallel (day 1).
- F2-2 needs F2-1; F2-3 needs F2-2; F2-5 needs F2-3+F2-4. Strict chain.
- F2-6 (acceptance) depends on F2-5.
- F2-7 (Strang) and F2-9 (resource metrics) can run in parallel with
  F2-6 once F2-5 lands.
- F2-10 (Phase C) and F2-11 (Phase D) are independent of each other but
  both downstream of F2-6.

**Cross-track:** zero shared code or test surface — Track 1 and Track 2
parallelize cleanly with separate owners.

### 4.4 Definition of done (per track)

**F2 done when:**
- F2-1 through F2-9 PRs merged.
- All five B.1a acceptance criteria pass.
- B.3 reports CX/depth metrics for q=3..6.
- (Optional, depending on scope cut) F2-10 through F2-13 also merged.

**F10 done when:**
- F10-1 through F10-14 PRs merged.
- All eight acceptance items in spec §11 produce their named artifacts.
- §14 reviewer checklist passes end-to-end.
- `paper_cole_hopf_circuit_q5_shots150k_mps` runs to completion and
  the postproc PNG shows the three bond-dim curves converging to the
  classical reference.

---

## 5. Validation matrix

| Item | F2 source | F10 source |
|---|---|---|
| Unit tests for new primitives | B.1a §6.1, §6.2 (`build_hamiltonian_mpo_ladder`, `build_wii_layer_ladder`) | P-B (acceptance items 11.1–11.6) |
| End-to-end SV agreement | B.3 (vs Phase A `tebd`, machine precision) | 11.2, 11.3 (vs `build_heat_propagator` dense, < 1e-6) |
| Shots-path agreement | Phase C (signed profiles vs classical) | 11.5 (`< 0.05` rel L2 at 150k shots) |
| Resource metrics | B.3 (CX count, depth, q=3..6) | §11 plus P-G bond-dim study |
| Noise robustness | Phase D (`tebd_noise_q5`) | 11.8 (`depolarizing_error(p=1e-3)`) |
| Paper-comparison artifact | Phase D animations | 11.6 + 11.7 |

---

## 6. Files touched (consolidated)

### F2 (B.1a + B.2 + B.3)

- `src/burgers_tebd.py` — new MPO builders; B.1 validation block updated.
- `src/burgers_trotter.py` — `tebd_circuit_step` calls MPO W-II + classical
  diffusion.
- `src/burgers_nonlinear.py` — new `diffusion_rhs`.
- `tests/test_tebd*.py` — new acceptance harness.
- `input/burgers_quantum.toml` — new `tebd_circuit_q{3..6}` groups.
- `docs/F2-PHASE-B1a-SPEC.md` — already authoritative; mark "implemented"
  on completion.

### F10 patches

- `src/burgers_cole_hopf_circuit.py` — P-A (Fork A or rename), P-C (single
  transpile), P-E (Möbius-only), P-G (MPS prep wiring), P-H (Hadamard-test
  readout).
- `src/burgers_solver.py` — P-A (rename or new propagator name), P-G
  (`--bond-dim` thread-through), P-H (`--readout` flag).
- `src/burgers_trotter.py` — P-G (`bond_dim` plumbing into
  `run_cole_hopf_circuit_simulation`).
- `tests/test_cole_hopf_circuit.py` — P-B (full acceptance harness),
  P-G (`test_mps_prep_used`, `test_bond_dim_truncation`).
- `input/burgers_quantum.toml` — P-D (paper-scale small-ν Dirichlet
  groups), P-G (`paper_cole_hopf_circuit_q5_shots150k_mps`).
- `docs/F10-IMPLEMENTATION-SPEC.md` — P-A, P-E, P-G spec edits.
- `.gitignore` — P-F.

---

## 7. Out-of-scope items (do not implement here)

Tracked in `FUTURE-WORK.md`:

- F11 Burgulence (depends on F10 small-ν stability).
- Forced Burgulence (F11.5).
- Encoding change (one-hot, block-encoded LCU shift, etc.).
- DST-based `qft-diagonal + dirichlet`.
- QSVT-polynomial alternative to ancilla-Ry.
- QROM-based θ(k) loading (q ≥ 7).
- Hardware execution.
- Direct u-space evolution via Carleman.
- F4 (Variational fast-forwarding).
- F5 (Krylov-MPS).
- F8 (Walters multimodal q=13 case).

---

## 8. Reference

- Strang, G. (1968). SIAM J. Numer. Anal. 5(3), 506–517.
- Vidal, G. (2003). Phys. Rev. Lett. 91, 147902. (TEBD)
- Vidal, G. (2004). Phys. Rev. Lett. 93, 040502. (TEBD/MPS)
- Zaletel et al. Phys. Rev. B 91, 165112 (2015). (W-II)
- Cole (1951); Hopf (1950). (CH transform)
- Ran, Phys. Rev. A 101 (2020). (MPS → circuit state preparation)
- Liu et al., PNAS 2023; Childs, Liu, Ostrander, arXiv:2011.06571.
  (Quantum algorithms for linearized nonlinear PDEs.)
- Meena, Murali et al., AIAA SciTech 2026. (Source paper for IC, ν, BC,
  CFL, shock-time conventions, shots count.)
