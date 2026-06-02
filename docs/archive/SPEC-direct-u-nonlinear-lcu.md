# SPEC: Direct-u Nonlinear LCU via Frozen-Coefficient Measure-Reprepare

Date: 2026-06-01
Paper: Gopalakrishnan Meena et al., AIAA-2026 ("MPS/MPO methods for
1D Burgers"); ladder operators §III.B.2 Eqs 9-12.
Related specs: `SPEC-F3-LCU-method.md` (CH-LCU / heat propagator),
`SPEC-measure-reprepare-evolution.md`, `SPEC-encoding-switch.md`,
`DEEP-OVERLAP-murali-vs-ucan.md`.

## 1. Motivation

The Pauli-decomposition path (`quantum_circuit`) confronts the Burgers
nonlinearity by absorbing it into a dense Hamiltonian and decomposing
into Pauli strings: O(4^q) terms. It does not scale.

The paper's primary method is MPO-on-MPS, not Pauli. Turning that into
a circuit is the open problem the paper itself names. The route taken
here is the paper's Appendix-A bridge -- Linear Combination of
Unitaries (LCU) -- applied to the **actual nonlinear evolution
operator** (not the Cole-Hopf heat propagator), evolving the velocity
field `u` directly.

This is distinct from the existing `cole_hopf_circuit` path, which
linearizes Burgers into the heat equation and so sidesteps the
nonlinear term. Here we keep the nonlinear operator and solve it
iteratively.

## 2. Hard constraint: no classical shadow

A **classical shadow** is a parallel classical solution of the field
amplitudes maintained in lock-step with the quantum state to feed
state-dependent operators (e.g. the F2 TEBD path's `shift_euler_step`
mirror that rebuilds H_n classically each step). This is forbidden.

**Measure-reprepare is NOT a classical shadow.** Reading out the
quantum state at a segment boundary, reconstructing the field, and
re-preparing a fresh circuit is iterative solving: the classical data
is the *measured output of a genuine quantum computation*, not a
parallel integrator running alongside it. This is the same mechanism
the CH path already uses (`evolution-mode = measure_reprepare`).

Consequence: the advection coefficient `diag(u_seg)` at the start of a
segment is **classically known from the measurement**, so it is built
as a known-value diagonal block-encoding -- not an amplitude-loading
oracle on an unknown state. This sidesteps the no-cloning obstacle
entirely (we never copy an unknown state; we re-prepare a known one).

## 3. Method

Direct-`u`, frozen-coefficient advection-diffusion on a
measure-reprepare loop:

1. **Reprepare** `|psi_seg> = sum_i u_i |i>` from the measured
   velocity field via Ran-2020 MPS prep (`classical_to_mps` ->
   `mps_to_circuit`). `--bond-dim` truncates the MPS of `u`.
2. **Freeze** the generator over the segment:
   `A_seg = nu*L - (1/2)*G*diag(u_seg)` (conservative flux form),
   built from the analytic ladder primitives (`L`, `G` from S+/S-)
   and the known diagonal `diag(u_seg)`.
3. **Evolve** coherently for `k` steps via a Taylor-LCU of
   `exp(A_seg * dt)` (see §5).
4. **Measure**, reconstruct `u`, return to step 1.

The frozen coefficient is the only approximation: a lagged (IMEX-style)
nonlinearity refreshed every `k` steps by measurement.

## 4. Encoding and state

- **Operator encoding: binary.** S+/S- are binary increment/decrement
  (`burgers_mpo.py`), matching the paper's Eqs 11-12. `G` and `L` are
  LCU of these. Gray encoding is deferred (only needed for the W-II MPO
  compilation, not the LCU path; tracked in `SPEC-encoding-switch.md`).
- **State: MPS of the velocity field `u`** (not the Cole-Hopf field).
  Ran-2020 prep; bond-dim knob carries over from the CH path.

## 5. Math: Taylor-LCU of the frozen generator

Semi-discrete Burgers in conservative (divergence) form:
`du/dt = nu*L*u - (1/2)*G*(u^2) = nu*L*u - (1/2)*G*diag(u)*u = A(u)*u`
(using `u^2 = diag(u)*u` elementwise). Freeze
`A_seg = nu*L - (1/2)*G*diag(u_seg)` over a segment. Over `k` steps
the field evolves by `exp(A_seg * dt)^k` (linear ODE, frozen `A`).

The advective form `nu*L - diag(u_seg)*G` is the same two factors in
swapped order (gradient before vs after the diagonal): identical
construction cost. It is the smooth-flow-only variant -- it does not
give the correct weak/shock solution -- so conservative is the
baseline.

Taylor expand one step:
```
exp(A_seg * dt) = sum_{m=0}^{M} (dt^m / m!) * A_seg^m
```
Each `A_seg^m` expands into products of:
- shift unitaries S+, S-, I (from `L`, `G`) -- unitary, and
- the diagonal `diag(u_seg)` -- non-unitary; block-encoded from its
  classically-known values (Mottonen UCRy, as in
  `diag_potential_block_encoding`), subnormalized by `||u_seg||_inf`.

Compose via the standard LCU recipe (`SPEC-F3-LCU-method.md` §4.1):
- **product** of block-encodings (within a power `A_seg^m`):
  subnormalization factors multiply,
- **sum** over the `M+1` Taylor terms: outer LCU layer
  (`lcu_block_encoding`).

Block-encoded operator is `exp(A_seg dt) / lambda`; post-select the
ancilla on `|0>`; success probability `~ ||exp(A_seg dt) psi||^2 /
lambda^2`. Per-factor subnormalization is
`lambda_A ~ nu*||L|| + (1/2)*||u_seg||_inf*||G||`, compounded through
the powers and the Taylor sum -- this is the dominant scaling
limiter (§8).

## 6. Components (reuse map)

| Piece | Status |
|---|---|
| `lcu_block_encoding` (SELECT/PREPARE, UnitaryGate-materialized) | exists (`burgers_lcu.py`) |
| S+/S- -> `G`, `L`; `_build_net_shift_circuit` | exists (`burgers_mpo.py`, `burgers_lcu.py`) |
| Known-diagonal block-encoding (Mottonen UCRy) | exists (`diag_potential_block_encoding`) -- adapt to encode `diag(u_seg)` |
| MPS prep of a field (Ran 2020) | exists (`classical_to_mps`, `mps_to_circuit`) |
| Measure-reprepare segmentation | exists (`_run_shots_measure_reprepare`, CH/phi) -- need direct-`u` variant |
| `advection_diffusion_taylor_lcu_terms` (Taylor of `exp(A_seg dt)` incl. `diag(u)*G`) | NEW -- mirrors `heat_taylor_lcu_terms` |
| Solver wiring: new propagator on the direct-`u` method | NEW |

## 7. Parcels (staged)

- **M1 -- Generator + single-segment SV correctness.** Build
  `advection_diffusion_taylor_lcu_terms`; block-encode
  `exp(A_seg dt)`; apply one segment to a sine IC at q=3..5,
  statevector, no shots. Acceptance: matches a classical conservative
  reference step of `nu*u_xx - (1/2)*d/dx(u^2)` (same spatial
  discretization; Godunov flux for the shock-resolving cases) within
  Taylor tolerance.
  *Status (done):* `advection_diffusion_taylor_lcu_terms` +
  `conservative_burgers_lcu_step_circuit` in `burgers_lcu.py`.
  Verified: `A@u` equals the conservative RHS to ~1e-15; the term sum
  equals the truncated Taylor `P_M` to ~1e-15 and is real; the full
  LCU circuit block reproduces `expm(A*dt)` to Taylor tolerance at
  q=3,4,5.  **Finding:** unlike the heat path, the diagonal phases do
  not collapse to net-shifts, so the flattened dense-SELECT term count
  grows fast (K ~ 37 / 167 / ~680 at taylor_order 2 / 3 / 4,
  q-independent), with `lambda ~ exp(dt*lambda_A)` and
  `lambda_A ~ 1/dx^2`.  The flattened circuit is feasible only at
  small `q*M` (validated at M=2 for q<=5); higher order needs the
  nested encoder (M4).
- **M2 -- Measure-reprepare loop (SV).** Direct-`u` measure-reprepare
  variant; sweep segment size `k`. Acceptance: frozen-coefficient
  error vs `k` characterized; full trajectory tracks FTCS within a
  stated tolerance for small `k`.
  *Status (done):* `burgers_direct_lcu.py`
  (`run_direct_lcu_simulation`) + `DirectLCUIntegrator` wired into
  `burgers_fw.py` and `--method direct_lcu` in the solver.  SV path
  applies the block-encoded operator
  (`conservative_burgers_lcu_operator`) directly.  Verified at q=5,
  nu=1e-2: k=1 matches the per-step expm reference to ~2e-12; frozen
  error grows monotonically with k (1.6e-4 at k=2 -> 6.5e-3 at k=40);
  Taylor convergence at k=1 (4.8e-7 / 1.1e-9 / 2.3e-12 for order
  2/3/4).  End-to-end CLI verified incl. the `--bond-dim` MPS-of-`u`
  truncation path.  M4 sweep TOML: `input/burgers_direct_lcu.toml`.
- **M3 -- Shots path.** Reconstruction + post-selection; report
  per-segment success probability and circuit cost (depth/cx, honest
  metric-basis lowering).
  *Status (done):* `_run_direct_lcu_shots` in `burgers_direct_lcu.py`,
  routed from `run_direct_lcu_simulation` when shots>0 and wired
  through `DirectLCUIntegrator` (backend + `--sign-recovery`).  One
  segment-spanning block-encoding of `exp(A_seg*T_segment)` per
  segment (A frozen); post-select ancilla=|0> via the shared
  `post_select_counts`; magnitude reconstruct; renormalise by
  `lambda*sqrt(p_success)`.  Requires `n_steps % segment_size == 0`
  (same as CH measure_reprepare; use `--auto-cadence`).  Verified at
  q=3,4: oracle-signed result converges to the statevector limit as
  shots grow (rel err 3e-2 -> 3.7e-3 over 5k -> 500k), p_success and
  lambda tracked, snapshots length n_steps+1.  Sign recovery: `none`
  (magnitudes), `classical_oracle` (DIAGNOSTIC, signs from the dense
  operator), and **`hadamard_test` (shadow-free)** all implemented;
  `dual_rail` is a follow-up.  Hadamard recovery (`_direct_lcu_hadamard_
  signs`) is the F9 interferometric test generalised to the
  block-encoding -- ONE extra circuit per segment (not per-bin):
  controlled-U_BE on a test ancilla against the signed reference,
  post-select the block ancilla, sign = sign(p0-p1)*sign(ref); the
  reference signs propagate from the IC each segment (no classical
  RHS).  The controlled block-encoding is materialised as a dense
  block_diag(I, U_BE) UnitaryGate (Aer can't assemble a controlled
  composite) -- same trick as QLBM, sim-only, small-q.  Verified at
  q=3 over 2 segments: sign agreement 1.00, recovers the sine's
  negative lobe, n_circuits=2, converges to the SV limit with shots.
  **Circuit depth/cx intentionally not synthesised:** the
  flattened dense SELECT would Shannon-decompose to a meaningless
  ~4^(q+m) cx count, so metrics record "unavailable"; honest gate
  counts come from the structured encoder (M4).  Meaningful M3 scaling
  indicators captured: `n_qubits`, `n_ancilla`, `lcu_lambda`,
  `p_success`.  Shots sweeps added to `input/burgers_direct_lcu.toml`.
- **M4a -- Scaling sweep (measurement; no new code).** Run
  `input/burgers_direct_lcu.toml` and read off where the method
  degrades.  The SV blocks map accuracy vs (q, taylor-order,
  segment-size, bond-dim); the shots blocks map post-select
  `p_success` vs (q, segment-size).  Runnable today on M1-M3 + the
  Hadamard path -- this is the "how far can it be pushed" deliverable.
- **M4b -- Structured (nested) encoder (the scaling fix; NEW -- the one
  substantial remaining build).**  The flattened dense-SELECT word LCU
  walls on two fronts M4b removes, plus one it can only soften:
  1. **K-explosion (Taylor order).** Replace the flattened word list
     (K ~ 37/167/680 at order 2/3/4, one giant dense SELECT) with the
     Berry-Childs-Kothari truncated-Taylor structure: block-encode
     `A_seg` ONCE from its 7 base terms, then apply it `m` times via a
     compact PREPARE-over-powers + repeated controlled-SELECT.  Ancilla
     becomes `O(a + log M)` instead of `O(log K)`; no dense word
     expansion.  The single biggest piece.
  2. **Honest gate counts.** Once SELECT is built from controlled-shift
     (S+/S-) and controlled-diagonal (`diag(u_seg)`) primitives instead
     of one dense `UnitaryGate`, the metric-basis depth/cx become real
     structured numbers (today they Shannon-decompose to a meaningless
     ~4^(q+m) and are recorded "unavailable").  Same work as (1).
  3. **lambda / p_success wall (physics, not artifact).**
     `lambda ~ exp(T_segment*lambda_A)`, `lambda_A ~ nu/dx^2 +
     (1/2)*||u||_inf/dx`, so post-select success `~ 1/lambda^2`
     collapses as `q` grows (smaller dx) or segments lengthen.
     Intrinsic to a non-unitary block-encoded step; M4b can SOFTEN it
     (oblivious amplitude amplification: `1/lambda^2 -> 1/lambda`;
     shorter segments) but not remove it.  M4a characterises exactly
     where it bites.

## 8. Scaling limiters (the "how far" experiment)

- **Segment size `k`** vs frozen-coefficient (lag) error.
- **LCU subnormalization `lambda`** -- `lambda_A ~ nu*||L|| +
  ||u_seg||_inf*||G||` through the Taylor powers; post-selection
  success `~ 1/lambda^2` caps reachable `q`, `dt`, Taylor order.
- **Ancilla width** (Taylor terms + diagonal BE ancilla).
- **Statevector sim memory** for the upper-`q` runs.

## 9. Tests

- `test_advection_diffusion_lcu_step_matches_classical` (M1):
  one-segment SV vs the classical conservative reference
  (`nu*u_xx - (1/2)*d/dx(u^2)`), q in {3,4,5}, sine IC, tol from
  Taylor order.
- `test_frozen_coefficient_error_vs_k` (M2): error monotone in segment
  size; converges to per-step refresh as `k -> 1`.
- `test_no_shadow`: assert no classical field integrator is invoked in
  the evolution loop (only measurement-derived reprep).
- `test_lambda_success_probability` (M3): measured post-select rate
  matches `1/lambda^2` estimate within shot noise.

## 10. Future work (enumerated)

Beyond M4b (the structured encoder, §7), in rough priority order:

1. **`dual_rail` sign recovery.** Alternative to the implemented
   `hadamard_test`: evolve `u = u+ - u-` on two non-negative rails and
   recombine classically (2x circuits/segment, no interference).
2. **Dirichlet BC.** Periodic (S+/S- wrap mod N) is the baseline;
   Dirichlet via the paper's one-sided boundary treatment
   (`shift_matrix` already supports `bc='dirichlet'`).
3. **Hardware execution.** Sim only (AerSimulator) today; the dense
   SELECT / controlled-block `UnitaryGate`s are sim-only and must be
   replaced by the structured primitives (M4b) before any backend run.
4. **Oblivious amplitude amplification** of the per-segment
   post-selection (`1/lambda^2 -> 1/lambda`); pairs with M4b.
5. **QSP / qubitisation** replacing the truncated-Taylor LCU -- a
   higher Hamiltonian-simulation tier; significant theoretical depth.
6. **Variational QNPU (Lubasch).** The alternative nonlinear route
   (state carried in a variational ansatz); a different parcel.
7. **Fast-forwarding / Krylov** amortization of the evolution across
   segments.
8. **Gray encoding.** Operator locality / lower MPS bond dim; see
   `SPEC-encoding-switch.md`.
9. **Advective form `u*du/dx`** behind a flag (smooth-flow-only
   variant; same construction cost as the conservative baseline).
10. **Amplitude-loading diag(u) from a coherent oracle.** Superseded by
    measure-reprepare's known-value diagonal (would re-introduce
    no-cloning); recorded as not-pursued.

## 11. Decisions and open questions

Decided:
1. **Generator form: conservative** `(1/2) d/dx(u^2)`
   (`A_seg = nu*L - (1/2)*G*diag(u_seg)`). Correct weak/shock
   solution; same construction cost as advective. Advective is a
   demoted variant (§10).
2. **BC: periodic baseline** (S+/S- wrap mod N). Dirichlet via the
   paper's one-sided treatment is a follow-up (§10.2).
3. **Time discretization: truncated Taylor**, matching the existing
   `heat_taylor_lcu_terms`.  SV path applies the per-step
   `exp(A_seg*dt)` operator `k` times; the shots path uses one
   segment-spanning `exp(A_seg*T_segment)` block-encoding per segment
   (A frozen, so both are exact in the frozen-A limit; only Taylor
   truncation differs, set by `--lcu-taylor-order`).

Open: none.
