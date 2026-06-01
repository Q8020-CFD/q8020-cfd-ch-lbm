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
- **M2 -- Measure-reprepare loop (SV).** Direct-`u` measure-reprepare
  variant; sweep segment size `k`. Acceptance: frozen-coefficient
  error vs `k` characterized; full trajectory tracks FTCS within a
  stated tolerance for small `k`.
- **M3 -- Shots path.** Reconstruction + post-selection; report
  per-segment success probability and circuit cost (depth/cx, honest
  metric-basis lowering).
- **M4 -- Scaling sweep.** Push `q`, `k`, Taylor order; map where
  success probability / depth / sim memory wall the method. This is
  the "how far can it be pushed" deliverable.

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

## 10. Out of scope

- **Amplitude-loading diag(u) from a coherent oracle.** Superseded by
  measure-reprepare's known-value diagonal; would re-introduce the
  no-cloning problem. Not pursued.
- **Gray encoding.** Optional extension; see `SPEC-encoding-switch.md`.
- **Advective form `u*du/dx`** as an alternative to the conservative
  baseline. Smooth-flow-only variant; same construction cost (factor
  order swap). Implement behind a flag only if a non-conservative
  comparison is wanted; not the baseline.
- **QSP / qubitisation / Berry-Childs-Kothari** replacing first-order
  Taylor. Future Hamiltonian-simulation upgrade.
- **Variational QNPU (Lubasch).** The alternative nonlinear route;
  different parcel.
- **Fast-forwarding / Krylov amortization** of the per-segment
  evolution. Future.
- **Hardware execution.** Sim only in v1.

## 11. Decisions and open questions

Decided:
1. **Generator form: conservative** `(1/2) d/dx(u^2)`
   (`A_seg = nu*L - (1/2)*G*diag(u_seg)`). Correct weak/shock
   solution; same construction cost as advective. Advective is a
   demoted variant (§10).
2. **BC: periodic baseline** (S+/S- wrap mod N). Dirichlet via the
   paper's one-sided treatment is a follow-up.

Open (decide at/before M1):
3. **Time discretization:** first-order Taylor of `exp(A_seg dt)` per
   step vs a single segment-spanning `exp(A_seg * k*dt)`.
