# Future Work

The canonical forward roadmap for `q8020-mps-burgers`. Supersedes the
F-section of `archive/IMPLEMENTATION-PLAN.md`; unfinished plan items are
folded in below. Each entry lists why it matters, rough scope, and what
it depends on. Listed roughly in order of likely priority; all items are
independent unless a dependency is noted.

## 1. Burgulence — statistical study of decaying 1-D Burgers turbulence

**Why.** Burgulence is the canonical toy model for 1-D shock-dominated
turbulence (Kida 1979); demonstrating the universal E(k) ~ k⁻² inertial
range and bifractal structure-function exponents on a quantum-simulated
Burgers solver would be a physics result, not just a demo. The
`--ic multimode` option produces a single deterministic Fourier field
with random phases — that is *not* Burgulence (a statistical object
requiring an ensemble). The Alhawwary–Wang §5.3 reproduction is the
paper-comparison benchmark target.

**Scope.** Five sub-tasks (originally plan items F11.1–5):

- **1.1 — Burgulence-correct IC sampler.** Gaussian random field with
  prescribed energy spectrum `E(k) = C · k⁻ᵝ · exp(−(k/k_c)²)`,
  drawing Fourier coefficients `A_k ~ N(0, √E(k))` and independent
  random phases. Distinct from `--ic multimode`'s deterministic
  `A_k = k⁻ᵅ`.
- **1.2 — Ensemble sweep driver.** Run `N_realizations` (~50–200)
  independent simulations at fixed `Re = U_rms · L / ν`, varying only
  `ic_seed`. Uses the existing q8020-sweep parameter-array machinery.
- **1.3 — Ensemble-averaged diagnostics.** Postprocessor that reads all
  realizations and computes: (a) `E(k,t) = ⟨|û(k,t)|²⟩` on log-log to
  test the k⁻² scaling, (b) structure functions
  `S_p(r,t) = ⟨|u(x+r) − u(x)|^p⟩` for `p ∈ {1,2,3,4,6}`, (c) PDF of
  velocity gradients `du/dx`, (d) shock density / shock statistics.
- **1.4 — Scale separation.** Requires `q ≥ 10` (`N ≥ 1024`) to have an
  inertial range between IC scale and Kolmogorov dissipation scale. At
  `q ≤ 6` the dissipation scale is comparable to the grid scale.
  Depends on the LCU `cole_hopf_circuit` path reaching `q ≥ 10`
  (efficient bond-dim, possibly hardware) — see items 5–6.
- **1.5 — Forced Burgulence (optional, harder).** Stochastic
  white-in-time forcing with prescribed spatial correlation; study
  steady-state statistics. Requires extending the solver loop for
  time-dependent stochastic `source_fn` with per-step variance
  rescaling.

**Depends on.** F10 closed at paper-target ν=1e-4 (i.e. P-G + P-H
merged, acceptance 11.7 / 11.8 equivalents passing). Sub-task 1.4 also
depends on item 5 (QSVT) or item 6 (QROM) to reach `q ≥ 10`. A stub
spec exists at `SPEC-alhawwary-wang-5.3-burgulence.md`.

## 2. Encoding change — binary → locality-preserving

**Why.** Binary amplitude encoding puts physical neighbors on qubits
that are Hamming-distant; any operator local in x is nonlocal on the
qubit chain. This is the root cause behind Zaletel W-II being rejected
for F2, and behind the dense-block fallback in F10. A Gray-code or
block-local encoding would make real Pauli-Trotter, Zaletel W-II, and
direct u-space evolution all tractable.

**Scope.** Large. New state-prep pipeline, rewrite of every
space-local operator, new CLI surface. F10 acceptance tests would
re-baseline.

**Depends on.** Nothing; this is the enabling prerequisite for items
3, 8, 9.

## 3. Direct u-space evolution via Carleman linearization

**Why.** Cole-Hopf linearization is only available for 1D Burgers
with no source; extending to 2D/3D Navier-Stokes or forced Burgers
requires a different linearization. Carleman embeds the nonlinearity
into an infinite linear hierarchy truncated at some order K.

**Scope.** ~F10-sized. New propagator, truncation-error analysis,
separate acceptance suite. Drops Cole-Hopf entirely.

**Depends on.** Nothing technical; but probably wants encoding
change (item 2) first for the linear operator to be local.

## 4. DST-based `qft-diagonal` + Dirichlet

**Why.** Today `qft-diagonal` silently falls back to `dense-block`
under Dirichlet BC because QFT diagonalizes only the periodic
Laplacian. A discrete-sine-transform variant would diagonalize the
Dirichlet Laplacian and restore qft-diagonal as a gate-count-optimal
path for the paper regime.

**Scope.** Moderate. New `dst_diagonal_step_circuit` in
`burgers_cole_hopf_circuit.py`, reuse conditional-Ry machinery on
sine-basis eigenvalues. No change to propagator dispatch CLI.

**Depends on.** Nothing.

## 5. QSVT polynomial alternative to ancilla-Ry

**Why.** The conditional-Ry Möbius expansion costs O(2^q) terms for
an exact polynomial fit; QSVT gives a polynomial-degree / error
tradeoff that scales better at q ≥ 7. Also removes the ancilla.

**Scope.** Substantial. QSVT phase-angle computation,
block-encoding setup, integration as a third `--propagator` option.

**Depends on.** Nothing.

## 6. QROM-based θ(k) loading for q ≥ 7

**Why.** At q ≥ 7 the Möbius expansion in `build_conditional_ry`
has 128+ terms and gate depth becomes the bottleneck. QROM loads
θ(k) as classical data into a quantum register in O(2^q) gates but
with shallower depth and better ancilla tradeoffs.

**Scope.** Moderate. QROM construction + controlled-rotation from
loaded register. Drop-in replacement for `build_conditional_ry` at
large q; keep both.

**Depends on.** Nothing. Becomes load-bearing only at q ≥ 7.

## 7. Hardware execution with error mitigation

**Why.** Everything so far is Aer. Running on real hardware is the
eventual goal. Needs noise-aware transpilation, zero-noise
extrapolation or probabilistic error cancellation, and a sweep
harness that tolerates queue latency.

**Scope.** Large. New backend abstraction, mitigation pipeline,
benchmark protocol. Starts with q=3–4 calibration circuits.

**Depends on.** F10 closed + P-H.1 readout (peaked-φ shots on
hardware would be catastrophic without it).

## 8. F2 `tebd_circuit` revival

**Why.** The original F2 proposal was a true TEBD circuit evolving
u directly (no Cole-Hopf). Shelved because Zaletel W-II is nonlocal
under binary encoding. Becomes tractable under item 2.

**Scope.** Revisit F2 spec; likely a full rewrite.

**Depends on.** Item 2 (encoding change).

## 9. True Pauli-Trotter propagator

**Why.** Originally pitched as Fork A in F10-REVIEW-PATCH.md. A
genuine product-formula expansion of exp(νLΔt) using Pauli strings
on the qubit chain. Currently nonlocal on binary encoding — the
string weight blows up with q. Was renamed to `dense-block` in F10
P-A (Fork B). Restores acceptance 11.4 first-order Trotter-error
convergence (currently vacuous against `dense-block`'s exact
eigendecomp) and gives Murali a Pauli-level object to reason about
directly, matching the paper's framing.

**Scope.** Moderate once encoding is local. New propagator alongside
`qft-diagonal` and `dense-block`; gate-count scaling study vs the two
existing variants.

**Depends on.** Item 2 (encoding change) to be competitive; on
binary encoding it is strictly worse than `dense-block`.

## 10. Peaked-φ shots readout — open gap

**Why.** At paper-target ν=1e-4, φ(x) = exp(−∫u/2ν) concentrates
almost all probability mass on ~1 grid bin, and √-of-counts readout
at `burgers_cole_hopf_circuit.py:504-509` wastes shots on the peak
while tail bins have p_i ≪ 1/shots. Observable symptom:
`test_11_5_shots_accuracy` runs at ν=0.1 instead of the spec's
ν=1e-2 because the low-ν path degrades past the 5% tolerance.
F10-REVIEW-PATCH-02.md P-H proposes Hadamard-test per bin as one
fix; if that proves too expensive at larger q, other options are
preconditioning the state to flatten φ before measurement,
importance sampling, or switching to a log-amplitude readout
altogether. No fix is merged.

**Scope.** Moderate per option. Hadamard-test path reuses the F9
sign-recovery ancilla wiring. Log-amplitude readout is a deeper
rewrite of the readout stage.

**Depends on.** Nothing. Load-bearing for any production sweep at
ν < 1e-3.

## 11. Qiskit 2.3 → 3 deprecation: `RYGate.control(annotated=None)`

**Why.** `build_conditional_ry` in `burgers_cole_hopf_circuit.py`
relies on `RYGate.control(annotated=None)`, deprecated in Qiskit
2.3 and slated for removal in Qiskit 3. At the Qiskit 3 bump the
conditional-Ry Möbius expansion stops transpiling and every
`cole_hopf_circuit` path breaks.

**Scope.** Small. Switch to the explicit `annotated=True` or
`annotated=False` form (decide based on transpiler-pass
compatibility), update tests.

**Depends on.** Nothing. Do this whenever we next touch Qiskit
pinning.

## 12. Cole-Hopf-exact analytic IC (plan F12.1) — DONE

Shipped.  See OVERVIEW §4.4 and `burgers_cole_hopf.py:
{initial_condition_cole_hopf_exact, analytic_solution_cole_hopf,
validate_cole_hopf_coeffs}`.  Wired via `--ic cole_hopf_exact` with
coefficients via `--ic-cole-hopf-coeffs "a0,a1,..."`.  When `--method`
is `cole_hopf` or `cole_hopf_circuit`, IC defaults to
`cole_hopf_exact` and the analytic `u(x,t)` is used as the reference
trajectory automatically; `--no-analytic-reference` falls back to
FTCS/Godunov.  Restricted to `--bc dirichlet` + `--source none` by the
math (Neumann-on-φ cosine basis; modes only stay decoupled in the
unforced case).

## 13. UCAN cross-validation (plan §3.2)

**Why.** Independent oracle check on our `shift` classical baseline
against the paper authors' own quimb-based MPO-on-MPS reference
(UCAN-1DBurgers, the Meena/Murali group's Stage-1 implementation).
Any discrepancy points to a BC or operator-construction bug. The plan
called for this at `q=5, 6` and it was never executed.

**Scope.** Small if UCAN is checked out. Clone UCAN, match its BC
(Dirichlet) and discretization (one-sided/upwind), run at `q ∈ {5,6}`
on a `sine` IC, diff our `--method shift --bc dirichlet` output
against UCAN's `u(x, t)` time series. Target agreement: floating
point per step (`< 1e-12`).

**Depends on.** Access to the UCAN-1DBurgers repo (not currently
checked out on this workspace).

## 14. `qlbm_circuit` real-backend shots path (QLBM F11-13) — DONE

Shipped as "Option A" (hybrid by construction; mirror of Meena
Appendix A.A for QLBM).  `run_qlbm_circuit_simulation` shots branch
now builds the same per-step circuit the statevector path builds,
transpiles, executes on the configured backend, and reconstructs
`f_post` via `|ψ_out_k| ≈ √(counts[k]/S)` followed by
`unflatten_distributions`.  `--sign-recovery {none, classical_oracle}`
both honored; `hadamard_test` deferred to #26.  Per-step metrics gain
`leakage` (mass in the unused `|11⟩` velocity block — noise sensor)
and `negative_mass` (classical-oracle signal for when sign recovery
matters).  See SPEC-qlbm-shots-and-sign-recovery.md and OVERVIEW
§5.2 for the full contract and the hybrid-vs-pure-quantum framing.
The pure-quantum QLBM alternatives are #27 / #28 / #29.

## 15. Neumann BC on `u` (plan F12.3)

**Why.** Current user-facing `--bc` choices are `{periodic,
dirichlet}`. Reflecting (Neumann, `du/dx = 0`) walls are the missing
canonical 1-D BC, useful for symmetry-plane / adiabatic-wall studies.
Internally, the Cole-Hopf `φ`-equation already uses Neumann BC (it is
the dual of `u` Dirichlet), but there is no `--bc neumann` for `u`.

**Scope.** Small. In the shift operators, ghost nodes copy their
interior neighbour: `(S⁺u)[N−1] = u[N−1]`, `(S⁻u)[0] = u[0]`.
Touches `burgers_mpo.py` (shift matrices), `burgers_nonlinear.py`
(`compute_rhs_shift` bc handling), and `burgers_solver.py` (grid
setup, `--bc` choices). Out of scope for `cole_hopf_circuit` since
Neumann-on-u is not a natural Cole-Hopf case.

**Depends on.** Nothing.

## 16. Gaussian IC (plan F12.2) — DONE

Shipped.  `initial_condition_gaussian` in `burgers_classical.py`,
`--ic gaussian` with `--ic-center` (default 0.5) and `--ic-sigma`
(default 0.1); amplitude via the existing `--ic-amplitude`.  No
closed-form Cole–Hopf analytic reference (the `∫u₀` is an erf, so
`φ₀` has no clean heat-equation evolution); pairs with FTCS/Godunov
as the classical reference.  Works with all methods including
`cole_hopf_circuit` and `qlbm*`; for LBM keep `--ic-amplitude < 1.0`
for D1Q3 stability.  See OVERVIEW §1.1.

## 17. RK4 time integration (plan F6)

**Why.** Paper uses forward Euler (first order). Our entire time loop
inherits the same first-order accuracy. RK4 would reduce
discretization error by orders of magnitude on smooth phases of the
solution and improve the signal-to-noise ratio of any Trotter-error
characterization (item 18). The paper itself calls RK4 out as a
future improvement.

**Scope.** Moderate per method. Per-step methods (`shift`,
`quantum_circuit`, …) require 4 Hamiltonian evaluations per step,
proportionally increasing circuit count; sign recovery and norm
tracking carry through. Delegating methods (`tebd`, `cole_hopf*`,
`qlbm*`) are mostly unaffected — they own their own loop.

**Depends on.** Nothing.

## 18. Trotter-error characterization at q = 6 (plan F7)

**Why.** Plan §3.4 / F7 had Trotter-rep sweeps at `q = 4, 5`. The
`q = 6` extension was deferred and never run. Closes the empirical
loop on "reps needed for Trotter error to fall below discretization
error" at the largest scale the Pauli-Trotter pathway supports.

**Scope.** TOML sweep extension only — no code change. Run
`quantum_circuit` and `quantum_exact` at `q = 6, trotter_reps ∈
{1, 2, 5, 10, 20}`, postproc with existing scripts.

**Depends on.** Available compute (q=6 quantum_exact is at the OOM
boundary).

## 19. Parallelized Pauli decomposition (plan F1)

**Why.** The `4^q` Pauli-string scaling becomes the wall at `q = 7`
(16384 terms) and `q = 8` (65536 terms). The `P_i|u⟩` precomputation
and the `S` matrix construction in `solve_pauli_coefficients` are
embarrassingly parallel.

**Scope.** Moderate. `joblib` or `multiprocessing` for local
parallelism; longer-term distribute across Frontier nodes. Targets
`burgers_nonlinear.solve_pauli_coefficients`. Pure-quantum Pathway 2
(`cole_hopf_circuit`) sidesteps this scaling, so this item is only
load-bearing if we want to keep Pathway 1 viable at `q ≥ 7`.

**Depends on.** Nothing.

## 20. Variational fast-forwarding (plan F4)

**Why.** Compresses `M` time steps into a single circuit of fixed
depth: `(exp(iHδt))^M ≈ W · D(M·δt) · W†`, where `W` is a variationally
trained diagonalizing unitary. Eliminates per-step circuit overhead,
relevant for the Pauli-Trotter pathway at large `n_steps`. This is one
of the two fast-forwarding proposals in Gopalakrishnan Meena et al.
AIAA-2026 **Appendix A.B** (Eq. 18); paper reference [32] (Cirstoiu
et al., "Variational fast forwarding"). The paper notes ref [33]
(Bittel & Kliesch, "Training VQAs is NP-hard") as a caveat on
variational optimisation difficulty.

**Scope.** Substantial. Variational optimization loop for `W`,
ansatz design, fidelity-vs-depth tradeoff study, integration as an
alternative circuit-build mode for `quantum_circuit`.

**Depends on.** Nothing technical.

## 21. Krylov subspace methods (plan F5)

**Why.** Alternative to Trotter for Hamiltonian simulation, bypassing
the variational optimisation hurdle of item 20. Builds a Krylov basis
`{|u⟩, H|u⟩, H²|u⟩, …}` and projects evolution into a low-dimensional
subspace. Potentially better gate-count / accuracy tradeoff than
first-order Trotter at fixed circuit depth. This is the second
fast-forwarding proposal in Gopalakrishnan Meena et al. AIAA-2026
**Appendix A.B**; paper references [31] (Tkachenko et al., "Quantum
Davidson"), [34] (Lanczos), [35] (Saad, "Iterative methods for sparse
linear systems").

**Scope.** Substantial. Krylov basis construction circuit, projected
evolution, integration as a third propagator option alongside
`quantum_circuit` and `quantum_exact`.

**Depends on.** Nothing.

## 22. Walters et al. multimodal test case (plan F8)

**Why.** The paper's second test case: `N = 8192` (`q = 13`), 12-mode
initial condition, `ν = 1e-5`, `CFL = 0.05`. Out of scope for a
serial Aer workflow on `q ≤ 6`; would be a headline-class result.

**Scope.** Large. Requires parallelism (item 19) for the Pauli path
or scaling of the LCU `cole_hopf_circuit` path (items 5, 6) plus
likely real hardware (item 7). Acceptance is reproduction of the
paper's energy-decay curve at `q = 13`.

**Depends on.** Item 5 (QSVT) or 6 (QROM) for `cole_hopf_circuit`
gate-count to fit `q = 13`; item 7 (hardware + mitigation) for any
real-device execution; item 19 (parallel Pauli) if pursued via
Pathway 1. Effectively a *combined* deliverable, not a standalone
item.

## 23. Lubasch QNPU — pure-quantum alternative to Appendix A.A

**Why.** Pathway 1 (`quantum_circuit`) implements Meena AIAA-2026
Appendix A.A faithfully, but A.A is hybrid *by design*: it skips
building `u·∇u` as a quantum operator by instead fitting Â per step
to a classical Euler trajectory and solving an `O(4^q × 4^q)`
least-squares system classically.  A pure-quantum implementation of
nonlinear 1-D Burgers in the same "direct-u, fit-Â-per-step" shape
requires replacing that classical fit with a quantum nonlinear
processing unit (QNPU) — Lubasch et al., Phys. Rev. A 101, 010301
(2020); Pool et al., Phys. Rev. Research 6, 033257 (2024) — which
Meena cites as refs [18, 19] and uses as the baseline that A.A is
proposed *against*.  QNPU builds `u·∇u` as a circuit on `|u⟩⊗|u⟩`
(second copy + gradient MPO + CNOT-multiplication) and trains the
per-step generator variationally on a parametrised ansatz.

**Scope.** Substantial new pathway, not a modification of
`quantum_circuit`.  Components: (a) two-copy state-prep harness
(reuse Ran-2020 MPS-prep machinery from `burgers_mps.py`); (b)
Hadamard-product-via-CNOT circuit for `u·∇u`; (c) parametrised
ansatz (hardware-efficient or problem-inspired) for the per-step
evolution unitary; (d) cost-function evaluator
`C = ‖exp(-iÂδτ)|u⟩ − |u_target⟩‖²` estimated via overlap circuits;
(e) variational optimisation loop (SPSA / SciPy / qiskit-algorithms).
New `--method qnpu` switch and a `burgers_qnpu.py` module.  Paper
ref [33] (Bittel & Kliesch, NP-hard training of VQAs) is the known
caveat — same caveat as item 20 (VFF).

**Depends on.** Nothing technical; item 2 (locality-preserving
encoding) helps gradient-MPO depth at `q ≥ 7` but is not required.
Item 7 (hardware + mitigation) for real-device runs.  Acceptance
target: reproduce the sine-IC trajectory at `q = 4–5`, `ν = 1e-3`
within ~5% L²-error against the classical reference, on Aer.

## 24. Plumb new IC / reference flags into `BurgersConfig` — DONE

Shipped.  `BurgersConfig` in `burgers_fw.py` now has fields
`ic_center`, `ic_sigma`, `ic_cole_hopf_coeffs`, `classical_reference`,
`analytic_reference`; `burgers_solver.py` passes them in from
`args`; `burgers_postprocess.py` records them in both the case
fragment and the JSON summary, conditional on the relevant `--ic` for
the IC-specific fields (so case fragments stay clean for unrelated
ICs).  See OVERVIEW §8.1.

## 25. Classical `cole_hopf` + `--bc dirichlet` BC mapping — DONE

Discovered during the #24 smoke test: `run_cole_hopf_simulation` was
periodic/Neumann-only on the phi side, so `--method cole_hopf --bc
dirichlet` crashed with a cryptic `ValueError: Unknown bc: 'dirichlet'`
from `build_laplacian_dense`.  This broke BC symmetry between the
classical and circuit CH paths and made the classical CH unusable as
a V&V oracle for `cole_hopf_circuit` under Dirichlet-on-u.

Shipped.  `run_cole_hopf_simulation` now mirrors the same `phi_bc =
"neumann" if bc == "dirichlet" else bc` mapping that
`burgers_cole_hopf_circuit.py:1911` uses on the circuit side
(per OVERVIEW §4.1: Dirichlet on u ↔ Neumann on phi).  Unsupported BC
values raise a clean `NotImplementedError` instead of leaking the
phi-side label up to the user.  Classical `cole_hopf` + `--bc
dirichlet` now runs and can serve as a cross-check against
`cole_hopf_circuit` at the same BC.

## 26. Hadamard per-bin sign test for `qlbm_circuit` (fast follow to #14)

**Why.** #14 v1 ships `--sign-recovery {none, classical_oracle}` for
`qlbm_circuit`.  `classical_oracle` works but reads signs from a
classical reference, so the run is no longer a stand-alone benchmark.
For shock-regime / non-positive-f cases the only stand-alone signal
recovery is interferometric: per-bin Hadamard test.

**Scope.** ~250 LOC, mirrors
`burgers_cole_hopf_circuit.py:1597–1851` (`hadamard_per_bin_circuit`,
`extract_hadamard_per_bin_amplitudes`, `_run_shots_hadamard_per_bin`).
Adds one ancilla, two applications of `U_step` per bin in
superposition with a reference prep; sign is `sign(Re(⟨k|U_step|ψ_in⟩))`.
Cost: `O(4N)` extra circuit executions per step on top of the
direct shots path.

**Depends on.** #14 v1 (the no-/classical_oracle path) shipped so
the dispatch already exists; flip `NotImplementedError` to the real
implementation.

## 27. Itani-style pure-quantum QLBM (QALB)

**Why.** Today's `qlbm_circuit` ("Option A") is hybrid by
construction: every step builds the collision unitary from a
classically-computed `f_post = collide_bgk(f_pre, tau)` via Householder
dilation.  The quantum circuit then *replays* the classical answer.
This is the QLBM analog of Meena Appendix A.A's hybrid design for
`quantum_circuit`.  Itani et al., *Phys. Fluids* 36, 2024 (ref [11]
of the Meena paper) propose a pure-quantum QALB where both `f` and
the equilibrium `f_eq` live on the quantum register in superposition
and the collision is a *state-independent* operator on the combined
encoding.  The result is a genuine pure-quantum QLBM, parallel to
the `cole_hopf_circuit` pathway.

**Scope.** Substantial — a new algorithm, not a modification of
`burgers_qlbm_circuit.py`.  Components: (a) extended register layout
that block-encodes `f_eq(ρ, u)` alongside `f` (extra ancilla
qubits); (b) state-independent collision unitary; (c) streaming
unchanged from today; (d) macroscopic-moment readout via overlap
circuits or amplitude estimation rather than direct measurement;
(e) cross-validation against the existing Option A as the hybrid
oracle.  Likely a new method name (`--method qalb`) coexisting with
the existing `qlbm_circuit` Option A, similar to how `quantum_exact`
coexists with `quantum_circuit`.

**Depends on.** Nothing technical, but item 2 (locality-preserving
encoding) helps the equilibrium-encoding subcircuit at `q ≥ 7`.
Item 7 (hardware + mitigation) for real-device runs.

## 28. Carleman linearization of BGK collision

**Why.** Alternative pure-quantum QLBM route to #27.  Carleman lift
`(f, f⊗f, f⊗f⊗f, …)` truncated at order `M` turns BGK's quadratic
nonlinearity into a *linear* sparse block-bidiagonal generator on
the lifted state.  Then evolution is a single fixed unitary —
state-independent, no per-step classical mirror.  Parallels item 3
(Carleman for direct-`u` Burgers); the BGK case is structurally
similar but operates on the lattice distributions rather than `u`
directly.

**Scope.** Substantial.  Lifted dimension `O(n^M)` for `n = 2^{q+2}`;
the order-`M` register needs `M·(q+2)` qubits.  Truncation-error
analysis, block-encoding of the lifted generator, integration as
a third QLBM method (`--method qlbm_carleman` or similar).
Convergence requires `R = ‖nonlinearity‖/‖dissipation‖ < 1`, i.e.
low effective Reynolds / `τ` not too close to 0.5.  State the
regime restriction explicitly.

**Depends on.** Item 2 (encoding) for locality of the lifted
operators at large `q`/`M`.

## 29. Linearized BGK collision (low-Mach pure-quantum QLBM)

**Why.** The cheapest pure-quantum route.  Linearise BGK around the
equilibrium: `f = f_eq + δf`, treat `δf` as the variable, drop the
`O(δf²)` term.  Resulting collision operator is *linear* in `δf`,
fixed, state-independent.  Streaming unchanged.  Pure-quantum, no
classical mirror.

**Scope.** Small to moderate.  New module
`burgers_qlbm_linear_circuit.py` with a fixed collision unitary
build, identical streaming.  Validation: agreement with full BGK in
the small-Mach, near-equilibrium regime.  Add as a third QLBM
method choice (`--method qlbm_linear` or similar).

**Depends on.** Nothing.

**Catch.** Only valid for smooth, low-Mach flows near equilibrium.
Loses shock physics — exactly what Burgers is most interesting for.
Useful as a pedagogical pure-quantum LBM benchmark, not as the
production solver.  Document the regime restriction prominently.
