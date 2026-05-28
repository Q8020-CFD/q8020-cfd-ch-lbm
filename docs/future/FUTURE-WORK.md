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

## 12. Cole-Hopf-exact analytic IC (plan F12.1)

**Why.** Currently all accuracy claims are relative to the classical
FTCS reference, which is itself only a numerical approximation. A
closed-form analytic reference would let us quantify quantum-solver
accuracy without a classical co-solver. Construct `u₀(x)` such that
the Cole-Hopf-transformed `φ₀(x)` is a finite cosine sum:

```
φ₀(x) = a₀ + Σ_{n=1..M} a_n · cos(n·π·x)
```

Each mode evolves independently under the heat equation:

```
φ(x, t) = a₀ + Σ_n a_n · cos(n·π·x) · exp(−ν · (n·π)² · t)
```

with inverse transform `u(x, t) = −2ν · φ_x(x, t) / φ(x, t)` giving
an analytic reference. Aligned with the F10 transform machinery,
which is already shipped in `burgers_cole_hopf.py`.

**Scope.** Small. Add `initial_condition_cole_hopf_exact(x, coeffs, nu)`
and `analytic_solution_cole_hopf(x, t, coeffs, nu)`; wire as a new
`--ic cole_hopf_exact` choice; pass `coeffs` via CLI or TOML.
Validation case: a single-mode `φ` (tanh-type `u` profile) tracked at
arbitrary `t`.

**Depends on.** Nothing — F10.1 (`burgers_cole_hopf.py`) is shipped.

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

## 14. `qlbm_circuit` real-backend shots path (QLBM F11-13)

**Why.** Today `qlbm_circuit` runs the real circuit only in
statevector mode; the shots path falls back to a classical BGK
collision-stream loop. The backend is plumbed through
`make_integrator` but never exercised. To make the QLBM cross-method
comparison honest at q ≥ 4 on hardware, we need a true measurement
+ reconstruction path.

**Scope.** Moderate. Add shot-based reconstruction of distributions
`f_i` from joint `|v⟩|p⟩` register counts; handle the per-step
amplitude rescaling for the non-unitary collision contraction;
integrate with the existing `q8020-cfd-qutil` backend abstraction.
Tests in `tests/test_shots_backend.py` extended for QLBM.

**Depends on.** Nothing technical; benefits from item 7 (hardware +
error mitigation) when we want to run on real backends.

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

## 16. Gaussian IC (plan F12.2)

**Why.** Smooth localized pulse `u₀(x) = A · exp(−((x − x₀)/σ)²)`
demonstrates shock formation from a single-lobe disturbance — useful
for pedagogical shock-formation animations and as a complement to the
periodic `sine` and randomized `multimode` ICs.

**Scope.** Trivial (~10 LOC). Add `initial_condition_gaussian` in
`burgers_classical.py` and `--ic gaussian` plus `--ic-amplitude`,
`--ic-center`, `--ic-sigma` flags.

**Depends on.** Nothing.

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
