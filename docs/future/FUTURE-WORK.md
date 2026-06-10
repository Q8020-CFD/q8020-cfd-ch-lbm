# Future Work

The canonical forward roadmap for `q8020-mps-burgers`. Supersedes the
F-section of `archive/IMPLEMENTATION-PLAN.md`; unfinished plan items are
folded in below. Each entry lists why it matters, rough scope, and what
it depends on. Listed roughly in order of likely priority; all items are
independent unless a dependency is noted.

> **Completed / resolved items** (#12, #14, #16, #24, #25, #26, #28, #29)
> are split out to
> [../archive/FUTURE-WORK-DONE.md](../archive/FUTURE-WORK-DONE.md).
> Numbers are **preserved** (referenced elsewhere), so this list has
> gaps rather than renumbering. #27 stays here — only partially shipped.

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

## 27. Itani-style pure-quantum QLBM (QALB) — PARTIALLY SHIPPED

The **collision + shots path is shipped** as the bare `qlbm_circuit`
(`burgers_qalb_circuit.py`, `QALBIntegrator`, `run_qalb_simulation`;
see OVERVIEW §5.3). What's done vs. what remains:

**Done.**
- App B value/Fock encoding (vacuum-displacement, `⟨q̂⟩` readout,
  machine-precision encode/decode).
- Normal-ordered Hermitised collision `e^{−iΔtĤ′}` (Itani Eq. 79–86) —
  exactly unitary, **no post-selection**; the normal-ordering of the
  quadratic term (`s²−I`) was the key that made the truncated collision
  reproduce the classical flow, convergent in `qc`.
- Per-site collision + exact streaming, measure-reprepare(k=1),
  `--fock-qubits`, `--seed`; routed through the shared shots helpers.
- Validation gates 1–7 in the module `__main__`; shots-vs-statevector
  agreement; subsumes #28 (the QALB *is* a Carleman/Kowalski scheme).
- **27.1 Trotter synthesis of `Ĥ′` (gate to hardware).** The dense
  per-site `UnitaryGate` of `e^{−iΔt Ĥ′}` (Quantum-Shannon, ~4^(3qc) CX)
  is now replaceable by a Suzuki-Trotter circuit of the Pauli
  decomposition of `Ĥ′` (`cell_collision_gate(..., trotter_reps>0)`,
  `--qalb-collision-trotter-reps`, gate7). A **single position-free
  unitary on exactly 3·qc qubits, no ancilla, exactly unitary (no
  post-selection)** — strictly better than an LCU block-encoding for
  this path, which would reintroduce ancilla + post-selection and break
  the App B virtue. Order-2 Trotter error ∝ 1/reps²; reps≈4 sits below
  the qc=2 Fock-truncation floor (gate7). **Measured depth (basis u/cx,
  opt-1):** at qc=3 the dense `UnitaryGate` is depth≈236k / 119k CX,
  while Trotter o=2 reps=1 is depth≈66k / 54k CX (**3.6× shallower**) and
  reps=2 is ≈132k / 109k CX (1.8× shallower) — the exponential
  Quantum-Shannon blowup is exactly what this fixes. At qc=2 (only 6
  qubits) the dense unitary is already cheap (depth 3522 / 1782 CX) so
  Trotter does *not* win there. *Caveat:* the Pauli words come from a
  brute-force `SparsePauliOp.from_operator` (194 terms at qc=2, 2948 at
  qc=3); the structured Itani monomial decomposition (Eqs. 132–133,
  `m=17` monomials × `qc²` words ≈ 68 at qc=2) would shrink the term
  count by ~30× and is the follow-on for the full depth advantage —
  tracked as **27.1a** below.

**Open (tracked in the session task list as "task #3").**
1. **27.1a — structured (compact) Pauli `Ĥ′`.** Replace `from_operator`
   with the Itani monomial decomposition (Eqs. 132–133) so the LCU/Trotter
   has `L = m·qc² ≈ 68` terms at qc=2 instead of 194; this is where the
   real hardware-honest depth advantage lives at qc≥3.
2. **#27.2 — coherent `k > 1` measure-reprepare + quantum streaming** —
   needs a full-lattice circuit (position register + quantum log-depth
   streaming, SPEC §3.6) so `k` collide+stream steps run before one
   measurement; bounded by Itani App A's logistic truncation divergence
   (keep `k` small at `Δt/τ = O(1)`). Today's k=1 measures every step +
   classical streaming. `build_streaming_circuit` (the `q+2`
   interleaved/linear encoding in `burgers_qlbm_circuit.py`) does **not**
   carry over; `build_qalb_streaming_unitary` is its QALB replacement.

   **Collision ↔ streaming interface contract (frozen — #27.1 landed
   against exactly this; the transducer + streaming compose against it):**
   - The per-site collision is `cell_collision_gate(tau, qc,
     collision_time, trotter_reps, trotter_order)` — a Qiskit gate on
     **exactly `3·qc` qubits, NO ancilla, exactly unitary (no
     post-selection)**. It is *not* an LCU block-encoding (an LCU would
     reintroduce an ancilla + post-selection and break the App B virtue),
     so there is **nothing to uncompute** on the collision and the
     transducer must not assume an LCU ancilla.
   - **Register order is frozen** as `kron(reg₋₁, reg₀, reg₊₁)` (the
     `encode_cell_B` convention). In Qiskit qubit indexing that is
     **reg₋₁ = qubits `[2·qc, 3·qc)` (MSB), reg₀ = `[qc, 2·qc)`,
     reg₊₁ = `[0, qc)` (LSB)**. Build the transducer + streaming against
     exactly these slots.
   - The collision is **position-free** (per-site), so it composes
     cleanly with whatever goes on the position register — there is no
     coupling between the collision gate and `build_qalb_streaming_unitary`.
3. **Backend wiring** — `make_integrator` builds no backend for
   `qlbm_circuit` and hardcodes `backend_type="sim"` elsewhere, so
   `--backend-type hardware` is ignored; wire `QALBIntegrator` to honor
   it (small).
4. **Validation** — full shots run at the aligned regime (`τ > 1`),
   qc-convergence vs FTCS, and the incompressible Mach ceiling at
   `amp ∈ {0.5, 0.8}`; plus error mitigation + an estimator-variance /
   shot-budget study.

**Note on the FTCS gap.** QALB is a *flow*-LBM (continuous BGK flow),
so it differs from FTCS/Euler by an O(Ω²)/step **scheme** gap
(~0.11 final error) that does *not* shrink with `qc` — distinct from
the Fock-truncation error, which does. See OVERVIEW §5.3 and
[SPEC](../future/SPEC-qlbm-pure-quantum-qalb.md).

**Depends on.** Item 2 (locality-preserving encoding) helps at `q ≥ 7`;
item 7 (hardware + mitigation) for real-device runs.

> #28 (Carleman lift of BGK) and #29 (linearised-BGK,
> `qlbm_circuit_linear`) are resolved — see
> [../archive/FUTURE-WORK-DONE.md](../archive/FUTURE-WORK-DONE.md).

## 30. Adaptive per-segment shot budgeting (`cole_hopf_circuit`)

**Why.** Shots are currently a hand-set constant (`--shots`, e.g.
150000 flat across the whole measure-reprepare run), so every segment
pays the same sampling cost regardless of how hard it is to resolve.
That is wasteful at both ends: the heat propagator damps high modes
fastest, so early segments lose the most amplitude (lowest
`p_success`, hardest to resolve) while late segments approach
`p_success → 1` and are over-sampled. A shot count *derived* from the
target accuracy would replace the magic number with a computed budget
and cut total shots at fixed final error.

The statistical floor is analytic and, crucially, **does not need a
statevector reference** — it is self-reported by the counts. From
`reconstruct_phi_from_counts` the per-bin amplitude estimate is
`φ̂_k = φ_norm·√(c_k/S)`, and the delta method gives a flat per-bin
variance `Var(φ̂_k) ≈ φ_norm²/(4S)` (independent of `p_success`). With
`M = --phi-modes` kept after the Fourier low-pass (white shot noise →
fraction `M/N` of the variance retained), the relative φ error is

   ε_φ ≈ √M / (2·√S·√p_success)   ⇒   S ≈ M / (4·p_success·ε_φ²).

Mapping φ-error to the physical `u`-error through the inverse
transform `u = −2ν ∂ₓ ln φ` multiplies by a condition number
κ ~ exp((U_max−U_min)/2ν) (the same `exp(·/2ν)` blow-up behind item 10
and the *"CH broken below ν≈0.015"* TOML note), giving
S ≈ M·κ² / (4·p_success·ε_u²). κ comes from the **macroscopic**
velocity range (IC-known / coarsely measurable), `M` is a user input,
and `p_success` is a scalar acceptance rate measurable to a few % with
a few-hundred-accepted-count pilot **at the real q** — none of these
require holding the 2^q state classically, so the scheme is
poly-cost and survives to large q (unlike calibrating off a `shots=0`
run, which presupposes the very SV solve the algorithm exists to
avoid).

**Scope.** Moderate; lives in `_run_shots_measure_reprepare`
(`burgers_cole_hopf_circuit.py`). Two viable forms:

- **Online/adaptive (preferred — no separate pilot pass).** Run each
  segment once; after `post_select_counts`, read the *actual*
  `p_success` and the self-reported binomial spread
  (`φ_norm²/(4·n_kept)`), then either keep sampling that segment until
  the spread meets ε (Aer lets you run more shots and merge counts) or
  set the *next* segment's shot count from this one's `p_success`.
  Removes the magic number entirely; the total budget is discovered as
  the run proceeds rather than fixed up front.
- **Naive pilot.** A low-shots (~1–2k) full trajectory is `K` circuits
  (one per segment — the re-prep chain means segment *i*'s
  `p_success` depends on all prior outcomes, so segments can't be
  probed in isolation), then extrapolate `S_target = S₀·(ε₀/ε_target)²`
  per segment. Cheap (~1–2% of a production pass) because
  construction/transpile cost is shots-independent and execution
  dominates — but the online form makes it unnecessary.

  Note that under segmentation errors add roughly in quadrature across
  re-preps (ε_total ≈ √K·ε_seg ⇒ shots-per-segment ∝ K), partly offset
  because shorter segments have much higher per-segment `p_success`
  (≈ p_success_single^(1/K)). This is the same `segment_size × nu`
  tuning surface flagged for the low-ν regime.

**Honest limit.** This budgets only the **statistical** floor. It says
nothing about systematic error — bond-dim truncation, MPS-prep,
Möbius angle pruning, the CH model itself — which does *not* shrink
with `S` and which, at large q, there is no cheap classical reference
to measure. The helper must print an explicit disclaimer that it bounds
statistical error alone. Same statistical machinery applies to QALB's
`⟨q̂⟩`-from-counts readout (`cell_collision_shots`); generalises the
shot-budget study noted in item 27 (Open #4).

**Depends on.** Nothing. Related to item 10 (peaked-φ readout — the
low-ν regime where this budget explodes via κ²) and item 27 Open #4
(QALB estimator-variance / shot-budget study).
