# SPEC — Pure-quantum QLBM (QALB) for 1-D Burgers

> **SUPERSEDED (2026-06-08).** This spec's chosen route (block-encoded
> collision) was found shots-impractical, and the App B/C reconciliation
> here is incomplete. The working construction (App B vacuum encoding +
> normal-ordering + Hermitisation, now in `burgers_qalb_circuit.py` with
> gates 5-6, shots path wired) and the current frontier live in the
> session task list / auto-memory, not here. Also note: Eq. (83)'s
> divergence is **−2/τ** for D1Q3 (the "−2/3τ" below is a typo; the Itani
> paper says −(1/τ)(Q−D) = −2/τ). Kept for the derivation history.

Self-contained handoff. Reader has **not** seen the prior conversation.

> **QALB** = *Quantum Algorithm for Lattice Boltzmann* — the name from
> the Itani et al. paper (below). Throughout this spec "QALB" means the
> pure-quantum QLBM method we are building under the `--method`
> name `qlbm_circuit` (Phase 2), as distinct from the retired hybrid
> `qlbm_circuit_hybrid` and the Phase-1 precursor `qlbm_circuit_linear`.

> **Status (2026-06-08).** Step 0 (rename to `qlbm_circuit_hybrid` +
> alias) **done**. **Phase 1 (`qlbm_circuit_linear`) implemented and
> integrated** — `burgers_qlbm_linear_circuit.py`, registered in
> `burgers_fw.py`/`burgers_solver.py`; runs via CLI statevector + shots.
> Gates pass: collision operator exactness (fixed point + O(u²)),
> block-encoding extraction, statevector == classical linearised LBM,
> and the shots measure-reprepare(k) pipeline (k=2 tracks statevector to
> 3e-3, p_success≈0.17 — the expected `p^k`).
> **Phase 2 (`qlbm_circuit`) foundation started** —
> `burgers_qalb_circuit.py`: App B truncated bosonic operators `q̂,p̂`
> (commutator Eq. B13 verified to 1e-15), value/Fock encoding + decode
> (machine precision), and `Ω` rederived against *this repo's*
> density-conserving equilibrium (rest `(0,1,0)`) in the δf variable.
> **Single-cell collision VALIDATED to machine precision.** Resolved via
> the App C finite-position embedding (`burgers_qalb_circuit.py`):
> `q̂_C|x⟩=x|x⟩` (Eq. C39), exact linear readout `amp₁/amp₀=x` (Eq. C20),
> and the collision as the Liouville flow generator `G=Σ Ωᵢ(q̂_C) Dᵢ`
> (`Dᵢ=d/dx`, exact on the truncated polynomial space). `exp(T·G)` decodes
> (scale-invariant readout cancels the deflation for the value) to the
> classical collision flow at **6e-14 for qc≥3** (qc=2 truncation-limited
> at 5e-3 — matches Itani's qc dependence). The earlier App B attempt
> (overlap 0.34) used the wrong operators.
> **Lattice assembled and running as `--method qlbm_circuit`** (the QALB
> claims the bare name; hybrid → `qlbm_circuit_hybrid`). The collision is
> local, so the validated (3·qc)-qubit operator is applied per site
> (state-independent, built once — no classical mirror) + exact
> streaming. **Assembly verified: full-lattice QALB == classical
> flow-LBM to 1e-12.** A state-independent collision operator naturally
> realises the *continuous BGK flow*, not the discrete Euler step
> (`δf→δf+Ω(δf)` needs `Ω(x)` — state-dependent, which is exactly what
> the hybrid computed classically). So QALB is a *flow-LBM*: correct, but
> distinct from Euler-LBM by O(Ω²)/step. `collision_time` is
> auto-calibrated to match the linear relaxation per step
> (`1−e^{−T/τ}=1/τ`, correct viscosity). End-to-end CLI vs FTCS, aligned
> low-Mach (q=6, qc=3): `final_error≈0.115` at correct viscosity (the
> flow-vs-Euler/FTCS scheme gap; classical Euler-LBM is 0.045).
> **Transpilable collision circuit demonstrated** (`cell_collision_circuit`):
> the fixed `U_cell=exp(T·G)` Sz-Nagy block-encoded (1 ancilla), exact vs
> the operator to **1.4e-17**, transpiles to depth≈14k/CX≈7.3k at qc=2
> (7 qubits) — but **post-selection `p_keep≈0.002`**: the block-encoding
> normalizes by `‖U_cell‖₂`, which truncation inflates (the derivative
> operator's high-Fock entries), rejecting 99.8% of shots. This
> concretely confirms the block-encoding route is shots-impractical and
> the **Hermitized-unitary collision** (Eq. 85, `e^{−iΔtĤ'}` unitary, *no*
> post-selection, deflation = deterministic scalar) is the required path.
> **Active milestone:** the Hermitized-unitary collision (resolving the
> App B/C operator reconciliation so the truncated `e^{−iΔtĤ'}`
> reproduces the validated flow). **Also remaining:** `--fock-qubits` CLI
> flag (qc, default 3), `k>1` coherent segments. Today's solver path is
> operator/statevector-faithful (`shots>0` runs it with a note).
> Elaborates
> [FUTURE-WORK.md](FUTURE-WORK.md) items **#27 (Itani QALB)**,
> **#28 (Carleman BGK)**, **#29 (linearized BGK)**. This spec shows
> #27 and #28 are the *same* construction (Itani's QALB *is* a
> Carleman/Kowalski second-quantized scheme), and that #29 is the
> staged precursor to #27, not a competitor. Reference:
> **W. Itani, K. R. Sreenivasan, S. Succi, "Quantum Algorithm for
> Lattice Boltzmann (QALB) Simulation of Incompressible Fluids with a
> Nonlinear Collision Term," Phys. Fluids 36, 017112 (2024)**;
> preprint **arXiv:2304.05915**. Equation numbers below in the form
> "Itani Eq. (n)" refer to the preprint.

---

## 0. Context

`q8020-mps-burgers` solves 1-D viscous Burgers three ways, all on a
common aligned grid (see
[../../input/burgers_aligned.toml](../../input/burgers_aligned.toml)):

| method | file | nonlinearity | purity |
|---|---|---|---|
| `cole_hopf_circuit` | `burgers_cole_hopf_circuit.py` | linearised (Cole-Hopf) | **pure quantum** (no classical mirror in the time loop) |
| `qlbm_circuit` (today) | `burgers_qlbm_circuit.py` | BGK collision | **hybrid** (classical collision mirror every step) |
| `ftcs_reference` | `burgers_lbm.py` / solver | — | classical reference |

This spec makes the QLBM pathway pure quantum, matching the purity
guarantee F10 established for the Cole-Hopf propagators.

### 0.1 Naming / retirement (decided, minimal footprint)

The existing hybrid (Option A) is **retired under its own name**
`qlbm_circuit_hybrid` and kept permanently as the cross-validation
oracle. The bare name `qlbm_circuit` is reserved for the pure-quantum
QALB target. The Phase-1 precursor (#29) registers as
`qlbm_circuit_linear`.

**Confirmed touch-set (this is the whole rename):**

| file | line(s) | change |
|---|---|---|
| `src/burgers_fw.py` | `_DELEGATING_METHODS` (~436), `make_integrator` dispatch (~480) | register `qlbm_circuit_hybrid` → `QLBMCircuitIntegrator` |
| `src/burgers_solver.py` | `--method` `choices` (~241), module docstring (~28), help text (~249) | add `qlbm_circuit_hybrid` |
| `docs/OVERVIEW-burgers-solver.md` | method table (~164), §5.2, scattered refs | rename + note retirement |

Everything else stays:

- **Internal identifiers unchanged** — `QLBMCircuitIntegrator`,
  `run_qlbm_circuit_simulation`, module `burgers_qlbm_circuit.py`, and
  the `[qlbm_circuit]` stderr log prefix are implementation, not the
  user-facing method string. Leave them. (Two stale code comments —
  `burgers_postprocess.py:256`, `burgers_lbm.py:132` — may be touched
  opportunistically; not required.)
- **Existing run artifacts untouched** — saved JSON keeps its
  `case_id` / `algorithm` strings. Historical reproductions are frozen.
- **TOMLs untouched** — `burgers_aligned.toml` keeps `[qlbm_circuit]`.

**Back-compat during the interim.** Because the bare name
`qlbm_circuit` is reserved for QALB but QALB does not exist yet, the
dispatcher **retains `qlbm_circuit` as an alias to the hybrid** so
existing TOMLs keep running. When Phase 2 lands, `qlbm_circuit` is
repointed to the QALB integrator and the alias dropped — at which
point an unchanged TOML automatically gets the pure-quantum method
(the intended end state). Register both names now; both route to
`QLBMCircuitIntegrator` until Phase 2 flips the target.

> **Implementer note.** Specified here, **not yet executed.** The
> rename is implementation step 0.

---

## 1. Why today's `qlbm_circuit` is hybrid, and why it is forced to k=1

Read `burgers_qlbm_circuit.py:399-520`. On the shots path each LBM
step does, in order:

1. **classical** `f_post = collide_bgk(f, tau)` inside
   `build_qlbm_step_circuit` (`burgers_lbm.py`), then Householder-
   dilate a unitary that maps `psi_in → psi_out` (the classical
   answer);
2. `qc.initialize(psi_in)` — re-prepare the current `f` as a state;
3. one collide+stream unitary;
4. `measure_all()`;
5. **classical** reconstruct `f` from counts (`sqrt(counts/S)` +
   `hadamard_test` sign recovery);
6. feed `f` to the next step's `initialize`.

So **today's method is already measure-and-reprepare with segment
length k = 1** — and worse, the collision physics is on the CPU.

**The root cause is the encoding.** Today `f` is *amplitude-encoded*
(the 4N populations are statevector amplitudes on a `q+2`-qubit
register; see `flatten_distributions` in `burgers_lbm.py:293`). The
BGK collision is *nonlinear in those amplitudes*
(`f_eq ∝ u²`, `u = Σ f_i c_i / ρ`), so the collision unitary
**depends on the current state** and must be rebuilt every step from
a classically known `f`. State-dependent collision ⟹ must know `f` ⟹
must measure every step ⟹ **k = 1 is mandatory, not a choice.**

This is the same architectural pattern F10 eliminated for the
direct-`u` Pauli path and Cole-Hopf eliminated via linearisation. The
QLBM fix is the subject of this spec.

---

## 2. The target: a *state-independent* collision unlocks k > 1

`--evolution-mode measure_reprepare` and `--segment-size` already
exist for Cole-Hopf (see
[../archive/SPEC-measure-reprepare-evolution.md](../archive/SPEC-measure-reprepare-evolution.md)).
A segment of `k` steps runs `k` propagator layers coherently in one
circuit, measures once, re-prepares, repeats. CH can do this because
its heat propagator is **state-independent** (fixed for all steps).

To give QLBM the same capability we need a collision operator that
does **not** depend on the current `f`. Itani's QALB provides exactly
this. The dial `k` (coherent LBM steps between measurements) then
trades circuit depth / NISQ noise against quantum coherence, with
`k = 1` recovering today's behaviour and `k = n_steps_lbm` being
fully coherent. **k is the honest measure of how pure the run is.**

---

## 3. The Itani QALB construction (specialised to 1-D D1Q3 Burgers)

### 3.1 Mode-coupling LB form

Itani Eq. (9):
`f_i(x + c_i Δt, t+Δt) − f_i(x,t) = (1−ω) f_i(x,t) + ω f_i^eq(x,t)`,
with `ω = Δt/τ`. Collision (local) then streaming (shift by `c_i`).
For us `D = 1`, `Q = 3`, `c = (−1, 0, +1)`, and the equilibrium is the
athermal-Burgers form already in `burgers_lbm.py:30`:
`f_0^eq = ρ(1−u²)`, `f_{±1}^eq = ρ(u² ± u)/2`.

### 3.2 The D1Q3 collision and its single quadratic nonlinearity

Itani Eq. (7), our indexing `f = (f_{-1}, f_0, f_{+1})`:
```
Ω(f) = −(1/τ) · [ ½(f_{-1}+f_{+1} − (f_{-1}−f_{+1})² − ⅓)
                  ( f_0          + (f_{-1}−f_{+1})² − ⅔)
                  ½(f_{-1}+f_{+1} − (f_{-1}−f_{+1})² − ⅓) ]
```
The **only** nonlinearity is the quadratic `(f_{-1}−f_{+1})² = (ρu)²`.
This is what makes the Carleman/Hermitization machinery tractable for
this case.

### 3.3 Encoding decision: value/Fock, not amplitude  ← the crux

This is the central architectural commitment of the whole spec.

- **Amplitude encoding (today).** A state-independent *full-BGK*
  collision is **impossible** here: the operator is nonlinear in the
  amplitudes, so no fixed unitary reproduces it. This is provably why
  Option A rebuilds the collision per step. (Linearised BGK *is* a
  fixed operator in amplitude encoding — that is Phase 1 / #29, §5.)

- **Value/Fock encoding (Itani).** Each discrete density `f_i` is the
  *value* stored in its own `qc`-qubit bosonic register (a truncated
  Fock space; Itani §VI, App. B). Position `q̂_i` and momentum `p̂_i`
  operators act on that register. In this encoding the collision can
  be written as a **fixed Hamiltonian** (next section) — *state
  independent* — because the nonlinearity is now an operator
  (`q̂_i²`), not a function of amplitudes. Qubit count is logarithmic
  in the Fock-space size.

**Adopting value/Fock encoding is the price of admission for #27.**
It is a different register layout from today's solver and is the bulk
of the implementation work.

Register layout (1-D, D1Q3, `N = 2^q` sites):

| register | qubits | purpose |
|---|---|---|
| densities | `Q · qc = 3·qc` | value of `f_{-1}, f_0, f_{+1}` |
| lattice position | `log₂ N = q` | site index, in superposition |
| lattice direction | `2D = 2` | stream control (Itani Table I) |

Total ≈ `3·qc + q + 2`. At `q = 6`, `qc = 2`: **≈ 14 qubits.**
Itani's numerics (§VIII.A.1) find **qc = 2 is the most accurate**
(lower error than qc = 3, 4; error grows with distance of `f_i` from
equilibrium, and 2nd-order LB has no truncation error). **Default
qc = 2**; expose `--fock-qubits` for sweeps.

### 3.4 Collision via Hermitization (recommended) + constant-divergence deflation

Itani §VII.A.1. Write the collision generator as a Hamiltonian
(Itani Eq. (79-80)):
```
Ĥ = ½ Σ_i [ p̂_i Ω_i(q̂) + Ω_i(q̂) p̂_i ] + ½ Σ_i [ p̂_i, Ω_i(q̂) ]
        └─────── Hermitian Ĥ' ───────┘   └──── anti-Hermitian ────┘
```
Evolve the **Hermitian part** unitarily, `e^{−iΔt Ĥ'}` (Itani
Eq. (85)). Ĥ' is **fixed** — built once from the collision functional
`Ω`, independent of the live state. *This is the property that makes
k > 1 possible.*

**The dissipation is a deterministic global rescale, not a
post-selection.** For the *incompressible* case the anti-Hermitian
(contractive) part is a **constant** divergence (Itani Eq. (82-83)):
```
Σ_i ∂Ω_i/∂q̂_i = −(1/τ)(Q − D) Î = −(2/3τ) Î    (D1Q3)
```
So the physical `f` is recovered from the unitarily-evolved state by
multiplying by the **known scalar** `e^{(t·Δt/2)(2/3τ)}` (Itani
Eq. (84,86)) — a uniform deflation applied in post-processing.

> **This corrects the earlier scoping assumption.** I previously
> expected a per-step block-encoded collision with success
> probability `p`, compounding as `p^k` and capping `k`. For the
> Hermitization route on incompressible D1Q3 that **does not happen**:
> the collision evolution is genuinely unitary and the dissipation is
> a deterministic constant deflation. There is **no `p^k`
> post-selection penalty from the collision.** This is precisely why
> Itani calls Hermitization the best of his routes (§VII.A.3):
> (1) no extra `D` velocity variables, (2) the dissipative part is
> exact, (3) deflation post-processing behaves well. The remaining
> cost moves into **Hamiltonian-simulation depth** for `e^{−iΔt Ĥ'}`
> (§3.7).

### 3.5 (Deferred) Hydrodynamic-variables collision

Itani §VII.A.2 gives an alternative: co-encode `|u⟩` and apply
collision as compute-`u` → relax → uncompute-`u`. Itani himself
recommends *against* it: resetting `u` to zero mid-circuit is not
exactly unitary (needs mid-circuit measurement), it adds `D` velocity
registers, and rescaling injects numerical error. **Do not
implement.** Documented only so a future reader does not re-derive it.

### 3.6 Streaming

Itani §VII.B. Streaming is the lattice-position increment/decrement
controlled by the direction register — exactly the permutation today's
`build_streaming_unitary` (`burgers_qlbm_circuit.py:72`) performs, but
Itani builds it as a **logarithmic-depth controlled-CX cascade**
(Itani Eq. (114-115), Fig. 6) instead of a dense `UnitaryGate`:
```
Ŝ_{d+} = X̂_0 · [CX]^1_{1,0} ··· [CX]^{log₂(N)-1}_{1...log₂(N)-1, 0}
Ŝ_{d-} = same with complemented control state
```
Periodic BC is free (cyclic increment). Reuse the existing
`_increment_gate`/`_decrement_gate` but emit the cascade form for
hardware-honest depth. Streaming is **exactly unitary** — no loss,
no deflation.

### 3.7 Hamiltonian simulation of Ĥ'

`e^{−iΔt Ĥ'}` is realised by Trotter or truncated-Taylor LCU
(Itani §IX uses the Berry et al. method [65]; the preprint TOC §X.A.1
is explicitly "LCU from a truncated Taylor series"). **We already
have LCU SELECT/PREPARE machinery** in `burgers_lcu.py` (built for the
Cole-Hopf LCU propagator, see
[../archive/SPEC-F3-LCU-method.md](../archive/SPEC-F3-LCU-method.md)).
Reuse it: Ĥ' is a fixed sparse operator in the value/Fock encoding, so
its LCU block-encoding is built **once** and applied each of the `k`
steps in a segment. Trotter error and Taylor truncation order are the
accuracy knobs here; expose `--trotter-reps` (already a solver flag)
and a Taylor order if LCU is used.

### 3.8 What Phase 2 costs, concretely (D1Q3, Q=3)

Itani §IX gives closed forms; specialised to `Q = 3`, `D = 1`,
`N_lat = 2^q` sites, `qc` Fock qubits per density (so each Fock
register holds `2^qc − 1` excitation levels, and `⌈log₂(N_fock+1)⌉ =
qc`):

- **Collision Hamiltonian = a fixed LCU of monomials.** The Hermitized
  BGK Ĥ' is a linear combination of `m = Q² + 2Q + 2` position/momentum
  monomials (Itani Eq. (132)). For `Q = 3`: **m = 17 monomials.** Each
  maps to `qc²` Pauli words (Itani Eq. (133)), so the LCU has
  `L = m·qc² =` **68 Pauli terms** at `qc = 2`. The collision LCU needs
  `O(log₂ L) ≈` **6–7 ancilla qubits** (Itani Eq. (142)); the Berry-et-al.
  `log/loglog` overhead is ≈ 1 in practice and is dropped to match the
  first-order-in-time LB accuracy (Itani Eq. (138–142)). Built **once**,
  reused every step in a segment.
- **Qubit budget** (collision + binary-position streaming, Itani
  Table III, best row): `O(log₂ G + Q)` data + `O(log₂(Q log₂ G))`
  ancilla. For `q = 6`, `qc = 2`: ≈ `3·2 + 6 + 2 = 14` data + ~7 LCU
  ancilla ≈ **~21 qubits.** Width is not the constraint.
- **Depth is the constraint, and it is the open quantum-advantage
  question.** The collision gate count scales as a **power of the
  number of timesteps `T` and of `Q`** (Itani Eq. (143), §IX.B): the
  position-embedded collision+streaming is "the only method that does
  not show linear scaling in volume `G`," but it pays *polynomial in
  `T`*, so **Itani explicitly states no quantum advantage is achieved
  yet.** This is the number to estimate before any Phase-2 hardware
  run, and it is exactly where our segmenting changes the picture
  (§4.4 — `T` per circuit becomes `k`, not `n_steps_lbm`).
- **Regime:** `τ > 1/2` for finite Reynolds (Itani Eq. (138), ref [53]);
  `Δt/τ < 2` (Itani Eq. (66)). Our aligned case (`τ ≈ 2.4` at
  `nu = 3e-2`) sits comfortably inside.

---

## 4. Architecture: measure-reprepare(k) IS Itani's measurement resolution

Itani §VIII.A proves a hard limitation: at the **full lattice**, a
*simultaneously* unitary collision (needs `f` values tensored,
Itani Eq. (117)) and unitary streaming (needs position superposition,
Itani Eq. (116)) are **incompatible** — acting on the shared position
register forces all densities at a site to stream the same direction
(Itani Eq. (120)). His resolutions: either scale qubits/gates
**linearly in lattice volume G** (Itani Eq. (121-122),
`O(min(G, T+log₂G)·Q·log₂(N+1))` qubits — kills the advantage), **or
break the superposition by measurement.**

**We choose measurement — which is exactly measure-reprepare.** So our
segmented architecture is doubly motivated: (a) NISQ depth limits,
(b) Itani's own streaming/collision incompatibility. Within a `k`-step
segment we keep things coherent; at the boundary we measure and
re-prepare. Nobody (including Itani, whose numerics are single-site
0-D collision only, §VIII.A.1) has demonstrated a clean full-lattice
*pure-unitary* QALB; the measure-reprepare(k) realisation on Burgers
is the deliverable. Set expectations accordingly.

### 4.1 Segment semantics and alignment (hard constraints)

- `k` is counted in **lattice (LBM) steps**, native clock
  `n_steps_lbm` (today `= round(cfl · n_steps) = 10` for the aligned
  case). Map `--segment-size` (caller steps) → lattice steps
  explicitly; do **not** silently round (see the alignment block in
  `burgers_aligned.toml`).
- Require **`k | n_steps_lbm`** so segment boundaries land on stored
  snapshots. For `n_steps_lbm = 10`, `k ∈ {1, 2, 5, 10}`.
- The common comparison grid `{0,10,…,100}` (caller steps) must stay
  valid; QALB snapshots at every `k`-th lattice step.

### 4.2 Reprep invariant: full distribution, never `f_eq` from moments

At a boundary, reconstruct and reload the **full off-equilibrium `f`**
(all `Q·N` values). **Never** measure moments `(ρ,u)` and re-prepare
`equilibrium(u)` — that re-equilibration is the classical BGK
collision smuggled back in, i.e. Option A. State this as an enforced
invariant and assert it in code (the reprep takes a measured `f`
array, not a `u` array). This mirrors CH, whose reprep re-prepares the
*measured* `φ` via MPS/`initialize`, never re-solves the heat step on
the CPU.

### 4.3 Shots / hardware readout (always-on)

Runs are **always shots-based and hardware-ready** (no statevector
endpoint). Route through the shared
`q8020_cfd_qutil.circuit` helpers (`transpile_circuit`,
`execute_circuit_counts`, honest-depth metric path) already used by
both shots methods, so `--optimization-level` / `--seed` / SamplerV2
behave consistently. Boundary readout in the value/Fock encoding is a
**per-register value read** of each `f_i` (qc qubits each) across the
`N` sites; specify the estimator and its shot-cost model
(`shots` vs. resolvable gradient at a front — the failure mode that
caused the historical step-90 blow-up in the hybrid method). Sign of
`f_i` is carried in the Fock-register value, so no separate
`hadamard_test` pass is needed *inside* a segment; characterise
whether it is needed at boundaries for the chosen estimator. Near-term
use raw shots (`1/√shots`); keep the readout interface clean so
amplitude estimation (Heisenberg `1/N`, deeper circuits) can drop in
later. See
[../archive/SPEC-qlbm-shots-and-sign-recovery.md](../archive/SPEC-qlbm-shots-and-sign-recovery.md).

### 4.4 Segmenting is what tames the Fock-truncation error — `k` is also an *accuracy* knob

This is the strongest argument for our architecture and it comes
straight from Itani's error analysis (Appendix A).

The error from truncating the bosonic Fock space at `qc` qubits
(`ε_N`, one truncation-tail unit per step) **does not propagate
linearly** — it obeys a **logistic-map recurrence** (Itani Eq. (A43)):
```
ε(t+1) = (Δt/τ) · ( C₁(ε_N + ε(t)) + C₀ )²
```
with `C₁ = O(Q)`, `C₀ = O(1/τ)` constants (Itani Eq. (A23–A33)). A
logistic map is quadratic and can **diverge**. Itani then proves
(§A.6, "Impossibility of a General Time-Independent Bound") that for
`Δt/τ = O(1)` **no time-independent error bound exists** — the
truncation error is only controllable as `Δt/τ → 0` (`Re → 0`) or over
*short* horizons (Itani Eq. (A84)).

**Measure-reprepare(`k`) is precisely the mechanism that caps the
horizon.** Three distinct effects, all per-segment:

1. **Bounded error horizon.** The logistic recurrence runs only over
   the `k` coherent steps inside a segment, not over `n_steps_lbm`. We
   keep `k` small enough that `ε(t)` stays in the pre-divergence basin
   of Eq. (A43). The run becomes `n_steps_lbm/k` independent
   short-horizon segments, each with controllable error — converting
   an unbounded-over-`T` divergence into bounded pieces.
2. **Fock-state reset.** At a boundary we measure `f`, reconstruct the
   classical values, and re-prepare a **fresh truncated position
   eigenstate** (Itani Eq. (A1/A7)). The accumulated higher-order
   Hermite leakage (the truncation tail) is projected back onto a
   clean truncated state, so each segment restarts at `ε(0) ≈ ε_N`
   rather than the accumulated `ε(t)`. Segmenting literally **resets
   the error accumulation** every `k` steps.
3. **Deflation stays numerically benign.** The constant-divergence
   deflation scalar (§3.4) grows as `e^{(t·Δt/2τ)(2/3)}`. Over the
   whole run that is a large factor that amplifies relative shot noise;
   per-segment, `t` resets to `≤ k`, keeping the deflation small.

So the *same* dial `k` that trades circuit depth ↔ purity (§2) and
that turns the `O(poly T)` collision cost into `O(poly k)` per circuit
(§3.8) **also** trades truncation-error-divergence ↔ readout overhead.
That triple role makes `k` the central tuning parameter of Phase 2.
The design task is to find the `k` that simultaneously: fits NISQ
depth, keeps `ε(t)` sub-divergent, and is affordable in boundary
readout — and to report it (a low `k` is honest about how much
coherence the run actually achieved).

### 4.5 MPS-compressed boundary reprepare (compression option)

The binding practical cost (Risk #2) is re-preparing the full
distribution `f` at each of the `n_steps_lbm/k` boundaries. **Reuse the
Cole-Hopf compression trick:** CH re-prepares its boundary field via
`classical_to_mps(psi, bond_dim) → mps_to_circuit`
(`burgers_mps.py`), trading a `--bond-dim` parameter (≈ `log₂` extra
qubits) for a low-depth prep instead of an exponential `initialize`.
CH finds **bond-dim 4 "converged for smooth φ."**

The same physics applies here: the distribution `f_i(x)` is **smooth
in `x`** in the pre-shock regime, i.e. **low entanglement across the
position register**, so an MPS over the position qubits compresses the
reprep with modest bond dimension. Concretely:

- Add `--bond-dim` to QALB; route the boundary reprep through
  `burgers_mps.py` exactly as CH does.
- **Register ordering matters.** Unlike CH's single amplitude-encoded
  field, the QALB boundary state is structured `|f_i⟩|e_i⟩|x⟩` with
  per-site Fock registers. Order the MPS so the **position qubits
  carry the spatial bond** (where the smoothness lives) and the small
  `Q·qc` Fock registers ride locally. Specify this ordering; it is the
  one non-trivial design point.
- **Bond dimension at the front is a free diagnostic.** As a shock
  forms, spatial entanglement rises and the required bond dimension
  climbs — the same trade CH sees. Track it: bond-dim-at-the-front is
  both the cost knob and a physical proxy for "how hard is this state
  to represent," and it tells you when `k` (or `qc`) must change.

This is the recommended answer to the compression question: MPS at the
**reprep boundary**, not as a replacement encoding for the evolution
(that would be a different, TEBD-style method — we already have `tebd`/
`mps`). It attacks Risk #2 directly and reuses shipped, tested code.

---

## 5. Staging: Phase 1 (#29) → Phase 2 (#27); #28 is subsumed

**#28 (Carleman) is not a separate method.** Itani's QALB *is* a
Carleman/Kowalski second-quantized scheme; the Hermitization of §3.4
is built on that lift. So #27 and #28 collapse into the single Phase-2
construction below. Do not implement a third "Carleman" method.

### Phase 1 — `qlbm_circuit_linear` (#29), low-Mach, **amplitude encoding**

Linearise BGK about the rest equilibrium `f_eq⁰ = (0,1,0)`
(`u_ref = 0`, `ρ = 1`); set `δf = f − f_eq⁰`. The collision is then a
**fixed linear operator** — and critically, **fixed in the existing
amplitude encoding**, so Phase 1 needs **no re-encoding**: same
`q+2`-qubit register as today, but the collision unitary is built
**once** (state-independent) and applied across a `k`-step segment.

The per-site operator (implemented and gate-verified in
`burgers_qlbm_linear_circuit.py`, `ω = 1/τ`):
```
f* = M₃ f + b₃,  M₃ = [[1-ω/2, 0, -ω/2],[0, 1-ω, 0],[-ω/2, 0, 1-ω/2]],
                  b₃ = (0, ω, 0)
```
Work in `δf`, **not** `f`: since `M₃ f_eq⁰ + b₃ = f_eq⁰`, the affine
offset cancels and `f* − f_eq⁰ = M₃ (f − f_eq⁰)` — a **purely linear**
map, no `+b₃` to embed. `f_eq⁰` is stream-invariant (only the
non-shifting rest population), so `δf` is consistent through both
collision and streaming. Block-encode `M₃ ⊗ I_N` on `δf`; the rest
equilibrium is added back classically at readout.

**Offline gate (done, spec §6 gate 1):** `M₃` reproduces `collide_bgk`
to machine precision at equilibrium and to `O(u²)` (slope 2.00) near
it — the correct statement for a linearised operator.

Purpose of Phase 1:
- Stand up and validate the `qlbm_circuit_linear` method end-to-end on
  the **converging low-Mach case** (the aligned toml at `nu=3e-2`,
  `amp=0.3`, which is well pre-shock — exactly this regime's validity
  domain), where FTCS gives ground truth.
- Build and test the **measure-reprepare(k)** plumbing, the
  **full-`f` boundary readout** (§4.2-4.3), and the `k | n_steps_lbm`
  alignment — with shocks and the value/Fock re-encoding held out, so
  only one new thing is in play.

Catch (document prominently): linearised BGK **loses shock physics**;
valid only for smooth, low-Mach, near-equilibrium flow. Phase 1 is a
de-risking scaffold and a pedagogical pure-quantum benchmark, **not**
the production solver.

### Phase 2 — `qlbm_circuit` (#27), full BGK + shocks, **value/Fock encoding**

The real target. Switch to value/Fock encoding (§3.3), Hermitization
collision Hamiltonian + constant-divergence deflation (§3.4),
log-depth streaming (§3.6), Hamiltonian-sim via reused LCU (§3.7), all
under measure-reprepare(k). Validity domain includes the interesting
(shock-forming) Burgers regime, bounded by: Fock truncation `qc`,
Trotter/Taylor error, and the incompressible assumption
`ρ = 1 + O(Ma²)` (a low-Mach caveat on *density*, but the `u²` velocity
nonlinearity is retained — unlike Phase 1). Flag the `Ma` ceiling and
check it for `amp ∈ {0.5, 0.8}`.

---

## 6. Module layout, coexistence, validation

- **`burgers_qlbm_circuit.py`** — unchanged code; method string
  becomes `qlbm_circuit_hybrid`. Permanent hybrid oracle.
- **`burgers_qlbm_linear_circuit.py`** (new) — Phase 1,
  `--method qlbm_circuit_linear`, amplitude encoding, fixed linear
  collision.
- **`burgers_qalb_circuit.py`** (new) — Phase 2,
  `--method qlbm_circuit`, value/Fock encoding, Hermitization.
- Reuse from `burgers_lbm.py`: `equilibrium`, `collide_bgk`, `stream`,
  `tau_from_nu` as the **classical oracle** and for building Ĥ'/the
  linear operator offline.
- Reuse from `burgers_lcu.py`: SELECT/PREPARE LCU for `e^{−iΔt Ĥ'}`.
- Reuse the CH measure-reprepare segment loop structure
  (`burgers_cole_hopf_circuit.py:~1090-1290`): segment build,
  per-segment `p_success`/`cumulative_norm` (here the deflation
  scalar, §3.4) tracking, `initialize`-based reprep.

**Validation gates (each phase):**
1. **Operator-exactness (offline, no circuit):** the constructed
   collision operator (linear op for P1; Ĥ' action for P2) reproduces
   `collide_bgk` (Itani Eq. (7)) to machine precision on random `f` in
   the valid regime. This is the gate before any circuit work.
2. **Statevector parity (dev only, not a shipped mode):** one
   internal statevector check that a `k`-step segment matches the
   classical LBM trajectory within Fock/Trotter error — used to
   isolate algorithm error from shot noise during bring-up. Shipped
   runs remain shots-only.
3. **Shots vs. `qlbm_circuit_hybrid`:** agreement on the aligned
   converging case, reusing `plot_method_compare.py` and the
   `{0,10,…,100}` grid.

---

## 7. Open questions / risks

1. **Hamiltonian-sim depth** of `e^{−iΔt Ĥ'}` (L ≈ 68 Pauli terms,
   §3.8) at `qc = 2`, `q = 6`, applied `k` times per segment — is it
   NISQ-plausible for `k ∈ {2,5}`? This, not post-selection, is the
   binding cost, and the `O(poly T)` collision scaling (no quantum
   advantage yet, Itani §IX.B) is exactly what segmenting reduces to
   `O(poly k)` per circuit (§4.4). Estimate transpiled depth/CX before
   any Phase-2 hardware run.
2. **Boundary readout + reprep cost** for the full `f` across `N`
   sites, and the shot budget to resolve a shock front (§4.3). Likely
   the practical bottleneck; the MPS-compressed reprep (§4.5) is the
   mitigation. Quantify on Phase 1 first.
3. **Fock truncation vs. shock amplitude.** qc = 2 is most accurate at
   low amplitude (Itani §VIII.A.1); a steep front may need larger `qc`
   *and* a smaller `k` to keep the logistic error (§4.4) sub-divergent
   — these couple. Sweep `--fock-qubits` × `k` against `amp`.
4. **Incompressible `Ma` ceiling** (`ρ = 1 + O(Ma²)`): validate the
   approximation holds at `amp = 0.5`/`0.8`.
5. **Data loading** initialisation is `O(G)` (the universal hurdle,
   Itani §VIII.A.1); acceptable here since `N` is small, but note it.

---

## 8. References

- W. Itani, K. R. Sreenivasan, S. Succi, "Quantum Algorithm for
  Lattice Boltzmann (QALB) Simulation of Incompressible Fluids with a
  Nonlinear Collision Term," *Phys. Fluids* **36**, 017112 (2024);
  arXiv:2304.05915.
- Sanavio, Succi et al., "Three Carleman routes to the quantum
  simulation of classical fluids," *Phys. Fluids* **36**, 057143
  (2024) — context for the Carleman framing of #28.
- In-repo: [FUTURE-WORK.md](FUTURE-WORK.md) #27-#29;
  [../archive/SPEC-measure-reprepare-evolution.md](../archive/SPEC-measure-reprepare-evolution.md);
  [../archive/SPEC-qlbm-shots-and-sign-recovery.md](../archive/SPEC-qlbm-shots-and-sign-recovery.md);
  [../archive/SPEC-F3-LCU-method.md](../archive/SPEC-F3-LCU-method.md);
  [../archive/F11-QLBM-SPEC.md](../archive/F11-QLBM-SPEC.md).
