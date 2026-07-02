# Quantum Methods for the 1-D Viscous Burgers Equation
## A technical reference for the `q8020-cfd-ch-lbm` solver

*May 2026 — LLM generated, human edited.*

---

## Preamble

The `q8020-cfd-ch-lbm` codebase (formerly `q8020-mps-burgers`) grew up ad
hoc, and has since had a haircut: the surviving methods are the ones below,
with the many earlier exploratory pathways removed. In the end the pathway
we actually care about scientifically is the **pure-quantum Cole–Hopf
pathway** (`cole_hopf_circuit`, this work); the classical (`shift`,
`ftcs_reference`, `lbm`) and kinetic (`qlbm_circuit` QALB) variants are kept
as cross-method comparisons and validation references. A front-end TOML
driver (see project repo q8020-cfd-metautil) feeds this solver, and its data
and metadata are harvested into the q8020 *cases × codes × backends* rollup.

The original goal was a curiosity: would it be possible to derive a quantum
solution from a published mathematical description — science publications
make good specifications, mathematics being a good specification language —
and to validate it against an existing classical solution, with AI tooling
providing "independent" review, plus human software and SME review.

We are also now interested, again in the *cases × codes × backends*
modality, in running the same case on more than one code implementing the
same Burgers equation, e.g. QLBM.

The Cole–Hopf substitution linearizes Burgers into the linear heat
equation, giving a *state-independent* generator that can be marched as a
fixed quantum circuit — no per-step Hamiltonian rebuild, no classical
mirror in the time loop. The circuit realises the heat propagator in its
eigenbasis (QFT for periodic, DCT-II for Dirichlet-on-`u`) with a single
block-encoding ancilla per step. We do not concede the phase problem, and
we do not require a classical solution to be computed simultaneously to
steer the quantum trajectory, as is so often the case in the literature.
We also added 1-D QLBM (the pure-quantum QALB) as a second quantum pathway
for cross-method comparison on the same case.

---

## 1. Governing equation and discretization

### 1.1 The PDE

The 1-D viscous Burgers equation with optional source:

```
∂u/∂t + u · ∂_x u = ν · ∂²_x u + g(x, t)                                (1)
```

with boundary conditions `u(0,t) = u(L,t) = 0` (homogeneous Dirichlet) or
`u(0,t) = u(L,t)` (periodic). The domain is `x ∈ [0, 1]` and the grid is
`N = 2^q` equispaced points `x_j = j · δx`, `j = 0, …, N−1`, with
`δx = 1/N`. The integer `q` is both the grid-refinement parameter and the
qubit count for the quantum methods. Both pathways encode the velocity as
a quantum state

```
|ψ⟩ = Σ_k u_k |k⟩ / ‖u‖                                                  (2)
```

on `q` qubits, where `k` indexes grid cells in binary.

Initial conditions: `sine` (single mode `u₀(x) = sin(2πx)`), `multimode`
(random sum of low-wavenumber modes), `gaussian` (localized pulse
`u₀(x) = A·exp(−((x − x₀)/σ)²)`, useful for shock-formation demos; no
analytic reference, pairs with FTCS), or `cole_hopf_exact` (a
Neumann cosine sum on `φ` that admits a closed-form analytic `u(x, t)`
under unforced Cole–Hopf heat evolution — see §4.3). Sources: `sine`
(Gopalakrishnan Meena AIAA-2026 reference test problem,
`g(x,t) = sin(2πx)·cos(2πt)`) or `none`.

The CLI is [`burgers_solver.py`](../src/burgers_solver.py); by default
it runs a reference trajectory alongside the chosen method and reports
the L² error against it. The reference is chosen automatically:

- `--ic cole_hopf_exact` (the default when `--method` is
  `cole_hopf_circuit`): the closed-form analytic `u(x, t)` evaluated at
  each saved step — exact, microsecond cost.
- Otherwise: a classical Forward-Time Central-Space (FTCS) Burgers
  solve on the quantum grid itself (`solve_burgers`, forward-Euler with
  shift-operator FD). A **resolution-decoupled** FTCS truth is also
  available as its own `--method ftcs_reference`: it runs on a refined
  grid of at least `--ref-points` nodes chosen so the `N = 2^q` quantum
  nodes are an exact subset (BC-aware), with internal substepping for
  explicit stability, then subsampled back to the quantum nodes for
  pointwise scoring (no interpolation error). This makes that method a
  converged "truth" rather than a same-grid coarse run.

Flags:

- `--ref-points`: minimum grid size for the `ftcs_reference` method's
  resolved FTCS solve (default 0 = solve on the q-grid, no refinement).
  No effect on the analytic reference (exact at any resolution).
- `--no-analytic-reference`: under `--ic cole_hopf_exact`, fall back to
  the FTCS reference (e.g. for cross-validating the analytic
  formula against the classical solver).
- `--no-classical-reference`: skip the reference trajectory entirely
  (no error metrics, no `speedup_ratio` in the analysis fragment).

Quantum trajectories may have pre-processing, but once rolling do not
refer back to the classical solution for steerage.

### 1.2 Spatial discretization

Spatial derivatives use the shift-operator central-difference stencil
(Meena et al. AIAA-2026 Eq. 9):

```
∇u   ≈ −(S⁺ − S⁻) u / (2·δx)                                            (3)
∇²u  ≈  (S⁺ + S⁻ − 2I) u / δx²                                          (4)
```

where `S⁺` and `S⁻` are the cyclic forward and backward shift matrices:
`(S⁺u)_j = u_{j−1}`, `(S⁻u)_j = u_{j+1}`. For Dirichlet BC the shift
wrap-around is suppressed: `S⁺[0, N−1] = S⁻[N−1, 0] = 0`, and the RHS at
boundary indices is forced to zero (`∂u/∂t = 0` at `x = 0, L`).

---

## 2. Method families and classification

`--method` selects the evolution scheme. Methods divide into three
families:

- **Direct-u**: march `u` directly (one method).
- **Cole–Hopf**: linearize via `u = −2ν ∂_x ln φ` and march `φ` (one method).
- **QLBM (kinetic)**: march mesoscopic distributions `f_i` and recover `u`
  by moments (two methods).

Each method is classified by how much classical machinery it carries:

| Class | Symbol | Meaning |
|---|---|---|
| Classical | C | No quantum objects |
| Pure quantum | PQ | No classical mirror in the time loop |

Method roster:

| `--method` | Family | Class | Notes |
|---|---|---|---|
| `shift` | Direct-u | C | Explicit forward-Euler shift-FD baseline |
| `ftcs_reference` | Direct-u | C | Resolution-decoupled FTCS truth trajectory |
| `cole_hopf_circuit` | Cole–Hopf | PQ | **Pure-quantum headline pathway (this work)** |
| `lbm` | QLBM | C | Classical D1Q3 BGK (renamed from `qlbm` — pure classical, no shots) |
| `qlbm_circuit` | QLBM | PQ | **Pure-quantum QALB (Itani-style; this work).** Value encoding (App C finite-position on the statevector path, App B bosonic on the shots path), normal-ordered Hermitised collision `e^{−iΔtĤ′}` (unitary, **no** post-selection) + streaming, `⟨q̂⟩` shots readout; `--fock-qubits` (qc, default 3). Collision is pure-quantum (no classical mirror); streaming classical, measure-reprepare k=1 today. See §5.2. |

The **headline pathway** is `cole_hopf_circuit` (pure-quantum; no
classical mirror in the time loop after `φ₀` is prepared from `u₀`). It
is the primary scientific object of this codebase and is detailed
mathematically in §4.2. `qlbm_circuit` is the **pure-quantum QALB**
(Itani-style; §5.2) — a second quantum pathway from a different
algorithmic family (kinetic), with a state-independent Hermitised
collision. The classical methods (`shift`, `ftcs_reference`, `lbm`)
exist as references and diagnostics.

---

## 3. Direct-u family

This family marches `u` directly. Only the classical `shift` baseline
survives; it doubles as the building block for the FTCS reference
trajectory.

### 3.1 `shift` (classical)

Explicit forward-Euler with central-difference shift-operator FD on `u`.
Pure classical baseline; no quantum objects. `O(N)` per step.

---

## 4. Cole–Hopf family

This family exploits the Cole–Hopf linearization to transform the
nonlinear Burgers equation into the linear heat equation, which can be
evolved with a time-independent quantum circuit — eliminating the
per-step Pauli decomposition entirely.

### 4.1 The Cole–Hopf substitution

Define the potential function

```
φ(x, t) = exp(−(1/(2ν)) ∫₀ˣ u(s, t) ds)                                 (10)
```

Then `φ > 0` everywhere, and the unforced Burgers equation reduces to
the linear heat equation

```
∂φ/∂t = ν ∇²φ                                                           (11)
```

The inverse transform recovers the velocity

```
u(x, t) = −2ν ∂_x ln φ(x, t)                                            (12)
```

The implementation uses cumulative trapezoidal integration for the
forward transform (Eq. 10) and a central-difference log-derivative for
the inverse (Eq. 12). For numerical stability at small `ν` the code
operates in log-space: it computes `e(x) = −(1/(2ν)) ∫ u`, centres the
exponent (`e − e_mid`), and converts to unit-norm `ψ` via log-sum-exp
normalization.

**BC mapping**: Dirichlet on `u` (`u(0) = u(L) = 0`) maps to Neumann on
`φ` (`∂φ/∂x = 0` at boundaries) via Eq. 12. This is why the eigenbasis
for Dirichlet problems is the DCT (cosine transform), not the DST or
QFT.

### 4.2 `cole_hopf_circuit` — Pure-Quantum headline pathway

The heat equation is marched as a quantum circuit. Because Cole–Hopf
turns Burgers into a *linear* PDE with a *state-independent* generator,
the per-step evolution unitary is fixed for the duration of the run: no
per-step Hamiltonian rebuild, no per-step lstsq fit, no classical RHS
computation inside the time loop. This is what makes it pure-quantum.

Initial `φ` amplitudes are loaded onto qubits via the Ran 2020
MPS-to-circuit state-prep pipeline (the MPS is used only for encoding,
not for evolution). A single propagator is implemented: the
**diagonal-in-eigenbasis** heat propagator, realised as QFT for periodic
BC and DCT-II for Dirichlet-on-`u` (§4.2.2). Shots-mode readout is
described in §4.2.3 and segmented evolution in §4.2.4. Sign recovery is
not needed because `φ > 0` by construction. Source forcing is **not
supported** on this pathway — a source `source_fn` raises
`NotImplementedError` (the position-diagonal potential does not share an
eigenbasis with the Laplacian).

**What residual classical work remains (and where).** The "pure-quantum"
label is meant honestly: nothing in the time loop reaches for a
classical PDE state. The classical work that does exist is at the
encode/decode boundary and one-time setup:

1. **Forward Cole–Hopf transform** (`lib_cole_hopf.cole_hopf_forward_centered`)
   — one-time at `t=0`: compute `φ₀ = exp(−(2ν)⁻¹∫u₀)` in log-space
   with mean-centring for numerical stability.
2. **MPS state-prep** (`lib_mps.classical_to_mps` →
   `mps_to_circuit`, Ran 2020) — one-time per circuit build: convert
   classical `φ₀` amplitudes into a state-prep subcircuit. In segmented
   mode this prep is repeated once per segment (§4.2.5).
3. **Propagator-coefficient build** — one-time per `(ν, δt, q)` choice:
   the damping factors `d_k = exp(ν·λ_k·δt)` and rotation angles
   `θ_k = arccos(d_k)` (`compute_theta_exact` for periodic,
   `compute_theta_dct` for Neumann-on-φ), plus their Möbius
   (inclusion–exclusion) coefficients for the conditional-Ry.
4. **Post-shot amplitude reconstruction** (`post_select_counts` →
   `reconstruct_phi_from_counts`) — at readout: discard shots where
   the ancilla flagged failure, then convert kept counts to
   amplitudes `φ̂ₖ = √(cntₖ/n_kept) · √(P_success) · ‖φ‖` (§4.2.3).
5. **Inverse Cole–Hopf transform** — at readout: `u = −2ν · ∂ₓ ln φ`
   via central differences in log-space.

None of the above is inside the per-step quantum time loop. That is what
makes the "pure-quantum" label honest for this pathway.

#### 4.2.1 Block-encoding of the heat propagator

The heat propagator `P = exp(ν·L·δt)` is a contractive operator (all
eigenvalues in `(0, 1]`) that must be embedded into a unitary circuit.
The block-encoding strategy is: an ancilla qubit rotated by
`arccos(d_k)` conditioned on the eigenstate index `k`, so that
post-selecting the ancilla in `|0⟩` implements the contraction

```
|k⟩|0⟩  →  |k⟩ (d_k|0⟩ + √(1−d_k²)|1⟩)                                  (13)
```

Post-selecting `ancilla = |0⟩` yields the state `Σ_k d_k ψ_k |k⟩` with
probability `P_success = Σ_k d_k² |ψ_k|²`. The success probability
degrades as `ν·δt` grows (stronger damping), requiring more shots in the
stochastic regime.

#### 4.2.2 The `qft-diagonal` propagator (periodic; Dirichlet-on-u via DCT-II)

This is the only heat propagator implemented in this repo
(`lib_cole_hopf_circuit.py`). The discrete periodic Laplacian is
diagonalized by the DFT. Its eigenvalues are

```
λ_j = −(4/δx²) sin²(π·j/N),   j = 0, …, N−1                             (14)
```

The damping factors are `d_j = exp(ν·λ_jᵖ·δt)`, and the rotation angles
are `θ_j = arccos(d_j)`. The per-step circuit is:

1. QFT on `q` data qubits (to Fourier eigenbasis).
2. Conditional-Ry(`2θ_k`) on ancilla, controlled by the `q`-bit index `k`.
3. QFT⁻¹ on `q` data qubits (back to computational basis).
4. Measure and reset ancilla.

The conditional rotation uses a Möbius (inclusion–exclusion) expansion:
`θ(k) = Σ_S a[S] ∏_{i ∈ S} b_i` where `k = Σ 2^i b_i`. Each non-zero
`a[S]` yields one (multi-)controlled `Ry` gate. Gate count: `O(2^q)` for
the conditional rotation, `O(q²)` for the QFT. CX counts measured at
`q = 3 / 4 / 5`: 12 / 60 / 240.

**Dirichlet-on-`u`.** Under `--bc dirichlet`, the φ-side BC is Neumann
(`∂φ/∂x = 0`), whose Laplacian eigenbasis is the DCT-II rather than
the DFT. The propagator dispatcher transparently swaps the QFT for the
DCT-II and uses the Neumann eigenvalues `λᵏⁿ = −(4/δx²)sin²(πk/(2N))`
(see `heat_dct_full_circuit` / `compute_theta_dct` in
`lib_cole_hopf_circuit.py`); the conditional-Ry / ancilla-rotation
structure is unchanged. The `qft-diagonal` label therefore covers both
BCs, just with different transform pairs (QFT/QFT⁻¹ for periodic,
DCT-II/DCT-II⁻¹ for Dirichlet-on-`u`). Internally `--bc dirichlet` is
remapped to `phi_bc="neumann"` before circuit build; any BC other than
periodic or Neumann-on-φ raises. Source forcing is not supported (the
source potential `V` is position-diagonal and does not share an
eigenbasis with the Laplacian), so `source_fn` must be `None`.

**Removed variants.** Earlier iterations of this codebase carried two
further propagators — a `dense-block` (full `N×N` eigendecomposition +
`UnitaryGate` block-encoding, exact per step, source-forcing-capable)
and an `lcu` (Taylor/Fourier–Bessel Linear-Combination-of-Unitaries with
`O(M·q)` gate scaling). **Neither is present in this repo**; there is no
`--propagator` flag, and `cole_hopf_circuit` is hardwired to the
qft-diagonal/DCT propagator. The paragraphs below on the hidden
eigendecomposition cost, `s_max` normalization, and LCU truncation are
retained only as historical description of those removed variants — none
of it runs here.

**(Historical) `dense-block`.** Built the full `N × N` matrix
`P = exp(ν·L·δt)`, eigendecomposed it (`P = V D V†`), and block-encoded
via `V†` → conditional-Ry(`2 arccos(d_k / s_max)`) → `V`. Exact per
step, `O(4^q)` gates from the dense `UnitaryGate`, and the only variant
that supported source-term forcing via a Strang-split potential `V(x,
t)`. Its `np.linalg.eigh` on the `N×N` propagator ran **once** in the
unforced case (reused for every timestep) and per step in the forced
case (`O(N³·n_steps)`).

**(Historical) `lcu`.** Achieved `O(M·q)` gate scaling via a Taylor
(periodic) or Fourier–Bessel / Jacobi–Anger (Neumann)
Linear-Combination-of-Unitaries with a PREPARE/SELECT/PREPARE† ancilla
sandwich, and supported source forcing through a Strang-split potential
`V(x, t)` with `V_x = g/(2ν)`. **Again: neither ships in this repo.** The
sole propagator here is qft-diagonal/DCT, unforced.

#### 4.2.3 Shots-mode readout and post-selection

In statevector mode the data-register amplitudes are read directly
after applying the evolution circuit, and the ancilla post-selection is
accounted for analytically.

In shots mode (`--shots N > 0`), the full circuit (including ancilla
measurements) is sampled `N` times. Reconstruction proceeds in two
stages (`post_select_counts` → `reconstruct_phi_from_counts`):

1. **Post-selection** (`post_select_counts`). Keep only shots in which
   every ancilla bit measured `|0⟩` (block-encoding success). Let
   `n_kept = Σ_{ancilla=0} cnt` and let `data_counts` be the marginal
   over the data register. Define `P_success ≈ n_kept / N`. Shots with
   any ancilla `|1⟩` are *dropped* — they correspond to the
   complement of the contractive embedding and carry no usable data.
2. **Amplitude reconstruction** (`reconstruct_phi_from_counts`). For
   each data bitstring `k`,
   `φ̂_k = √(data_counts[k] / n_kept) · √(P_success) · ‖φ‖_t`.
   The `√(P_success)` factor restores the contraction that
   post-selection removes; `‖φ‖_t` is the classical norm tracked
   alongside the unitary evolution.

`P_success` degrades as `ν·δt·t` grows (stronger cumulative damping →
fewer ancilla-`|0⟩` shots), so long evolutions in the
shot-dominated regime require either a larger `--shots` budget or the
segmented-evolution mode of §4.2.4.

**Experimental Hadamard-per-bin readout.** An alternative interferometric
readout (`--readout hadamard_per_bin`) that estimates each bin's signed
amplitude via a Hadamard test is implemented in `hadamard_per_bin_circuit`
/ `extract_hadamard_per_bin_amplitudes` / `_run_shots_hadamard_per_bin`
(`lib_cole_hopf_circuit.py`). It is not the default path (`--readout
direct` is), is incompatible with `measure_reprepare`, and raises on
hardware. Intended for the deep-shot regime at small `ν` where standard
post-selection becomes statistics-starved; not validated for production
yet.

#### 4.2.4 Measure-and-reprepare (segmented) evolution mode

For long runs in shots mode, `--evolution-mode measure_reprepare` splits the
`n_steps` total into `K = n_steps / segment_size` segments. Each segment is
a self-contained circuit (`build_segment_circuit`) comprising state-prep
+ `segment_size` propagator layers + measurement. Between segments
(`_run_shots_measure_reprepare` in `lib_cole_hopf_circuit.py`):

1. **Decode**: post-select and reconstruct `φ̂` from the segment's shots
   (§4.2.3).
2. **Re-encode**: run `classical_to_mps(φ̂)` → `mps_to_circuit(...)` to
   build a fresh state-prep subcircuit for the next segment's starting
   amplitudes. The MPS bond dim follows `--bond-dim`.
3. **Run next segment** with the fresh prep.

This trades cumulative post-selection survival (which falls
exponentially in `segment_size`) against information loss from
classical decode/re-encode at each segment boundary (limited by the MPS
truncation). The classical norm and `P_success` factors compose
multiplicatively across segments
(`cumulative_norm *= segment_norm; cumulative_p_success *= segment_p_success`),
so the reconstructed `φ̂` at any snapshot still represents physical
amplitudes. No classical PDE physics enters the loop — only amplitude
IO at the segment boundaries. Segmented mode defaults to `--backend-type
sim`; running it on real hardware goes through the standalone F12
driver `burgers_ch_hw_runner.py`, which sets the opt-in `allow_hardware`
kwarg and wraps the serial segments in one held IBM Runtime `Session`
(§8.6).

#### 4.2.5 Complexity summary

Per-step cost of the implemented `qft-diagonal`/DCT propagator (data
qubits `q`, `N = 2^q`):

| Propagator | Gate count | Ancillas | Classical setup | Source forcing |
|---|---|---|---|---|
| `qft-diagonal` (periodic/Dirichlet-on-u) | `O(2^q)` cond-Ry + `O(q²)` (Q/D)FT | 1 | `O(N)` per `(ν, δt)` for `θ(k)` | not supported |

The removed `dense-block` (`O(4^q)`, source-capable) and `lcu`
(`O(M·q)`, source-capable) variants are described historically in
§4.2.2 but do not run in this repo.

Per-step `P_success` lower-bounds (smallest contraction across the
spectrum): `min_k d_k² = exp(2·ν·λ_min·δt)`, with `λ_min ≈ −4/δx²`
on a periodic grid. Worst-case survival to step `n` is therefore
`exp(2·ν·λ_min·δt·n)`; the spectrum-weighted average tracked by the
code via `P_success = Σ_k d_k² |ψ_k|²` is typically much larger
because mass concentrates on the smooth low-`k` modes that are
weakly damped.

### 4.3 Cole–Hopf analytic IC and reference (`--ic cole_hopf_exact`)

For unforced Burgers on `[0, L_box]` with `--bc dirichlet`, picking
`φ₀(x)` as a finite Neumann cosine sum yields a closed-form analytic
`u(x, t)`. Each cosine mode evolves independently under the heat
equation (eigenfunctions of the Neumann Laplacian), so coefficients
just decay:

```
φ₀(x)    = a₀ + Σ_{n=1..M} aₙ · cos(nπx/L_box)                          (22)
φ(x,t)   = a₀ + Σₙ aₙ · cos(nπx/L_box) · exp(−ν(nπ/L_box)² t)            (23)
u(x,t)   = −2ν · φ_x(x,t) / φ(x,t)                                       (24)
```

**Positivity guard** (`validate_cole_hopf_coeffs` in
`lib_cole_hopf.py`): `u = −2ν ∂ₓ ln φ` is well-defined only when
`φ > 0` everywhere. Sufficient condition: `a₀ > Σ |aₙ|` for `n ≥ 1`.
Since modes only decay, satisfying this at `t = 0` carries to all
`t > 0`. Violation raises `ValueError`.

**Constraints.** This IC family is restricted to:

- `--bc dirichlet` (`u(0) = u(L_box) = 0`). The cosine basis matches
  Neumann-on-φ, which is the Cole–Hopf dual of Dirichlet-on-u.
- `--source none`. A source couples the cosine modes (via the
  Cole–Hopf potential `V`) and breaks the independent-mode decay.

**Usage.** When `--method` is `cole_hopf_circuit`, `--ic` defaults to
`cole_hopf_exact` and the analytic `u(x, t)` is emitted as the reference
trajectory. For other methods, `--ic cole_hopf_exact` can be selected
explicitly (the analytic formula is method-agnostic — it's just the
exact PDE solution for this IC). Pass coefficients as a comma-list:
`--ic-cole-hopf-coeffs "1.0,0.3"` (default).

**Validation case.** Single mode `a = (1.0, 0.3)`, `ν = 1e-2`, `q = 5`,
`n_steps = 50`. Gives a tanh-bump `u₀(x)` profile that decays
monotonically. Method accuracy claims against this reference are
**independent of any classical co-solver** — the headline
"validated against another approximation" reviewer objection
disappears for this test family.

**Suppressing the analytic reference.** `--no-analytic-reference`
falls back to FTCS even with `--ic cole_hopf_exact`, useful
for cross-checking the analytic formula itself against the classical
solver during V&V. `--no-classical-reference` skips the reference
trajectory entirely (no error metrics, no `speedup_ratio`).

---

## 5. QLBM (kinetic) family

D1Q3 lattice Boltzmann: evolve three mesoscopic distributions
`f_i(x, t)` on lattice velocities `c = (−1, 0, +1)` by a BGK collision
plus streaming step, and recover the macroscopic velocity by moments:

```
ρ = Σ_i f_i,    ρ u = Σ_i c_i f_i                                        (21)
```

with athermal-Burgers weights `w_i = 1/3`. The LBM-native timestep is
`δt_lbm = δx` (one lattice site per step); the caller's `δτ · n_steps`
window is remapped to an integer number of lattice steps. Reference:
Quartey & Zhong, RPI/IBM 2025 (the F11 source paper).

### 5.1 `lbm` (classical D1Q3)

Pure-classical D1Q3 BGK solver. Collision relaxes `f` toward the
equilibrium `f^eq(u)` with rate `1/τ`; streaming shifts each `f_i` by
its velocity `c_i`. Periodic or bounce-back (Dirichlet) BC. `O(N)` per
step. The classical kinetic baseline / reference for `qlbm_circuit`.
(Previously named `qlbm`; the "q" was misleading since this method
ignores `--shots` entirely and has no quantum content.  Module:
`lib_lbm.py`; entry function `run_lbm_simulation`; wrapped by the
`LBMIntegrator` in `lib_fw.py`.)

### 5.2 `qlbm_circuit` (pure-quantum QALB, Itani-style)

The realised pure-quantum QLBM, following Itani et
al. (*Phys. Fluids* 36, 2024; arXiv:2304.05915). Module
`lib_qalb_circuit.py`; integrator `QALBIntegrator`; function
`run_qalb_simulation`. The collision is a **fixed, state-independent
unitary** with **no classical `collide_bgk` mirror** in the loop.

**Value encoding — two paths.** Each density `δf_i = f_i − f_eq⁰`
(rest equilibrium `f_eq⁰ = (0,1,0)`) is stored as the *value* in its
own `qc`-qubit register. There are two encodings, chosen by mode, not by
a flag:
- **shots = 0 (statevector) — App C finite-position embedding.** A value
  `x ∈ [−1,1]` is `|x⟩ = Σ_n P_n(x)|n⟩`, where `P_n` are the monic
  polynomials orthogonal to the truncated-Gaussian functional (Itani
  Eqs. C1/C4). The finite-position operator `q̂_C` (Eq. C39) satisfies
  `q̂_C|x⟩ = x|x⟩` *exactly* and the linear readout `⟨1|x⟩/⟨0|x⟩ = x` is
  exact (Eq. C20). This is what makes the operator path
  statevector-faithful; the earlier App B physicists' eigenstates only
  approximately diagonalise, so they are **not** used for encoding.
- **shots > 0 — App B bosonic encoding.** A vacuum displaced in the
  position quadrature, `|δf_i⟩ = e^{−i δf_i p̂}|0⟩` with
  `q̂ = (â+â†)/√2`, `p̂ = i(â†−â)/√2`, `[q̂,p̂]=iI`. Readout is the
  normalised expectation `⟨q̂_i⟩` from marginal counts.

**Hermitised collision (Itani Eq. 60–63, 83, 85; App B).** The D1Q3 BGK
collision
functional `Ω(q̂)` (this repo's density-conserving equilibrium) is
written as the flow generator and split into a Hermitian part
`Ĥ′ = ½ Σ_i (p̂_i Ω_i(q̂) + Ω_i(q̂) p̂_i)`. `e^{−iΔt Ĥ′}` is **exactly
unitary** — applied per site, built **once**, no post-selection. The
anti-Hermitian part is the *constant* divergence `−2/τ·I` (Eq. 83),
recovered as a deterministic scalar deflation that cancels in the
`⟨q̂⟩` ratio readout. **The quadratic collision term is normal-ordered**
(`s² → s²−I`, subtracting the vacuum variance) — this is what makes the
truncated Hermitised collision reproduce the classical flow; it
converges in `qc` (single-cell error 0.073 → 0.027 → 0.0043 at
qc = 2,3,4).

**Architecture.** Per lattice step: apply the per-site collision
unitary to each site, then **exact streaming** (a permutation). Shots
mode prepares `|δf_i⟩`, applies `W = e^{−iΔt Ĥ′}`, rotates each register
into the `q̂` eigenbasis, measures, and estimates `δf_i′ = ⟨q̂_i⟩` from
the marginal counts (`cell_collision_shots`). Routes through the shared
`q8020_cfd_qutil.circuit` helpers, so `--shots`, `--seed`,
`--optimization-level`, `--backend-type` behave as elsewhere.
`shots = 0` runs an operator/statevector-faithful collision instead.

**Knobs / regime.**
- `--fock-qubits` (qc, default 3). **qc = 2 is too coarse** — it
  under-dissipates and the amplitude grows; qc = 3 tracks the
  reference. qc = 4 is more faithful but costly.
- Valid only for `τ > 1` (`Δt/τ < 1`); `τ ≤ 1` is Itani's divergent
  regime (App A: no time-independent error bound) and emits a warning.
- It is a **flow-LBM**: the state-independent collision realises the
  *continuous* BGK flow, not the discrete Euler step, so it differs
  from FTCS by an O(Ω²)/step **scheme gap** (~0.11 final error) that
  does *not* shrink with qc — distinct from the Fock-truncation error
  that does.

**Status / honesty.** k = 1 measure-reprepare (measure every lattice
step, reconstruct `f` classically, re-prepare); streaming is classical
(imported `stream` from `lib_lbm`; log-depth quantum streaming is
still open). The default collision is a **dense `UnitaryGate`**
(Quantum-Shannon synthesis, ~`4^{3qc}` depth), so it runs on a simulator
but is heavy. A **hardware-honest Trotter synthesis** of the Pauli `Ĥ′`
is implemented and selectable via `--qalb-collision-trotter-reps > 0`
(with `--trotter-order 1|2`) — a single position-free unitary on `3·qc`
qubits, no ancilla, exactly unitary (no post-selection); order-2 Trotter
error `∝ 1/reps²`. Coherent `k > 1` segments and quantum streaming
remain open. The module's `__main__` validation gates cover encoding,
collision convergence, the exact-unitary property, and
shots-vs-statevector agreement (see also `tests/test_qalb_circuit.py`).

---

## 6. Comparative analysis

### 6.1 Gate complexity per step

| Pathway | Propagator | CX per step (`q=5`) | Asymptotic |
|---|---|---|---|
| Cole–Hopf (`cole_hopf_circuit`) | `qft-diagonal` | ~240 | `O(2^q + q²)` |
| QLBM (`qlbm_circuit`, QALB) | per-site `e^{−iΔtĤ′}` | dense `UnitaryGate` ~`4^{3qc}` depth/site (qc=3) × N sites; Trotter path `∝ reps` | LCU synthesis pending |

The QALB per-site dense collision dominates the gate budget; the
`--qalb-collision-trotter-reps` path replaces it with a Suzuki–Trotter
circuit whose depth scales with the rep count instead of `4^{3qc}`.

### 6.2 Accuracy characteristics

- **Cole–Hopf** — `qft-diagonal`/DCT uses exact rotation angles via the
  Möbius expansion (no series truncation); the only quantum-side error is
  the block-encoding post-selection cost. The Cole–Hopf transform itself
  contributes
  discretization error from the trapezoidal quadrature (Eq. 10) and the
  FD log-derivative (Eq. 12), both `O(δx²)`.
- **QLBM** — accuracy carries the standard `O(δx²)` LBM discretization
  error, plus two circuit-specific terms: a **Fock-truncation error**
  that shrinks with `qc` (0.073 → 0.027 → 0.0043 at qc = 2,3,4 for a
  single cell), and a **flow-vs-Euler scheme gap** (~0.11 final error)
  that does *not* shrink with `qc` because the state-independent
  collision realises the continuous BGK flow rather than the discrete
  Euler step. The Hermitised collision is exactly unitary (no
  post-selection, no amplitude-rescaling error).

### 6.3 Boundary-condition support

| Pathway / variant | Periodic | Dirichlet |
|---|---|---|
| Cole–Hopf, `qft-diagonal` (QFT periodic / DCT-II Dirichlet-on-u) | ✓ | ✓ |
| QLBM (`lbm`, `qlbm_circuit`) | ✓ | ✓ (bounce-back) |

---

## 7. Per-method CLI options

Most flags only apply to a subset of methods. The CLI will raise an
explicit "not supported" error for invalid combinations.

### 7.1 Propagator / encoding — no flags

There is **no `--propagator` flag and no `--encoding` flag** in this
repo. `cole_hopf_circuit` is hardwired to the qft-diagonal (periodic)
/ DCT-II (Dirichlet-on-u) propagator with binary index encoding; the
QALB encoding is fixed (App C on the statevector path, App B on the
shots path) and tuned only via `--fock-qubits` (§7.2).

### 7.2 QALB collision knobs (`--fock-qubits`, `--qalb-collision-trotter-reps`, `--trotter-order`)

Used by `qlbm_circuit` (QALB). `--fock-qubits` (qc, default 3) is the
number of bosonic Fock qubits per density register; qc = 2 is too coarse
for a full run, qc = 3 converges. `--qalb-collision-trotter-reps` selects
the collision synthesis: `0` (default) = dense `UnitaryGate` of
`e^{−iΔtĤ′}`; `> 0` = Suzuki/Lie–Trotter of the Pauli `Ĥ′` at that rep
count (hardware-honest depth), with `--trotter-order {1,2}` picking the
splitting order.

### 7.3 MPS bond dimension (`--bond-dim`)

Used by the state-prep stage of `cole_hopf_circuit`. `--bond-dim None`
keeps full rank; a finite value truncates the singular-value spectrum
during the MPS compression of the amplitudes to be loaded.

### 7.4 Sign recovery (`--sign-recovery`)

Choices `{none, classical_oracle, hadamard_test, dual_rail}`, default
`none`. It is a no-op for `cole_hopf_circuit` (`φ > 0` by construction,
so amplitudes are non-negative) and is **rejected** by `qlbm_circuit`
(its `⟨q̂⟩` readout is already signed) — a non-`none` value there raises
`ValueError`. Retained as a CLI knob for magnitude-readout experiments;
the postproc rolls up its cost metrics generically.

### 7.5 Readout (`--readout {direct,hadamard_per_bin}`)

`cole_hopf_circuit` shots only. `direct` (default) = post-selected
amplitude estimation (§4.2.3). `hadamard_per_bin` = signed `Re(ψ_k)` via
a per-bin Hadamard test; incompatible with `measure_reprepare` and
raises on hardware.

### 7.6 Fourier low-pass (`--phi-modes`)

`cole_hopf_circuit` shots only. Keep only the lowest `N` Fourier modes of
the reconstructed `φ` before the inverse Cole–Hopf transform, to suppress
shot noise. `0` (default) = no filter.

### 7.7 Shots and backend

`--shots N` (0 means statevector). `--backend-type {sim,fake,hardware}`,
`--backend NAME`, `--t1`, `--t2`, `--coupling-map`, `--seed`,
`--optimization-level`. `--metric-transpile-timeout` caps the isolated
basis-transpile used *only* to report depth/gate counts (never affects
results; `0` = uncapped).

### 7.8 Evolution mode (`--evolution-mode {single,measure_reprepare}`, `--segment-size`, `--auto-cadence`)

`cole_hopf_circuit` shots-mode only. `single` = one big circuit with
`n_steps` inlined step layers (today's default). `measure_reprepare`
(segmented) = break the evolution into `K`-step segments, read out and
re-prep amplitudes between segments. Trades depth-per-circuit against
shot-noise compounding. `--auto-cadence` auto-picks `--segment-size` (the
nearest divisor of `n_steps`) and `--save-every` so segments stay aligned
at any `q`.

### 7.9 Source forcing (`--source {sine,none}`)

`sine` gives `g(x,t) = sin(2πx)·cos(2πt)`; `none` disables it. The
classical methods (`shift`, `ftcs_reference`, `lbm`) and the QALB accept
it; **`cole_hopf_circuit` does not support forcing and raises** if a
non-`none` source is passed (§4.2.2). Cole–Hopf analytic-IC runs force
`--source none` (§4.3).

### 7.10 Time-window (`--shock-pct`, `--n-steps`)

Either a percentage of the inviscid shock-formation time
`t_shock = 1 / max|du₀/dx|` (resolves to an `n_steps` from the fixed
CFL-derived `dt`), or an explicit step count.

### 7.11 Initial condition (`--ic`, `--ic-*`)

`--ic {sine,multimode,gaussian,cole_hopf_exact}`.  Default is method-
dependent: `cole_hopf_exact` when `--method` is `cole_hopf_circuit`,
`sine` otherwise.  IC-specific knobs:

- `--ic-amplitude` (all ICs).  Scales `u₀` after construction;
  required `< 1.0` for QLBM D1Q3 stability.
- `--ic-modes`, `--ic-seed`, `--ic-alpha` (multimode only).
- `--ic-center`, `--ic-sigma` (gaussian only).  Pulse centre and
  width.  Pick `σ` small enough that `u(boundary)` is negligible
  under `--bc dirichlet` or the boundary discontinuity radiates
  spurious shocks.
- `--ic-cole-hopf-coeffs "a0,a1,..."` (cole_hopf_exact only).
  Comma-separated Neumann cosine-sum coefficients; default `"1.0,0.3"`.
  Must satisfy `a₀ > Σ|aₙ|` for n ≥ 1 (positivity of φ); the solver
  raises `ValueError` otherwise.  Forces `--bc dirichlet` and
  `--source none`.  See §4.3 for the math and the analytic-reference
  pairing.

Passing an IC-specific flag with the wrong `--ic` emits a warning and
ignores the flag.

### 7.12 Reference trajectory (`--no-classical-reference`, `--no-analytic-reference`)

By default the solver runs a reference trajectory alongside the
chosen method and reports L² error against it.  The reference is
chosen automatically: closed-form analytic when `--ic cole_hopf_exact`
(microsecond cost), otherwise a classical FTCS run (`solve_burgers` on
the q-grid, or `solve_burgers_subsampled` on a `--ref-points`-refined
grid when running `--method ftcs_reference`; default `--ref-points 0`
= no refinement).  Two flags suppress:

- `--no-analytic-reference` — under `--ic cole_hopf_exact`, fall
  back to FTCS instead of the closed-form.  Useful for
  cross-validating the analytic formula itself against the classical
  solver during V&V.  No effect for other ICs.
- `--no-classical-reference` — skip the reference trajectory
  entirely.  The analysis fragment emits null `classical_wall_time_s`,
  null `speedup_ratio`, null `u_final_classical`, and NaN error
  metrics.  Saves wall time on long sweeps where errors are computed
  offline from the raw amplitudes.

The two flags compose: `--no-classical-reference` takes precedence
and suppresses both paths.

---

## 8. Framework integration (`solverfw`)

The package is wired to the framework in
[`lib_fw.py`](../src/lib_fw.py).

### 8.1 Components

- **Config**: `BurgersConfig(SolverConfig)` adds every CLI parameter
  in §7 as a dataclass field (including the §7.11 IC-specific knobs
  and the §7.12 reference-suppression flags). `describe()` returns a
  serializable summary; the case-fragment writer records the relevant
  IC knobs conditional on `--ic` so fragments stay clean for unrelated
  ICs.
- **Grid**: `Grid1D.from_qubits(q, bc=...)`. Interior depends on BC:
  Dirichlet includes both endpoints (so `u = 0` is satisfied by the
  sine IC); periodic excludes the right endpoint since `x = 0` and
  `x = 1` are identified.
- **State**: `DenseState`. Burgers' state is a 1-D float array of
  length `N` — the protocol's default implementation is sufficient.
- **SpatialOperator**: `ShiftFD` — central-difference RHS using shift
  operators on `u`. Used only by the per-step family below; delegating
  methods build their own spatial structures internally.
- **TimeIntegrator**: a different concrete subclass per method (tables
  in §8.2, §8.3).
- **MainLoop**: standard, unchanged from the framework.

### 8.2 Per-step integrators (use the framework loop)

The `shift` baseline plugs into `MainLoop` the normal way — `step()`
advances one timestep, the framework owns the loop:

| `--method` | Integrator class | Step function it calls |
|---|---|---|
| `shift` | `ShiftEulerIntegrator` | `lib_fd.shift_euler_step` |

The integrator pulls the source value at time `t` from
`config._source_fn(grid.xc, t)` and forwards it to the underlying step
function along with `bc`, `nu`, `dt`. Returns
`(DenseState(u_new), metrics_dict)`.

### 8.3 Delegating integrators (own their own loop)

The remaining methods carry circuit / kinetic state across timesteps,
or pre-build a propagator once and reuse it. They do not fit the
per-step model. They use the delegating-integrator idiom from
[SPEC-solverfw.md](../../q8020-cfd-metautil/docs/SPEC-solverfw.md) §5:
subclass `TimeIntegrator`, but in `step()` run the *entire* multi-step
simulation internally and return all snapshots via sentinel keys in the
metrics dict.

| `--method` | Integrator class | Inner driver |
|---|---|---|
| `cole_hopf_circuit` | `ColeHopfCircuitIntegrator` | `lib_cole_hopf_circuit.run_cole_hopf_circuit_simulation` |
| `lbm` | `LBMIntegrator` | `lib_lbm.run_lbm_simulation` |
| `qlbm_circuit` | `QALBIntegrator` | `lib_qalb_circuit.run_qalb_simulation` |

The set of delegating methods is recorded as
`_DELEGATING_METHODS` in `lib_fw.py` (`cole_hopf_circuit`, `lbm`,
`qlbm_circuit`). `make_integrator` handles `shift` plus these three;
`ftcs_reference` is **not** in the integrator registry — it is dispatched
directly in `burgers_solver.py` (via the local `make_reference_grid` /
`solve_burgers_subsampled`) rather than through `run_simulation_fw`.

### 8.4 Dispatcher

`run_simulation_fw(config, grid, u0, source_fn)` attaches `source_fn` to
`config._source_fn`, builds the right integrator via
`make_integrator(config)`, and either:

- **Delegating method**: calls `integrator.step()` once with full
  `n_steps`, pulls `_delegated_solutions` and `_delegated_metrics` out
  of the returned metrics dict, and returns those.
- **Per-step method**: hands the integrator to `MainLoop().run()` and
  returns its output.

In either case the public return shape is
`(solutions, step_metrics, genuine_steps)` — identical to the framework
contract, identical across all methods. `genuine_steps` is the set of
caller-step indices actually computed (coarser for the LBM family, whose
lattice step is `δx`), or `None` when every caller step is genuine.

### 8.5 Backend management

The quantum-circuit methods build their backend lazily.
`cole_hopf_circuit` builds its backend inside the integrator's
`_run_all` because it may not be needed (`shots=0` path is
statevector-only). `qlbm_circuit` consumes the backend in the shots
path: every step's circuit is transpiled via
`q8020_cfd_qutil.circuit.transpile_circuit` and executed via
`execute_circuit_counts` (the same shared helpers `cole_hopf_circuit`'s
shots paths use), with counts reconstructed back to `f` (see §5.2).  In
statevector mode the backend is unused and the step is simulated via
`Statevector.evolve`.

The shared `qutil` execution layer means **all real-circuit shots
paths** (`cole_hopf_circuit` batched or segmented, `qlbm_circuit`
per-step) honour the same flag contract: `--shots`, `--seed`,
`--optimization-level`, and `--backend-type {sim,hardware}` produce
comparable behaviour across methods. `execute_circuit_counts`
transparently dispatches to `backend.run` for Aer and to `SamplerV2`
for IBM-runtime backends, so methods transition from simulator to
hardware without per-method code changes.

### 8.6 Standalone tooling (outside the framework loop)

`burgers_ch_hw_runner.py` (F12) sits alongside the solver but is not a
`--method` — a standalone driver that runs the Cole–Hopf
*measure-reprepare* segment loop on sim, a fake backend, or real IBM
hardware, with TREX measurement mitigation and dynamical decoupling. It
reuses `run_cole_hopf_circuit_simulation` unchanged; the only solver-side
hook is the opt-in `allow_hardware` kwarg. The segments are intrinsically
serial (segment `k+1` is built from segment `k`'s measured counts), so on
hardware they are wrapped in one held `Session`. It carries its own
`CASES` dict (`shock`, `smooth`, `smooth_stunt`) and writes per-segment
job-id / quantum-seconds audit metadata. Sim-testable end-to-end with no
IBM credentials.

---

## 9. Implementation map

```
q8020-cfd-ch-lbm/
├── postproc/                        # q8020 postproc plotters (§10.1)
│   ├── plot_circuit_resources_bakeoff.py
│   ├── plot_cost_scaling.py
│   ├── plot_dataset_movie.py
│   ├── plot_method_compare.py       # FTCS vs Cole–Hopf vs LBM animation
│   ├── plot_regime_crossover.py
│   ├── plot_regime_pair.py
│   ├── plot_seg_depth_tradeoff.py
│   └── Z-Keep/                      # retained older plotters
├── tests/                           # test_cole_hopf_circuit, test_qalb_circuit,
│                                    # test_shots_backend
└── src/
    ├── burgers_solver.py            # CLI entry point; ftcs_reference
    │                                # dispatch + reference-trajectory
    │                                # selection (analytic / FTCS / skipped)
    │                                # per --ic and the two --no-*-reference
    │                                # flags
    ├── lib_fw.py                    # solverfw bindings (§8): BurgersConfig,
    │                                # ShiftFD, integrators, make_integrator
    ├── lib_classical.py             # ICs (sine, multimode, gaussian) +
    │                                # FTCS solve_burgers + source_term_sine
    │                                # + resolved reference (make_reference_grid,
    │                                # solve_burgers_subsampled)
    ├── lib_fd.py                    # shift-operator FD: compute_rhs_shift,
    │                                # shift_euler_step, shift_matrix,
    │                                # compute_error
    ├── lib_mps.py                   # Ran 2020 MPS state-prep helpers
    │                                # (used by cole_hopf_circuit)
    ├── lib_cole_hopf.py             # Forward/inverse CH transforms
    │                                # (Eqs. 10, 12); log-domain stability;
    │                                # cole_hopf_exact analytic IC + ref
    │                                # (§4.3, Eqs. 22–24)
    ├── lib_cole_hopf_circuit.py     # CH quantum-circuit pipeline;
    │                                # qft-diagonal + DCT-II propagator;
    │                                # Möbius conditional-Ry; shots readout;
    │                                # measure-reprepare; hadamard-per-bin
    ├── lib_lbm.py                   # classical D1Q3 LBM (F11)
    ├── lib_qalb_circuit.py          # pure-quantum QALB (Itani-style):
    │                                # App C operator path + App B shots path
    ├── burgers_ch_hw_runner.py      # F12 standalone Cole–Hopf hardware
    │                                # runner (measure-reprepare on IBM QPU)
    └── lib_postprocess.py           # output writers, q8020 metrics dump
```

Everything the doc's earlier revisions attributed to `burgers_nonlinear.py`,
`burgers_trotter.py`, `burgers_mpo.py`, `burgers_tebd.py`, `burgers_lcu.py`,
`burgers_potential.py`, `burgers_encoding.py`, or `burgers_sign_recovery.py`
has been removed or folded elsewhere — **none of those modules exist in this
repo.** The shift-FD kernels now live in `lib_fd.py`.

Method-to-module crosswalk for "where does the actual physics live":

| `--method` | Module |
|---|---|
| `shift` | `lib_fd.py::shift_euler_step` (+ `compute_rhs_shift`) |
| `ftcs_reference` | `burgers_solver.py::solve_burgers_subsampled` (also in `lib_classical.py`) |
| `cole_hopf_circuit` | `lib_cole_hopf_circuit.py::run_cole_hopf_circuit_simulation` |
| `lbm` | `lib_lbm.py::run_lbm_simulation` |
| `qlbm_circuit` | `lib_qalb_circuit.py::run_qalb_simulation` |

---

## 10. Running cases

### 10.1 Sweep harness (TOML)

The q8020 sweeper (see project repo q8020-cfd-metautil) drives this
solver from TOML case files, each key a `burgers_solver.py` flag. A
representative Cole–Hopf circuit case (unforced, shots, on a simulator):

```toml
[cole_hopf_circuit_q5_smoke]
"--method"        = "cole_hopf_circuit"
"--ic"            = "cole_hopf_exact"
"--bc"            = "dirichlet"
"--source"        = "none"
"--nu"            = 0.01
"--cfl"           = 0.1
"--n-steps"       = 50
"--q"             = 5
"--shots"         = 50000
"--backend-type"  = "sim"
"--seed"          = 42
"--save-every"    = 1
_group_postproc = "python ./postproc/plot_method_compare.py"
```

The sweeper converts this to a CLI invocation of `burgers_solver.py`
and runs it; the postproc receives the resulting JSON dump (built by
`lib_postprocess.py`) and renders the comparison plot. (Note there
is no `--propagator` or `--source`-forcing option for
`cole_hopf_circuit`; see §7.1 and §7.9.)

### 10.2 CLI quick reference

Classical `shift` baseline (periodic):

```sh
python burgers_solver.py --q 5 --method shift \
  --bc periodic --n-steps 10 --nu 1e-2 --noshow
```

FTCS reference trajectory as the method itself:

```sh
python burgers_solver.py --q 5 --method ftcs_reference \
  --bc periodic --n-steps 10 --nu 1e-2 --noshow
```

Classical D1Q3 LBM baseline:

```sh
python burgers_solver.py --q 5 --method lbm \
  --bc periodic --n-steps 10 --nu 1e-2 --ic-amplitude 0.5 --noshow
```

Pure-quantum QALB (Itani-style):

```sh
python burgers_solver.py --q 5 --method qlbm_circuit \
  --bc periodic --n-steps 10 --nu 1e-2 --fock-qubits 3 \
  --ic-amplitude 0.5 --noshow
```

Headline pathway (pure-quantum Cole–Hopf, periodic, statevector):

```sh
python burgers_solver.py --q 5 --method cole_hopf_circuit \
  --bc periodic --n-steps 10 --nu 1e-2 --source none --noshow
```

Cole–Hopf, Dirichlet (DCT-II propagator), statevector:

```sh
python burgers_solver.py --q 5 --method cole_hopf_circuit \
  --bc dirichlet --n-steps 10 --nu 1e-2 --source none --noshow
```

Cole–Hopf, Dirichlet, with shots:

```sh
python burgers_solver.py --q 5 --method cole_hopf_circuit \
  --bc dirichlet --n-steps 10 --nu 1e-2 --source none \
  --shots 10000 --noshow
```

Cole–Hopf against the closed-form analytic reference (`--ic` defaults
to `cole_hopf_exact` for the CH method; reference is exact, no FTCS):

```sh
python burgers_solver.py --q 5 --method cole_hopf_circuit \
  --bc dirichlet --n-steps 50 --nu 1e-2 --source none \
  --ic-cole-hopf-coeffs "1.0,0.3" --noshow
```

Same case but cross-checking the analytic vs FTCS as the reference:

```sh
python burgers_solver.py --q 5 --method cole_hopf_circuit \
  --bc dirichlet --n-steps 50 --nu 1e-2 --source none \
  --ic-cole-hopf-coeffs "1.0,0.3" --no-analytic-reference --noshow
```

Cole–Hopf, Dirichlet, segmented (measure-reprepare) shots run:

```sh
python burgers_solver.py --q 5 --method cole_hopf_circuit \
  --bc dirichlet --n-steps 50 --nu 1e-2 --source none \
  --shots 20000 --evolution-mode measure_reprepare --auto-cadence --noshow
```

---

## 11. What this solver does NOT do

- **No 2-D / 3-D.** Strictly 1-D. The framework is general enough for
  higher dimensions; the application is not.
- **No adaptive `dt`.** Fixed `dt = cfl · dx` (set via `--cfl`).
- **No mesh refinement.**
- **No real hardware execution by default.** Runs default to
  `--backend-type sim`. Real IBM-hardware Cole–Hopf runs go through the
  standalone F12 driver `burgers_ch_hw_runner.py` (§8.6), not the plain
  `--method` path.
- **No physics beyond viscous Burgers + source.** No reaction term, no
  compressibility coupling, no multi-component flow.

---

## 12. References and further reading

### Primary publications

- Meena, M. Gopalakrishnan et al., "A Tensor Network–based Quantum
  Algorithm for the Nonlinear 1-D Burgers' Equation," AIAA 2026 — the
  source of the shift-operator central-difference discretization
  (Eqs. 3–4), the Ran-2020 MPS state-prep, and the general
  quantum-Burgers program.
- Quartey, B. & Zhong, X., "Beyond the Simulator: A Practical
  Demonstration of Quantum Lattice Boltzmann Methods on IBM Quantum,"
  RPI / IBM, Nov 2025 — the source paper for the QLBM family.
- Itani, W., Sreenivasan, K. R. & Succi, S., "Quantum Algorithm for
  Lattice Boltzmann (QALB) Simulation of Incompressible Fluids with a
  Nonlinear Collision Term," *Phys. Fluids* **36**, 017112 (2024);
  arXiv:2304.05915 — the source for the pure-quantum QALB
  (`qlbm_circuit`, §5.2); App B (bosonic encoding, Hermitised collision)
  and App C (finite-position embedding) are both used.
- Cole, J. D., "On a quasi-linear parabolic equation occurring in
  aerodynamics," *Quart. Appl. Math.* **9**, 225 (1951); Hopf, E., "The
  partial differential equation u_t + u u_x = ν u_xx," *Commun. Pure
  Appl. Math.* **3**, 201 (1950) — the Cole–Hopf substitution
  underlying the headline pathway.
- Ran, S.-J., "Encoding of matrix product states into quantum circuits
  of one- and two-qubit gates," *Phys. Rev. A* **101**, 032310 (2020) —
  the MPS state-prep used by `cole_hopf_circuit`'s initial-state
  loading.
