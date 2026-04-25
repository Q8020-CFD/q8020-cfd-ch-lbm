# F10 — Quantum-Circuit Cole-Hopf Burgers Solver (Comprehensive Spec)

**Status**: authoritative. This document is the single source of truth for completing
F10. It supersedes any prior F10 / F2-Phase-B drafts in this folder.

**Purpose**: implement a genuinely quantum-native solver for 1D viscous Burgers,
aligned in spirit with Meena/Murali AIAA-2026 (amplitude encoding, MPS prep, circuit
evolution, shot-based readout), while eliminating the paper's classical co-solver
side-channel. The linearization that makes this possible is the Cole-Hopf transform.

## 0. Read-this-first constraints

1. **No classical co-solver inside a quantum timestep.** The paper's rotation
   Hamiltonian `A = i(|δ⟩⟨ψ| − |ψ⟩⟨δ|)` requires a classical `u_next = u + dt·rhs(u)`
   predictor at every step. That is rejected here. The only classical step is the
   one-shot Cole-Hopf transform at `t=0` and its inverse at `t=T` — not per-step
   steering.
2. **Amplitude (binary) encoding is fixed.** N = 2^q grid points on q qubits.
   Changing encoding is explicit **future work** (see §13) and is out of scope.
3. **Qubit-chain locality is not physical-grid locality.** Under binary encoding,
   physical-grid-nearest-neighbor operators (shift, Laplacian) are not
   qubit-chain-NN. This is acknowledged, not solved. Gate cost per timestep is
   accepted as `poly(q)` with non-NN two-qubit gates; Zaletel W-II / ladder-MPO is
   **not** used.
4. **Murali-the-person expects TEBD flavor.** That is honored by
   *per-timestep Trotterized circuit cadence*: the circuit has N_steps layers,
   each applying one heat-propagator increment `exp(ν·L·Δt)`, exactly as a TEBD
   loop would. The efficient-by-basis-choice variant (QFT-diagonal) is retained
   as an alternative; the spec mandates both are runnable so the author can choose.
5. **Phi is positive; sign recovery is a classical post-step.** `φ > 0` by
   construction of the Cole-Hopf transform. The quantum amplitudes therefore
   carry no sign ambiguity. Sign of `u(x, T)` is recovered by the classical
   inverse transform after readout (u can be negative; the sign lives in the
   derivative of log φ, not in the quantum state).
6. **Target platform**: Aer statevector simulator with optional shots and
   depolarizing noise. No hardware backend in scope.
7. **CFD case matches Murali**: 1D Burgers, sine IC, ν = 1e-4, Dirichlet or
   periodic BC, T up to ≈ 1.5 · t_shock where t_shock = 1/(2π) ≈ 0.159.
   Small-ν handling (§9) is required to make this physical ν runnable.

## 1. Current state of the codebase

- `burgers_cole_hopf.py`: classical pipeline exists (forward/inverse CH,
  dense Laplacian, `build_heat_propagator`, MPO variant). `method=cole_hopf`
  runs end-to-end on statevector-via-dense and MPS-via-MPO. Reference only.
- `burgers_mps.py`: MPS state prep (Ran 2020 → quimb MPS → qubit register
  amplitudes) exists and is used by `method=mps`. Reuse unchanged.
- `burgers_solver.py`: CLI dispatcher. Takes `--method {shift, quantum_exact,
  quantum_circuit, mps, tebd, tebd_circuit, cole_hopf}`. Will gain
  `cole_hopf_circuit` here.
- `burgers_trotter.py`: `run_cole_hopf_simulation` (classical) and the existing
  `quantum_circuit` / `mps` circuit machinery live here. The new method is
  added here.
- Everything under `method=tebd_circuit` (F2 Phase B.2) is **not a dependency**
  of F10. It remains broken in the repo but is frozen; F10 does not build on it.

## 2. What F10 delivers (Phase B)

A new CLI method `--method cole_hopf_circuit` that, given `(q, ν, T, bc, shots,
trotter_steps, propagator_variant)`:

1. Classically computes `φ₀(x) = exp(−(1/2ν) ∫₀ˣ u₀ ds)` from the user-chosen IC.
2. Normalizes and amplitude-encodes `|φ₀⟩` on q qubits via the Ran 2020
   MPS-to-circuit pipeline in `burgers_mps.py` (`classical_to_mps` +
   `mps_to_circuit`, right-canonical). `--bond-dim` controls the MPS
   truncation and is threaded through both the shots and the statevector
   paths; `None` means full rank. Faithful to Murali/Meena AIAA-2026
   Eq. 5-6 + Ref [27].
3. Executes a Qiskit `QuantumCircuit` on `q + n_anc` qubits that iteratively
   applies the non-unitary heat propagator `exp(ν · L · Δt)` in `N_steps`
   Trotter-cadence layers, reaching evolution time `T = N_steps · Δt`.
4. Reads out amplitudes via statevector (noiseless reference) **and** shots
   with post-selection on the ancilla register, producing `|⟨x_i|φ(T)⟩|`.
5. Classically inverts Cole-Hopf → `u(x, T)`. Writes NPZ + PNG per existing
   sweep/animation conventions.

Two propagator variants must both exist and be selectable at the CLI:

- `--propagator qft-diagonal` (default, recommended). One QFT + one diagonal
  phase-damping layer + one QFT⁻¹ per Trotter step. O(q²) two-qubit gates per
  step. Non-unitary via one ancilla per step (shared + reset; see §5).
- `--propagator dense-block`. Exact eigendecomposition of exp(ν·L·dt),
  block-encoded via V^H + conditional Ry + V with ancilla post-selection.
  No Trotter error (exact per step). Handles any BC including Dirichlet
  (via Neumann on φ). Tractable for q ≤ 5 (O(2^q) gates per step).
  **Note**: a true Pauli-Trotter variant (SparsePauliOp decomposition +
  commuting-group LCU) is deferred to future work (§13).

## 3. Math recap

Cole-Hopf:

    u(x, t) = −2ν · ∂_x ln φ(x, t)         (inverse)
    φ(x, 0) = exp(−(1/2ν) ∫₀ˣ u(s, 0) ds)  (forward)
    ∂_t φ = ν · ∂_xx φ                     (heat equation for φ)

Discrete Laplacian `L ∈ ℝ^{N×N}` (periodic or Neumann for Dirichlet-in-u; see
§8). `L` is real-symmetric, negative-semidefinite. The heat propagator

    P(Δt) = exp(ν · L · Δt)

has eigenvalues in `(0, 1]`, i.e. **contractive, non-unitary**. In the Fourier
basis (periodic BC) `L` is diagonal with eigenvalues `−(2π k / L_box)²` for
`k = 0, ±1, ..., ±N/2`, so

    P(Δt)|_Fourier = diag(exp(−ν · (2π k / L_box)² · Δt))

This is the identity the `qft-diagonal` variant exploits.

## 4. Architecture decision summary

| Concern | Decision |
|---|---|
| Encoding | Amplitude (binary) on q qubits. Future work to change. |
| State prep | MPS → Ran 2020 → `mps_to_circuit` from `burgers_mps.py` (right-canonical). `--bond-dim` threads through `classical_to_mps` truncation in both shots and SV paths. |
| Evolution cadence | `N_steps` TEBD-style Trotter layers, each applying `P(Δt)`. |
| Propagator per step (default) | QFT → diag(exp(−ν · (2π k / L_box)² · Δt)) via ancilla conditional rotation → QFT⁻¹. |
| Propagator per step (alt) | Dense-block: exact eigendecomposition + block-encoding (q ≤ 5). |
| Non-unitarity | Single ancilla qubit implementing `|0⟩ → cos θ(k)|0⟩ + sin θ(k)|1⟩`. Post-select `|0…0⟩` over ancilla history at end. |
| Ancilla reuse | One ancilla shared across all N_steps with reset between steps for `qft-diagonal` (and measurement-captured per-step post-select bits). |
| Readout | Statevector (reference) or shots with post-select on all ancilla history = `|0…0⟩`. `√counts/total` → `φ(x, T)`. |
| Normalization | Quantum state stays unit-norm; contraction lives in the post-select success probability. Rescale by `√(P_success)` and known `‖φ₀‖` after readout. |
| Sign of u | Recovered classically in `cole_hopf_inverse` (derivative of log φ). Zero quantum-side sign work. |
| Small-ν handling | Centered-exponent trick (§9). |

**Explicitly rejected alternatives**: W-II / ladder-MPO (needs NN qubit-chain
locality — category error under binary encoding); state-dependent rotation-H
(requires classical predictor — the whole reason we are in F10); QSVT polynomial
approximation of `P(Δt)` (implementation effort disproportionate to benefit
at small q); Carleman linearization (bigger scope; separate feature).

## 5. Circuit construction — default `qft-diagonal` variant

Per Trotter layer `ℓ = 1..N_steps`:

1. **QFT** on the q-qubit data register → basis of momentum states `|k⟩`.
2. **Conditional ancilla rotation**: for each basis state `|k⟩` on data, rotate
   one ancilla `|0⟩_anc → cos θ(k)|0⟩_anc + sin θ(k)|1⟩_anc` where

       θ(k) = arccos(exp(−ν · λ(k) · Δt))

   and `λ(k) = (2π k / L_box)²` is the Laplacian eigenvalue magnitude (with
   Qiskit QFT k-index convention — map computational basis integer back to the
   signed wavenumber; see §5.1).
3. **QFT⁻¹** on the q-qubit data register.
4. **Measure-and-record** ancilla, then **reset** it to `|0⟩` for the next
   layer. The measurement outcome per layer is used downstream to post-select
   `|0⟩` on every layer — equivalent to one accumulated ancilla register.
   On the last layer, do not reset; measure and post-select.

Post-select ancilla history = all-zeros (statevector: project onto
`|0⟩_anc^{⊗N_steps}` subspace; shots: filter by the classical-bit history).
Post-select success probability `P_success = ‖φ(T)‖² / ‖φ₀‖²` in normalized
encoding.

### 5.1 Conditional rotation implementation

θ is diagonal in the momentum basis, so the conditional rotation is a single-
ancilla operation controlled by the q data qubits' computational-basis value.
Realization:

- **Exact Möbius route** (current implementation). Compute exact `θ(k)` for all
  `2^q` values, apply the Möbius (inclusion-exclusion) transform to obtain
  multilinear coefficients, then emit one (multi-)controlled-Ry per nonzero
  coefficient. Gate count is O(2^q) — honest about the cost at the q values
  we run. No polynomial fitting; no accuracy loss.
- **Direct QROM/multi-controlled route** (for q ≥ 7 if needed). Load θ(k) into
  ancilla via a lookup circuit; out of scope for this spec beyond a TODO.

### 5.2 Ancilla accounting

- `qft-diagonal`: **1 ancilla** reused across all N_steps via measure-reset.
  Quantum width = `q + 1`. Classical bits = `N_steps + q`. Depth grows as
  `N_steps · (QFT + Ry-ladder + QFT⁻¹)`.
- `dense-block`: **1 ancilla** reused across all N_steps via measure-reset
  (same as qft-diagonal). Width `q + 1`. Depth `N_steps · (V_dag + Ry-ladder + V)`.

## 6. Circuit construction — alternative `dense-block` variant

Build the full propagator `P = exp(ν · L · dt)` classically via
`scipy.linalg.expm`, eigendecompose `P = V · D · V^H`, and block-encode:

1. `V^H` on data register (to eigenbasis).
2. For each eigenvalue `d_k`, controlled-Ry(2·arccos(d_k/s_max)) on ancilla,
   controlled by `|k⟩` eigenbasis state.
3. `V` on data register (back to computational basis).
4. Measure-reset ancilla.

This is exact per step (no Trotter error). Handles any BC including Dirichlet
(via Neumann on φ). Gate count is O(2^q) per step from the controlled-Ry ladder.

**Future work (§13)**: a true Pauli-Trotter variant using `SparsePauliOp`
decomposition and commuting-group LCU would have first-order Trotter error
`O(Δt²)` per step, testable via convergence plots. This was originally §6
but was deferred due to implementation complexity.

## 7. Readout and rescaling

1. Run circuit. Measure all q data qubits in computational basis at end.
   Ancilla history is already recorded classically (per §5).
2. Keep only shots where the full ancilla history is all-`|0⟩`. Let
   `N_kept / N_total = P_success`.
3. For each data bitstring `x_i`: `|φ̂(x_i, T)|² = counts[x_i] / N_kept`.
4. `φ̂(x_i, T) = +√(counts[x_i] / N_kept) · √(P_success) · ‖φ₀‖` where `‖φ₀‖`
   is the classical prep-time norm (tracked externally). Positive root because
   `φ > 0` by Cole-Hopf.
5. Classical inverse Cole-Hopf → `u(x_i, T)`.

Statevector mode: same math, with `counts[x_i]` replaced by
`|⟨x_i, 0_anc^{⊗N_steps}|ψ⟩|²` and `P_success = ⟨ψ|Π_{anc_history=0}|ψ⟩`.

## 8. Boundary conditions

- `--bc periodic`: Fourier diagonalization is exact; `qft-diagonal` is
  preferred. `L_box = 2π` (repo convention).
- `--bc dirichlet`: represented on φ as Neumann (zero-flux) because
  `u = −2ν ∂_x ln φ` with `u = 0` at boundaries implies `∂_x φ = 0`. Use
  `build_laplacian_dense(..., bc="neumann")` (already exists). QFT is not the
  right basis (DST would be); for this spec, `dirichlet` runs default to
  `dense-block` variant. `qft-diagonal + dirichlet` raises
  `NotImplementedError` with a clear message pointing to §13 future work.

## 9. Small-ν handling (centered exponent)

At ν = 1e-4 the classical CH forward transform overflows `float64` because
`exp(−∫u/(2ν))` swings through `exp(±10³)` in a single field. Repo currently
clamps ν ≥ 1e-2 in `input/burgers_quantum.toml` for this reason. F10 must
lift that.

**Mitigation**: compute `e(x) = −∫u/(2ν)` in log-domain, shift by
`e_mid = 0.5 · (max(e) + min(e))`, and carry `φ̃ = exp(e − e_mid)` as the
quantum-encoded amplitudes. `e_mid` is a classical scalar that rides the
simulation out-of-band and is re-applied at inverse-CH time. The heat
propagator commutes with scalar multiplication so the centering does not
alter the physics, only the numeric range.

Acceptance: `cole_hopf_circuit` runs end-to-end at ν = 1e-4 for q ∈ {3,4,5}.

## 10. CLI surface

New method in `burgers_solver.py`:

    --method cole_hopf_circuit
    --propagator {qft-diagonal, dense-block}       (default: qft-diagonal)
    --trotter-steps INT                            (default: N, one step per dt)
    --shots INT                                    (existing; 0 = statevector)
    --bc {periodic, dirichlet}                     (existing)
    --bond-dim INT                                 (existing; None = full rank)

Keeps `--method cole_hopf` (classical reference) unchanged. Do not repurpose
flags. The method dispatcher in `burgers_trotter.py` gains a new branch
`method == "cole_hopf_circuit"` → `run_cole_hopf_circuit_simulation(...)`.

## 11. Acceptance criteria

Each item is pass/fail. Implementer must produce the named artifact for the
code-reviewer session.

### 11.1 Classical reference reproduction (no regressions)

`--method cole_hopf` (existing) continues to match `--method shift` within
2% L2 at `q=5, ν=1e-2, T=0.5·t_shock`. Artifact: `test_cole_hopf_classical.py`
pytest PASS.

### 11.2 Statevector correctness — `qft-diagonal`

For `(q=4, ν=1e-2, bc=periodic, T=0.05, N_steps=10)`:
`‖φ_circuit − φ_dense‖₂ / ‖φ_dense‖₂ < 1e-6` where `φ_dense` is the classical
`build_heat_propagator` result. Artifact: NPZ `cole_hopf_qft_q4_verify.npz`
+ plot.

### 11.3 Statevector correctness — `dense-block`

Same case as 11.2: `‖φ_circuit − φ_dense‖₂ / ‖φ_dense‖₂ < 1e-6` (exact
eigendecomposition, no Trotter error). Artifact: pytest PASS in
`test_cole_hopf_circuit.py::test_11_3_dense_block_statevector`.

### 11.4 Trotter-error convergence

Not applicable for current propagators: `qft-diagonal` uses exact Möbius
rotation angles and `dense-block` uses exact eigendecomposition. Both have
zero Trotter error per step. A future Pauli-Trotter variant (§13) would
restore this acceptance item.

### 11.5 Shots correctness

At `shots=150k` and the 11.2 configuration:
`‖u_circuit − u_dense‖₂ / ‖u_dense‖₂ < 0.05`. Post-select retention
`P_success > 0.3` (if lower, flag in logs; do not silently proceed).
Artifact: `cole_hopf_shots_q4.png` overlaying circuit vs dense.

### 11.6 Small-ν endurance

`(q ∈ {3,4,5}, ν=1e-4, bc=dirichlet, T=0.8 · t_shock, shots=150k)` runs
without overflow and produces a visually recognizable Burgers profile with
the forming shock. Artifact: `paper_cole_hopf_circuit_q{3,4,5}_shots150k.png`.

### 11.7 Sweep + animation

`[cole_hopf_circuit_q5_shots150k]` group in `burgers_quantum.toml` works with
`q8020-sweep`. An existing animation script (`animate_tebd_comparison.py` or
a new `animate_cole_hopf_comparison.py` — implementer's call) shows the
classical `cole_hopf` curve alongside `cole_hopf_circuit` over all saved
frames. Artifact: MP4.

### 11.8 Noiseless → noisy degradation

Run the 11.6 q=5 case through an Aer `depolarizing_error(p=1e-3)` on all
two-qubit gates. Solution degrades but remains recognizable at T = 0.5 ·
t_shock. Artifact: `noise_sensitivity.png`.

## 12. Parcels for dole-out

Dependencies: `A → B` means B starts after A is merged. Parcels with the same
letter prefix are independent of each other and may run in parallel.

- **P1 — CH classical hardening + small-ν centering** (blocks P5). File:
  `burgers_cole_hopf.py`. Add `cole_hopf_forward_centered(u, dx, nu)
  -> (phi_tilde, e_mid)` and matching `cole_hopf_inverse_centered(phi_tilde,
  e_mid, dx, nu) -> u`. Update the classical `run_cole_hopf_simulation` to
  use the centered pair when `|e_max − e_min| > 50` or ν < 1e-3. Unit tests:
  round-trip `u → φ̃ → u` exact to 1e-10 at ν ∈ {1e-2, 1e-3, 1e-4}. Delivers:
  precondition for acceptance 11.6.

- **P2 — Ancilla-conditional Möbius-Ry primitive** (blocks P3). New file
  `burgers_cole_hopf_circuit.py`. Implement `build_conditional_ry(data_qubits,
  ancilla, theta_exact) -> QuantumCircuit` using exact Möbius transform of
  `2·θ(k)` into multilinear coefficients, then one (multi-)controlled-Ry per
  nonzero coefficient. Test harness verifies statevector action matches
  `cos(θ(k))|0⟩ + sin(θ(k))|1⟩` within 1e-10 for q ≤ 6. O(2^q) gates.

- **P3 — `qft-diagonal` propagator per Trotter step** (depends on P2). Same
  file. Add `heat_qft_step_circuit(q, nu, dt, L_box, bc) -> QuantumCircuit`
  composed of QFT + polynomial-Ry(θ) + QFT⁻¹ + measure-and-reset ancilla.
  Add `heat_qft_full_circuit(q, nu, T, N_steps, L_box, bc) -> QuantumCircuit`
  stacking N_steps layers. Acceptance item 11.2. Statevector path only.

- **P4 — `dense-block` propagator** (independent of P3; parallelizable).
  Same file. `heat_dense_block_step_circuit(q, nu, dt, L_box, bc) -> QuantumCircuit`
  and `heat_dense_block_full_circuit(...)`. Exact eigendecomposition of
  `exp(ν·L·dt)`, block-encoded via V^H + conditional Ry + V. Acceptance
  item 11.3.

- **P5 — CLI wiring + `run_cole_hopf_circuit_simulation`** (depends on P1,
  P3, P4). In `burgers_trotter.py`, add `method == "cole_hopf_circuit"`
  branch and the run function. Dispatches to the chosen propagator variant,
  runs on Aer statevector, writes NPZ consistent with existing
  `quantum_circuit` NPZ schema. `burgers_solver.py`: add the two new flags
  and pass through. No shots in this parcel.

- **P6 — Shots + post-selection + noise** (depends on P5). In the run
  function, add shots path: assemble `QuantumCircuit` with measurement,
  execute on Aer with `shots`, filter by ancilla-history-all-zero, compute
  `P_success`, rescale. Emit warning if `P_success < 0.3`. Plumbing for
  `depolarizing_error(p)` via existing Aer noise-model hooks. Acceptance
  items 11.5 + 11.8.

- **P7 — Neumann / Dirichlet adaptation** (can start after P3 lands). For
  `--bc dirichlet`, use `dense-block` variant with
  `build_laplacian_dense(..., bc="neumann")`. For `qft-diagonal + dirichlet`,
  raise `NotImplementedError("DST-based propagator is future work; see §13")`.

- **P8 — TOML groups, sweep integration, animation** (depends on P5). Add
  `[cole_hopf_circuit_q{3,4,5,6}]` and `[cole_hopf_circuit_q5_shots150k]`
  groups to `input/burgers_quantum.toml`. Add `_group_postproc` that
  generates the comparison animation. Acceptance item 11.7.

- **P9 — Tests + acceptance artifacts** (depends on all). In
  `analysis/test_cole_hopf_circuit.py`. Pytest PASS for all acceptance items
  expressible as asserts (11.1–11.5, 11.6 smoke). Plots for the rest.

**Recommended sequencing**: P1 + P2 immediately in parallel. Then P3 + P4 +
P7 in parallel (three agents). Then P5 (single agent, serializing point).
Then P6 + P8 in parallel. P9 last.

## 13. Explicit future work (out of scope)

- **Encoding change.** Swap binary amplitude encoding for one that makes
  physical-grid operators qubit-chain-local, so true TEBD with NN two-qubit
  gates (Zaletel W-II) becomes the right tool. Candidates: one-hot (N qubits
  — trivially local but exponentially wider), block-encoded LCU with an
  explicit shift register, or a tree/interleaved encoding. A 1D chain of q
  qubits cannot be both log-wide *and* physical-grid-NN for shift operators;
  this is an encoding-design project of its own.
- **Direct u-space evolution.** Keeps amplitude encoding but drops Cole-Hopf.
  Requires Carleman linearization or equivalent embedding. Scope comparable
  to F10 itself; separate feature.
- **DST-based `qft-diagonal + dirichlet`.** Replace QFT with a Discrete Sine
  Transform circuit so the Dirichlet-on-u / Neumann-on-φ case diagonalizes
  with logarithmic ancilla rotations.
- **Hardware execution.** Depth of `qft-diagonal` at q=5, N_steps=50 is
  already heavy for NISQ. Hardware run is a separate effort with its own
  error-mitigation story.
- **QSVT-polynomial alternative** to the ancilla-Ry construction. Tighter
  asymptotics; unnecessary at q ≤ 6.
- **F11 Burgulence.** Multi-mode stochastic IC ensemble. Depends on F10
  landing with stable small-ν behavior.

## 14. Validation checklist for the code-reviewer session

A separate code-reviewer session at the end should confirm:

- [ ] No branch of `cole_hopf_circuit` reads `u` state inside a timestep.
- [ ] `φ̂` from readout is positive everywhere before inverse-CH.
- [ ] Post-selection `P_success` is logged and surfaced in the NPZ.
- [ ] Classical `cole_hopf` pipeline is unchanged by this work (no regressions
      in existing sweep groups).
- [ ] `method=tebd_circuit` is not touched and is not a dependency.
- [ ] `--bc dirichlet` either works via `dense-block` or fails with a
      clear `NotImplementedError` in `qft-diagonal`; it does not silently
      produce wrong output.
- [ ] Small-ν centering (§9) is on by default when applicable and off-path
      when not, with a log line explaining the choice.
- [ ] Prepared ψ matches `reconstruct_from_mps(classical_to_mps(ψ₀,
      canonical="right"))` to 1e-12 at full rank (MPS prep is wired, not
      bypassed via `QuantumCircuit.initialize`).
- [ ] `--bond-dim` visibly truncates in both shots and SV paths (bd=1
      produces a different prepared ψ than full rank).
- [ ] No agent has introduced a new `.claude/` dir in the repo (Best-Practices
      Rule 23).
- [ ] PEP 8 + 88-char line width + venv invocation (Best-Practices Rules 3, 5, 7).

## 15. Reference

- Meena, Murali et al., "Quantum algorithm for nonlinear Burgers …", AIAA
  SciTech 2026. Paper-source for IC, ν, BC, CFL, shock-time convention, and
  shots count. F10 deliberately **diverges** from the paper on evolved
  variable (φ not u) and on the evolution operator (heat propagator not
  rotation-H); all other choices track the paper.
- Cole (1951); Hopf (1950). The transform.
- Ran, Phys. Rev. A 101 (2020). MPS → circuit state preparation used at prep.
- Liu et al., PNAS 2023; Childs, Liu, Ostrander, arXiv:2011.06571. Quantum
  algorithms for linearized nonlinear PDEs — literature defense for using
  Cole-Hopf as the linearization route.
