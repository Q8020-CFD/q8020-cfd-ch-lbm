# F11 — Quantum Lattice Boltzmann Method for 1-D Burgers

**Status:** Proposed (spec only).
**Depends on:** None (independent of F2/F10; reuses framework plumbing).
**Reference:** Quartey & Zhong, "Beyond the Simulator: A Practical
Demonstration of Quantum Lattice Boltzmann Methods on IBM Quantum,"
RPI / IBM, Nov 2025.

---

## 0. Motivation

We currently have two pure-quantum pathways for viscous Burgers:

| Pathway | Encodes | Nonlinearity handling | Scaling bottleneck |
|---|---|---|---|
| Pauli-Trotter (`quantum_circuit`) | u(x) directly | Pauli decomposition per step | O(8^q) overlap matrix |
| Cole-Hopf LCU (`cole_hopf_circuit`) | φ(x) via transform | Eliminated (linear heat eq.) | O(M·q) LCU gates |

QLBM adds a third, fundamentally different approach: encode the
**mesoscopic distribution functions** f_i(x,t) of the lattice Boltzmann
equation. The nonlinearity enters through the equilibrium distribution
in the collision step (a local, diagonal operation), while streaming is
a conditional shift — both are efficient as quantum circuits.

**Why this matters:**
1. No Pauli decomposition, no Cole-Hopf transform — direct kinetic encoding.
2. Collision is O(N) diagonal gates; streaming is O(q) controlled increments.
3. Natural extension to 2-D/3-D (just add velocity directions).
4. Three-way comparison (Trotter vs Cole-Hopf vs QLBM) on identical
   IC/BC/grid is a strong narrative for the Frontier proposal.
5. QLBM has been demonstrated on real IBM hardware (Quartey & Zhong),
   giving us a hardware-readiness reference point.

---

## 1. Classical Lattice Boltzmann for 1-D Burgers

### 1.1 Stencil: D1Q3

For 1-D viscous Burgers, the minimal stencil is D1Q3 — three discrete
velocities on a 1-D lattice:

```
c = (-1, 0, +1) · δx/δt
```

At each lattice site j, there are three distribution functions:
f₋₁(j), f₀(j), f₊₁(j). The macroscopic fields are:

```
ρ(j) = f₋₁ + f₀ + f₊₁           (density, conserved)
ρu(j) = -f₋₁ + f₊₁              (momentum)
```

For Burgers (constant density ρ=1), the equilibrium distributions are:

```
f₋₁ᵉᵍ = (1/3)(1 - u + u²)       (Eq. 1)
f₀ᵉᵍ  = (1/3)(1 - u²)           (Eq. 2)  [note: can also use 2/3 weight]
f₊₁ᵉᵍ = (1/3)(1 + u + u²)       (Eq. 3)
```

The weights (1/3, 1/3, 1/3) are specific to the athermal D1Q3 model
for Burgers. Standard D1Q3 for Navier-Stokes uses (1/6, 2/3, 1/6).
The choice depends on the target PDE — for Burgers we need u² in the
equilibrium to recover the u·∂u/∂x nonlinearity via Chapman-Enskog.

### 1.2 BGK collision + streaming

One LBM timestep consists of:

**Collision (local, per-site):**
```
f_i*(j) = f_i(j) - (1/τ)(f_i(j) - f_iᵉᵍ(j))      (Eq. 4)
```

where τ is the relaxation time, related to viscosity by:
```
ν = (τ - 1/2) · δx² / δt                             (Eq. 5)
```

**Streaming (non-local, shift):**
```
f_i(j + c_i, t + δt) = f_i*(j, t)                    (Eq. 6)
```

The collision step is purely local (each site j is independent).
The streaming step shifts f₋₁ left by one site, f₀ stays, f₊₁
right by one site.

### 1.3 Boundary conditions

**Periodic:** Shift wraps cyclically. Natural for the shift circuit.

**Dirichlet (u=0 at boundaries):** Bounce-back: at wall sites, the
outgoing distribution is reflected back as the incoming one:
```
f₊₁(0) = f₋₁(0)      (left wall)
f₋₁(N-1) = f₊₁(N-1)  (right wall)
```

This enforces zero velocity at walls — exact Dirichlet u=0.

---

## 2. Quantum encoding

### 2.1 Register layout

We use the **interleaved** encoding from Quartey & Zhong:

```
|i⟩|j⟩  =  |velocity_direction⟩ ⊗ |position⟩
```

- **Velocity register:** 2 qubits to index 3 directions (|00⟩=f₋₁,
  |01⟩=f₀, |10⟩=f₊₁; |11⟩ is unused/ancilla).
- **Position register:** q qubits for N = 2^q grid points.

**Total data qubits:** q + 2.

The quantum state encodes the distribution functions as amplitudes:
```
|ψ⟩ = (1/‖f‖) Σᵢ Σⱼ f_i(j) |i⟩|j⟩                  (Eq. 7)
```

where ‖f‖ = √(Σᵢⱼ |f_i(j)|²) is the normalization.

**Alternative encoding (compact):** Use a single (q+2)-qubit register
where the top 2 bits index velocity direction and bottom q bits index
position. Same qubit count, different gate topology.

### 2.2 State preparation

Initialize from the classical IC u(x):

1. Compute u(xⱼ) for j = 0, …, N-1.
2. Compute equilibrium distributions: f_i^eq(j) from Eqs. 1–3.
3. Flatten into a 3N-length vector (padding the 4th slot |11⟩ with 0).
4. Normalize and prepare via Qiskit `initialize()` or MPS state prep
   (Ran 2020 pipeline, already implemented for Cole-Hopf).

### 2.3 Measurement and reconstruction

After evolution, measure to get amplitudes (statevector) or
probabilities (shots). Reconstruct:

1. Extract f_i(j) from the 3N amplitudes (ignore |11⟩ slot).
2. Rescale by tracked norm ‖f‖.
3. Recover macroscopic velocity: u(j) = (f₊₁(j) - f₋₁(j)) / ρ(j).

**Sign recovery:** Not needed if f_i > 0 everywhere (which is
physically guaranteed for LBM — distributions are non-negative).
This is a major advantage over the Pauli-Trotter path.

---

## 3. Quantum circuit construction

### 3.1 Collision operator

The BGK collision (Eq. 4) is **diagonal in the position basis** — it
acts independently on each site j. For each site, it mixes the three
velocity components:

```
f_i*(j) = (1 - 1/τ) f_i(j) + (1/τ) f_iᵉᵍ(j)        (Eq. 8)
```

This is a 3×3 linear map M(u(j)) on the vector (f₋₁, f₀, f₊₁) at
each site. The matrix M depends on the local velocity u(j), making
the collision **state-dependent** (like Pauli-Trotter, unlike Cole-Hopf).

**Circuit implementation options:**

**Option A — Linearized collision (statevector only):**
Pre-compute M(u(j)) classically at each timestep. Build the full
3N × 3N block-diagonal matrix and apply as a UnitaryGate. Cost: O(4^q)
gates (dense unitary on q+2 qubits). Same scaling as dense-block.

**Option B — Block-encoded collision:**
Block-encode M(u(j)) via ancilla + controlled rotations. The collision
matrix is sparse (block-diagonal with 3×3 blocks), so the block
encoding can exploit this structure. Requires 1 ancilla qubit.
Post-select on |0⟩.

**Option C — Fixed-point collision (τ-independent):**
For specific τ values (e.g., τ=1 → over-relaxation; τ=0.5+ε →
near-incompressible), the collision simplifies. At τ=1:
f_i* = f_iᵉᵍ — just prepare the equilibrium directly. This eliminates
the collision circuit entirely but restricts the viscosity.

**Recommendation for v1:** Option A (linearized). Same philosophy as
Pauli-Trotter (classical pre-computation per step). Allows direct
comparison. Upgrade to Option B for hardware runs.

### 3.2 Streaming operator

Streaming shifts each velocity population by its lattice velocity:

```
f₋₁: shift left by 1   → S⁻ on position register, conditioned on |i⟩=|00⟩
f₀:  no shift           → identity, conditioned on |i⟩=|01⟩
f₊₁: shift right by 1  → S⁺ on position register, conditioned on |i⟩=|10⟩
```

**Circuit:** Two controlled-increment circuits on the position register,
each controlled by one configuration of the velocity register.

```
C-S⁻:  if velocity = |00⟩, apply decrement on position register
C-S⁺:  if velocity = |10⟩, apply increment on position register
```

Each controlled shift is a cascade of q controlled-NOT gates (ripple
increment/decrement). Gate count: O(q) CX per shift, O(q) total for
streaming.

**This is the key efficiency win:** streaming is O(q) regardless of N.
Compare to Pauli-Trotter's O(4^q · r) Trotter gates.

### 3.3 Full timestep circuit

One LBM timestep:

```
|ψₙ⟩ → [Collision] → [Streaming] → [Measure/Reset ancilla] → |ψₙ₊₁⟩
```

For multi-step evolution, iterate. If collision uses block encoding
(Option B), measure and reset the collision ancilla each step.

### 3.4 Gate complexity summary

| Component | Gate count | Notes |
|---|---|---|
| State prep | O(2^(q+2)) | Qiskit initialize, one-time |
| Collision (Option A) | O(4^(q+2)) | Dense unitary, per step |
| Collision (Option B) | O(3N) = O(3·2^q) | Block-diagonal encoding, per step |
| Streaming | O(q) | Two controlled increments, per step |
| Measurement | O(1) | Ancilla post-selection if block-encoded |

**Per-step total (Option A):** O(4^q) — same as dense-block Cole-Hopf.
**Per-step total (Option B):** O(2^q + q) — better than Pauli-Trotter.

---

## 4. Integration into mps-burgers

### 4.1 New files

| File | Role |
|---|---|
| `burgers_qlbm.py` | D1Q3 equilibrium, collision matrix, streaming, classical LBM solver |
| `burgers_qlbm_circuit.py` | Quantum circuit construction: collision + streaming gates |
| `tests/test_qlbm.py` | Unit tests: equilibrium recovery, mass conservation, viscosity |
| `tests/test_qlbm_circuit.py` | Circuit tests: SV match vs classical LBM |

### 4.2 CLI integration

```
python burgers_solver.py --q 5 --method qlbm --bc dirichlet \
    --nu 1e-2 --source none --ic sine --save-every 1 --noshow
```

And the circuit variant:

```
python burgers_solver.py --q 5 --method qlbm_circuit --bc dirichlet \
    --nu 1e-2 --source none --ic sine --shots 0 --save-every 1 --noshow
```

New CLI args:
- `--tau` (float): BGK relaxation time. Default: computed from ν via Eq. 5.
- `--collision-mode` (str): `dense` (Option A) or `block` (Option B).
  Default: `dense`.

### 4.3 Framework wiring

Add to `burgers_fw.py`:
- `QLBMIntegrator` class implementing `step()` interface.
- `QLBMCircuitIntegrator` class (quantum circuit version).

Both return u(x,t) at each saved step (not f_i — the framework expects
velocity fields). Internal state is (f₋₁, f₀, f₊₁) but output is
reconstructed u = (f₊₁ - f₋₁) / ρ.

### 4.4 TOML groups

```toml
# Classical LBM reference
[qlbm_classical_q5]
"--q" = 5
"--nu" = 1e-2
"--shock-pct" = 100.0
"--method" = "qlbm"
"--bc" = "dirichlet"
"--source" = "none"
"--save-every" = 1

# QLBM circuit (statevector)
[qlbm_circuit_q5]
"--q" = 5
"--nu" = 1e-2
"--shock-pct" = 100.0
"--method" = "qlbm_circuit"
"--collision-mode" = "dense"
"--bc" = "dirichlet"
"--source" = "none"
"--shots" = 0
"--save-every" = 1

# Three-way comparison
[pq_compare_q5_qlbm]
"--q" = 5
"--nu" = 1e-2
"--shock-pct" = 100.0
"--method" = "qlbm_circuit"
"--collision-mode" = "dense"
"--bc" = "dirichlet"
"--source" = "none"
"--shots" = 0
"--save-every" = 1
```

### 4.5 Comparison plot

Extend `plot_pq_compare.py` to detect `qlbm_circuit` as a third method.
Three-way overlay: Pauli-Trotter (red), Cole-Hopf LCU (purple),
QLBM (green), + classical FTCS (blue) + Godunov (black).

---

## 5. Implementation plan

### Phase 1: Classical LBM baseline (F11-1 through F11-4)

| ID | Task | Acceptance |
|---|---|---|
| F11-1 | `burgers_qlbm.py`: D1Q3 equilibrium, collision, streaming | Unit test: f_iᵉᵍ recovers ρ, ρu |
| F11-2 | Classical LBM time-stepper: `run_qlbm_simulation()` | Matches FTCS Burgers to O(δx²) for sine IC |
| F11-3 | Wire `--method qlbm` through `burgers_fw.py` | CLI runs, metadata emitted |
| F11-4 | Viscosity calibration: verify ν = (τ-½)δx²/δt | Measured decay rate matches target ν ±1% |

### Phase 2: Quantum circuit (F11-5 through F11-8)

| ID | Task | Acceptance |
|---|---|---|
| F11-5 | Collision circuit (Option A: dense unitary) | SV matches classical LBM collision to 1e-12 |
| F11-6 | Streaming circuit (controlled increment/decrement) | SV matches classical LBM streaming to 1e-14 |
| F11-7 | Full `qlbm_circuit` integrator + CLI wiring | `--method qlbm_circuit --shots 0` matches classical LBM |
| F11-8 | Resource metrics: CX count, depth, ancilla count | Table comparing to Trotter and Cole-Hopf at q=3..5 |

### Phase 3: Comparison and validation (F11-9 through F11-11)

| ID | Task | Acceptance |
|---|---|---|
| F11-9 | TOML groups for three-way comparison | All three methods run from single sweeper invocation |
| F11-10 | Three-way plot script | GIF with Trotter + Cole-Hopf + QLBM + classical |
| F11-11 | Accuracy/cost table in technical doc | Update quantum_pathways_technical_reference.docx |

### Phase 4: Hardware-ready (F11-12, F11-13) — deferred

| ID | Task | Acceptance |
|---|---|---|
| F11-12 | Block-encoded collision (Option B) | SV match + ancilla post-selection working |
| F11-13 | Shots path with noise model | Fidelity > 0.9 at q=3, noise_model=ibm_brisbane |

---

## 6. Key differences from Quartey & Zhong

Their poster targets general 2-D flow (lid-driven cavity, channel flow)
on IBM hardware. Our adaptation:

1. **1-D only** (Burgers, not Navier-Stokes) — simpler stencil, fewer qubits.
2. **Statevector first** — prove algorithmic correctness before hardware.
3. **Integrated comparison** — same IC/BC/grid as Pauli-Trotter and Cole-Hopf.
4. **MPS state prep** — leverage Ran 2020 pipeline already in codebase.
5. **No hardware runs initially** — defer to Phase 4 after SV validation.

Their circuit structure (collision as controlled rotations, streaming as
conditional shifts) maps directly to our D1Q3 simplification. The main
simplification is going from D2Q9 (their stencil) to D1Q3.

---

## 7. Open questions

1. **Equilibrium weights:** Standard D1Q3 for Burgers uses equal weights
   (1/3, 1/3, 1/3). Some formulations use (1/6, 2/3, 1/6). The choice
   affects the recovered PDE — need Chapman-Enskog analysis to confirm
   which weights recover viscous Burgers (not isothermal Navier-Stokes).

2. **Norm tracking:** LBM distributions are non-negative, so ‖f‖ is
   well-behaved. But the collision operator (Eq. 8) is not unitary —
   it's contractive when τ > 1/2. Block encoding (Option B) handles
   this via post-selection. Option A just applies the dense matrix.
   Need to track the contraction factor per step for amplitude rescaling.

3. **Stability at small ν:** Small ν → τ → 1/2 (Eq. 5), which makes
   the collision nearly singular. Same numerical challenge as Cole-Hopf
   at small ν. May need τ > 0.5 + ε guard.

4. **Source term:** Forcing g(x,t) can be added to LBM via a source
   term in the collision: f_i* += w_i · g · δt. Straightforward to
   wire if needed, but start unforced for the comparison.
