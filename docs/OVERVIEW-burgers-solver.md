# Quantum Methods for the 1-D Viscous Burgers Equation
## A technical reference for the `q8020-mps-burgers` solver

*May 2026 — LLM generated, human edited.*

---

## Preamble

The `q8020-mps-burgers` codebase grew up ad hoc, and as such probably needs
a haircut. We wired in many CLI switches to trigger classical and hybrid
pathways. In the end the two pathways we actually care about scientifically
are the **hybrid Pauli–Trotter pathway** (`quantum_circuit`, Meena
AIAA-2026 Appendix A.A) and the **pure-quantum Cole–Hopf pathway**
(`cole_hopf_circuit`, this work); the other classical and hybrid
variants are kept as validation references. A front-end TOML driver
(see project repo q8020-cfd-metautil) feeds this solver, and its data
and metadata are harvested into the q8020 *cases × codes × backends*
rollup.

The original goal was a curiosity: would it be possible to derive a quantum
solution from a published mathematical description — science publications
make good specifications, mathematics being a good specification language —
and to validate it against an existing classical solution, with AI tooling
providing "independent" review, plus human software and SME review.

We are also now interested, again in the *cases × codes × backends*
modality, in running the same case on more than one code implementing the
same Burgers equation, e.g. QLBM.

In the solver we do an MPS encoding. There is variation around what
"nearest neighbor" means for a given encoding, but that is workable. We
then decompose into Pauli strings — this is 4^q and does not scale well,
though it offers embarrassing parallelism. We Trotterize, then measure.
Additional measurements may be required for sign extraction, which we
implement. We do not concede the phase problem, and we do not require a
classical solution to be computed simultaneously to steer the quantum
trajectory, as is so often the case in the literature.

The 4^q cost is a per-iteration burden, and the same scaling appears in
quantum algorithms across other domains. In CFD the qubit count `q` may
be large, so a method that scales better is wanted. The Cole–Hopf
substitution linearizes Burgers into the heat equation, bypasses the 4^q cost, and runs much faster than Pauli–Trotter; LCU can be used for the time-step. This
pathway is not in the Meena et al. AIAA paper; we added it for
comparison. We also added 1-D QLBM as a third pathway for cross-method
comparison on the same case.

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
analytic reference, pairs with FTCS/Godunov), or `cole_hopf_exact` (a
Neumann cosine sum on `φ` that admits a closed-form analytic `u(x, t)`
under unforced Cole–Hopf heat evolution — see §4.4). Sources: `sine`
(Gopalakrishnan Meena AIAA-2026 reference test problem,
`g(x,t) = sin(2πx)·cos(2πt)`) or `none`.

The CLI is [`burgers_solver.py`](../src/burgers_solver.py); by default
it runs a reference trajectory alongside the chosen method and reports
the L² error against it. The reference is chosen automatically:

- `--ic cole_hopf_exact` (the default when `--method` is `cole_hopf` or
  `cole_hopf_circuit`): the closed-form analytic `u(x, t)` evaluated at
  each saved step — exact, microsecond cost.
- Otherwise: a classical Forward-Time Central-Space (FTCS) Burgers
  solve, or Godunov via `--classical-baseline godunov`.

Flags:

- `--no-analytic-reference`: under `--ic cole_hopf_exact`, fall back to
  the FTCS/Godunov reference (e.g. for cross-validating the analytic
  formula against the classical solver).
- `--no-classical-reference`: skip the reference trajectory entirely
  (no error metrics, no `speedup_ratio` in the analysis fragment).

Quantum trajectories may have pre-processing, but once rolling do not
refer back to the classical solution for steerage. (The solver does
support quantum-classical hybrid modes for diagnostic comparison.)

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

- **Direct-u**: march `u` directly (six methods).
- **Cole–Hopf**: linearize via `u = −2ν ∂_x ln φ` and march `φ` (two methods).
- **QLBM (kinetic)**: march mesoscopic distributions `f_i` and recover `u`
  by moments (two methods).

Each method is classified by how much classical machinery it carries:

| Class | Symbol | Meaning |
|---|---|---|
| Classical | C | No quantum objects |
| Hybrid | H | Operator rebuilt from classical state each step (classical mirror in the time loop) |
| Near-pure quantum | NQ | Classical mirror confined to operator construction; evolution itself is fully quantum |
| Pure quantum | PQ | No classical mirror in the time loop |

Method roster:

| `--method` | Family | Class | Notes |
|---|---|---|---|
| `shift` | Direct-u | C | FTCS reference, baseline |
| `quantum_exact` | Direct-u | H | Diagnostic upper bound (no Trotter error) |
| `quantum_circuit` | Direct-u | H | **Hybrid Pathway 1 (Meena AIAA-2026 Appendix A.A)** |
| `mps` | Direct-u | H | MPS state-prep + dense evolution |
| `tebd` | Direct-u | C | Tensor-network classical |
| `tebd_circuit` | Direct-u | NQ | W-II quantum circuit |
| `cole_hopf` | Cole–Hopf | C | Classical reference for the CH pathway |
| `cole_hopf_circuit` | Cole–Hopf | PQ | **Pure-quantum Pathway 2 (this work)** |
| `lbm` | QLBM | C | Classical D1Q3 BGK (renamed from `qlbm` — pure classical, no shots) |
| `qlbm_circuit` | QLBM | H | Quantum-circuit D1Q3 ("Option A": classical collision shadow + Householder dilation; statevector and shots both real) |

The two **headline pathways** are Pathway 1 = `quantum_circuit`
(hybrid; Pauli decomposition fit per step to a classical Euler
reference, then Trotterised) and Pathway 2 = `cole_hopf_circuit`
(the only pure-quantum pathway in the codebase; no classical mirror
in the time loop after `φ₀` is prepared from `u₀`). They are the
primary scientific objects of this codebase and are detailed
mathematically in §§3.3 and 4.3. `qlbm_circuit` is the
cross-comparison code from a different algorithmic family. All other
methods exist as references and diagnostics.

---

## 3. Direct-u family

### 3.1 `shift` (classical)

Explicit forward-Euler with central-difference shift-operator FD on `u`.
Pure classical baseline; no quantum objects. `O(N)` per step.

### 3.2 `quantum_exact` (hybrid, statevector)

At each step, freeze the nonlinear Burgers RHS at the current state, fit
a Hermitian operator `Â` to it via Pauli decomposition, then apply
`expm(−i·Â·δt)` (SciPy dense matrix exponential). No Trotter error, but a
classical mirror (the frozen RHS) is required every step. Diagnostic /
upper-bound reference for the circuit methods. The Pauli decomposition
builds a `4^q × N` matrix and solves a least-squares system —
`O(4^q · N)` per step, which is the bottleneck. The `expm` adds `O(N³)`.
Out-of-memory at `q ≥ 6`.

### 3.3 `quantum_circuit` — Hybrid Pathway 1 (Meena AIAA-2026 Appendix A.A)

Same per-step Pauli Hamiltonian as `quantum_exact`, but evolved via a
Suzuki–Trotter circuit instead of exact `expm`. This pathway directly
follows Gopalakrishnan Meena et al. AIAA-2026 **Appendix A.A** (Eqs. 16–17),
which the paper itself labels as a proposed/future-work alternative to
their §V.C implementation. The paper's §V.C MPS/MPO classical-Euler
pipeline (velocity stored as MPS, spatial operators applied as `quimb`
MPOs, then `.to_dense()` and Euler-stepped classically) is **not**
implemented in this method — we use it externally as a cross-check
reference. The shift-operator FD used here for the classical reference
RHS matches the *form* of §V.C Eq. 15 but not its MPS/MPO apparatus.

**This is a hybrid pathway, not a pure-quantum one.** The Hamiltonian Â
is fit per step to a classical Euler trajectory, so the time loop
carries an inseparable classical mirror. Per-step classical work that
runs alongside the circuit (all in `burgers_nonlinear.py` /
`burgers_trotter.py`):

1. **Classical Euler RHS** (`compute_rhs_shift`) — `ν∇²u − u·∇u + g`
   built with shift-matrix FD on the un-normalised field.
2. **Classical predictor** — `u_next = u + δτ·rhs` in
   `build_evolution_hamiltonian`.
3. **Classical least-squares solve** (`solve_pauli_coefficients`) —
   builds `2·Re(S)` and `b` on `4^q` Pauli strings and calls
   `np.linalg.lstsq`. Dominant cost; `O(4^q · N)` wall at large `q`.
4. **Classical norm tracking** — `‖u + δτ·rhs‖` carried on the
   classical side and reapplied via `u_evolved_norm · norm_next` after
   the unitary step (the unitary preserves `‖|u⟩‖ = 1`).
5. **Classical Dirichlet projection** (`_project_dirichlet`) — when
   `bc='dirichlet'`, zero the boundary amplitudes of the evolved state
   and renormalise.
6. **Classical sign recovery** (when `shots > 0` and `--sign-recovery`
   ≠ `none`) — the `classical_oracle` mode literally copies signs from
   the classical reference; even `hadamard_test` / `dual_rail` leave
   classical norm/sign bookkeeping in the loop.

The genuinely quantum core is the unitary application: state-prep
`initialize(u_norm)` + the Trotterised `PauliEvolutionGate` +
statevector readout (or shots-mode amplitude reconstruction). That
core implements Appendix A.A's Eq. 17, but it sits inside the
classical scaffold above.

Supports periodic and Dirichlet boundary conditions; statevector and
shots modes; optional sign recovery.

#### 3.3.1 Classical target state (paper Eq. 15–16)

The quantum Hamiltonian is fitted per timestep to reproduce the
forward-Euler update. This is the defining design decision of the Meena
approach: the quantum pipeline (state prep → Hamiltonian simulation →
measurement) is validated against a known classical trajectory, not used
as an independent solver.

At time `t_n`:

1. Compute the classical RHS: `Δ = ν∇²u − u·∇u + g` (Euler step from
   paper Eq. 15).
2. Predict the next physical state: `u_{n+1} = u_n + δτ · Δ`.
3. Normalize both states: `|u⟩ = u/‖u‖`, `|u_next⟩ = u_{n+1}/‖u_{n+1}‖`.
4. Form the target rate of change: `δ₀ = (|u_next⟩ − |u⟩) / δτ`.

The norm `‖u_{n+1}‖` is tracked classically; it is not recovered from the
quantum circuit. The circuit evolves only the normalized direction.

#### 3.3.2 Pauli decomposition (paper Appendix A.A, Eq. 16)

The Hermitian operator `Â = Σ_i c_i P̂_i` is determined by solving a
linear system that matches `exp(−i·δτ·Â)|u⟩ ≈ |u⟩ + δτ · δ₀` to first
order. There are `4^q` Pauli string labels
`P̂_i ∈ {I, X, Y, Z}^⊗q`. The system is

```
b_i  = −2 Im(⟨u|P̂_i|δ₀⟩)                                                (5)
S_ij = ⟨P̂_i u | P̂_j u⟩                                                  (6)
2 Re(S) c = b                                                            (7)
```

This is an overdetermined system (`4^q` unknowns, rank ≤ `N²`). It is
solved via NumPy least-squares (`lstsq`). Near-zero coefficients
(`|c_i| < 10⁻¹⁵`) are filtered for circuit efficiency. The resulting `Â`
is verified Hermitian to `10⁻¹²` tolerance. Cost: building the full
`4^q × N` overlap matrix `S` each timestep is `O(4^q · N) = O(8^q)` per
step — the dominant scaling bottleneck.

#### 3.3.3 Trotterized evolution circuit (paper Eq. 17)

Given `Â = Σ_i c_i P̂_i`, the evolution operator `exp(−i·δτ·Â)` is
approximated by a Suzuki–Trotter product:

```
exp(−i·δτ·Â) ≈ ∏_i exp(−i·δτ·c_i · P̂_i)                                 (8)
```

Each factor `exp(−i·δτ·c_i·P̂_i)` is a Pauli rotation gate, synthesized
by Qiskit's `PauliEvolutionGate` with `SuzukiTrotter(order=k, reps=r)`.
Order `k=1` gives Lie–Trotter (`O(δτ²)` error per step); `k=2` gives the
symmetric Strang splitting (`O(δτ³)` error per step). The full per-step
circuit is:

1. State preparation: initialize `q` qubits to `|u⟩ = u/‖u‖`.
2. Apply the Trotter circuit `exp(−i·δτ·Â)` to the `q`-qubit register.
3. Measurement (shots mode) or statevector extraction.
4. Rescale: `u_{n+1} = ‖u_{n+1}‖ · |u_evolved⟩` (classical norm).

#### 3.3.4 Sign recovery in shots mode

Measurement yields probabilities `p_k = |u_k|²/‖u‖²`, which lose the sign
of `u_k`. Three strategies are implemented (`--sign-recovery`):

- **`hadamard_test`** — an ancilla qubit initialized to `|+⟩` controls
  the evolution circuit; ancilla statistics yield `Re(⟨k|U|u⟩)`, from
  which `sign(u_k)` is extracted per grid bin. Fully quantum — no
  classical oracle.
- **`dual_rail`** — the state is split into positive and negative
  components, each evolved on a separate circuit; recombination
  recovers the signed amplitudes. Doubles the circuit count.
- **`classical_oracle`** — signs copied from the classical reference
  solution. Diagnostic; defeats the standalone-solver purpose.
- **`none`** — only valid when the true solution is known non-negative.

#### 3.3.5 Dirichlet boundary enforcement

The Pauli least-squares fit correctly targets `δ₀[0] = δ₀[N−1] = 0`
(zero RHS at boundaries). However, the unitary `exp(−i·δτ·Â)` couples all
amplitudes, causing `O(δτ)` leakage into boundary indices each step. We
project back onto the Dirichlet subspace after each evolution:

```
u'[0] = u'[N−1] = 0,   then renormalize: u' ← u' / ‖u'‖                  (9)
```

This is the standard approach for enforcing hard BCs on quantum PDE
solvers: the evolution is slightly non-physical at boundaries, and
post-projection restores the constraint.

#### 3.3.6 Complexity summary

| Operation | Cost per step | Notes |
|---|---|---|
| Pauli decomposition (build `S`, solve) | `O(8^q)` | Dominant bottleneck |
| Trotter circuit depth | `O(4^q · r)` | `r` = Trotter reps |
| State initialization | `O(N)` | Qiskit `initialize()` |
| Norm tracking | `O(N)` | Classical, 1 inner product |

The `8^q` scaling makes this pathway impractical beyond `q ≈ 5–6`. At
`q=7` (128 grid points) the overlap matrix alone has 16384 rows and 128
columns, and must be rebuilt every timestep because the Hamiltonian is
state-dependent.

### 3.4 `mps` (hybrid, tensor)

Encode the current `u` into an MPS circuit using the Ran 2020
state-prep decomposition (with optional bond-dim truncation), simulate
the circuit to obtain the quantum state, then apply the exact dense
Hamiltonian evolution. Useful for studying MPS compression fidelity
independently of time-integration error. Uses the same `O(4^q · N)`
Pauli Hamiltonian as `quantum_exact`.

### 3.5 `tebd` (classical, tensor)

Build the dense Hermitian evolution generator directly from shift
operators (`O(N²)`, bypassing the `O(4^q)` Pauli decomposition), compute
`expm` once per step (`O(N³)`), convert to an MPO via
`quimb.MatrixProductOperator.from_dense`, and apply the MPO to the MPS
state with bond-dim truncation (`O(N · χ³)` per step). Multi-step
delegating path. Scales to `q=12` (`N=4096`); the bottleneck is the
dense `expm`, not the MPO ops.

### 3.6 `tebd_circuit` (near-pure quantum)

TEBD-style quantum circuit: MPS state-prep followed by a W-II (Zaletel)
gate layer that encodes one evolution step. Uses the same `O(N²)` dense
Hamiltonian as `tebd` (no Pauli decomposition). Per-step path through
the framework loop. Not pure-quantum: the Hamiltonian is rebuilt from
the current `u` each step (classical mirror). However the evolution
itself — state-prep, gate layer, measurement — is entirely quantum. The
classical mirror is confined to operator construction, not state readout
or steerage.

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

### 4.2 `cole_hopf` (classical, tensor reference)

Apply Eq. 10, build the heat propagator `exp(ν·L·δt)` once as a dense
matrix (`O(N³)` one-time cost), convert to MPO, and reuse it every step
(state-independent). Per-step cost is `O(N · χ³)` for the MPO-on-MPS
apply. Invert via Eq. 12 with log-domain finite differences. Multi-step
delegating path.

### 4.3 `cole_hopf_circuit` — Pure-Quantum Pathway 2

Same Cole–Hopf linearization, but the heat equation is marched as a
quantum circuit. Because Cole–Hopf turns Burgers into a *linear* PDE
with a *state-independent* generator, the per-step evolution unitary
is fixed for the duration of the run: no per-step Hamiltonian rebuild,
no per-step lstsq fit, no classical RHS computation inside the time
loop. This is what makes Pathway 2 pure-quantum in a sense that
Pathway 1 isn't.

Initial `φ` amplitudes are loaded onto qubits via the Ran 2020
MPS-to-circuit state-prep pipeline (the MPS is used only for encoding,
not for evolution). Three propagator variants are available
(§§4.3.2–4.3.4); shots-mode readout is described in §4.3.6 and segmented
evolution in §4.3.7. Sign recovery is not needed because `φ > 0` by
construction.

**What residual classical work remains (and where).** The "pure-quantum"
label is meant honestly: nothing in the time loop reaches for a
classical PDE state. The classical work that does exist is at the
encode/decode boundary and one-time setup:

1. **Forward Cole–Hopf transform** (`burgers_cole_hopf.cole_hopf_forward_centered`)
   — one-time at `t=0`: compute `φ₀ = exp(−(2ν)⁻¹∫u₀)` in log-space
   with mean-centring for numerical stability.
2. **MPS state-prep** (`burgers_mps.classical_to_mps` →
   `mps_to_circuit`, Ran 2020) — one-time per circuit build: convert
   classical `φ₀` amplitudes into a state-prep subcircuit. In segmented
   mode this prep is repeated once per segment (§4.3.7).
3. **Propagator-coefficient build** — one-time per `(ν, δt, q)` choice
   (`compute_theta_exact`, `compute_theta_dct`, or the LCU PREPARE
   coefficients `α_k = √(|c_k|/λ)`). For `dense-block` this includes
   a one-time `O(N³)` `np.linalg.eigh` of the dense propagator; for
   `dense-block` *with source forcing*, the eigendecomposition is
   repeated **per step** because `V(x, t)` changes the operator (see
   §4.3.3).
4. **Post-shot amplitude reconstruction** (`post_select_counts` →
   `reconstruct_phi_from_counts`) — at readout: discard shots where
   the ancilla flagged failure, then convert kept counts to
   amplitudes `φ̂ₖ = √(cntₖ/n_kept) · √(P_success) · ‖φ‖` (§4.3.6).
5. **Inverse Cole–Hopf transform** — at readout: `u = −2ν · ∂ₓ ln φ`
   via central differences in log-space.

None of the above is inside the per-step quantum time loop. By
contrast, Pathway 1's hybrid mirror is — that is the structural
difference between the two pathways.

#### 4.3.1 Block-encoding of the heat propagator

The heat propagator `P = exp(ν·L·δt)` is a contractive operator (all
eigenvalues in `(0, 1]`) that must be embedded into a unitary circuit.
All three variants use the same block-encoding strategy: an ancilla
qubit rotated by `arccos(d_k)` conditioned on the eigenstate index `k`,
so that post-selecting the ancilla in `|0⟩` implements the contraction

```
|k⟩|0⟩  →  |k⟩ (d_k|0⟩ + √(1−d_k²)|1⟩)                                  (13)
```

Post-selecting `ancilla = |0⟩` yields the state `Σ_k d_k ψ_k |k⟩` with
probability `P_success = Σ_k d_k² |ψ_k|²`. The success probability
degrades as `ν·δt` grows (stronger damping), requiring more shots in the
stochastic regime.

#### 4.3.2 Propagator A — `qft-diagonal` (periodic, Dirichlet-on-u via DCT)

The discrete periodic Laplacian is diagonalized by the DFT. Its
eigenvalues are

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
DCT and uses the Neumann eigenvalues `λᵏⁿ = −(4/δx²)sin²(πk/(2N))`
(see `heat_dct_step_circuit` in `burgers_cole_hopf_circuit.py`); the
conditional-Ry / ancilla-rotation structure is unchanged. The
`qft-diagonal` propagator label therefore covers both BCs, just with
different transform pairs (QFT/QFT⁻¹ for periodic, DCT/DCT⁻¹ for
Dirichlet-on-`u`). True u-Neumann (mapping to phi-Dirichlet, needing
the DST) is FUTURE-WORK item #4. Source forcing under `qft-diagonal`
is not supported (the source potential `V` is position-diagonal and
does not share an eigenbasis with the Laplacian); use `dense-block`
or `lcu` instead.

#### 4.3.3 Propagator B — `dense-block` (any BC, exact)

For Dirichlet BC the eigenbasis is the DCT-II (cosine transform), with
Neumann Laplacian eigenvalues

```
λ_k = −(4/δx²) sin²(π·k/(2N)),   k = 0, …, N−1                          (15)
```

The dense-block propagator builds the full `N × N` matrix
`P = exp(ν·L·δt)`, eigendecomposes it (`P = V D V†`), and block-encodes
via:

1. `V†` on `q` data qubits (to eigenbasis via `UnitaryGate`).
2. Conditional-Ry(`2 arccos(d_k / s_max)`) on ancilla, controlled by `k`.
3. `V` on `q` data qubits (back to computational basis).
4. Measure and reset ancilla.

The `s_max` normalization ensures all rotation arguments lie in
`[−1, 1]`. This propagator is exact per step (no Trotter error) and
supports source-term forcing via a Strang-split potential `V(x, t)`.
Gate count: `O(4^q)` due to the dense `UnitaryGate` — acceptable for
`q ≤ 5` but limits scalability.

**Hidden classical cost.** `dense-block` runs `np.linalg.eigh` on the
`N×N` propagator matrix to obtain `(V, D)`. In the unforced case
(constant `V(x)`) the eigendecomposition runs **once** at circuit-build
time and the resulting step circuit is reused for every timestep
(`heat_dense_block_full_circuit` pre-builds a single shared step_qc;
see `burgers_cole_hopf_circuit.py:524-528`). In the forced case
(`source_fn is not None`), the potential `V(x, tₙ)` changes the
operator so a **fresh eigendecomposition runs per step**, giving an
`O(N³ · n_steps)` classical setup cost in addition to the
`O(4^q)`-gate quantum circuit. This is the dominant classical cost on
the dense-block + source path at `q ≥ 5`.

#### 4.3.4 Propagator C — `lcu` (polynomial gate scaling)

The Linear Combination of Unitaries (LCU) propagator achieves
`O(M · q)` gate scaling, where `M` is the series truncation order.
This is the key innovation for scaling to `q = 7–8` on Frontier-class
machines.

**Periodic BC — Taylor LCU.** For periodic BC the heat propagator is
expanded directly in the Laplacian:

```
P_M = Σ_{k=0}^{M} (ν·δt)^k L^k / k!                                     (16)
```

Each power `L^k` of the discrete Laplacian
`L = (S⁺ + S⁻ − 2I)/δx²` expands via the trinomial theorem into a sum of
shift operators `S^n` with integer net shift `n = a − b`:

```
L^k = (1/δx²)^k Σ [k!/(a! b! c!)] (−2)^c (S⁺)^a (S⁻)^b,  a+b+c = k     (17)
```

Terms are aggregated by net shift, and each unique shift `S^n` is
implemented as `n` cascaded increment/decrement circuits on `q` qubits.

**Dirichlet BC — Fourier–Bessel LCU.** For Neumann BC (Dirichlet-on-`u`
via Cole–Hopf) the propagator is diagonal in the DCT-II eigenbasis with
entries

```
d_k = exp(ν·λ_k·δt) = exp(−A/2) · exp((A/2) cos(π·k/N))                 (18)
```

where `A = 4ν·δt/δx²`. The second factor is expanded via the
Jacobi–Anger (modified Bessel) identity:

```
exp(x cos φ) = I₀(x) + 2 Σ_{j=1}^{M} I_j(x) cos(j·φ)                    (19)
```

where `I_j(x)` is the modified Bessel function of the first kind.
Crucially, all Bessel coefficients are positive for `x > 0`, so no sign
absorption is needed in the SELECT oracle. Each cosine term decomposes
into two diagonal unitaries:

```
cos(j·π·k/N) = ½ [V_j⁺ + V_j⁻],   V_j±[k,k] = exp(±i·j·π·k/N)           (20)
```

Each `V_j±` is implemented as `q` single-qubit phase gates:
`P(j·π · 2^l / N)` on qubit `l` for `l = 0, …, q−1`. This gives `2q`
gates per Bessel term, for a total SELECT cost of `O(M · q)`.

The full per-step circuit is:

1. DCT on `q` data qubits (to cosine eigenbasis).
2. PREPARE on `m = ⌈log₂(2M+1)⌉` ancilla qubits: `|0⟩^m → Σ_k α_k |k⟩`.
3. SELECT: apply `V_k` conditioned on ancilla state `|k⟩`.
4. PREPARE† (uncompute ancilla).
5. DCT† on `q` data qubits (back to computational basis).
6. Post-select all ancilla qubits on `|0⟩`.

where `α_k = √(|c_k|/λ)` with `λ = Σ |c_k|`. Post-selecting
`ancilla = |0⟩^m` yields the heat propagator divided by `λ`. The LCU
normalization factor `λ` is tracked for amplitude rescaling. Verified:
machine-precision agreement (`2.2 × 10⁻¹⁴`) with the dense-block
reference at `q = 3`, `ν = 0.1`.

#### 4.3.5 Source forcing

With `--source` enabled, the per-step circuit becomes a Strang sandwich
`exp(−V·δt/2) · P_heat · exp(−V·δt/2)`, where `V(x, t)` is the
Cole–Hopf-mapped potential of the source `g(x, t)`. This adds two
ancillas (one per half-step) and introduces `O(δt²)` Strang error. The
`dense-block` and `lcu` propagators support source forcing;
`qft-diagonal` raises `NotImplementedError`.

**Deriving `V(x, t)` from `g(x, t)`.** Under Cole–Hopf, a source
`g(x, t)` on the velocity equation transforms into a *multiplicative*
potential on `φ`:

```
∂φ/∂t = ν∇²φ − V(x, t)·φ,    V_x = +g/(2ν)                              (21)
```

`V` is the spatial antiderivative of `g/(2ν)`. The code
(`burgers_potential.potential_from_source`) integrates `g/(2ν)` via the
trapezoid rule on the grid and gauge-fixes `V` to zero spatial mean
(adding a constant to `V` only multiplies `φ` by a global
time-dependent factor, which cancels in `u = −2ν∂ₓlnφ`). Choosing the
mean-zero gauge prevents `φ` from accumulating multiplicative blow-up
or decay over long runs.

#### 4.3.6 Shots-mode readout and post-selection

In statevector mode the data-register amplitudes are read directly
after applying the evolution circuit, and the LCU normalisation /
ancilla post-selection are accounted for analytically.

In shots mode (`--shots N > 0`), the full circuit (including ancilla
measurements) is sampled `N` times. Reconstruction proceeds in two
stages (`burgers_cole_hopf_circuit.py:993-1025`):

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
segmented-evolution mode of §4.3.7.

**Experimental Hadamard-per-bin readout.** An alternative interferometric
readout that estimates each bin's signed amplitude via a Hadamard test
is implemented in `hadamard_per_bin_circuit` /
`extract_hadamard_per_bin_amplitudes` /
`_run_shots_hadamard_per_bin` (~250 lines, lines 1597–1851 of
`burgers_cole_hopf_circuit.py`). It is not the default path; it is
the workbench for FUTURE-WORK item #10 ("Peaked-φ shots readout"),
intended for the deep-shot regime at small `ν` where standard
post-selection becomes statistics-starved. Not validated for
production yet.

#### 4.3.7 Measure-and-reprepare (segmented) evolution mode

For long runs in shots mode, `--evolution-mode measure_reprepare` splits the
`n_steps` total into `K = n_steps / segment_size` segments. Each segment is
a self-contained circuit comprising state-prep + `segment_size`
propagator layers + measurement. Between segments
(`_run_shots_measure_reprepare` in `burgers_cole_hopf_circuit.py:1031`):

1. **Decode**: post-select and reconstruct `φ̂` from the segment's shots
   (§4.3.6).
2. **Re-encode**: run `classical_to_mps(φ̂)` → `mps_to_circuit(...)` to
   build a fresh state-prep subcircuit for the next segment's starting
   amplitudes. The MPS bond dim follows `--bond-dim`/`--mps-threshold`.
3. **Run next segment** with the fresh prep.

This trades cumulative post-selection survival (which falls
exponentially in `segment_size`) against information loss from
classical decode/re-encode at each segment boundary (limited by the MPS
truncation). The classical norm and `P_success` factors compose
multiplicatively across segments
(`cumulative_norm *= segment_norm; cumulative_p_success *= segment_p_success`),
so the reconstructed `φ̂` at any snapshot still represents physical
amplitudes. No classical PDE physics enters the loop — only amplitude
IO at the segment boundaries. Hardware backend support for segmented
evolution is deferred to v2; segmented mode currently requires
`--backend-type sim`.

#### 4.3.8 Complexity summary

Per-step cost for the three propagator variants (data qubits `q`,
`N = 2^q`; `M` is the LCU truncation order):

| Propagator | Gate count | Ancillas | Classical setup | Source forcing |
|---|---|---|---|---|
| `qft-diagonal` (periodic/Dirichlet-on-u) | `O(2^q)` cond-Ry + `O(q²)` (Q/D)FT | 1 | `O(N)` per `(ν, δt)` for `θ(k)` | not supported |
| `dense-block` (any BC, exact) | `O(4^q)` from dense `UnitaryGate` | 1 | `O(N³)` eigendecomp — *once* if unforced, *per step* if forced | supported |
| `lcu` (any BC, truncated) | `O(M · q)` SELECT + `O(N log N)` (Q/D)FT | `⌈log₂(2M+1)⌉` | `O(M)` Bessel/Taylor coefficient build, once | supported |

Measured CX counts (no source, periodic, statevector mode):
`qft-diagonal` 12 / 60 / 240 at `q = 3 / 4 / 5`. `lcu` verified to
machine precision (`2.2 × 10⁻¹⁴`) against `dense-block` at `q = 3`,
`ν = 0.1`.

Per-step `P_success` lower-bounds (smallest contraction across the
spectrum): `min_k d_k² = exp(2·ν·λ_min·δt)`, with `λ_min ≈ −4/δx²`
on a periodic grid. Worst-case survival to step `n` is therefore
`exp(2·ν·λ_min·δt·n)`; the spectrum-weighted average tracked by the
code via `P_success = Σ_k d_k² |ψ_k|²` is typically much larger
because mass concentrates on the smooth low-`k` modes that are
weakly damped.

### 4.4 Cole–Hopf analytic IC and reference (`--ic cole_hopf_exact`)

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
`burgers_cole_hopf.py`): `u = −2ν ∂ₓ ln φ` is well-defined only when
`φ > 0` everywhere. Sufficient condition: `a₀ > Σ |aₙ|` for `n ≥ 1`.
Since modes only decay, satisfying this at `t = 0` carries to all
`t > 0`. Violation raises `ValueError`.

**Constraints.** This IC family is restricted to:

- `--bc dirichlet` (`u(0) = u(L_box) = 0`). The cosine basis matches
  Neumann-on-φ, which is the Cole–Hopf dual of Dirichlet-on-u.
- `--source none`. A source couples the cosine modes (via the
  Cole–Hopf potential `V`) and breaks the independent-mode decay.

**Usage.** When `--method` is `cole_hopf` or `cole_hopf_circuit`,
`--ic` defaults to `cole_hopf_exact` and the analytic `u(x, t)` is
emitted as the reference trajectory. For other methods, `--ic
cole_hopf_exact` can be selected explicitly (the analytic formula is
method-agnostic — it's just the exact PDE solution for this IC). Pass
coefficients as a comma-list:
`--ic-cole-hopf-coeffs "1.0,0.3"` (default).

**Validation case.** Single mode `a = (1.0, 0.3)`, `ν = 1e-2`, `q = 5`,
`n_steps = 50`. Gives a tanh-bump `u₀(x)` profile that decays
monotonically. Method accuracy claims against this reference are
**independent of any classical co-solver** — the headline
"validated against another approximation" reviewer objection
disappears for this test family.

**Suppressing the analytic reference.** `--no-analytic-reference`
falls back to FTCS/Godunov even with `--ic cole_hopf_exact`, useful
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
`burgers_lbm.py`; integrator: `LBMIntegrator`; function:
`run_lbm_simulation`.)

### 5.2 `qlbm_circuit` (hybrid, quantum-circuit D1Q3)

The same D1Q3 algorithm as a quantum circuit on `q + 2` qubits (`q`
position qubits + 2 velocity qubits, interleaved encoding `|v⟩|p⟩`).
Streaming is a controlled increment/decrement on the position register;
collision is a dense unitary embedding ("Option A") rebuilt each step
from the current distributions and applied with amplitude rescaling to
account for its non-unitary contraction.

**This is a hybrid pathway, not a pure-quantum one** — the analog of
`quantum_circuit` (§3.3) in the QLBM family.  Every step the
collision unitary `U_collision(f_pre)` is constructed from
`f_post = collide_bgk(f_pre, τ)` *computed classically*, via two
Householder reflections that send the normalised `f_pre` state to the
normalised `f_post` state.  The quantum circuit then replays that
classical answer on the register.  The contraction factor
`‖f_post‖/‖f_pre‖` is likewise a classical scalar, tracked alongside
`cumulative_norm`.  This is the QLBM analog of Meena AIAA-2026
Appendix A.A — fit a per-step unitary to a known classical update.
The only pure-quantum pathway in this codebase remains
`cole_hopf_circuit`.

Three roads to a genuine pure-quantum QLBM are documented in
FUTURE-WORK as alternative algorithms, not modifications of this
method: #27 (Itani-style QALB), #28 (Carleman lift of BGK), and #29
(linearised-BGK).  Option A here exists as a hybrid validation /
cross-comparison tool, parallel to how `quantum_circuit` exists as
the hybrid baseline for Pathway 1.

**Shots are real, not a fallback.** Statevector and shots both
execute the same `build_qlbm_step_circuit` output.  The shots path
routes through the shared `q8020_cfd_qutil.circuit.transpile_circuit`
and `execute_circuit_counts` helpers (same execution layer as
`cole_hopf_circuit`'s shots paths) so `--optimization-level`,
`--seed`, and `--backend-type {sim,hardware}` are honoured uniformly
across both methods.  Reconstruction:
`|ψ_out_k| ≈ √(counts[k]/S)`, unflatten to `f_post`, rescale by
`cumulative_norm`.  Three sign-recovery modes are wired through
`--sign-recovery` (mirroring §3.3.4 for `quantum_circuit`):

- `none` (default) — non-negative magnitudes only.  Correct when
  `f_i ≥ 0` throughout (typical smooth-flow regime).  Per-step
  metric `negative_mass` reports `Σ|f⁻|/Σ|f|` from the classical
  oracle so you can see when this assumption breaks.
- `classical_oracle` — copy signs from a parallel
  `collide_bgk + stream` step.  Diagnostic grade: the *signs* now
  come from a classical reference, so this branch is doubly hybrid;
  use for shock-regime debugging, not as a stand-alone benchmark.
- `hadamard_test` — stand-alone interferometric sign recovery.  For
  each bin `k` a per-bin Hadamard test estimates
  `Re(⟨k|U_step|ψ_in⟩)`; since every QLBM operator (collision
  Householder + real streaming permutation) is real, that real part
  *is* `ψ_out_k`, so its sign is the recovered sign and is combined
  with the direct magnitude `√(counts[k]/S)`.  Unlike
  `classical_oracle` the signs come from the circuit itself, not a
  classical reference, so the run stays a stand-alone benchmark.
  Cost: `O(4N)` extra circuit executions per step (`_qlbm_hadamard_signs`,
  mirroring `burgers_cole_hopf_circuit.hadamard_per_bin_circuit`);
  per-step metric `hadamard_p_kept` reports the mean post-selection
  acceptance.  Diagnostic grade — for shock-regime deep dives, not
  production sweeps.

**Leakage diagnostic.** The velocity register has 4 states but D1Q3
only uses 3 (`|11⟩` is unused).  Per-step metric `leakage` reports
the probability mass landing in the `|11⟩` block — structurally zero
on noise-free Aer, nonzero on hardware or under transpilation error.

**No `P_success` decay.** Unlike `cole_hopf_circuit`'s block-encoding,
QLBM has no post-selection ancilla — the contraction lives in a
classical scalar.  Every shot is "usable".  Shot budgets therefore
scale linearly with the noise floor rather than multiplicatively with
`n_steps`.  See SPEC-qlbm-shots-and-sign-recovery.md for the full
contract.

---

## 6. Comparative analysis

### 6.1 Gate complexity per step

| Pathway | Propagator | CX per step (`q=5`) | Asymptotic |
|---|---|---|---|
| Pauli–Trotter (`quantum_circuit`) | — | ~10 000+ | `O(8^q · r)` |
| Cole–Hopf (`cole_hopf_circuit`) | `qft-diagonal` | 240 | `O(2^q + q²)` |
| Cole–Hopf (`cole_hopf_circuit`) | `dense-block` | 1 280 | `O(4^q)` |
| Cole–Hopf (`cole_hopf_circuit`) | `lcu` | `O(M · q)` | `O(M · q)` |
| QLBM (`qlbm_circuit`) | (Option A) | dense `O(N²)` collision per step | `O(N²)` (Option B/QSVT pending) |

The LCU propagator with Bessel truncation order `M = 8` requires
`m = ⌈log₂(17)⌉ = 5` ancilla qubits and 16 diagonal unitaries × `q`
phase gates each = `16q` CX-free gates per step. The PREPARE/PREPARE†
state preparation adds `O(2^m)` gates but `m` is logarithmic in `M`.

### 6.2 Accuracy characteristics

- **Pauli–Trotter** — accuracy is determined by (a) the `lstsq` residual
  of the Pauli decomposition, (b) the Trotter splitting error
  `O(δτ^{k+1})` per step, and (c) sign-recovery fidelity in shots mode.
  The Hamiltonian is re-fitted each step against the classical
  reference, so errors do not compound — each step independently targets
  the exact next state.
- **Cole–Hopf** — the `dense-block` propagator is exact per step (no
  Trotter error). `qft-diagonal` uses exact rotation angles via Möbius
  expansion (no truncation). `lcu` introduces Taylor (periodic) or
  Bessel (Dirichlet) series truncation, and all variants pay
  post-selection cost. The Cole–Hopf transform itself contributes
  discretization error from the trapezoidal quadrature (Eq. 10) and the
  FD log-derivative (Eq. 12), both `O(δx²)`.
- **QLBM** — accuracy carries the standard `O(δx²)` LBM discretization
  error; the circuit `qlbm_circuit` adds amplitude-rescaling error from
  the non-unitary collision embedding (statevector path is otherwise
  bit-exact to the classical reference).

### 6.3 Boundary-condition support

| Pathway / variant | Periodic | Dirichlet |
|---|---|---|
| Pauli–Trotter (`quantum_circuit`) | ✓ | ✓ (post-step projection) |
| Cole–Hopf, `qft-diagonal` | ✓ | — |
| Cole–Hopf, `dense-block` | ✓ | ✓ |
| Cole–Hopf, `lcu` | ✓ | ✓ |
| QLBM (`lbm`, `qlbm_circuit`) | ✓ | ✓ (bounce-back) |

---

## 7. Per-method CLI options

Most flags only apply to a subset of methods. The CLI will raise an
explicit "not supported" error for invalid combinations.

### 7.1 Encoding (`--encoding {binary,gray}`)

Used by `cole_hopf_circuit`. `binary` is index-aligned (default); `gray`
uses the reflected Gray-code permutation `π(i) = i ⊕ (i >> 1)` on the
Laplacian / propagator matrix. The encoding choice affects which
two-qubit gates are nearest-neighbour after transpilation. See
[SPEC-encoding-switch.md](future/SPEC-encoding-switch.md).

### 7.2 Propagator (`--propagator {qft-diagonal,dense-block,lcu}`)

Used by `cole_hopf_circuit`. Selects the heat-equation circuit
construction (§4.3.2 – §4.3.4). Source forcing (`--source sine`) is
supported on `dense-block` and `lcu`; `qft-diagonal` + source raises
`NotImplementedError`.

### 7.3 Trotter order / reps (`--trotter-order`, `--trotter-reps`)

`quantum_circuit` only. Suzuki–Trotter order (1 or 2) and number of
sub-step repetitions per timestep.

### 7.4 MPS bond dimension (`--bond-dim`, `--mps-threshold`)

Used by `mps`, `tebd`, `tebd_circuit`, and the state-prep stage of
`cole_hopf_circuit`. `--bond-dim None` keeps full rank;
`--mps-threshold` is a singular-value cutoff during compression.

### 7.5 Sign recovery (`--sign-recovery`)

Applies to `quantum_circuit`, `mps`, `tebd_circuit` when reading out via
shots. Methods that go through Cole–Hopf do not need it because `φ > 0`
by construction. Choices and semantics in §3.3.4.

### 7.6 Shots and backend

`--shots N` (0 means statevector). `--backend-type {sim,fake,hardware}`,
`--backend NAME`, `--t1`, `--t2`, `--coupling-map`, `--seed`,
`--optimization-level`. See
[SPEC-shots-backend.md](archive/SPEC-shots-backend.md).

### 7.7 Evolution mode (`--evolution-mode {single,measure_reprepare}`, `--segment-size`)

`cole_hopf_circuit` shots-mode only. `single` = one big circuit with
`n_steps` inlined step layers (today's default). `measure_reprepare`
(segmented) = break the evolution into `K`-step segments, read out and
re-prep amplitudes between segments. Trades depth-per-circuit against
shot-noise compounding. See
[SPEC-measure-reprepare-evolution.md](archive/SPEC-measure-reprepare-evolution.md).

### 7.8 Source forcing (`--source {sine,none}`)

All methods accept it. The `dense-block` path threads it through to a
per-step `V(x, t)` potential in the heat propagator (see
[SPEC-source-forcing.md](archive/SPEC-source-forcing.md)). Other paths
inject it directly into the `u` RHS or the φ-equation.

### 7.9 Time-window (`--shock-pct`, `--n-steps`)

Either a percentage of the inviscid shock-formation time
`t_shock = 1 / max|du₀/dx|` (resolves to an `n_steps` from the fixed
CFL-derived `dt`), or an explicit step count.

### 7.10 Initial condition (`--ic`, `--ic-*`)

`--ic {sine,multimode,gaussian,cole_hopf_exact}`.  Default is method-
dependent: `cole_hopf_exact` when `--method` is `cole_hopf` or
`cole_hopf_circuit`, `sine` otherwise.  IC-specific knobs:

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
  `--source none`.  See §4.4 for the math and the analytic-reference
  pairing.

Passing an IC-specific flag with the wrong `--ic` emits a warning and
ignores the flag.

### 7.11 Reference trajectory (`--no-classical-reference`, `--no-analytic-reference`)

By default the solver runs a reference trajectory alongside the
chosen method and reports L² error against it.  The reference is
chosen automatically: closed-form analytic when `--ic cole_hopf_exact`
(microsecond cost), otherwise FTCS via `solve_burgers` (or Godunov via
`--classical-baseline godunov`).  Two flags suppress:

- `--no-analytic-reference` — under `--ic cole_hopf_exact`, fall
  back to FTCS/Godunov instead of the closed-form.  Useful for
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
[`burgers_fw.py`](../src/burgers_fw.py).

### 8.1 Components

- **Config**: `BurgersConfig(SolverConfig)` adds every CLI parameter
  in §7 as a dataclass field (including the §7.10 IC-specific knobs
  and the §7.11 reference-suppression flags). `describe()` returns a
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

These five methods plug into `MainLoop` the normal way — `step()`
advances one timestep, the framework owns the loop:

| `--method` | Integrator class | Step function it calls |
|---|---|---|
| `shift` | `ShiftEulerIntegrator` | `burgers_trotter.shift_euler_step` |
| `quantum_exact` | `QuantumExactIntegrator` | `burgers_trotter.quantum_exact_step` |
| `quantum_circuit` | `QuantumCircuitIntegrator` | `burgers_trotter.quantum_circuit_step` (or `dual_rail_quantum_step` if `--sign-recovery dual_rail`) |
| `mps` | `MPSIntegrator` | `burgers_trotter.mps_step` |
| `tebd_circuit` | `TEBDCircuitIntegrator` | `burgers_trotter.tebd_circuit_step` |

Each integrator pulls the source value at time `t` from
`config._source_fn(grid.xc, t)` and forwards it to the underlying step
function along with `bc`, `nu`, `dt`. Returns
`(DenseState(u_new), metrics_dict)`.

### 8.3 Delegating integrators (own their own loop)

Five methods carry tensor / circuit / kinetic state across timesteps,
or pre-build a propagator once and reuse it. They do not fit the
per-step model. They use the delegating-integrator idiom from
[SPEC-solverfw.md](../../q8020-cfd-metautil/docs/SPEC-solverfw.md) §5:
subclass `TimeIntegrator`, but in `step()` run the *entire* multi-step
simulation internally and return all snapshots via sentinel keys in the
metrics dict.

| `--method` | Integrator class | Inner driver |
|---|---|---|
| `tebd` | `TEBDIntegrator` | `burgers_tebd.run_tebd_simulation` |
| `cole_hopf` | `ColeHopfIntegrator` | `burgers_cole_hopf.run_cole_hopf_simulation` |
| `cole_hopf_circuit` | `ColeHopfCircuitIntegrator` | `burgers_cole_hopf_circuit.run_cole_hopf_circuit_simulation` |
| `lbm` | `LBMIntegrator` | `burgers_lbm.run_lbm_simulation` |
| `qlbm_circuit` | `QLBMCircuitIntegrator` | `burgers_qlbm_circuit.run_qlbm_circuit_simulation` |

The set of delegating methods is recorded as
`_DELEGATING_METHODS = {"tebd", "cole_hopf", "cole_hopf_circuit",
"lbm", "qlbm_circuit"}` in `burgers_fw.py`.

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
`(solutions: list[np.ndarray], step_metrics: list[dict] | None)` —
identical to the framework contract, identical across all ten methods.

### 8.5 Backend management

For the three quantum-circuit methods that shoot at a backend
(`quantum_circuit`, `mps`, `tebd_circuit`) the backend is built once in
`make_integrator()` via `q8020_cfd_qutil.backend.get_backend(...)` and
stored on the integrator instance. `cole_hopf_circuit` builds its
backend lazily inside the integrator's `_run_all` because it may not be
needed (`shots=0` path is statevector-only). `qlbm_circuit` consumes
the backend in the shots path: every step's circuit is transpiled via
`q8020_cfd_qutil.circuit.transpile_circuit` and executed via
`execute_circuit_counts` (the same shared helpers
`cole_hopf_circuit`'s shots paths use — `burgers_cole_hopf_circuit.py:
1182, 1187`), with counts reconstructed back to `f` (see §5.2).  In
statevector mode the backend is unused and the step is simulated via
`Statevector.evolve`.

The shared `qutil` execution layer means **all real-circuit shots
paths** (`quantum_circuit` per-step, `cole_hopf_circuit` batched or
segmented, `qlbm_circuit` per-step) honour the same flag contract:
`--shots`, `--seed`, `--optimization-level`, and `--backend-type
{sim,hardware}` produce comparable behaviour across methods.
`execute_circuit_counts` transparently dispatches to `backend.run`
for Aer and to `SamplerV2` for IBM-runtime backends, so methods
transition from simulator to hardware without per-method
code changes.

---

## 9. Implementation map

```
q8020-mps-burgers/
├── input/
│   └── burgers_quantum.toml         # q8020 sweep cases
├── postproc/
│   ├── plot_cole_hopf_circuit_evolution.py
│   ├── plot_hadamard_time_evolution.py
│   ├── plot_method_compare.py
│   ├── plot_paper_aligned.py
│   └── plot_pq_compare.py
└── src/
    ├── burgers_solver.py            # CLI entry point; reference-trajectory
    │                                # dispatcher (analytic / FTCS / Godunov /
    │                                # skipped) per --ic and the two
    │                                # --no-*-reference flags
    ├── burgers_fw.py                # solverfw bindings (§8)
    ├── burgers_classical.py         # ICs (sine, multimode, gaussian) +
    │                                # FTCS solve_burgers + source_term_sine
    ├── burgers_nonlinear.py         # compute_rhs_shift used by ShiftFD;
    │                                # Pauli decomp (Eqs. 5–7) + Trotter
    │                                # circuit synthesis (Eq. 8)
    ├── burgers_trotter.py           # per-step quantum kernels;
    │                                # sign-recovery dispatch; Dirichlet
    │                                # projection (Eq. 9)
    ├── burgers_mps.py               # Ran 2020 MPS prep + helpers
    ├── burgers_mpo.py               # heat MPO for the cole_hopf path
    ├── burgers_tebd.py              # TEBD multi-step driver
    ├── burgers_cole_hopf.py         # Forward/inverse CH transforms
    │                                # (Eqs. 10, 12); log-domain stability;
    │                                # cole_hopf_exact analytic IC + ref
    │                                # (§4.4, Eqs. 22–24)
    ├── burgers_cole_hopf_circuit.py # CH quantum-circuit pipeline;
    │                                # qft-diagonal + dense-block;
    │                                # Möbius conditional-Ry; DCT matrix
    ├── burgers_lcu.py               # PREPARE/SELECT/LCU primitives;
    │                                # Taylor LCU (periodic, Eqs. 16–17);
    │                                # Fourier–Bessel LCU (Neumann,
    │                                # Eqs. 18–20); Strang-split potential
    ├── burgers_lbm.py               # classical D1Q3 LBM (F11)
    ├── burgers_qlbm_circuit.py      # quantum-circuit D1Q3 LBM (F11)
    ├── burgers_potential.py         # V(x,t) for source-forced CH
    ├── burgers_encoding.py          # binary/gray encoding helpers
    ├── burgers_sign_recovery.py     # F9 sign-recovery strategies
    └── burgers_postprocess.py       # output writers, q8020 metrics dump
```

Method-to-module crosswalk for "where does the actual physics live":

| `--method` | Module |
|---|---|
| `shift` | `burgers_classical.py` (FTCS too) + `burgers_nonlinear.py` |
| `quantum_exact` | `burgers_trotter.py::quantum_exact_step` |
| `quantum_circuit` | `burgers_trotter.py::quantum_circuit_step` + `burgers_nonlinear.py` |
| `mps` | `burgers_trotter.py::mps_step` + `burgers_mps.py` |
| `tebd` | `burgers_tebd.py::run_tebd_simulation` |
| `tebd_circuit` | `burgers_trotter.py::tebd_circuit_step` |
| `cole_hopf` | `burgers_cole_hopf.py::run_cole_hopf_simulation` |
| `cole_hopf_circuit` | `burgers_cole_hopf_circuit.py::run_cole_hopf_circuit_simulation` + `burgers_lcu.py` |
| `lbm` | `burgers_lbm.py::run_lbm_simulation` |
| `qlbm_circuit` | `burgers_qlbm_circuit.py::run_qlbm_circuit_simulation` |

---

## 10. Running cases

### 10.1 Sweep harness (TOML)

`input/burgers_quantum.toml` contains q8020-sweeper cases. A
representative one for the Cole–Hopf circuit on a forced run:

```toml
[cole_hopf_circuit_forced_q5_smoke]
"--method"        = "cole_hopf_circuit"
"--propagator"    = "dense-block"
"--ic"            = "sine"
"--source"        = "sine"
"--nu"            = 0.1
"--cfl"           = 0.1
"--shock-pct"     = 100.0
"--q"             = 5
"--shots"         = 50000
"--backend-type"  = "sim"
"--seed"          = 42
"--save-every"    = 1
_group_postproc = "python ./q8020-mps-burgers/postproc/plot_cole_hopf_circuit_evolution.py"
```

The sweeper converts this to a CLI invocation of `burgers_solver.py`
and runs it; the postproc receives the resulting JSON dump (built by
`burgers_postprocess.py`) and renders the comparison plot.

### 10.2 CLI quick reference for the headline pathways

Pathway 1 (hybrid Pauli–Trotter, periodic):

```sh
python burgers_solver.py --q 5 --method quantum_circuit \
  --bc periodic --n-steps 10 --nu 1e-2 --noshow
```

Pathway 1 (hybrid Pauli–Trotter, Dirichlet):

```sh
python burgers_solver.py --q 5 --method quantum_circuit \
  --bc dirichlet --n-steps 10 --nu 1e-2 --source none --noshow
```

Pathway 2 (pure-quantum Cole–Hopf, `qft-diagonal`, periodic):

```sh
python burgers_solver.py --q 5 --method cole_hopf_circuit \
  --propagator qft-diagonal --bc periodic --n-steps 10 --nu 1e-2 \
  --source none --noshow
```

Pathway 2 (Cole–Hopf, `lcu`, Dirichlet):

```sh
python burgers_solver.py --q 5 --method cole_hopf_circuit \
  --propagator lcu --bc dirichlet --n-steps 10 --nu 1e-2 \
  --source none --noshow
```

Pathway 2 (Cole–Hopf, `dense-block`, Dirichlet, with shots):

```sh
python burgers_solver.py --q 5 --method cole_hopf_circuit \
  --propagator dense-block --bc dirichlet --n-steps 10 --nu 1e-2 \
  --source none --shots 10000 --noshow
```

Pathway 2 against the closed-form analytic reference (`--ic` defaults
to `cole_hopf_exact` for CH methods; reference is exact, no FTCS):

```sh
python burgers_solver.py --q 5 --method cole_hopf_circuit \
  --propagator lcu --bc dirichlet --n-steps 50 --nu 1e-2 \
  --source none --ic-cole-hopf-coeffs "1.0,0.3" --noshow
```

Same case but cross-checking the analytic vs FTCS as the reference:

```sh
python burgers_solver.py --q 5 --method cole_hopf_circuit \
  --propagator lcu --bc dirichlet --n-steps 50 --nu 1e-2 \
  --source none --ic-cole-hopf-coeffs "1.0,0.3" \
  --no-analytic-reference --noshow
```

---

## 11. What this solver does NOT do

- **No 2-D / 3-D.** Strictly 1-D. The framework is general enough for
  higher dimensions; the application is not.
- **No adaptive `dt`.** Fixed `dt = cfl · dx` (or `--dt` override).
- **No mesh refinement.**
- **No real hardware execution by default.** `--backend-type hardware`
  submits jobs and records placeholders; result harvest is a separate
  workflow (see [SPEC-shots-backend.md](archive/SPEC-shots-backend.md)
  §10).
- **No physics beyond viscous Burgers + source.** No reaction term, no
  compressibility coupling, no multi-component flow.

---

## 12. References and further reading

### Primary publications

- Meena, M. Gopalakrishnan et al., "A Tensor Network–based Quantum
  Algorithm for the Nonlinear 1-D Burgers' Equation," AIAA 2026 — the
  source paper for Pathway 1. Local copy:
  [`refs/AIAA2026_QC_final.pdf`](refs/AIAA2026_QC_final.pdf).
- Quartey, B. & Zhong, X., "Beyond the Simulator: A Practical
  Demonstration of Quantum Lattice Boltzmann Methods on IBM Quantum,"
  RPI / IBM, Nov 2025 — the source paper for the QLBM family.
- Cole, J. D., "On a quasi-linear parabolic equation occurring in
  aerodynamics," *Quart. Appl. Math.* **9**, 225 (1951); Hopf, E., "The
  partial differential equation u_t + u u_x = ν u_xx," *Commun. Pure
  Appl. Math.* **3**, 201 (1950) — the Cole–Hopf substitution
  underlying Pathway 2.
- Ran, S.-J., "Encoding of matrix product states into quantum circuits
  of one- and two-qubit gates," *Phys. Rev. A* **101**, 032310 (2020) —
  the MPS state-prep used by `mps` and by `cole_hopf_circuit`'s
  initial-state loading.
- Vidal, G., "Efficient classical simulation of slightly entangled
  quantum computations," *Phys. Rev. Lett.* **91**, 147902 (2003) — TEBD.
- Zaletel, M. P. et al., "Time-evolving a matrix product state with
  long-ranged interactions," *Phys. Rev. B* **91**, 165112 (2015) —
  W-II construction used by `tebd_circuit`.
- Alhawwary, M. & Wang, Z. J., "Comparative analysis of high-order
  methods for the Burgers equation," *J. Comput. Phys.* **373**, 835
  (2018) — §5.3 Burgulence case (pending implementation, see
  [SPEC-alhawwary-wang-5.3-burgulence.md](future/SPEC-alhawwary-wang-5.3-burgulence.md)).
  Local copy: [`refs/AlhawwaryWang2018_…_Journal_of_Computational_Physics).pdf`](refs/AlhawwaryWang2018_Fourier_analysis_and_evaluation_of_DG,_Journal_of_Computational_Physics%29.pdf).

### In-repo design specs and reviews

| Topic | Doc |
|---|---|
| Framework itself | [`q8020-cfd-metautil/docs/SPEC-solverfw.md`](../../q8020-cfd-metautil/docs/SPEC-solverfw.md) |
| Cole-Hopf circuit (Pathway 2) details | [`F10-IMPLEMENTATION-SPEC.md`](archive/F10-IMPLEMENTATION-SPEC.md) |
| QLBM family (F11) | [`F11-QLBM-SPEC.md`](archive/F11-QLBM-SPEC.md) |
| LCU propagator (F3) | [`SPEC-F3-LCU-method.md`](archive/SPEC-F3-LCU-method.md), [`SPEC-F3-LCU-source-forcing.md`](archive/SPEC-F3-LCU-source-forcing.md) |
| Source forcing | [`SPEC-source-forcing.md`](archive/SPEC-source-forcing.md), [`SPEC-source-forcing-REVIEW.md`](archive/SPEC-source-forcing-REVIEW.md) |
| Shots / backend / noise | [`SPEC-shots-backend.md`](archive/SPEC-shots-backend.md) |
| Measure-and-reprepare (segmented) evolution | [`SPEC-measure-reprepare-evolution.md`](archive/SPEC-measure-reprepare-evolution.md) |
| Encoding (binary vs gray) | [`SPEC-encoding-switch.md`](future/SPEC-encoding-switch.md) |
| Paper-fidelity review | [`REVIEW-murali-paper-fidelity.md`](archive/REVIEW-murali-paper-fidelity.md), [`DEEP-OVERLAP-murali-vs-ucan.md`](archive/DEEP-OVERLAP-murali-vs-ucan.md) |
| Future work / open gaps | [`FUTURE-WORK.md`](future/FUTURE-WORK.md) |
| 2-D / QLBM-vs-MPS comparison design | [`DISCUSSION-2D-rotational-and-qlbm-comparison.md`](future/DISCUSSION-2D-rotational-and-qlbm-comparison.md) |
