# F2 Phase B.1a SPEC — Physical Hamiltonian via Operator Splitting

Target: replace the rank-2 rotation-generator `build_hamiltonian_dense`
with a physically meaningful Hamiltonian that (a) a CFD reader will
accept as real, (b) has local ladder-form MPO structure so that the
Zaletel W-II gate fusion yields genuine O(dt³) unitary evolution.

## 1. Why B.1 failed and why we cannot just "refactor to ladder form"

The current `build_hamiltonian_dense(u, dx, dt, nu, ...)` returns

```
A = i·|δ⟩⟨ψ| - i·|ψ⟩⟨δ|
```

where `|ψ⟩ = u/‖u‖` and `|δ⟩ = (ψ_next - ψ)/dt`. This is the *minimal
Hermitian rotation* that maps `|ψ⟩` to `|ψ_next⟩` in one step. It is
mathematically correct, but:

- It is rank 2 and **globally dense** (the matrix elements mix every
  site with every other via the outer products of the state vectors).
- It has no spatial locality. Every site couples to every site.
- Its MPO representation is bond-2 but **not in ladder form**. There is
  no identity-pass-through structure to exploit.
- Zaletel W-II fuses local MPO tensors into local gates assuming a
  spatially-local H with identity pass-through. Polar-unitarizing the
  fused tensors of a non-local H leaves an operator that is unitary
  but bears no resemblance to `exp(-iAdt)`. Measured behavior: per-step
  damping factor ~0.25 on a q=4 sine; field collapses in ~5 steps.

**Conclusion.** W-II on this `A` is not salvageable. We need a
physically motivated, spatially local `H` instead.

## 2. The dissipative/Hamiltonian tension

Burgers in viscous form:

```
  ∂_t u + u ∂_x u = ν ∂_x² u      (1)
```

is **parabolic / dissipative**. Kinetic energy `½∫u² dx` decreases
monotonically while ν > 0. A Hamiltonian generates **unitary** (norm-
preserving) evolution. There is no way to write `∂_t u = -iHu` for a
Hermitian `H` that reproduces (1): unitary evolution preserves
`‖u‖`, dissipative evolution does not.

Literature escapes:
  (L1) **Cole-Hopf**: linearize (1) into the heat equation in
       ψ = exp(-(1/2ν)∫u dx). Heat equation is still dissipative but
       linear — amenable to block-encoding or LCU. F10 pursues this.
  (L2) **LCU / block-encoding of `exp(νL·dt)`**: embed the non-unitary
       heat-flow operator into a larger unitary with ancilla and
       post-selection. Research-grade; outside F2 scope.
  (L3) **Operator splitting (Strang 1968; Chorin 1967)**: split (1)
       into a conservative transport piece and a dissipative piece,
       advance each by a well-chosen sub-step. **This is the standard
       CFD approach.** F2 Phase B.1a adopts it.

### 2.1 Split (1) as

```
  advection:    ∂_t u + u ∂_x u = 0                      (A)
  diffusion:    ∂_t u = ν ∂_x² u                         (D)
```

(A) is conservative. Linearizing about the snapshot `u_n`, the
generator `H_adv := -½i·{diag(u_n), D_x}` (symmetrized Burgers
advection) is Hermitian; `exp(-i·dt·H_adv)` is unitary and
represents physical advection.

(D) is the heat equation; non-unitary, decays high-k modes.

### 2.2 Split scheme

**Lie-Trotter** (order 1 in dt):

```
  u* = exp(-i·dt·H_adv(u_n)) · u_n       (quantum, unitary)
  u_{n+1} = u* + dt·ν·L·u*               (classical, explicit Euler
                                          on Laplacian)
```

**Strang** (order 2 in dt):

```
  u* = exp(ν·L·dt/2) · u_n               (classical half-diffusion)
  u** = exp(-i·dt·H_adv(u*)) · u*        (quantum, full advection)
  u_{n+1} = exp(ν·L·dt/2) · u**          (classical half-diffusion)
```

Phase B.1a ships Lie-Trotter first (simpler); Strang is a follow-up.

### 2.3 What a CFD reader sees

- Advection + diffusion splitting is textbook. Strang (1968) is the
  canonical reference. Chorin's fractional-step projection for
  incompressible NSE is a direct descendant.
- The quantum step advects conservatively under a Hermitian generator
  that is literally `u·∂/∂x` symmetrized. No hidden kludges.
- The diffusion step is identical to a classical FTCS update. Fully
  auditable.
- Accuracy claims are splitting-error-dominated (O(dt²) for Strang),
  not "O(dt³) from W-II". We are honest about that.

## 3. Build `H_adv` as a ladder-form MPO

Work on q qubits, N = 2^q physical grid, periodic BC.

### 3.1 Shift operator `S`

`S|k⟩ = |(k+1) mod N⟩` — binary increment. On the standard MSB-first
qubit ordering, `S` is a **bond-2 MPO**: a single bit carries the
"increment/carry" signal along the chain. Per-site tensor (physical
in→out 2×2, bond left-right 2×2):

```
  left bond | bit_in  -> bit_out  | right bond
    c = 1      0          1            0      (flip, carry absorbed)
    c = 1      1          0            1      (flip, carry propagates)
    c = 0      0          0            0      (identity)
    c = 0      1          1            0      (identity)
```

Boundary vectors close periodically (the leading carry must match the
trailing carry — this is what makes it mod N). For the LSB site,
inject `c = 1`. This is standard; see e.g. Motta et al. for the
arithmetic-MPO construction.

`S†` is the decrement MPO, obtained by reversing bond signals.

### 3.2 First derivative `D_x`

Central difference on the periodic grid:

```
  D_x = (S - S†) / (2·dx)
```

Direct MPO sum → bond-4 MPO (two bond-2 chains summed). Compress to
bond 2 or 3 via standard MPO compression (exact, no error).

`D_x` is **anti-Hermitian** (symmetric Jacobian under D.B.).

### 3.3 Diagonal of u as MPO

`diag(u)` on q qubits: factor `u` as an MPS with bond dim χ_u (depends
on u; for a pure sine χ_u = 2, for a shock χ_u ≤ O(q) in practice),
then lift to a **bond-χ_u diagonal MPO**:

```
  [diag(u)]_{kk} = u_k,  [diag(u)]_{ij} = 0 for i ≠ j
```

Per-site tensor: `T[p_in, p_out, l, r] = δ(p_in, p_out) · M[p_in, l, r]`
where `M` is the MPS tensor of u at that site.

### 3.4 Symmetrized advection Hamiltonian

```
  H_adv = -(i/2) · (diag(u_n)·D_x + D_x·diag(u_n))             (2)
```

Each product is an MPO multiplication, bond dim ≤ 2·χ_u. The sum of
two is ≤ 4·χ_u. `H_adv` is Hermitian by construction (anticommutator
of Hermitian and anti-Hermitian, times `-i/2`, is Hermitian).

Ladder form of `H_adv`: this is the hand-constructed lower-triangular
MPO for a sum of local nearest-neighbor operators. For a generic sum
`H = Σ A_j B_{j+1} + h.c.`, the canonical ladder form is

```
  W_j = [[ I       0       0  ]
         [ B_j     0       0  ]
         [ h_j    A_j      I  ]]
```

with boundary vectors v_L = [0, 0, 1], v_R = [1, 0, 0]ᵀ. For H_adv
specifically, A_j and B_j carry the diag-u/shift factors; the exact
site decomposition is what §3.5 must implement.

### 3.5 Deliverable: `build_hamiltonian_mpo_ladder`

Add to `burgers_tebd.py`:

```python
def build_hamiltonian_mpo_ladder(
    u: np.ndarray,
    dx: float,
    bc: str = "periodic",
) -> qtn.MatrixProductOperator:
    """Ladder-form MPO for H_adv = -(i/2){diag(u), D_x}.

    Bond dim scales as O(chi_u) where chi_u is the MPS bond dim of u.
    For periodic BC only in this spec; Dirichlet TBD.

    Returns a quimb MPO whose per-site tensors are in lower-triangular
    ladder form with identity pass-through, suitable for Zaletel W-II
    gate fusion.
    """
```

The existing `build_hamiltonian_dense` stays for reference and for
the Phase A classical TEBD (which continues to work). The new
function is consumed only by the W-II / tebd_circuit path.

## 4. Wire into the W-II layer

### 4.1 `build_wii_layer` currently takes a dense H

Change `tebd_circuit_step` (burgers_trotter.py L538-644) to:

1. Call `build_hamiltonian_mpo_ladder(u, dx, bc=bc)` → MPO H_adv.
2. Pass the MPO directly to `build_wii_layer` (new signature taking
   MPO instead of dense matrix; old dense path kept as fallback for
   unit tests).
3. After the quantum sub-step decodes `u_half`, apply **classical
   half-diffusion** twice (Strang) or one full classical diffusion
   step (Lie-Trotter) on the physical grid. Use the existing
   `compute_rhs_shift` for the Laplacian piece only.

### 4.2 `build_wii_layer` for ladder MPO

`build_wii_layer_ladder(H_mpo, dt)` — Zaletel's construction on a
spatially local H. Per-site gate dim = 2·D where D = MPO bond dim.
Unit tests:

- Accepts a bond-2 MPO (e.g., `H = X⊗I + I⊗X` on 2 qubits).
- Reconstructs exp(-iHdt) to frobenius err < 10·dt³.
- Passes the X2 unitarity assertion (err < 1e-10).

## 5. Classical diffusion update

Reuse `compute_rhs_shift` from `burgers_nonlinear.py` but strip the
advection term. New helper:

```python
def diffusion_rhs(
    u: np.ndarray, dx: float, nu: float, bc: str = "periodic",
) -> np.ndarray:
    """Returns ν·∂²u/∂x² (the Laplacian piece only)."""
```

Then:

```python
u_next = u_half + dt * diffusion_rhs(u_half, dx, nu, bc=bc)
```

for Lie-Trotter. Strang wraps it with half-step diffusion before and
after.

## 6. Acceptance criteria

Ordered from strictest to softest. All at q=4, dt=1e-4, nu=1e-2,
periodic BC, sine IC.

### 6.1 Unitarity (hard)

```
  ‖U·U† - I‖_F < 1e-10
```

where U is the full circuit unitary for one step (advection only).
Inherited from X2; must still pass.

### 6.2 Advection-only accuracy (hard)

With ν = 0 and a traveling sine `u(x, 0) = sin(2π(x - c t))` for
small constant c (set u_n = c for the generator), run the quantum
advection step alone and compare against the exact translated sine.
Expect err_F = C·dt³ + O(dt⁴) with fitted exponent p ∈ [2.8, 3.2]
over dt ∈ [1e-4, 1e-3].

This is the real O(dt³) check that X2 could not meet. With a
physically local `H_adv`, Zaletel W-II should deliver it.

### 6.3 Full-step accuracy with Lie-Trotter (soft)

Compare `tebd_circuit` (advection quantum + diffusion classical)
against `shift-Euler` over 200 steps at q=4, dt=1e-4, nu=1e-2, sine
IC. Expect:

```
  max|u_tebd_circuit - u_shift| < 5e-2         (absolute)
  max relative error < 5%                       (normalized)
  ‖u_tebd_circuit‖ stable (no amplitude collapse)
```

### 6.4 Stability through shock (soft)

Run to t = 0.2 (past shock time ≈ 0.16). The field must remain
finite and non-degenerate — no collapse to the zero vector, no NaN.

### 6.5 Animation reproduction

Regenerate `tebd_circuit_comparison_q4_shock.gif` via
`animate_tebd_comparison.py --q 4 --steps 1800 --include-circuit`.
The green line should now track the shift/TEBD lines through the
shock, with visible small deviation from splitting error, not a
dying-amplitude collapse.

## 7. Out of scope

- Full quantum diffusion via LCU / block-encoding. Rejected for F2;
  belongs in F10 Phase B or a new F12.
- Quantum Imaginary Time Evolution (QITE). Same reason.
- Hadamard-test sign recovery for `tebd_circuit`. F2 Phase C.
- Dirichlet BC for the ladder MPO. Open (will need boundary bond
  closure logic beyond the periodic case).
- Non-power-of-2 N. Unchanged; we inherit the existing log₂(N) = q
  constraint.

## 8. Code surface

**New** (burgers_tebd.py):
- `build_hamiltonian_mpo_ladder(u, dx, bc) -> MPO`
- `build_wii_layer_ladder(H_mpo, dt) -> list[np.ndarray]`
- `build_shift_mpo(q, direction=+1) -> MPO` (bond-2 increment)
- Optional: `build_laplacian_mpo(q, dx) -> MPO`

**Modified** (burgers_tebd.py):
- Phase B.1 validation block (L~659-776): swap the rank-2 `A` for
  `build_hamiltonian_mpo_ladder`. Re-run unitarity + O(dt³) checks.

**Modified** (burgers_trotter.py):
- `tebd_circuit_step` (L538-644): call the MPO-taking W-II layer,
  append classical diffusion update.

**New** (burgers_nonlinear.py):
- `diffusion_rhs(u, dx, nu, bc) -> np.ndarray` (pure Laplacian).

**Unchanged**:
- `run_tebd_simulation` / classical `tebd` path. That still uses
  dense H → MPO via `from_dense`. Works fine for the classical path.
- `build_hamiltonian_dense` stays as a private helper. Marked
  "used by Phase A only" in its docstring.

## 9. Validation plan (for reviewer = this session)

Before accepting the agent's PR I will check:

1. **Hermiticity**: assert `‖H_adv - H_adv†‖_F / ‖H_adv‖_F < 1e-12`
   on a random u at q=3,4,5.
2. **MPO bond dims** printed at every step; cap at 4·χ_u; flag if
   W-II gate dim exceeds 16 (hardware-relevance sanity).
3. **Advection-only Fourier test (§6.2)** passes with fitted slope
   ∈ [2.8, 3.2].
4. **Energy monotonicity**: `½Σu² decreases` under the full split
   step (since advection conserves energy exactly and diffusion is
   dissipative).
5. **Mass conservation** (periodic BC): `Σu` unchanged by advection
   step to machine precision; changes only via source terms.
6. The animation regenerated in §6.5 looks physical.

## 10. References

- Strang, G. (1968). *On the construction and comparison of
  difference schemes.* SIAM J. Numer. Anal. 5(3), 506–517.
- Chorin, A. (1967). Numerical method for solving incompressible
  viscous flow problems. *J. Comp. Phys.* 2, 12–26.
- Zaletel, M. et al. (2015). Time-evolving a matrix product state
  with long-ranged interactions. *Phys. Rev. B* 91, 165112.
- Hatano, N.; Suzuki, M. (2005). Finding exponential product
  formulas of higher orders. *Quantum Annealing and Other
  Optimization Methods*, 37–68. (Strang ≡ S₂.)
- Motta, M.; Ceperley, D. et al. arithmetic MPO constructions for
  the increment/decrement operator (multiple refs; any standard
  TN textbook chapter on MPOs).
- F2-IMPLEMENTATION-SPEC.md — parent spec.
- burgers_tebd.py L78-120 — current rank-2 `A`; replaced here.
