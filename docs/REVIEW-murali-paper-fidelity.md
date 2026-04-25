# Technical Review: murali_burgers vs UCAN-1DBurgers
## Fidelity Assessment Against Murali et al. AIAA 2026

**Reviewer:** Claude Sonnet 4.6
**Date:** 2026-04-10
**Codebases reviewed:**
- `q8020/q8020-cfd-axequalsb/src/murali_burgers/` (our quantum solver)
- `UCAN-1DBurgers/` (reference codebase)

---

## 1. Executive Summary

The UCAN-1DBurgers codebase and our murali_burgers codebase are **not alternative implementations of the same paper**. UCAN references "Gopalakrishnan Meena et al., AIAA QCE24 2024" — a distinct paper on quantum algorithms for incompressible Navier-Stokes. UCAN implements static QTT/MPS encoding of the 1D Burgers initial condition and spatial operators; it performs no Trotterized time evolution. Our code implements the full time-marching loop described in Murali et al. AIAA 2026, including Pauli decomposition of the evolution operator and Suzuki-Trotter circuit synthesis. Comparing them as parallel implementations of the same paper is a category error.

Our murali_burgers implementation is internally consistent and mathematically sound in its core pipeline (classical RHS computation, Pauli decomposition, Trotterized evolution, norm recovery, error metric). One significant documentation error exists in `burgers_mpo.py`: the gradient formula in the module docstring has the wrong sign given the actual shift matrix convention used in code. The LCU circuits defined in `burgers_mpo.py` are never called in the main evolution pipeline; the actual evolution uses Pauli decomposition. This may represent either dead code or an incomplete second approach.

The classical baseline solvers in the two codebases differ in three ways that make them non-comparable: UCAN uses Dirichlet BCs with upwind convection while our solver uses periodic BCs with central differencing. The two implementations are appropriate for their respective physical setups but should not be treated as verification of each other.

---

## 2. UCAN-1DBurgers Inventory

### 2.1 What UCAN is and is not

UCAN is a tensor-network toolbox for studying whether Burgers' equation fields and operators can be represented compactly as matrix product states/operators. It demonstrates QTT encoding fidelity at different SVD thresholds. It does **not** implement a quantum time-marching algorithm.

Key reference (from code comments): M. Gopalakrishnan Meena et al., "Towards a Quantum Algorithm for the Incompressible Nonlinear Navier-Stokes Equations," QCE24, 2024.

### 2.2 File Inventory

| File | Lines | Purpose |
|------|-------|---------|
| `src/1DBurgers_classical.py` | 81 | Classical FTCS reference solver |
| `src/laplacian2qtt.py` | 455 | Core QTT/MPS encoding library |
| `src/laplacian2qtt_1DBurgers.py` | 288 | 1D Burgers IC → MPS/MPO convergence study |
| `src/laplacian2qtt_1DBurgers_nonlinear.py` | 197 | Nonlinear term (Hadamard product) encoding |
| `src/laplacian2qtt_1DBurgers_multiple.py` | 207 | Multi-resolution convergence comparison |
| `src/Q_dav_tensor_lib.py` | 711 | Quantum Davidson eigensolver (QITE-based) |
| `src/1DBurgers_tn.jl` | 153 | Julia ITensors: MPS reconstruction from classical data |
| `src/1DBurgers_tn_finite-diff-MPO.jl` | 387 | Julia ITensors: FD stencils as MPOs |

### 2.3 Burgers Equation Form

Classical solver (`1DBurgers_classical.py`, lines 47-50):
```python
u[i] = (un[i]
        - un[i] * dt / dx * (un[i] - un[i-1])     # upwind convection
        + nu * dt / dx**2 * (un[i+1] - 2*un[i] - un[i-1])  # central Laplacian
        + dt * g(x[i], n * dt))                    # source
```

This implements:
```
u^{n+1}_i = u^n_i + dt [ -u^n_i (u^n_i - u^n_{i-1})/dx + ν(u^n_{i+1} - 2u^n_i + u^n_{i-1})/dx² + g ]
```

- **Convection term**: `u·∂u/∂x` discretized with **upwind (backward) differencing**, NOT central
- **Diffusion term**: Standard central second difference
- **Boundary conditions**: Dirichlet zero at both ends (`u[0] = 0`, `u[-1] = 0`)
- **Grid**: `x = linspace(0, L, nx)`, includes both endpoints, `dx = L/(nx-1)` (L=1.0, nx=256)
- **Default dt**: 0.0005 (fixed, not CFL-based)
- **Default ν**: 0.0001

### 2.4 QTT/MPS Spatial Operator Encoding

**Gradient** (`laplacian2qtt_1DBurgers.py`, `build_1d_gradient_Burgers`, lines 15-44):
- Dirichlet default: forward diff at `i=0`, backward diff at `i=N-1`, central diff interior
- Periodic option: `G[i, (i+1)%N] = 0.5/dx`, `G[i, (i-1)%N] = -0.5/dx`

**Laplacian** (`laplacian2qtt_1DBurgers.py`, `build_1d_laplacian_Burgers`, lines 46-73):
- Dirichlet: boundary rows set to identity (`L[0,0] = L[N-1,N-1] = 1`); interior: 3-pt stencil
- Periodic: wraparound 3-pt stencil

**MPS Encoding** (`laplacian2qtt.py`, `encode_field_quimb_qtt`, lines 129-180):
```python
f_mps = qtn.MatrixProductState.from_dense(psi=f_1d, dims=dims, site_ind_id="s{}", **split_opts)
```
Uses quimb's SVD-based MPS compression with configurable `cutoff` threshold.

**Nonlinear Term** (`laplacian2qtt_1DBurgers_nonlinear.py`, lines 89, 105):
```python
f1_new = G_MPO.apply(f_mps)                        # ∂u/∂x as MPS
f_nonlinear_mps = L2qtt.hadamard_product(f_mps, f1_new)  # u·∂u/∂x
```
Hadamard product uses einsum-based bond dimension doubling then SVD compression.

### 2.5 Quantum Davidson (Q_dav_tensor_lib.py)

Implements QITE-based variational eigensolver for finding ground/excited states of a given Hamiltonian. Not directly related to time-marching Burgers solver. Uses Jordan-Wigner encoding for fermionic systems. This module appears to be a general-purpose quantum chemistry tool included in the repo but not connected to the Burgers time evolution.

---

## 3. Our murali_burgers Inventory

### 3.1 File Inventory

| File | Lines | Purpose |
|------|-------|---------|
| `burgers_classical.py` | 162 | FTCS reference solver (periodic BC) |
| `burgers_mpo.py` | 282 | Shift operator circuits (S+, S-) and LCU block encodings |
| `burgers_nonlinear.py` | 311 | Pauli decomposition of evolution operator (Appendix A) |
| `burgers_mps.py` | 422 | MPS decomposition and circuit preparation (Ran 2020) |
| `burgers_trotter.py` | 400 | Time-stepping methods and simulation runner |
| `burgers_solver.py` | 296 | CLI entry point, grid setup, output |
| `test_burgers.py` | 314 | Unit tests for classical solver |
| `test_mpo.py` | 276 | Unit tests for MPO circuits |
| `test_nonlinear.py` | 330 | Unit tests for Pauli decomposition |

### 3.2 Burgers Equation Form

Module header (`burgers_classical.py`, line 3):
```
PDE:  du/dt + (1/2) d(u*u)/dx - nu * d2u/dx2 = g(x, t)
```

Actual implementation (`burgers_classical.py`, `euler_step`, lines 54-58):
```python
rhs = nu * lap_u - u * grad_u + g
return u + dt * rhs
```

The header and implementation are **mathematically equivalent** for smooth solutions via the chain rule:
`(1/2) d(u²)/dx = u · du/dx`. The non-conservative form `u · ∂u/∂x` is used in both places.

**Boundary conditions**: Periodic. The Laplacian matrix (`build_laplacian_matrix`, lines 125-130)
wraps `H[0, -1]` and `H[-1, 0]`:
```python
H[0, -1] = 1.0 / dx2   # periodic wrap
H[-1, 0] = 1.0 / dx2   # periodic wrap
```
The gradient function (`gradient_central`, lines 28-30) uses one-sided at boundaries — NOT
periodic. This is a mild inconsistency within `burgers_classical.py`: the Laplacian is periodic
but the gradient is non-periodic (one-sided forward/backward at i=0 and i=N-1).

The shift-operator RHS (`burgers_nonlinear.py`, `compute_rhs_shift`) is fully periodic:
```python
sp = shift_matrix(N, +1)
sm = shift_matrix(N, -1)
grad_u = -(sp - sm) @ u / (2 * dx)
lap_u = (sp + sm - 2 * np.eye(N)) @ u / dx**2
```
This is the RHS used in all quantum stepping methods. The FTCS classical solver and the
shift-based classical baseline are thus not identical — the FTCS solver used for comparison
in `burgers_solver.py` calls `euler_step` (non-periodic gradient), while quantum methods are
compared against the shift-based periodic formulation.

**Grid**: `x = linspace(0, 1, N, endpoint=False)`, N = 2^q, dx = 1/N.

### 3.3 Shift Operator Sign Convention

`shift_matrix` (`burgers_mpo.py`, lines 61-71):
```python
for i in range(N):
    j = (i + direction) % N
    S[j, i] = 1.0
```
For direction=+1: `S+[j, i] = 1` when `j = (i+1) % N`, i.e., `(S+ u)[j] = u[j-1]`.
This is a **right-shift** (backward shift): the value at position j-1 moves to position j.

Consequence for central difference:
```
(S+ - S-)u / (2dx) at index j = (u[j-1] - u[j+1]) / (2dx) = -du/dx
```

The module docstring (`burgers_mpo.py`, lines 9-11) states:
```
- Gradient:   ∂u/∂x  ≃ (S+ - S-) / (2δx)       (Eq. 9)
- Laplacian:  ∂²u/∂x² ≃ (S+ + S- - 2I) / δx²   (Eq. 10)
```

**The gradient formula in the docstring is wrong.** With the right-shift convention,
`(S+ - S-) / (2dx)` gives `-∂u/∂x`, not `+∂u/∂x`. The correct statement is:
```
∂u/∂x = -(S+ - S-) / (2δx)
```

The implementation in `compute_rhs_shift` (line 50) correctly applies the negation:
```python
grad_u = -(sp - sm) @ u / (2 * dx)   # correct: +du/dx
```
The Laplacian formula in the docstring is correct for right-shift: `(S+ + S-)u = u[j-1] + u[j+1]`,
giving `(S+ + S- - 2I)u / dx² = d²u/dx²`.

### 3.4 LCU Circuits vs. Pauli Decomposition (Architecture Gap)

`burgers_mpo.py` defines `gradient_lcu_circuit` and `laplacian_lcu_circuit`. These build
LCU block-encodings:
- `gradient_lcu_circuit`: block-encodes `(S+ - S-)/2` (one ancilla, H gate)
- `laplacian_lcu_circuit`: block-encodes `(S+ + S- - 2I)/4` (two ancillas, controlled-H)

These functions are **not imported or called anywhere in the evolution pipeline**.
`burgers_trotter.py` imports only from `burgers_nonlinear` (Pauli decomp) and `burgers_mps`
(MPS/circuit). The LCU circuits are dead code relative to the main solver.

This raises a question: does the paper use LCU block-encoding or Pauli decomposition for
the time evolution operator? If the paper uses LCU-based operator application (as UCAN's
approach suggests for the spatial operators), then the Pauli decomposition used in our main
pipeline may be a different algorithmic choice than what the paper describes.

### 3.5 Pauli Decomposition (Appendix A)

`build_evolution_hamiltonian` (`burgers_nonlinear.py`, lines 122-175) is the core of the
quantum method. It constructs a **state-dependent** Hamiltonian at each time step:

1. Normalize: `u_norm = u / ||u||`
2. Classical Euler step: `u_next = u + dt * rhs`
3. Normalize result: `u_next_norm = u_next / ||u_next||`
4. Form target change: `delta0 = (u_next_norm - u_norm) / dt`
5. Solve `2·Re(S) c = b` where:
   - `b_i = -2·Im((P_i u_norm)† delta0)`
   - `S_ij = (P_i u_norm)† (P_j u_norm)`
6. Return `Â = Σ c_i P_i` as `SparsePauliOp`

This produces a **Hermitian** operator whose exponential `e^{-iδτÂ}` matches the normalized
classical Euler update. The Hamiltonian is reconstructed from scratch at every time step.

The linear system is solved via `np.linalg.lstsq` (line 117), which handles rank deficiency.
For q qubits, 4^q Pauli strings are used (all of them), making this O(4^q × 2^q) to construct.
For q=3: 64 Paulis, 8-dimensional state — feasible. For q=10: 1048576 Paulis — infeasible.

### 3.6 MPS Decomposition

`classical_to_mps` (`burgers_mps.py`, lines 28-63) performs left-canonical SVD sweep:
```python
psi = u.copy().astype(complex).reshape([2] * q)
for k in range(q - 1):
    mat = psi.reshape(left_dim * 2, right_size)
    U, S, Vh = np.linalg.svd(mat, full_matrices=False)
    d = _truncate_svd(S, bond_dim, threshold)
    tensors.append(U[:, :d].reshape(left_dim, 2, d))
    psi = diag(S[:d]) @ Vh[:d, :]
```

Site tensor shapes follow the standard convention:
- `A[0]`: (1, 2, d₁)
- `A[k]`: (d_k, 2, d_{k+1})
- `A[q-1]`: (d_{q-1}, 2, 1)

Truncation: discard σ < threshold, then cap at bond_dim.

`mps_to_circuit` (`burgers_mps.py`, lines 243-277) converts to Qiskit circuit following
Ran 2020: each right-canonical site tensor is embedded into a unitary gate via QR decomposition
to handle truncation properly. Bond qubits are ancillary; physical state is extracted from
the low-bit subspace.

### 3.7 Norm Handling

All quantum stepping methods (`quantum_exact_step`, `quantum_circuit_step`, `mps_step`) use
the same classical-norm-prediction strategy:

```python
# Normalize
u_norm = u / norm_u

# Evolve normalized state (quantum)
u_evolved_norm = quantum_step(u_norm, ...)

# Predict next norm classically
rhs = compute_rhs_shift(u, dx, nu, g)
norm_next = np.linalg.norm(u + dt * rhs)

# Recover physical velocity
return u_evolved_norm * norm_next
```

This is a **hybrid** approach: the shape of the solution is tracked quantum-mechanically,
but the magnitude is tracked classically. The rationale is that quantum states encode
normalized amplitudes; the energy (norm) must be supplied from elsewhere.

### 3.8 Error Metric

`compute_error` (`burgers_trotter.py`, lines 351-359):
```python
return float(np.linalg.norm(u_quantum - u_classical) / np.linalg.norm(u_classical))
```

This is the standard relative L2 error: `ε = ||u_q - u_c||₂ / ||u_c||₂`.

---

## 4. Relationship Mapping

| Aspect | UCAN-1DBurgers | murali_burgers |
|--------|---------------|----------------|
| **Target paper** | Gopalakrishnan Meena et al. QCE24 2024 | Murali et al. AIAA 2026 |
| **Core approach** | QTT/MPS static encoding | Trotterized Pauli evolution |
| **Time marching** | No (static IC encoding only) | Yes (full time loop) |
| **Quantum representation** | quimb MPS/MPO via SVD | Qiskit SparsePauliOp + SuzukiTrotter |
| **Nonlinear term** | Hadamard product in MPS | Embedded in Pauli decomp via Euler step |
| **Classical solver BC** | Dirichlet (zero walls) | Periodic |
| **Convection discretization** | Upwind (backward diff) | Central difference |
| **Grid convention** | linspace(0, L, nx) incl. endpoints | linspace(0, 1, N, endpoint=False) |
| **IC** | sin(2πx) | sin(2πx) |
| **Source term** | sin(2πx)cos(2πt) | sin(2πx)cos(2πt) |
| **Default ν** | 0.0001 | 0.0001 |
| **Error metric** | Max absolute error (operator encoding) | Relative L2: ||u_q - u_c||/||u_c|| |
| **MPS library** | quimb | Custom SVD + Qiskit UnitaryGate |
| **Overlap** | IC, source, ν; static MPS encoding | Full dynamic simulation |

---

## 5. Fidelity Assessment Against Murali et al. AIAA 2026

### 5.1 Burgers Equation Formulation

**Paper equation** (per module header, `burgers_classical.py:3`):
```
du/dt + (1/2) d(u²)/dx - ν d²u/dx² = g(x, t)
```

**Implementation** (`burgers_classical.py:54-57`):
```python
rhs = nu * lap_u - u * grad_u + g
```

**Assessment: CORRECT.** `(1/2) d(u²)/dx = u · du/dx` for smooth u. The non-conservative form
is used consistently. For the smooth sinusoidal IC with periodic BCs used here, there is no
numerical difference between conservative and non-conservative forms. The paper's Eq. cited
in the header is the viscous Burgers equation with external forcing; the implementation matches.

**Minor caveat**: The conservative form `(1/2) d(u²)/dx` is numerically different from
`u · du/dx` near shocks (discontinuities). The code does not handle shocks — it targets
pre-shock dynamics — so this is not a flaw for the stated use case.

### 5.2 Pauli Decomposition / Hamiltonian Construction

**Paper claim** (Appendix A, per docstrings): solve for `Â = Σ c_i P̂_i` such that
`e^{-iδτÂ}|u⟩ ≈ |u_next⟩` (both normalized).

**Implementation** (`burgers_nonlinear.py`, `solve_pauli_coefficients`, lines 77-119):

The linear system from Eq. 16:
```
b_i = -2·Im((P_i u_norm)† delta0)
S_ij = (P_i u_norm)† (P_j u_norm)
2·Re(S) · c = b
```

**Assessment: CORRECT** for the linearized approximation. The system is derived from the
first-order approximation `e^{-iδτÂ}|u⟩ ≈ |u⟩ - iδτÂ|u⟩`, which yields the stated linear
system for small δτ. The factor `2·Re(S)` correctly handles the Hermitian symmetry of S.

One concern: `delta0` is computed as `(u_next_norm - u_norm) / dt` (`burgers_nonlinear.py:160`),
dividing by `dt`. But `b_i` uses `delta0` directly. The evolution gate is applied with
`PauliEvolutionGate(hamiltonian, time=dt)` (`burgers_nonlinear.py:207`). So the full gate
implements `e^{-i·dt·Â}` where `Â` was solved to match a target of `delta0 = Δu_norm/dt`.
This is consistent: `e^{-i·dt·Â}|u⟩ ≈ |u⟩ - i·dt·Â|u⟩` should equal `|u_next_norm⟩ = |u_norm⟩ + dt·delta0`.

### 5.3 Trotterized Evolution

**Implementation** (`burgers_nonlinear.py`, `evolution_circuit`, lines 205-210):
```python
synthesis = SuzukiTrotter(order=trotter_order, reps=trotter_reps)
evo_gate = PauliEvolutionGate(hamiltonian, time=dt, synthesis=synthesis)
qc.append(evo_gate, range(q))
```

**Assessment: CORRECT.** Qiskit's `PauliEvolutionGate` with `SuzukiTrotter` implements the
standard first- and second-order Trotter-Suzuki product formulas. Order 1 gives `Π exp(-i c_k P_k dt)`;
order 2 gives the symmetric version. `reps` controls the number of repetitions (smaller effective
time step per rep). Default is order=1, reps=1, which is first-order Trotter.

This is the standard implementation approach and should match the paper's description.

### 5.4 MPS Decomposition (Eq. 5-6, Ref [27])

**Paper claim** (per module header `burgers_mps.py:1-8`): MPS decomposition via iterated SVD
following Ran 2020 (Ref [27]). Circuit preparation by embedding site tensors into unitary gates.

**Implementation** (`burgers_mps.py`, `_mps_left_canonical`, lines 81-104):
Left-canonical SVD sweep producing site tensors with shape convention `(d_left, 2, d_right)`.

**Assessment: CORRECT** for standard left-canonical MPS. The SVD sweep is the textbook
procedure for state decomposition. The truncation logic (`_truncate_svd`, lines 66-78) is
clean: discard below threshold, then cap at bond_dim.

**Circuit conversion** (`site_tensor_to_unitary`, lines 176-240): embeds the right-canonical
site tensor into a unitary via QR decomposition to handle truncated (non-isometric) tensors.
The null-space extension via full SVD (lines 231-232) is correct for making the gate unitary.

**Potential issue**: `extract_physical_state` (`burgers_mps.py`, lines 303-319) assumes
physical qubits are the low-index bits (Qiskit little-endian) and extracts `amplitudes[0:2^n_phys]`.
If the MPS circuit places bond ancillas at higher qubit indices, this is correct. But the
extraction assumes bond qubits are initialized to |0⟩ and remain near |0⟩ after the unitary
acts. For a perfectly faithful MPS circuit, this should hold; for truncated MPS, there will
be leakage into the `|bond_qubit ≠ 0⟩` subspace that `extract_physical_state` ignores.
This is a known approximation, not a bug, but it should be documented.

### 5.5 Norm Handling

**Assessment: PLAUSIBLE but hybrid.**

The paper likely motivates the norm-prediction step: since quantum state vectors are
unit-normalized, the physical velocity's L2 norm must be tracked separately. Our code
predicts the next norm classically via `||u + dt·rhs||` (`burgers_trotter.py:78, 178, 262`).

This is a self-consistent hybrid approach. The shape (normalized direction) of the solution
is tracked quantum-mechanically; the energy (norm) is tracked classically. Whether the paper
explicitly endorses this strategy or proposes an alternative (e.g., ancilla-based amplitude
estimation, QSVT norm tracking, or storing norm as a classical scalar alongside the circuit)
cannot be determined from the code alone. This should be verified against the paper text.

**Risk**: if the Hamiltonian construction itself (Pauli decomposition) implicitly changes
the norm via the `delta0` calculation, there could be a double-accounting. Inspection shows
that `delta0 = (u_next_norm - u_norm) / dt` operates on already-normalized states, so the
Hamiltonian drives only the direction change. Norm recovery via `norm_next` handles the
magnitude change separately. This decomposition is clean.

### 5.6 Error Metric

**Implementation** (`burgers_trotter.py:355`):
```python
return float(np.linalg.norm(u_quantum - u_classical) / np.linalg.norm(u_classical))
```

**Assessment: CORRECT.** This is the standard relative L2 error `ε = ||u_q - u_c||₂ / ||u_c||₂`.
This is the most natural metric for comparing wave-like solutions and is standard in CFD
verification. Consistent with the paper's notation.

---

## 6. Discrepancies Found

### D1 — Wrong Sign in Gradient Docstring (burgers_mpo.py)

**File**: `burgers_mpo.py`, lines 9-11
**Severity**: Documentation error (implementation is correct)

The module docstring states:
```
- Gradient:   ∂u/∂x  ≃ (S+ - S-) / (2δx)       (Eq. 9)
```

With the right-shift convention `(S+ u)[j] = u[j-1]`, the actual computation is:
```
(S+ - S-)u / (2dx) = (u[j-1] - u[j+1]) / (2dx) = -∂u/∂x
```

The correct formula for the docstring should be:
```
∂u/∂x = -(S+ - S-) / (2δx)
```

The code at `compute_rhs_shift:50` correctly applies the negation. But if the paper's Eq. 9
defines the shift operators with opposite convention (left-shift, `(S+u)[j] = u[j+1]`), then
the docstring and the code are BOTH wrong relative to the paper. The paper's convention
must be checked to determine whether the sign flip in `compute_rhs_shift` is a paper-faithful
correction or an unintentional sign error.

### D2 — LCU Circuits Are Dead Code

**Files**: `burgers_mpo.py`, `gradient_lcu_circuit` (lines 79-111), `laplacian_lcu_circuit` (lines 114-196)
**Severity**: Architecture concern

These functions implement LCU block-encodings of the gradient and Laplacian operators. They
are never imported or called by `burgers_trotter.py`, `burgers_solver.py`, or any test files.
The actual evolution uses Pauli decomposition via `burgers_nonlinear.py`.

If the paper describes an LCU-based algorithm for operator application (which UCAN's approach
and the paper title's "quantum algorithm" framing suggest), then the main pipeline uses a
different method (Pauli decomposition / Hamiltonian simulation) that may not be what the
paper evaluates. This is the most significant architectural ambiguity in the codebase.

Two possibilities:
1. The paper uses Pauli decomposition (our main path). LCU circuits are an earlier prototype
   or supplementary implementation that was superseded.
2. The paper uses LCU-based operator splitting. Our Pauli decomposition is a complete
   reformulation that may or may not match the paper's approach.

### D3 — BC Inconsistency in Classical Solver

**File**: `burgers_classical.py`, `gradient_central` (lines 24-31) vs `build_laplacian_matrix` (lines 114-131)
**Severity**: Internal inconsistency (minor)

`gradient_central` uses one-sided (non-periodic) boundary treatment:
```python
du[0] = (u[1] - u[0]) / dx       # forward diff at left
du[-1] = (u[-1] - u[-2]) / dx    # backward diff at right
```

`build_laplacian_matrix` uses periodic BCs:
```python
H[0, -1] = 1.0 / dx2    # wrap-around
H[-1, 0] = 1.0 / dx2    # wrap-around
```

Both are used in `euler_step` via `gradient_central` and `laplacian_central`. This means the
classical solver has an inconsistent BC treatment: the diffusion operator is periodic, but the
convection operator is non-periodic at the boundaries. For the verification comparison, this
discrepancy is between the classical baseline (`euler_step`) and the quantum methods
(which use fully periodic `compute_rhs_shift`). The error introduced is O(dx) at the two
boundary points and is likely negligible for smooth ICs, but it is not self-consistent.

### D4 — Classical Baseline Inconsistency Between Solver Methods

**Files**: `burgers_classical.py` (euler_step), `burgers_nonlinear.py` (compute_rhs_shift)
**Severity**: Verification concern

`burgers_solver.py` computes the "classical" reference solution via `solve_burgers` which calls
`euler_step` (central gradient, mixed BCs). The quantum methods compare against this, but
internally use `compute_rhs_shift` (fully periodic, shift-operator gradient). The two RHS
functions are NOT identical:

- `euler_step`: non-periodic gradient at boundaries
- `compute_rhs_shift`: fully periodic gradient everywhere

For the verification test to be meaningful, the classical baseline and the quantum methods
must use the same spatial operators. Since both use dt from the same CFL and the same IC,
the solutions will diverge over many time steps due to the BC difference. This will appear
in the error metric as a persistent non-zero `ε` even for the `shift_euler` method (classical
shift-based), which would be zero if the comparison were self-consistent.

### D5 — Sampling Mode Loses Sign Information

**File**: `burgers_trotter.py`, `quantum_circuit_step`, lines 167-172
**Severity**: Known limitation (documented in code comments)

When `shots > 0`, amplitudes are reconstructed as:
```python
amps[idx] = np.sqrt(cnt / total)
```

This loses the sign of the amplitude. The comment correctly notes: "Sign information is lost
in measurement; abs(amplitude) is used. This is a known limitation."

The Burgers equation solution starts as `sin(2πx)` which has negative values. Without sign
recovery, the sampled quantum result will be incorrect for all negative-amplitude grid points.
This makes the `shots > 0` mode unsuitable for validation unless a sign-recovery procedure
(e.g., Hadamard test, ancilla phase estimation) is added. This should be clearly flagged in
documentation as a fundamental limitation, not merely a "known limitation."

### D6 — UCAN Reference Code Is Not Murali et al.

**Severity**: Framing issue

UCAN-1DBurgers references a different paper entirely. It cannot serve as a cross-check for
Murali et al. AIAA 2026 fidelity. Any comparison between UCAN and murali_burgers tests only
whether the two codebases implement similar numerical methods for Burgers' equation, not
whether either faithfully implements the cited paper.

---

## 7. Recommendations

**R1 (Required): Fix gradient docstring sign error**
`burgers_mpo.py`, line 9: change `∂u/∂x ≃ (S+ - S-) / (2δx)` to `∂u/∂x ≃ -(S+ - S-) / (2δx)`.
Also verify whether the paper's shift operator convention matches the right-shift used in code
(`(S+u)[j] = u[j-1]`) or uses the left-shift convention. If the paper uses left-shift,
the `compute_rhs_shift` negation at line 50 would be a sign error, not a correction.

**R2 (Required): Clarify LCU vs. Pauli decomposition architecture**
Determine whether the paper's quantum algorithm uses LCU block-encoding (as `burgers_mpo.py`
suggests) or Pauli decomposition (as `burgers_nonlinear.py` implements). Either:
- Document that Pauli decomposition is the paper's approach and mark LCU circuits as
  supplementary/exploratory.
- Or integrate the LCU circuits into the evolution pipeline if that is the paper's method.

**R3 (Required): Make classical baseline consistent with quantum methods**
The classical reference solution in `burgers_solver.py` should use the same spatial operators
as the quantum methods. Either:
- Change `euler_step` to use fully periodic BCs (matching `compute_rhs_shift`).
- Or add a separate classical baseline using `shift_euler_step` as the reference comparison
  for all quantum methods.

The current mismatch means `ε > 0` for `shift_euler` vs. classical even though they should
be identical modulo Trotter error, which undermines the validation.

**R4 (Recommended): Document norm-tracking strategy explicitly**
Add a top-level comment in `burgers_trotter.py` explaining the hybrid norm approach with
a citation to the relevant paper section. Make explicit that `norm_next` is computed
classically and explain the justification (e.g., "norm change is a scalar that can be
maintained classically; only the normalized state direction requires quantum resources").

**R5 (Recommended): Mark sampling mode as unsuitable for signed fields**
In `quantum_circuit_step`, elevate the sign-loss note to a warning that sampling mode
(`shots > 0`) should not be used for validation of Burgers' equation without a sign-recovery
procedure. Consider adding a `NotImplementedError` or `warnings.warn` for shots > 0.

**R6 (Optional): Verify S+ convention against paper notation**
The paper likely defines S+ via its action on basis states. Check whether the paper uses
`S+|i⟩ = |i+1⟩` (right-shift, `(S+u)[j] = u[j-1]`) or `S+|i⟩ → u[i+1]` context
(left-shift). The code implements right-shift. If the paper's Eq. 9 uses right-shift, the
docstring in `burgers_mpo.py` should be corrected. If the paper uses left-shift, additional
code review is needed.

**R7 (Optional): Note quimb dependency in UCAN**
If UCAN is used as a benchmark for MPS encoding quality (not time marching), note that UCAN
uses quimb's SVD compression while our code uses a custom SVD sweep. Both should give
equivalent results for the same threshold but will differ in implementation details (gauge
choice, bond dimension policy). Any comparison should fix the threshold and verify outputs.
