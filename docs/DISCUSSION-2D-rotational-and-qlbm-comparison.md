# DISCUSSION — Extending to 2D, the rotational nonlinearity, and MPS-vs-QLBM comparison

*Design discussion, 2026-05-25. No code was written; this captures the
reasoning and the conclusions reached so they can inform later work.
Companion to [`OVERVIEW-burgers-solver.md`](OVERVIEW-burgers-solver.md)
(note: that overview is stale — it predates the F11 QLBM methods).*

## Context

`q8020-mps-burgers` is one package within a broader study of quantum
computing for CFD/PDE solving, structured along three axes:
**PDE case × code/algorithm × backend**. Individual deliverables are cells
in that matrix. The shared `solverfw` harness (in `q8020-cfd-metautil`)
deliberately controls the *case* and *backend* axes so that any comparison
isolates the *algorithm* axis. One concrete planned result: take the MPS
implementation (1D, later 2D) and compare it against another quantum code
(e.g. QLBM) on the same case.

This discussion worked through what a 2D extension would involve, whether it
is worth doing, and how it serves the comparison study.

---

## 1. Which 2D Burgers?

Three increasingly honest formulations:

- **Scalar 2D**: one field `u(x,y,t)`, `∂_t u + u(∂_x+∂_y)u = ν∇²u`. A toy.
- **Vector 2D (general)**: `u=(u,v)`, `∂_t u + (u·∇)u = ν∇²u` — two coupled
  PDEs with cross-advection (`v ∂_y u`, etc.). The "real" 2D Burgers.
- **Vector 2D, potential/irrotational** (`u = ∇ψ`, curl-free): the only
  multi-D regime where Cole-Hopf survives.

The choice is not cosmetic — it determines whether the linearization that
powers this codebase's pure-quantum methods still applies.

## 2. Cole-Hopf only linearizes the potential case

The pure-quantum story rests on Cole-Hopf turning Burgers into the linear
heat equation. In multi-D the transform **only works when the velocity field
is a gradient**: `u = ∇ψ`, `φ = exp(−ψ/2ν)`, then `∂_t φ = ν∇²φ` exactly and
`u = −2ν ∇(ln φ)`. A rotational (nonzero-curl) component cannot be written as
a single scalar gradient, so the linearization is lost. General rotational 2D
Burgers is essentially **2D Navier–Stokes minus the pressure projection**.

## 3. The heat half of 2D generalizes nearly for free

The 2D discrete Laplacian is a **Kronecker sum**: `L₂ = L_x ⊗ I + I ⊗ L_y`.
Two consequences fall straight out of the existing 1D code:

- **qft-diagonal**: `L₂` is still diagonal in the 2D Fourier basis, with
  eigenvalues `λ_{kx} + λ_{ky}`. The propagator becomes
  `(QFT_x⊗QFT_y) → conditional-Ry on summed eigenvalues → inverse`. The
  existing `laplacian_eigenvalues` / `compute_theta_exact` machinery extends
  by summing per-axis spectra. Still `O((2q)²)` gates.
- **dense-block / lcu**: because `L_x⊗I` and `I⊗L_y` commute,
  `exp(ν dt L₂) = exp(ν dt L_x) ⊗ exp(ν dt L_y)` **exactly** — no
  Strang/Trotter splitting error between x- and y-diffusion. The 1D
  propagator builder is reusable per axis and tensored.

## 4. Where the real work and risk sit (irrotational 2D)

- **MPS qubit ordering** (`burgers_mps.py`, Ran 2020 prep): encoding
  `φ(x,y)` on `2q` qubits forces a choice of x/y bit interleaving. Row-major
  vs. snake vs. Hilbert-curve ordering changes entanglement structure and
  therefore bond-dim growth and prep fidelity. This is the one genuine
  research unknown; locality-preserving orderings are the known mitigation.
- **Cole-Hopf forward/inverse** (`burgers_cole_hopf.py`): currently a 1D
  cumulative-trapezoid (`u→∫u→φ`) and a log-derivative inverse. In 2D the
  forward step recovers the potential `ψ` from `(u,v)` via a line-integral /
  Poisson solve (path-independent only if truly curl-free; otherwise
  Helmholtz-project first). The inverse is clean: both `u,v` are gradients of
  the single scalar `ln φ`.
- **Classical FTCS reference** (`burgers_classical.py`): cost goes
  `2^q → 2^{2q}`, so the L2-against-classical validation window shrinks
  (~`q≈6` per axis).
- **Grid2D**: cheap — the framework `Grid` ABC already declares `ndim`; add a
  `Grid2D(ndim=2)` with `qx, qy` and a tensor-product `xc`. The dispatcher in
  `burgers_fw.py` is dimension-agnostic at the contract level.

## 5. The direct-`u` family is a dead end for 2D

`shift`, `quantum_exact`, `quantum_circuit`, `mps`, `tebd`, `tebd_circuit`
march `u` and rebuild operators via the `O(4^q)` Pauli decomposition.
`quantum_exact` already OOMs at `q=6` in 1D → `q=3` in 2D, and the vector
problem adds cross-advection these kernels don't model. Scope 2D as
**Cole-Hopf-only**.

---

## 6. Is 2D irrotational interesting on its own? (Honest verdict)

**As physics, thin. As a comparison substrate, valuable.**

In the irrotational case the nonlinearity is effectively fake: for curl-free
`u`, `(u·∇)u = ½∇|u|²` — a pure gradient that folds into the potential and is
annihilated by Cole-Hopf. What remains to evolve is the **linear 2D heat
equation**, which the quantum community already solves and which generalizes
from the 1D code almost for free (§3). Curl-free is *preserved* by the
dynamics (`u = −2ν∇ln φ` is a gradient for all time), which is tidy but is
exactly what guarantees you never touch the hard part. A reviewer could
fairly call it "the heat equation in disguise."

What genuine interest exists is concentrated in:
- **"Encode once, evolve purely, decode once."** The pure-quantum advantage
  requires Cole-Hopf *once* into `φ`, pure-quantum heat evolution, then invert
  *once*. Irrotational 2D is the only multi-D regime where that property
  survives. (If you decode/re-encode every step, the advantage is gone.)
- **2D state encoding** — the MPS ordering / bond-dim study (§4), publishable
  on its own but really a tensor-network-encoding result, studyable on the
  heat equation directly.

**Conclusion:** build 2D irrotational as a **stepping-stone / pipeline-scaling
demo** and as the common substrate for cross-code comparison — not as a fluids
result in its own right.

---

## 7. Handling the nonlinear term in the rotational case

When the flow is rotational, Cole-Hopf cannot linearize and you must march the
genuine nonlinear advection `(u·∇)u`. The advection term already exists in
scalar quadratic form in the code: `burgers_nonlinear.py` computes
`u * grad_u = diag(u)·D·u`, with diffusion `νLu` as the linear part — i.e.
`du/dt = F₁u + F₂(u⊗u)`.

Five families of approach, ordered by fit to existing machinery:

1. **Carleman linearization — recommended.** Lift the state to
   `[u, u⊗u, u⊗u⊗u, …]` truncated at order `M`, turning the quadratic ODE into
   a *linear*, sparse, block-bidiagonal system that is directly
   block-encodable / LCU-able — reusing the `S±` shift primitives in
   `burgers_lcu.py`. Burgers' pure-quadratic nonlinearity is the textbook
   Carleman demo.
   - *Cost*: lifted dimension `O(n^M)` for `n=2^{2q}`; the order-`M` register
     needs `M·2q` qubits (Carleman order multiplies qubit count); the `F`
     matrices are sparse.
   - *Catch*: convergence requires `R = ‖nonlinearity‖/‖dissipation‖ < 1`,
     i.e. low effective Reynolds / viscosity-dominated. State this as a
     limitation, not a bug.
2. **Schrödingerization / Koopman–von Neumann (level-set).** Exact linear
   embedding (no truncation error); price is a phase-space blowup and a less
   direct fit to the circuit-construction style.
3. **Extend the per-step Hermitian-fit** (`quantum_exact`/`quantum_circuit`).
   Conceptually trivial to add 2D cross-advection, but `O(4^{2q})` Pauli cost
   (OOMs at `q=3` in 2D) and not pure-quantum. Useful only as a validation
   oracle.
4. **Operator splitting.** Per step: Cole-Hopf heat solve for the
   potential/diffusion part, then a rotational-advection correction. Runnable
   soonest and reuses most of the codebase, but the correction step still
   contains the nonlinearity (needs 1–3 underneath) and incurs `O(dt)`/`O(dt²)`
   splitting error.
5. **Variational (McLachlan).** Avoids the lifting blowup but heuristic
   (barren plateaus, optimizer dependence) and stylistically off.

**Recommendation:** Carleman is the principled next rung — keeps everything
linear, reuses the LCU/block-encoding code, and `F₂` already exists in scalar
form. Scope it as low-Reynolds, order-2 first; validate against the per-step
Hermitian-fit oracle on tiny grids.

---

## 8. MPS-vs-QLBM comparison (the concrete result)

**QLBM is already wired in this package**: `--method qlbm` (classical D1Q3,
`burgers_qlbm.py`) and `--method qlbm_circuit` (`burgers_qlbm_circuit.py`),
both delegating integrators in `burgers_fw.py`. So the **1D MPS-ColeHopf vs
QLBM comparison is runnable today** through the same harness.

**Why it is a clean result:** both methods share `solverfw` — same case
definition (IC, BC, ν, grid, `t_final`), same FTCS reference, same error
metric, same shots/backend plumbing. The case and backend axes are controlled,
so the comparison isolates the algorithm axis.

**The caveat that makes it a result, not a horse race** — the two encode
different things, so report distinct error budgets and footprints, not just
an L2 number:

| | MPS-ColeHopf | QLBM (D1Q3) |
|---|---|---|
| Encodes | macroscopic `φ` on `q` qubits (clean `N=2^q`) | 3 mesoscopic `f_i`, `log₂(3N)` qubits (factor-3 packs awkwardly) |
| Error sources | bond-dim truncation + Ran state-prep fidelity | collision/BGK + moment reconstruction + athermal-Burgers Re/Ma limits |
| Nonlinearity | irrotational only (Cole-Hopf) | genuinely nonlinear-capable (incl. rotational in 2D) |
| Scaling knob | χ (bond dim) | lattice resolution |

**Comparison dimensions:** qubit count, two-qubit-gate depth, shots-to-target
accuracy, noise resilience on fake/hardware backends, and the per-method
scaling knob — all against the exact Cole-Hopf / FTCS ground truth.

**2D requires lifting both codes, asymmetrically:** MPS→2D is the easy
structural part (§3) plus the encoding study (§4); QLBM D1Q3→D2Q5/D2Q9 is its
own lift (new equilibrium, collision, streaming). The **irrotational case is
the common-validity ground** where both can be compared directly; QLBM can
push into rotational 2D where MPS-ColeHopf cannot follow — a natural
"divergence" result and a motivation for the Carleman route on the MPS side.

---

## 9. Suggested sequencing

1. Run the **1D MPS-ColeHopf vs QLBM** head-to-head on matched cases
   (same `q`, ν, IC, backend); produce the comparative metric table. Runnable
   now.
2. Build **2D irrotational** as pipeline-scaling + the MPS encoding-ordering
   study (Cole-Hopf only, `qft-diagonal` and `dense-block` first).
3. Lift **QLBM to 2D** (D2Q9) for the 2D side of the comparison.
4. Pursue **rotational 2D via Carleman** as the genuinely novel quantum-CFD
   result, with the per-step Hermitian-fit oracle for validation.
