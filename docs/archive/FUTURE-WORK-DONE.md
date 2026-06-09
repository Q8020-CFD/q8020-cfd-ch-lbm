# Future Work — Completed / Resolved

Items split out of [../future/FUTURE-WORK.md](../future/FUTURE-WORK.md)
once shipped or resolved. **Original item numbers are preserved** (other
docs and code comments reference them, e.g. "#14", "#26"); FUTURE-WORK
keeps the gaps rather than renumbering. Open items stay in FUTURE-WORK.

---

## 12. Cole-Hopf-exact analytic IC (plan F12.1) — DONE

Shipped.  See OVERVIEW §4.4 and `burgers_cole_hopf.py:
{initial_condition_cole_hopf_exact, analytic_solution_cole_hopf,
validate_cole_hopf_coeffs}`.  Wired via `--ic cole_hopf_exact` with
coefficients via `--ic-cole-hopf-coeffs "a0,a1,..."`.  When `--method`
is `cole_hopf` or `cole_hopf_circuit`, IC defaults to
`cole_hopf_exact` and the analytic `u(x,t)` is used as the reference
trajectory automatically; `--no-analytic-reference` falls back to
FTCS/Godunov.  Restricted to `--bc dirichlet` + `--source none` by the
math (Neumann-on-φ cosine basis; modes only stay decoupled in the
unforced case).

## 14. `qlbm_circuit` real-backend shots path (QLBM F11-13) — DONE

Shipped as "Option A" (hybrid by construction; mirror of Meena
Appendix A.A for QLBM).  `run_qlbm_circuit_simulation` shots branch
now builds the same per-step circuit the statevector path builds,
transpiles, executes on the configured backend, and reconstructs
`f_post` via `|ψ_out_k| ≈ √(counts[k]/S)` followed by
`unflatten_distributions`.  `--sign-recovery {none, classical_oracle}`
both honored; `hadamard_test` since shipped under #26.  Per-step metrics gain
`leakage` (mass in the unused `|11⟩` velocity block — noise sensor)
and `negative_mass` (classical-oracle signal for when sign recovery
matters).  See SPEC-qlbm-shots-and-sign-recovery.md and OVERVIEW
§5.2 for the full contract and the hybrid-vs-pure-quantum framing.

> **Note.** This was the *hybrid* shots path, now named
> `qlbm_circuit_hybrid`. The pure-quantum QALB that claimed the bare
> `qlbm_circuit` name is #27 (its collision + shots path are now
> shipped; remaining items stay open in FUTURE-WORK).

## 16. Gaussian IC (plan F12.2) — DONE

Shipped.  `initial_condition_gaussian` in `burgers_classical.py`,
`--ic gaussian` with `--ic-center` (default 0.5) and `--ic-sigma`
(default 0.1); amplitude via the existing `--ic-amplitude`.  No
closed-form Cole–Hopf analytic reference (the `∫u₀` is an erf, so
`φ₀` has no clean heat-equation evolution); pairs with FTCS/Godunov
as the classical reference.  Works with all methods including
`cole_hopf_circuit` and `qlbm*`; for LBM keep `--ic-amplitude < 1.0`
for D1Q3 stability.  See OVERVIEW §1.1.

## 24. Plumb new IC / reference flags into `BurgersConfig` — DONE

Shipped.  `BurgersConfig` in `burgers_fw.py` now has fields
`ic_center`, `ic_sigma`, `ic_cole_hopf_coeffs`, `classical_reference`,
`analytic_reference`; `burgers_solver.py` passes them in from
`args`; `burgers_postprocess.py` records them in both the case
fragment and the JSON summary, conditional on the relevant `--ic` for
the IC-specific fields (so case fragments stay clean for unrelated
ICs).  See OVERVIEW §8.1.

## 25. Classical `cole_hopf` + `--bc dirichlet` BC mapping — DONE

Discovered during the #24 smoke test: `run_cole_hopf_simulation` was
periodic/Neumann-only on the phi side, so `--method cole_hopf --bc
dirichlet` crashed with a cryptic `ValueError: Unknown bc: 'dirichlet'`
from `build_laplacian_dense`.  This broke BC symmetry between the
classical and circuit CH paths and made the classical CH unusable as
a V&V oracle for `cole_hopf_circuit` under Dirichlet-on-u.

Shipped.  `run_cole_hopf_simulation` now mirrors the same `phi_bc =
"neumann" if bc == "dirichlet" else bc` mapping that
`burgers_cole_hopf_circuit.py:1911` uses on the circuit side
(per OVERVIEW §4.1: Dirichlet on u ↔ Neumann on phi).  Unsupported BC
values raise a clean `NotImplementedError` instead of leaking the
phi-side label up to the user.  Classical `cole_hopf` + `--bc
dirichlet` now runs and can serve as a cross-check against
`cole_hopf_circuit` at the same BC.

## 26. Hadamard per-bin sign test for `qlbm_circuit` (fast follow to #14) — DONE

Shipped.  `--sign-recovery hadamard_test` is now a stand-alone
(non-hybrid) sign-recovery mode for `qlbm_circuit` (the hybrid, now
`qlbm_circuit_hybrid`).  New helper `_qlbm_hadamard_signs` in
`burgers_qlbm_circuit.py` runs a per-bin Hadamard test (one ancilla,
controlled `X_k · U_step · prep(ψ_in)`) that estimates
`Re(⟨k|U_step|ψ_in⟩)` for each of the `4N` basis indices.  Every QLBM
operator (collision Householder + real streaming permutation) is real,
so that real part *is* `ψ_out_k`; its sign is the recovered sign and
is combined with the direct magnitude `√(counts[k]/S)`.  Unlike
`classical_oracle` the signs come from the circuit itself, keeping the
run a stand-alone benchmark.  Cost is `O(4N)` extra circuit executions
per step; per-step metric `hadamard_p_kept` reports the mean
post-selection acceptance.  Mirrors
`burgers_cole_hopf_circuit.hadamard_per_bin_circuit`.  Verified at
`q=3` against the statevector path (exact agreement at 2e5 shots).
See OVERVIEW §5.2.

## 28. Carleman linearization of BGK collision — SUBSUMED by #27

Resolved as *not a separate method*.  The Itani QALB (#27) **is** a
Carleman/Kowalski second-quantised scheme: the value/Fock encoding lifts
the BGK quadratic nonlinearity into a linear operator on the bosonic
registers, and the Hermitisation is built on that lift.  So #27 and #28
collapse into the single construction shipped under #27 — there is no
separate `qlbm_carleman` method.  (Original scoping note retained below
for history.)

> **Why (original).** Alternative pure-quantum QLBM route to #27.
> Carleman lift `(f, f⊗f, …)` truncated at order `M` turns BGK's
> quadratic nonlinearity into a linear sparse block-bidiagonal generator;
> evolution is then a single fixed, state-independent unitary.  Scope was
> substantial (lifted dimension `O(n^M)`, `M·(q+2)` qubits, truncation
> analysis), with convergence requiring `R = ‖nonlin‖/‖dissip‖ < 1`.

## 29. Linearized BGK collision (low-Mach pure-quantum QLBM) — DONE

Shipped as `--method qlbm_circuit_linear`
(`burgers_qlbm_linear_circuit.py`).  Linearises BGK about the rest
equilibrium (`f = f_eq⁰ + δf`, drop `O(δf²)`), giving a **fixed,
state-independent linear collision `M₃`** in the amplitude encoding
(same `q+2`-qubit register as the hybrid — no value/Fock re-encoding).
Block-encoded collision + streaming run as `k`-step measure-reprepare
segments (`_run_shots_segments`); gates pass (collision exactness,
block-encoding extraction, statevector==classical, shots(k) tracking).
This is the de-risking scaffold for #27 and the direct
measure-reprepare(k) comparison vehicle against Cole–Hopf.  **Catch
(by design):** linearisation drops the `u²` term, so it loses shock
physics — valid only for smooth, low-Mach, near-equilibrium flow; a
pedagogical benchmark, not the production solver.  See OVERVIEW §5.4.
