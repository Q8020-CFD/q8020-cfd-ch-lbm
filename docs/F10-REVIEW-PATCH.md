# F10 — Review Patch

Patch document addressing the code review of the first F10 implementation pass.
Apply on top of the currently-merged F10 code. Do not re-scope; the
**F10-IMPLEMENTATION-SPEC.md** remains the authoritative spec and is not
being revised except where this patch says so explicitly.

Patches are ordered by dependency. Parcels can be dispatched in parallel
where marked `par:`.

## P-A — Rename or re-implement `pauli-trotter` (BLOCKER)

**Problem.** The current `--propagator pauli-trotter` path in
`burgers_cole_hopf_circuit.py:272-321` (`heat_pauli_step_circuit`) builds
`P = expm(νLdt)`, eigendecomposes it, and applies a dense `UnitaryGate` of
the eigenvectors with a ladder of multi-controlled-Ry per eigenvalue. No
Pauli decomposition, no commuting-group split, no first-order Trotter. The
flag name lies and the downstream acceptance 11.4 (first-order convergence
slope vs N_steps) becomes un-testable because the propagator is already
exact per step.

**Choose one fork.** Whichever the author picks, do it in full — do not
leave both.

### Fork A — Implement real Pauli-Trotter (restores 11.4)

Drop the eigendecomposition path. Replace with:

1. `L_dense = build_laplacian_dense(N, dx, bc=bc)` — already available.
2. `paulis = SparsePauliOp.from_operator(Operator(L_dense))` — Qiskit gives
   Pauli string + complex coefficient pairs. `L` is real-symmetric so the
   coefficients are real; assert this.
3. **Commuting-group partition.** Use `SparsePauliOp.group_commuting(
   qubit_wise=False)` so that each group exponentiates with a single
   basis-change + Rz ladder. For tridiagonal-with-wrap L this gives O(q)
   groups.
4. **Per-group non-unitary exponentiation via LCU-of-two-unitaries.** For
   a Hermitian `L_g` with real spectrum of mixed sign, write

       exp(ν · L_g · dt) = cosh(ν |L_g| dt) · I + sinh(ν |L_g| dt) · sign(L_g)

   which equals `(1/2)(exp(ν |L_g| dt · U_+) + exp(−ν |L_g| dt · U_−))` for
   a suitable pair `U_±`. Implementation: Hadamard on ancilla, controlled
   `PauliEvolutionGate(L_g, time=+ν·dt)` on `|0⟩_anc`, controlled
   `PauliEvolutionGate(L_g, time=−ν·dt)` on `|1⟩_anc`, Hadamard,
   measure-and-reset ancilla. Standard LCU-of-2.
5. **Product of groups per Trotter step:** `Π_g LCU_g`. First-order; error
   `O(Δt²)` per step; `O(Δt)` globally. Matches spec §6.
6. Ancilla: one per group per step with reset between groups, as the spec
   already says at §5.2.

**Acceptance 11.4** then becomes testable: drive `N_steps ∈ {10, 20, 50,
100, 200}` at fixed `T`, plot `‖φ_circuit − φ_dense‖₂ / ‖φ_dense‖₂` vs
`N_steps`, expect slope ≈ −1 on log-log.

### Fork B — Rename and narrow (avoids 11.4)

Keep the existing exact-eigendecomposition implementation but stop calling
it Pauli-Trotter. Do all of:

- Rename `--propagator pauli-trotter` → `--propagator dense-block` in
  `burgers_solver.py:122-125`.
- Rename `heat_pauli_step_circuit` → `heat_dense_block_step_circuit`
  (and `_full_` analogously) in `burgers_cole_hopf_circuit.py`.
- Rename the `_pauli` TOML groups in `input/burgers_quantum.toml` to
  `_dense` prefix.
- Edit `F10-IMPLEMENTATION-SPEC.md` §2, §4, §6, §11.3, §11.4, §12.P4 to
  replace `pauli-trotter` with `dense-block` and describe the eigendecomp
  implementation honestly. Delete acceptance item 11.4 (Trotter-error
  convergence does not apply to two exact propagators). Move the Pauli
  path to §13 future work.

Fork A is the spec-faithful choice and gives Murali the Pauli-level object
he expects. Fork B is the minimum-churn choice if the author judges the
LCU-of-2 implementation too expensive for the current sprint. **Recommend
Fork A** unless there is a specific reason not to.

## P-B — Add `test_cole_hopf_circuit.py` (BLOCKER)

`par: A`

New file `analysis/test_cole_hopf_circuit.py`. Pytest asserts for each
numbered acceptance item that can be expressed as an assert. Use the
local venv for all runs (Best-Practices rule 7).

Minimum tests required:

- `test_11_1_classical_no_regression`: run `--method cole_hopf` and
  `--method shift` at `q=5, ν=1e-2, T=0.5·t_shock` via `run_simulation`
  directly. Assert `‖u_ch − u_shift‖₂ / ‖u_shift‖₂ < 0.02`.
- `test_11_2_qft_diagonal_statevector`: `(q=4, ν=1e-2, bc=periodic,
  T=0.05, N_steps=10)`. Run `run_cole_hopf_circuit_sv` with
  `propagator='qft-diagonal'`, compare final `φ` against
  `build_heat_propagator(...) @ φ₀`. Assert relative L2 `< 1e-6`.
- `test_11_3_second_variant_statevector`: same as 11.2 against whichever
  propagator came out of P-A. For Fork A, assert `< 1e-3` at `N_steps=10`
  and `< 1e-6` at `N_steps=200`. For Fork B, assert `< 1e-6` at
  `N_steps=10` (exact).
- `test_11_4_trotter_convergence`: **Fork A only.** `N_steps` sweep, fit
  log-log slope, assert slope `< -0.9` (i.e. at least first-order).
  Fork B: omit this test.
- `test_11_5_shots_accuracy`: `shots=150000` at the 11.2 config. Assert
  `‖u_circuit − u_dense‖₂ / ‖u_dense‖₂ < 0.05` and `P_success > 0.3`.
  Mark as `@pytest.mark.slow` so it can be deselected in quick runs.
- `test_11_6_smoke_small_nu`: `(q=3, ν=1e-4, bc=dirichlet, T=0.1·t_shock,
  shots=0)`. Assert the run completes and produces finite u. Full
  small-ν acceptance is the artifact from P-D.

All tests: invoke via `./.venv/bin/python -m pytest
analysis/test_cole_hopf_circuit.py -v`. Honour line-width 88 and PEP 8.

## P-C — Collapse the shots path to one circuit (SERIOUS)

`par: A, B`

**Problem.** `_run_shots_path` at
`burgers_cole_hopf_circuit.py:475-567` rebuilds the full circuit and
re-transpiles once per snapshot. For `save_every=1, n_steps=50` that is
50 independent circuit builds and runs (~50× the correct cost).

**Fix.** Two acceptable patterns:

1. **Transpile once, run N_steps times** (easier). In
   `run_cole_hopf_circuit_simulation` shots branch, build a cumulative
   family of circuits `{circuit_1, circuit_2, ..., circuit_{n_steps}}`
   where `circuit_k` is `k` Trotter layers + final data measurement,
   but transpile them in one `transpile([...], backend)` call so the
   compilation work is batched. Run each with the target `shots`.

2. **One circuit with mid-circuit snapshots** (preferred, but requires
   Aer save-instruction support). Add `save_statevector(label=f"t_{k}")`
   inside the layer loop; run once with `shots` (or `shots=0` for SV).
   Post-process every labelled intermediate state. Limit this to the
   SV+snapshots case; shots with post-selected history at every
   intermediate time is not well-defined without per-snapshot
   post-selection.

Option 1 is the pragmatic choice. Enforce at most one `AerSimulator()`
instantiation and one `transpile(...)` call per `run_cole_hopf_circuit_
simulation` invocation.

## P-D — Paper-scale small-ν Dirichlet groups (BLOCKER for 11.6)

`par: A, B, C`

**Problem.** All `[cole_hopf_circuit_*]` groups in
`input/burgers_quantum.toml` run periodic BC at `ν ∈ {0.01, 0.05, 0.1}`.
Spec §11.6 requires `ν=1e-4 + bc=dirichlet + shots=150k`. The small-ν
centering code path (§9) is therefore never exercised in a sweep.

**Add** the following TOML groups. Use `propagator = "pauli-trotter"` if
Fork A was taken (real Pauli-Trotter handles Dirichlet via Neumann-on-φ);
use `propagator = "dense-block"` if Fork B was taken.

```toml
[paper_cole_hopf_circuit_q3_shots150k]
"--q" = 3
"--nu" = 1e-4
"--shock-pct" = 80.0
"--method" = "cole_hopf_circuit"
"--propagator" = "pauli-trotter"   # or "dense-block"
"--bc" = "dirichlet"
"--shots" = 150000
"--save-every" = 2

[paper_cole_hopf_circuit_q4_shots150k]
"--q" = 4
"--nu" = 1e-4
"--shock-pct" = 80.0
"--method" = "cole_hopf_circuit"
"--propagator" = "pauli-trotter"
"--bc" = "dirichlet"
"--shots" = 150000
"--save-every" = 4

[paper_cole_hopf_circuit_q5_shots150k]
"--q" = 5
"--nu" = 1e-4
"--shock-pct" = 80.0
"--method" = "cole_hopf_circuit"
"--propagator" = "pauli-trotter"
"--bc" = "dirichlet"
"--shots" = 150000
"--save-every" = 8
_group_postproc = ["python ./q8020-cfd-axequalsb/src/murali_burgers/analysis/plot_cole_hopf_circuit_evolution.py"]
```

Run the q=5 group. Acceptance artifact:
`paper_cole_hopf_circuit_q5_shots150k.png` matching spec §11.6 visual
criterion (visible forming shock, not crazy). Include a log line showing
the centering decision (`use_centering=True`, `e_mid=…`) so §14 checklist
item 7 is verifiable.

## P-E — Resolve the unused polynomial fit (SERIOUS)

`par: A, B, C, D`

**Problem.** `fit_theta_polynomial` at
`burgers_cole_hopf_circuit.py:59-77` returns coefficients that are then
used to evaluate `θ(k)` at all `2^q` points
(`burgers_cole_hopf_circuit.py:122-126`), Möbius-transformed, and emitted
as up to `2^q` multi-controlled-Ry gates. The polynomial fit is used
only for a warning threshold check. It produces no gate-count reduction.
Spec §4 promises O(q²) gates per step; this path delivers O(2^q).

**Choose one fork.**

### Fork E1 — Use the polynomial, deliver on O(q²)

Replace `build_polynomial_ry` with a polynomial-native emission:

- Decompose `k = Σ_i 2^i · b_i` where `b_i = (I − Z_i)/2`. A monomial
  `k^m` expands into at most `C(q+m-1, m)` products of Z-diagonal terms
  over the data qubits. Each product of k `Z_j` operators compiles into
  one controlled-Ry on the ancilla with k controls (Z-diagonal → diagonal
  rotation).
- For `degree d`, emit O(q^d) controlled-Ry gates, each with ≤ d controls.
  At `d=6, q=5` that is ~8000 gates — worse than current 2^q=32. Not a
  win.

So Fork E1 is **not actually cheaper** at the q we run. Reject it.

### Fork E2 — Drop the polynomial fit, keep Möbius (recommended)

Delete `fit_theta_polynomial` and the `poly_degree` parameter threading.
Build Möbius coefficients directly from exact `θ(k)` via
`_mobius_transform(compute_theta_exact(...), q)`. No accuracy loss (the
fit was the only source of error; Möbius is exact). Update the spec §4
"O(q²)" claim to `O(2^q)`. Update §5.1 polynomial-fit paragraph to
describe the Möbius construction honestly.

Gate count at q=5 is 32 controlled-Ry per layer, well within Aer. The
"O(q²)" promise was unrealistic for arbitrary conditional rotations
anyway; the honest version is still fast enough.

### Fork E3 — Direct QROM lookup

Out of scope for this patch; leave as `# TODO(q>=7): QROM-based θ(k)
loading` in the code and in §5.1 future work.

**Recommend Fork E2.**

## P-F — Repo hygiene (BLOCKER; trivial)

`par: A, B, C, D, E`

- `rm -rf q8020-cfd-axequalsb/.claude`. Best-Practices rule 23 (no
  per-repo `.claude` dirs).
- Add `.claude/` to the repo `.gitignore` so it does not come back.
- Remove `src/murali_burgers/.DS_Store` and add `.DS_Store` to
  `.gitignore`.
- Remove scratch `src/murali_burgers/q8020_*_0.json` files (they belong
  under the output directory, not in source).
- Do **not** run `git add`, `git commit`, or `git push` as part of this
  patch — leave the user to stage and commit (Best-Practices rule 20).

## Ordering and sequencing

```
  P-A  ──┐
         ├──▶  P-B  ──┐
  P-C  ──┤            │
         ├──▶  P-D  ──┼──▶  run & verify
  P-E  ──┤            │
         │            │
  P-F  ──┘            │
```

P-A drives spec edits and API names; P-B depends on P-A's final flag
names and acceptance set; P-C, P-D, P-E, P-F are independent and can run
in parallel with P-A as soon as a single agent owns the P-A fork choice.

## Acceptance for this patch

- `./.venv/bin/python -m pytest analysis/test_cole_hopf_circuit.py -v`
  all non-slow tests PASS in under 2 minutes.
- Slow tests PASS in under 30 minutes.
- `q8020-sweep q8020-cfd-axequalsb/input/burgers_quantum.toml` executes
  `paper_cole_hopf_circuit_q5_shots150k` to completion and the
  postproc PNG renders a recognizable shock.
- `git status` shows no `.claude/`, `.DS_Store`, or scratch JSONs
  tracked or untracked in the repo.
- The §14 checklist in `F10-IMPLEMENTATION-SPEC.md` passes end to end.

## Out of scope (reaffirming)

Out-of-scope items from the main spec §13 remain out of scope:
encoding change, direct u-space evolution, DST-based Dirichlet QFT
variant, hardware execution, QSVT alternative, F11 Burgulence.
