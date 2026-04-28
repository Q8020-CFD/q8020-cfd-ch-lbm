# SPEC — Forced Burgers via heat-with-potential, pure quantum

Self-contained handoff for an agent that has not seen prior
conversations.

## 0. Context

`q8020-mps-burgers` is a Cole-Hopf quantum-circuit pipeline for 1D
Burgers. Source code at `/Users/agallojr/proj/src/q8020-mps-burgers/`.
The `cole_hopf_circuit` method today silently drops the `--source`
flag at the dispatch in
[burgers_trotter.py:713-720](../src/burgers_trotter.py:713) — it
solves **unforced** Burgers regardless. The classical FTCS baseline
in `burgers_solver.py` keeps the source. So when a user runs
`--ic sine --source sine` the two legs are solving different PDEs
and the L2 error number is dominated by that mismatch, not by
algorithm error.

The paper's Sec. III.A reference test problem (Murali/Meena AIAA-2026)
uses g(x,t) = sin(2πx)·cos(2πt). To reproduce it in our quantum path
we need forced Burgers support.

The temptation is to add the source via operator-splitting — half-step
of g classically, full heat circuit, half-step of g classically. That
is **classical oracle steerage** inside the time loop and breaks the
pure-quantum claim that the P-G work explicitly bought (state prep
via Ran 2020, propagator as a real Qiskit circuit, no mid-loop
classical evolution). This spec rejects that approach.

## 1. The math (paper-faithful, pure quantum)

Forced Burgers:

```
∂u/∂t + u·∂_x u = ν·∂²_x u + g(x,t)
```

Cole-Hopf substitution u = −2ν·∂_x ln φ gives a φ-equation with a
time-varying potential:

```
∂φ/∂t = ν·∂²_x φ − V(x,t)·φ                             (*)
```

where (standard textbook derivation; see e.g. Hopf 1950, Cole 1951
with source extension)

```
V(x,t) = −(1/(2ν)) · ∫_0^x g(x', t) dx'   + C(t)
```

C(t) is a gauge degree of freedom — it shifts φ by a multiplicative
factor that cancels in u = −2ν·∂_x ln φ. Pick C(t) to keep V
spatially mean-zero per step (numerically convenient).

Equation (*) is **linear in φ** — heat plus diagonal multiplicative
potential — so it admits a pure-quantum propagator:

```
φ(t + dt) = exp(dt · (ν·∂² − V(x,t))) · φ(t)
         ≈ exp(dt · M_n) · φ(t),  where M_n = ν·L − diag(V_n)
```

Per step n we evaluate V at the midpoint t_n + dt/2 (Strang midpoint
rule, second-order in dt). M_n is real symmetric, so exp(dt·M_n) is
positive symmetric; it block-encodes via the same eigendecomposition
+ controlled-Ry + ancilla-post-select pattern the dense-block
propagator already uses for the unforced case.

For the paper case g = sin(2πx)·cos(2πt):

```
∫_0^x g dx' = cos(2πt) · (1 − cos(2πx)) / (2π)
V_raw(x,t) = −cos(2πt) · (1 − cos(2πx)) / (4πν)
V(x,t)     = V_raw − ⟨V_raw⟩_x
            = cos(2πx) · cos(2πt) / (4πν)             (gauge-fixed)
```

At ν = 0.1 the prefactor is 1/(4π·0.1) ≈ 0.796 — physically
significant, not a perturbation.

## 2. Goal

`python burgers_solver.py --method cole_hopf_circuit --propagator
dense-block --source sine --ic sine --nu 0.1 --shock-pct 100 ...`
solves the forced Burgers equation through a pure-quantum CH path
(state prep + propagator + measurement on the simulator/hardware,
classical only for forward CH, post-select, and inverse CH — same
pattern as the unforced case today).

The acceptance gate: matches the classical FTCS forced baseline
within Trotter+shot-noise tolerance at q=5, ν=0.1, sine source —
i.e., the L2 error drops from today's ~0.21 to whatever the
algorithm's true floor is (estimate: a few percent dominated by
shot noise at 100k shots, plus second-order Strang error in dt).

## 3. Non-goals

- `--propagator qft-diagonal` with source. K = ν·∂² is diagonal in
  the Fourier basis; V(x,t) is diagonal in the position basis.
  They don't share an eigenbasis, so the qft-diagonal route needs
  Strang splitting (`exp(−V·dt/2)·exp(K·dt)·exp(−V·dt/2)`) with V
  block-encoded separately. Doable but mechanically distinct.
  Defer to a follow-up parcel.
- Classical operator splitting (the steerage path). Explicitly
  excluded.
- Source forwarding for any other method (`tebd_circuit`,
  `quantum_circuit`, etc.). Those paths already accept source via
  their own dispatch and aren't broken.
- Generic `g(x,t)` with no closed-form antiderivative. This spec
  assumes either an analytic form (paper's sin·cos) or trapezoidal
  numerical integration on the grid. Both are cheap.
- Time-step adaptivity, `dt` refinement studies. Use the same dt as
  the unforced case.
- Sign recovery interactions. φ stays positive throughout (heat +
  potential preserves positivity if φ₀ > 0 and V is real and
  bounded). No sign-recovery surface-area increase.

## 4. CLI surface

No new flags. `--source {sine,none}` already exists in
[burgers_solver.py:103-105](../src/burgers_solver.py:103). The
implementation must thread `source_fn` (already constructed at
[:203](../src/burgers_solver.py:203)) through dispatch into the
quantum CH propagator builder.

If we ever support more sources, add them as `--source` choices in
the existing argparse — out of scope for this spec.

## 5. Plumbing changes

### 5.1 Dispatch — `burgers_trotter.py::run_simulation`

The `cole_hopf_circuit` branch at
[burgers_trotter.py:713-720](../src/burgers_trotter.py:713) currently
does NOT pass `source_fn`. Add it:

```python
if method == "cole_hopf_circuit":
    sols, mets = run_cole_hopf_circuit_simulation(
        u0, x, nu, dt, n_steps, bc=bc,
        propagator=propagator, shots=shots,
        snapshot_interval=snapshot_interval,
        bond_dim=bond_dim, encoding=encoding,
        source_fn=source_fn,                # NEW
        # ... whatever new kwargs the shots-backend spec adds
    )
    return sols, mets
```

### 5.2 `burgers_cole_hopf_circuit.py::run_cole_hopf_circuit_simulation`

Add `source_fn: Callable | None = None` parameter. Forward to the
propagator-building code paths.

**Guard:** if `source_fn is not None and propagator != "dense-block"`,
raise `NotImplementedError(
    "Source forcing requires --propagator dense-block in this "
    "release. qft-diagonal+source is on the roadmap."
)`. Keeps the contract honest until the qft-diagonal path lands.

### 5.3 New module — `burgers_potential.py`

A small classical helper module. Computes V(x,t) on the grid for
the propagator builder.

```python
"""Source-induced potential V(x,t) for forced Burgers via Cole-Hopf.

For ∂u/∂t + u·∂_x u = ν·∂²_x u + g(x,t), the Cole-Hopf
substitution u = −2ν·∂_x ln φ yields ∂φ/∂t = ν·∂²_x φ − V(x,t)·φ
with V_x = −g/(2ν).  Returned V is gauge-fixed to spatial
mean-zero so that φ does not accumulate multiplicative
blow-up/decay between snapshots.
"""

import numpy as np


def potential_from_source(
    source_fn,           # callable (x, t) -> g(x,t) on grid, or None
    x: np.ndarray,
    t: float,
    nu: float,
    bc: str = "periodic",
) -> np.ndarray:
    """Return V(x, t) on the grid; zeros if source_fn is None.

    Integration is trapezoidal on the grid x.  For periodic BC the
    integration constant is chosen so V has zero spatial mean.
    For dirichlet BC ditto; the gauge choice is irrelevant to u.
    """
    N = x.size
    if source_fn is None:
        return np.zeros(N)
    g = source_fn(x, t)
    dx = x[1] - x[0]
    # antiderivative G(x) = ∫_0^x g(x') dx' on grid (trapezoid)
    G = np.concatenate(([0.0], np.cumsum((g[:-1] + g[1:]) * dx / 2)))
    V_raw = -G / (2.0 * nu)
    return V_raw - V_raw.mean()   # gauge: spatial mean zero
```

### 5.4 `heat_dense_block_step_circuit` — accept potential

Add a `V: np.ndarray | None = None` parameter at
[burgers_cole_hopf_circuit.py:225-279](../src/burgers_cole_hopf_circuit.py:225).
Modify the propagator math:

```python
def heat_dense_block_step_circuit(
    q, nu, dt, L_box,
    bc="periodic",
    encoding="binary",
    V: np.ndarray | None = None,        # NEW: potential on grid (length N)
) -> QuantumCircuit:
    N = 1 << q
    dx = L_box / N
    L_dense = build_laplacian_dense(N, dx, bc=bc)
    L_dense = permute_operator(L_dense, q, encoding)

    M = nu * L_dense * dt
    if V is not None:
        # V is in grid order; permute to encoded order for consistency
        # with L_dense.  permute_operator on a diagonal: same as
        # permuting the diagonal vector then re-diagonalising.
        V_perm = permute_to_encoding(V, q, encoding) \
            if encoding != "binary" else V
        M = M - np.diag(V_perm) * dt

    # ... existing eigendecomp + block-encoding code unchanged ...
```

(`permute_to_encoding` already exists in `burgers_encoding.py`.)

The eigendecomposition, controlled-Ry, ancilla-post-select machinery
all work unchanged on the modified M. M is still real symmetric.

### 5.5 `heat_dense_block_full_circuit` — per-step V

Today this builder constructs ONE `step_qc` and inlines it `N_steps`
times ([burgers_cole_hopf_circuit.py:282-321](../src/burgers_cole_hopf_circuit.py:282)).
With time-dependent V, each step needs its own step_qc. Restructure
to a per-step build loop:

```python
def heat_dense_block_full_circuit(
    q, nu, T, N_steps, L_box,
    bc="periodic", encoding="binary",
    source_fn=None, x=None,           # NEW
):
    """If source_fn is given, also pass x (grid coords)."""
    from burgers_potential import potential_from_source
    dt = T / N_steps
    # ... register setup unchanged ...

    for step_idx in range(N_steps):
        if source_fn is not None:
            t_mid = (step_idx + 0.5) * dt
            V_n = potential_from_source(source_fn, x, t_mid, nu, bc=bc)
        else:
            V_n = None
        step_qc = heat_dense_block_step_circuit(
            q, nu, dt, L_box, bc=bc, encoding=encoding, V=V_n,
        )
        # ... existing inline + remap-ancilla-bit code, unchanged ...
```

`source_fn` and `x` flow down from `_run_shots_batch` (and the SV
path's `_build_step_sv` for shots=0).

### 5.6 `_run_shots_batch` — forward source

Add `source_fn=None`, `x=None` kwargs. Pass through to
`heat_dense_block_full_circuit`. The SV path
(`run_cole_hopf_circuit_sv` + `_build_step_sv`) needs the same
treatment for the statevector branch — it currently builds a single
step_qc and reuses it ([burgers_cole_hopf_circuit.py:408-462](../src/burgers_cole_hopf_circuit.py:408)).
For source, the SV path also needs per-step propagator rebuild.

Implementation note: `_build_step_sv` with V plumbing has the same
shape as 5.5 above — replace single step_qc construction with a
per-step builder list `step_qcs[s]`.

### 5.7 Caller — `run_cole_hopf_circuit_simulation`

Inside this function ([burgers_cole_hopf_circuit.py:592+](../src/burgers_cole_hopf_circuit.py:592)),
construct nothing new — just thread `source_fn` and `x` through to
`_run_shots_batch` (shots path) and `run_cole_hopf_circuit_sv`
(shots=0 path).

The forward Cole-Hopf transform on `u0` is unchanged: φ₀ = exp(−U/(2ν))
where U is the antiderivative of u₀. The source affects the
EVOLUTION operator, not the IC.

The inverse Cole-Hopf at the end is also unchanged.

## 6. q8020 TOML smoke case

Add a sibling to the existing sine-wave case in
`input/burgers_quantum.toml`:

```toml
[cole_hopf_circuit_forced_q5_smoke]
"--method" = "cole_hopf_circuit"
"--propagator" = "dense-block"
"--ic" = "sine"
"--source" = "sine"
"--nu" = 0.1
"--cfl" = 0.1
"--shock-pct" = 100.0
"--q" = 5
"--shots" = 50000
"--backend-type" = "sim"
"--seed" = 42
"--save-every" = 1
"_group_postproc" = "python ./q8020-mps-burgers/docs/plot_cole_hopf_circuit_evolution.py"
```

Smoke target: `final_error < 0.05` (vs ~0.21 today on the broken-
forcing config).

## 7. Tests

In `tests/test_source_forcing.py`:

1. **`test_potential_unforced_returns_zero`** — `source_fn=None` →
   `V` is exactly the zero vector. Regression gate; the dense-block
   path with `V=None` must produce bitwise-identical circuits to
   the pre-change code.
2. **`test_potential_paper_source_matches_analytical`** — for
   `source_fn=source_term_sine` at t=0.05 on a q=4 grid, compare
   the trapezoidal `potential_from_source` output to the closed-form
   `cos(2πx)·cos(2πt)/(4πν)` (gauge-fixed mean-zero). Tolerance
   1e-3 on q=4, tightening with q.
3. **`test_potential_gauge_invariance`** — run
   `run_cole_hopf_circuit_simulation` with V replaced by `V + c`
   for arbitrary c. The reconstructed `u(x, T)` must match within
   shot-noise tolerance. Proves the gauge choice is harmless.
4. **`test_unforced_regression`** — full pipeline run with
   `source_fn=None` matches the pre-change pipeline output to 1e-12
   on the SV path and within shot-noise tolerance on the shots
   path. Catches an accidental V≠0 leakage into the unforced case.
5. **`test_forced_quantum_matches_classical_dense_block_sv`** — at
   q=5, ν=0.1, source=sine, run cole_hopf_circuit on the
   shots=0/SV path and compare against classical FTCS forced.
   Acceptance: relative L2 < 0.02 (no shot noise in this leg).
6. **`test_forced_quantum_matches_classical_dense_block_shots`** —
   same as #5 but shots=50000. Acceptance: relative L2 < 0.05
   (shot noise dominates).
7. **`test_qft_diagonal_with_source_raises`** — guard from §5.2
   fires cleanly with the documented error message.

## 8. Acceptance

- All 7 §7 tests pass.
- §6 smoke case from q8020 produces a `final_error < 0.05` and the
  evolution GIF shows a forced sine wave (sustained amplitude under
  the cos(2πt) drive in the time window) rather than the pure
  exponential decay seen today.
- Pure-quantum claim from P-G is intact: simulation core (state
  prep + propagator + measurement) contains no classical evolution
  step. The propagator builder is classical (it always was — it
  generates a Qiskit circuit) but the EVOLUTION inside the time
  loop is still a quantum circuit, no mid-loop decode/re-encode.

## 9. Implementation order

S1: §5.3 `burgers_potential.py` + tests #1, #2, #3. Pure classical,
    self-contained, fast feedback.
S2: §5.4 + §5.5 + §5.6 + §5.7 plumbing. Plus test #4 (regression)
    to catch any leakage on the unforced case.
S3: §5.1 dispatch wire-up. At this point CLI `--source sine` reaches
    the propagator. Tests #5 and #6 land.
S4: §5.2 guard for qft-diagonal+source. Test #7 lands.
S5: §6 TOML case. Manual q8020 sweep to confirm the GIF.

S1 is ~1-2 hours. S2 is the load-bearing change (~half a day —
mostly mechanical, but two paths to refactor: shots and SV). S3-S5
are short.

## 10. Out of scope (future work)

- `--propagator qft-diagonal` with source. Needs Strang split with
  diagonal-V block-encoded as a separate ancilla layer. Worth its
  own spec; cost is modest (extra controlled-phase layer per step
  and one extra ancilla, OR re-using the existing ancilla with
  reset).
- Adaptive `dt` for stiff V (high ν⁻¹ regions). Current spec uses
  fixed `dt`. If V is large enough that ‖exp(dt·M_n)‖ blows up
  P_success badly, refining dt is the answer.
- Multiple source modes (e.g., g(x,t) = sum of harmonics). Add as
  new `--source` choices when needed.
- `bond_dim` interaction with source forcing. The state-prep MPS
  bond-dim sweep should still work — V doesn't change the IC, only
  the propagator. But verify on a small case before claiming.
- Time-dependent ν. Out of scope; ν stays constant.
