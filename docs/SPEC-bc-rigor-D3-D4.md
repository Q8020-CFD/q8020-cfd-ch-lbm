# SPEC — Close out D3/D4 boundary-condition rigor

Self-contained handoff. Reader has not seen the prior conversation.

## 0. Context

Two findings from the original
[REVIEW-murali-paper-fidelity.md](REVIEW-murali-paper-fidelity.md):

- **D3** ([REVIEW §6.D3](REVIEW-murali-paper-fidelity.md#d3)) — the
  legacy `burgers_classical.py::gradient_central` used one-sided FD
  at boundaries while `build_laplacian_matrix` used periodic
  wrap-around. Internal inconsistency in the legacy classical
  baseline.
- **D4** ([REVIEW §6.D4](REVIEW-murali-paper-fidelity.md#d4)) — the
  classical reference `solve_burgers` used the inconsistent legacy
  RHS while the quantum methods used `compute_rhs_shift` (fully
  periodic, shift-operator). The two paths solved subtly different
  PDEs at the boundaries; quoted L2 errors were contaminated at
  O(dx).

**Status today (2026-04-26).** Largely fixed in code:

- [`burgers_classical.py:140-191`](../src/burgers_classical.py:140) —
  `solve_burgers` calls `compute_rhs_shift(u, dx, nu, g, bc=bc)`,
  the same RHS the quantum methods use. The legacy
  `gradient_central`, `laplacian_central`, `euler_step`,
  `build_gradient_matrix`, and `build_laplacian_matrix` remain in
  the file but have docstrings marking them as "Legacy — not used
  by the solver." They are reachable only from the `__main__` smoke
  test and from `test_mpo.py`.
- [`burgers_nonlinear.py::compute_rhs_shift`](../src/burgers_nonlinear.py)
  takes `bc` and passes through to
  [`burgers_mpo.py::shift_matrix`](../src/burgers_mpo.py:61) which
  zeroes the wrap entries on `bc=dirichlet`. Plumbing is correct.

What remains is **rigor**, not refactor: prove the paths are
actually consistent, ringfence the legacy code, and re-state any
error numbers the team previously cited under the inconsistent
regime.

## 1. Goal

After this parcel:

1. There is a regression test that fails if the classical reference
   and any quantum method ever drift apart due to BC handling on
   the same `(IC, ν, T, dt, bc)`.
2. The legacy functions are either deleted or moved to a clearly-
   labelled `legacy/` location and explicitly excluded from
   production paths and from new tests.
3. Prior performance / fidelity numbers in our docs are audited:
   either confirmed still valid post-fix, or re-measured and
   updated, or marked "measured pre-D3/D4 fix; superseded by
   re-run YYYY-MM-DD."

## 2. Non-goals

- **Add Dirichlet support to direct-`u` quantum methods** that
  don't have it today (`quantum_circuit`, `mps`, `tebd_circuit`).
  Out of scope. They run periodic-only by design — see
  [OVERVIEW-burgers-solver.md §4.2](OVERVIEW-burgers-solver.md).
  Adding Dirichlet is a separate parcel; this spec only proves the
  *existing* BC handling is internally consistent.
- **Change the FTCS scheme.** `solve_burgers` stays forward-Euler +
  shift-operator FD; we are not switching to RK4 or upwind here.
- **Re-derive the BCs from the paper.** [DEEP-OVERLAP](DEEP-OVERLAP-murali-vs-ucan.md)
  notes the paper uses Dirichlet but our quantum paths default
  periodic. That meta-question is left where it is; this spec
  closes the *internal* consistency loop only.

## 3. Plumbing changes

### 3.1 Ringfence the legacy functions

Two options. Pick one:

**Option A (clean):** delete the legacy functions outright.

- Remove from
  [`burgers_classical.py`](../src/burgers_classical.py):
  `gradient_central`, `laplacian_central`, `euler_step`,
  `build_gradient_matrix`, `build_laplacian_matrix`.
- Remove the `__main__` smoke test (or rewrite it to call
  `solve_burgers` directly).
- Update `test_mpo.py` to import the dense matrices from a new
  helper module if it really needs them; or delete the relevant
  tests if the value they provide is reproduced by other tests.

**Option B (preserve for cross-validation):** move them to a
sibling module, named explicitly.

- Create `src/burgers_classical_legacy.py` containing exactly the
  five functions plus a top-of-file docstring stating "Legacy
  reference implementations retained for offline cross-validation
  against quimb MPO. Not on the solver code path. Do not import
  from production modules."
- Move the `__main__` smoke test from `burgers_classical.py` to
  the new module.
- Update `test_mpo.py` imports.
- `burgers_classical.py` then contains only `initial_condition_*`,
  `source_term_sine`, and `solve_burgers`.

Recommendation: **Option B** unless `test_mpo.py` is also being
deleted. The matrices are useful for validating the LCU circuits
in F3 (see [SPEC-F3-LCU-method.md](SPEC-F3-LCU-method.md)) — keep
them but offline.

### 3.2 New test file — `tests/test_bc_rigor.py`

Three regression tests. None of them require new physics; they
just lock in the current contract.

```python
"""Regression tests proving D3/D4 are closed.

If any of these fail, the classical reference and the quantum
methods have drifted apart on boundary handling.
"""

import numpy as np
import pytest

from burgers_classical import solve_burgers, initial_condition_sine
from burgers_nonlinear import compute_rhs_shift


# ── Test 1: solve_burgers and compute_rhs_shift agree byte-for-byte
#    on a single explicit-Euler step ────────────────────────────────

def test_solve_burgers_uses_compute_rhs_shift():
    """One Euler step of solve_burgers == one manual compute_rhs_shift.
    Catches any future regression where solve_burgers re-introduces
    its legacy RHS path."""
    N, q = 32, 5
    x = np.linspace(0, 1, N, endpoint=False)
    dx = x[1] - x[0]
    nu, dt = 0.1, 0.001

    u0 = initial_condition_sine(x)
    g = np.zeros_like(u0)

    sols = solve_burgers(u0, x, nu, dt, n_steps=1, source_fn=None,
                         bc="periodic")
    rhs = compute_rhs_shift(u0, dx, nu, g, bc="periodic")
    expected = u0 + dt * rhs

    np.testing.assert_allclose(sols[1], expected, atol=1e-14, rtol=0)


# ── Test 2: BC parameter actually affects the result ──────────────

def test_bc_periodic_vs_dirichlet_differ():
    """Run both BCs on the same IC and confirm the boundaries
    differ.  Catches regressions where bc= silently no-ops."""
    N = 32
    x = np.linspace(0, 1, N, endpoint=False)
    nu, dt = 0.1, 0.001

    # Use a non-symmetric IC so periodic vs dirichlet diverges:
    u0 = np.sin(2 * np.pi * x) + 0.3 * np.cos(4 * np.pi * x)

    sols_p = solve_burgers(u0, x, nu, dt, n_steps=10,
                           source_fn=None, bc="periodic")
    sols_d = solve_burgers(u0, x, nu, dt, n_steps=10,
                           source_fn=None, bc="dirichlet")

    # Boundary points must differ; interior may be similar
    assert not np.isclose(sols_p[10][0], sols_d[10][0])
    assert not np.isclose(sols_p[10][-1], sols_d[10][-1])


# ── Test 3: classical reference and shift-method match exactly ────

def test_shift_method_matches_classical_reference():
    """The 'shift' --method and the FTCS reference solve the SAME
    PDE on the SAME grid with the SAME BCs.  They must agree to
    machine precision (both are forward-Euler + compute_rhs_shift,
    just on different code paths)."""
    from burgers_fw import BurgersConfig, run_simulation_fw
    from q8020_cfd_metautil.solverfw import Grid1D

    N, q = 32, 5
    x = np.linspace(0, 1, N, endpoint=False)
    dx = x[1] - x[0]
    nu, dt, n_steps = 0.1, 0.001, 50

    u0 = initial_condition_sine(x)

    # Classical reference path
    sols_ref = solve_burgers(u0, x, nu, dt, n_steps,
                             source_fn=None, bc="periodic")

    # Shift method via solverfw
    grid = Grid1D.from_qubits(q, bc="periodic")
    cfg = BurgersConfig(q=q, nu=nu, cfl=0.1, dt=dt,
                        n_steps=n_steps, bc="periodic",
                        method="shift", ic="sine", source="none")
    sols_fw, _ = run_simulation_fw(cfg, grid, u0, source_fn=None)

    np.testing.assert_allclose(sols_ref[-1], sols_fw[-1],
                               atol=1e-12, rtol=0,
                               err_msg="shift method has drifted "
                                       "from FTCS reference; D3/D4 "
                                       "regression!")
```

### 3.3 Test gate

Add to the project's CI / pre-commit set. If anyone re-introduces
the legacy `gradient_central` into `solve_burgers` (or otherwise
desyncs the paths), Test 3 fails immediately.

## 4. Documentation audit

Files to inspect for prior error numbers and re-validate or
annotate:

- [`REVIEW-murali-paper-fidelity.md`](REVIEW-murali-paper-fidelity.md)
  — D3, D4 sections. Update with "Resolved YYYY-MM-DD; see
  [SPEC-bc-rigor-D3-D4.md](SPEC-bc-rigor-D3-D4.md) and test_bc_rigor.py."
- [`F10-IMPLEMENTATION-SPEC.md`](F10-IMPLEMENTATION-SPEC.md) and
  [`F10-REVIEW-PATCH.md`](F10-REVIEW-PATCH.md),
  [`F10-REVIEW-PATCH-02.md`](F10-REVIEW-PATCH-02.md) — any quoted
  L2 / fidelity numbers from runs predating the D3/D4 fix should
  carry a "measured pre-fix; see audit log" footnote, OR be
  re-measured. List the runs explicitly.
- [`HANDOFF-burgers-pipeline.md`](HANDOFF-burgers-pipeline.md) —
  same sweep.
- Any commit messages that quoted error numbers — leave alone
  (history), but the audit log should map old runs to "this number
  is no longer comparable" / "this number reproduces post-fix to
  within X."
- README.md tables of method-vs-error if any.

The audit log is a short markdown file:

```
docs/AUDIT-D3-D4-error-numbers.md

| Doc | Quoted number | Pre/Post-fix | Action |
|---|---|---|---|
| F10-REVIEW-PATCH-02.md §4 | final_error=0.21 | pre | re-run, replace |
| HANDOFF-burgers-pipeline.md "shift" baseline | ε=0.003 | pre | annotate |
| ... | ... | ... | ... |
```

This is the deliverable that gives "rigor of every error number we
cite." Without it, you've fixed the code but not the citations.

## 5. q8020 TOML — re-baseline runs

Add a small group of cases that produce the canonical error
numbers we'll cite from now on. These become the "measured post-
fix" reference set. Expected total wall time: a few minutes.

```toml
[bc_rigor_baseline_shift_periodic]
"--method" = "shift"
"--ic" = "sine"
"--source" = "none"
"--nu" = 0.1
"--cfl" = 0.1
"--shock-pct" = 100.0
"--q" = 5
"--bc" = "periodic"

[bc_rigor_baseline_shift_dirichlet]
"--method" = "shift"
"--ic" = "sine"
"--source" = "none"
"--nu" = 0.1
"--cfl" = 0.1
"--shock-pct" = 100.0
"--q" = 5
"--bc" = "dirichlet"

[bc_rigor_baseline_quantum_exact_periodic]
"--method" = "quantum_exact"
"--ic" = "sine"
"--source" = "none"
"--nu" = 0.1
"--cfl" = 0.1
"--shock-pct" = 100.0
"--q" = 5
"--bc" = "periodic"

[bc_rigor_baseline_cole_hopf_circuit_sv]
"--method" = "cole_hopf_circuit"
"--propagator" = "dense-block"
"--ic" = "sine"
"--source" = "none"
"--nu" = 0.1
"--cfl" = 0.1
"--shock-pct" = 100.0
"--q" = 5
"--shots" = 0
"--bc" = "periodic"
```

After running, record `final_error` for each in the audit log
above. These are now the post-fix reference numbers.

## 6. Acceptance

- [ ] §3.1 ringfence chosen and applied; legacy functions are
      either gone or in `burgers_classical_legacy.py` only.
- [ ] §3.2 three regression tests added; all pass.
- [ ] §3.3 tests run in the project's normal test suite.
- [ ] §4 audit document `AUDIT-D3-D4-error-numbers.md` exists and
      lists every previously-cited error number with its disposition.
- [ ] §5 four reference baseline cases run cleanly, numbers
      recorded in the audit log.
- [ ] [REVIEW-murali-paper-fidelity.md](REVIEW-murali-paper-fidelity.md)
      D3 and D4 sections gain a "Resolved" annotation pointing at
      this spec.

## 7. Implementation order

R1: §3.1 ringfence (decide A or B; do it). 30–60 min.
R2: §3.2 regression tests. ~1 hour.
R3: §5 re-baseline runs. 10 min walltime + 10 min recording.
R4: §4 audit pass — read each doc in the list, populate the audit
    table. ~1 hour because there's no dodging the read.

Total: half a day.

## 8. What this does NOT prove

This spec closes the *internal* consistency loop: classical
reference and quantum-method-on-shift are now provably solving the
same PDE. It does **not** prove either of them solves what the
*paper* solves — the paper uses Dirichlet on the direct-`u`
methods, our quantum direct-`u` methods are periodic-only.
Closing that gap is a separate parcel ("Add Dirichlet support to
quantum_circuit / mps / tebd_circuit"), called out as a non-goal
in §2.

The error numbers we cite *are* now internally consistent; whether
they match the paper's quoted numbers under matching BCs is a
different question that the audit in §4 is not asked to answer.
