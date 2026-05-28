# SPEC — End-to-end shots support for `cole_hopf_circuit` (sim → fake → hardware)

Self-contained handoff for an agent that has not seen prior
conversations.

## 0. Context

`q8020-mps-burgers` is a standalone Burgers / Cole-Hopf quantum-circuit
pipeline at `/Users/agallojr/proj/src/q8020-mps-burgers/`. Three
sibling repos provide reused infrastructure (registered as workspace
members of `/Users/agallojr/proj/src/q8020/`):

- `q8020-cfd-metautil` — argparse helpers, sweep harness, metadata
  fragment writers
- `q8020-cfd-qutil` — backend factory, transpile + execute helpers,
  hardware job submission

The end goal is **the full quantum path of `cole_hopf_circuit` runs on
real IBM hardware**. This spec is the next step toward that: get a
clean shots path with proper backend abstraction (sim / fake-backend
/ hardware-async) for the **`cole_hopf_circuit` method only**. Other
methods (`quantum_circuit`, `mps`, `tebd_circuit`) already use the
qutil backend factory — leave them alone.

## 1. Current state (verified, do not relitigate)

`burgers_solver.py` already exposes the flags via metautil's
`add_standard_quantum_args` (see
`q8020-cfd-metautil/src/q8020_cfd_metautil/args.py`):

- `--shots` (default 1024)
- `--backend` (fake-backend name, e.g. `manila`)
- `--t1`, `--t2` (custom thermal noise)
- `--coupling-map {default, all-to-all}`
- `--optimization-level {0,1,2,3}` (default 1)
- `--seed`

These already make it from CLI → `run_simulation` in
`burgers_trotter.py`. The methods `quantum_circuit`, `mps`, and
`tebd_circuit` correctly call
`q8020_cfd_qutil.backend.get_backend(name=backend_name,
backend_type="sim", t1=t1, t2=t2)` and pass the resulting AerSimulator
through every step ([burgers_trotter.py:735-737](../../src/burgers_trotter.py:735)).

**The gap is `cole_hopf_circuit`.** Two specific bugs:

1. **Dispatch drops the args.** In
   [burgers_trotter.py:713-720](../../src/burgers_trotter.py:713) the
   call to `run_cole_hopf_circuit_simulation` does not forward
   `backend_name`, `t1`, `t2`, `optimization_level`, `seed`, or
   `coupling_map`.
2. **Backend is hardcoded.** In
   [burgers_cole_hopf_circuit.py:537](../../src/burgers_cole_hopf_circuit.py:537)
   `_run_shots_batch` does `backend = AerSimulator()` with no noise,
   no coupling map, no shots-seed plumbing.

Net effect today: `cole_hopf_circuit` shots runs are ideal Aer
regardless of `--backend` / `--t1` / `--t2`. There is no path to a
fake backend, and no path to real hardware.

The shots=0 (statevector) path uses `qiskit.quantum_info.Statevector`
directly and does not need a backend. Leave that path alone.

## 2. Goal

`python burgers_solver.py --method cole_hopf_circuit --shots 4096
--backend manila --backend-type sim` runs a noisy AerSimulator with
Manila topology + Manila's calibration-derived noise.

Same for `--backend-type fake` (FakeBackendV2 directly via SamplerV2)
and `--backend-type hardware` (real IBM Quantum, async-submit semantics
to be designed in §6).

Across all three modes, the pipeline:

1. Builds the circuit batch (one circuit per snap_step) — unchanged.
2. Transpiles with the requested optimization level against the
   chosen backend — replaces the current hard-coded `optimization_level=1`.
3. Executes via the right path (Aer `backend.run` for sim,
   `SamplerV2` for fake/hardware) — `q8020_cfd_qutil.circuit`
   already abstracts this.
4. Performs ancilla post-selection on the counts dict — unchanged.
5. Reconstructs `phi_hat` and returns metrics including transpile
   info, gate counts, depth, P_success — extended with the new
   metrics from `transpile_circuit` and `execute_circuit_counts`.

## 3. Non-goals

- Refactoring methods other than `cole_hopf_circuit`. The
  q8020-mps-burgers repo carries other methods that work fine today
  with the qutil factory — do not touch them.
- Error mitigation (TREX, ZNE, dynamical decoupling) — separate spec
  if/when needed.
- Hardware-async session/batch optimisation. The hardware path uses
  qutil's existing `submit_job` / `get_job_result` (fire-and-poll);
  no Session/Batch modes in this spec.
- Any change to the shots=0 statevector path
  (`run_cole_hopf_circuit_sv`).
- Any change to `burgers_solver.py` argparse — all needed flags are
  there except one (§4).

## 4. CLI surface

One new flag, added by metautil for shared use across the project.

In `q8020-cfd-metautil/src/q8020_cfd_metautil/args.py`, extend
`add_backend_args`:

```python
parser.add_argument(
    "--backend-type", type=str, default="sim",
    choices=["sim", "fake", "hardware"],
    help="Backend execution mode (default: sim)",
)
```

This is a metautil change because every method in this repo (and
sibling repos like axequalsb) eventually wants the same surface. It
defaults to `sim` so existing TOML cases keep working.

`burgers_solver.py` doesn't need argparse edits — `--backend-type` is
picked up from `add_standard_quantum_args`.

## 5. Plumbing changes

### 5.1 `burgers_solver.py`

At the call site of `run_simulation`
([burgers_solver.py:237-254](../../src/burgers_solver.py:237)), add the
new args (already collected by argparse):

```python
sols_method, step_metrics = run_simulation(
    ...
    backend_type=args.backend_type,
    coupling_map=args.coupling_map,
    optimization_level=args.optimization_level,
    seed=args.seed,
    ...  # everything already passed today
)
```

In `make_case_meta` and the analysis-fragment dict, add:

- `backend_type=args.backend_type`
- `coupling_map=args.coupling_map`
- `optimization_level=args.optimization_level`
- `seed=args.seed`

So sweeps record what was actually run.

### 5.2 `burgers_trotter.py::run_simulation`

Extend signature with `backend_type: str = "sim"`,
`coupling_map: str = "default"`, `optimization_level: int = 1`,
`seed: int | None = None`.

In the `cole_hopf_circuit` branch
([burgers_trotter.py:713-720](../../src/burgers_trotter.py:713)),
construct the backend ONCE (matching the pattern at
[:734-737](../../src/burgers_trotter.py:734)) and forward it plus the
relevant args:

```python
if method == "cole_hopf_circuit":
    backend = None
    if shots > 0:
        from q8020_cfd_qutil.backend import get_backend
        backend = get_backend(
            name=backend_name,
            backend_type=backend_type,
            t1=t1, t2=t2,
            coupling_map=coupling_map,
        )
    sols, mets = run_cole_hopf_circuit_simulation(
        u0, x, nu, dt, n_steps, bc=bc,
        propagator=propagator, shots=shots,
        snapshot_interval=snapshot_interval,
        bond_dim=bond_dim, encoding=encoding,
        backend=backend,
        backend_type=backend_type,
        backend_name=backend_name,
        optimization_level=optimization_level,
        seed=seed,
    )
    return sols, mets
```

Build the backend only when `shots > 0`. Statevector path doesn't
need it.

### 5.3 `burgers_cole_hopf_circuit.py`

#### 5.3.1 `run_cole_hopf_circuit_simulation`

Extend signature:

```python
def run_cole_hopf_circuit_simulation(
    u0, x, nu, dt, n_steps,
    bc="periodic",
    propagator="qft-diagonal",
    snapshot_interval=1,
    shots=0,
    bond_dim=None,
    encoding="binary",
    backend=None,                  # NEW
    backend_type="sim",            # NEW
    backend_name=None,             # NEW (only used in hardware-async msg)
    optimization_level=1,          # NEW
    seed=None,                     # NEW (forwarded to AerSimulator)
):
```

Forward all five new params into `_run_shots_batch`.

#### 5.3.2 `_run_shots_batch`

Replace the hard-coded `backend = AerSimulator()` block at
[burgers_cole_hopf_circuit.py:536-547](../../src/burgers_cole_hopf_circuit.py:536)
with a call into qutil:

```python
from q8020_cfd_qutil.circuit import (
    transpile_circuit,
    execute_circuit_counts_joint,
)

# Caller built the backend; we just transpile + execute.
# transpile_circuit takes a single circuit; loop over the batch.
# (Future P-* parcel: extend qutil to accept batches.)
transpile_metrics: list[dict] = []
results: list[tuple[np.ndarray, dict]] = []
for raw_qc, s in zip(raw_circs, snap_steps):
    qc_t, t_info = transpile_circuit(
        raw_qc, backend, optimization_level=optimization_level,
    )
    transpile_metrics.append(t_info)
    counts, exec_info = execute_circuit_counts_joint(
        qc_t, backend, shots=shots, seed=seed,
    )
    # ... existing post-selection on `counts` (joint dict) ...
    met.update({
        "transpile": t_info,
        "execute": exec_info,
    })
    results.append((phi_hat, met))
```

**The crucial detail: joint counts across registers.** The shots
circuits have two classical registers, `data` (q bits) and
`anc_hist` (one bit per Trotter step). Post-selection at
[:548-578](../../src/burgers_cole_hopf_circuit.py:548) needs the joint
distribution — counts of `(data=X AND anc=Y)` on the SAME shot.
This is NOT recoverable from per-register marginals.

The four data sources differ here:

| Path | What you get | Joint? |
|---|---|---|
| `AerSimulator.run().get_counts()` | one dict, keys `"data anc"` (space-joined) | yes |
| `SamplerV2 ... data.<reg>.get_counts()` | one Counter PER register | no — marginals only |
| `SamplerV2 ... data.<reg>.array` (per-shot bits) | ndarray, one row per shot | yes (zip rows) |
| `SamplerV2 ... pub_result.join_data().get_counts()` | one dict, joined bitstring | yes |

The current `q8020_cfd_qutil.circuit.execute_circuit_counts`
([circuit.py:114-127](../../../q8020-cfd-qutil/src/q8020_cfd_qutil/circuit.py:114))
returns marginals — first register only — which silently breaks our
post-selection on V2-Sampler paths. We need joint counts.

**Recommended fix: extend qutil with a joint-counts helper.**
Adding `execute_circuit_counts_joint(qc, backend, shots, seed=None)`
to `q8020-cfd-qutil/src/q8020_cfd_qutil/circuit.py` is the right
answer because:

- Other consumers in the workspace (axequalsb, future repos) will
  hit the same multi-register issue. Solving it once in qutil is
  cheaper than solving it three times in three repos.
- Named registers carry semantic meaning (`anc_hist` vs `data`).
  Collapsing into one creg in the burgers circuit would lose that
  signal in transpiler output and in any downstream debugging.
- The fix is small (~20 lines).

Implementation in qutil (new function alongside the existing
`execute_circuit_counts`, do not break the existing one):

```python
def execute_circuit_counts_joint(
    qc_transpiled, backend, shots=1024, seed=None,
):
    """Execute and return JOINT counts across all classical registers.

    Returns (counts, exec_info) where `counts` is a dict whose keys
    are bitstrings concatenated across registers in
    most-recently-added-first order — matching Aer's
    `result.get_counts()` convention for multi-register circuits
    (with the inter-register space stripped).
    """
    is_aer = _is_aer(backend)
    t0 = time.time()

    if is_aer:
        kwargs = {"shots": shots}
        if seed is not None:
            kwargs["seed_simulator"] = seed
        result = backend.run(qc_transpiled, **kwargs).result()
        raw = result.get_counts()
        # Aer joins multi-creg with spaces — strip to align with V2 path.
        counts = {k.replace(" ", ""): v for k, v in raw.items()}
        exec_info = {
            "wall_time": time.time() - t0,
            "shots_requested": shots,
            "shots_executed": result.results[0].shots,
            "backend_time": result.results[0].time_taken,
        }
    else:
        from qiskit_ibm_runtime import SamplerV2 as Sampler
        sampler = Sampler(backend)
        if seed is not None:
            sampler.options.seed_simulator = seed   # primitive option
        job = sampler.run([qc_transpiled], shots=shots)
        pub_result = job.result()[0]
        # join_data gives one BitArray covering all classical registers,
        # preserving the per-shot correlation we need.
        counts = pub_result.join_data().get_counts()
        exec_info = {
            "wall_time": time.time() - t0,
            "shots_requested": shots,
            "job_id": job.job_id(),
        }

    return counts, exec_info
```

Note the bitstring layout: Aer's `get_counts()` returns
`"<data> <anc_hist>"` (later-added register on the LEFT, with a
space separator). After stripping the space, the burgers
post-selection reads `data_bits = bitstring[:q]` and
`anc_bits = bitstring[q:]` — which matches the existing fallback
branch at [burgers_cole_hopf_circuit.py:556-557](../../src/burgers_cole_hopf_circuit.py:556)
already. V2's `join_data().get_counts()` produces the same layout
(no space). Either way the post-selection code at [:551-562]
needs ONE small change: drop the `bitstring.split()` branch and
always slice by position. Verify with the parity test in §7.

**Fallback (do not pursue unless qutil owners push back):** collapse
the two registers in the burgers circuits to a single
`ClassicalRegister(q + N_steps, "all")` and slice positions in
Python. Keeps qutil unchanged, but loses semantic register names in
transpiled circuits and forces a circuit-construction edit at three
places ([:198-220, :297-318, and the heat_qft_full_circuit
counterpart](../../src/burgers_cole_hopf_circuit.py:198)).

#### 5.3.3 Hardware mode (async)

When `backend_type == "hardware"`, do NOT execute synchronously inside
`_run_shots_batch`. Instead:

1. Build all `raw_circs` (already done).
2. Call `q8020_cfd_qutil.job.submit_job(raw_circs, backend_name=...,
   shots=shots, optimization_level=optimization_level)`. This
   transpiles internally and submits a SamplerV2 batch.
3. Print the returned `job_id` to stderr.
4. Return early with placeholder `phi_hat = np.full(N, np.nan)` and
   metrics containing `job_id`, `backend_name`, `submitted_at`,
   `shots_per_circuit`, so the q8020 sweeper records the submission.

A separate post-processor (out of scope: see §10) polls
`get_job_result(job_id)` — and once it has the
`PrimitiveResult`, calls `pub_result.join_data().get_counts()` to
get joint counts (the same shape `execute_circuit_counts_joint`
returns), then runs the post-selection / phi reconstruction /
cole_hopf_inverse offline. Reuse the same post-selection code path.

`get_job_result` ([qutil/job.py:99-101](../../../q8020-cfd-qutil/src/q8020_cfd_qutil/job.py:99))
currently calls `_extract_counts` which mirrors the marginal-only
behaviour of `execute_circuit_counts`. The hardware harvester
script will need either an `_extract_counts_joint` helper added to
qutil/job.py or to call `pub_result.join_data().get_counts()`
directly. Track this under future-work item §10.

This spec only delivers fire-and-record for hardware. Synchronous
hardware execution is not safe — IBM jobs queue for hours.

#### 5.3.4 Seed plumbing

`--seed` from the CLI becomes `args.seed`, threaded all the way to
the executor. Three seeds matter and they are NOT the same:

1. **Simulator-shot seed** — the RNG that draws shots from the
   final-state distribution. Aer reads it as
   `seed_simulator`; SamplerV2 reads it as
   `sampler.options.seed_simulator` (V2 primitives expose the same
   underlying option but through `options`, not the run kwargs).
   Both are honoured by the new `execute_circuit_counts_joint`
   above when `seed is not None`.
2. **Transpiler seed** — `transpile(..., seed_transpiler=seed)`
   makes layout/routing reproducible. Add this to qutil's
   `transpile_circuit` (it currently has no seed argument). Pass
   `seed` down from `_run_shots_batch` through `transpile_circuit`.
3. **Multimode IC seed** — already exists as `--ic-seed`,
   independent of `--seed`. Do not collapse them; `--seed` is for
   the quantum stack, `--ic-seed` is for the classical IC. Document
   this in the `--seed` help string.

Hardware mode (§5.3.3): IBM hardware does not honour
`seed_simulator` (it's literal hardware). `submit_job` must
silently ignore the seed for hardware paths but still record it in
the case-meta JSON for traceability.

Acceptance: two back-to-back invocations with the same `--seed`,
same `--shots`, same `--backend-type sim` produce byte-identical
counts dicts (test §7.6).

### 5.4 Metrics propagation

Whatever `transpile_info` and `execute_info` come back from qutil
gets folded into the per-step metrics dict and surfaces in the
`analysis_data["per_step_metrics"]` JSON via the existing fragment
writer. Specifically:

- `transpile.before.depth`, `transpile.after.depth`
- `transpile.before.gate_counts`, `transpile.after.gate_counts`
- `transpile.wall_time`
- `execute.wall_time`, `execute.shots_executed`,
  `execute.backend_time` (Aer only)
- `execute.job_id` (V2 Sampler / hardware only)

This makes A/B noise studies harvestable from q8020 sweeps without
extra plumbing.

## 6. q8020 TOML cases

Add three smoke cases to
`q8020-mps-burgers/input/burgers_quantum.toml` to validate each
backend mode end-to-end (numbers chosen to keep wall time low):

```toml
[cole_hopf_circuit_shots_smoke_ideal]
"--method" = "cole_hopf_circuit"
"--q" = 4
"--shots" = 4096
"--backend-type" = "sim"
"--n-steps" = 4
"--cfl" = 0.1
"--propagator" = "dense-block"

[cole_hopf_circuit_shots_smoke_noisy]
"--method" = "cole_hopf_circuit"
"--q" = 4
"--shots" = 4096
"--backend" = "manila"
"--backend-type" = "sim"
"--n-steps" = 4
"--cfl" = 0.1
"--propagator" = "dense-block"

[cole_hopf_circuit_shots_smoke_fake]
"--method" = "cole_hopf_circuit"
"--q" = 4
"--shots" = 4096
"--backend" = "manila"
"--backend-type" = "fake"
"--n-steps" = 4
"--cfl" = 0.1
"--propagator" = "dense-block"
```

The hardware case is documented in `docs/HARDWARE.md` (out of scope
for this parcel) — needs IBM credentials and a deliberate run.

## 7. Tests

Two test files. Most go in `q8020-mps-burgers/tests/`, but the joint-
counts helper lives in qutil so its test belongs there.

### 7a. In `q8020-cfd-qutil/src/q8020_cfd_qutil/test_circuit_joint.py`

1. **`test_joint_counts_aer_vs_v2_match_distribution`** — Build a
   small two-creg circuit (q=2 data + 1 anc, simple known state).
   Run via the new `execute_circuit_counts_joint` against
   AerSimulator and against `SamplerV2(FakeManilaV2)`. Both return
   joint dicts with the same key shape (`q+anc` total bits, no
   space). Assert key sets match exactly. Assert per-key shot
   counts agree within Poisson tolerance at shots=20000.
2. **`test_joint_counts_recover_correlation`** — Build a circuit
   that entangles the data and anc registers (e.g., Bell pair
   across the boundary). Joint dict must show ONLY the correlated
   bitstrings; if we accidentally call per-register get_counts and
   multiply marginals, the (anti-)correlation is lost. Asserts
   that the four-bin marginal-product distribution is NOT a match
   for the joint distribution returned by the helper. Sanity gate
   for "did we actually fix the marginal bug?"
3. **`test_seed_reproducibility_aer`** — Same circuit, same seed,
   two back-to-back calls to `execute_circuit_counts_joint`.
   Counts dicts must be byte-identical.
4. **`test_seed_reproducibility_v2_fake`** — Same as #3 but
   backend=FakeManilaV2 via SamplerV2. Counts must be
   byte-identical (V2 honours seed_simulator on fake backends).

### 7b. In `q8020-mps-burgers/tests/test_shots_backend.py`

5. **`test_cole_hopf_shots_ideal_sim_no_regression`** — Run the
   pre-change shots path at q=3, shots=2048, backend_type=sim,
   backend=None, t1=None, t2=None, seed=42. Assert `phi_hat`
   matches a stored reference within shot-noise tolerance
   (±0.05 L2). Proves the refactor introduced no regression on
   the ideal-Aer path.
6. **`test_cole_hopf_shots_noisy_finite`** — t1=50, t2=70,
   shots=2048, seed=42. Result is finite, `p_success > 0.1`.
   Proves noise plumbing reaches the simulator end-to-end.
7. **`test_cole_hopf_shots_fake_backend`** — backend=manila,
   backend_type=fake, shots=2048, seed=42. Result is finite.
   Proves V2 Sampler path works for fake backends end-to-end —
   this is the integration test that the joint-counts fix and the
   bitstring layout match for the actual circuits we ship.
8. **`test_seed_reproducibility_end_to_end`** — Two
   back-to-back invocations of `run_cole_hopf_circuit_simulation`,
   same seed, same args, backend_type=sim. `phi_hat` arrays
   identical bit-for-bit. (This is the §5.3.4 acceptance gate.)
9. **`test_metrics_carry_transpile_info`** — shots=2048, inspect
   returned metrics, assert `transpile.after.depth` and
   `execute.shots_executed` are present and sensible. Also assert
   `seed` is recorded in case-meta.

Hardware path: no automated test (would burn IBM time and is async).
Manual smoke documented in `docs/HARDWARE.md` (separate parcel).

## 8. Acceptance

- All three §6 TOML cases run cleanly via q8020 from
  `q8020-mps-burgers/`.
- All five §7 tests pass.
- For the noisy-sim case, `analysis.json` contains
  `t1`, `t2`, `backend_name=manila`, `backend_type=sim`,
  `coupling_map=default`, `optimization_level=1`, and per-step
  metrics include transpile depth before vs after.
- `pytest tests/` from the repo root: previous tests still green,
  plus the five new ones.

## 9. Implementation order

E1: **qutil extension** — `execute_circuit_counts_joint` +
    `seed_transpiler` plumbing in `transpile_circuit`. Tests 1-4
    in §7a land here. Lowest risk because the existing
    `execute_circuit_counts` is untouched. **Land this first** —
    everything downstream depends on it.
E2: §4 metautil flag (`--backend-type`).
E3: §5.1 + §5.2 plumbing (`burgers_solver.py` + `run_simulation`
    forwarding). At this point `cole_hopf_circuit` is no worse
    than today, but the new args reach the simulation function.
E4: §5.3.1 + §5.3.2 + §5.3.4 (`run_cole_hopf_circuit_simulation`
    and `_run_shots_batch` refactor onto qutil joint-counts +
    seed). This is where shots actually become noisy and
    reproducible. §7b tests 5, 6, 8, 9 land here.
E5: §5.3.3 hardware-async fire-and-record path. §7b test 7 (fake)
    also lands here since fake uses the same V2 Sampler code.
E6: §6 TOML cases.

E1 is in qutil (~half a day; small but in another repo, so
coordinate with whoever owns it). E2+E3 are pure plumbing
(~1-2 hours). E4 is the load-bearing change in this repo
(~half a day). E5 is small. E6 is configuration.

## 10. Out of scope (future work)

- A `harvest_hardware_jobs.py` post-processor that reads job IDs from
  results JSON, polls `get_job_result`, completes the post-selection
  / cole_hopf_inverse / writes a "hardware_completed" fragment. This
  parcel also adds `_extract_counts_joint` (or equivalent) to
  `q8020-cfd-qutil/job.py` so the harvester gets joint counts —
  see §5.3.3 caveat.
- Batched transpilation in qutil (`transpile_circuit` currently takes
  one circuit; we loop. A batched form would amortise transpile cost
  and is on `q8020-cfd-qutil`'s roadmap, not this repo's).
- Session/Batch mode for hardware to amortise queue time across
  snap_steps.
- Error mitigation (TREX, ZNE, DD).
- A `--backend-type ideal` alias for `sim` with no name. Today
  `backend_type=sim` + `backend=None` already gives ideal Aer; alias
  is sugar.

## 11. Addendum — fold `_joint` into `execute_circuit_counts`

Decision: replace the existing `execute_circuit_counts` body in-place
with the joint-counts implementation. Drop the `_joint` name.

**Why safe:** all three current callers (`ax_equals_b_hhl.py:419`,
`ax_equals_b_modular_hhl.py:549`, `ax_equals_b_hhl_qrisp.py:325`)
use `measure_all()` → single creg → joint and "first marginal" produce
identical dicts. Behaviour is unchanged for them.

Spec deltas applied at implementation time:

- **§5.3.2:** dropped the rename. `execute_circuit_counts` used
  everywhere. The change is in-place.
- **§5.3.3:** `_extract_counts` in `qutil/job.py` upgraded to use
  `join_data().get_counts()` directly.
- **§7a:** tests renamed `test_counts_*`; they exercise the upgraded
  `execute_circuit_counts`.
- **§9 E1:** "qutil extension" → "qutil upgrade — replace
  `execute_circuit_counts` body with joint-counts impl, add
  `seed_transpiler` to `transpile_circuit`."
- **§10:** struck the `_extract_counts_joint` callout;
  `get_job_result` already returns the right thing once
  `_extract_counts` is upgraded.

**Gotcha:** Aer's `result.get_counts()` produces space-joined keys for
multi-creg circuits. The upgraded function strips the space so
single-bitstring keys are the contract regardless of backend or register
count. Single-creg callers see no space to strip — no-op. Documented
in the function docstring.

**Migration:** none required. axequalsb continues to work unchanged.
`q8020-mps-burgers` `_run_shots_batch` slices
`data_bits = bitstring[:q]; anc_bits = bitstring[q:]`.
