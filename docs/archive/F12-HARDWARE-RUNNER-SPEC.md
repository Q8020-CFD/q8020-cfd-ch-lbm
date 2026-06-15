# F12 — Cole-Hopf Hardware Runner Spec

Date: 2026-06-14
Scope: a new standalone driver app that executes the existing Cole-Hopf
measure-reprepare segment loop on **real IBM Quantum hardware** (Heron-class,
e.g. Boston) — and, identically, on a simulator for dry-run testing — while
leaving the existing CH solver pathway essentially untouched.

Companion docs: F10-IMPLEMENTATION-SPEC.md (segmented evolution v1, sim-only),
HANDOFF-tebd-and-cole-hopf.md, AB-BAKEOFF-guide.md (q4 hardware-probe TOMLs).


## 1. Motivation and the one non-negotiable constraint

The q4 A-B probe set established Cole-Hopf at q=4 as the only hardware-runnable
candidate: ~10 qubits, ~1006 CX/circuit, depth ~3060, post-selection success
0.96–0.997, sim rel-L2 ≈ 0.08–0.18. (qlbm is dead: 119k CX, rel-L2 0.87.)

The CH solver runs a **measure-reprepare segment loop**: each segment is one
circuit (prep + `segment_size` Trotter steps), measured; amplitudes are
reconstructed classically from finite counts (`sqrt(n_kept/shots)`), an MPS is
re-prepared from that *measured, noisy* profile, and the next segment continues.

**Non-negotiable: the segments are intrinsically serial and MUST be run as N
separate QPU invocations.** Segment k+1's circuit cannot be built until segment
k's counts return. The seam between segments — finite-shot readout noise,
sqrt-amplitude collapse, post-selection renormalization, MPS re-prep of a
measured profile — *is part of what the hardware run measures*. Collapsing the
chain into independent t0→tk circuits (valid in principle because CH is linear)
would delete the seam error and is therefore explicitly **out of scope** for the
primary run. ch_smooth = 3 invocations; shock = 6.


## 2. Design principle: do not touch the CH pathway

The CH solver already passes its `backend` object *into*
`_run_shots_measure_reprepare` (burgers_cole_hopf_circuit.py:1077, default
`None`); it does not construct it internally. The execution seam is already
backend-polymorphic:

- `transpile_circuit(raw_qc, backend, ...)` — burgers_cole_hopf_circuit.py:1223
- `execute_circuit_counts(qc_t, backend, shots, seed)` — :1250
- `execute_circuit_counts` (qutil/circuit.py:213) **already branches**: AerSimulator
  → `backend.run().result()`; otherwise → `SamplerV2(backend).run(...).result()`
  (qutil/circuit.py:254). So a real `IBMBackend` already flows through correctly
  at the execute layer.

There is exactly **one** thing in the CH path that blocks hardware: the
top-of-function guard

```python
if backend_type == "hardware":          # burgers_cole_hopf_circuit.py:1098-1102
    raise NotImplementedError(
        "segmented evolution v1: sim only; hardware segmenting deferred to v2"
    )
```

### 2.1 The single permitted CH-side change

Relax that guard so a hardware backend is allowed through. Two acceptable forms,
in order of preference:

- **Preferred (zero behavioral change for existing runs):** gate the raise behind
  a new opt-in kwarg, e.g. `allow_hardware: bool = False`, defaulting `False`.
  Existing callers see identical behavior (still raises). The new runner passes
  `allow_hardware=True`. This is a ~3-line, backward-compatible change.
- Alternative: drop the raise entirely. Rejected — it silently changes the
  contract for every existing caller and removes a useful guard for accidental
  hardware submission from the sweeper.

**No other edit to burgers_cole_hopf_circuit.py, burgers_fw.py, or
burgers_solver.py is in scope.** The `isinstance(backend, AerSimulator)` checks
(:1196-1202, the LCU transpile-skip) already do the right thing for a non-Aer
backend (they evaluate False → transpile runs, correct for hardware). The
`execute_circuit_counts` SamplerV2 branch already exists. Nothing else breaks.

### 2.2 The Session problem (why a thin wrapper, not just a backend swap)

`execute_circuit_counts` calls `SamplerV2(backend).run(...).result()` and blocks
on `.result()` (qutil/circuit.py:254-262). On hardware that means **each of the
N segments opens its own primitive job and eats the queue separately** — N queue
waits. We want one queue wait, with the device held across the N serial
round-trips (classical reconstruction between segments is sub-second).

A `qiskit_ibm_runtime.Session` does exactly this: opened around the whole
segment loop, every `SamplerV2.run` inside it is dispatched within the held
session. **Crucially, `Session` is established by context, not by argument** —
when a `SamplerV2` is constructed inside an active `Session`/with a `mode=session`
handle, its jobs route through that session. So the runner can open a Session and
call straight into the *unmodified* CH loop; the existing
`execute_circuit_counts` → `SamplerV2(backend)` calls will be captured by the
active session automatically. This is the "session trick": **no change to
execute_circuit_counts is required** if the runner establishes the session as
ambient context around the call into the CH simulation.

**Decision (supersedes the "ambient capture" sketch above): the runner
constructs the Sampler.** Because error mitigation (§2.3) must be set on the
Sampler object via `sampler.options`, ambient-session capture is insufficient —
we need a handle to the Sampler to attach options. The plan is therefore the
additive qutil helper: `execute_circuit_counts(..., session=None,
sampler_options=None)` — when a session/options are passed (non-Aer backend), it
builds `SamplerV2(mode=session)` and applies the options; when omitted, the path
is **byte-identical** to today (existing sim/hardware callers unaffected). The
runner opens the `Session`, builds the options dict once, and threads both into
the CH call so every segment's Sampler is session-bound and mitigated. This is a
small *additive* change to qutil/circuit.py, not a behavioral change to the
default path.


### 2.3 Error mitigation: TREX (+ dynamical decoupling)

**Chosen technique: TREX — Twirled Readout Error eXtinction**, native to
SamplerV2 (`options.resilience.measure_mitigation` / `options.twirling`), **no
new dependency**.

Rationale (the fork resolved):
- **ZNE/PEC are inapplicable** — they mitigate Estimator *observable expectation
  values*; our output is a raw sampled histogram (Sampler). Eliminated by the
  readout architecture (§8), not by preference.
- **The corrupted quantity is a measurement.** We read a 16-bin histogram, take
  `sqrt(counts/n_kept)`, then differentiate the log — readout assignment error
  biases every bin and the sqrt+derivative *amplify* it. Measurement mitigation
  targets exactly what corrupts our specific output.
- **TREX over M3:** M3 needs `mthree` (confirmed NOT in uv.lock — would be a new
  dep) and mainly pays off at larger qubit counts. At 10 qubits TREX suffices and
  is dependency-free / native to the Sampler we already use.
- **Companion, not a second technique: dynamical decoupling.** Enabled in the
  *same* options block (`options.dynamical_decoupling.enable = True`), it fills
  idle windows over the depth-3060 circuit (~130 µs vs T2 ~100–200 µs — the
  "secondary limit" from the q4 analysis). One extra line, same wiring.

What this stack does NOT fix: the ~1006 CX gate-error wall (the dry-run, §7
step 3, quantifies it). TREX cleans the most directly-correctable,
most-workflow-aligned error; ancilla post-selection independently rejects some
error events. TREX + DD + post-selection is the v1 mitigation stack.

Options are surfaced as CLI flags (default ON for `--target hardware`):
`--measure-mitigation/--no-measure-mitigation`, `--dynamical-decoupling/--no-dd`.
On `--target sim` they default OFF (ideal sim) but may be forced on to exercise
the wiring against a noise model.


## 3. The new app

### 3.1 Location and name

`q8020-mps-burgers/src/burgers_ch_hw_runner.py` — a standalone CLI driver. It is
**not** a new solver method; it is an orchestrator that reuses
`run_cole_hopf_circuit_simulation` / `_run_shots_measure_reprepare`.

### 3.2 What it does (happy path)

1. Parse CLI (§4). Resolve the case (shock | smooth) → physics + segment params.
2. Build the backend via the existing `q8020_cfd_qutil.backend.get_backend`:
   - `--target sim` → `backend_type="sim"` (optionally a fake-backend noise model
     via `--fake-backend boston`-style name, reusing existing noise plumbing).
   - `--target hardware` → `backend_type="hardware"`, name from `--backend-name`
     (e.g. `ibm_boston`), credentials resolved by the existing
     `get_service` chain (arg token → `IBM_QUANTUM_TOKEN` → saved account).
3. **Free pre-flight transpile** against the resolved backend's coupling map
   (heavy-hex for hardware) to report the *true* CX/depth before spending shots.
   Use `transpile_circuit` on the first segment's circuit; print and record.
   Honor a `--dry-run` flag that stops here (no execution).
4. Open a `Session(backend=...)` (hardware) — or no session for sim — and build
   the `sampler_options` dict (§2.3: TREX measure-mitigation + DD when enabled).
   Invoke `run_cole_hopf_circuit_simulation(..., backend=backend,
   backend_type=<...>, allow_hardware=True, evolution_mode="measure_reprepare",
   segment_size=<case>, session=<session>, sampler_options=<opts>, ...)`. The CH
   loop runs its N serial segments; each segment's Sampler is built session-bound
   and mitigated via the §2.2 qutil helper.
5. Consume the return value: `list[tuple[np.ndarray, dict]]` — the snapshot
   phi-states and per-segment metric dicts (the same `per_step_metrics` shape
   already produced: `shots, n_kept, p_success, segment_idx, circuit_depth,
   gate_counts, job_id, execution_time_s, ...`).
6. Post-process: inverse Cole-Hopf (`burgers_cole_hopf.cole_hopf_inverse`) on the
   final phi to recover u(x); compute rel-L2 vs the FTCS/classical reference if
   available; emit the standard plots only if `--show`/not `--noshow`.
7. Write the full q8020 metadata bundle (§5) including the backend calibration
   snapshot and every per-segment `job_id`.

### 3.3 What it explicitly does NOT do

- Does not implement its own circuit construction, prep, Trotter, or
  post-selection — all reused from the CH module.
- Does not batch segments or run independent t0→tk circuits (§1).
- Does not modify argparse/flags of burgers_solver.py.


## 4. CLI (AS BUILT)

```
burgers_ch_hw_runner.py
  --case {shock,smooth}        required. Selects physics + segment count.
                               shock  -> A=0.4, nu=0.03, n_steps=60, seg=5 (12 segs)
                               smooth -> A=0.3, nu=0.08, n_steps=30, seg=5 (6 segs)
                               (values mirrored from the q4 probe TOMLs; see §4.1)
                               NOTE: segments = n_steps/segment_size = the number
                               of SERIAL QPU invocations (12 / 6), NOT the snapshot
                               count. Earlier "6/3" in drafts were save-every
                               snapshots; corrected here.
  --target {sim,local,hardware}  default sim.
                               sim      = AerSimulator (+ optional fake-backend
                                          noise via --backend-name). Aer path,
                                          no Sampler/Session, no mitigation effect.
                               local    = SamplerV2(mode=FakeBackendV2) qiskit
                                          local-testing mode. Exercises the EXACT
                                          hardware code path (SamplerV2 + Session +
                                          TREX/DD plumbing) with NO credentials and
                                          realistic device noise. CAVEAT: local mode
                                          ACCEPTS but numerically IGNORES TREX/DD
                                          (the runner prints this). Use it to
                                          validate the path + see the noisy gate-
                                          error floor; not the mitigation benefit.
                               hardware = real IBM via QiskitRuntimeService.
  --backend-name NAME          IBM backend for hardware (e.g. ibm_boston); or a
                               fake-backend name for sim noise (e.g. boston).
  --shots N                    default 150000 (matches probe runs).
  --segments N                 override the case default (3 or 6). Optional.
  --optimization-level {0..3}  default 3 for hardware, 1 for sim.
  --dry-run                    transpile + report CX/depth, then stop. No shots.
  --no-session                 disable the Session wrapper (debug; N separate jobs).
  --outdir PATH                sweep-compatible output root (default ~/q8020).
  --seed N                     default 1234.
  --token / --channel / --instance   IBM creds passthrough (else env/saved).
  --noshow                     suppress plots (default for hardware).
```

### 4.1 Case parameter source

The shock and smooth parameter sets are taken verbatim from the existing probe
TOMLs (`q8020_burgers_ab_shock_q4.toml`,
`q8020_burgers_ab_show_q4_ch_smooth.toml`): q=4, cfl=0.1, bc=periodic, ic=sine,
source=none, propagator=qft-diagonal, bond-dim=8, phi-modes=8,
evolution-mode=measure_reprepare. The runner hard-codes a small dict of these
two cases rather than re-parsing TOML, to stay a thin driver. (Smooth's exact
n_steps/segment-size to be read from its TOML at implementation time.)


## 5. Output / provenance (reuse metautil)

Reuse `q8020_cfd_metautil.meta_fragment` writers so output drops into the same
sweep directory structure the postproc + harvest tooling already understands:

- `make_experiment_meta`, `make_case_meta`, `make_code_meta`,
  `make_backend_meta(backend)` — the last captures the **real device calibration
  snapshot** (already used by job.py:get_job_result).
- Per-segment: persist each `job_id` (so a run can be reconstructed/audited
  after the fact), `p_success`, `n_kept`, transpiled depth/CX, `execution_time_s`.
- `write_results` / `write_analysis` for the recovered u(x), rel-L2, and timing.
- One run = one case = one experiment dir, N segment records inside.

This is the reason to let the runner write metautil fragments directly rather
than inventing its own format: the existing `plot_method_compare.py` /
harvest / compare tooling keeps working.


## 6. Sweeper relationship

Two viable wirings; pick at implementation time:

- **(A) Standalone, sweeper-launched (recommended).** The runner is a normal
  CLI script. A thin TOML (`burgers_ch_hw_<case>.toml`) with
  `_script = "python .../burgers_ch_hw_runner.py"` and `--target`,
  `--backend-name`, etc. lets the existing sweeper invoke it and own the
  output-dir hashing / `_group_postproc` exactly as for solver runs. This
  preserves sweep dir organization for free and is the path of least surprise.
- **(B) Pure standalone.** Run the script directly; it writes the metautil
  bundle itself (§5). Used for one-off hardware shots outside a sweep.

Both are supported by the same script; (A) is just (B) under the sweeper's
`_script` injection. No sweeper code change is required for (A) — it already
injects `--outdir` and runs arbitrary scripts.


## 7. Test / validation plan (sim first, then hardware)

1. **Sim parity (no noise):** `--target sim --case shock` must reproduce the
   existing q4 shock per-segment metrics (CX ~1007, depth ~3063, p_success
   0.977→0.988, 6 segments) — proving the runner reuses the CH loop faithfully
   and changes nothing numerically.
2. **Sim with Boston noise model + mitigation:** `--target sim --backend-name
   boston --measure-mitigation --dynamical-decoupling` — exercises the
   SamplerV2 + Session + TREX/DD options path end-to-end without QPU cost.
   Confirms recovered u(x) degrades gracefully AND that mitigation-on beats
   mitigation-off on the noisy sim (the cheap proxy for the hardware benefit).
3. **Dry-run heavy-hex transpile:** `--target hardware --backend-name ibm_boston
   --dry-run` — reports the true transpiled CX/depth on the real coupling map.
   Free (no QPU time). Gates the go/no-go and sets the error budget.
4. **Hardware, smooth first:** smallest serial chain (3 segments), highest
   post-selection headroom (0.994–0.997). Estimated billed QPU time ~1–2 min on
   Heron; wall clock dominated by one queue wait (Session-held). Then shock (6).

Compilation is not a test (Best-Practices §15); step 1 numeric parity is the
acceptance gate.


## 8. Open decisions (need human call before/within implementation)

- **Readout observable:** full 16-bin histogram → sqrt → log-derivative (current
  CH path, shot-hungry, gradient amplifies noise) vs a few moments of u(x). The
  current spec keeps the existing histogram path (zero CH change). Switching to
  an Estimator/observable path would be a larger change and is deferred.
- **Error mitigation: DECIDED — TREX + dynamical decoupling (§2.3).** This drove
  the wiring choice: Sampler must be runner/helper-constructed to attach options,
  so §2.2 (additive qutil `session=`/`sampler_options=`) is now the plan, not a
  contingency. Resolved; no longer open.
- **Session vs Batch:** Session (interactive, holds device) fits the serial
  reconstruct-between-segments pattern. Batch (fire-and-collect) does not, since
  segments are not independent. Spec assumes Session.


## 9. Summary of code touch surface

| File | Change | Size |
|---|---|---|
| burgers_cole_hopf_circuit.py | add `allow_hardware=False` kwarg; gate the line-1098 raise behind it; thread `allow_hardware`, `session`, `sampler_options` from `run_cole_hopf_circuit_simulation` → `_run_shots_measure_reprepare` → `execute_circuit_counts` | ~10 lines, backward-compatible (new kwargs default to today's behavior) |
| qutil/circuit.py | add optional `session=None`, `sampler_options=None` to `execute_circuit_counts`; when set (non-Aer), build `SamplerV2(mode=session)` + apply TREX/DD options | additive, default path byte-identical |
| burgers_ch_hw_runner.py | **new** standalone driver (§3, §4, §5); opens Session, builds mitigation options, post-process + metautil bundle | new file |
| TOML (optional) | `burgers_ch_hw_shock.toml` / `_smooth.toml` for sweeper launch (§6A) | config only |

Everything else — backend construction, transpile, execute/SamplerV2 branch,
post-selection, MPS prep, inverse transform, metadata writers — is reused as-is.


## 10. AS-BUILT STATUS (2026-06-14)

Implemented and validated on sim/local. Files:
- `q8020-cfd-qutil/src/q8020_cfd_qutil/circuit.py`: `execute_circuit_counts`
  gains `session=None`, `sampler_options=None` (+ `_apply_sampler_options`
  helper). Default path byte-identical.  The SamplerV2 branch also best-effort
  captures `job.metrics()` into `exec_info["job_metrics"]` and lifts
  `usage.quantum_seconds` to `exec_info["quantum_seconds"]` (the billed QPU
  time; absent/0 on sim/local, populated on real hardware).
- `q8020-mps-burgers/src/burgers_cole_hopf_circuit.py`:
  `_run_shots_measure_reprepare` and `run_cole_hopf_circuit_simulation` gain
  `allow_hardware=False`, `session=None`, `sampler_options=None`. The line-1098
  guard now fires only when `backend_type=="hardware" and not allow_hardware`.
  (Companion refactor by a parallel agent: segment construction extracted to
  `build_segment_circuit` — single source of truth shared by the loop and the
  dry-run; held-Session runs skip the in-loop classical metric-transpile, with
  `circuit_metrics_exact`/`_from_segment` honesty labels.)
- `q8020-mps-burgers/src/burgers_ch_hw_runner.py`: new driver.  Records
  per-segment `quantum_seconds` + `job_metrics` and a summed
  `total_quantum_seconds`; on real hardware it harvests per-segment
  execution-time metrics + calibration snapshots via the qutil builtin
  `get_job_result` (keyed on captured job_ids), written as a second results
  fragment.  All runtime metric payloads are run through a `json_safe` coercion
  (job.metrics() carries datetimes the fragment writer cannot serialize).
- `q8020-mps-burgers/input/ch-q4-hw/burgers_ch_hw_{smooth,shock}.toml`: sweeper
  launchers.

**Metrics provenance (the code-review fix):** the runner now gathers both
circuit-side stats (CX/depth/p_success/timing, via the CH loop) AND
hardware-side stats (billed `quantum_seconds`, execution-time calibration via
`get_job_result`) — the latter using the purpose-built `qutil/job.py` builtin
rather than re-deriving it.  On sim/local `quantum_seconds` is 0/None and the
hardware harvest is skipped (no creds); on real hardware both populate.

**Regression check:** `test_cole_hopf_circuit.py` — 16 pass; 2 fail
(`test_shots_convergence`, `test_11_5_shots_accuracy`). Both failures are
**pre-existing** (verified identical on a pristine git checkout via stash):
they call `run_cole_hopf_circuit_simulation(..., shots=...)` with `backend=None`,
which a newer `qiskit-ibm-runtime` rejects (`SamplerV2(None)` no longer defaults
a backend). Unrelated to F12 and out of scope. The measure_reprepare path F12
touches is covered by passing tests.

**Validated runs (credential-free):**
- `--target sim --dry-run` (shock & smooth): reports 12 / 6 segments, 300 2q
  gates / depth 679 against the ideal-sim basis.
- `--target local` (FakeCairo, 27q): full SamplerV2 + Session + options
  plumbing runs end to end; transpiled to 1535 2q gates on the fake heavy-hex
  layout — and the noisy fake backend drives p_success to ~2–3% and rel-L2
  off the scale, i.e. it correctly demonstrates the gate-error wall the q4
  analysis predicted. Metadata bundle written; `make_backend_meta` does not
  support FakeBackendV2 (non-fatal, caught — real IBM backends are supported).

**Handoff procedure for the IBM-access operator** (run top to bottom):
1. `--target sim --case smooth --dry-run` — sanity, free.
2. `--target local --case smooth` — exercise the exact hardware path, no creds.
3. `--target hardware --backend-name ibm_boston --case smooth --dry-run` —
   **the gating number**: true Boston heavy-hex CX/depth. Free (no QPU).
4. `--target hardware --backend-name ibm_boston --case smooth` — the real run.
   6 serial segments in one held Session; TREX + DD default ON. ~1–2 min billed
   QPU; wall clock = one queue wait. Then `--case shock` (12 segments).

**Known limitations / deferred (unchanged from §8):**
- Readout observable stays the histogram→sqrt→log-derivative path.
- `--target local` cannot show the *numerical* benefit of TREX/DD (qiskit local
  mode ignores them); only real hardware does. Runner warns about this.
- `make_backend_meta` lacks a FakeBackendV2 branch — cosmetic; add later if
  fake-backend provenance is wanted in local-run bundles.
