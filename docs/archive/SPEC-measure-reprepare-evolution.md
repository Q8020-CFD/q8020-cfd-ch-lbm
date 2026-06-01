# SPEC — Measure-and-reprepare (segmented) evolution for shots-mode Cole-Hopf circuit (formerly "chunked evolution")

Self-contained handoff. Reader has not seen prior conversation.

## 0. Context

`q8020-mps-burgers` runs a Cole-Hopf quantum-circuit pipeline for 1D
Burgers. The shots path (`_run_shots_batch` in
[burgers_cole_hopf_circuit.py:532](../../src/burgers_cole_hopf_circuit.py:532))
today builds, for each requested snapshot at step `s`, an
*independent* full circuit that prepares the IC and inlines `s` step
layers, then runs `shots` shots. Two pathologies:

**Snapshot quadratic.** With `--save-every 1` and `n_steps = N`,
snap_steps = [1, 2, …, N]. Total step-layer construction +
simulation cost across all snapshots is `1+2+…+N = O(N²)`.
Concrete: a forced shock-pct=100 (n_steps=51) smoke run takes ~25
minutes of wall time, mostly back-loaded on the deep snapshots.

**Deep-circuit hang.** The shock-pct=1000 case (n_steps=513) builds
one circuit with 513 inlined dense-block step layers; transpiled
basic-gate depth is in the hundreds-of-thousands range. Mid-circuit
`measure(anc) + reset(anc)` between layers forces Aer into per-shot
trajectory simulation rather than evolve-once-sample-many. The case
hung overnight without producing output.

A third pathology, latent at deeper N: post-selection success
`P_success = ∏_step p_step ≈ p^N` decays multiplicatively with depth.
At `p ≈ 0.95`, `p^513 ≈ 10⁻¹¹` — effectively unobservable.

## 1. The architectural shift

**Allowed:** between quantum segments, classically read out
post-selected amplitudes, re-prep them as a fresh IC for the next
segment, continue evolving. The classical step is amplitude IO
(decode counts → re-encode via MPS prep), **not** PDE physics. No
classical solver advances `u` or `φ`; the only thing the classical
side knows how to do is "given amplitude vector, produce a state-prep
circuit." Same code path as t=0.

**Forbidden (unchanged):** classical co-solver running PDE physics
alongside the quantum (the paper's QNPU steerage). This spec does
not introduce that and explicitly rejects it.

The shift dissolves all three pathologies above.

## 2. Goal

Add a chunked-evolution mode for the shots path. Evolution is
partitioned into K chunks of `chunk_size = n_steps / K` steps each.
Per chunk: prep IC → propagator(chunk_size) → measure data →
reconstruct φ_chunk amplitudes → use as IC for next chunk. Snapshots
naturally land at chunk boundaries.

Acceptance: forced shock-pct=100 smoke run completes in < 5 min
(vs ~25 min today); shock-pct=1000 case completes at all (today:
hangs); final-time L2 vs FTCS within shot-noise+compounding floor
(estimate below).

## 3. Non-goals

- **SV path (`shots=0`).** [`run_cole_hopf_circuit_sv`](../../src/burgers_cole_hopf_circuit.py:420)
  already does one matvec per step and snapshots classically. No
  redundancy. Untouched.
- **Hardware path (`backend_type == "hardware"`).** Real quantum
  hardware can run chunked too, but the round-trip latency between
  shots-readout and re-submit makes it expensive. Default to
  single-circuit for hardware in v1; flag chunked as opt-in for
  hardware in v2. v1: chunked mode raises `NotImplementedError` for
  `backend_type == "hardware"`.
- **qft-diagonal propagator.** Same chunking scheme applies, but
  source-forcing is not yet implemented for qft-diagonal (see
  [SPEC-source-forcing.md](SPEC-source-forcing.md) §3 non-goals).
  v1 chunked supports qft-diagonal *unforced* and dense-block
  forced+unforced. Forced qft-diagonal lands when its source-forcing
  parcel does.
- **Adaptive chunk size.** Fixed `chunk_size` per run; no tuning at
  runtime based on observed P_success. Future work.
- **Sign of φ.** Cole-Hopf guarantees φ > 0; reconstruction from
  counts gives `phi = sqrt(probabilities) ≥ 0`. No sign-recovery
  surface area. (`u` can be negative; sign lives in the inverse CH,
  unchanged.)

## 4. Math: noise compounding

Single-circuit (today): one round of post-selection across all N
ancilla measurements, one round of data-readout shot noise. Final
amplitude std error per bin ≈ `1/√(P_succ_total · shots)` where
`P_succ_total = ∏ p_step`.

Chunked: K rounds, each with its own post-select and shot-noise.
Per chunk: `σ_chunk ≈ 1/√(P_succ_chunk · shots)` where
`P_succ_chunk = p^(chunk_size)`. Across K chunks the IC noise
propagates additively in variance under linear evolution (heat is
contractive — actually noise *decays* through the heat propagator,
which helps), but sampling noise is freshly injected at each chunk.
Final-time noise:

```
σ_final² ≈ K · σ_chunk² · (effective decay factor < 1)
σ_final  ≈ √K · σ_chunk
```

Compared to single-circuit `σ_single = 1/√(p^N · shots)`:

- At small N: single is fine, chunked adds √K overhead.
- At large N: `p^N` collapses, single becomes infeasible; chunked
  with `p^(N/K)` per chunk is bounded.

There exists a chunk_size `K* ≈ N · ln(p) / ln(σ_target/something)`
that minimizes total noise. For a smoke spec we pick a sane default
and let the user override.

**Default heuristic.** Choose `chunk_size` so `P_succ_chunk ≥ 0.5`,
i.e., `chunk_size ≤ ln(0.5)/ln(p)`. With `p ≈ 0.97` (typical
unforced ν=0.1, q=5), `chunk_size ≤ ~22`. Default to 10 — comfortable
margin, K=5 chunks for n_steps=51, K=51 for n_steps=513.

## 5. CLI surface

One new flag, one mode switch, no breaking change.

```
--evolution-mode {single, chunked}    default: single
--chunk-size INT                      default: 10 (only used when chunked)
```

`single` is exactly today's behavior. `chunked` activates the new
path.

For chunked + `--save-every K`: enforce `chunk_size == save_every`
(or document that snapshots may not align with chunk boundaries; v1
enforces equality with a clear error). This collapses two knobs that
fundamentally do the same thing.

Add to argparse in [`burgers_solver.py`](../../src/burgers_solver.py)
alongside the existing `--save-every`.

## 6. Plumbing changes

### 6.1 New driver — `_run_shots_chunked`

Sibling to `_run_shots_batch` in `burgers_cole_hopf_circuit.py`.
Signature mirrors the existing batch driver. Pseudocode:

```python
def _run_shots_chunked(
    psi0, q, snap_steps, dt, L_box, bc, propagator,
    shots, chunk_size, *, bond_dim, encoding, backend,
    backend_type, backend_name, optimization_level, seed,
    source_fn=None, x=None,
) -> list[tuple[np.ndarray, dict]]:
    """Chunked evolution for shots-mode shotted runs.
    snap_steps must align with multiples of chunk_size."""

    assert backend_type != "hardware", "chunked v1: sim only"
    assert all(s % chunk_size == 0 for s in snap_steps), \
        "snap_steps must be multiples of chunk_size"

    N = 1 << q
    psi_norm, init_norm = normalize_state(psi0)
    cumulative_norm = init_norm           # tracks ||phi|| across chunks
    snapshots: dict[int, tuple[np.ndarray, dict]] = {}

    psi_current = psi_norm                # input amplitudes for next chunk
    for chunk_idx in range(max(snap_steps) // chunk_size):
        # 1. Build prep circuit from current amplitudes
        tensors = classical_to_mps(psi_current, bond_dim=bond_dim,
                                    canonical="right")
        prep_qc = mps_to_circuit(tensors)

        # 2. Build chunk_size-step evolution circuit
        t_start = chunk_idx * chunk_size * dt
        full_qc = (heat_qft_full_circuit if propagator == "qft-diagonal"
                   else heat_dense_block_full_circuit)(
            q, nu, dt * chunk_size, chunk_size, L_box, bc=bc,
            encoding=encoding, source_fn=source_fn, x=x,
            t_start=t_start,                # NEW: see §6.4
        )

        # 3. Compose, transpile, run shots
        chunk_qc = init_qc + prep + full_qc + measurements
        qc_t, t_info = transpile_circuit(chunk_qc, backend,
                                          optimization_level, seed)
        counts, exec_info = execute_circuit_counts(qc_t, backend,
                                                    shots, seed)

        # 4. Post-select on ancilla=all-zero, reconstruct amplitudes
        n_kept, data_counts = post_select_joint(counts, q, chunk_size)
        p_success = n_kept / shots
        if n_kept == 0:
            # Chunk failed — propagate NaN snapshot, abort
            psi_current = np.full(N, np.nan)
            cumulative_norm = float("nan")
        else:
            psi_current = reconstruct_amplitudes(data_counts, n_kept, N)
            cumulative_norm *= np.sqrt(p_success)
            psi_current /= np.linalg.norm(psi_current)  # re-normalize
            # Note: amplitude norm = sqrt(p_success) of pre-post-select state.
            # After re-norm psi_current is unit; cumulative_norm tracks the
            # prefactor for inverse CH at the end.

        step_at_chunk_end = (chunk_idx + 1) * chunk_size
        if step_at_chunk_end in snap_steps:
            phi = psi_current * cumulative_norm   # rescale for output
            snapshots[step_at_chunk_end] = (phi, {
                "shots": shots, "p_success": p_success,
                "n_kept": n_kept, "step": step_at_chunk_end,
                "chunk_idx": chunk_idx, "chunk_size": chunk_size,
                "cumulative_norm": cumulative_norm,
                "transpile": t_info, "execute": exec_info,
            })

    return [snapshots[s] for s in snap_steps]
```

Two helper routines lift cleanly out of the existing
`_run_shots_batch`:

- `post_select_joint(counts, q, n_anc_bits) -> (n_kept, data_counts)`
- `reconstruct_amplitudes(data_counts, n_kept, N) -> np.ndarray` (the
  `phi_hat[idx] = np.sqrt(cnt / n_kept)` block, but un-rescaled).

Refactor those out of `_run_shots_batch` for shared use; both drivers
import them.

### 6.2 Dispatch — `run_cole_hopf_circuit_simulation`

Add `evolution_mode: str = "single"` and `chunk_size: int = 10`
parameters. In the shots branch (`shots > 0`), select driver:

```python
if shots > 0:
    if evolution_mode == "chunked":
        if propagator == "qft-diagonal" and source_fn is not None:
            raise NotImplementedError(
                "chunked + qft-diagonal + source pending; "
                "use --propagator dense-block"
            )
        results = _run_shots_chunked(...)
    else:
        results = _run_shots_batch(...)
```

### 6.3 Top-level dispatch — `run_simulation`

In [`burgers_trotter.py`](../../src/burgers_trotter.py) thread
`evolution_mode` and `chunk_size` from CLI args through to
`run_cole_hopf_circuit_simulation`. Mechanical kwarg add at the
existing dispatch block.

### 6.4 Time-dependent V across chunks

When `source_fn is not None`, each chunk's propagator builds V(x,t)
at midpoints `t = (step_idx + 0.5) * dt` where `step_idx` is the
*global* step, not chunk-local. Today `heat_dense_block_full_circuit`
([../src/burgers_cole_hopf_circuit.py:294](../../src/burgers_cole_hopf_circuit.py:294))
implicitly assumes `step_idx` starts at 0. Add a `t_start: float = 0.0`
parameter and use `t_mid = t_start + (step_idx + 0.5) * dt` so chunk
N+1 evaluates V at the correct global time.

Backwards-compatible: `t_start=0.0` default reproduces today.

### 6.5 CLI thread-through — `burgers_solver.py`

Add argparse:

```python
parser.add_argument("--evolution-mode", choices=["single", "chunked"],
                    default="single")
parser.add_argument("--chunk-size", type=int, default=10)
```

Validation: when `--evolution-mode chunked`, require
`n_steps % chunk_size == 0` and `save_every % chunk_size == 0` (or
chunk_size is a multiple of save_every). Fail fast with a helpful
error if not.

## 7. q8020 TOML smoke cases

Add two siblings to existing forced cases. Smoke (small) and the
deep one that hangs today:

```toml
[cole_hopf_circuit_forced_q5_chunked_smoke]
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
"--optimization-level" = 0
"--save-every" = 10
"--evolution-mode" = "chunked"
"--chunk-size" = 10
_group_postproc = "python ./q8020-mps-burgers/docs/plot_cole_hopf_circuit_evolution.py"

[cole_hopf_circuit_forced_q5_chunked_deep]
"--method" = "cole_hopf_circuit"
"--propagator" = "dense-block"
"--ic" = "sine"
"--source" = "sine"
"--nu" = 0.1
"--cfl" = 0.1
"--shock-pct" = 1000.0
"--q" = 5
"--shots" = 100000
"--backend-type" = "sim"
"--seed" = 42
"--optimization-level" = 0
"--save-every" = 10
"--evolution-mode" = "chunked"
"--chunk-size" = 10
_group_postproc = "python ./q8020-mps-burgers/docs/plot_cole_hopf_circuit_evolution.py"
```

`save-every=10` + `chunk-size=10` means snapshots align with chunk
boundaries. n_steps=51 → 5 chunks, 5 snapshots; n_steps=513 → 51
chunks, 51 snapshots.

## 8. Tests

`tests/test_chunked_evolution.py`:

1. **`test_chunked_unforced_matches_single_sv`** — at q=5, ν=0.1,
   no source, n_steps=20, chunk_size=5, shots=0 (SV reference):
   chunked output matches `run_cole_hopf_circuit_sv` to 1e-12 at
   every snapshot. Proves chunking is mathematically equivalent at
   zero-noise.
2. **`test_chunked_unforced_matches_single_shots`** — chunked vs
   single (today's `_run_shots_batch`) at q=5, ν=0.1, n_steps=10,
   chunk_size=5, shots=50000. Acceptance: relative L2 < 0.05 at the
   final snapshot. Within shot-noise, the two paths agree.
3. **`test_chunked_forced_matches_classical`** — at q=5, ν=0.1,
   source=sine, n_steps=50, chunk_size=10, shots=50000. Compare
   final u(x,T) to FTCS forced. Acceptance: relative L2 < 0.10
   (looser than test #2 because forcing + compounding noise).
4. **`test_chunked_t_start_correct`** — at chunk_size=5, n_steps=20,
   forced. Verify that V(x,t) for chunk k is evaluated at
   `t_start = k*5*dt`, not at the chunk-local 0. Numerical check
   against expected closed-form `cos(2πx)·cos(2πt)/(4πν)` at the
   midpoint of each chunk.
5. **`test_chunked_p_success_per_chunk_logged`** — metrics dict
   from each snapshot includes `p_success`, `chunk_idx`,
   `cumulative_norm`. Schema test only.
6. **`test_chunked_failure_propagates_nan`** — synthetic backend
   that returns zero kept counts; chunked driver returns NaN
   snapshot for that chunk and beyond, doesn't crash.
7. **`test_chunked_alignment_validation`** — `n_steps=51`,
   `chunk_size=10` (mismatch) raises an informative error before
   the run starts.
8. **`test_chunked_hardware_rejected`** — `backend_type="hardware"`
   + `evolution_mode="chunked"` raises `NotImplementedError` with a
   clear pointer.

## 9. Acceptance

- All 8 §8 tests pass.
- Smoke case (§7 first one) completes in < 5 minutes, GIF shows
  forced sine evolution, final L2 vs FTCS < 0.10.
- Deep case (§7 second one) completes in < 1 hour (today: hangs).
  Final L2 vs FTCS < 0.15 (looser — 51 chunks of compounding noise).
- Single-mode regression: today's `_smoke` and `_anim` cases
  (`evolution-mode=single` by default) produce bitwise-identical
  output to pre-change runs. Catches accidental coupling.

## 10. Implementation order

C1: §6.1 helpers — refactor `post_select_joint` and
    `reconstruct_amplitudes` out of `_run_shots_batch`, no behavior
    change. Tests: existing single-mode cases still green.
    (~1-2 hours)
C2: §6.4 `t_start` parameter on
    `heat_dense_block_full_circuit`. Default 0.0; existing callers
    unchanged. (~30 min)
C3: §6.1 `_run_shots_chunked` driver. Wire to test #1 and #2 first
    (unforced). (~half day)
C4: Dispatch wire-up §6.2, §6.3, §6.5. Tests #3, #5, #6, #7, #8.
    (~2 hours)
C5: §7 TOML smokes + manual q8020 sweep. Confirm GIFs.
    (~30 min)

Total: ~1.5 days. C1 and C2 are zero-risk refactors that can land
independently and stay reverted-friendly.

## 11. Out of scope (future work)

- **Hardware chunking.** v2: relax the §3 hardware non-goal. Each
  chunk submitted as separate hardware job, results read back, IC
  re-prepped. Cost is round-trip latency per chunk; may be infeasible
  for hardware where job queue dominates wall time.
- **Adaptive chunk size.** Monitor running `p_success`; if a chunk's
  P_succ drops below threshold, halve `chunk_size` for subsequent
  chunks. Reject if it doesn't recover.
- **Re-prep cost amortization.** Today each chunk runs its own MPS
  prep (`classical_to_mps` + `mps_to_circuit`). For very long runs
  the prep cost dominates. Possible: cache MPS tensors and update
  incrementally. Probably not worth it until measured.
- **qft-diagonal + source + chunked.** Lands when forced
  qft-diagonal does (its own SPEC).
- **Sign of u-amplitudes via two-circuit phase trick.** Out of scope
  here — `φ > 0` so amplitudes stay non-negative; sign lives in the
  classical inverse CH only.
- **Chunk-boundary error analysis** (rigorous, not heuristic). The
  §4 sketch is order-of-magnitude. A real noise model (variance
  propagation through heat operator with diagonal V) would tighten
  acceptance gates and inform the default `chunk_size`. Useful but
  not blocking v1.
