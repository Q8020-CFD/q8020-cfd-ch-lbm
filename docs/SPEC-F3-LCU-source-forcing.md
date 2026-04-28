# SPEC — F3 LCU source forcing (Strang-split CH-LCU)

Self-contained handoff. Reader has not seen the prior conversation.

## 0. Context

The pure-quantum CH-LCU propagator
([SPEC-F3-LCU-method.md](SPEC-F3-LCU-method.md), variant 1.B)
shipped in the `q8020-mps-burgers` repo as `--propagator lcu`. It
solves the unforced heat equation `∂φ/∂t = ν·∂_xx φ` via a truncated
Taylor LCU built from `S+`/`S-` ladder primitives, with `q + m`
qubits (m ancillas, `m ≈ ⌈log₂(2·taylor_order + 1)⌉`).

This spec adds source forcing to the LCU propagator so that
`--method cole_hopf_circuit --propagator lcu --source sine` runs
end-to-end. It mirrors the source-forcing work already done for
`--propagator dense-block` ([SPEC-source-forcing.md](SPEC-source-forcing.md)),
keeping `lcu`'s pure-quantum-in-the-time-loop guarantee intact.

**Status going in**: today's
[`run_cole_hopf_circuit_simulation`](../src/burgers_cole_hopf_circuit.py)
has a guard at lines 1116–1123:

```python
if source_fn is not None and propagator not in ("dense-block",):
    raise NotImplementedError(
        "Source forcing requires --propagator dense-block in this "
        "release. qft-diagonal/lcu+source is on the roadmap."
    )
```

This spec replaces "qft-diagonal/lcu" with "qft-diagonal" — i.e.,
`lcu + source` becomes supported; `qft-diagonal + source` stays
deferred.

## 1. Goal

After this parcel:

- `--method cole_hopf_circuit --propagator lcu --source sine`
  runs end-to-end at q=5, ν=0.1, shock-pct=100, statevector mode,
  and produces a final L2 vs FTCS forced reference under 0.05.
- The same pipeline works under `--shots N` and chunked evolution.
- Per-step P_success ≥ 0.7 at the canonical `(q=5, ν=0.1, dt)`,
  comparable to the unforced LCU baseline.
- The pure-quantum-in-the-time-loop property holds: V(x, t) is
  built classically before each step (same model as dense-block
  source-forcing) but the *evolution* — state-prep, propagator,
  measurement — is fully quantum with no classical re-injection.

## 2. Non-goals

- **`--propagator qft-diagonal` + source.** Stays deferred. The
  Fourier basis diagonalises `L` but not `diag(V)`; would need
  Strang splitting around the diagonal QFT layer with V block-
  encoded separately. Out of scope here.
- **Generic non-analytic `g(x, t)`.** Spec assumes either an
  analytic source (paper's `sin(2πx)·cos(2πt)`) or a callable that
  returns a numpy array on the grid; trapezoidal antiderivative
  handles the rest. Same shape as the existing
  `burgers_potential.potential_from_source`.
- **Adaptive Trotter / Strang order.** Fixed first-order Strang
  for v1. Higher orders deferred.
- **`--bc dirichlet` for LCU.** LCU is periodic-only in v1 (the
  guard at
  [burgers_cole_hopf_circuit.py:1125-1129](../src/burgers_cole_hopf_circuit.py:1125)
  stays). Forced + Dirichlet is a separate parcel.
- **Direct non-commuting Taylor expansion of
  `exp((νL − V)·dt)`.** Considered; rejected. Strang is cheaper,
  parallel to the unforced LCU and dense-block paths, and
  introduces only O(dt²) error which matches what dense-block
  source-forcing already pays.

## 3. Math: Strang split

Forced φ-equation under Cole-Hopf:

```
∂φ/∂t = ν·∂_xx φ − V(x, t)·φ           (M = νL − diag(V))
```

Where V comes from the source via
[`burgers_potential.potential_from_source`](../src/burgers_potential.py)
(`V_x = +g/(2ν)`, gauge-fixed mean-zero — see SPEC-source-forcing
§1, with the corrected sign confirmed in
[SPEC-source-forcing-REVIEW.md](SPEC-source-forcing-REVIEW.md)).

`L` and `diag(V)` do NOT commute (`L` has off-diagonals; `V` is
position-diagonal). Use first-order Strang:

```
exp((νL − V)·dt) ≈ exp(−V·dt/2) · exp(ν·L·dt) · exp(−V·dt/2)
                                                    + O(dt²)
```

The outer `exp(−V·dt/2)` factors are real, positive, diagonal,
non-unitary. They block-encode via the same controlled-Ry +
ancilla post-select trick `dense-block` already uses for the heat
propagator. The inner `exp(ν·L·dt)` is exactly today's unforced
LCU.

V is evaluated at the **Strang midpoint** `t_mid = t_n + dt/2` of
each step. (Same convention dense-block uses; see
[burgers_cole_hopf_circuit.py:334](../src/burgers_cole_hopf_circuit.py:334).)

### 3.1 Per-step P_success

The Strang circuit is three block-encodings composed sequentially
with measure+reset of the V-ancilla between halves:

```
P_success_step = (s_max_V)² · p_heat · (s_max_V)²
              = s_max_V⁴ · p_heat
```

Where `s_max_V = max_k exp(−V_k·dt/2)`. For paper-case V at ν=0.1
peaking around `1/(4πν) ≈ 0.8`, dt ≈ 0.003, V·dt/2 ≈ 0.0012, so
`s_max_V ≈ exp(0.0012) ≈ 1.0012`. Forth-power factor ≈ 1.005;
contribution to attenuation < 1% per step. P_success_step is
dominated by `p_heat` (~0.8 at the canonical case), unchanged from
unforced.

### 3.2 Operator construction order

Per step, in the order applied to |φ_n⟩:

1. `B_V(t_mid)` — block-encodes `diag(exp(−V_n·dt/2))/s_max_V_n` on
   `q + 1` qubits. Post-select ancilla=|0⟩.
2. `B_L` — block-encodes the unforced heat LCU
   `(Σ_k (νdt)^k L^k/k!) / λ_heat` on `q + m_heat` qubits. Post-
   select ancilla=|0^m_heat⟩.
3. `B_V(t_mid)` again — same V_n, same s_max.

The two V applications can share ONE ancilla qubit with measure +
reset between them. Total ancillas per step: `m_heat + 1`. Total
classical bits per N_steps run: `N_steps · (m_heat + 2) + q`
(two V measurements per step, m_heat heat measurements, plus q
data measurements at the end).

## 4. Plumbing changes

### 4.1 New function — `diag_potential_block_encoding`

In [`burgers_lcu.py`](../src/burgers_lcu.py), add after
`heat_lcu_step_circuit`:

```python
def diag_potential_block_encoding(
    q: int,
    V: np.ndarray,           # length N=2^q, position-diagonal
    dt_half: float,          # the Strang half-step (dt/2)
) -> tuple[QuantumCircuit, float]:
    """Block-encode diag(exp(-V*dt_half)) on q+1 qubits.

    Mirrors heat_dense_block_step_circuit's controlled-Ry layer.
    Returns (qc, s_max) where post-selecting the q-th qubit on
    |0> leaves data in diag(exp(-V*dt_half))/s_max applied to the
    input state.

    s_max = max_k |exp(-V_k*dt_half)| is the operator norm of the
    diagonal; needed by the caller to bookkeep cumulative scaling.
    """
    from qiskit.circuit.library import RYGate

    N = 1 << q
    if V.shape != (N,):
        raise ValueError(
            f"V must have length {N}, got {V.shape}"
        )
    diag_vals = np.exp(-V * dt_half)             # real, positive
    s_max = float(np.max(np.abs(diag_vals)))
    if s_max < 1e-300:
        raise ValueError("diag(exp(-V*dt_half)) is identically zero")

    qc = QuantumCircuit(q + 1, name="V_half")
    data = list(range(q))
    anc = q
    for k in range(N):
        if abs(diag_vals[k]) < 1e-15:
            continue
        theta = 2.0 * np.arccos(
            np.clip(diag_vals[k] / s_max, -1.0, 1.0),
        )
        if abs(theta) < 1e-15:
            continue
        ctrl_state = format(k, f"0{q}b")
        gate = RYGate(theta).control(
            q, ctrl_state=ctrl_state,
        )
        qc.append(gate, data + [anc])
    return qc, s_max
```

This is the same controlled-Ry pattern used by dense-block at
[burgers_cole_hopf_circuit.py:488-495](../src/burgers_cole_hopf_circuit.py:488)
and [:482-497](../src/burgers_cole_hopf_circuit.py:482). Copy-
paste-adapt.

### 4.2 New function — `heat_lcu_with_potential_step_circuit`

In `burgers_lcu.py`, add a sibling to `heat_lcu_step_circuit`:

```python
def heat_lcu_with_potential_step_circuit(
    q: int,
    nu: float,
    dt: float,
    L_box: float,
    V: np.ndarray,                # midpoint potential, length N
    taylor_order: int = 4,
) -> tuple[QuantumCircuit, float]:
    """Strang-split LCU step: V/2 -> heat LCU -> V/2.

    Returns (qc, lam_total) on q + m_heat + 1 qubits where:
      - q data qubits (low indices)
      - m_heat heat ancilla qubits (LCU PREPARE state register)
      - 1 V ancilla (highest index), measured + reset between halves

    lam_total = s_max_V * lam_heat * s_max_V
              = lam_heat * s_max_V^2

    Post-selecting heat-ancilla=|0^m_heat> AND V-ancilla=|0> at
    each of the two V applications gives:
        (1/lam_total) * exp(-V*dt/2) * P_M(nu*L*dt) * exp(-V*dt/2)

    Layout decision: the V ancilla shares one qubit between halves
    via mid-circuit measure+reset.  m_heat ancillas are kept across
    the heat block (no measure+reset; heat is a single LCU block).
    """
    from qiskit.circuit import ClassicalRegister

    N = 1 << q
    if V.shape != (N,):
        raise ValueError(f"V must have length {N}, got {V.shape}")

    # Build the three sub-blocks
    v_qc, s_max_V = diag_potential_block_encoding(q, V, dt / 2.0)
    heat_qc, lam_heat = heat_lcu_step_circuit(
        q, nu, dt, L_box, bc="periodic", taylor_order=taylor_order,
    )
    m_heat = heat_qc.num_qubits - q

    # Total qubit layout:
    #   data: 0..q-1
    #   heat anc: q..q+m_heat-1
    #   V anc: q+m_heat
    total_q = q + m_heat + 1
    v_anc = q + m_heat

    # The V block targets data + V-ancilla (q+1 qubits); the heat
    # block targets data + heat-ancillas (q+m_heat qubits).
    qc = QuantumCircuit(total_q, name="V_half_heat_V_half")

    # Mid-circuit measurement target for the two V-ancilla reads.
    # Caller (full_circuit builder) will add this; per-step builder
    # exposes the V-ancilla and the per-step reset is performed
    # OUTSIDE this function via measure + reset on v_anc between
    # halves.
    qc.compose(
        v_qc,
        qubits=list(range(q)) + [v_anc],
        inplace=True,
    )
    # Caller measures v_anc here, resets it, before the heat block.
    qc.compose(
        heat_qc,
        qubits=list(range(q)) + list(range(q, q + m_heat)),
        inplace=True,
    )
    # Caller measures heat ancillas here, resets between steps.
    qc.compose(
        v_qc,
        qubits=list(range(q)) + [v_anc],
        inplace=True,
    )
    # Caller measures v_anc here (second half).

    lam_total = lam_heat * (s_max_V ** 2)
    qc.metadata = {
        "lcu_lambda": lam_total,
        "lcu_lambda_heat": lam_heat,
        "lcu_s_max_V": s_max_V,
    }
    return qc, lam_total
```

**Note for implementer**: the comments `# Caller measures...`
delineate where mid-circuit measurements need to be inserted by
`heat_lcu_full_circuit` (shots path) and by the SV-path simulator
(which handles post-selection by direct projection on the
statevector — no measurements in the circuit, see §4.4). Two
options for how to expose this:

- **Option A (cleaner)**: have this function return a list of
  "post-select points" (qubit index, name) so the full-circuit
  builder knows where to insert measurements. Three post-select
  points per step: V-anc after first V-half, heat-ancs after heat
  block, V-anc after second V-half.
- **Option B (simpler, ship-friendly)**: don't bake measurements
  in here at all; have this function return a measurement-free
  composite circuit AND a metadata dict listing the ancilla
  qubits with their post-select expectations. Caller builds the
  full circuit by inlining + measure-reset at the right points.

Recommend **Option B**. Less abstract, easier to debug.

### 4.3 Modify `_build_step_sv` to accept V (LCU branch)

In [`burgers_cole_hopf_circuit.py`](../src/burgers_cole_hopf_circuit.py),
the LCU branch at lines 498–506 currently ignores V:

```python
elif propagator == "lcu":
    from burgers_lcu import heat_lcu_step_circuit
    lcu_qc, lam = heat_lcu_step_circuit(
        q, nu, dt, L_box, bc=bc,
        taylor_order=taylor_order,
    )
    lcu_qc.metadata = {"lcu_lambda": lam}
    return lcu_qc
```

Replace with:

```python
elif propagator == "lcu":
    from burgers_lcu import (
        heat_lcu_step_circuit,
        heat_lcu_with_potential_step_circuit,
    )
    if V is None:
        lcu_qc, lam = heat_lcu_step_circuit(
            q, nu, dt, L_box, bc=bc,
            taylor_order=taylor_order,
        )
        lcu_qc.metadata = {"lcu_lambda": lam}
    else:
        lcu_qc, lam = heat_lcu_with_potential_step_circuit(
            q, nu, dt, L_box, V,
            taylor_order=taylor_order,
        )
        # Metadata already set by builder (includes s_max_V)
    return lcu_qc
```

(The `V` parameter on `_build_step_sv` already exists — see
line 447. The LCU branch just needs to honour it.)

### 4.4 Modify the SV path post-selection projection

`run_cole_hopf_circuit_sv` projects on `sv[:N]` per step
([burgers_cole_hopf_circuit.py:614](../src/burgers_cole_hopf_circuit.py:614)).
This works for any propagator with a single contiguous
"all-ancillas-zero" subspace at the front of the statevector.

For the Strang-split LCU step circuit, ancilla layout is:
`data (q) | heat_anc (m) | V_anc (1)`. Total ancillas = `m + 1`.
Statevector index `j`: data bits = `j % N`; heat_anc bits = `(j //
N) % 2^m`; V_anc bit = `j // (N * 2^m)`.

The SV path's existing logic already handles arbitrary `n_anc =
step_qc.num_qubits - q` (line 558, 562, 606). The "all-zero
ancilla" subspace is `sv[:N]` regardless of how many ancillas there
are — that's still correct. **No change needed to the SV
projection logic.**

The per-step V_n rebuild already exists in `run_cole_hopf_circuit_sv`
(line 595-604). The LCU branch needs to enter it when V_n is set.
Verify: the existing condition `if shared_step is not None`
(line 593) gates the rebuild path correctly — when `source_fn is
not None`, `shared_step is None` and per-step rebuild fires for
all propagators, including LCU. So step (4.3) suffices for the SV
path.

### 4.5 Modify `heat_lcu_full_circuit` for per-step rebuild

Today's
[`heat_lcu_full_circuit`](../src/burgers_cole_hopf_circuit.py:388)
builds one `step_qc` and inlines `N_steps` times — correct for
unforced. With source, V_n changes per step, so each step needs a
fresh `step_qc`. Mirror the pattern in
[`heat_dense_block_full_circuit`](../src/burgers_cole_hopf_circuit.py:294)
(specifically lines 322–349 which guard on `source_fn is None`).

Add `source_fn`, `x`, `t_start: float = 0.0` parameters; branch
on `source_fn is None` (build once, inline) vs `source_fn is not
None` (rebuild per step with `V_n` at midpoint
`t_start + (step_idx + 0.5) * dt`).

The mid-circuit measurement schema also changes when V is on:
each step has `measure(v_anc) → reset → heat block →
measure(heat_anc) → reset → measure(v_anc) → reset` (V-anc gets
measured twice per step). Update the classical-register sizing:

```python
# Old: N_steps * m_heat ancilla bits.
# New: N_steps * (m_heat + 2) ancilla bits  (2 V-anc reads/step)
```

Use one `ClassicalRegister` per ancilla type for clarity:

```python
v_anc_cr   = ClassicalRegister(N_steps * 2, "v_anc_hist")
heat_anc_cr = ClassicalRegister(N_steps * m_heat, "heat_anc_hist")
data_cr    = ClassicalRegister(q, "data")
```

Post-selection in `post_select_counts` already strips and
checks ALL ancilla bits = 0; the function doesn't care which
register. Verify the bitstring layout after Qiskit serializes the
counts (registers added later go to higher bit positions); should
work without changes since the existing `data_bits = bitstring[:q]`
+ `anc_bits = bitstring[q:]` slice still isolates data correctly,
and "all anc bits zero" is the same predicate.

### 4.6 Source guard — `run_cole_hopf_circuit_simulation`

[burgers_cole_hopf_circuit.py:1116–1123](../src/burgers_cole_hopf_circuit.py:1116):

```python
if source_fn is not None and propagator not in ("dense-block",):
    raise NotImplementedError(...)
```

Replace `("dense-block",)` with `("dense-block", "lcu")`. Update
the error message to reflect that only `qft-diagonal` is now the
unsupported combination.

### 4.7 Chunked driver — `_run_shots_chunked`

[burgers_cole_hopf_circuit.py:751–755](../src/burgers_cole_hopf_circuit.py:751)
already passes `source_fn` and `x` only to the dense-block branch.
Extend the LCU branch to do the same:

```python
elif propagator == "lcu":
    full_qc = heat_lcu_full_circuit(
        q, nu, T_chunk, chunk_size, L_box, bc=bc,
        taylor_order=taylor_order,
        source_fn=source_fn, x=x,                # NEW
        t_start=t_start_chunk,                   # NEW
    )
```

Same `t_start_chunk = chunk_idx * chunk_size * dt` calculation
that's already in the dense-block branch.

### 4.8 Batch driver — `_run_shots_batch`

[burgers_cole_hopf_circuit.py:931–935](../src/burgers_cole_hopf_circuit.py:931):

```python
elif propagator == "lcu":
    full_qc = heat_lcu_full_circuit(
        q, nu, dt * s, s, L_box, bc=bc,
        taylor_order=taylor_order,
        source_fn=source_fn, x=x,                # NEW
    )
```

(Batch driver has `t_start` implicit at 0; each circuit `s`
evolves from t=0 for `s` steps.)

## 5. Tests

`tests/test_lcu_source.py` (new file). Six tests:

1. **`test_diag_potential_block_encoding_correctness`** — for
   `V = [0.1, -0.2, 0.05, 0.0]` on q=2 and `dt_half = 0.5`,
   extract the block-encoded operator from
   `diag_potential_block_encoding`'s circuit and compare to
   `diag(exp(-V*dt_half))/s_max` to 1e-12.
2. **`test_strang_step_correctness_unitary`** — at q=4, ν=0.1,
   `dt = 0.001`, `V = sin(2πx)·cos(0)/(4πν)` (paper case at t=0),
   build `heat_lcu_with_potential_step_circuit`, extract block-
   encoded operator at all-ancilla-zero, compare to classical
   `expm(-V*dt/2) @ expm(ν*L*dt) @ expm(-V*dt/2) / lam_total`.
   Tolerance 1e-10. Proves Strang is implemented correctly modulo
   the dt² error vs the true `expm`.
3. **`test_strang_vs_full_expm_dt_squared_scaling`** — at fixed q,
   ν, sweep `dt ∈ {0.01, 0.005, 0.0025, 0.00125}`. Compute
   `‖U_Strang(dt) − expm(M·dt)‖_2` and confirm error scales as
   O(dt²).
4. **`test_ch_lcu_with_source_matches_dense_block_sv`** — at q=5,
   ν=0.1, source=sine, n_steps=10, shots=0: relative L2 between
   `--propagator lcu` and `--propagator dense-block` < 0.05 at the
   final time. Both paths solve the same forced PDE; bound is
   loose because Strang LCU is dt²-accurate while dense-block is
   exact-per-step.
5. **`test_ch_lcu_with_source_matches_ftcs`** — at q=5, ν=0.1,
   source=sine, n_steps=20, shots=0: relative L2 vs classical FTCS
   forced reference < 0.10. Sanity gate that the implementation
   solves the *physics*, not just internal consistency.
6. **`test_ch_lcu_p_success_with_source`** — at q=5, ν=0.1,
   source=sine, n_steps=10, shots=0 (statevector): every
   `step_metrics["p_success_step"] > 0.7`. The V·dt/2 attenuation
   should be tiny at this case size (back-of-envelope: ≤ 1%
   contribution per step).

`tests/test_lcu.py` regression: re-run the existing 6 unforced
tests to confirm no breakage.

## 6. q8020 TOML smoke

Append to [`burgers_quantum.toml`](../input/burgers_quantum.toml)
in the LCU block already established by the unforced smoke:

```toml
# Forced LCU smoke: paper-case sine forcing at q=5.
# Acceptance: final L2 vs FTCS forced < 0.10; per-step p_success > 0.7.
[cole_hopf_circuit_lcu_forced_q5]
"--method" = "cole_hopf_circuit"
"--propagator" = "lcu"
"--ic" = "sine"
"--source" = "sine"
"--nu" = 0.1
"--cfl" = 0.1
"--shock-pct" = 100.0
"--q" = 5
"--shots" = 0
"--bc" = "periodic"
"--lcu-taylor-order" = 4
"--save-every" = 1
_group_postproc = ["python ./q8020-mps-burgers/docs/plot_cole_hopf_circuit_evolution.py"]

# Forced LCU + chunked shots: end-to-end realism.
# Acceptance: final L2 vs FTCS forced < 0.15.
[cole_hopf_circuit_lcu_forced_chunked_q5]
"--method" = "cole_hopf_circuit"
"--propagator" = "lcu"
"--ic" = "sine"
"--source" = "sine"
"--nu" = 0.1
"--cfl" = 0.1
"--shock-pct" = 100.0
"--q" = 5
"--shots" = 100000
"--backend-type" = "sim"
"--seed" = 42
"--optimization-level" = 0
"--bc" = "periodic"
"--lcu-taylor-order" = 4
"--save-every" = 10
"--evolution-mode" = "chunked"
"--chunk-size" = 10
_group_postproc = ["python ./q8020-mps-burgers/docs/plot_cole_hopf_circuit_evolution.py"]
```

## 7. Acceptance

- [ ] All six §5 tests pass.
- [ ] All six existing `tests/test_lcu.py` tests still pass.
- [ ] §6 SV smoke (`cole_hopf_circuit_lcu_forced_q5`) reports
      `final_error < 0.10` against FTCS forced; the postproc GIF
      shows a forced sine evolution (sustained amplitude under
      the cos(2πt) drive), not a pure decay.
- [ ] §6 chunked shots smoke
      (`cole_hopf_circuit_lcu_forced_chunked_q5`) reports
      `final_error < 0.15`.
- [ ] Per-step `step_metrics` carries `lcu_lambda`,
      `lcu_lambda_heat`, `lcu_s_max_V`, `p_success_step`. (For
      the shots path: per-snapshot `met` carries the same.)
- [ ] [SPEC-F3-LCU-method.md](SPEC-F3-LCU-method.md) §3
      non-goal "qft-diagonal/lcu+source" updated to
      "qft-diagonal+source" — LCU+source is no longer a non-goal.
- [ ] [OVERVIEW-burgers-solver.md](OVERVIEW-burgers-solver.md)
      §3 source-forcing section: "Currently supported only on
      `dense-block`" → "Currently supported on `dense-block` and
      `lcu`."

## 8. Implementation order

S1: §4.1 `diag_potential_block_encoding` + §5 test #1.
    ~30–60 min.
S2: §4.2 `heat_lcu_with_potential_step_circuit` + §5 tests #2 and
    #3. The two-V-half + heat sandwich is the core; getting
    measurements vs SV-projection right is the only fiddly bit.
    ~2–3 hours.
S3: §4.3 `_build_step_sv` LCU+V branch + §5 test #4 (SV path
    only). ~1 hour.
S4: §4.5 `heat_lcu_full_circuit` per-step rebuild. ~1–2 hours
    (mostly mechanical, mirror dense-block).
S5: §4.6, §4.7, §4.8 plumbing — source guard, batch and chunked
    drivers. ~30 min.
S6: §5 tests #5, #6 (full pipeline + p_success). ~30 min.
S7: §6 TOML smokes + manual q8020 sweep. Confirm GIF visually.
    ~30 min.
S8: §7 doc updates (acceptance checklist final two items). 15 min.

Total: ~6–8 hours, comfortably one day.

S1 and S2 are zero-risk (new code, no integration). S3 onward
modifies hot paths; run the existing
`test_ch_lcu_propagator_matches_dense_block_at_q5` regression
after each.

## 9. Out of scope / future work

- **`qft-diagonal + source`.** Same Strang trick applies, but the
  inner block becomes the qft-diagonal heat layer instead of LCU.
  Worth its own spec; modest cost.
- **Higher-order Strang** (Strang-Yoshida-4 or Trotter-Suzuki
  4th-order). Trades depth for accuracy. Useful at large dt; not
  needed for current case.
- **Direct non-commuting-Taylor LCU of `exp((νL − V)·dt)`.**
  Considered in §2; rejected. May be revisited if Strang's
  O(dt²) error becomes a bottleneck.
- **Time-integrated V** for stiff source regimes — when `dt · V`
  is not small. Not the regime we run; defer.
- **Adaptive `taylor_order` based on (ν, dt, V)**. Fixed at runtime
  via CLI; runtime adaptation is a follow-up.
- **LCU + source + Dirichlet BC.** Periodic-only in v1, same as
  unforced LCU. Dirichlet would need a different shift basis (no
  cyclic wrap); separate parcel.
