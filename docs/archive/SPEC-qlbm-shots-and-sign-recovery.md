# SPEC — `qlbm_circuit` shots path + sign correction (FUTURE-WORK #14)

Self-contained handoff. Reader has not seen prior conversation.

## 0. Context

`q8020-mps-burgers` includes a quantum-circuit D1Q3 LBM solver
(`burgers_qlbm_circuit.py`) for cross-method comparison against the
direct-`u` (`quantum_circuit`) and Cole–Hopf (`cole_hopf_circuit`)
families. It is classified as **hybrid (H)** in OVERVIEW §2 because
the collision operator depends on the current distribution state and
must be rebuilt every timestep from a classical mirror.

**Current state (statevector only).** `run_qlbm_circuit_simulation`
(`burgers_qlbm_circuit.py:201`) supports two paths:

- `shots == 0`: real circuit, `Statevector(psi_in).evolve(qc)`, exact
  amplitudes, reconstruction is trivial.
- `shots > 0`: **classical fallback** — calls `collide_bgk(f, tau)` +
  `stream(f, bc=bc)` from `burgers_qlbm.py` and ignores the backend
  entirely (`burgers_qlbm_circuit.py:272-276`). The `backend`
  parameter is plumbed through but never used.

OVERVIEW §2 method roster Notes column already flags this:
`qlbm_circuit | QLBM | H | Quantum-circuit D1Q3 (statevector real;
shots = classical fallback)`. This SPEC closes the gap.

## 1. Goal

Make `--method qlbm_circuit --shots N > 0` execute the same quantum
circuit the statevector path builds, sampled `N` times on the
configured backend, then reconstruct the post-step distributions `fᵢ`
from the measured counts and rescale them by the classical
cumulative-norm trajectory. Add sign correction with the same
strategy menu (`--sign-recovery`) that `quantum_circuit` exposes,
since the QLBM amplitude register can carry negative components in
the shock / non-positive-`fᵢ` regime.

When done: a sweep can include `--method qlbm_circuit --shots 8192
--backend-type {sim,hardware}` and produce honest measurements +
reconstructions, no classical bypass.

## 2. Math contract

### 2.1 Register layout

`n_qubits = q + 2`. Top 2 qubits are the velocity register (D1Q3 has 3
populated velocity states; the 4th, `|11⟩`, is unused). Bottom `q`
qubits are the position register. Full Hilbert space dim is `4N`
where `N = 2^q`.

```
basis label       (v1 v0)(p_{q-1} ... p_0)     interpretation
|00⟩ |j⟩          0    0    j                  f_{-1}(x_j)   (left-moving)
|01⟩ |j⟩          0    1    j                  f_0(x_j)      (rest)
|10⟩ |j⟩          1    0    j                  f_{+1}(x_j)   (right-moving)
|11⟩ |j⟩          1    1    j                  unused (must remain 0)
```

The `unflatten_distributions` helper (`burgers_qlbm.py:298`) maps the
flattened `4N`-vector to the `(3, N)` distribution array and drops
the `|11⟩` block.

### 2.2 Single step

Per step the circuit is:

```
qc = build_qlbm_step_circuit(f_pre, tau, q)        # one (q+2)-qubit gate
```

decomposing internally into

```
U_step = U_stream · U_collision(f_pre, tau)
```

where:

- `U_collision(f_pre, tau)` is the *Householder dilation* that maps
  the normalised pre-collision state `psi_in = vec(f_pre) / ‖vec(f_pre)‖`
  to the normalised post-collision state
  `psi_out = vec(f_post) / ‖vec(f_post)‖`, where
  `f_post = collide_bgk(f_pre, tau)`
  (`burgers_qlbm_circuit.py:105-146`). Returns the `contraction =
  ‖vec(f_post)‖ / ‖vec(f_pre)‖` factor as a scalar tracked classically.
- `U_stream` is the block-diagonal unitary `block_diag(dec, eye, inc,
  eye)` on velocity-subspaces (`burgers_qlbm_circuit.py:65-86`).

### 2.3 Cumulative norm

```
cumulative_norm_0 = ‖vec(f_0)‖
cumulative_norm_n = cumulative_norm_{n-1} · contraction_n             (1)
```

The circuit evolves *normalised* states; the physical distributions
are recovered by multiplying by `cumulative_norm_n` after measurement.
This already works in the statevector path
(`burgers_qlbm_circuit.py:248, 265`).

### 2.4 Reconstruction from counts

Sample `S` shots of the full `(q+2)`-qubit circuit. Each shot yields a
bitstring `b = b_{q+1} b_q b_{q-1} … b_0`. Group by bitstring, get
`counts[b]`. Reconstruct the normalised post-step amplitude as

```
|psi_out_k| ≈ sqrt(counts[k] / S)                                     (2)
```

then unflatten:

```
f_post_flat = sign_vec · |psi_out| · cumulative_norm_n                (3)
f_post = unflatten_distributions(f_post_flat, N)                      (4)
```

where `sign_vec ∈ {−1, +1}^{4N}` carries the sign recovery (§3).
After unflatten, post-select on the `|11⟩` velocity-register block
being empty:

```
leakage = sum( |psi_out_k|^2 for k in |11⟩-block ) / sum_all          (5)
```

`leakage > 1e-3` (heuristic threshold) flags numerical drift in the
streaming/collision unitaries or transpiler-induced error. Surfaced
in per-step metrics dict.

## 3. Sign correction

### 3.1 Why signs matter for QLBM

In the *physical* regime (low Mach, smooth flow, well-conditioned
BGK), all distributions `fᵢ(x, t) ≥ 0` for all `t`. Then `psi_out` is
component-wise non-negative and `|psi_out| = psi_out`; vanilla
`sqrt(counts/S)` reconstruction is exact up to shot noise. No sign
recovery needed.

In the *shock* regime (high amplitude, near-singular `tau`, or
near-BGK-stability-boundary), the collision step can produce
`fᵢ(x) < 0` at some grid points — a known LBM stability artefact, not
a quantum-circuit bug. When this happens, `psi_out` has negative
components and `|psi_out|² = psi_out²` discards their sign;
reconstruction via `sqrt(counts/S)` returns
`|amplitude|`, missing the sign bit.

The user CLI already exposes `--sign-recovery {none, classical_oracle,
hadamard_test, dual_rail}` for `quantum_circuit` (§7.5 of OVERVIEW).
This SPEC extends the same menu to `qlbm_circuit`, with the same
semantics.

### 3.2 Strategies

**none** (default). Reconstruct as `+sqrt(counts/S)`. Correct when
distributions stay non-negative; silently introduces L² error
proportional to `‖f_negative_part‖` when they don't.

**classical_oracle**. After each shot-mode step:
1. Reconstruct magnitudes `|f_post|` from shot data (§2.4 without
   sign).
2. Run a parallel **classical** step:
   `f_ref = collide_bgk(f_pre, tau); f_ref = stream(f_ref, bc=bc)`.
3. Copy signs: `f_post = sign(f_ref) · |f_post|`.

This mirrors `classical_oracle_signs` in `burgers_sign_recovery.py:38`
and the dispatch pattern in `burgers_trotter.py:222-225`. Strict
honesty caveat: this is a hybrid path (signs from a classical
reference), so it is **not** a stand-alone benchmark — it is a
diagnostic / debugging mode. Documented as such.

**hadamard_test** (per bin, optional, expensive). Mirror the
`hadamard_per_bin_circuit` machinery in
`burgers_cole_hopf_circuit.py:1597-1851`. Adds one ancilla, two
applications of `U_step` per bin in superposition with a reference
prep, measures real part of overlap. Sign is the sign of `Re(⟨k|
U_step |psi_in⟩ - ⟨k| |psi_in⟩)`. Cost: `O(4N)` additional circuit
executions per step. Recommended only for diagnostic deep dives, not
production sweeps.

**dual_rail** — **not in scope for this SPEC.** Dual-rail would
require splitting `f` into positive and negative parts and evolving
two QLBM lattices in parallel; the LBM collision is non-linear in `f`,
so the split-and-evolve scheme that works for the linear `quantum_circuit`
generator does **not** carry over. Defer to a follow-up SPEC if ever
needed.

### 3.3 Recommended default

`--sign-recovery none` for sweeps in the well-behaved regime (low
Mach, periodic, smooth IC). The default at the framework level stays
`none` for `qlbm_circuit`, matching the `quantum_circuit` default.

Flag any run that emits a per-step warning `[qlbm_circuit] WARNING:
negative-magnitude detected in classical f_ref (max |f_neg|=…);
consider --sign-recovery classical_oracle.` so the user knows when
to switch.

## 4. Code touchpoints

### 4.1 `src/burgers_qlbm_circuit.py`

The function to gut and rewrite is the shots-mode branch of
`run_qlbm_circuit_simulation` (lines 272-276 today). Replace with:

```python
else:  # shots > 0: real circuit, measure, reconstruct
    qc, contraction = build_qlbm_step_circuit(f, tau, q)
    cumulative_norm *= contraction

    vec_in = flatten_distributions(f)
    norm_in = float(np.linalg.norm(vec_in))
    if norm_in < 1e-15:
        f = np.zeros_like(f)
        continue
    psi_in = vec_in / norm_in

    # Run on backend
    qc_full = QuantumCircuit(n_qubits, n_qubits)
    qc_full.initialize(psi_in.tolist(), range(n_qubits))
    qc_full.compose(qc, inplace=True)
    qc_full.measure_all()
    qc_t = transpile(qc_full, backend, optimization_level=1)
    result = backend.run(qc_t, shots=shots).result()
    counts = result.get_counts()

    # Reconstruct (signs default to +; classical_oracle path below)
    psi_out_mag = np.zeros(dim)
    for bitstr, cnt in counts.items():
        idx = int(bitstr.replace(" ", "")[::-1], 2)
        psi_out_mag[idx] = np.sqrt(cnt / shots)

    leakage = _compute_leakage(psi_out_mag, q)   # §2.4 Eq. 5

    if sign_recovery == "classical_oracle":
        f_ref = collide_bgk(f, tau)
        f_ref = stream(f_ref, bc=bc)
        vec_ref = flatten_distributions(f_ref)
        signs = np.sign(vec_ref)
        signs[signs == 0] = 1.0
        psi_out = signs * psi_out_mag
    elif sign_recovery == "hadamard_test":
        psi_out = _qlbm_hadamard_signs(
            psi_in, qc, n_qubits, shots, backend, q,
        )
    else:  # 'none'
        psi_out = psi_out_mag

    f_out_flat = psi_out * cumulative_norm
    f = unflatten_distributions(f_out_flat, N)

    if bc == "dirichlet":
        f[2, 0] = f[0, 0]
        f[0, -1] = f[2, -1]
```

New helpers in the same file:

- `_compute_leakage(psi_vec, q) -> float`: sum of `|psi_vec[k]|²` over
  the `|11⟩` block divided by total mass.
- `_qlbm_hadamard_signs(psi_in, qc, n_qubits, shots, backend, q)`:
  per-bin Hadamard-test signs; deferred implementation OK if
  `--sign-recovery hadamard_test` raises `NotImplementedError` in v1.
  v1 ships `none` + `classical_oracle`.

Endianness note: Qiskit's `get_counts()` bitstrings are little-endian.
The current statevector path uses the natural endianness of
`Statevector.data` (big-endian). The `int(bitstr[::-1], 2)` reversal
in the snippet above must match the endianness assumed by
`unflatten_distributions`; verify against a single-step statevector
round-trip in v1 acceptance.

### 4.2 `src/burgers_qlbm_circuit.py` — signature changes

`run_qlbm_circuit_simulation` gains a `sign_recovery: str = "none"`
kwarg. The existing `backend: Any = None  # noqa: ARG001` becomes
a real positional consumer of the backend; remove the `noqa`.

### 4.3 `src/burgers_fw.py`

The `QLBMCircuitIntegrator` (already wired in §8.3 dispatcher) needs
to pass `config.sign_recovery` and `config.backend` into the new
kwarg. `BurgersConfig.sign_recovery` already exists (set up for
`quantum_circuit`); no schema change.

### 4.4 `src/burgers_solver.py`

No CLI changes. `--sign-recovery` is already a valid CLI flag;
documenting that `qlbm_circuit` now honors it goes in OVERVIEW
(§4 covers QLBM family; add a paragraph in §5.2 mirroring the
quantum_circuit sign-recovery treatment in §3.3.4).

### 4.5 `src/burgers_postprocess.py`

Per-step metrics dict gains:

- `leakage: float` — fraction of probability mass landing in the
  `|11⟩` block.
- `negative_mass: float | None` — when sign_recovery == "classical_oracle",
  fraction of |f_ref| that was negative (signal of the regime where
  sign correction matters).

These flow through the same `step_metrics` channel that
`quantum_circuit` already uses.

### 4.6 `tests/`

Extend `tests/test_shots_backend.py` (referenced in #14 scope) with
QLBM cases:

- **Statevector ↔ shots agreement** at `shots = 2¹⁴`, `q = 3`, sine
  IC, periodic, `ν = 1e-2`, 10 steps: L² error < 5% between shot-mode
  reconstruction and statevector reference.
- **Sign-recovery correctness**: synthetic case where classical
  `f_ref` has known negative components; verify
  `--sign-recovery classical_oracle` reproduces them and
  `--sign-recovery none` does not.
- **Backend abstraction**: same case run with default Aer vs an
  AerSimulator with no noise model, byte-for-byte equivalent results
  modulo shot RNG seed.
- **Hardware path (optional, slow)**: same case on a small ibm fake
  backend with a depolarising noise model; loose tolerance.

## 5. Failure modes

1. **Cumulative `P_success` is not a thing for QLBM.** Unlike the
   Cole–Hopf block-encoding, the QLBM step uses a *dilated unitary*
   on the full `4N` space — there are no ancilla success bits to
   post-select on. The `contraction_factor` accounts for the
   non-unitarity classically. So no exponential
   shots-budget decay with `n_steps`. Document this in OVERVIEW §5.2
   for the chunked-mode comparison.

2. **`|11⟩`-block leakage.** A perfect quantum simulator preserves
   the velocity-register subspace structure exactly. Hardware /
   noise breaks this. Track via `leakage` metric (§4.5). Threshold
   `1e-3` is a starting point; tune based on observed values from
   the no-noise sim baseline.

3. **Negative-magnitude regime without sign recovery.** Silent
   accuracy loss in shock-dominated runs. Per-step warning (§3.3) is
   the user signal.

4. **Endianness mismatch.** Qiskit bitstring → index conversion in
   the shots branch must match the statevector path's assumed
   ordering. v1 acceptance test verifies via a single-step round
   trip: `f_post_statevector` and `f_post_shots(S=10⁶, no noise)`
   agree to `O(10⁻³)` L² (shot noise floor).

5. **Per-step circuit rebuild cost.** Currently dominant at large q:
   building `U_collision` as a dense `(4N × 4N)` `UnitaryGate` is
   `O(N²)` memory and per-step. The shots path inherits this. Out of
   scope for v1; flagged as a follow-up gate-decomposition item.

## 6. Acceptance

v1 ships when:

1. `--method qlbm_circuit --shots 8192 --backend-type sim --bc
   periodic --ic sine --nu 1e-2 --n-steps 20 --q 4` runs to
   completion, produces a finite `final_error` against the FTCS
   reference, and matches the statevector reference (same params,
   `--shots 0`) to within `5%` L² at the final step.
2. `--sign-recovery classical_oracle` reproduces the statevector
   result on a synthetic-negative-f test case; `--sign-recovery
   none` does not.
3. The `leakage` metric is `< 1e-6` on the noise-free sim baseline.
4. `tests/test_shots_backend.py::test_qlbm_shots_statevector_agreement`
   and `::test_qlbm_sign_recovery` pass on Aer.
5. OVERVIEW §5.2 updated: drop "shots = classical fallback" from the
   method-roster Notes column; add a sign-recovery subsection
   mirroring §3.3.4 (parts that apply: classical_oracle, hadamard_test).

FUTURE-WORK #14 → DONE.

## 7. Out of scope for this SPEC

- **Hardware noise mitigation.** ZNE/PEC for QLBM is FUTURE-WORK #7
  (hardware execution with error mitigation), not this SPEC. v1
  ships the plumbing; the mitigation layer is separate.
- **Gate-level decomposition of `U_collision` / `U_stream`.** Both
  are dense `UnitaryGate`s today. Decomposition (e.g., controlled
  increment/decrement chains for streaming; a parametric ansatz +
  per-step coefficient set for collision) is its own optimisation
  effort, expected to be needed before `q ≥ 7` is tractable on real
  hardware.
- **`dual_rail` sign recovery.** §3.2 explains why it doesn't fit
  QLBM's non-linear collision; defer to a separate SPEC if needed.
- **Multi-lattice (D1Q5 / D2Q9 / D3Q19) extensions.** Today's code
  hardcodes D1Q3; the velocity-register width and the streaming
  unitary structure both depend on the lattice choice. Multi-lattice
  is out of scope for #14.
- **`--evolution-mode chunked` for QLBM.** Unlike Cole–Hopf,
  collision is state-dependent so a "chunk" of multiple steps cannot
  be precompiled; each step is its own circuit. The chunked-mode
  knob has no useful semantics for QLBM. If the user passes
  `--evolution-mode chunked --method qlbm_circuit`, raise a clean
  `NotImplementedError` (or silently treat as `single`).

## 8. Open questions for the implementer

These are concrete decisions to lock down before coding. Most have a
recommended default in parentheses.

1. **`hadamard_test` in v1 or v2?** Implementing the per-bin Hadamard
   path here would add ~250 LOC mirroring
   `burgers_cole_hopf_circuit.py:1597-1851`. v1 likely ships
   `none` + `classical_oracle` only and raises `NotImplementedError`
   on `--sign-recovery hadamard_test --method qlbm_circuit`.
   *Recommendation: defer.*

2. **`negative_mass` warning threshold.** When `max|f_negative| / max|f|`
   exceeds what value does the per-step warning fire? Below the
   threshold the run is in the "physical" regime and `--sign-recovery
   none` is fine. *Recommendation: 1e-3.*

3. **`leakage` panic threshold.** Above what `leakage` value should
   the run abort with `RuntimeError` instead of just warning?
   *Recommendation: 1e-1 (10% mass leaked from the D1Q3 subspace
   means the simulator/hardware is fundamentally broken for this
   circuit; downstream numbers are meaningless).*

4. **`--evolution-mode chunked --method qlbm_circuit`**:
   `NotImplementedError` (loud) or silently treat as `single` (quiet)?
   *Recommendation: `NotImplementedError` with a helpful message.*

5. **Shots-mode wall-time budget.** The collision unitary is rebuilt
   classically every step. For a shots run, the wall-time profile
   is: classical collision build (O(N²)) + transpile + shot execution
   + classical reconstruction. At `q = 4`, `n_steps = 20`,
   `shots = 8192`, what is the rough budget we should target? If it's
   minutes-not-seconds, document that explicitly so sweep configs can
   plan accordingly. *Recommendation: measure during v1 and document
   in OVERVIEW §5.2.*

## 9. References

- `src/burgers_qlbm_circuit.py` — module to modify.
- `src/burgers_qlbm.py` — classical BGK + streaming kernels reused
  as oracles.
- `src/burgers_sign_recovery.py:38, 69` — `classical_oracle_signs`,
  `hadamard_test_sign_circuit`; same patterns reused.
- `src/burgers_cole_hopf_circuit.py:1597-1851` — Hadamard-per-bin
  reference implementation for §3.2 (if/when we revive
  `hadamard_test`).
- OVERVIEW §2 (method roster), §5 (QLBM family), §7.5 (sign-recovery
  CLI) — doc surfaces to update at the end of v1.
- FUTURE-WORK #14 — the parent item.
- FUTURE-WORK #7 — hardware execution + error mitigation (out of
  scope here; consumes the plumbing this SPEC delivers).
