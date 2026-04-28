# SPEC — F3: LCU SELECT/PREPARE method

Self-contained handoff. Reader has not seen the prior conversation.

> **Status (2026-04-28).** Variant 1.B (pure-quantum CH-LCU) shipped
> for unforced runs. Source forcing for the LCU propagator landed
> via [SPEC-F3-LCU-source-forcing.md](SPEC-F3-LCU-source-forcing.md)
> using a Strang-split sandwich (`exp(-V·dt/2) · LCU_heat ·
> exp(-V·dt/2)`). Both forced and unforced runs supported via
> `--method cole_hopf_circuit --propagator lcu`. Variant 1.A
> (direct-`u` LCU, paper-faithful) remains deferred.

## 0. Context

[IMPLEMENTATION-PLAN.md:202](IMPLEMENTATION-PLAN.md:202) reserves
F3 for the LCU (Linear Combination of Unitaries) SELECT/PREPARE
construction. [REVIEW R2](REVIEW-murali-paper-fidelity.md#r2) flags
that LCU may be the paper's preferred quantum-circuit path, and
that wiring the existing dead-code stubs into a real method would
close the question.

We have unwired raw materials in
[`burgers_mpo.py`](../src/burgers_mpo.py):

- `increment_circuit(q)` — `S+` ladder operator (binary increment
  mod 2^q).
- `decrement_circuit(q)` — `S-` (= `(S+)†`).
- `gradient_lcu_circuit(q)` — block-encodes `(S+ − S-)/2` with one
  ancilla; post-select `anc=|1⟩`.
- `laplacian_lcu_circuit(q)` — block-encodes `(S+ + S- − 2I)/4`
  with two ancillas; post-select `anc=|00⟩`.
- `extract_block_encoded_operator(...)` — utility for verifying a
  block-encoding numerically.

These functions are tested in their `__main__` smoke. They are not
called by any integrator. F3 wires them up.

## 0.1 Scope directive (read this before §1)

**Implement variant 1.B (pure-quantum CH-LCU) only. Variant 1.A
(paper-faithful direct-`u` LCU) is documented in §1.A and §5.2 for
future reference but is NOT in scope right now.**

Rationale: variant 1.A reintroduces a per-step classical mirror
(the `diag(u_n)` factor in the Burgers Hamiltonian is state-
dependent; computing the LCU PREPARE coefficients each step
requires knowing `u_n` classically). That is the same architectural
pattern as the existing `quantum_circuit` method and the same
"steerage" concern that F10 was built to eliminate. We are not
adding another classical-mirror method to the codebase right now.

The pure-quantum path is the priority: linearise via Cole-Hopf so
the heat propagator is state-independent, then build it as an LCU.
No classical mirror in the time loop. Same purity guarantee F10
established for the `dense-block` and `qft-diagonal` propagators,
extended to LCU.

For the implementer:

- **Skip §5.2** (`lcu_circuit` method wiring) entirely.
- **Skip the `lcu_circuit_*` TOML cases in §5.6.**
- **Skip tests #4, #5 in §5.5.**
- **Skip step L3 in §7's implementation order.** Total work drops
  from ~3 days to ~1.5 days (L1, L2, L4, L5, L6).
- **Acceptance gates relating to `lcu_circuit` in §6 are moot.**
- **REVIEW R2 is NOT closed by this parcel.** The R2 annotation
  in §6 acceptance applies to variant 1.A only; defer R2
  resolution to a future `SPEC-F3-direct-u-lcu.md` parcel that
  picks up variant 1.A if and when we want a classical-mirror
  LCU path. Note in REVIEW-murali-paper-fidelity.md R2 that "F3
  pure-quantum CH-LCU shipped via SPEC-F3-LCU-method.md;
  paper-faithful direct-`u` LCU path remains open."

## 1. Two variants and what they buy

The LCU *circuits* listed above are pure-quantum: ancillas + gates
+ post-selection, no classical mirror inside the circuit. The
operators they encode (gradient, Laplacian) are
**state-independent** spatial discretisations.

Burgers, however, has a **state-dependent** Hamiltonian
`H = ν·L − diag(u)·G + diag(g)`. The `diag(u)` factor is the
nonlinearity. There are two distinct ways to use the LCU stubs to
make a Burgers method:

### 1.A Paper-faithful direct-`u` LCU (closes R2)

Build `H_n = ν·L − diag(u_n)·G + diag(g_n)` each step (state-
dependent), expand into Pauli (or LCU) coefficients
`H_n = Σ c_k P_k`, apply `exp(−i·dt·H_n)` via SELECT/PREPARE LCU
instead of sequential Trotter rotations.

- **Pure quantum?** No. The LCU *circuit* is pure quantum, but the
  PREPARE oracle's amplitudes (the `c_k`) depend on `u_n`. Computing
  them requires a classical mirror per step — same architectural
  pattern as today's `quantum_circuit` (Pauli-Trotter) method.
- **What it buys**: scaling. LCU applies the sum of unitaries with
  one PREPARE-SELECT-UNPREPARE block + ancilla post-selection,
  ancilla width O(log K) for K terms; replaces the long sequential
  Trotter chain that today's `quantum_circuit` synthesises. This
  is the bottleneck the paper's Appendix A flags as the key to
  scaling beyond q=8.
- **Closes R2?** Yes — this is the paper-claim path.
- **Method name**: `lcu_circuit`.

### 1.B Pure-quantum CH-LCU (extends F10)

Apply Cole-Hopf transform `u → φ`, march the heat equation
`∂φ/∂t = ν·∂_xx φ` whose evolution operator `P = exp(ν·L·dt)` is
**state-independent**. Build `P` as an LCU SELECT/PREPARE.

- **Pure quantum?** Yes. State-independent; coefficients of the
  LCU PREPARE depend only on `(ν, dt, L)`, set once, no per-step
  classical predictor. This is the marriage with F10.
- **What it buys**: a third propagator option for
  `cole_hopf_circuit` alongside `qft-diagonal` and `dense-block`.
  Compared to `dense-block`'s explicit eigendecomposition, LCU may
  have better depth scaling at large `q` because the operator
  decomposes into `O(poly(q))` ladder primitives instead of
  `O(2^q)` controlled rotations.
- **Closes R2?** Indirectly — it answers "is the paper's LCU
  approach realisable on our infrastructure?" affirmatively, even
  though the *method is not the paper's direct-`u` method.*
- **Propagator name**: `--propagator lcu`.

This spec covers **both** because they share most of the work
(SELECT, PREPARE, post-select machinery; testing harness; CLI
plumbing). Implementer can ship 1.A first to close R2 explicitly,
or 1.B first if pure-quantum scaling is the priority. They do not
block each other.

## 2. Goal

After this parcel:

- A new direct-`u` quantum method `--method lcu_circuit` runs on
  the simulator end-to-end: state prep → SELECT/PREPARE-based
  evolution → post-select on ancilla → readout. Numerically
  matches `quantum_circuit` to within Trotter+post-select
  tolerance; scales to at least q=8 where today's
  `quantum_circuit` Pauli-Trotter chain becomes painful.
- A new propagator choice `--propagator lcu` for
  `cole_hopf_circuit` runs on the simulator end-to-end. Matches
  `--propagator dense-block` to within numerical tolerance at
  q ≤ 5 (the regime where dense-block is exact); produces working
  output at q ≥ 6 where dense-block's explicit eigendecomp gets
  unwieldy.

## 3. Non-goals

- **LCU on real hardware.** Sim only in v1. Real hardware has its
  own ancilla-management surface; defer.
- **Variational fast-forwarding (F4).** Different parcel; LCU here
  is per-step, not amortised across steps.
- **Krylov subspace (F5).** Different parcel.
- **Replace `quantum_circuit` with `lcu_circuit`.** Both coexist;
  `quantum_circuit` stays as the Pauli-Trotter reference.
- **Replace `dense-block` with `lcu`.** Both coexist; `dense-block`
  stays as the eigendecomp reference for small q.
- **MPO-driven LCU (Lubasch QNPU).** That is a third path, not
  this one. F3 here uses Pauli/ladder LCU only.
- **Hadamard / dual-rail sign recovery integration.** F9 already
  ships that surface for the direct-`u` shots path; F3's
  `lcu_circuit` reuses it unchanged. CH-LCU doesn't need it
  (φ ≥ 0).

## 4. Math: SELECT/PREPARE

### 4.1 The standard LCU recipe

Given a target operator `A = Σ_k c_k U_k` with each `U_k` unitary
and `c_k ∈ ℝ` (without loss of generality; absorb signs into
`U_k`), define `λ = Σ_k |c_k|`, `α_k = √(|c_k|/λ)`. Then with `m`
ancilla qubits where `K = 2^m ≥` number-of-terms:

```
PREPARE: |0⟩^m  →  Σ_k α_k |k⟩         (state preparation)
SELECT:  |k⟩|ψ⟩ →  |k⟩ sign(c_k) U_k |ψ⟩
UNPREP:  inverse of PREPARE

Block-encoded: ⟨0|^m · UNPREP · SELECT · PREPARE · |0⟩^m  =  A / λ
```

Post-select on ancilla = `|0⟩^m` after UNPREP; success probability
`‖A|ψ⟩‖² / λ²`. The structure is the same whether `A` is the
Hamiltonian (paper-faithful, 1.A) or the heat propagator (CH-LCU,
1.B).

### 4.2 Variant 1.A: direct-`u` Hamiltonian

Per timestep `n` with current state `u_n`:

1. Compute `H_n = ν·L − diag(u_n)·G + diag(g_n)` classically
   (state-dependent — this is the classical mirror).
2. Decompose into a sum of cheap unitaries:
   `H_n = Σ_k c_k(u_n) U_k` where the `U_k` are products of `S+`,
   `S-`, identity, and computational-basis-diagonal phase factors.
   The `diag(u_n)` factor expands into a sum of one-hot phase
   gates weighted by `u_n[i]`; the `diag(g_n)` similarly.
3. Apply `exp(−i·dt·H_n)` via Trotter on the LCU-encoded `H_n`,
   or via QSP / qubitisation if depth permits (out of scope for
   v1; v1 uses LCU + first-order Trotter).
4. Post-select on all ancilla bits = 0 across all SELECT blocks.
5. Read out / reconstruct `u_{n+1}`.

Coefficient count `K_n = O(N)` per step (the `diag(u_n)` and
`diag(g_n)` contribute `N` terms each; `L` and `G` contribute `O(1)`
LCU terms each). Ancilla width: `m = ⌈log₂ K_n⌉ ≈ q + O(1)`. Per-
step depth: SELECT is `O(K_n)` controlled-unitaries on the system;
PREPARE / UNPREP are state preparations of `α_k` which are
themselves `O(K_n)` gates in the worst case. Net: `O(N · poly(q))`
per step on `2q + O(1)` qubits. Compare with `quantum_circuit`'s
`O(4^q)` Pauli-Trotter chain — a substantial scaling improvement.

### 4.3 Variant 1.B: CH-LCU heat propagator

The φ-equation propagator
`P(dt) = exp(ν · L · dt) = Σ_k=0^∞ (ν·dt)^k · L^k / k!` is real,
symmetric, contractive (eigenvalues in (0, 1]).

Two construction options:

- **Truncated Taylor LCU** — pick truncation order `M`, build
  `P_M = Σ_k=0^M (ν·dt)^k · L^k / k!` as an LCU. Each `L^k` is a
  product of `(S+ + S- − 2I)/dx²` terms; expand into a sum over
  `S+`/`S-`/`I` strings. Pure quantum. Coefficient count grows as
  `O(M · 3^M)` — manageable for `M ≤ 4`.
- **Linear combination of `cos(√(ν·dt)·k_j)` factors** — sin/cos
  decomposition approach used in Hamiltonian simulation literature.
  More complex to derive; defer to v2 if needed.

V1 ships truncated Taylor; v2 considers cos/sin if depth-vs-error
trade is unfavourable.

Post-select on all-ancilla-zero per step; cumulative
`P_success ≈ p^N_steps` (multiplicative across steps), so chunked-
evolution mode (already in tree, see
[SPEC-chunked-evolution.md](SPEC-chunked-evolution.md)) is the
right pairing for long horizons.

## 5. Plumbing changes

### 5.1 New module — `burgers_lcu.py`

Top-level driver for both variants. Signature mirrors
`run_cole_hopf_circuit_simulation` and the per-step quantum
methods. Outline:

```python
def build_select_circuit(
    operators: list[QuantumCircuit],     # K unitaries
    n_system: int,
    n_ancilla: int,                       # must be ⌈log₂ K⌉
) -> QuantumCircuit:
    """Construct SELECT: controlled-apply U_k when ancilla = k."""

def build_prepare_circuit(
    coefficients: np.ndarray,             # length K, real
    n_ancilla: int,
) -> QuantumCircuit:
    """Construct PREPARE: |0⟩^m → Σ_k √(|c_k|/λ) |k⟩."""
    # Use Qiskit StatePreparation on the ancilla register.
    # Sign of c_k absorbed into the corresponding U_k via a
    # diagonal phase wrap before the SELECT block.

def lcu_block_encoding(
    operators: list[QuantumCircuit],
    coefficients: np.ndarray,
    n_system: int,
) -> tuple[QuantumCircuit, float]:
    """Return (full_circuit, lambda) where lambda = Σ |c_k|.
    The circuit block-encodes A/λ where A = Σ c_k U_k."""
    n_anc = int(np.ceil(np.log2(len(operators))))
    qc = QuantumCircuit(n_system + n_anc, name="LCU")
    qc.compose(build_prepare_circuit(coefficients, n_anc),
               qubits=list(range(n_system, n_system + n_anc)),
               inplace=True)
    qc.compose(build_select_circuit(operators, n_system, n_anc),
               inplace=True)
    qc.compose(build_prepare_circuit(coefficients, n_anc).inverse(),
               qubits=list(range(n_system, n_system + n_anc)),
               inplace=True)
    return qc, float(np.sum(np.abs(coefficients)))
```

These primitives are shared by both variants. Reuse `S+`, `S-`,
`gradient_lcu_circuit`, `laplacian_lcu_circuit` from
`burgers_mpo.py` as the canonical state-independent unitaries.

### 5.2 Variant 1.A driver — `lcu_circuit` method

In a new function `run_lcu_circuit_simulation(...)`:

1. Each step `n`, build `H_n` classically (the mirror).
2. Decompose `H_n` into `(operators, coefficients)`. For Burgers:
   - `ν·L`: one term, operator = laplacian_lcu_circuit, coefficient
     = `ν / dx²`.
   - `−diag(u_n)·G`: `N` terms, operator k = `|k⟩⟨k| ⊗
     gradient_lcu_circuit`, coefficient k = `−u_n[k] / dx`. The
     `|k⟩⟨k|` is implemented by a multi-controlled phase on the
     system register.
   - `diag(g_n)`: `N` terms, operator k = `|k⟩⟨k|` (diagonal phase),
     coefficient k = `g_n[k]`.
3. Form the unitary `exp(−i·dt·H_n)` via first-order Trotter on
   the LCU components: `Π_term exp(−i·dt·c_k U_k)`. (For each
   term, `exp(−i·dt·c·U)` for unitary `U` is itself unitary; this
   step does not use LCU per se, only the LCU primitives as
   building blocks. Pure LCU-based Hamiltonian simulation —
   Berry/Childs/Kothari truncated Taylor — is a follow-up; v1
   uses LCU as a circuit-synthesis tool inside Trotter.)
4. Compose with state prep + measurement; run; post-select.

Wire into [`burgers_fw.py`](../src/burgers_fw.py) as a per-step
integrator (not a delegating one — same shape as
`QuantumCircuitIntegrator`):

```python
class LCUCircuitIntegrator(TimeIntegrator):
    def step(self, state, spatial_op, grid, config, dt, t=0.0):
        from burgers_lcu import lcu_circuit_step
        u = state.to_dense()
        g = config._source_fn(grid.xc, t) if config._source_fn else None
        u_new, metrics = lcu_circuit_step(
            u, grid.dx, dt, config.nu, g, bc=grid.bc,
            shots=config.shots, backend=self.backend,
            sign_recovery=config.sign_recovery,
            t1=config.t1, t2=config.t2,
            backend_name=config.backend_name,
        )
        return DenseState(u_new), metrics
```

Add `"lcu_circuit"` to the argparse choices in
[`burgers_solver.py:113-117`](../src/burgers_solver.py:113) and to
the registry in
[`burgers_fw.py::make_integrator`](../src/burgers_fw.py:340).

### 5.3 Variant 1.B driver — `lcu` propagator for `cole_hopf_circuit`

Extends [`burgers_cole_hopf_circuit.py`](../src/burgers_cole_hopf_circuit.py).
Add `propagator="lcu"` branch alongside `qft-diagonal` and
`dense-block` in
[`heat_dense_block_full_circuit`](../src/burgers_cole_hopf_circuit.py:294)
and its SV twin.

```python
def heat_lcu_step_circuit(
    q: int, nu: float, dt: float, L_box: float,
    bc: str = "periodic",
    encoding: str = "binary",
    V: np.ndarray | None = None,        # source-forced potential
    taylor_order: int = 4,
) -> QuantumCircuit:
    """LCU block-encoding of the heat propagator for one timestep.

    Builds P_M = Σ_{k=0}^{taylor_order} (ν·dt)^k · L^k / k!
    as an LCU using S+/S- ladder primitives.  V (optional) adds
    a diagonal potential layer, same role as in the dense-block
    path.
    """
```

The `--propagator` argparse choice (`burgers_solver.py:122`) gains
`"lcu"`. Source-forcing guard
([SPEC-source-forcing.md](SPEC-source-forcing.md) §5.2) extends
to: `lcu + V` is supported in v1 (V is diagonal, fits in the LCU
naturally as a diagonal phase term added to `L`).

### 5.4 CLI flags

Two new flags, both optional:

```
--lcu-taylor-order INT       default: 4   (CH-LCU only)
--lcu-trotter-reps INT       default: 1   (lcu_circuit method only;
                                          mirrors --trotter-reps)
```

Both default to values that should give acceptable accuracy at
the typical `(q, ν, dt)` we run. Tune via convergence studies.

### 5.5 Tests

`tests/test_lcu.py`:

1. **`test_lcu_block_encoding_correctness`** — for a synthetic
   `A = c_0 X + c_1 Z` on 1 qubit, verify
   `extract_block_encoded_operator` of the LCU circuit returns
   `A / (|c_0| + |c_1|)` to 1e-12.
2. **`test_lcu_select_prepare_handles_K_not_pow2`** — with K=3 on
   2 ancillas (one ancilla state unused), confirm post-selection
   isolates the correct subspace.
3. **`test_gradient_lcu_matches_compute_rhs_shift`** — apply
   `gradient_lcu_circuit(q)` to a sine state, post-select, compare
   to classical `compute_rhs_shift`'s gradient component. Tolerance
   1e-10. Already half-tested in the `__main__` smoke; promote to
   a real test.
4. **`test_lcu_circuit_method_matches_quantum_exact`** — at q=4,
   ν=0.1, n_steps=10, source=none, statevector path: relative
   L2 < 0.05 between `lcu_circuit` and `quantum_exact` (both
   solve the same direct-`u` Burgers; LCU+Trotter has Trotter
   error, exact does not).
5. **`test_lcu_circuit_method_scales_to_q8`** — at q=8, ν=0.1,
   n_steps=5, source=none: completes in finite time, no OOM.
   `quantum_circuit` at q=8 with default Pauli-Trotter should be
   the slow comparison; LCU should win.
6. **`test_ch_lcu_propagator_matches_dense_block_at_q5`** — at
   q=5, ν=0.1, n_steps=20, source=none, shots=0: relative L2
   between `--propagator lcu` and `--propagator dense-block` <
   0.02 at the final time.
7. **`test_ch_lcu_propagator_with_source`** — at q=5, ν=0.1,
   source=sine, n_steps=10, shots=0: matches FTCS forced
   reference within 0.05.
8. **`test_ch_lcu_p_success_above_threshold`** — sanity: with
   shots=10000 and chunked mode (chunk_size=5), per-chunk
   `p_success > 0.3` at q=5, ν=0.1.

### 5.6 q8020 TOML smoke cases

```toml
[lcu_circuit_smoke_q4]
"--method" = "lcu_circuit"
"--ic" = "sine"
"--source" = "none"
"--nu" = 0.1
"--cfl" = 0.1
"--shock-pct" = 100.0
"--q" = 4
"--shots" = 0
"--bc" = "periodic"

[lcu_circuit_scale_q8]
"--method" = "lcu_circuit"
"--ic" = "sine"
"--source" = "none"
"--nu" = 0.1
"--cfl" = 0.1
"--shock-pct" = 50.0
"--q" = 8
"--shots" = 0
"--bc" = "periodic"

[cole_hopf_circuit_lcu_smoke_q5]
"--method" = "cole_hopf_circuit"
"--propagator" = "lcu"
"--ic" = "sine"
"--source" = "none"
"--nu" = 0.1
"--cfl" = 0.1
"--shock-pct" = 100.0
"--q" = 5
"--shots" = 0
"--bc" = "periodic"
"--lcu-taylor-order" = 4

[cole_hopf_circuit_lcu_forced_q5]
"--method" = "cole_hopf_circuit"
"--propagator" = "lcu"
"--ic" = "sine"
"--source" = "sine"
"--nu" = 0.1
"--cfl" = 0.1
"--shock-pct" = 100.0
"--q" = 5
"--shots" = 50000
"--backend-type" = "sim"
"--seed" = 42
"--evolution-mode" = "chunked"
"--chunk-size" = 10
"--lcu-taylor-order" = 4
```

## 6. Acceptance

- [ ] All eight §5.5 tests pass.
- [ ] `lcu_circuit_smoke_q4` final L2 vs FTCS < 0.05.
- [ ] `lcu_circuit_scale_q8` completes; `quantum_circuit` at the
      same `(q, n_steps)` either also completes (and is slower in
      wall-time) or visibly OOMs / hangs in transpile. The
      scaling claim is empirical.
- [ ] `cole_hopf_circuit_lcu_smoke_q5` final L2 vs FTCS < 0.05;
      matches `--propagator dense-block` to within 0.02.
- [ ] `cole_hopf_circuit_lcu_forced_q5` final L2 vs FTCS forced <
      0.10 (looser bound — chunked + Taylor truncation + shots).
- [ ] [REVIEW R2](REVIEW-murali-paper-fidelity.md#r2) annotated
      "Resolved by SPEC-F3-LCU-method.md, variant 1.A.
      `lcu_circuit` method available."
- [ ] [IMPLEMENTATION-PLAN.md F3](IMPLEMENTATION-PLAN.md:202)
      annotated with status pointer to this spec.
- [ ] [OVERVIEW-burgers-solver.md](OVERVIEW-burgers-solver.md) §2
      table updated with `lcu_circuit` method and `lcu`
      propagator.

## 7. Implementation order

L1: §5.1 `burgers_lcu.py` primitives + tests #1, #2. Pure utility,
    no method wiring. Half day.
L2: §5.5 test #3 promoting the existing `__main__` smoke.
    30 min.
L3: §5.2 variant 1.A `lcu_circuit` method end-to-end + tests #4,
    #5. ~1 day. This closes R2.
L4: §5.3 variant 1.B `--propagator lcu` for cole_hopf_circuit +
    tests #6, #7, #8. ~1 day. This is the pure-quantum extension.
L5: §5.6 TOML smokes + manual q8020 sweep. 30 min.
L6: §6 doc annotations and review-doc updates. 30 min.

Total: ~3 days, with L3 and L4 independent (either order).

## 8. Out of scope (future work)

- **Berry/Childs/Kothari truncated-Taylor Hamiltonian simulation.**
  V1 uses LCU as a circuit-synthesis tool inside first-order
  Trotter; replacing the outer Trotter with QSP / qubitisation /
  truncated Taylor is the next step up in Hamiltonian-simulation
  technology. Significant theoretical depth; defer.
- **MPO-driven LCU (Lubasch QNPU).** The paper's *primary*
  classical path uses MPO + dense-detour for the nonlinear term.
  Converting that into a quantum circuit is the unsolved problem
  the paper itself names. Out of scope.
- **Variational-PREPARE compilation.** PREPARE for `Σ √(|c_k|/λ)
  |k⟩` is exponentially costly in the worst case; a variational
  ansatz could compress it. Future work.
- **Hardware execution.** Submission stub may accept LCU circuits
  unchanged; harvest workflow may need adjustment for the larger
  ancilla register. Defer until the harvester is robust on
  smaller circuits.
- **`lcu` propagator in single-evolution mode at large depth.**
  Per-step ancilla post-select compounds; chunked mode is the
  right pairing. Single-evolution `lcu` is supported at small
  `n_steps` for sanity but not as a production path.
- **Adaptive `taylor_order` based on `(ν·dt·‖L‖)`.** Fixed in v1.
  Adaptive selection requires a runtime norm estimate; reasonable
  follow-up.
