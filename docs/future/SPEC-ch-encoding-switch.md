# SPEC — CLI-switchable state encoding for `cole_hopf_circuit`

Mini-spec for FUTURE-WORK.md item 2 (encoding change). Target: a
`--encoding` flag that makes the current binary amplitude encoding
(our novel F10 implementation) A/B-comparable against a
locality-preserving encoding that restores paper-faithful W-II / true
Pauli-Trotter evolution.

## 0. Goal

Run the same paper config two ways on the same CLI and compare:

1. `--encoding binary` + existing propagators (qft-diagonal,
   dense-block) — F10 as shipped. "Novel" path.
2. `--encoding gray` + new locality-aware propagator (W-II ladder and/or
   true Pauli-Trotter) — Murali-paper-faithful path.

Answers: does the paper-faithful route give us anything the binary
route doesn't (gate count, accuracy, scaling), or are they equivalent
up to a classical permutation?

## 1. Current state

F10 hard-codes binary amplitude encoding: grid index `i` → qubit state
`|i⟩` with `i = b₀ + 2·b₁ + … + 2^{q-1}·b_{q-1}`. The tridiagonal
Laplacian is nonlocal on the qubit chain (adjacent grid points can be
Hamming-distance q apart). W-II / true Pauli-Trotter were rejected
per F10-IMPLEMENTATION-SPEC.md §4 for exactly this reason.

## 2. Proposed encodings

Pick one (recommended order):

- `binary` — default, no-op, current behavior.
- `gray` — reflected Gray code permutation π(i) = i XOR (i >> 1).
  Same qubit count (N = 2^q). Adjacent grid points are Hamming-1 on
  the qubit chain → the tridiagonal Laplacian decomposes into
  nearest-neighbor qubit-chain terms. W-II ladder and true
  Pauli-Trotter become local products. This is the target
  locality-preserving encoding for the paper comparison.
- `unary` — one-hot, N qubits for N grid points. Perfect locality,
  O(N) qubits, only tractable at tiny q. Deferred to a separate spec.

## 3. CLI surface

```
--encoding {binary, gray}              (default: binary)
```

Plumb through `burgers_solver.py` argparse → `run_simulation` →
`run_cole_hopf_circuit_simulation`.

## 4. Code pieces

### 4.1 New module `burgers_encoding.py`

```python
def permute_to_encoding(v: np.ndarray, encoding: str) -> np.ndarray: ...
def permute_from_encoding(v: np.ndarray, encoding: str) -> np.ndarray: ...
def encoding_permutation(q: int, encoding: str) -> np.ndarray: ...
```

For `gray`: `π[i] = i ^ (i >> 1)`. Pure index permutation on classical
vectors; no quantum-circuit machinery at state-prep time — the
permutation is absorbed into `classical_to_mps(π(ψ))`.

### 4.2 State prep (`run_cole_hopf_circuit_simulation`)

After computing `ψ₀` via Cole-Hopf forward, apply
`permute_to_encoding(ψ₀, encoding)` before `classical_to_mps`. MPS
now decomposes the permuted vector; `mps_to_circuit` is unchanged.

### 4.3 Propagator

Three options for first cut:

- `dense-block` under `gray`: trivially supported — build Laplacian in
  grid basis, permute to encoded basis (`L_enc = Π L Πᵀ`), then
  eigendecompose. No circuit-level changes; encoding is classical
  pre-conditioning of the operator.
- `qft-diagonal` under `gray`: QFT diagonalises the Laplacian only in
  the binary basis. Forbid the combo at the CLI layer for now
  (raise `NotImplementedError` with a clear message). Future work:
  chain permutation-circuit + QFT + inverse-permutation-circuit.
- `wii-ladder` under `gray`: new `--propagator wii-ladder` option,
  gated to `encoding=gray`. Implements Murali's MPO evolution as the
  paper describes. Can reuse F2 Phase B.1 work (`burgers_tebd.py`)
  since Zaletel W-II now has the locality it needs.

### 4.4 Readout

In `_run_shots_batch` and `run_cole_hopf_circuit_sv`: after extracting
`φ̂` in the encoded-index order, apply
`permute_from_encoding(φ̂, encoding)` to return to grid-index order
before `cole_hopf_inverse`.

## 5. Tests (`analysis/test_encoding_switch.py`)

- `test_encoding_roundtrip`: `permute_from(permute_to(v)) == v` for
  both encodings, all q in {3, 4, 5}.
- `test_gray_equivalence_dense_block`: `encoding=gray
  propagator=dense-block` produces the same `u(x, T)` as
  `encoding=binary propagator=dense-block` to 1e-10, same config.
  Proves the encoding layer is a bijection under an equivalent
  propagator — sanity gate for any downstream comparison.
- `test_qft_diagonal_gray_raises`: asserts the forbidden combo errors
  cleanly at the CLI layer.
- `test_gray_laplacian_locality`: decompose `L_gray` into Pauli
  strings; assert max weight ≤ 2 (NN on qubit chain) vs binary's
  O(q) weight. Proves Gray is actually doing what we claim.
- `test_wii_ladder_gray_matches_dense_block_gray` (E-2 only):
  W-II evolution under `gray` matches `dense-block` under `gray`
  within the published W-II Trotter tolerance on a small config.

## 6. Parcels

- **E-1** — `burgers_encoding.py` + CLI flag + permute hooks in
  `run_cole_hopf_circuit_simulation`. `dense-block` works under both
  encodings; `qft-diagonal + gray` raises. Tests: roundtrip,
  equivalence, forbidden combo, locality.
- **E-2** — `--propagator wii-ladder` gated to `encoding=gray`.
  Reuses F2 Phase B.1 ladder-MPO code where possible. Acceptance
  test: matches `dense-block + gray` on q=4 within W-II tolerance.
- **E-3** — (optional) `--propagator true-pauli-trotter` gated to
  `encoding=gray`. Revives F10 acceptance 11.4 first-order Trotter
  convergence (currently vacuous against `dense-block`'s exact
  eigendecomp — see FUTURE-WORK.md item 9).

E-1 is a standalone parcel (~1 day). E-2 and E-3 depend on E-1 and
are the paper-faithful comparison target.

## 7. Acceptance = A/B comparison study

Once E-2 lands, run the paper config at q=5, ν=1e-4, shots=150k
(plus bond-dim sweep from P-G) under four combinations:

| Encoding | Propagator    | Role                          |
|----------|---------------|-------------------------------|
| binary   | qft-diagonal  | F10 novel — minimum gate count |
| binary   | dense-block   | F10 novel — exact per step     |
| gray     | dense-block   | E-1 sanity gate                |
| gray     | wii-ladder    | E-2 — paper-faithful target    |

Deliverable: per-variant gate count + `u(x, T)` error vs the
classical `cole_hopf` reference, on one PNG. Answers whether the
locality-preserving path is worth the encoding complexity for this
problem size, and grounds the choice for F11 / hardware targets.

## 8. Out of scope

- `unary` encoding (separate spec if ever wanted).
- Changing classical-side methods (`shift`, `cole_hopf` classical) —
  encoding is strictly a quantum-side concern.
- Peaked-φ readout mitigation (FUTURE-WORK.md item 10) — orthogonal
  to encoding; both encodings hit the same peak.
- Hardware execution (FUTURE-WORK.md item 7).
