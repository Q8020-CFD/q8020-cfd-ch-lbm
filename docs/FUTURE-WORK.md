# Future Work

Items out of scope for F10 and its two review patches
(F10-REVIEW-PATCH.md, F10-REVIEW-PATCH-02.md). Each entry lists why it
matters, rough scope, and what it depends on. Listed roughly in order
of likely priority; all are independent unless a dependency is noted.

## 1. F11 — Burgulence (multi-mode stochastic IC ensemble)

**Why.** Alhawwary-Wang §5.3 is the next paper-comparison benchmark
after single-shock Burgers. Validates the solver on a randomized IC
with a broadband energy spectrum.

**Scope.** Ensemble driver on top of F10: N IC realizations, per-run
statistics aggregation (spectra, structure functions). No new circuit
machinery.

**Depends on.** F10 closed at paper-target ν=1e-4 (i.e. P-G + P-H
merged, acceptance 11.7 / 11.8 equivalents passing). A stub spec exists
at `SPEC-alhawwary-wang-5.3-burgulence.md`.

## 2. Encoding change — binary → locality-preserving

**Why.** Binary amplitude encoding puts physical neighbors on qubits
that are Hamming-distant; any operator local in x is nonlocal on the
qubit chain. This is the root cause behind Zaletel W-II being rejected
for F2, and behind the dense-block fallback in F10. A Gray-code or
block-local encoding would make real Pauli-Trotter, Zaletel W-II, and
direct u-space evolution all tractable.

**Scope.** Large. New state-prep pipeline, rewrite of every
space-local operator, new CLI surface. F10 acceptance tests would
re-baseline.

**Depends on.** Nothing; this is the enabling prerequisite for items
3, 8, 9.

## 3. Direct u-space evolution via Carleman linearization

**Why.** Cole-Hopf linearization is only available for 1D Burgers
with no source; extending to 2D/3D Navier-Stokes or forced Burgers
requires a different linearization. Carleman embeds the nonlinearity
into an infinite linear hierarchy truncated at some order K.

**Scope.** ~F10-sized. New propagator, truncation-error analysis,
separate acceptance suite. Drops Cole-Hopf entirely.

**Depends on.** Nothing technical; but probably wants encoding
change (item 2) first for the linear operator to be local.

## 4. DST-based `qft-diagonal` + Dirichlet

**Why.** Today `qft-diagonal` silently falls back to `dense-block`
under Dirichlet BC because QFT diagonalizes only the periodic
Laplacian. A discrete-sine-transform variant would diagonalize the
Dirichlet Laplacian and restore qft-diagonal as a gate-count-optimal
path for the paper regime.

**Scope.** Moderate. New `dst_diagonal_step_circuit` in
`burgers_cole_hopf_circuit.py`, reuse conditional-Ry machinery on
sine-basis eigenvalues. No change to propagator dispatch CLI.

**Depends on.** Nothing.

## 5. QSVT polynomial alternative to ancilla-Ry

**Why.** The conditional-Ry Möbius expansion costs O(2^q) terms for
an exact polynomial fit; QSVT gives a polynomial-degree / error
tradeoff that scales better at q ≥ 7. Also removes the ancilla.

**Scope.** Substantial. QSVT phase-angle computation,
block-encoding setup, integration as a third `--propagator` option.

**Depends on.** Nothing.

## 6. QROM-based θ(k) loading for q ≥ 7

**Why.** At q ≥ 7 the Möbius expansion in `build_conditional_ry`
has 128+ terms and gate depth becomes the bottleneck. QROM loads
θ(k) as classical data into a quantum register in O(2^q) gates but
with shallower depth and better ancilla tradeoffs.

**Scope.** Moderate. QROM construction + controlled-rotation from
loaded register. Drop-in replacement for `build_conditional_ry` at
large q; keep both.

**Depends on.** Nothing. Becomes load-bearing only at q ≥ 7.

## 7. Hardware execution with error mitigation

**Why.** Everything so far is Aer. Running on real hardware is the
eventual goal. Needs noise-aware transpilation, zero-noise
extrapolation or probabilistic error cancellation, and a sweep
harness that tolerates queue latency.

**Scope.** Large. New backend abstraction, mitigation pipeline,
benchmark protocol. Starts with q=3–4 calibration circuits.

**Depends on.** F10 closed + P-H.1 readout (peaked-φ shots on
hardware would be catastrophic without it).

## 8. F2 `tebd_circuit` revival

**Why.** The original F2 proposal was a true TEBD circuit evolving
u directly (no Cole-Hopf). Shelved because Zaletel W-II is nonlocal
under binary encoding. Becomes tractable under item 2.

**Scope.** Revisit F2 spec; likely a full rewrite.

**Depends on.** Item 2 (encoding change).

## 9. True Pauli-Trotter propagator

**Why.** Originally pitched as Fork A in F10-REVIEW-PATCH.md. A
genuine product-formula expansion of exp(νLΔt) using Pauli strings
on the qubit chain. Currently nonlocal on binary encoding — the
string weight blows up with q. Was renamed to `dense-block` in F10
P-A (Fork B). Restores acceptance 11.4 first-order Trotter-error
convergence (currently vacuous against `dense-block`'s exact
eigendecomp) and gives Murali a Pauli-level object to reason about
directly, matching the paper's framing.

**Scope.** Moderate once encoding is local. New propagator alongside
`qft-diagonal` and `dense-block`; gate-count scaling study vs the two
existing variants.

**Depends on.** Item 2 (encoding change) to be competitive; on
binary encoding it is strictly worse than `dense-block`.

## 10. Peaked-φ shots readout — open gap

**Why.** At paper-target ν=1e-4, φ(x) = exp(−∫u/2ν) concentrates
almost all probability mass on ~1 grid bin, and √-of-counts readout
at `burgers_cole_hopf_circuit.py:504-509` wastes shots on the peak
while tail bins have p_i ≪ 1/shots. Observable symptom:
`test_11_5_shots_accuracy` runs at ν=0.1 instead of the spec's
ν=1e-2 because the low-ν path degrades past the 5% tolerance.
F10-REVIEW-PATCH-02.md P-H proposes Hadamard-test per bin as one
fix; if that proves too expensive at larger q, other options are
preconditioning the state to flatten φ before measurement,
importance sampling, or switching to a log-amplitude readout
altogether. No fix is merged.

**Scope.** Moderate per option. Hadamard-test path reuses the F9
sign-recovery ancilla wiring. Log-amplitude readout is a deeper
rewrite of the readout stage.

**Depends on.** Nothing. Load-bearing for any production sweep at
ν < 1e-3.

## 11. Qiskit 2.3 → 3 deprecation: `RYGate.control(annotated=None)`

**Why.** `build_conditional_ry` in `burgers_cole_hopf_circuit.py`
relies on `RYGate.control(annotated=None)`, deprecated in Qiskit
2.3 and slated for removal in Qiskit 3. At the Qiskit 3 bump the
conditional-Ry Möbius expansion stops transpiling and every
`cole_hopf_circuit` path breaks.

**Scope.** Small. Switch to the explicit `annotated=True` or
`annotated=False` form (decide based on transpiler-pass
compatibility), update tests.

**Depends on.** Nothing. Do this whenever we next touch Qiskit
pinning.
