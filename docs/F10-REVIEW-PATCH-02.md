# F10 — Review Patch 02

Second patch. Two parcels. Addresses the two remaining load-bearing gaps
identified in the standing-inventory review: (1) `cole_hopf_circuit`
bypasses the paper-faithful MPS state-prep pipeline, and (2) shots
readout collapses at small ν because φ concentrates on ~one bin.

Apply on top of the currently-merged F10 code (post F10-REVIEW-PATCH.md).
**F10-IMPLEMENTATION-SPEC.md** remains authoritative; this patch only
revises it where called out below.

Parcels are independent; can be dispatched in parallel.

## P-G — Wire MPS / Ran 2020 state prep into `cole_hopf_circuit`

**Problem.** Today
[burgers_cole_hopf_circuit.py:463-464](../burgers_cole_hopf_circuit.py)
initializes the quantum state with Qiskit's generic
`QuantumCircuit.initialize(init_sv, range(q + 1))`. This bypasses
`burgers_mps.py` entirely. Consequences:

- Spec §2 item 2 and §4 "State prep" row claim MPS → Ran 2020 → circuit,
  faithful to Meena/Murali AIAA-2026 Eq. 5-6 + Ref [27]. The claim is
  currently false for F10 outputs.
- Dense `initialize` is O(4^q) gates; Ran 2020 is O(q · D²). Irrelevant
  at q=5; load-bearing at q ≥ 8.
- `--bond-dim` (used by `--method mps`) has no effect on
  `cole_hopf_circuit`, so no bond-dim sweep / truncation study is
  available for the F10 route.

**Fix.** Replace the `initialize` call with the Ran 2020 pipeline that
already exists and is under test in `burgers_mps.py`:

    from burgers_mps import (
        classical_to_mps, mps_to_circuit, normalize_state,
    )

    psi_norm, _ = normalize_state(psi_prep)
    tensors = classical_to_mps(
        psi_norm, bond_dim=bond_dim, canonical="right",
    )
    prep_qc = mps_to_circuit(tensors)      # acts on data + bond qubits

Compose `prep_qc` onto the data register of the q+1-qubit circuit;
ancilla stays in `|0⟩`. The input `psi_prep` is the classical length-N
vector that currently feeds `initialize` (i.e. `psi0` after Cole-Hopf
forward + optional centering normalization). Do not change the
normalization / centering logic — §9 is untouched.

**Thread `--bond-dim` through.**

- [burgers_solver.py:244](../burgers_solver.py) already passes
  `bond_dim` to `run_simulation`. Extend the `cole_hopf_circuit` branch
  at [burgers_trotter.py:712-718](../burgers_trotter.py) to accept and
  forward `bond_dim=bond_dim`.
- Add `bond_dim: int | None = None` parameter to
  `run_cole_hopf_circuit_simulation` and
  `_run_shots_batch` in
  [burgers_cole_hopf_circuit.py](../burgers_cole_hopf_circuit.py).
- Pass `bond_dim` through into `classical_to_mps(...)`.
- Default `None` preserves the current behaviour (full rank); any finite
  value truncates the MPS.

**Statevector path parity.** The existing statevector driver
`run_cole_hopf_circuit_sv` at
[burgers_cole_hopf_circuit.py:418](../burgers_cole_hopf_circuit.py)
constructs its own statevector seed directly from `psi0` — it does not
invoke a prep circuit. For fairness with the shots path, the SV driver
must optionally run through the same MPS-prep circuit so bond-dim
truncation is visible in both paths. Add a `use_mps_prep: bool = True`
flag; when True, apply `prep_qc` to `|0…0⟩` and use the resulting state
instead of the raw `psi0`.

**New TOML group** (per-parcel sweep with bond-dim study):

```toml
[paper_cole_hopf_circuit_q5_shots150k_mps]
"--q" = 5
"--nu" = 1e-4
"--shock-pct" = 80.0
"--method" = "cole_hopf_circuit"
"--propagator" = "dense-block"
"--bc" = "dirichlet"
"--source" = "none"
"--shots" = 150000
"--bond-dim" = [1, 2, 4]
"--save-every" = 8
_group_postproc = ["python ./q8020-cfd-axequalsb/src/murali_burgers/analysis/plot_cole_hopf_circuit_evolution.py"]
```

**Acceptance.**

1. New test `test_mps_prep_used` in
   `analysis/test_cole_hopf_circuit.py`: run `cole_hopf_circuit` at q=4,
   `bond_dim=None`, and assert the prepared statevector equals the
   Ran-2020 reconstruction `reconstruct_from_mps(classical_to_mps(psi0))`
   to 1e-12. Guards against any future silent fall-back to `initialize`.
2. New test `test_bond_dim_truncation`: same config with
   `bond_dim=1`, assert the reconstructed ψ differs from the full-rank ψ
   (i.e. truncation is actually happening) and both paths complete
   without error.
3. All existing acceptance tests (11.1, 11.2, 11.3, 11.5, 11.6) still
   PASS unchanged — MPS prep with full rank should be a no-op
   numerically vs dense `initialize`.
4. `q8020-sweep` on `[paper_cole_hopf_circuit_q5_shots150k_mps]` runs
   to completion; the postproc PNG shows the three bond-dim curves
   converging to the full-rank / classical curve.

**Spec edits.** Update F10-IMPLEMENTATION-SPEC.md:

- §2 item 2: confirm MPS-prep is wired (drop any hedging language).
- §4 "State prep" row: add "`--bond-dim` passes through to
  `classical_to_mps` truncation."
- §10 CLI surface: add `--bond-dim INT` to the list.
- §14 validation checklist: add "Prepared ψ matches
  `reconstruct_from_mps(classical_to_mps(ψ₀))` to 1e-12."

## P-H — Peaked-φ shots readout (low-ν regime)

**Problem.** At paper-target ν=1e-4, φ(x) = exp(−∫u/2ν) concentrates
almost all its probability mass on ~1 grid bin (the location of the
minimum of ∫u). The shots path at
[burgers_cole_hopf_circuit.py:504-509](../burgers_cole_hopf_circuit.py)
reconstructs φ(x_i) = √(counts[x_i]/N_kept) with √-of-counts noise
scaling. At N_bins − 1 "tail" bins with expected p_i ≪ 1/shots, the
relative noise explodes. Concretely: test_11_5 had to deviate from the
spec's ν=1e-2 → ν=0.1 config to pass. Production sweeps at ν=1e-4
will give qualitatively wrong u in the tail region even at 150k shots.

This is a known failure mode of direct amplitude sampling on peaked
states; not a bug, an algorithmic gap.

**Three candidate mitigations**, ordered by implementation effort.

### P-H.1 — Hadamard-test per bin (recommended, modest effort)

For each basis state `|x_i⟩`, run a Hadamard test against the
post-evolution state to directly extract `Re⟨x_i|U_prep · U_evo|0⟩`.

- Circuit per bin: 1 ancilla + H + controlled-`U_prep·U_evo` +
  controlled-`|x_i⟩⟨0|` (reflection) + H + measure ancilla.
- Shots per bin: far fewer than peak-bin-dominated direct sampling,
  because each bin is now measured as a ±1 expectation with variance
  1/shots independent of bin population. For q=5 that's 32 circuits;
  1000 shots each = 32k total shots, vs the 150k shots currently
  wasted on the peak.
- Output: real-valued amplitude `φ̂(x_i)` directly. No sqrt + sign
  work because φ > 0; pass through.

**Cost.** 2q extra gates per circuit (controlled reflection on q
qubits = multi-controlled-Z). At q=5 well within Aer.

**Acceptance.** Extend `test_11_5_shots_accuracy` to cover ν=1e-2
(spec's original config) with the Hadamard-test path and assert
`‖u_circuit − u_dense‖₂ / ‖u_dense‖₂ < 0.05`. Remove the nu=0.1
deviation from the existing test docstring.

**Spec edits.** Add §7.A describing the Hadamard-test readout as an
alternative to direct √-counts sampling; add `--readout {direct,
hadamard_per_bin}` CLI flag with `direct` as default for high-ν and
`hadamard_per_bin` recommended for ν < 1e-3.

### P-H.2 — Adaptive two-pass (lighter implementation)

Round 1: spend 10% of shots on direct sampling; identify the peak bin
`k*`. Round 2: apply a classical-controlled rotation that swaps `|k*⟩`
out of the computational basis (e.g. CNOT-ladder into an auxiliary
register that is post-selected OUT), and sample the remaining 90% of
shots on the tail-only distribution. Combine.

**Cons.** Adaptive logic inside a sweep is awkward; two-pass increases
latency and branches the codepath. Not recommended unless P-H.1 proves
too expensive at larger q.

### P-H.3 — Shadow-tomography / classical shadows (research)

Out of scope for a patch. Flag as future work.

**Recommend P-H.1.** The Hadamard-test-per-bin circuit already exists
conceptually in the F9 sign-recovery machinery (`--sign-recovery
hadamard_test` flag in [burgers_solver.py:126-130](../burgers_solver.py)),
which can be adapted: sign recovery returns ±1 amplitude; here we
return amplitude only since φ > 0. Reuse the ancilla wiring.

## Ordering

```
  P-G  ──┐
         ├──▶  run paper_cole_hopf_circuit_q5_shots150k_mps  ──▶  F10 closed
  P-H  ──┘
```

Independent parcels; dispatch in parallel. Both converge on the same
paper-comparison artifact as the F10 acceptance close-out.

## Out of scope

Bucket 3 future work (encoding change, Carleman, DST-Dirichlet, QSVT,
QROM, hardware, F11, F2 revival, true Pauli-Trotter) is documented in
`FUTURE-WORK.md` (companion file); not touched here.
