# Code Review: F9 Sign Recovery

Date: 2026-04-19
Scope: F9 task from IMPLEMENTATION-PLAN.md -- four sign-recovery
options for the shots>0 quantum_circuit / mps paths.

Files reviewed:
  burgers_sign_recovery.py   (new module, all four options)
  burgers_trotter.py         (integration: quantum_circuit_step,
                              mps_step, dual_rail_quantum_step,
                              run_simulation)
  burgers_solver.py          (--sign-recovery CLI)
  input/burgers_quantum.toml (sign_recovery_sweep_q5 group)


## Coverage vs plan

All four options are present and wired end-to-end.

  Option 1 (none)              -- quantum_circuit_step:186-196
  Option 2 (hadamard_test)     -- burgers_sign_recovery.py:64-170
  Option 3 (classical_oracle)  -- burgers_sign_recovery.py:36-57
  Option 4 (dual_rail)         -- burgers_sign_recovery.py:177-206
                                  + dual_rail_quantum_step in trotter

CLI and sweep groups cover all four as a parameter array plus
targeted singletons.


## High-priority findings

1. dual_rail_quantum_step builds a per-rail Hamiltonian.

   In dual_rail_quantum_step the inner _evolve_rail calls
   quantum_circuit_step(u_rail, ...). quantum_circuit_step line 136
   builds H from whatever u it receives, so u+ evolves under H(u+)
   and u- under H(u-).

   F9 Option 4 describes splitting a single evolution:
     U(u)*u+ - U(u)*u- = U(u)*u.
   With per-rail H the implementation computes
     U(u+)*u+ - U(u-)*u-
   which is a different operator and does not converge to the
   Burgers solution as dt -> 0.

   Fix: build H(u) once in dual_rail_quantum_step, pass it into
   each rail evolution (requires a small refactor so that
   quantum_circuit_step can accept a pre-built Hamiltonian), or
   document this as an intentional alternative scheme with its
   own error analysis.

2. <DONE> Hadamard test recovers sign(Re psi_k) * |psi_k|, not Re(psi_k).

   Derivation: for complex psi = e^{-iHdt}phi and real phi,
     P(0,k) - P(1,k) = Re(phi_k* psi_k)
     2*(P(0,k) + P(1,k)) - phi_k^2 = |psi_k|^2

   The code uses sign(p0-p1) * sqrt(|psi_k|^2), which equals
   Re(psi_k) only when psi_k is real. For a single Trotter step
   at small dt the state is approximately real and this is fine,
   but the systematic error grows with dt and trotter_reps.

   The rest of the code takes .real of the evolved statevector
   elsewhere (trotter:173), so the comparison target itself
   discards Im(psi). That is consistent but should be noted in
   the docstring so a future reader does not chase phantom
   sign errors.

3. Dual-rail metrics are dropped.

   trotter:352: metrics = {"sign_recovery": "dual_rail"}. The
   two inner quantum_circuit_step calls produce circuit depth,
   gate counts, and timings that are discarded. Downstream
   plots show dual_rail as "free". Merge the rails' metrics
   (sum times, combine gate_counts, take max depth) before
   returning.


## Medium

4. <DONE> mps_step only supports classical_oracle.

   trotter:289-291 applies signs only for classical_oracle.
   hadamard_test / dual_rail are silently ignored under the
   mps method. Either raise NotImplementedError for unsupported
   combinations, or gate the CLI choices by method.

5. <DONE> Classical-oracle near-zero sign flap.

   burgers_sign_recovery.py:56: np.sign(u_next) with the
   s==0 -> +1 tiebreaker is safe in exact arithmetic but flips
   under discretization noise near zero crossings (e.g. x ~ 0.5
   for the sine IC). A tolerance (|u_next| < eps -> keep
   sign(u_prev) or +1) reduces speckle where it matters.


## Low / style

6. <DONE> Circular-import workaround is unnecessary.

   burgers_sign_recovery.py:53 imports compute_rhs_shift at call
   time. burgers_nonlinear does not import burgers_sign_recovery,
   so the top-level import is safe and aligns with the "imports
   at top, grouped" rule.

7. <IGNORE> No unit or smoke coverage for Option 2 correctness.

   Given (2) and the bitstring indexing in
   extract_signs_from_hadamard_counts, a small test that builds
   a known signed state, runs the Hadamard-test circuit at high
   shots, and checks recovered signs against ground truth would
   catch regressions cheaply. No such test exists today.

8. <IGNORE> Register-width assumption in the counts parser.

   burgers_sign_recovery.py:149-152 assumes bits[0] is the
   ancilla and bits[1:] is the data register. True for a fresh
   QuantumCircuit(q+1) + measure_all, but brittle if transpilation
   later adds ancillae or registers. Mapping via named
   ClassicalRegister instances is safer.


## Summary

Feature-complete against F9's scope. The dual-rail implementation
diverges from the F9 description in a way that materially changes
what is being solved (1) and hides its cost (3); these are the two
items to fix before trusting the comparison plots. The Hadamard-test
math is right but leaks a Trotter-scale systematic into the "signs"
result (2) -- worth documenting at minimum.
