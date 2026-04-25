# F2 Implementation Spec: TEBD for Burgers on Amplitude-Encoded MPS

Date: 2026-04-22
Scope: Pure-quantum time evolution of 1D Burgers on q <= 6 qubits.
Status: Design spec (no code yet). Supersedes the one-paragraph F2 stub
in IMPLEMENTATION-PLAN.md.

## 1. What the plan calls for

From IMPLEMENTATION-PLAN.md item F2:

> Time-evolving block decimation (TEBD, Ref. 30). An alternative to
> Trotterization that works directly in MPS form by applying two-site
> gates and re-compressing. Would avoid the Pauli decomposition entirely.
> Not implemented.

Ref. 30 in Meena et al. AIAA 2026 is:

> Vidal, G. (2003), "Efficient classical simulation of slightly entangled
> quantum computations", Phys. Rev. Lett. 91, 147902.

(The paper cites Vidal 2003; the 2004 PRL "Efficient simulation of
one-dimensional quantum many-body systems" is the companion. Both
describe TEBD; use the 2003 paper as the primary source for the
canonical-form updates and the 2004 paper for the open-system / MPS
formulation.)

Meena defers TEBD to future work (Sec. VI) and gives no implementation
detail. The spec below fills that gap.

## 2. Refined goal (from user)

The pure-quantum version of F2 — not a classical TEBD reference — has
three pieces:

 1. The initial condition u_0(x) is encoded as an MPS and state-prepared
    into a quantum circuit via the existing Ran-2020 synthesis in
    burgers_mps.py (mps_to_circuit).

 2. Time evolution is executed entirely on the circuit, as a sequence
    of 2-qubit gate layers — no per-step classical RHS evaluation, no
    per-step Pauli refit, no classical norm rescaling. This is the
    structural distinction from burgers_trotter.py /
    burgers_nonlinear.py, which are classical-driven.

 3. Phase (sign) discernment is performed via a quantum protocol so the
    readout recovers signed amplitudes, not just |psi_k|^2. The F9
    Hadamard-test scheme is the baseline (IMPLEMENTATION-PLAN.md lines
    258-271).

Item 2 is the hard part. The discussion below is mostly about item 2.

## 3. Why vanilla TEBD does not map directly

Vidal's TEBD targets an MPS of a 1D lattice quantum system. The
physical indices are the sites of the lattice; "adjacent sites" are
physically neighboring particles; and the Hamiltonian is a sum of
nearest-neighbor terms:

    H = sum_i h_{i, i+1}

Second-order Trotter produces a brick-wall of 2-site gates:

    U(dt) ~ [prod_{i odd}  exp(-i h_{i,i+1} dt/2)]
          * [prod_{i even} exp(-i h_{i,i+1} dt)]
          * [prod_{i odd}  exp(-i h_{i,i+1} dt/2)]

Each gate is a fixed 4x4 unitary.

Meena's encoding is fundamentally different. N = 2^q grid points are
amplitude-encoded into q qubits: u(x_k) -> amplitude psi_k of basis
state |k>, with k written as a q-bit binary string. MPS sites here are
the bits of the grid INDEX, not physical lattice sites. Two
consequences:

 a) Adjacent qubits do NOT correspond to adjacent grid points. Qubit 0
    is the MSB of the grid index; flipping it moves by N/2 grid points.

 b) The Burgers spatial operators (Laplacian, gradient) are NOT local in
    qubit space. The shift operators S+/S- are MPOs with bond dim 3 and
    long-range entanglement across the qubit register.

So "TEBD for Burgers in amplitude encoding" is not brick-wall-of-fixed-
gates-on-a-lattice. It is: express the evolution operator as an MPO and
apply it to the MPS with bond-dim truncation at each layer. The
nearest-neighbor structure lives in the MPS/MPO virtual-bond graph, not
in physical space.

This is the MPO-W-II / time-evolving-MPO approach that generalizes
Vidal's TEBD to long-range Hamiltonians. References:

 - Zaletel, Mong, Karrasch, Moore, Pollmann (2015), "Time-evolving a
   matrix product state with long-ranged interactions", Phys. Rev. B
   91, 165112. [MPO-W-I, MPO-W-II]
 - Paeckel, Kohler, Swoboda, Manmana, Schollwock, Hubig (2019), "Time-
   evolution methods for matrix-product states", Ann. Phys. 411, 167998.
   [comprehensive review of TEBD / TDVP / W-II / Krylov-MPS]

These are the academic anchors for F2 beyond the bare Vidal citation in
Meena.

## 4. The nonlinearity problem (unchanged from F10 notes)

For viscous Burgers, u_t + u u_x = nu u_xx, the evolution operator
depends on the current state via the u u_x term. A fixed set of 2-site
gates cannot represent a state-dependent Hamiltonian. Two routes
around this:

 A. Linearize via Cole-Hopf (F10 territory). phi_t = nu phi_xx is
    linear; TEBD/W-II applies cleanly; readout needs
    u = -2 nu phi_x / phi. Covered in F10; not duplicated here.

 B. Freeze the Hamiltonian within a step. Build H(u_n) once per step
    from the current classical-equivalent state, construct its MPO-W-II
    representation, apply it as a quantum circuit layer. This is still
    classically-driven per-step (fails the "classical-free mid-flight"
    criterion), but it sidesteps Cole-Hopf and keeps the existing
    sine/multimode test cases unchanged.

F2 as stated in the plan is route B (direct on Burgers). F10 is route A.
Both are valuable; this spec is scoped to route B because that is what
F2 means in the current doc. Route-A work belongs under F10.

Acknowledge up front: route B does not demonstrate quantum advantage
(the H(u_n) build requires classical work). Its value is replacing
exponential classical work with polynomial classical work:

    path              | per-step classical cost | scaling in q
    ------------------+-------------------------+-------------------
    quantum_circuit   | solve 4^q Pauli coeffs  | O(4^q) (exponential)
                      | per step                |
    tebd (route B)    | build H_n MPO, W-II     | O(q * chi^3)
                      | compile, gate extract   | (polynomial)
    F10 (route A)     | fixed gates at setup    | O(1) in time loop
                      | only                    |

Exponential-to-polynomial on the classical side is the F2 contribution.
True classical-free evolution is F10.

## 5. Pure-quantum pipeline

Per step n (state |psi_n>, time t_n = n dt):

 1. Build linearized-within-step Hamiltonian H_n as an MPO.
      H_n = nu * Laplacian_MPO  -  diag(u_n) * Gradient_MPO  +  diag(g_n)
    where u_n is the classical-equivalent vector (needed only for the
    diag(u_n) term). The MPOs for Laplacian and Gradient are the bond-
    dim-3 MPOs from Meena (S+-, S-). The diag(u_n) MPO has bond dim
    bounded by the MPS bond dim of u_n.

 2. Convert exp(-i H_n dt) to a circuit layer via MPO-W-II:
      W-II gives a single layer of bond-dim-2 local gates that
      approximates exp(-i H_n dt) to second order in dt. The layer
      consists of nearest-neighbor 2-qubit gates in the MPS-site
      ordering (qubits ordered MSB -> LSB). Depth is O(q).

 3. Append the W-II layer to the running circuit. No measurement, no
    state reset, no reconstruction. The MPS stays implicit in the
    circuit amplitudes.

 4. (Periodically or at snapshot steps only) perform phase-discerned
    readout. See Section 8.

 5. For the next step's H_{n+1}, maintain a classical mirror u_n
    propagated by classical Euler in parallel. The mirror is load-
    bearing: it is what keeps per-step classical cost polynomial rather
    than exponential. Replacing the mirror with quantum tomography of
    the register re-introduces exp(q) shots per step and destroys F2's
    contribution. Reading u_n from the mirror also matches how the
    existing classical baseline is already computed for every case, so
    no new classical infrastructure is needed.

Circuit depth per step: O(q) 2-qubit gates. Over M steps: O(M q) gates.
Compare to burgers_trotter.py: O(4^q) Pauli rotations per step. TEBD
wins at q >= 5.

Qubit count: q data qubits + O(1) ancilla for phase discernment.

## 6. Quimb: what it is useful for, and what it is not

quimb (Gray 2018, Ref. 26 in Meena) is the tensor-network library
already used in the project. Assessment for F2:

 Useful for:

 - Classical verification oracle. quimb.tensor.TEBD and the MPO-apply
   primitives let us run a classical TEBD/W-II simulation in parallel
   with the quantum path, for validation at small q. This is the
   cheapest way to test W-II correctness before wiring it into a
   circuit.

 - MPO construction. MatrixProductOperator.from_dense is already the
   path Meena uses for Laplacian / Gradient MPOs (paper Sec. III.B.2,
   Fig. 6). Same machinery builds H_n; we get it for free.

 - W-II approximation construction. quimb exposes
   MatrixProductOperator.apply_op_MPO-style primitives and a
   make_bondary_compressed_MPO helper that implement the Zaletel
   W-II construction. This avoids hand-coding the W-II tensor reshuffle.
   (Exact API: quimb.tensor.tensor_1d_tebd has time-evolution
   utilities; quimb.tensor.circuit has the classical MPO->gate helpers
   we would wrap.)

 - Fidelity and overlap for validation. quimb computes <psi_quantum|
   psi_classical_tebd> directly once both are in MPS form.

 Not useful for:

 - Generating the actual Qiskit circuit. quimb has no native export to
   Qiskit. The W-II 2-qubit gates come out as numpy arrays that we then
   wrap in qiskit.circuit.library.UnitaryGate — same pattern as
   burgers_mps.py mps_to_circuit already uses for Ran 2020. Do not try
   to use quimb.tensor.circuit.Circuit as the executor; we want
   qiskit-aer for shots/noise consistency with the rest of the project.

 - Running on IBM hardware. quimb is classical. The quimb path is a
   verification reference; production runs must go through Qiskit.

Conclusion: quimb is load-bearing for the build-and-verify stage of F2
(constructing H_n MPOs, constructing W-II gates, running a classical
TEBD reference) but is NOT the execution engine. Same role it plays in
the existing MPS state-prep pipeline.

## 7. Mathematical core: W-II construction

The key move in turning H = sum_n h_n (where h_n is an MPO tensor at
bond n, not a two-site term) into a single shallow circuit layer. Given
H as an MPO with virtual bond dim D, write each local MPO tensor
W^{s s'} with virtual indices a, b (a in [0, D)):

    W^{s s'}_{a b}

Zaletel et al. show that for small dt, exp(-i H dt) is approximated by
an MPO whose local tensors are

    V^{s s'}_{a b} = delta_{s s'} delta_{a b}
                   + (-i dt) h^{s s'}_{a b}   [for specific (a,b) cells]
                   + O(dt^2)

with a specific sparse structure (W-II form) that gives second-order
accuracy in dt per step. The resulting evolution MPO has bond dim D.
Applying it to an MPS of bond dim chi produces an MPS of bond dim
D*chi, which is then SVD-truncated back to chi.

In quantum-circuit form: the W-II MPO factorizes into a single layer of
2-site unitaries acting on adjacent MPS sites. Each 2-site unitary is a
4x4 matrix derived from the W tensor. Compile to qiskit via UnitaryGate.

Deliverables for this section:

 a) Derivation of the W-II tensors for H_n (Laplacian, Gradient,
    diag(u_n), diag(g)). Write these out explicitly in the spec
    appendix once we prototype — paper references give the recipe but
    not the numerical tensors.

 b) Unit test: compare exp(-i H dt) W-II MPO against
    scipy.linalg.expm(-i H_dense dt) for q=3,4 at several dt. Expect
    agreement O(dt^3) in Frobenius norm.

 c) Unitarity check: each 2-site gate produced must be unitary to 1e-12.
    The W-II form is approximately unitary; exact unitarization via
    polar decomposition may be needed and is standard practice.

## 8. Phase discernment

Amplitude encoding readout by shots gives sqrt(counts/total), which
drops sign. F9 already catalogs four options (IMPLEMENTATION-PLAN.md
lines 249-291). For F2, the pure-quantum choice is Option 2: Hadamard
test against the previous-step state. Summary:

 - Maintain a reference |phi_n> = (circuit that prepares signed state at
   step n). Initially |phi_0> = MPS-circuit(u_0), with signs known.

 - At snapshot step, prepare |phi_n> on register B and the current
   |psi_{n+1}> on register A, with an ancilla in |+>. Controlled-SWAP
   between A and B, then H on ancilla, measure ancilla and register A.

 - For each basis state k of register A:
     P(anc=0, k) - P(anc=1, k) = Re(<phi_n|k><k|psi_{n+1}>)
   which carries the sign of the overlap Re(psi_{n+1,k} * conj(phi_{n,k})).
   Combined with the known sign of phi_{n,k}, this gives sign(psi_{n+1,k}).

 - Cost: 1 ancilla, 1 controlled-SWAP layer, and a doubling of data
   qubits for the duration of the snapshot measurement (register B
   exists only at snapshot steps).

Bootstrapping: |phi_0> signs are known from the IC. At step n+1, update
the reference to |phi_{n+1}> = (quantum circuit that produced the
current state, with signs now known from this snapshot's result). This
keeps the reference one step behind the quantum evolution and requires
no separate classical sign oracle.

Alternative for budget-constrained runs: Option 3 (classical sign
oracle from the classical mirror), which the project already computes
anyway. This is a fallback for debugging — if Option 2 gives nonsense,
compare to Option 3 to localize the bug.

## 9. Phased deliverables

Phase A — Classical W-II reference (quimb-backed, no circuit)

 A.1  burgers_tebd_classical.py: implement H_n construction as a quimb
      MPO; implement the W-II second-order time-step as an MPO apply
      with bond-dim truncation. Output: u_n at each step. Use
      MatrixProductOperator.from_dense on the dense H_n and quimb's
      built-in apply_MPO with compression.

 A.2  Validation: compare against burgers_trotter.quantum_exact_step
      (scipy expm) and burgers_classical.solve_burgers for
      q in {3, 4, 5}, sine IC, nu=1e-4. Expect agreement to O(dt^2)
      time-step error plus O(threshold) compression error.

 A.3  Bond-dim sweep: vary chi_max in {4, 8, 16, 32, full}; document
      accuracy-vs-chi_max tradeoff. Reuse plot_mps_bond_sweep.py scaffolding.

Phase B — Circuit W-II (no phase discernment yet)

 B.1  burgers_tebd.py: given H_n MPO from A.1, extract the W-II local
      tensors, unitarize each, wrap as Qiskit UnitaryGate, assemble into
      a per-step circuit layer. Connect to the existing MPS state-prep
      circuit (burgers_mps.mps_to_circuit) as the initial layer.

 B.2  Wire into burgers_trotter.run_simulation as method=tebd, parallel
      to quantum_circuit, quantum_exact, mps, shift. Reuse existing
      shots/noise/backend plumbing.

 B.3  Validation: statevector (shots=0) agreement with Phase A to
      machine precision. Resource metrics: 2-qubit gate count, depth,
      transpiled CX count vs quantum_circuit at same q.

Phase C — Phase discernment (Option 2 Hadamard test)

 C.1  Extend burgers_tebd.py with a snapshot-measurement subroutine
      that takes (current circuit, reference state preparation, ancilla)
      and returns signed amplitudes.

 C.2  Validation: shots != 0 runs should produce signed profiles that
      track the classical reference across the shock transition at
      x > 0.5, which is the failure mode the F9 doc calls out.

 C.3  Comparison: same q, same shots, method=quantum_circuit vs
      method=tebd, vs method=tebd with sign recovery. Quantify the
      sign-recovery accuracy vs Option 3 (classical oracle) baseline.

Phase D — Integration and sweep

 D.1  input/burgers_quantum.toml: add tebd_q3..q6 cases (statevector),
      tebd_shots_q5 (shots study), tebd_noise_q5 (T1/T2).

 D.2  Plot scripts: extend plot_paper_aligned.py and plot_shots_study.py
      to include method=tebd rows. New plot_tebd_depth.py for depth /
      CX-count vs q.

 D.3  q8020 metadata: add tebd-specific fields to the analysis fragment
      (chi_max, W-II order, sign_recovery_method, ancilla_count).

## 9a. CLI surface

Single control: `--method`. A three-axis split (state-prep x evolution
x sign-recovery) was considered and rejected — the axes are
syntactically orthogonal but physically coupled (see Section 9b), and
most of the 32 combinations are invalid, degenerate, or uninteresting.
Safer to enumerate the curated set:

    --method {shift, quantum_exact, quantum_circuit, mps,
              mps_pauli, tebd, tebd_signed}

Mapping to the underlying (state-prep, evolution, default sign-recovery)
triple:

    shift           -> (none,      shift_classical, none)
    quantum_exact   -> (amplitude, exact_expm,      none)
    quantum_circuit -> (amplitude, pauli_trotter,   none)   [Meena main]
    mps             -> (mps_ran,   exact_expm,      none)
    mps_pauli       -> (mps_ran,   pauli_trotter,   none)   [new: isolates
                                                             state-prep
                                                             contribution
                                                             to Meena path]
    tebd            -> (mps_ran,   tebd_wii,        none)   [F2 core]
    tebd_signed     -> (mps_ran,   tebd_wii,        hadamard_test)
                                                    [pure-quantum headline]

The first four are existing; the last three are added by F2. No TOML
schema change — existing cases keep working.

`--sign-recovery` remains as an optional override for the F9-style
ablation study on quantum_circuit and mps_pauli (where sign recovery is
a legitimate independent axis). For tebd_signed the sign-recovery is
definitional; overriding it produces a different method, not a variant.

Arg-parse validation rejects incompatible overrides at startup rather
than silently producing a meaningless result. Rejected cases:

 - hadamard_test requested on any amplitude-prep method (reference-
   state prep mismatch; see Section 9b coupling 3).
 - hadamard_test requested on shift (no quantum register).
 - dual_rail requested with mps_ran at q where sparse-zero regions
   trigger Ran-2020 canonical-form degeneracy (warn, don't reject).

Rationale for curating: the paper build-up wants ~6 comparison points,
not ~30. Each curated method answers a specific ablation question:

    quantum_circuit vs mps_pauli  -> what does MPS state prep cost on
                                     the existing Pauli path?
    mps_pauli vs tebd             -> Pauli-Trotter vs W-II at fixed
                                     state-prep?
    tebd vs tebd_signed           -> cost of adding sign recovery to
                                     F2?
    quantum_circuit vs tebd_signed -> full F2 contribution vs Meena
                                     baseline.
    quantum_exact vs mps          -> pure state-prep error at exact
                                     evolution.
    shift vs quantum_exact        -> quantum-register round-trip error
                                     only.

That's the TOML sweep target for Phase D.1.

## 9b. Why the axes are not orthogonal

The CLI is a single `--method` knob because the underlying (state-prep,
evolution, sign-recovery) triple has real physical couplings. Five of
them, documented here so the curated whitelist in 9a can be revised
intelligently if new methods are added later.

 1. State-prep fidelity is the floor under evolution fidelity.
    mps_ran injects an O(epsilon_chi) compression error at t=0 which
    every subsequent evolution step carries. amplitude has zero
    compression error but produces transpiled depth that explodes with
    q. So pauli_trotter and tebd_wii have different useful regimes
    depending on which prep feeds them. At q=6 with mps_ran + chi=8,
    trotter error and state-prep error are comparable and an ablation
    study is meaningful; with amplitude at q=6 on hardware, state-prep
    depth dominates the error budget.

 2. "Pure quantum" collapses the axes. Qiskit's Statevector.initialize
    is classically computed under the hood (Schmidt-decomposition
    inside the transpiler) for q <= ~15. If the headline paper claim
    is a pure-quantum pipeline, mps_ran is the only qualifying
    state-prep option. The pure-quantum triple is therefore locked:
    mps_ran + tebd_wii + hadamard_test. That is the tebd_signed
    method.

 3. Hadamard-test sign recovery needs matching prep. The reference
    state |phi_n> must be prepared by the same state-prep method as
    |psi_n>. Mixing (|psi_n> via mps_ran, |phi_n> via amplitude)
    leaves an unresolved compression-error phase and the Hadamard
    overlap measures a meaningless sign. Hard constraint.

 4. Dual-rail sign recovery pairs naturally with amplitude. u+ and
    u- have many exact zeros where the original u is nonnegative or
    nonpositive respectively. Ran 2020 on states with clusters of
    zeros has canonical-form degeneracy (zero singular values have
    non-unique singular vectors). amplitude handles zeros trivially.
    Dual_rail is near-free with amplitude and needs a regularization
    convention with mps_ran.

 5. W-II classical-verification assumes bounded-chi MPS. Phase A
    (quimb reference) assumes the MPS bond dim stays bounded across
    the step. At q >= 8 with an amplitude-init'd shock profile, the
    "cheap classical TEBD" argument for the reference weakens. Not
    in scope for q <= 6 but flagged for later.

The curated method set in 9a respects all five couplings by
construction. Adding a new method to that list means re-checking it
against this list.

## 10. Risks and open questions

 1. diag(u_n) bond dim is the real ceiling. Lead risk. The Laplacian
    and Gradient MPOs have the advertised bond dim 3 only in their
    specific ladder form; diag(u_n) has bond dim equal to the MPS bond
    dim of u_n, which grows toward the shock. The composite H_n bond
    dim is dominated by diag(u_n), not by the advertised-3 Laplacian /
    Gradient. At q <= 6 with chi <= 8 this is fine; at q >= 8 it is
    the first thing expected to break. Validation plan: measure H_n
    bond dim vs time and vs q in Phase A, publish the curve.

 2. W-II unitarity. The raw W-II tensors are not unitary; polar
    decomposition is the standard fix but introduces an O(dt^2) error
    of its own. Need to confirm this does not dominate the Trotter
    error budget already characterized in Phase 3 of the plan.

 3. State-dependent H_n defeats classical-free evolution. This is the
    known limitation from Section 4. Document it as the F2 ceiling;
    point toward F10 (Cole-Hopf) for the genuine quantum-advantage
    claim.

 4. Reference-state drift in Option 2 phase discernment. If the
    reference |phi_n> is itself prepared from a lossy snapshot, errors
    compound step-over-step. Mitigation: re-anchor the reference to
    |phi_0> periodically and re-derive all intermediate signs; or
    fall back to Option 3 on divergence.

 5. quimb's TEBD uses left/right-canonical MPS forms; the Ran 2020
    circuit produces one canonical form. Must verify the canonicality
    convention matches between quimb (classical) and Qiskit (circuit)
    paths so the W-II tensors align.

 6. Dirichlet boundary conditions. The shift MPOs already support
    Dirichlet (Phase 1.2 of the plan); need to re-verify the W-II
    construction preserves the boundary rows correctly. Unit test at
    q=3 with u_0 = sin(pi x) (vanishing at 0 and 1) is the sanity check.

## 10a. Acceptance criteria (coding-team handoff)

Per-phase go/no-go for the coding team. Each phase must pass its
criteria before the next one starts; regressions on earlier phases block
merge.

Phase A (classical W-II reference):
 - A.1 module imports; builds H_n MPO from quimb for q=3,4,5.
 - A.2 single-step agreement with scipy expm to < 1e-10 at dt=1e-4,
   for q=3,4,5, sine IC.
 - A.2 full-trajectory max L2 error vs burgers_classical.solve_burgers
   below 2x the existing quantum_exact error at the same parameters
   (i.e. W-II adds no more than its own dt^2 contribution).
 - A.3 bond-dim sweep plot produced; chi_max at which error plateaus
   is reported per q.
 - H_n composite bond dim measured per step and plotted vs time per q
   (validation of risk 1).

Phase B (circuit W-II, no sign recovery):
 - B.1 W-II 2-qubit gates unitary to < 1e-12 after polar step.
 - B.2 `--method tebd` runs end-to-end through burgers_solver.py for
   q=3,4,5,6 at shots=0 (statevector).
 - B.2 statevector output matches Phase A classical reference to
   < 1e-12 at q=3,4,5 (no approximation in this comparison — pure
   representation check).
 - B.3 transpiled CX count and depth reported for q=3..6; compared to
   `quantum_circuit` at the same q.

Phase C (sign recovery, Hadamard test):
 - C.1 `--method tebd_signed` runs end-to-end, shots > 0, on aer.
 - C.2 signed profile tracks classical reference across x > 0.5
   shock region with L2 error <= 2x the x < 0.5 error at the same
   shot budget. This is the explicit F9 failure mode the method is
   fixing.
 - C.3 tebd_signed vs tebd-with-sign-recovery=classical_oracle
   agreement within statistical error of the shot budget.

Phase D (sweep integration):
 - D.1 TOML cases added; q8020-sweep completes all new cases without
   error.
 - D.2 plots include `tebd` and `tebd_signed` method rows.
 - D.3 metadata fragment records method, state-prep, evolution,
   sign-recovery fields per case.

Cross-phase code-hygiene:
 - No modifications to burgers_classical.py, burgers_mps.py, or
   burgers_nonlinear.py beyond what's strictly required. F2 is
   additive. If a shared change is needed, flag it in PR description.
 - New module burgers_tebd.py is the only required addition beyond
   burgers_tebd_classical.py (Phase A).
 - All Best-Practices rules apply (PEP 8, venv, grouped imports, no
   trailing whitespace, <= 88 char lines, strong typing).

## 10b. Successor tasks (scope boundary)

F2 delivers `tebd` and `tebd_signed` methods as described above. It does
NOT deliver:

 - F10 (Cole-Hopf + TEBD). Removes the classical mirror entirely; per-
   step classical cost becomes O(1). Is the quantum-advantage story.
   Independent module burgers_cole_hopf.py (see plan F10.1). Depends
   on F2's W-II machinery for the evolution circuit.

 - F11 (Burgulence statistical study). Gaussian-random-field ICs,
   ensemble sweep, E(k) scaling study. Depends on F10, not on F2
   directly. Note for scheduling: F10 at the Burgulence nu=1e-5 regime
   hits a Cole-Hopf dynamic-range problem (phi = exp(integral / nu)
   overflows float64 at realistic random-field IC amplitudes); that
   needs its own design pass before F11 can run at the inertial-range
   Re. Recommended interim target is nu ~ 1e-3 for decaying Burgulence
   to validate the pipeline end-to-end.

 - Forced Burgulence (F11.5). Stochastic forcing turns the Cole-Hopf
   target equation into a heat equation with a time-dependent
   multiplicative potential; breaks F10's fixed-gates-once promise.
   Separate project.

The above are out of scope for the F2 pull request. Do not implement
them under the F2 ticket; they are tracked as successor tasks in
IMPLEMENTATION-PLAN.md.

## 11. What we are explicitly NOT doing under F2

 - Cole-Hopf linearization (that is F10).
 - Variational fast-forwarding (that is F4).
 - Krylov-MPS (that is F5 / W-II-adjacent but different).
 - TDVP. It is a principled alternative to W-II for long-range H and
   may be better for shocks, but starting with W-II matches the
   plan's "TEBD" wording and the quimb-native path. Revisit after
   Phase B if bond-dim growth is the bottleneck.
 - Any extension beyond 1D Burgers.

## 12. References

Primary:

 - Meena, Jones, Zhang, Gao (2026), "A Tensor Network-based Quantum
   Algorithm for the Nonlinear 1D Burgers' Equation", AIAA 2026.
   [the current project's anchor paper; Ref 30 there is Vidal]
 - Vidal, G. (2003), "Efficient classical simulation of slightly
   entangled quantum computations", PRL 91, 147902. [TEBD, original]
 - Vidal, G. (2004), "Efficient simulation of one-dimensional quantum
   many-body systems", PRL 93, 040502. [TEBD, MPS form]

W-II and long-range TEBD:

 - Zaletel, Mong, Karrasch, Moore, Pollmann (2015), "Time-evolving a
   matrix product state with long-ranged interactions", PRB 91, 165112.
 - Paeckel et al. (2019), "Time-evolution methods for matrix-product
   states", Ann. Phys. 411, 167998. [review]

State preparation and tooling:

 - Ran, S.-J. (2020), "Encoding of matrix product states into quantum
   circuits of one- and two-qubit gates", PRA 101, 032310.
   [state prep, already implemented in burgers_mps.py]
 - Gray, J. (2018), "quimb: A python package for quantum information
   and many-body calculations", JOSS 3(29), 819.
   [already a project dependency]

Phase discernment:

 - F9 section of IMPLEMENTATION-PLAN.md, options 2-4.
 - No new reference beyond the standard Hadamard test; implementation
   cross-references burgers_sign_recovery.py for the Option 3 baseline.
