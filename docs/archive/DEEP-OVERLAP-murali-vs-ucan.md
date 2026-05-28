# Deep Overlap Analysis: murali_burgers vs UCAN-1DBurgers

Date: 2026-04-10
Paper: Meena et al., "MPS/MPO Methods for 1D Burgers Equation," AIAA 2026

## Critical Context

UCAN-1DBurgers IS the paper's reference implementation. "Murali" =
Muralikrishnan Gopalakrishnan Meena (ORNL). The two codebases are not
independent implementations of different papers -- they implement
different stages of the same research pipeline described in the paper.

## Pipeline Mapping

The paper describes a three-stage pipeline:

```
Stage 1: Classical validation of TN operations
  (dense -> MPS, operator -> MPO, apply MPO to MPS, compare to dense)

Stage 2: Quantum circuit construction from TN
  (MPS -> circuit via Ran 2020, MPO -> circuit via LCU/Trotter)

Stage 3: Execution on quantum hardware
  (transpile, run on Aer/IBM backends, measure, reconstruct)
```

UCAN-1DBurgers implements Stage 1 only. Our murali_burgers implements
Stages 1-3. The paper itself only validates Stage 1 and proposes
Stages 2-3 as future work.


## Function-Level Overlap

### MPS Encoding (State Preparation)

Both codebases convert a classical vector u (length N=2^q) into MPS.

UCAN: `qtn.MatrixProductState.from_dense(u_normalized)` in
`laplacian2qtt.py`. Single call to quimb, which internally performs
the SVD sweep. Bond dimension controlled by a singular value
threshold (1e-6 for sine wave).

murali_burgers: `classical_to_mps(u, bond_dim, form)` in
`burgers_mps.py:15`. Manual iterated SVD via numpy. Supports both
left-canonical and right-canonical forms. Bond dimension controlled
by explicit max chi or threshold.

Assessment: Algorithmically identical (iterated SVD). Our code is
more explicit and gives direct control over the decomposition. UCAN
delegates to quimb. No code reuse opportunity here -- our version
is more appropriate for the quantum circuit pipeline because we need
the individual tensors to build gates.

### Operator Construction

This is where the codebases diverge fundamentally.

UCAN: Builds dense NxN matrices for gradient and Laplacian
operators, then `qtn.MatrixProductOperator.from_dense(A)` to get MPO
form. The MPO has bond dimension 3 (sufficient for these tridiagonal
operators). For the nonlinear term u*du/dx, it applies the gradient
MPO to the MPS, converts both back to dense via `.to_dense()`, does
element-wise multiplication in dense space, then re-encodes as MPS.
This is the QNPU concept from Lubasch et al.

murali_burgers: Builds the full evolution Hamiltonian
H = nu*Laplacian - diag(u)*Gradient + diag(g) as a dense matrix
in `burgers_nonlinear.py`, then decomposes into Pauli strings via
`SparsePauliOp.from_operator()`. This yields 4^q terms. The Pauli
Hamiltonian is then Trotterized via `PauliEvolutionGate` +
`SuzukiTrotter` synthesis.

Assessment: Completely different representations of the same
operators. UCAN uses MPO (bond dim 3, O(d^3) cost). We use Pauli
decomposition (4^q terms, exponential). The paper discusses both
approaches: MPO is the primary method, Pauli/LCU is in Appendix A
as an alternative for actual quantum circuits.

### Time-Stepping

Both use explicit Euler: u(t+dt) = u(t) + dt * RHS.

UCAN: Applies MPOs to MPS states via quimb tensor contraction. All
operations stay in TN form. The nonlinear term requires a
dense-space detour (QNPU bottleneck).

murali_burgers: Three paths:
  - "shift": Classical Euler with shift-operator FD (burgers_classical.py)
  - "quantum_exact": Pauli Hamiltonian + matrix exponential (no Trotter error)
  - "quantum_circuit": Pauli Hamiltonian + Trotterized circuit on Aer

The quantum paths rebuild the Hamiltonian each step (because u changes
the nonlinear term), decompose into Pauli strings, and evolve.

Assessment: Both use Euler. The operator application mechanism is
different (MPO contraction vs Pauli evolution). Our code goes further
by actually constructing and executing quantum circuits.

### Source Term

UCAN: g(x,t) = sin(2*pi*x) * cos(2*pi*t) -- time-varying, matches paper.

murali_burgers: `source_term_sine(x, t)` in burgers_classical.py:17
returns `sin(2*pi*x) * cos(2*pi*t)` -- also time-varying.
Time t = step * dt is correctly passed at each step (burgers_trotter.py:331).

Assessment: Both match the paper. No discrepancy.

### Boundary Conditions

UCAN: Dirichlet (u=0 at both ends). Uses one-sided differences at
boundaries in the dense operator matrices.

murali_burgers classical path: Mixed. `gradient_central()` uses
one-sided at boundaries. `laplacian_central()` uses periodic wrapping
(u[-1] at left, u[0] at right).

murali_burgers quantum path: Fully periodic. Shift operators S+/S-
wrap mod N by construction.

Paper: One-sided at boundaries (matches UCAN).

Assessment: Our classical baseline has an inconsistency (periodic
Laplacian, one-sided gradient). Our quantum path is fully periodic,
which differs from the paper's Dirichlet BCs. This is a known issue
flagged in the earlier review. For sine IC with u(0)=u(1)=0, the
difference is negligible, but it matters for validation rigor.

### MPS-to-Circuit Conversion

UCAN: Has a stub `mps2qc()` in `laplacian2qtt.py:383` -- literally
`pass`. Never implemented. No Qiskit imports in the Burgers pipeline.
There is unused Pauli infrastructure in `Q_dav_tensor_lib.py`.

murali_burgers: `mps_to_circuit(tensors)` in `burgers_mps.py`
implements the Ran 2020 algorithm: QR-based unitary completion of
each MPS tensor, decomposed into multi-controlled rotations. Fully
functional and tested.

Assessment: Our code implements what UCAN left as a stub. This is
the main value-add of our codebase.


## Should One Feed the Other?

### What UCAN could give us

1. MPO operator representation. The paper's primary method uses MPOs
   (bond dim 3) rather than Pauli decomposition (4^q terms). At q=8,
   Pauli decomposition produces ~65,000 terms. The MPO has fixed cost.
   However, converting an MPO to a quantum circuit is non-trivial --
   the paper explicitly calls this out as unsolved for non-unitary MPOs.

2. Nonlinear term handling. UCAN's QNPU approach (apply gradient MPO,
   convert to dense, element-wise multiply) is the same conceptual
   bottleneck we face: the nonlinear term breaks the linear algebra.
   Neither codebase has a clean quantum solution for this. The paper
   proposes variational QNPU (Lubasch et al.) as the quantum path
   but does not implement it.

3. Validation oracle. UCAN's quimb-based MPS/MPO operations are a
   known-correct reference for our from-scratch implementations.
   We could use UCAN outputs as test oracles for our MPS encoding.

### What we give UCAN

1. Actual quantum circuits. Our code fills UCAN's stub: MPS state
   prep via Ran 2020, Trotterized evolution, circuit execution on
   Aer and (potentially) IBM hardware.

2. Noise modeling. Our integration with q8020-cfd-qutil gives
   thermal relaxation noise (T1/T2), shot noise, and fake backend
   topologies.

3. Measurement and reconstruction. The shots>0 path with
   counts-to-amplitudes reconstruction (though the sign-loss issue
   needs fixing for Burgers solutions with negative values).

### Recommendation

The codebases are complementary, not competitive. The natural
integration path is:

1. Use UCAN as a validation oracle for our MPS encoding (compare
   our `classical_to_mps()` output against quimb's
   `MatrixProductState.from_dense()`).

2. Do NOT adopt UCAN's MPO approach as a replacement for Pauli
   decomposition. The MPO representation is efficient on classical
   hardware (where quimb can contract tensors directly), but
   converting non-unitary MPOs to quantum circuits is the unsolved
   problem the paper identifies. Our Pauli decomposition, while
   expensive, is the standard route to actual quantum circuits.

3. The real bottleneck to address is Pauli decomposition cost (4^q
   scaling). The paper's Appendix A proposes LCU as the bridge:
   express the operator as a sum of Pauli strings (which we already
   do), then use linear combination of unitaries for efficient
   circuit implementation. This is the path forward, and it builds
   on our existing Pauli infrastructure, not UCAN's MPO code.

4. Fix the boundary condition inconsistency in our classical
   baseline to match UCAN/paper (Dirichlet).


## Corrections to Prior Review (REVIEW-murali-paper-fidelity.md)

The earlier review document incorrectly stated:
- "UCAN references Gopalakrishnan Meena et al. QCE24 2024 and is a
  different paper." WRONG. It is the same paper, same authors.
- "Source term discrepancy: our code uses static g=sin(2*pi*x)."
  WRONG. Our code does use the time-varying form g=sin(2*pi*x)*cos(2*pi*t).
- "Comparing the two as implementations of the same paper is the wrong
  framing." PARTIALLY WRONG. They ARE implementations of the same paper,
  but at different stages of the pipeline.

These need to be corrected in the review document.


## Summary Table

| Aspect               | UCAN-1DBurgers            | murali_burgers            |
|----------------------|---------------------------|---------------------------|
| Library              | quimb                     | numpy + qiskit            |
| MPS encoding         | from_dense() (quimb SVD)  | manual iterated SVD       |
| Operators            | MPO (bond dim 3)          | Pauli decomp (4^q terms)  |
| Nonlinear term       | QNPU (dense detour)       | absorbed into Hamiltonian  |
| Time-stepping        | Euler (TN contraction)    | Euler (circuit evolution)  |
| Quantum circuits     | stub (not implemented)    | full (Trotter + Aer)      |
| Noise model          | none                      | T1/T2 thermal relaxation  |
| Shots/measurement    | none                      | yes (sign-loss bug)       |
| BCs                  | Dirichlet                 | periodic (quantum path)   |
| Source term           | sin(2pi*x)*cos(2pi*t)    | sin(2pi*x)*cos(2pi*t)    |
| Paper stage          | Stage 1 (TN validation)   | Stages 1-3 (full circuit) |
