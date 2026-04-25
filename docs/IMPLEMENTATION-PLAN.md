# Implementation Plan: Meena et al. AIAA 2026
# "A Tensor Network-based Quantum Algorithm for the Nonlinear 1D Burgers' Equation"

Date: 2026-04-10
Scope: q <= 6 (no parallelism). Driven by q8020-sweep. Instrumented for q8020 metadata.


## What the Paper Describes

The paper lays out a three-stage approach:

Stage 1 (implemented in paper, implemented in UCAN):
  Classical validation of tensor network operations -- encode solution
  as MPS, construct differential operators as MPOs, apply MPOs to MPS,
  compare against dense classical solutions.

Stage 2 (proposed in paper, partially implemented in our code):
  Convert tensor network representations to quantum circuits -- MPS
  state preparation via Ran 2020, operator application via Pauli/LCU
  decomposition and Trotterization.

Stage 3 (proposed in paper, partially implemented in our code):
  Execute circuits on quantum simulators and hardware -- Aer statevector,
  Aer with shots/noise, IBM backends.

The paper's time-stepping algorithm (Eq. 15, explicit Euler):

  u(x, t+dt) = u(x, t) + dt * [nu * Laplacian(u) - u * Gradient(u) + g(x,t)]

where:
  - Laplacian(u) = (S+ + S- - 2I)u / dx^2        (Eq. 10)
  - Gradient(u)  = (S+ - S-)u / (2*dx)            (Eq. 9)
  - g(x,t) = sin(2*pi*x) * cos(2*pi*t)           (Sec. III.A)
  - S+, S- are ladder (shift) operators            (Eqs. 11-12)

Operator representation: MPO form (bond dim 3 for linear ops) via
quimb's MatrixProductOperator.from_dense(). Nonlinear term handled
via Hadamard product (QNPU: apply gradient MPO, convert back to
dense, element-wise multiply, re-encode).

Alternative quantum circuit route (Appendix A.A): express evolution
operator as sum of Pauli strings via LCU. Coefficients found by
solving linear system (Eq. 16). Evolution: e^{-i*dt*A_hat} (Eq. 17).


## What We Already Have

File-by-file status in murali_burgers/:

burgers_solver.py -- CLI + q8020 metadata integration
  DONE: argparse with q, nu, cfl, n_steps, shock_pct, ic, source,
        method, trotter_order, trotter_reps, bond_dim, mps_threshold,
        shots, backend, t1, t2, save_every, noshow
  DONE: q8020 metadata fragments (case, results, analysis, artifacts)
  DONE: always runs classical baseline for comparison
  DONE: wired into q8020-sweep via TOML

burgers_classical.py -- Classical Euler baseline
  DONE: initial_condition_sine, source_term_sine (time-varying)
  DONE: gradient_central, laplacian_central, euler_step, solve_burgers
  BUG:  Mixed BCs -- laplacian uses periodic wrapping, gradient uses
        one-sided. Paper uses one-sided (Dirichlet) throughout.

burgers_nonlinear.py -- Pauli decomposition (Appendix A approach)
  DONE: generate_pauli_labels, solve_pauli_coefficients
  DONE: build_evolution_hamiltonian (LCU Eq. 16-17)
  DONE: evolution_circuit (SuzukiTrotter synthesis)
  DONE: exact_evolution_matrix (scipy.linalg.expm)
  NOTE: This implements the paper's Appendix A, not the main body's
        MPO approach. The Appendix A route is the one that maps to
        actual quantum circuits, so this is correct for our purposes.

burgers_mpo.py -- Shift operator circuits
  DONE: shift_matrix (dense S+/S-), increment_circuit, decrement_circuit
  DONE: gradient_lcu_circuit, laplacian_lcu_circuit
  NOTE: Only shift_matrix() is used in the CLI pipeline (called by
        burgers_nonlinear.py for dense RHS computation). The quantum
        circuits (increment, decrement, gradient_lcu, laplacian_lcu)
        are validated building blocks not yet wired into time-stepping.
        They would become the controlled unitaries inside a future
        LCU SELECT/PREPARE circuit.

burgers_mps.py -- MPS state preparation
  DONE: classical_to_mps (iterated SVD, left/right canonical)
  DONE: mps_to_circuit (Ran 2020 algorithm)
  DONE: mps_fidelity, mps_circuit_fidelity

burgers_trotter.py -- Time-stepping orchestration
  DONE: quantum_circuit_step (Pauli decomp + Trotter per step)
  DONE: quantum_exact_step (Pauli decomp + expm per step)
  DONE: shift_euler_step (classical shift-operator Euler)
  DONE: mps_step (MPS state prep + exact evolution)
  DONE: run_simulation (dispatch to any method, collect metrics)
  DONE: shots/noise wiring through all paths

input/burgers_quantum.toml -- Sweep configuration
  DONE: classical baselines q=3..6, quantum circuit q=3..6,
        quantum exact q=3..5, Trotter convergence, MPS bond sweeps,
        shots studies, noise studies, viscosity study


## What Needs to Be Done

### Phase 1: Fix Known Discrepancies

1.1  <DONE> Fix classical baseline boundary conditions
     File: burgers_classical.py
     Problem: gradient_central uses one-sided at boundaries but
     laplacian_central uses periodic wrapping. The paper uses
     one-sided (Dirichlet, u=0 at boundaries) for both.
     Fix: Change laplacian_central to use one-sided differences
     at boundaries, matching gradient_central.
     Impact: Affects error metric (classical is the validation oracle).

1.2  <DONE> Add Dirichlet BC option to quantum path
     File: burgers_mpo.py, burgers_nonlinear.py
     Problem: Shift operators S+/S- wrap mod N (periodic BC).
     The paper uses Dirichlet (u=0 at boundaries). For sine IC
     with u(0)=u(1)=0 the difference is negligible, but for
     correctness we should support both.
     Fix: Add a --bc flag (periodic/dirichlet). For Dirichlet,
     modify the dense operator matrices to zero the boundary rows
     before Pauli decomposition.
     Priority: Low for q<=6 sine case. Higher for Walters case.

1.3  <DONE> Verify source term propagation through quantum path
     File: burgers_trotter.py
     Status: VERIFIED. Source term g(x,t) = sin(2*pi*x)*cos(2*pi*t)
     is correctly evaluated with t = step*dt at each step and passed
     to the Hamiltonian builder. Matches paper.


### Phase 2: q8020 Metadata and Sweeper Integration

2.1  <DONE> Enrich metadata fragments
     DONE: Added bc to case fragment and JSON summary.
     DONE: Added n_qubits to per-step circuit metrics.
     DONE: Added avg_cx_gates and n_qubits to analysis fragment.
     Remaining: Trotter error estimate (diff between quantum_exact
     and quantum_circuit at same parameters) -- deferred to Phase 3
     validation runs where both are available for comparison.

2.2  <DONE> TOML sweep completeness
     DONE: Added Dirichlet BC comparison groups:
       - dirichlet_classical_q3..q6 (shift method, bc=dirichlet)
       - dirichlet_qexact_q3..q5 (quantum_exact, bc=dirichlet)
       - dirichlet_qcircuit_q3..q5 (quantum_circuit, bc=dirichlet)
     Existing periodic groups retained for direct comparison.

2.3  <DONE for now> Post-processing scripts
     Existing scripts in analysis/ cover MPS bond sweeps, shots
     studies, and wave overlays. Trotter convergence and circuit
     resource plots deferred until Phase 3 validation data exists.


### Phase 3: Validation Runs (q <= 6)

3.1  <DONE> Run the full TOML sweep
     Command: q8020-sweep q8020-cfd-axequalsb/input/burgers_quantum.toml
     Verify all cases complete without error.

3.2  Cross-validate against UCAN
     For q=5, q=6: run UCAN's quimb-based MPS/MPO path on the same
     parameters. Compare our classical baseline output against UCAN's.
     This validates that our shift-operator FD matches quimb's MPO
     application. Any discrepancy here points to a BC or operator
     construction bug.

3.3  <DONE - test case in toml> Validate quantum_exact == classical (to machine precision)
     The quantum_exact path (Pauli decomposition + expm) should match
     the classical shift-operator path to floating-point precision.
     If it does not, there is a bug in the Pauli coefficient solve
     or the normalization round-trip. Run at q=3..5 and check
     errors are O(1e-14).

3.4  <DONE> Characterize Trotter error budget
     At q=5, q=6: compare quantum_circuit vs quantum_exact at
     trotter_reps = [1, 2, 5, 10, 20]. Determine the reps needed
     for Trotter error to be below discretization error.


## What We Are NOT Implementing (Future Work)

### From our roadmap:

F1.  Parallelized Pauli decomposition
     The 4^q scaling wall. At q=7 (16,384 terms) and q=8 (65,536
     terms), serial Pauli decomposition becomes the bottleneck.
     Approaches: multiprocessing/joblib on local hardware, or
     distribute across Frontier nodes. The solve_pauli_coefficients
     function in burgers_nonlinear.py is the target -- the P_i|u>
     precomputation and the S matrix construction are embarrassingly
     parallel.

### From the paper's future work (Sec. V):

F2.  Time-evolving block decimation (TEBD, Ref. 30)
     An alternative to Trotterization that works directly in MPS
     form by applying two-site gates and re-compressing. Would
     avoid the Pauli decomposition entirely. Not implemented.

F3.  LCU SELECT/PREPARE circuit
     Our code implements the Pauli coefficient solve (Eq. 16) and
     Hamiltonian construction. What we do NOT yet implement is the
     LCU circuit itself -- the ancilla-based SELECT/PREPARE protocol
     that efficiently applies the sum of unitaries without chaining
     all 4^q rotations sequentially. The shift operator circuits in
     burgers_mpo.py (increment, decrement, gradient_lcu, laplacian_lcu)
     would become the controlled unitaries inside the SELECT oracle.
     This is the key to scaling beyond q=8.

F4.  Variational fast-forwarding (Appendix A.B, Refs. 31, 34, 35)
     Compresses M time steps into a single circuit of fixed depth:
     (e^{iH*dt})^M = W * D(M*dt) * W_dag. Would eliminate the
     per-step circuit overhead. Requires variational optimization
     of the diagonalizing unitary W. Not implemented.

F5.  Krylov subspace methods (Refs. 31, 34, 35)
     Alternative to Trotter for Hamiltonian simulation. Builds a
     Krylov basis {|u>, H|u>, H^2|u>, ...} and projects the
     evolution into this low-dimensional subspace. Not implemented.

F6.  RK4 time integration
     The paper uses Euler (first-order). They note RK4 as a
     future improvement for accuracy. Would require multiple
     Hamiltonian evaluations per step (4 for RK4), increasing
     circuit count proportionally.

F7.  Trotter error characterization
     The paper does not use Trotterization (it uses classical Euler).
     Our code introduces Trotter error on top of the Euler scheme.
     Tasks:
     a) Trotter convergence TOML cases already exist for q=4
     b) Extend to q=5, q=6
     c) Compare quantum_circuit vs quantum_exact at each q
        to isolate Trotter error from discretization error

F8.  Walters et al. multimodal test case
     Paper's second test case: N=8192 (q=13), 12-mode initial
     condition, nu=1e-5, CFL=0.05. Way beyond our q<=6 scope.
     Requires parallelism and likely real quantum hardware.

F9.  <DONE> Sign recovery for measurement-based (shots>0) path
     The counts-to-amplitudes reconstruction uses sqrt(counts/total),
     which loses the sign of the wavefunction.  For Burgers with
     sin(2pi*x) IC the solution goes negative past the shock front
     (~x>0.5), so the shots path produces ε~1 in that region.
     Currently plots are clipped to x<0.5 as a workaround.

     Four options, forming a spectrum from classical to fully quantum:

     Option 1 — As-is (no sign recovery)
       Keep the current sqrt(counts/total) reconstruction.  The
       solution is always non-negative.  Useful as a baseline that
       *demonstrates* the sign-loss problem and quantifies its effect
       on solution error.  Already implemented.

     Option 2 — Hadamard test / interferometric approach (quantum sign recovery)
       Use interference with a reference state |φ⟩ (previous time
       step's signed solution) to recover signs without classical
       computation of the solution itself.
       Circuit: one ancilla + controlled-PauliEvolutionGate.
         anc |0⟩ ── H ──── •(e^{-iHdt}) ── H ── M_anc
         reg |0⟩ ── U_φ ────────────────────── M_reg
       Measure (anc, data=k) jointly.  Then:
         sign(ψ_k) = sign(P(anc=0,k) - P(anc=1,k)) * sign(φ_k)
       Cost: +1 ancilla qubit, ~2x circuit depth (controlled evo gate).
       Bootstrapping: use known IC signs at t=0; propagate step-by-step.
       This is the "tradeoff" option — quantum sign recovery without
       doubling qubit count.

     Option 3 — Classical sign oracle (hybrid reference)
       The classical RHS is already computed every step for norm
       prediction.  Extract signs from the classical Euler update:
         u_classical_next = u + dt * compute_rhs_shift(u, dx, nu, g)
         u_signed = np.sign(u_classical_next) * np.abs(u_quantum)
       Zero extra quantum cost.  Quantum gives magnitudes (at potential
       advantage for large N); classical gives signs for free.  This
       is the practical near-term baseline and the reference against
       which Options 2 and 4 are compared.

     Option 4 — Dual-rail encoding (idealized, fully quantum)
       Decompose u = u⁺ - u⁻ where u⁺_k = max(u_k,0) and
       u⁻_k = max(-u_k,0).  Encode each as a separate non-negative
       quantum state, evolve separately, subtract classically.
       Eliminates the sign problem entirely; evolution stays fully
       quantum.  Cost: 2x qubits and 2x circuit executions per step.
       Establishes the upper bound on accuracy for the shots path.

     Recommended implementation order: 3 (free, immediate), then 1
     (already done), then 2 (quantum paper contribution), then 4
     (theoretical completeness).

F10. Cole-Hopf + TEBD (classical-free mid-flight quantum solver)

     Motivation
     ----------
     The Meena/Murali Pauli-decomposition method implemented in this
     codebase is structurally classical-driven: at every step
     build_evolution_hamiltonian runs a classical Euler update
     (u_next = u + dt * compute_rhs_shift(u)) and fits ~4^q Pauli
     coefficients so that the quantum unitary reproduces the classical
     one-step target.  The quantum circuit is a trailing executor of a
     classically-designed operator.  In addition, quantum_circuit_step
     applies a post-readout rescaling by the classical norm prediction.
     This method cannot asymptotically outperform its own classical
     subroutine and therefore cannot demonstrate quantum advantage for
     nonlinear PDE simulation.

     The obstacle is the nonlinear convective term u * d(u)/dx: because
     it is quadratic in u, the Hamiltonian depends on the current state
     and a fixed 2-site gate decomposition is impossible on a naive
     state-only encoding.

     Approach
     --------
     Apply the Cole-Hopf transformation to linearize the viscous
     Burgers equation into the heat equation:

         u = -2 * nu * phi_x / phi         (definition)
         u_t + u * u_x = nu * u_xx   <==>  phi_t = nu * phi_xx

     phi satisfies a pure linear heat equation.  Then simulate phi
     with Time-Evolving Block Decimation (TEBD), a tensor-network
     algorithm that applies local 2-site gates directly to an MPS
     representation of the state:

         H_heat = (nu / dx^2) * sum_i (I - shift(i, i+1))

     is a sum of nearest-neighbor terms.  Second-order Trotter:

         U(dt) = prod_{odd i}  exp(-i H_{i,i+1} dt / 2)
               * prod_{even i} exp(-i H_{i,i+1} dt)
               * prod_{odd i}  exp(-i H_{i,i+1} dt / 2)

     Each 2-site gate is a fixed 4x4 matrix computed ONCE at setup
     from the Hamiltonian structure.  No classical RHS evaluation
     inside the time loop; no per-step Pauli refit; no norm rescaling.
     After each gate application the MPS is SVD-truncated to bond
     dimension chi_max.

     Classical work (confined to setup and readout only)
     ---------------------------------------------------
     Setup (once):
       phi_0(x) = exp( -(1 / (2 * nu)) * integral_0^x u_0(s) ds )
       MPS encoding of phi_0 with target chi_max.
       Construction of fixed 2-site gates exp(-i h_{i,i+1} dt).

     Readout (post-simulation, or at each requested snapshot):
       u(x, t) = -2 * nu * phi_x(x, t) / phi(x, t)
       Evaluated via quantum expectation values <phi|O|phi> for the
       required observables, or classically after full statevector
       reconstruction for verification at small q.

     Between setup and readout the time evolution is entirely quantum:
     a fixed linear Hamiltonian, fixed gates, MPS bond-dimension
     truncation only.  This is the first path in the codebase that
     qualifies as a genuine quantum simulation of Burgers.

     Caveats and open design questions
     ---------------------------------
     a) Cole-Hopf validity: requires phi(x,t) > 0 everywhere.  True for
        smooth positive-exponential ICs; must verify for shocks.
        (The heat equation preserves positivity, so if phi_0 > 0 then
        phi(t) > 0 for all t.  u can still develop shocks because the
        nonlinearity is absorbed into the logarithmic derivative.)

     b) Dirichlet BC on u translates to Neumann-like BC on phi:
        u(0) = u(1) = 0  =>  phi_x(0)/phi(0) = phi_x(1)/phi(1) = 0,
        i.e. reflecting/insulated BC on the heat equation.  The 2-site
        gate structure at boundaries must reflect this.

     c) Division at readout: u = -2*nu*phi_x/phi is singular where
        phi -> 0.  For smooth viscous Burgers this should not occur
        (phi stays positive), but needs monitoring.

     d) TEBD vs MPS state-prep circuits: TEBD is a *tensor-network*
        algorithm that can be executed classically for verification
        (quimb) and on quantum hardware via MPS-to-circuit synthesis
        (existing burgers_mps.py machinery).  The quantum circuit
        implementation of TEBD itself is non-trivial and may require
        a second sub-task; a classical TEBD reference is the first
        deliverable.

     Deliverables (proposed sub-tasks)
     ---------------------------------
     F10.1  burgers_cole_hopf.py: forward/inverse transforms
             u <-> phi, analytical test against known solutions.
     F10.2  burgers_tebd.py: classical TEBD reference using quimb
             tensor network primitives.  2nd-order Trotter gates,
             bond-dim sweep, verification against analytic heat
             equation solutions (Fourier modes decay as
             exp(-nu * (2*pi*k)^2 * t)).
     F10.3  Integration: wire TEBD through run_simulation so
             --method=tebd is a selectable option in burgers_solver.py
             with all the existing sweep/plot infrastructure.
     F10.4  Verification: F10 vs Pauli-decomposition path at
             q in {3,4,5}.  Expect agreement on smooth evolution;
             small discrepancies at shock locations due to bond-dim
             truncation.  Sweep chi_max to characterize the trade-off.
     F10.5  Quantum TEBD circuits (optional, later): synthesize
             2-site gates into quantum circuits; execute via
             MPS state preparation + circuit evolution.  This is
             where the method meets actual quantum hardware.

     References
     ----------
     * Cole, J.D. (1951), "On a quasi-linear parabolic equation
       occurring in aerodynamics", Quart. Appl. Math. 9: 225-236.
     * Hopf, E. (1950), "The partial differential equation u_t +
       u u_x = mu u_xx", Comm. Pure Appl. Math. 3: 201-230.
     * Vidal, G. (2004), "Efficient simulation of one-dimensional
       quantum many-body systems", PRL 93, 040502. [TEBD]
     * Schollwöck, U. (2011), "The density-matrix renormalization
       group in the age of matrix product states", Ann. Phys. 326,
       96-192.  [TEBD/MPS algorithmic review]

F11. Burgers turbulence (Burgulence) statistical study

     Motivation
     ----------
     The --ic multimode option (initial_condition_multimode) produces a
     single deterministic multi-mode Fourier field with random phases.
     This is NOT Burgers turbulence despite what an earlier version of
     the code was misleadingly named.  True Burgulence is a statistical
     object requiring an ensemble of realizations with prescribed IC
     statistics (or stochastic forcing), analyzed via ensemble-averaged
     energy spectra, structure functions, and velocity-gradient PDFs.
     The classic result is E(k) ~ k^{-2} in the inertial range,
     emerging universally from shock dominance regardless of IC
     details (Kida 1979).

     Scope of this task
     ------------------
     Implement a statistical study of decaying Burgulence on the
     best-available quantum solver path (presumably F10 Cole-Hopf +
     TEBD by the time this is undertaken).

     Deliverables
     ------------
     F11.1  Burgulence-correct IC sampler
            Gaussian random field with prescribed energy spectrum
            E(k) = C * k^{-beta} * exp(-(k/k_c)^2), drawing Fourier
            coefficients A_k ~ N(0, sqrt(E(k))) and independent random
            phases.  Distinct from --ic multimode which uses
            deterministic A_k = k^{-alpha}.

     F11.2  Ensemble sweep driver
            Run N_realizations (~50-200) independent simulations at
            fixed Re = U_rms * L / nu, varying only ic_seed.  Use the
            existing q8020-sweep infrastructure with a parameter
            array over seed values.

     F11.3  Ensemble-averaged diagnostics
            Post-processor that reads all realizations from a sweep
            and computes:
              a) E(k, t) = <|u_hat(k, t)|^2> averaged over seeds;
                 plot vs k on log-log to test k^{-2} scaling.
              b) Structure functions S_p(r, t) =
                 <|u(x+r) - u(x)|^p> for p = 1, 2, 3, 4, 6.
              c) PDF of velocity gradients du/dx.
              d) Shock density / shock statistics.
            All diagnostics time-resolved to show decay dynamics.

     F11.4  Scale separation
            Requires q >= 10 (N >= 1024) to have an inertial range
            between forcing/IC scale and Kolmogorov dissipation
            scale.  At q <= 6 the dissipation scale is comparable
            to the grid scale and no inertial range exists.  This
            task therefore depends on F10 achieving q >= 10 via
            TEBD bond-dim efficiency, or on hardware scaling.

     F11.5  Forced Burgulence (optional, harder)
            Add stochastic white-in-time forcing with prescribed
            spatial correlation.  Study steady-state statistics.
            Requires extending the solver loop to accept a
            time-dependent source_fn with stochastic sampling and
            variance rescaling per-step.

     Dependencies
     ------------
     Depends on F10.  Ensemble runs are expensive; need efficient
     per-seed execution, which the Cole-Hopf/TEBD path provides
     more naturally than the Pauli-decomposition path (no per-step
     classical refit).

     Why this is interesting
     -----------------------
     Burgulence is the canonical toy model for 1D shock-dominated
     turbulence.  Demonstrating k^{-2} scaling and bifractal
     structure-function exponents (zeta_p = min(p/3, 1)) on a
     quantum-simulated Burgers solver would be a genuine physics
     result, not just a demo.  It requires scales that only
     tensor-network compression can reach within current quantum
     hardware, making it a natural headline result for F10.

     References
     ----------
     * Kida, S. (1979), "Asymptotic properties of Burgers turbulence",
       J. Fluid Mech. 93, 337-377.
     * E, Khanin, Mazel, Sinai (2000), "Invariant measures for
       Burgers equation with stochastic forcing", Ann. Math. 151,
       877-960.
     * Bec, J. & Khanin, K. (2007), "Burgers turbulence",
       Phys. Rep. 447, 1-66.  [comprehensive review]
     * Frisch, U. & Bec, J. (2001), "Burgulence", in New Trends in
       Turbulence, Les Houches 2000.

F12. IC and BC extensions

     Motivation
     ----------
     Current options are narrow: IC in {sine, multimode}, BC in
     {periodic, dirichlet(u=0)}.  This limits both physical coverage
     and verification rigor.  In particular, no current IC provides
     an analytic reference solution -- all accuracy claims are
     relative to a classical FTCS reference which is itself only a
     numerical approximation.

     Deliverables
     ------------

     F12.1  Cole-Hopf-exact IC (highest priority)
            ---------------------------------------
            Construct u_0(x) such that the Cole-Hopf-transformed
            phi_0(x) is a finite sum of cosine modes:
                phi_0(x) = a_0 + sum_{n=1..M} a_n * cos(n*pi*x)
            Each mode evolves independently under the heat equation:
                phi(x, t) = a_0 + sum_n a_n * cos(n*pi*x)
                              * exp(-nu * (n*pi)^2 * t)
            with inverse transform giving an analytic
                u(x, t) = -2*nu * phi_x(x, t) / phi(x, t).
            This is a genuine analytic reference for quantifying
            quantum-solver accuracy without a classical cosolver.
            Aligned with F10: the same transform machinery serves
            both the Cole-Hopf+TEBD solver and this verification IC.

            Deliverable: initial_condition_cole_hopf_exact(x, coeffs, nu),
            and analytic_solution_cole_hopf(x, t, coeffs, nu) returning
            the exact u(x,t) for the given Fourier coefficients.

            Validation: choose coeffs giving a single-mode phi
            (analytically a tanh-type u profile), verify quantum
            solver tracks the exact u(x,t) within ε bounds at
            arbitrary t.

     F12.2  Gaussian IC
            -----------
            u_0(x) = A * exp(-((x - x0) / sigma)^2)
            Smooth localized pulse; demonstrates shock formation from
            a single-lobe disturbance.  With Dirichlet BC and sigma
            small compared to domain, u_0 is effectively zero at the
            boundaries.  Useful for pedagogical shock-formation
            animations.  Small implementation (~10 lines).

     F12.3  Neumann BC
            ----------
            du/dx = 0 at x = 0 and x = 1 (zero-flux / reflecting walls).
            Implementation: in the shift operators, ghost nodes copy
            their interior neighbor:
                (S+ u)[N-1] = u[N-1]    (was 0 for dirichlet)
                (S- u)[0]   = u[0]      (was 0 for dirichlet)
            Equivalent to a "symmetry plane" or "adiabatic wall".
            Changes touch burgers_mpo.py (shift matrices),
            burgers_nonlinear.py (compute_rhs_shift bc handling),
            and burgers_solver.py (grid setup and --bc choices).

     Out of scope for F12
     --------------------
     * Riemann / step ICs: discontinuous, hostile to amplitude
       encoding state preparation; would need smoothing anyway.
     * Nonzero Dirichlet: easy extension of F12.3 but no physical
       case in the current roadmap needs it.
     * Robin / mixed BC: low-priority for canonical Burgers demos.
     * Gaussian random field IC: absorbed into F11.

     Dependencies
     ------------
     F12.1 shares the Cole-Hopf transform code with F10; implement
     F10.1 (cole_hopf.py) first.  F12.2 and F12.3 are independent
     and can happen anytime.

