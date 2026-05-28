# Complete Code Review: murali_burgers

Date: 2026-04-19
Scope: all seven modules under src/murali_burgers, the CLI, and the
TOML sweep.  Reviewed against IMPLEMENTATION-PLAN.md and the paper
(Meena/Murali AIAA 2026).

Companion document: F9-CODE-REVIEW.md covers the sign-recovery task
in detail.  Items C1, C2, C8 below reference findings duplicated
there for continuity.


## Overall assessment

The module is cohesive and does what the plan describes.  Each solver
method builds on the previous; the CLI is paper-faithful; the q8020
metadata wiring is complete.  The mathematical spine (Pauli
decomposition via Eq. 16/17, Ran-2020 MPS prep with a QR extension
for truncated states) is correctly implemented.  Finite issues below,
split into correctness, redundancy, consistency, math clarity,
style, and testing.


## Update 2026-04-19 (2): Multi-mode Fourier IC

(Renamed from the earlier "turbulence IC" draft -- the symbol now in
the codebase is initial_condition_multimode, the CLI flag is
--ic multimode, and the docstring/help text explicitly disclaim any
Burgulence interpretation.  See F11 in the implementation plan for
the statistical object.)

What was added: initial_condition_multimode in burgers_classical.py,
wired through the CLI as --ic multimode with --ic-modes, --ic-seed,
--ic-alpha, plus two sweep groups in input/burgers_quantum.toml --
hadamard_multimode_q5 (q=5, n_modes=4) and hadamard_multimode_q6
(q=6, n_modes=6), both seed=42, alpha=1.0, bc=dirichlet, source=none,
shock_pct=150.0.

What is good:

  + Naming is now honest.  The docstring, CLI help, and TOML comments
    all spell out "NOT Burgers turbulence" and point at F11.  That
    removes the misleading physics claim without hiding the useful
    demonstrator.
  + Amplitude law A_k = k^-alpha with alpha=1 still gives a k^-2
    energy spectrum per mode, which is the right default for a
    multi-shock demonstrator.
  + Seeded RNG + post-normalization to max|u0|=1 keeps the CFL
    predictable and runs reproducible.
  + Correct argparse wiring; TOML params flow through to the IC
    unchanged.

Issues (most carry over verbatim from the earlier draft -- the rename
did not touch the math):

MM1. <DONE> The IC is NOT Dirichlet-compatible.  Critical.  [was TU1]
     Docstring at burgers_classical.py:38 still claims:
         "All modes are pure sines of integer wavenumber so the IC
          satisfies u(0) = u(1) = 0 automatically (Dirichlet-
          compatible)."
     But the implementation at burgers_classical.py:55 is
         u += A_k * sin(2*pi*k*x + phi_k)
     with phi_k uniform in [0, 2*pi).  For integer k:
         sin(2*pi*k*0 + phi_k) = sin(phi_k)
         sin(2*pi*k*1 + phi_k) = sin(2*pi*k + phi_k) = sin(phi_k)
     So u(0) = u(1) = sum_k A_k sin(phi_k), generally nonzero and
     equal at the two endpoints.  With the TOML defaults for
     hadamard_multimode_q5 (seed=42, n_modes=4, alpha=1.0, q=5):
         u0[0]  = -0.7086
         u0[N-1]= -0.7086
         max|u0|=  1.0000
     The IC violates its own docstring by ~71% of the state magnitude.
     Combined with B1 (Dirichlet not enforced in time), the ~0.71
     endpoint value persists and contaminates the Laplacian stencil
     for the entire run.
     Fixes:
       (a) drop the phase: u += A_k * sin(2*pi*k*x).  Retain the
           seeded RNG by randomizing AMPLITUDES instead
           (A_k = rng.normal() * k^-alpha).
       (b) sine basis: u += A_k * sin(pi*k*x).  Zero at x=0, x=1 for
           any integer k >= 1.  This is the Dirichlet eigenbasis on
           [0,1] and matches the standard "sine series" convention.
     Pick one and update the docstring to match.  The multi-shock
     interaction character that motivates the IC is preserved by
     either fix.

MM2. <DONE> shock_pct semantics silently change for multi-mode.  [was TU2]
     burgers_solver.py:169 uses t_shock = 1/(2*pi), derived for the
     sine IC where max|u'(x=0)| = 2*pi.  For the multi-mode IC with
     A_k = k^-alpha the un-normalized max|u'| ~ 2*pi*sum_k k*A_k, and
     after max|u|=1 rescaling the effective shock time differs.  So
     --shock-pct=150.0 in hadamard_multimode_q5/q6 means "150% of the
     sine shock time", not 150% of the multi-mode shock time.  Frames
     labelled post-shock in the plot may actually be mid-formation
     for n_modes > 1 -- especially worse for q6 (n_modes=6).
     Fix options: compute t_shock = 1/max|du0/dx| when ic != "sine",
     or introduce an explicit --t-end flag and deprecate --shock-pct
     for non-sine ICs.

MM3. <DONE> IC params are not recorded in case metadata or summary.
     [was TU3]  burgers_solver.py:248-271 passes ic=args.ic to
     make_case_meta but not ic_modes / ic_seed / ic_alpha.  The
     summary dict at burgers_solver.py:327-348 omits all four
     (including ic itself).  Consequence: the two existing
     multi-mode sweeps differ only in q and ic_modes, and the latter
     is not in the fragments; ic_seed/ic_alpha variations in future
     sweeps would collide in metadata space entirely.  Add ic,
     ic_modes, ic_seed, ic_alpha to make_case_meta and to the JSON
     summary.  Harmless None for sine runs.

MM4. <DONE> No validation of n_modes vs N.  [was TU4]
     Default n_modes=6 aliases for q<=3 (N<=8, Nyquist=4) and barely
     fits q=4 (N=16).  Docstring says "Keep <= N/4" as advice but
     nothing enforces it.  hadamard_multimode_q5 (N=32, n_modes=4) is
     safe; hadamard_multimode_q6 (N=64, n_modes=6) is safe; but a
     user copying the group and dropping q would not get a warning.
     Add
         n_modes = min(n_modes, N // 2 - 1)
     (or raise) at the top of solve_burgers or in the IC once N is
     known.  Cheap insurance.

MM5. <DONE> m == 0 silently returns a zero array.  [was TU5]
     burgers_classical.py:56-58: if amplitudes conspire to cancel,
     m=0 and the function returns zeros.  Downstream the solver will
     blow up in norm_u at step 0.  Either raise in the IC ("IC
     amplitude collapsed to zero -- pick a different seed") or drop
     the guard and let division by zero surface it loudly.

MM6. <DONE> Interaction with B1 amplifies MM1.  [was TU6]
     Fixing MM1 alone yields u(0)=u(1)=0 at t=0 but B1 still lets the
     boundary drift from zero over time.  Fixing B1 alone pins the
     endpoints at whatever nonzero value the IC happens to produce.
     For the Dirichlet sweeps (hadamard_multimode_q5/q6) both must be
     fixed together to get a consistent Dirichlet solver.

MM7. <DONE> Unused IC args have no effect on sine runs.  [was TU7]
     --ic-modes 20 --ic sine is accepted and silently ignored.  Not
     harmful, but a one-line warning ("--ic-modes/--ic-seed/--ic-alpha
     only apply to --ic multimode") would save a future debugging
     session.  Cheap.

MM8. <DONE> Two near-duplicate sweep groups.  [new, post-rename]
     hadamard_multimode_q5 and hadamard_multimode_q6 differ only in q
     and ic_modes.  Everything else is copy-pasted.  Once the TOML
     sweeper supports per-group matrix expansion, collapse these to
     a single group with paired lists; until then, at least add a
     comment pointing out that the two groups must stay in lockstep
     on shots / save-every / bc / sign-recovery.  Otherwise drift
     between them will silently bias the q5-vs-q6 comparison the
     animation postproc is meant to produce.

Priority: MM1 is blocking for any Dirichlet multi-mode run (both
current sweeps are in this state).  MM3 blocks post-processing
reproducibility.  MM2 is interpretive -- either fix it or document
the sine-calibrated shock_pct convention in the group comment.  The
rest are cleanup.


## Update 2026-04-19: Boundary-condition refactor

A refactor in another IDE changed the classical baseline to route
through the shift-operator kernel and introduced a grid-endpoint
toggle.  Two files affected:

  burgers_classical.py:
    solve_burgers now accepts bc and calls
    burgers_nonlinear.compute_rhs_shift(u, dx, nu, g, bc=bc) directly
    instead of the local euler_step/gradient_central/laplacian_central.

  burgers_solver.py:
    Grid is np.linspace(0,1,N,endpoint=True) for bc="dirichlet" and
    endpoint=False for bc="periodic".  bc is threaded into
    solve_burgers.

What is good:

  + Classical reference and quantum path now see the exact same
    stencil, closing Phase 1.1's divergence concern at its root
    rather than by patching laplacian_central.
  + The endpoint toggle is semantically right: for Dirichlet the
    grid includes x=0 and x=1 so the sine IC has u[0]=u[N-1]=0
    naturally; for periodic it excludes x=1 since that point is
    identified with x=0.

Remaining issues introduced or still present:

B1. <DONE> Dirichlet boundaries are not enforced in time.
    compute_rhs_shift with bc="dirichlet" gives the correct one-sided
    stencils at i=0 and i=N-1 *assuming* u there stays zero, but
    nothing clamps u[0] and u[N-1] to zero after the update.  At
    i=0 with u[0]=0 and g[0]=sin(2pi*0)*cos(2pi*t)=0, the update
    becomes
        u_next[0] = 0 + dt*(nu * u[1]/dx^2)   != 0
    so u[0] drifts.  Empirical measurement (q=4, nu=1e-4,
    cfl=0.01, 95 steps): u[0] ~ +4.7e-4 (0.04% of interior max),
    symmetric at u[N-1].  Because the *classical* baseline now
    uses the same kernel, both sides drift identically and the
    epsilon plots are unaffected -- but this means the validation
    sweep cannot detect the BC-enforcement bug, and cross-validation
    against UCAN (plan section 3.2, UCAN enforces u[0]=u[N-1]=0
    strictly) will show a delta that is not a bug in either code.
    Fix: after each u update in run_simulation (and solve_burgers)
    when bc=="dirichlet", set u[0] = u[N-1] = 0.  Alternatively,
    zero rows 0 and N-1 of compute_rhs_shift's output when
    bc=="dirichlet" -- same effect, cheaper to justify as a
    stencil-level choice.

B2. <IGNORE - comment> dx differs between periodic and Dirichlet at the same q.
    endpoint=True gives dx = 1/(N-1); endpoint=False gives dx = 1/N.
    Since dt = cfl*dx, periodic and Dirichlet runs at the same q run
    at slightly different dt and cover slightly different physical
    time after n_steps.  This is intentional (paper uses Dirichlet
    with N points spanning [0,1]) but the periodic-vs-Dirichlet
    comparison groups in input/burgers_quantum.toml now compare
    results on different temporal grids.  Worth a one-line comment
    in the sweep config so a future reader does not chase a phantom
    bias between groups.

B3. <DONE> Module docstring is stale.
    burgers_classical.py header says "Uses forward-time central-space
    (FTCS) explicit Euler".  solve_burgers no longer calls euler_step;
    it wraps compute_rhs_shift.  Update the header to describe the
    shift-operator FD kernel and note that gradient_central,
    laplacian_central, and euler_step are retained only for the
    module's __main__ demo.

B4. <DONE> Dead code in burgers_classical.py.
    With solve_burgers routed through compute_rhs_shift, the
    functions gradient_central, laplacian_central, euler_step,
    build_gradient_matrix, build_laplacian_matrix are only exercised
    by the module's __main__.  Supersedes D1 in this review.  Delete
    or keep with a "reference, not used" comment.

B5. <DONE> Local import inside solve_burgers.
    burgers_classical.py:94 imports compute_rhs_shift at call time.
    burgers_nonlinear does not import burgers_classical, so a
    top-level import is safe and matches the "imports at top, grouped"
    rule (Best-Practices 10).

B6. <DONE> shift_matrix still returns complex (K3 below).
    With solve_burgers now calling compute_rhs_shift, the classical
    baseline runs in complex dtype every step and the result is
    downcast via .real only at the top of run_simulation (or
    discarded implicitly by the output writer).  Fixing shift_matrix
    to return real dtype cleans up the RuntimeWarnings visible when
    the classical baseline blows up (e.g. at shock time).

Priority: B1 is a real BC correctness bug that is currently masked
by the "same kernel on both sides" symmetry.  Fix before the UCAN
cross-validation in Phase 3.2.  Others are cleanup.


## Correctness

C1. dual_rail_quantum_step builds a per-rail Hamiltonian.
    Covered in F9-CODE-REVIEW.md.  Calling
    quantum_circuit_step(u_rail, ...) at burgers_trotter.py:340 means
    the Burgers convection term is linearized around u+ (resp. u-)
    rather than u, so the split does not recombine back to a Burgers
    step.  If dual_rail is supposed to be a sign-recovery technique
    for the same evolution, it is wrong.  If it is supposed to be an
    alternative physical scheme, the plan says otherwise.

C2. <IGNORE - see other doc> Hadamard-test returns sign(Re psi) * |psi| for complex psi.
    Covered in F9-CODE-REVIEW.md.

C3. <DONE> Sign-convention reconciliation in compute_rhs_shift is implicit.
    With the code's shift_matrix, S+ acts as (S+ u)_j = u_{j-1}
    (column-based), so the standard central-difference gradient
    (u_{j+1} - u_{j-1}) / (2dx) equals -(S+ - S-) u / (2dx).  The
    code at burgers_nonlinear.py:54 has the required leading minus
    sign, so the math is right.  But paper Eq. 9 writes
    (S+ - S-) / (2dx) without the minus, because the paper's S+
    convention is the opposite.  A reader comparing code line-by-line
    against the paper will flag this as a bug.  Add a comment on the
    convention mismatch and a sign-propagation regression test that
    catches a future "fix" breaking it.

C4. <DONE> Statevector mode silently discards noise models.
    burgers_trotter.py:151: if a caller passes a T1/T2-configured
    backend via backend=... with shots=0, the code creates a fresh
    AerSimulator(method="statevector") and runs the circuit
    noiselessly.  A sweep row with shots=0, t1=30 looks like a noisy
    statevector run but is not.  Either raise when shots=0 and noise
    params are set, or drop the shortcut and run on the supplied
    backend with save_statevector() (Aer supports this).

C5. avg_cx_gates divides by n even when some step-metrics lack
    gate_counts.
    burgers_solver.py:281 does
        sum(m["gate_counts"].get("cx",0) for m in step_metrics
            if "gate_counts" in m) / n
    In dual_rail runs all entries are {"sign_recovery":"dual_rail"}
    (C1's sibling bug), so the numerator sums to 0 over 0 terms but
    the denominator is n = n_steps.  Report becomes "avg_cx_gates = 0",
    which is wrong in two ways.  Divide by
        len([m for m in step_metrics if "gate_counts" in m])
    or set to None when no entries have the key.

C6. <DONE> mps_step returns only the array, not metrics.
    burgers_trotter.py:233 and 440-443: run_simulation never
    populates circuit_metrics for method="mps" (left as None).  MPS
    sweeps cannot be plotted for depth/gate/time comparisons against
    quantum_circuit.  If MPS is ever going into a side-by-side plot,
    it needs to return the same-shape metrics.

C7. <DONE> mps_step does not renormalize after post-selection.
    burgers_trotter.py:279-285: keeps only bond=|0> bitstrings, then
    sqrt(amps / total).  total here is the post-selected count, not
    the full shot count, so the returned u_prepared is already
    renormalized to the bond=0 subspace.  If bond-leakage is large
    this gives a pseudo-state that understates the MPS-truncation
    error.  For validation runs this is desirable (MPS-error isolated
    from bond-leakage noise), but the docstring should say so.

C8. <IGNORE - see other doc> hadamard_test_sign_circuit counts parser assumes a specific
    register layout.  Covered in F9-CODE-REVIEW.md.


## Redundancy / dead code

D1. <DONE> build_gradient_matrix / build_laplacian_matrix in
    burgers_classical.py are unused.  The shift_matrix path supersedes
    them.  Either delete or note that they are kept for reference
    against quimb MPO validation in UCAN (plan section 3.2).

D2. <IGNORE> n_pauli_terms recomputed at burgers_solver.py:267-270.  The
    per-step metrics already include n_pauli_terms
    (burgers_trotter.py:215), and build_evolution_hamiltonian is the
    expensive step (4^q Pauli-string evaluation).  Pull the first
    entry out of step_metrics, or compute once in run_simulation.

D3. <DONE> quantum_evolution_step in burgers_nonlinear.py is never called.
    Only exists as a wrapper for the module's __main__.  Inline into
    __main__ or remove.

D4. <DONE> step_fn dispatch in run_simulation is half-used.
    burgers_trotter.py:414-419 builds a dict of four callables but
    only the shift / quantum_exact branch uses it (via the else path).
    The other three branches call the function by name anyway.  Drop
    the dict, or normalize all four signatures so the dispatch
    actually does work.


## Consistency

K1. <DONE> Return types diverge across stepping functions.
    quantum_circuit_step returns (array, metrics); mps_step returns
    array; quantum_exact_step returns array; dual_rail_quantum_step
    returns (array, metrics).  Pick one.  Uniform (array, metrics)
    (with metrics=None or {} where not applicable) would simplify
    run_simulation's branching and close C6.

K2. <IGNORE> not shots vs shots == 0 used interchangeably.
    quantum_circuit_step (trotter:143) uses "not shots"; mps_step
    (trotter:258) uses "if not shots:".  Pick one convention.

K3. <DONE>shift_matrix returns complex dtype for a real +/-1 permutation.
    burgers_mpo.py:71.  Makes downstream "@ u" allocate a complex
    result from real u; harmless but wasteful at q>=5 per step.  Use
    real dtype unless a caller needs complex.

K4. <DONE> Hot path rebuilds AerSimulator per step in mps_step.
    burgers_trotter.py:268 creates one each call; quantum_circuit_step
    correctly reuses a pre-built backend hoisted out of run_simulation
    (trotter:410-412).  Hoist once for MPS too.


## Math / physics clarity

M1. <DONE> The quantum evolution is a fit to classical Euler, not an
    independent Burgers solver.  build_evolution_hamiltonian solves
    for Pauli coefficients such that
        e^{-iHdt} |u_norm> ~= |u_next_classical_norm>
    (Eq. 16 in the paper).  This is intentional -- the paper is about
    validating the pipeline, not about demonstrating quantum
    advantage.  But newcomers reading only the module header will
    miss this.  One paragraph at the top of burgers_nonlinear.py
    ("this Hamiltonian is fit per-step to match classical Euler;
    quantum_exact and shift must therefore agree to lstsq residual,
    not just to machine precision") would save a lot of re-derivation.

M2. <DONE> Hermiticity is not checked at runtime.  The __main__ in
    burgers_nonlinear checks it; the solver path does not.
    solve_pauli_coefficients enforces real coefficients so H is
    Hermitian by construction, but lstsq on a rank-deficient
    2*Re(S) can produce coefficients that satisfy the equation up to
    the residual, not exactly.  A single
        assert np.allclose(H_mat - H_mat.conj().T, 0)
    inside build_evolution_hamiltonian catches anyone who later
    extends the basis with non-Hermitian operators.  Cheap insurance.

M3. <DONE> quantum_circuit_step docstring undersells "shots>0".
    burgers_trotter.py:114-117 says "Sign information is lost in
    measurement; abs(amplitude) is used".  True for Option 1, but not
    for Options 2/3/4.  Update the docstring to name which options it
    supports and point at the sign_recovery parameter.


## Style / Best-Practices

S1. <DONE> Function-local import of compute_rhs_shift in
    burgers_sign_recovery.py.  Top-level is safe (no back-edge) and
    aligns with the "imports at top, grouped" rule.

S2. <DONE> quantum_circuit_step is ~125 lines with four branches.  Split
    into _statevector_branch, _sampling_hadamard_branch,
    _sampling_basic_branch for readability.

S3. <DONE> Duplication between quantum_exact_step and the statevector branch
    of quantum_circuit_step.  Both build H, initialize u_norm, apply
    the evolution, rescale by norm_next.  If H-build becomes a shared
    helper (needed for C1 fix anyway), the two collapse to a few
    lines each.

S4. <IGNORE> Module-level script at bottom of burgers_solver.py (~300 lines of
    logic under if __name__ == "__main__":).  Conventional to extract
    a main(args) function; makes the module importable without side
    effects and enables a minimal smoke test.

S5. <DONE - enough> __main__ sections of submodules save to /tmp/*.png and print.
    Fine for local debugging, but shadows the sweep's actual outputs.
    Best-Practices rule 12 permits /tmp writes, so this is allowed --
    noting only because a new user running "python burgers_mps.py"
    will get files they did not expect.

S6. <IGNORE> No logging; all progress via print(..., file=sys.stderr,
    flush=True).  Consistent across the module, so not a bug.  Worth
    a future harmonization with the rest of q8020 if that package
    uses logging.


## Testing

T1. <IGNORE> No unit tests in this subdirectory.  The __main__ blocks in each
    submodule are de facto smoke tests.  This is OK per
    Best-Practices rule 6 ("do not write your own test modules unless
    told").  But the F9-specific claims (Hadamard-test sign recovery,
    dual-rail correctness) are non-obvious enough that a small
    regression script (not a test module) that runs the four
    sign-recovery options on q=3 and asserts max-epsilon bounds would
    earn its keep.  Add once C1/C2 are fixed.


## Sweep / config

G1. sign_recovery_sweep_q5 uses shots=150000.  Plausible for Options
    1 and 3.  For Option 2 (Hadamard test, one extra qubit,
    controlled evolution) each of 2^(q+1)=64 bins sees ~2300 counts,
    std ~48, relative noise ~2%.  For Option 4 (dual-rail, two
    independent evolutions, subtract) noise compounds.  May be fine;
    once C1/C3 are fixed, check whether 150k is actually enough to
    resolve the claimed ordering of options.


## What is in good shape

- The Ran-2020 MPS prep with QR-completion for truncated states
  (burgers_mps.py:206-240) is a genuine extension beyond the paper,
  implemented carefully and with a clear comment on why.  Keep.
- Phase 1.1 BC fix is in: laplacian_central now uses one-sided at
  boundaries matching gradient_central (burgers_classical.py:34-41).
  Plan item closed.
- shift_matrix(bc="dirichlet") correctly zeros the wrap entries
  (burgers_mpo.py:75-80).  Phase 1.2 delivered.
- build_evolution_hamiltonian filters near-zero Pauli coefficients
  at 1e-15 before returning (burgers_nonlinear.py:170-175).  Reduces
  circuit terms and is a sensible default.
- compute_error is the paper's epsilon metric, handled correctly for
  near-zero classical norm (burgers_trotter.py:452-460).
- Metadata-fragment wiring is complete: case, results, analysis,
  artifacts.  q8020-sweep harvests without custom code.


## Priority order for fixes

1. C1  dual-rail Hamiltonian -- changes what Option 4 plots mean.
2. C3  sign-convention comment + regression -- prevents a future
       "correctness fix" from introducing a real bug.
3. C5 + D2  CX-gate averaging and n_pauli_terms double-compute --
            plot numbers become trustworthy.
4. C6 + K1  uniform return shape; MPS metrics -- unblocks
             cross-method comparison plots.
5. C2  Hadamard-test docstring caveat -- doc-only, cheap.
6. Everything else (dead code, style) -- opportunistic.

Net: the module is close to done on its stated scope.  The above is
a punch list, not a rewrite.
