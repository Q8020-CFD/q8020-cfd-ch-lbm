"Chunking" here is a way to run a long time evolution as a sequence of shorter circuits, stitched together by reading the
  state out to classical numbers and re-loading it — instead of building one giant circuit for all the time steps.

  The problem it solves

  This is a shots-based solver. Every time step uses a non-unitary heat propagator that's block-encoded with an ancilla, and
  you only keep the runs where the ancilla measures |0⟩ ("post-selection"). The probability of keeping a run is p_success
  per step.

  If you build one circuit that does all n_steps at once, the survival probabilities multiply: p_total = p₁ · p₂ · ... · pₙ.
  After many steps that product gets tiny, so almost all your shots get thrown away and the signal collapses. The circuit
  also gets very deep.

  What chunking does

  Split n_steps into n_chunks pieces of chunk_size steps each (line 1081: n_chunks = n_steps_total // chunk_size). Then loop
  (the for chunk_idx loop, line 1094):

  1. Prep — load the current amplitudes psi_current into a fresh circuit using the MPS state-prep (line 1100-1104).
  2. Evolve — run only chunk_size heat steps (line 1112-1130).
  3. Measure + post-select — keep only the all-ancilla-|0⟩ shots (post_select_counts, line 1192).
  4. Read out to classical numbers — turn the surviving counts back into an amplitude vector psi_new (line 1207-1215). Note
  √(count/n_kept) — it reconstructs magnitudes from measurement frequencies.
  5. Re-normalize and carry forward — psi_current = psi_new, and track the discarded normalization in cumulative_norm *= 
  √p_success (line 1214) so the final scale is still correct.

  Then the next chunk starts fresh from that re-loaded state.

  The key trade-off

  The comment at line 1056-1058 says it plainly:

  ▎ Between chunks, classically read out post-selected amplitudes and re-prep as fresh IC. No classical PDE physics — only 
  ▎ amplitude IO.

  - Win: p_success resets to ~1 at the start of each chunk instead of decaying over all n_steps. Shallower circuits, far
  less post-selection loss.
  - Cost / honesty caveat: the readout at step 4 only recovers magnitudes (√(count/n_kept)), so sign/phase information is 
  lost at every chunk boundary. For the Cole-Hopf φ field, which is positive (φ = exp(...) > 0), that's usually fine — but
  it's the reason this is "amplitude IO" and not a free lunch. You're paying with a classical measurement-and-reprepare
  bottleneck between chunks, which on real hardware reintroduces full state-prep cost each chunk.

  How you reach it

  Set evolution_mode="chunked" with a chunk_size (line 2023). It requires:
  - n_steps divisible by chunk_size (line 2025),
  - snapshot steps aligned to chunk boundaries (line 2030),
  - simulator only — hardware chunking raises NotImplementedError (line 1065-1069).

  The alternatives are "single" (_run_shots_batch, one circuit per snapshot, full depth) and the hadamard_per_bin readout.
  Chunking sits between them: it trades a deep one-shot circuit for many shallow circuits plus classical hops in between.



hadamard_per_bin trades doing N circuits per snapshot for the ability to read out signed amplitudes with
  good signal-to-noise — built for the low-viscosity regime where the default magnitude-only readout breaks down.


  The trade-off vs. direct

  ┌───────────────────────┬─────────────────────┬─────────────────────────────────────────────────┐
  │                       │       direct        │                hadamard_per_bin                 │
  ├───────────────────────┼─────────────────────┼─────────────────────────────────────────────────┤
  │ Circuits per snapshot │ 1                   │ N = 2^q                                         │
  ├───────────────────────┼─────────────────────┼─────────────────────────────────────────────────┤
  │ Recovers sign?        │ No (magnitude only) │ Yes (signed real part)                          │
  ├───────────────────────┼─────────────────────┼─────────────────────────────────────────────────┤
  │ Best regime           │ normal ν            │ low ν / weak signal                             │
  ├───────────────────────┼─────────────────────┼─────────────────────────────────────────────────┤
  │ Cost                  │ cheap               │ N× more circuits + an annotated controlled gate │
  └───────────────────────┴─────────────────────┴─────────────────────────────────────────────────┘


  ┌──────────────────────────────────┬─────────────────────────────────┬────────────────────────────┐
  │                                  │     CH (cole_hopf_circuit)      │    QLBM (qlbm_circuit)     │
  ├──────────────────────────────────┼─────────────────────────────────┼────────────────────────────┤
  │ Multi-step circuit?              │ Yes (single mode)               │ No — always 1 step/circuit │
  ├──────────────────────────────────┼─────────────────────────────────┼────────────────────────────┤
  │ Chunking knob?                   │ Yes (evolution_mode/chunk_size) │ N/A (always chunk_size=1)  │
  ├──────────────────────────────────┼─────────────────────────────────┼────────────────────────────┤
  │ Needs sign recovery?             │ No (φ ≥ 0)                      │ Yes when f_i < 0           │
  ├──────────────────────────────────┼─────────────────────────────────┼────────────────────────────┤
  │ Sign + per-step readout coexist? │ No (guard line 2000)            │ Yes (line 469 × 482-489)   │
  └──────────────────────────────────┴─────────────────────────────────┴────────────────────────────┘




  QLBM — stop-motion, one tick at a time

  Lattice Boltzmann doesn't track the velocity directly. It tracks little "populations" of
  fictitious particles sitting on a grid, and each tick does two things:

  1. Collide: at every cell, nudge the populations toward their local equilibrium (this is where
  the nonlinear physics lives).
  2. Stream: shuffle populations one cell left/right.

  Then you add up the populations to recover u(x) — that's your frame.

  The crucial constraint: streaming moves things exactly one cell per tick, so one tick = one
  grid-spacing of time. Over the whole run that's only ~10 ticks at this resolution. It's
  genuinely stop-motion: frame N is posed from frame N−1, in sequence, and there are only ~10 real
  poses. (The padding that used to fake it up to 98 frames is the "fairness" thing we just
  removed.) On a quantum circuit, each tick is its own little unitary that gets rebuilt from the
  current state, run, and measured before the next tick.

  Cole-Hopf (CH) — change the problem so you can fast-forward

  Cole-Hopf does something clever: a change of variables (u → φ = exp(−∫u/2ν)) that turns the
  nonlinear Burgers equation into the plain linear heat equation (a bump just smoothly diffusing).
  Solve the easy linear problem, then transform back to u.

  Why that matters for the movie: a linear equation has a "time machine" property — you can write
  down one operator that jumps straight to any future time. You don't have to crawl tick by tick.
  That's what splits into two flavors:

  Unchunked CH ("single") — each page computed fresh from the start.
  To get the frame at t = 0.2, take the initial state and apply one operator that fast-forwards
  straight to 0.2. The frame at 0.1 and the frame at 0.2 are independent computations, both
  launched from t = 0 — like plugging different t values into a formula. No frame depends on any
  other frame.

  The catch: to run this on a quantum circuit you have to smuggle the (non-unitary) evolution in
  via an extra "ancilla" qubit and then keep only the runs where the ancilla reads 0
  (post-selection). The longer the single jump, the more the useful signal bleeds away, so fewer
  runs survive → noisier readout for the late frames.

  Chunked CH — fast-forward in short hops, with a photocopy at each handoff.
  Instead of one big jump, break the time axis into chunks (here 7 steps each): evolve the first
  chunk, measure and read out the state, re-prepare that measured state as the new starting point,
  evolve the next chunk, and so on. Like sprinting 7 steps, snapping a Polaroid, then using the
  Polaroid as your new start line.

  Trade-off, and this is exactly what your two-CH comparison is testing:
  - Upside: each hop is short, so the per-circuit post-selection stays healthy (more surviving
  runs) and the circuits stay shallow.
  - Downside: it's now sequential again (each chunk starts from the previous chunk's measured
  result), and every handoff is a lossy photocopy — measurement + reconstruction error that
  accumulates chunk after chunk. Snapshots can also only land on chunk boundaries.
  
  So: unchunked = one clean long jump per frame but noisier reads at late times; chunked =
  depth-bounded and better per-hop signal but error piles up across handoffs.

  Side by side

  ┌────────────┬──────────────────────────────────┬─────────────────────┬────────────────────┐
  │            │     How frames are generated     │  Frame depends on   │   Genuine frame    │
  │            │                                  │      previous?      │       budget       │
  ├────────────┼──────────────────────────────────┼─────────────────────┼────────────────────┤
  │ QLBM       │ Collide+stream, one grid-cell    │ Yes (sequential,    │ ~10                │
  │            │ tick at a time                   │ stop-motion)        │ (lattice-locked)   │
  ├────────────┼──────────────────────────────────┼─────────────────────┼────────────────────┤
  │ CH         │ One direct fast-forward from t=0 │ No (each            │ Any save step      │
  │ unchunked  │  to each snapshot time           │ independent)        │                    │
  ├────────────┼──────────────────────────────────┼─────────────────────┼────────────────────┤
  │ CH chunked │ Fast-forward in 7-step hops,     │ Yes (sequential, by │ Chunk boundaries   │
  │            │ measure & re-prep between        │  chunk)             │ only               │
  └────────────┴──────────────────────────────────┴─────────────────────┴────────────────────┘

  And one thing common to all three: at the end of every frame you have quantum amplitudes, and
  you have to turn those into the actual velocity line. QLBM reconstructs the populations from
  shot counts (and needs the Hadamard-test trick to recover signs, since populations can go
  negative). CH reads out φ from counts (no sign problem — φ is always positive) and then undoes
  the log transform to get u.

  That's why, in the figure, the CH curve tends to look smoothly diffusive (it is solving a
  smoothed linear problem) while the QLBM/LBM curve fights the nonlinearity directly and is more
  prone to steepening and blowing up.


viscosity

  ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²

  ν (nu) scales the diffusive term ∂²u/∂x². Larger ν → more smoothing, smoother/wider shock fronts;
  smaller ν → sharper steepening toward the inviscid limit. In the sweep --nu = 1e-2 is, per the
  archived comment, "well-behaved viscosity, well inside the regime where cole_hopf_circuit
  shots+post-selection is healthy" — the Cole–Hopf circuit pathway gets numerically pathological
  below ν < 1e-3.



  Background: Cole–Hopf turns Burgers into the linear heat equation for φ, which the circuit
  encodes as a quantum state ψ. ψ0 is the encoded initial condition; ψ(t) is that state evolved
  forward to time t by the propagator.

  So "one ψ0→ψ(t) per saved step" means: for each saved snapshot time t_k (the steps 0, 7, 14, …,
  98), the solver builds one independent circuit that evolves directly from the initial state ψ0 
  all the way to ψ(t_k) in a single shot. There's no chaining between snapshots — every saved frame
  is its own fresh ψ0 → ψ(t_k) propagation.

  Contrast with the other case in the same toml, --evolution-mode = measure_reprepare (segmented):

  - single: ψ0 ──U(t_k)──▶ ψ(t_k) independently for each k. No intermediate measurement, no
  post-selection chaining. One direct evolution per saved step.
  - measure_reprepare: evolve in chunks of --segment-size = 7; at each segment boundary you
  measure, post-select, and re-prepare a new initial state from the measured result, then continue.
  State is threaded through measurements segment by segment.

  Why both are in the sweep: comparing single vs measure_reprepare isolates how much accuracy the 
  segmented post-selection/re-preparation chaining costs relative to the clean single-window
  baseline. single is "more correct" numerically (no per-segment measurement collapse / re-prep
  error) but each circuit is deeper (full-time evolution), whereas measure_reprepare bounds
  per-circuit depth at the price of that chaining error.




  1. n_steps — derived (the one computed quantity).
  dx = 1/2^q ,  dt = cfl·dx ,  n_steps = round(t_end / dt)
  ∝ 2^q. q=5 → 98. q=4 → ~49.
  
  2. --segment-size — must divide n_steps (Cole–Hopf measure_reprepare only).
  The segmented evolution tiles the full run into equal chunks, so you need n_steps % segment_size 
  == 0, or the last segment doesn't land on n_steps.
  - q=5: 98 = 14 × 7 → segment_size=7 gives 14 segments. ✓
  - q=4: 49 = 7 × 7 → segment_size=7 gives 7 segments. ✓ (still works)

  3. --save-every — must be a multiple of segment_size.
  In measure_reprepare, a snapshot can only be emitted at a segment boundary (the
  measure-and-reprepare points). So every step you want to save must itself be a multiple of
  segment_size. Setting save_every = segment_size = 7 is the simplest choice that guarantees this:
  saved steps 0, 7, 14, … are exactly the segment boundaries.




The three propagators

  qft-diagonal (the argparse default) — burgers_cole_hopf_circuit.py:739
  Diffusion is diagonal in Fourier space, so it diagonalizes the heat operator with a QFT (or a DCT
  for Neumann), applies a conditional rotation, inverts the transform. One ancilla.
  - ✅ Exact (no truncation — the heat equation really is diagonal under QFT), most
  circuit-efficient (QFT ~O(q²) gates), genuinely quantum-native.
  - ❌ Only periodic or Neumann BC (line 1951 errors otherwise), binary encoding only (line 1969),
  no source forcing.

  dense-block (your current choice) — line 757
  Builds the exact dense propagator expm(ν·L·Δt), eigendecomposes it classically, then
  block-encodes via controlled-RY rotations.
  - ✅ Exact, supports any BC including Dirichlet, any encoding, and source forcing (the V
  potential term). Transpiles to basis gates → fast shot sampling (~2 ms/shot).
  - ❌ Requires a classical eigh of a 2^q×2^q matrix → doesn't scale (fine for q≤5–6, hopeless at
  large q). It's a faithful circuit playback of the exact answer, not a scalable quantum algorithm.

  lcu — line 789
  Linear-Combination-of-Unitaries block-encoding of the propagator's Taylor series.
  - ✅ Genuinely scalable in principle — no classical eigendecomposition, so it's the only one
  that's a "real" quantum algorithm at large q.
  - ❌ Taylor-truncated (approximate, taylor_order), needs m ancillas, and its per-step ancilla 
  resets defeat Aer's shot-sampling (~27 ms/shot → ~19 h at 150k shots, per your archived note).
  
  So what's "best"?

  ┌───────────────────────────────┬──────────────┬────────────────────────────────────────────┐
  │       If your goal is…        │    Best      │                    Why                     │
  │                               │  propagator  │                                            │
  ├───────────────────────────────┼──────────────┼────────────────────────────────────────────┤
  │ This sweep (gaussian,         │ dense-block  │ Dirichlet rules out qft-diagonal; it's     │
  │ Dirichlet, q=5, 150k shots)   │ ✅           │ exact and shot-samples fast. The right     │
  │                               │              │ validation tool.                           │
  ├───────────────────────────────┼──────────────┼────────────────────────────────────────────┤
  │ Periodic/Neumann BC, no       │              │ Exact and the most circuit-efficient; the  │
  │ source (e.g. the paper's sine │ qft-diagonal │ natural quantum-native choice              │
  │  IC)                          │              │                                            │
  ├───────────────────────────────┼──────────────┼────────────────────────────────────────────┤
  │ Demonstrating a scalable      │              │ Only option that avoids the classical      │
  │ quantum propagator at large q │ lcu          │ eigendecomposition — accept truncation +   │
  │                               │              │ slowness                                   │
  ├───────────────────────────────┼──────────────┼────────────────────────────────────────────┤
  │ Forced Burgers (nonzero       │ dense-block  │ Only one that supports the potential term  │
  │ --source)                     │              │ (line 1963)                                │
  └───────────────────────────────┴──────────────┴────────────────────────────────────────────┘

  Bottom line for your gaussian toml: dense-block is correct and best. The chain is: you chose
  Dirichlet (because the gaussian's nonzero mean breaks periodic Cole–Hopf), Dirichlet rules out
  qft-diagonal, and lcu is too slow for 150k shots and only approximate — so dense-block is the
  exact, fast, BC-compatible choice. Your archived comment captured exactly this.

  Two things worth knowing as you present it:
  - dense-block being exact-but-classically-assisted means it's a fidelity ceiling / validation 
  baseline, not a claim of quantum advantage. If a reviewer asks "is this a scalable quantum
  algorithm?", the honest answer is "no — lcu is the scalable pathway; dense-block validates the
  circuit against the exact propagator at small q."
  - If you ever switch the demo to periodic BC (sine IC like the paper), switching to qft-diagonal
  would be both more efficient and still exact — a nice point to make about matching propagator to
  BC.



    In the MPS/MPO picture for this Cole–Hopf circuit path:

  - MPS = the state. The encoded object is the heat field φ (the Cole–Hopf variable) as a q-qubit
  amplitude vector, optionally bond-dim-truncated via the Ran-2020 prep. That's the matrix-product
  state.
  - The operator ("MPO") = the heat propagator P = exp(ν·Δt·∂ₓₓ). LCU is one implementation of that
  linear operator — a Taylor block-encoding expressed as a linear combination of unitaries. So
  yes, LCU sits in the operator slot. dense-block (eigendecomposition) and qft-diagonal are two
  other implementations of the same propagator. Calling it "the MPO part" is fair as a role
  analogy, with one caveat: LCU is a block-encoding/circuit, not literally a bond-dimension
  operator train. A true MPO (the tensor-train operator) only actually appears in this repo on the
  TEBD path ("Dense-H → MPO → MPS", Zaletel W-II), not on the LCU path.
  - Cole–Hopf is not the operator. CH is the nonlinear change of variables u ↔ φ = exp(−∫u/2ν).
  It's the linearization — the thing that makes a linear propagator exist at all — applied as
  classical pre/post-processing around the evolution (and at each segment boundary under
  measure_reprepare). It defines what variable the MPS encodes, rather than acting as an operator
  on the encoded state.

  So the accurate phrasing: the MPS holds φ; the LCU block-encoding is the operator (MPO-role) 
  applied to it; Cole–Hopf is the linearizing transform that brackets the whole evolution, sitting 
  outside the MPS/MPO pair rather than being the MPO.


Why skip:
  1. Aer doesn't need it — AerSimulator simulates arbitrary high-level gates (UnitaryGate, StatePreparation,
  multi-controlled composites) directly by applying their matrices, so lowering them to a cx/rz/sx basis is pointless
  work for a sim.
  2. It dodges a crash — the comment at the branch says transpiling the LCU circuit's multi-controlled SELECT gates can
  hit a Qiskit qs_decomposition segfault. Skipping avoids that.
  
  Important catch given what you're running: that branch is gated on propagator == "lcu". Your Gaussian sweep uses
  --propagator dense-block, which is not skipped — it transpiles every segment. So a log line saying (skip transpile — 
  Aer+LCU) means that run was an LCU run, not one of your dense-block CH cases. If you saw this in the gaussian sweep
  output, double-check which case/config produced it.

  How it ties into the metrics we just added:
  - Timing: skipped segments correctly report transpilation_time_s = 0 (no transpile happened) — so the rollup stays
  honest.
  - Circuit stats: the skip's t_info has no after key, so circuit_depth/gate_counts/n_qubits come back None for LCU+Aer
  segments — i.e., LCU runs won't get gate/depth metrics (there's nothing transpiled to count). Dense-block is unaffected
  and reports real counts. If you ever want gate metrics for the LCU path too, we'd add a circuit_stats_in_basis call
  there (like I did for qlbm) — but that's the same qs_decomposition that segfaults, so it'd need care.
