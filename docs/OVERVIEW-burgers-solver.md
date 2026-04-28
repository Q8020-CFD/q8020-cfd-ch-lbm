# OVERVIEW — Burgers solver

This document describes what the `q8020-mps-burgers` solver does,
the various evolution methods it offers, the encoding and propagator
options that vary per method, and how the package is layered on top
of `solverfw` (see
[SPEC-solverfw.md](../../q8020-cfd-metautil/docs/SPEC-solverfw.md)
for the framework itself).

## 1. What the solver solves

One-dimensional viscous Burgers equation with optional source:

```
∂u/∂t + u·∂_x u = ν · ∂²_x u + g(x, t)
```

with periodic or homogeneous-Dirichlet (`u=0`) boundary conditions.
Initial conditions: `sine` (single mode) or `multimode` (random
sum of low-wavenumber modes). Sources: `sine` (g(x,t) = sin(2πx)·
cos(2πt), the Murali/Meena AIAA-2026 reference test problem) or
`none`.

The grid is N = 2^q points uniform on [0, 1]. `q` controls grid
resolution and the qubit count for quantum methods.

The CLI is [`burgers_solver.py`](../src/burgers_solver.py); it
always runs a classical Forward-Time Central-Space (FTCS) reference
alongside the chosen method and reports L2 error against it.

Quantum solutions may have pre-processing, but once rolling, do not refer back to the classical solution for steerage. The solver does have additional runtime modes which permit a quantum-classical hybrid for comparison to the quantum-centric solver variants. 


## 2. The method variations

`--method` selects the evolution scheme. Methods divide into two
families: those that march `u` directly, and those that linearize
via Cole-Hopf and march `φ = exp(−U/(2ν))` instead.

**Legend:**
🔧 classical — no quantum objects
🔬 quantum with classical mirror — operator rebuilt from classical state each step
🔭 near-pure quantum — classical mirror confined to operator construction; evolution is fully quantum
⚛ pure quantum — no classical mirror in the time loop

### Direct-`u` family (six)

- 🔧 **`shift`** *(classical)* —
  Explicit forward-Euler with central-difference shift-operator FD
  on `u`. Pure classical baseline; no quantum objects at all.
  O(N) per step where N = 2^q.

- 🔬 **`quantum_exact`** *(quantum, statevector)* —
  At each step, freeze the nonlinear Burgers RHS at the current
  state, fit a Hermitian operator `Â` to it via Pauli decomposition,
  then apply `expm(-i·Â·dt)` (scipy dense matrix exponential). No
  Trotter error, but requires a classical mirror (the frozen RHS)
  every step. Diagnostic / upper-bound reference for the circuit
  methods. The Pauli decomposition builds a (4^q × N) matrix and
  solves a least-squares system — O(4^q · N) per step, which is the
  bottleneck. `expm` adds O(N^3). OOMs at q >= 6.

- 🔬 **`quantum_circuit`** *(quantum, circuit)* —
  Same per-step Pauli Hamiltonian as `quantum_exact`, but evolved
  via a Suzuki-Trotter circuit instead of exact `expm`. Trotter
  order (1 or 2) and repetition count are configurable. Supports
  statevector and shots modes, with optional sign recovery. The
  Pauli decomposition is still O(4^q · N) per step; the circuit
  construction and simulation replace the O(N^3) `expm`.

- 🔬 **`mps`** *(quantum/tensor)* —
  Encode the current `u` into an MPS circuit using the Ran 2020
  state-prep decomposition (with optional bond-dim truncation),
  simulate the circuit to obtain the quantum state, then apply the
  exact dense Hamiltonian evolution. Useful for studying MPS
  compression fidelity independently of time-integration error.
  Uses the same O(4^q · N) Pauli Hamiltonian as `quantum_exact`.

- 🔬 **`tebd`** *(classical/tensor)* —
  Build the dense Hermitian evolution generator directly from shift
  operators — O(N^2), bypassing the O(4^q) Pauli decomposition.
  Compute `expm` once per step — O(N^3). Convert to an MPO via
  `quimb.MatrixProductOperator.from_dense`, and apply the MPO to
  the MPS state with bond-dim truncation — O(N · chi^3) per step.
  Multi-step delegating path (owns its own loop). Scales to q=12
  (N=4096); the bottleneck is the dense `expm`, not the MPO ops.

- 🔭 **`tebd_circuit`** *(quantum, circuit)* —
  TEBD-style quantum circuit: MPS state-prep followed by a W-II
  (Zaletel) gate layer that encodes one evolution step. Uses the
  same O(N^2) dense Hamiltonian as `tebd` (no Pauli decomposition).
  Per-step path through the framework loop. Not pure-quantum: the
  Hamiltonian is rebuilt from the current `u` each step (classical
  mirror). However, the *evolution itself* — state-prep, gate
  layer, measurement — is entirely quantum. The classical mirror
  is confined to operator construction, not state readout or
  steerage.

### Cole-Hopf family (two methods, three propagators)

- 🔧 **`cole_hopf`** *(classical/tensor)* —
  Apply the Cole-Hopf transform `u → phi = exp(-U/(2*nu))`,
  converting nonlinear Burgers into the linear heat equation. Build
  the heat propagator `exp(nu*L*dt)` once as a dense matrix —
  O(N^3) one-time cost — convert to MPO, and reuse it every step
  (the propagator is state-independent). Per-step cost is
  O(N · chi^3) for the MPO-on-MPS apply. Invert back to `u` via
  log-domain finite differences. Multi-step delegating path.

- ⚛ **`cole_hopf_circuit`** *(quantum, circuit)* —
  Same Cole-Hopf linearization, but the heat equation is marched as
  a quantum circuit. Initial `φ` amplitudes are loaded onto qubits
  via the Ran 2020 MPS-to-circuit state-prep pipeline (the MPS is
  used only for encoding, not for evolution). In chunked mode the
  MPS prep is repeated at each chunk boundary (decode counts,
  re-encode). Two propagator choices: `qft-diagonal` (QFT ->
  conditional-Ry on Fourier eigenvalues -> inverse QFT; O(q^2)
  gates; periodic BC only) or `dense-block` (eigendecomposition
  encoded as a block of a unitary with one ancilla + post-selection;
  O(N^3) one-time build; supports any BC and source forcing) or
  `lcu` (Taylor-expansion LCU block-encoding of the heat propagator
  using S+/S- shift-operator primitives; periodic BC; gate count
  O(M*q) per step where M = taylor_order; see
  [SPEC-F3-LCU-method.md](SPEC-F3-LCU-method.md)).
  Pure-quantum inside the time loop. Multi-step delegating path.
  Supports statevector, shots, and chunked evolution modes.

The classical FTCS baseline that runs alongside every method is
implemented in [`burgers_classical.py`](../src/burgers_classical.py)
(`solve_burgers`) — that is the reference, *not* one of the switchable
methods.

## 3. Per-method options

Most CLI flags only apply to a subset of methods. You might see some "not supported" errors for odd combinations.

### Encoding (`--encoding {binary,gray}`)

Used by `cole_hopf_circuit`. `binary` is index-aligned (default);
`gray` uses reflected Gray code permutation `π(i) = i ^ (i >> 1)`
on the Laplacian/propagator matrix. The encoding choice affects
which two-qubit gates are nearest-neighbour after transpilation.

### Propagator (`--propagator {qft-diagonal,dense-block,lcu}`)

Used by `cole_hopf_circuit`. Selects the heat-equation circuit
construction:

- `qft-diagonal`: QFT → conditional-Ry on momentum eigenvalues →
  inverse QFT. Exploits the fact that the Laplacian is diagonal in
  the Fourier basis. Periodic BC only.
- `dense-block`: build `M = ν·L·dt`, exponentiate via
  eigendecomposition, encode as a block of a unitary using one
  ancilla qubit + post-selection on `anc=0`. Supports any BC and
  any diagonal potential (used by source forcing).
- `lcu`: Taylor-expansion LCU (Linear Combination of Unitaries)
  block-encoding of `exp(ν·L·dt)` using S+/S- shift-operator
  primitives. Ancilla count = ceil(log2(K)) where K is the number
  of distinct shift terms. Periodic BC only. Controlled by
  `--lcu-taylor-order` (default 4). With `--source` enabled, the
  per-step circuit becomes a Strang sandwich
  `exp(-V·dt/2) · LCU_heat · exp(-V·dt/2)` with two extra V
  ancillas (one per half-step), introducing O(dt²) Strang error.
  See [SPEC-F3-LCU-method.md](SPEC-F3-LCU-method.md) and
  [SPEC-F3-LCU-source-forcing.md](SPEC-F3-LCU-source-forcing.md).

Source forcing (`--source sine`) is currently supported on
`dense-block` and `lcu`. `qft-diagonal` + source raises
`NotImplementedError`. See
[SPEC-source-forcing.md](SPEC-source-forcing.md) and
[SPEC-F3-LCU-source-forcing.md](SPEC-F3-LCU-source-forcing.md).

### Trotter order / reps (`--trotter-order`, `--trotter-reps`)

`quantum_circuit` only. Suzuki-Trotter order (1 or 2) and number of
sub-step repetitions per timestep.

### MPS bond dimension (`--bond-dim`, `--mps-threshold`)

Used by `mps`, `tebd`, `tebd_circuit`, and the state-prep stage of
`cole_hopf_circuit`. `--bond-dim None` keeps full rank;
`--mps-threshold` is a singular-value cutoff during compression.

### Sign recovery (`--sign-recovery`)

Applies to `quantum_circuit`, `mps`, `tebd_circuit` when reading
out via shots. Methods that go through Cole-Hopf don't need it
because `φ > 0` by construction. Choices:

- `none` — no sign recovery (only valid when the true solution is
  known non-negative).
- `classical_oracle` — recover sign from the classical FTCS
  reference (diagnostic; cheats by definition).
- `hadamard_test` — quantum sign extraction via interference.
- `dual_rail` — encode `±` in a paired register.

### Shots and backend

`--shots N` (0 means statevector). `--backend-type {sim,fake,
hardware}`, `--backend NAME`, `--t1`, `--t2`, `--coupling-map`,
`--seed`, `--optimization-level`. See
[SPEC-shots-backend.md](SPEC-shots-backend.md).

### Evolution mode (`--evolution-mode {single,chunked}`, `--chunk-size`)

`cole_hopf_circuit` shots-mode only. `single` = one big circuit
with `n_steps` inlined step layers (today's default). `chunked` =
break the evolution into K-step segments, read out and re-prep
amplitudes between segments. Trades depth-per-circuit against
shot-noise compounding. See
[SPEC-chunked-evolution.md](SPEC-chunked-evolution.md).

### Source forcing (`--source {sine,none}`)

All methods accept it. The `dense-block` path threads it through
to a per-step `V(x,t)` potential in the heat propagator (see
[SPEC-source-forcing.md](SPEC-source-forcing.md)). Other paths
inject it directly into the `u` RHS or the φ-equation.

### Time-window (`--shock-pct`, `--n-steps`)

Either a percentage of the inviscid shock-formation time
`t_shock = 1 / max|du₀/dx|` (resolves to an `n_steps` from the
fixed CFL-derived dt), or an explicit step count.

## 4. How the methods map to solverfw

The package is wired to the framework in
[`burgers_fw.py`](../src/burgers_fw.py).

### 4.1 The pieces

- **Config**: `BurgersConfig(SolverConfig)` adds every parameter in
  §3 as a dataclass field. `describe()` returns a serialisable
  summary.
- **Grid**: `Grid1D.from_qubits(q, bc=...)`. Interior of the grid
  depends on BC: Dirichlet includes both endpoints (so `u=0` is
  satisfied by the sine IC); periodic excludes the right endpoint
  since `x=0` and `x=1` are identified.
- **State**: `DenseState`. Burgers' state is a 1-D float array of
  length N — the protocol's default impl is sufficient.
- **SpatialOperator**: `ShiftFD` — central-difference RHS using
  shift operators on `u`. This is *only* used by methods in the
  per-step family below; delegating methods build their own
  spatial structures internally.
- **TimeIntegrator**: a different concrete subclass per method
  (table below).
- **MainLoop**: standard, unchanged from the framework.

### 4.2 Per-step integrators (use the framework loop)

These five methods plug into `MainLoop` the normal way — `step()`
advances one timestep, framework owns the loop:

| `--method` | Integrator class | Step function it calls |
|---|---|---|
| `shift` | `ShiftEulerIntegrator` | `burgers_trotter.shift_euler_step` |
| `quantum_exact` | `QuantumExactIntegrator` | `burgers_trotter.quantum_exact_step` |
| `quantum_circuit` | `QuantumCircuitIntegrator` | `burgers_trotter.quantum_circuit_step` (or `dual_rail_quantum_step` if `--sign-recovery dual_rail`) |
| `mps` | `MPSIntegrator` | `burgers_trotter.mps_step` |
| `tebd_circuit` | `TEBDCircuitIntegrator` | `burgers_trotter.tebd_circuit_step` |

Each integrator pulls the source value at time `t` from
`config._source_fn(grid.xc, t)` and forwards it to the underlying
step function along with `bc`, `nu`, `dt`. Returns
`(DenseState(u_new), metrics_dict)`.

### 4.3 Delegating integrators (own their own loop)

Three methods carry tensor / circuit state across timesteps, or
pre-build a propagator once and reuse it. They do not fit the
per-step model. They use the delegating-integrator idiom from
[SPEC-solverfw.md](../../q8020-cfd-metautil/docs/SPEC-solverfw.md)
§5: subclass `TimeIntegrator`, but in `step()` run the *entire*
multi-step simulation internally and return all snapshots via
sentinel keys in the metrics dict.

| `--method` | Integrator class | Inner driver |
|---|---|---|
| `tebd` | `TEBDIntegrator` | `burgers_tebd.run_tebd_simulation` |
| `cole_hopf` | `ColeHopfIntegrator` | `burgers_cole_hopf.run_cole_hopf_simulation` |
| `cole_hopf_circuit` | `ColeHopfCircuitIntegrator` | `burgers_cole_hopf_circuit.run_cole_hopf_circuit_simulation` |

The set of delegating methods is recorded as
`_DELEGATING_METHODS = {"tebd", "cole_hopf", "cole_hopf_circuit"}`
in `burgers_fw.py`.

### 4.4 The dispatcher

`run_simulation_fw(config, grid, u0, source_fn)` attaches
`source_fn` to `config._source_fn`, builds the right integrator via
`make_integrator(config)`, and either:

- **Delegating method**: calls `integrator.step()` once with full
  `n_steps`, pulls `_delegated_solutions` and `_delegated_metrics`
  out of the returned metrics dict, returns those.
- **Per-step method**: hands the integrator to `MainLoop().run()`
  and returns its output.

In either case the public return shape is
`(solutions: list[np.ndarray], step_metrics: list[dict] | None)` —
identical to the framework contract, identical across all eight
methods.

### 4.5 Backend management

For the three quantum-circuit methods that shoot at a backend
(`quantum_circuit`, `mps`, `tebd_circuit`) the backend is built
once in `make_integrator()` via
`q8020_cfd_qutil.backend.get_backend(...)` and stored on the
integrator instance. `cole_hopf_circuit` builds its backend lazily
inside the integrator's `_run_all` because it may not be needed
(shots=0 path is statevector-only).

## 5. Directory map of the implementation

```
q8020-mps-burgers/
├── input/
│   └── burgers_quantum.toml         # q8020 sweep cases
└── src/
    ├── burgers_solver.py            # CLI entry point + FTCS reference
    ├── burgers_fw.py                # solverfw bindings (this doc §4)
    ├── burgers_classical.py         # FTCS solve_burgers + source_term_sine
    ├── burgers_nonlinear.py         # compute_rhs_shift used by ShiftFD
    ├── burgers_trotter.py           # per-step quantum kernels
    ├── burgers_mps.py               # Ran 2020 MPS prep + helpers
    ├── burgers_mpo.py               # heat MPO for the cole_hopf classical path
    ├── burgers_tebd.py              # TEBD multi-step driver
    ├── burgers_cole_hopf.py         # CH classical pipeline + run_cole_hopf_simulation
    ├── burgers_cole_hopf_circuit.py # CH quantum-circuit pipeline (full,sv,shots,chunked)
    ├── burgers_potential.py         # V(x,t) for source-forced CH (SPEC-source-forcing)
    ├── burgers_encoding.py          # binary/gray encoding helpers
    ├── burgers_sign_recovery.py     # F9 sign-recovery strategies
    └── burgers_postprocess.py       # output writers, q8020 metrics dump
```

Method-to-module crosswalk for "where does the actual physics
live":

| `--method` | Module |
|---|---|
| `shift` | `burgers_classical.py` (FTCS too) + `burgers_nonlinear.py` |
| `quantum_exact` | `burgers_trotter.py::quantum_exact_step` |
| `quantum_circuit` | `burgers_trotter.py::quantum_circuit_step` |
| `mps` | `burgers_trotter.py::mps_step` + `burgers_mps.py` |
| `tebd` | `burgers_tebd.py::run_tebd_simulation` |
| `tebd_circuit` | `burgers_trotter.py::tebd_circuit_step` |
| `cole_hopf` | `burgers_cole_hopf.py::run_cole_hopf_simulation` |
| `cole_hopf_circuit` | `burgers_cole_hopf_circuit.py::run_cole_hopf_circuit_simulation` |

## 6. Run shape — a typical sweep case

`input/burgers_quantum.toml` contains q8020-sweeper cases. A
representative one for the Cole-Hopf circuit on a forced run:

```toml
[cole_hopf_circuit_forced_q5_smoke]
"--method" = "cole_hopf_circuit"
"--propagator" = "dense-block"
"--ic" = "sine"
"--source" = "sine"
"--nu" = 0.1
"--cfl" = 0.1
"--shock-pct" = 100.0
"--q" = 5
"--shots" = 50000
"--backend-type" = "sim"
"--seed" = 42
"--save-every" = 1
_group_postproc = "python ./q8020-mps-burgers/docs/plot_cole_hopf_circuit_evolution.py"
```

The sweeper converts this to a CLI invocation of `burgers_solver.py`
and runs it; the postproc receives the resulting JSON dump (built
by `burgers_postprocess.py`) and renders the comparison plot.

## 7. What this solver does NOT do

- **No 2-D / 3-D**. Strictly 1-D. The framework is general enough
  for higher dims; the application is not.
- **No adaptive `dt`**. Fixed dt = `cfl * dx` (or `--dt` override).
- **No mesh refinement**.
- **No real hardware execution by default**. `--backend-type
  hardware` submits jobs and records placeholders; result harvest
  is a separate workflow (see SPEC-shots-backend §10).
- **No physics beyond viscous Burgers + source.** No reaction term,
  no compressibility coupling, no multi-component.

## 8. Where to read more

| Topic | Doc |
|---|---|
| Framework itself | [`q8020-cfd-metautil/docs/SPEC-solverfw.md`](../../q8020-cfd-metautil/docs/SPEC-solverfw.md) |
| Cole-Hopf circuit details | [`F10-IMPLEMENTATION-SPEC.md`](F10-IMPLEMENTATION-SPEC.md) |
| Source forcing | [`SPEC-source-forcing.md`](SPEC-source-forcing.md), [`SPEC-source-forcing-REVIEW.md`](SPEC-source-forcing-REVIEW.md) |
| Shots / backend / noise | [`SPEC-shots-backend.md`](SPEC-shots-backend.md) |
| Chunked evolution | [`SPEC-chunked-evolution.md`](SPEC-chunked-evolution.md) |
| Encoding (binary vs gray) | [`SPEC-encoding-switch.md`](SPEC-encoding-switch.md) |
| Paper alignment | [`REVIEW-murali-paper-fidelity.md`](REVIEW-murali-paper-fidelity.md), [`DEEP-OVERLAP-murali-vs-ucan.md`](DEEP-OVERLAP-murali-vs-ucan.md) |
| Pipeline handoff (legacy, may be stale) | [`HANDOFF-burgers-pipeline.md`](HANDOFF-burgers-pipeline.md) |
