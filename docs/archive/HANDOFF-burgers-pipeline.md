# Burgers Quantum Pipeline: Integration Handoff

> **Goal**: Wire the existing Burgers quantum solver modules into the q8020 sweeper infrastructure so the full pipeline can be run from a TOML config via `q8020-sweep`.

---

## 1. What Exists (Done)

Five Python modules in `/Users/agallojr/proj/src/q8020/q8020-cfd-axequalsb/src/`:

| Module | Purpose |
|--------|---------|
| `burgers_classical.py` | FTCS solver, initial conditions, source terms, FD matrices |
| `burgers_mps.py` | MPS decomposition + quantum circuit state preparation |
| `burgers_mpo.py` | S+/S- ladder circuits, gradient LCU, Laplacian LCU, block encoding |
| `burgers_nonlinear.py` | Pauli decomposition of evolution operator (Appendix A), Hamiltonian construction |
| `burgers_trotter.py` | Time-stepping orchestrator: shift/quantum_exact/quantum_circuit methods |

Tests: `test_burgers.py` (20/20), `test_mpo.py` (33/35), `test_nonlinear.py`.

The modules have no CLI entry point. They run via `__main__` blocks or pytest.

---

## 2. What Needs to Be Built

### 2a. CLI Entry Point: `burgers_solver.py`

Create a new file `burgers_solver.py` that follows the same pattern as `ax_equals_b_hhl.py`. It must:

1. **Accept CLI args** using the metautil helpers:

```python
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

from q8020_cfd_metautil.args import add_standard_quantum_args
from q8020_cfd_metautil.meta_fragment import (
    make_case_meta,
    write_case,
    write_results,
    write_analysis,
    write_artifacts,
)

from burgers_classical import initial_condition_sine, source_term_sine, solve_burgers
from burgers_trotter import run_simulation, compute_error
```

2. **Define problem-specific args** (these become TOML keys):

```python
parser = argparse.ArgumentParser(
    description="Solve 1D Burgers equation via quantum tensor-network algorithm"
)
add_standard_quantum_args(parser)  # --backend, --shots, --seed, --outdir, --experiment-id, --workflow-id, --optimization-level

# Problem parameters
parser.add_argument("--q", type=int, default=3, help="Number of qubits (N=2^q grid points)")
parser.add_argument("--nu", type=float, default=1e-4, help="Kinematic viscosity")
parser.add_argument("--cfl", type=float, default=0.1, help="CFL number (dt = cfl * dx)")
parser.add_argument("--n-steps", type=int, default=50, help="Number of time steps")
parser.add_argument("--ic", type=str, default="sine", choices=["sine"], help="Initial condition")
parser.add_argument("--source", type=str, default="sine", choices=["sine", "none"], help="Source term")

# Method selection
parser.add_argument("--method", type=str, default="shift",
                    choices=["shift", "quantum_exact", "quantum_circuit"],
                    help="Evolution method")
parser.add_argument("--trotter-order", type=int, default=1, help="Suzuki-Trotter order")
parser.add_argument("--trotter-reps", type=int, default=1, help="Trotter repetitions")

# MPS parameters (for future MPS state-prep pipeline)
parser.add_argument("--bond-dim", type=int, default=None, help="MPS bond dimension (None=full)")

# Reporting
parser.add_argument("--save-every", type=int, default=0,
                    help="Save solution every N steps (0=only final)")
parser.add_argument("--noshow", action="store_true", help="Suppress plots")
```

3. **Main logic** — the simulation loop:

```python
args = parser.parse_args()

q = args.q
N = 2**q
x = np.linspace(0, 1, N, endpoint=False)
dx = x[1] - x[0]
dt = args.cfl * dx
nu = args.nu
n_steps = args.n_steps

# IC and source
ic_fn = {"sine": initial_condition_sine}[args.ic]
source_fn = {"sine": source_term_sine, "none": lambda x, t: None}[args.source]
u0 = ic_fn(x)

# Always run classical baseline
t0 = time.time()
sols_classical = solve_burgers(u0, x, nu, dt, n_steps, source_fn=source_fn)
t_classical = time.time() - t0

# Run selected method
t0 = time.time()
sols_method = run_simulation(
    u0, x, nu, dt, n_steps,
    source_fn=source_fn,
    method=args.method,
    trotter_order=args.trotter_order,
    trotter_reps=args.trotter_reps,
)
t_method = time.time() - t0

# Compute error at each saved step
errors = {}
save_steps = list(range(0, n_steps + 1, max(1, args.save_every))) if args.save_every > 0 else [0, n_steps]
for step in save_steps:
    errors[step] = compute_error(sols_method[step], sols_classical[step])
```

4. **Write metadata fragments** (same pattern as HHL solver):

```python
outdir = Path(args.outdir) if args.outdir else Path.cwd()
exp_id = args.experiment_id

# Case fragment
case_data = make_case_meta(
    name="burgers_1d",
    q=q, N=N, nu=nu, dx=dx, dt=dt, cfl=args.cfl,
    n_steps=n_steps, ic=args.ic, source=args.source,
    method=args.method,
    trotter_order=args.trotter_order,
    trotter_reps=args.trotter_reps,
    bond_dim=args.bond_dim,
)
write_case(outdir, case_data, experiment_id=exp_id)

# Results fragment
results_data = {
    "u_initial": u0.tolist(),
    "u_final_classical": sols_classical[-1].tolist(),
    "u_final_method": sols_method[-1].tolist(),
    "errors_by_step": {str(k): v for k, v in errors.items()},
    "final_error": errors[n_steps],
    "max_error": max(errors.values()),
}
write_results(outdir, results_data, experiment_id=exp_id)

# Analysis fragment
analysis_data = {
    "final_error_epsilon": errors[n_steps],
    "max_error_epsilon": max(errors.values()),
    "classical_wall_time_s": t_classical,
    "method_wall_time_s": t_method,
    "speedup_ratio": t_classical / max(t_method, 1e-9),
    "n_pauli_terms": None,  # filled below for quantum methods
}
if args.method != "shift":
    from burgers_nonlinear import build_evolution_hamiltonian
    H = build_evolution_hamiltonian(u0, dx, dt, nu, source_fn(x, 0) if args.source != "none" else None)
    analysis_data["n_pauli_terms"] = len(H)
write_analysis(outdir, analysis_data, experiment_id=exp_id)

# Artifacts fragment
artifacts_data = {
    "grid": x.tolist(),
    "solution_steps": {str(s): sols_method[s].tolist() for s in save_steps},
}
write_artifacts(outdir, artifacts_data, experiment_id=exp_id)

# JSON summary to stdout (harvested by sweep)
summary = {
    "experiment_id": exp_id,
    "algorithm": f"burgers_{args.method}",
    "q": q, "N": N, "nu": nu, "n_steps": n_steps,
    "method": args.method,
    "final_error": errors[n_steps],
    "max_error": max(errors.values()),
    "classical_time_s": t_classical,
    "method_time_s": t_method,
}
print(json.dumps(summary, indent=2))
```

### 2b. TOML Sweep Config: `input/burgers_quantum.toml`

Create in `/Users/agallojr/proj/src/q8020/q8020-cfd-axequalsb/input/burgers_quantum.toml`:

```toml
# Burgers 1D Quantum Solver Sweep
# Run: q8020-sweep q8020-cfd-axequalsb/input/burgers_quantum.toml

[global]
_output_dir = "~/q8020"
_inject_outdir = "--outdir"
_env = "./q8020-cfd-axequalsb/.venv"
_script = "python ./q8020-cfd-axequalsb/src/burgers_solver.py"
"--noshow" = true

# ── Classical baseline ──────────────────────────────────
[classical_N8]
"--q" = 3
"--nu" = 1e-4
"--cfl" = 0.1
"--n-steps" = 50
"--method" = "shift"
"--save-every" = 10

[classical_N16]
"--q" = 4
"--nu" = 1e-4
"--cfl" = 0.1
"--n-steps" = 100
"--method" = "shift"
"--save-every" = 20

[classical_N256]
"--q" = 8
"--nu" = 1e-4
"--cfl" = 0.1
"--n-steps" = 200
"--method" = "shift"
"--save-every" = 50

# ── Quantum exact (matrix expm, no Trotter error) ──────
[qexact_N8]
"--q" = 3
"--nu" = 1e-4
"--cfl" = 0.1
"--n-steps" = 50
"--method" = "quantum_exact"
"--save-every" = 10

[qexact_N16]
"--q" = 4
"--nu" = 1e-4
"--cfl" = 0.1
"--n-steps" = 50
"--method" = "quantum_exact"
"--save-every" = 10

# ── Quantum circuit (Trotterized) ──────────────────────
[qcircuit_N8_T1R1]
"--q" = 3
"--nu" = 1e-4
"--cfl" = 0.1
"--n-steps" = 50
"--method" = "quantum_circuit"
"--trotter-order" = 1
"--trotter-reps" = 1
"--save-every" = 10

[qcircuit_N8_T2R1]
"--q" = 3
"--nu" = 1e-4
"--cfl" = 0.1
"--n-steps" = 50
"--method" = "quantum_circuit"
"--trotter-order" = 2
"--trotter-reps" = 1
"--save-every" = 10

# ── Trotter convergence study ──────────────────────────
[trotter_convergence]
"--q" = 3
"--nu" = 1e-4
"--cfl" = 0.1
"--n-steps" = 20
"--method" = "quantum_circuit"
"--trotter-order" = 1
"--trotter-reps" = [1, 2, 4, 8]
"--save-every" = 5

# ── Viscosity study ────────────────────────────────────
[viscosity_study]
"--q" = 3
"--cfl" = 0.1
"--n-steps" = 50
"--nu" = [1e-2, 1e-3, 1e-4]
"--method" = "quantum_exact"
"--save-every" = 10
```

---

## 3. Infrastructure Conventions to Follow

### 3a. Argparse Helpers (from `q8020_cfd_metautil.args`)

`add_standard_quantum_args(parser)` adds these args automatically:
- `--backend`, `--coupling-map` (backend selection)
- `--shots`, `--seed` (execution)
- `--outdir` (output directory, injected by sweeper)
- `--experiment-id`, `--workflow-id` (injected by sweeper)
- `--optimization-level` (transpile)
- `--t1`, `--t2` (noise)

**Do not redefine these.** Just call the helper and add problem-specific args after.

### 3b. Metadata Fragments (from `q8020_cfd_metautil.meta_fragment`)

Write these fragments to `args.outdir`:
- `write_case()` — problem definition (grid, viscosity, method)
- `write_results()` — solution data
- `write_analysis()` — error metrics, timing
- `write_artifacts()` — any files/arrays worth archiving

Each takes `(outdir, data_dict, experiment_id=exp_id)`.

### 3c. stdout JSON

The solver must print a JSON summary to stdout. The sweeper captures this and uses it for harvest.

### 3d. TOML Key Conventions

- CLI args use `"--key"` with double dash (long form)
- Boolean flags: `"--noshow" = true` becomes `--noshow` (flag only)
- Lists expand to multiple cases: `"--nu" = [1e-2, 1e-3]` becomes 2 cases
- Underscore-prefixed keys (`_output_dir`, `_script`, etc.) are sweeper metadata

### 3e. Execution

```bash
# From the q8020 monorepo root:
cd /Users/agallojr/proj/src/q8020

# Run all cases sequentially:
q8020-sweep q8020-cfd-axequalsb/input/burgers_quantum.toml

# Run specific case groups only:
q8020-sweep q8020-cfd-axequalsb/input/burgers_quantum.toml classical_N8 qexact_N8

# Parallel:
# Add _run_mode = "parallel" to [global] in the TOML
```

---

## 4. Module Dependency Map

```
burgers_solver.py  (NEW — CLI entry point)
 ├── burgers_classical.py    (IC, source, FTCS baseline)
 ├── burgers_trotter.py      (run_simulation, compute_error)
 │    └── burgers_nonlinear.py (Pauli decomp, Hamiltonian, evolution circuit)
 │         └── burgers_mpo.py  (shift_matrix for RHS computation)
 ├── burgers_mps.py          (MPS state prep — not yet used in pipeline)
 ├── q8020_cfd_metautil.args
 ├── q8020_cfd_metautil.meta_fragment
 └── q8020_cfd_qutil          (get_backend, get_circuit_info — for future hardware runs)
```

---

## 5. Known Limitations / Future Work

### Scale bottleneck
The Pauli decomposition in `burgers_nonlinear.py` solves a 4^q x 4^q linear system per time step. Practical limits:
- q=3 (N=8): instant
- q=4 (N=16): ~1s per step
- q=5 (N=32): ~minutes per step
- q=8 (N=256): infeasible per step

For the paper's Fig. 11 (N=256), only `method="shift"` is practical.

### MPS state-prep circuit not in pipeline
`burgers_mps.py` can build quantum circuits that prepare |u> from MPS decomposition, but `burgers_trotter.py` currently feeds the state vector directly to `Statevector.evolve()`. Wiring MPS prep into the circuit pipeline (MPS prep circuit + evolution circuit composed) is a future enhancement.

### Nonlinear term
The nonlinear convective term u*du/dx is handled by the Pauli decomposition — it linearizes around the current state each step. This is faithful to the paper's Appendix A but means the Hamiltonian changes every step.

### Hardware execution
The solver currently uses statevector simulation. To run on real hardware:
1. Use `get_backend(backend_type="hardware", ...)` from qutil
2. Add measurement + post-selection logic
3. This only makes sense at small q (3-4) where circuits fit

---

## 6. Test Validation

Before wiring to sweeper, verify everything works:

```bash
cd /Users/agallojr/proj/src/q8020/q8020-cfd-axequalsb/src

# Unit tests
/Users/agallojr/proj/src/q8020/.venv/bin/python -m pytest test_burgers.py -v  # 20/20
/Users/agallojr/proj/src/q8020/.venv/bin/python -m pytest test_mpo.py -v      # 33/35 (2 known issues)
/Users/agallojr/proj/src/q8020/.venv/bin/python -m pytest test_nonlinear.py -v

# Smoke tests (each module's __main__)
/Users/agallojr/proj/src/q8020/.venv/bin/python burgers_classical.py
/Users/agallojr/proj/src/q8020/.venv/bin/python burgers_mps.py
/Users/agallojr/proj/src/q8020/.venv/bin/python burgers_mpo.py
/Users/agallojr/proj/src/q8020/.venv/bin/python burgers_nonlinear.py
/Users/agallojr/proj/src/q8020/.venv/bin/python burgers_trotter.py
```

### test_mpo.py known issues (2 failures):
1. `test_gradient_on_sine_wave` — tolerance too tight for N=16 central difference. Fix: relax tolerance or increase N.
2. `test_gradient_lcu_on_mps_state` — missing `from qiskit.circuit import QuantumCircuit` import in test file.

---

## 7. File Locations Summary

| What | Path |
|------|------|
| Solver modules | `/Users/agallojr/proj/src/q8020/q8020-cfd-axequalsb/src/burgers_*.py` |
| Argparse helpers | `/Users/agallojr/proj/src/q8020/q8020-cfd-metautil/src/q8020_cfd_metautil/args.py` |
| Fragment writers | `/Users/agallojr/proj/src/q8020/q8020-cfd-metautil/src/q8020_cfd_metautil/meta_fragment.py` |
| Sweep runner | `/Users/agallojr/proj/src/q8020/q8020-cfd-metautil/src/q8020_cfd_metautil/sweep.py` |
| Backend utils | `/Users/agallojr/proj/src/q8020/q8020-cfd-qutil/src/q8020_cfd_qutil/` |
| Example solver (HHL) | `/Users/agallojr/proj/src/q8020/q8020-cfd-axequalsb/src/ax_equals_b_hhl.py` |
| Example TOML | `/Users/agallojr/proj/src/q8020/q8020-cfd-axequalsb/input/ax_equals_b_hhl.toml` |
| Python venv | `/Users/agallojr/proj/src/q8020/.venv/bin/python` (Qiskit 2.3.1, Python 3.12.10) |
| Paper PDF | `/Users/agallojr/proj/src/research-notes/quantum/2026/refs-artifacts/Murali-AIAA2026_QC_final.pdf` |

---

## 8. Checklist for the Implementing Bot

- [ ] Create `burgers_solver.py` — CLI entry point following the pattern above
- [ ] Fix `test_mpo.py` two failures (import + tolerance)
- [ ] Create `input/burgers_quantum.toml` sweep config
- [ ] Test standalone: `python burgers_solver.py --q 3 --method shift --n-steps 10`
- [ ] Test with sweeper: `q8020-sweep q8020-cfd-axequalsb/input/burgers_quantum.toml classical_N8`
- [ ] Verify metadata fragments are written correctly
- [ ] Optionally: create a harvester in `q8020-cfd-experiments/codes/burgers/` for post-processing
- [ ] Optionally: add plotting post-processor that generates Fig. 11-style time evolution plots
