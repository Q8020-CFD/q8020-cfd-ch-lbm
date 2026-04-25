"""Solve 1D Burgers equation via quantum tensor-network algorithm.

Implements the hybrid classical-quantum pipeline from Murali et al. AIAA 2026.
At each time step:
  u(t+δτ) = u(t) + δτ [ν∇²u - u·∇u + g]

Four evolution methods are supported:
- shift: classical Euler with shift-operator FD (periodic BC)
- quantum_exact: Pauli decomposition + exact matrix exponential
- quantum_circuit: Pauli decomposition + Trotterized circuit
- mps: MPS state-prep circuit + exact Hamiltonian evolution

Integrates with q8020 sweeper via argparse CLI and metadata fragment writers.
"""

#pylint: disable=invalid-name

import argparse
import json
import sys
import time
from math import pi
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from q8020_cfd_metautil.args import add_standard_quantum_args
from q8020_cfd_metautil.meta_fragment import (
    make_case_meta,
    write_case,
    write_results,
    write_analysis,
    write_artifacts,
)

from burgers_classical import (
    initial_condition_sine,
    initial_condition_multimode,
    source_term_sine,
    solve_burgers,
)
from burgers_trotter import run_simulation, compute_error


# *****************************************************************************
# main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve 1D Burgers equation via quantum tensor-network algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python burgers_solver.py --q 3 --method shift --n-steps 10
  python burgers_solver.py --q 3 --method quantum_exact --n-steps 5 --nu 1e-4
  python burgers_solver.py --q 3 --method quantum_circuit --trotter-order 2 --n-steps 5
""")
    add_standard_quantum_args(parser)

    # Problem parameters
    parser.add_argument(
        "--q", type=int, default=3,
        help="Number of qubits (N=2^q grid points)",
    )
    parser.add_argument(
        "--nu", type=float, default=1e-4,
        help="Kinematic viscosity",
    )
    parser.add_argument(
        "--cfl", type=float, default=0.1,
        help="CFL number (dt = cfl * dx)",
    )
    parser.add_argument(
        "--n-steps", type=int, default=50,
        help="Number of time steps",
    )
    parser.add_argument(
        "--shock-pct", type=float, default=None,
        help=(
            "Fraction of inviscid shock time (%%); overrides --n-steps. "
            "T_end = (shock_pct/100) * 1/(2π)"
        ),
    )
    parser.add_argument(
        "--ic", type=str, default="sine", choices=["sine", "multimode"],
        help="Initial condition (multimode = random-phase Fourier IC, "
             "NOT Burgers turbulence -- see F11 in implementation plan)",
    )
    parser.add_argument(
        "--ic-modes", type=int, default=6,
        help="Number of Fourier modes (multimode IC only)",
    )
    parser.add_argument(
        "--ic-seed", type=int, default=42,
        help="RNG seed for random phases (multimode IC only)",
    )
    parser.add_argument(
        "--ic-alpha", type=float, default=1.0,
        help="Velocity spectrum exponent A_k~k^-alpha (multimode IC only)",
    )
    parser.add_argument(
        "--source", type=str, default="sine", choices=["sine", "none"],
        help="Source term",
    )

    # Boundary conditions
    parser.add_argument(
        "--bc", type=str, default="periodic",
        choices=["periodic", "dirichlet"],
        help="Boundary conditions (periodic or dirichlet u=0)",
    )

    # Method selection
    parser.add_argument(
        "--method", type=str, default="shift",
        choices=[
            "shift", "quantum_exact", "quantum_circuit", "mps",
            "tebd", "tebd_circuit", "cole_hopf",
            "cole_hopf_circuit",
        ],
        help="Evolution method",
    )
    parser.add_argument(
        "--propagator", type=str, default="qft-diagonal",
        choices=["qft-diagonal", "dense-block"],
        help="Heat propagator variant (cole_hopf_circuit only)",
    )
    parser.add_argument(
        "--encoding", type=str, default="binary",
        choices=["binary", "gray"],
        help="State encoding (cole_hopf_circuit only)",
    )
    parser.add_argument(
        "--sign-recovery", type=str, default="none",
        choices=["none", "classical_oracle", "hadamard_test", "dual_rail"],
        help="Sign recovery strategy for shots>0 path (F9)",
    )
    parser.add_argument(
        "--trotter-order", type=int, default=1,
        help="Suzuki-Trotter order (quantum_circuit only)",
    )
    parser.add_argument(
        "--trotter-reps", type=int, default=1,
        help="Trotter repetitions (quantum_circuit only)",
    )

    # MPS parameters
    parser.add_argument(
        "--bond-dim", type=int, default=None,
        help="MPS bond dimension (None=full, mps method only)",
    )
    parser.add_argument(
        "--mps-threshold", type=float, default=0.0,
        help="MPS singular value truncation threshold (mps method only)",
    )

    # Reporting
    parser.add_argument(
        "--save-every", type=int, default=0,
        help="Save solution every N steps (0=only initial and final)",
    )
    parser.add_argument(
        "--noshow", action="store_true",
        help="Suppress plots",
    )

    args = parser.parse_args()

    # Grid setup
    q = args.q
    N = 2**q
    # Grid depends on BC: Dirichlet includes both endpoints so that
    # u[0]=u[N-1]=0 is satisfied naturally by the sine IC; periodic
    # excludes the right endpoint since x=0 and x=1 are identified.
    if args.bc == "dirichlet":
        x = np.linspace(0, 1, N, endpoint=True)
    else:
        x = np.linspace(0, 1, N, endpoint=False)
    dx = x[1] - x[0]
    dt = args.cfl * dx
    nu = args.nu

    # IC and source functions
    if args.ic != "multimode":
        _mm_defaults = {"ic_modes": 6, "ic_seed": 42, "ic_alpha": 1.0}
        for _k, _v in _mm_defaults.items():
            if getattr(args, _k) != _v:
                print(
                    f"[burgers] WARNING: --{_k.replace('_', '-')} is ignored "
                    f"for --ic {args.ic} (only applies to --ic multimode)",
                    file=sys.stderr, flush=True,
                )

    if args.ic == "sine":
        u0 = initial_condition_sine(x)
    elif args.ic == "multimode":
        u0 = initial_condition_multimode(
            x, n_modes=args.ic_modes, seed=args.ic_seed, alpha=args.ic_alpha,
        )
    else:
        raise ValueError(f"Unknown IC: {args.ic}")
    source_fn = {"sine": source_term_sine, "none": None}[args.source]

    # Shock formation time: t_shock = 1 / max|du0/dx| (inviscid estimate).
    # For sine IC this equals 1/(2*pi); for other ICs it is computed from
    # the actual initial gradient.
    du0dx = np.gradient(u0, dx)
    max_grad = np.max(np.abs(du0dx))
    t_shock = 1.0 / max_grad if max_grad > 0 else 1.0 / (2 * pi)

    shock_pct = args.shock_pct
    if shock_pct is not None:
        t_end = (shock_pct / 100.0) * t_shock
        n_steps = round(t_end / (args.cfl * dx))
    else:
        n_steps = args.n_steps

    print(
        f"[burgers] q={q} N={N} nu={nu:.1e} cfl={args.cfl} "
        f"dt={dt:.4e} n_steps={n_steps} method={args.method}",
        file=sys.stderr, flush=True,
    )

    # Always run classical FTCS baseline
    print("[burgers] running classical FTCS (Forward-Time Central-Space) baseline ...", file=sys.stderr, flush=True)
    t0 = time.time()
    sols_classical = solve_burgers(
        u0, x, nu, dt, n_steps, source_fn=source_fn, bc=args.bc,
    )
    t_classical = time.time() - t0
    print(f"[burgers] classical done {t_classical:.2f}s", file=sys.stderr, flush=True)

    # Run selected method
    print(f"[burgers] running method={args.method} ...", file=sys.stderr, flush=True)
    t0 = time.time()
    sols_method, step_metrics = run_simulation(
        u0, x, nu, dt, n_steps,
        source_fn=source_fn,
        method=args.method,
        trotter_order=args.trotter_order,
        trotter_reps=args.trotter_reps,
        bond_dim=args.bond_dim,
        mps_threshold=args.mps_threshold,
        shots=args.shots,
        backend_name=args.backend,
        t1=args.t1,
        t2=args.t2,
        bc=args.bc,
        sign_recovery=args.sign_recovery,
        propagator=args.propagator,
        snapshot_interval=max(1, args.save_every),
        encoding=args.encoding,
    )
    t_method = time.time() - t0
    print(f"[burgers] method done {t_method:.2f}s", file=sys.stderr, flush=True)

    # Determine which steps to save and compute errors at
    if args.save_every > 0:
        save_steps = list(range(0, n_steps + 1, args.save_every))
        if n_steps not in save_steps:
            save_steps.append(n_steps)
    else:
        save_steps = [0, n_steps]

    errors = {step: compute_error(sols_method[step], sols_classical[step])
              for step in save_steps}

    final_error = errors[n_steps]
    max_error = max(errors.values())

    print(
        f"[burgers] final_error={final_error:.4e} max_error={max_error:.4e}",
        file=sys.stderr, flush=True,
    )

    # *******************************************************************************
    # Write metadata fragments

    outdir = Path(args.outdir) if args.outdir else Path.cwd()
    exp_id = args.experiment_id

    # Case fragment: problem definition
    case_data = make_case_meta(
        name="burgers_1d",
        q=q,
        N=N,
        nu=nu,
        dx=float(dx),
        dt=float(dt),
        cfl=args.cfl,
        n_steps=n_steps,
        shock_pct=shock_pct,
        ic=args.ic,
        ic_modes=args.ic_modes if args.ic == "multimode" else None,
        ic_seed=args.ic_seed if args.ic == "multimode" else None,
        ic_alpha=args.ic_alpha if args.ic == "multimode" else None,
        source=args.source,
        method=args.method,
        trotter_order=args.trotter_order,
        trotter_reps=args.trotter_reps,
        bond_dim=args.bond_dim,
        mps_threshold=args.mps_threshold,
        shots=args.shots,
        backend_name=args.backend,
        t1=args.t1,
        t2=args.t2,
        bc=args.bc,
        sign_recovery=args.sign_recovery,
    )
    write_case(outdir, case_data, experiment_id=exp_id)

    # Results fragment: solution data
    results_data = {
        "u_initial": u0.tolist(),
        "u_final_classical": np.array(sols_classical[-1]).real.tolist(),
        "u_final_method": np.array(sols_method[-1]).real.tolist(),
        "errors_by_step": {str(k): v for k, v in errors.items()},
        "final_error": final_error,
        "max_error": max_error,
    }
    write_results(outdir, results_data, experiment_id=exp_id)

    # Analysis fragment: error metrics and timing
    analysis_data = {
        "final_error_epsilon": final_error,
        "max_error_epsilon": max_error,
        "classical_wall_time_s": t_classical,
        "method_wall_time_s": t_method,
        "speedup_ratio": t_classical / max(t_method, 1e-9),
        "n_pauli_terms": None,
        "shots": args.shots,
        "t1": args.t1,
        "t2": args.t2,
        "backend_name": args.backend,
    }
    if args.method != "shift":
        from burgers_nonlinear import build_evolution_hamiltonian
        g0 = source_fn(x, 0) if source_fn is not None else None
        H = build_evolution_hamiltonian(u0, dx, dt, nu, g0)
        analysis_data["n_pauli_terms"] = len(H)
    if step_metrics:
        n = len(step_metrics)
        analysis_data["total_pauli_decomp_time"] = sum(m.get("pauli_decomp_time_s", 0) for m in step_metrics)
        analysis_data["total_circuit_construction_time"] = sum(m.get("circuit_construction_time_s", 0) for m in step_metrics)
        analysis_data["total_transpilation_time"] = sum(m.get("transpilation_time_s", 0) for m in step_metrics)
        analysis_data["total_execution_time"] = sum(m.get("execution_time_s", 0) for m in step_metrics)
        depths = [m["circuit_depth"] for m in step_metrics if "circuit_depth" in m]
        analysis_data["avg_circuit_depth"] = sum(depths) / len(depths) if depths else None
        gate_totals = [sum(m["gate_counts"].values()) for m in step_metrics if "gate_counts" in m]
        analysis_data["avg_gate_count"] = sum(gate_totals) / len(gate_totals) if gate_totals else None
        analysis_data["avg_cx_gates"] = sum(m["gate_counts"].get("cx", 0) for m in step_metrics if "gate_counts" in m) / n
        analysis_data["n_qubits"] = step_metrics[0].get("n_qubits") if step_metrics else None
        analysis_data["sign_recovery"] = step_metrics[0].get("sign_recovery", "none")
        analysis_data["per_step_metrics"] = step_metrics
    write_analysis(outdir, analysis_data, experiment_id=exp_id)

    # Artifacts fragment: grid and solution snapshots
    artifacts_data = {
        "grid": x.tolist(),
        "solution_steps": {str(s): np.array(sols_method[s]).real.tolist() for s in save_steps},
    }
    write_artifacts(outdir, artifacts_data, experiment_id=exp_id)

    # JSON summary to stdout (harvested by sweeper)
    summary = {
        "experiment_id": exp_id,
        "algorithm": f"burgers_{args.method}",
        "q": q,
        "N": N,
        "nu": nu,
        "n_steps": n_steps,
        "shock_pct": shock_pct,
        "ic": args.ic,
        "ic_modes": args.ic_modes if args.ic == "multimode" else None,
        "ic_seed": args.ic_seed if args.ic == "multimode" else None,
        "ic_alpha": args.ic_alpha if args.ic == "multimode" else None,
        "method": args.method,
        "bc": args.bc,
        "trotter_order": args.trotter_order,
        "trotter_reps": args.trotter_reps,
        "shots": args.shots,
        "t1": args.t1,
        "t2": args.t2,
        "backend_name": args.backend,
        "final_error": final_error,
        "max_error": max_error,
        "classical_time_s": t_classical,
        "method_time_s": t_method,
        "exit_code": 0,
    }
    print(json.dumps(summary, indent=2))

    sys.exit(0)
