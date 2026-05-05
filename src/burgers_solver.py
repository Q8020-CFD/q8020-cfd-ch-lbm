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

from burgers_classical import (
    initial_condition_sine,
    initial_condition_multimode,
    source_term_sine,
    solve_burgers,
    solve_burgers_godunov,
)
from burgers_fw import BurgersConfig, run_simulation_fw
from burgers_postprocess import BurgersPostProcessor
from burgers_trotter import compute_error
from q8020_cfd_metautil.solverfw import Grid1D


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
        "--ic-amplitude", type=float, default=1.0,
        help="Scale factor for IC amplitude (LBM requires < 1.0)",
    )
    parser.add_argument(
        "--classical-baseline", type=str, default="ftcs",
        choices=["ftcs", "godunov"],
        help="Classical reference solver for error computation",
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
            "cole_hopf_circuit", "qlbm", "qlbm_circuit",
        ],
        help="Evolution method",
    )
    parser.add_argument(
        "--propagator", type=str, default="qft-diagonal",
        choices=["qft-diagonal", "dense-block", "lcu"],
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
    parser.add_argument(
        "--splitting", type=str, default="lie",
        choices=["lie", "strang"],
        help="Operator splitting for tebd_circuit: lie (Lie-Trotter) or strang (Strang/symmetric)",
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

    # Shots post-processing
    parser.add_argument(
        "--phi-modes", type=int, default=0,
        help="Fourier low-pass: keep N modes in phi before "
             "inverse CH (0=no filter, shots path only)",
    )

    # Chunked evolution
    parser.add_argument(
        "--evolution-mode", type=str, default="single",
        choices=["single", "chunked"],
        help="Evolution mode (cole_hopf_circuit shots only)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=10,
        help="Steps per chunk (chunked mode only)",
    )
    parser.add_argument(
        "--lcu-taylor-order", type=int, default=4,
        help="Taylor truncation order for LCU propagator",
    )
    parser.add_argument(
        "--readout", type=str, default="direct",
        choices=["direct", "hadamard_per_bin"],
        help="Shots readout strategy (cole_hopf_circuit and tebd_circuit shots): "
             "'direct' = post-selected amplitude estimation; "
             "'hadamard_per_bin' = signed Re(psi_k) via per-bin "
             "Hadamard test (F2-10, F10-12).",
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
    if args.ic_amplitude != 1.0:
        u0 = u0 * args.ic_amplitude
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

    # Classical baseline
    bl = args.classical_baseline
    print(
        f"[burgers] running classical {bl.upper()} baseline ...",
        file=sys.stderr, flush=True,
    )
    t0 = time.time()
    if bl == "godunov":
        sols_classical = solve_burgers_godunov(
            u0, x, nu, dt, n_steps, source_fn=source_fn, bc=args.bc,
        )
    else:
        sols_classical = solve_burgers(
            u0, x, nu, dt, n_steps, source_fn=source_fn, bc=args.bc,
        )
    t_classical = time.time() - t0
    print(f"[burgers] classical done {t_classical:.2f}s", file=sys.stderr, flush=True)

    # Build solverfw config and grid
    grid = Grid1D.from_qubits(q, bc=args.bc)
    fw_config = BurgersConfig(
        q=q, nu=nu, cfl=args.cfl,
        dt=dt, n_steps=n_steps, bc=args.bc,
        method=args.method,
        ic=args.ic, source=args.source,
        trotter_order=args.trotter_order,
        trotter_reps=args.trotter_reps,
        bond_dim=args.bond_dim,
        mps_threshold=args.mps_threshold,
        shots=args.shots,
        backend_name=args.backend,
        backend_type=args.backend_type,
        coupling_map=args.coupling_map,
        optimization_level=args.optimization_level,
        seed=args.seed,
        t1=args.t1, t2=args.t2,
        sign_recovery=args.sign_recovery,
        splitting=args.splitting,
        propagator=args.propagator,
        encoding=args.encoding,
        evolution_mode=args.evolution_mode,
        chunk_size=args.chunk_size,
        phi_modes=args.phi_modes,
        taylor_order=args.lcu_taylor_order,
        readout=args.readout,
        save_every=args.save_every,
        shock_pct=shock_pct,
    )

    # Run selected method via solverfw
    print(f"[burgers] running method={args.method} ...", file=sys.stderr, flush=True)
    t0 = time.time()
    sols_method, step_metrics = run_simulation_fw(
        fw_config, grid, u0, source_fn=source_fn,
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

    errors = {}
    for step in save_steps:
        u_m = sols_method[step]
        if not np.all(np.isfinite(u_m)):
            continue
        errors[step] = compute_error(u_m, sols_classical[step])

    final_error = errors.get(n_steps, float("nan"))
    max_error = max(errors.values()) if errors else float("nan")

    print(
        f"[burgers] final_error={final_error:.4e} max_error={max_error:.4e}",
        file=sys.stderr, flush=True,
    )

    # *******************************************************************************
    # Write metadata fragments via PostProcessor

    outdir = Path(args.outdir) if args.outdir else Path.cwd()
    exp_id = args.experiment_id

    post = BurgersPostProcessor(
        classical_solutions=sols_classical,
        source_fn=source_fn,
        x=x,
        output_dir=outdir,
        experiment_id=exp_id,
        save_steps=save_steps,
    )
    # Feed step metrics collected during simulation
    if step_metrics:
        post.step_metrics = step_metrics

    summary = post.write_fragments(
        config=fw_config,
        u0=u0,
        solutions=sols_method,
        t_classical=t_classical,
        t_method=t_method,
        args=args,
    )

    # JSON summary to stdout (harvested by sweeper)
    print(json.dumps(summary, indent=2))

    sys.exit(0)
