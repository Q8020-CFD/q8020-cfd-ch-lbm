"""Solve 1D Burgers equation via quantum tensor-network algorithm.

The `quantum_circuit` method implements Gopalakrishnan Meena et al.
AIAA-2026 Appendix A.A (Eqs. 16-17): per-step Pauli decomposition of
the classical Euler RHS, then Trotterized circuit evolution of the
fitted unitary e^{-iÂδτ}.  This is a HYBRID pathway, not pure quantum:
every step runs a classical Euler RHS, a classical least-squares fit
of the Pauli coefficients, and classical norm tracking around the
unitary application.  The only pure-quantum pathway in this codebase
is `cole_hopf_circuit`.  The paper's §V.C pipeline (classical Euler
with `quimb` MPS/MPO spatial operators) is NOT implemented here; we
use it externally as a reference path.

At each time step the underlying classical update has the §V.C Eq. 15
form: u(t+δτ) = u(t) + δτ [ν∇²u - u·∇u + g], but the spatial operators
are plain shift-matrix FD, not MPS/MPO via quimb.

Methods supported (see docs/OVERVIEW-burgers-solver.md for the full
list and classification):
- shift: classical Euler with shift-operator FD
- quantum_exact: Pauli decomposition + exact matrix exponential
- quantum_circuit: Pauli decomposition + Trotterized circuit (Appendix A.A)
- mps: MPS state-prep circuit + exact Hamiltonian evolution
- tebd / tebd_circuit: TEBD classical / quantum
- cole_hopf / cole_hopf_circuit: Cole-Hopf linearisation, MPS / pure quantum
- lbm: classical D1Q3 lattice Boltzmann (no quantum content -- no
  shots, no backend)
- qlbm_circuit: quantum-circuit D1Q3 (hybrid Option A; shots honoured)

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
    initial_condition_gaussian,
    initial_condition_sine,
    initial_condition_multimode,
    source_term_sine,
    make_reference_grid,
    solve_burgers_subsampled,
)
from burgers_cole_hopf import (
    analytic_solution_cole_hopf,
    initial_condition_cole_hopf_exact,
    validate_cole_hopf_coeffs,
)
from burgers_fw import BurgersConfig, run_simulation_fw
from burgers_postprocess import BurgersPostProcessor
from burgers_trotter import compute_error
from q8020_cfd_metautil.solverfw import Grid1D


# Preferred steps-per-segment for --auto-cadence (matches the q=5/n_steps=98
# design point of 7); the actual value is the nearest divisor of n_steps.
_PREFERRED_SEGMENT = 7


def _nearest_divisor(n: int, target: int) -> int:
    """Divisor of n closest to target (ties favour the larger divisor)."""
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    return min(divisors, key=lambda d: (abs(d - target), -d))


def construct_ic(x, args, nu, ch_coeffs=None):
    """Build the initial velocity field on grid ``x`` from CLI args.

    Pure array construction (no validation/warnings -- those run once in
    main).  Factored out so the resolved FTCS reference can re-evaluate
    the same IC on its refined grid instead of interpolating a coarse u0.
    ``ch_coeffs`` is required only for --ic cole_hopf_exact.
    """
    if args.ic == "sine":
        u0 = initial_condition_sine(x)
    elif args.ic == "multimode":
        u0 = initial_condition_multimode(
            x, n_modes=args.ic_modes, seed=args.ic_seed, alpha=args.ic_alpha,
        )
    elif args.ic == "gaussian":
        u0 = initial_condition_gaussian(
            x, amplitude=1.0, center=args.ic_center, sigma=args.ic_sigma,
        )
    elif args.ic == "cole_hopf_exact":
        u0 = initial_condition_cole_hopf_exact(x, ch_coeffs, nu, L_box=1.0)
    else:
        raise ValueError(f"Unknown IC: {args.ic}")
    if args.ic_amplitude != 1.0:
        u0 = u0 * args.ic_amplitude
    return u0


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
        "--ic", type=str, default=None,
        choices=["sine", "multimode", "gaussian", "cole_hopf_exact"],
        help=(
            "Initial condition.  sine = u0=sin(2*pi*x).  multimode = "
            "random-phase Fourier IC (NOT Burgers turbulence; see F11 "
            "in implementation plan).  gaussian = localized pulse "
            "u0=A*exp(-((x-x0)/sigma)^2) (no analytic CH reference; "
            "uses FTCS).  cole_hopf_exact = u0 derived from a "
            "Neumann cosine sum phi_0(x)=a_0+sum a_n*cos(n*pi*x), "
            "yielding a closed-form analytic u(x,t) reference under "
            "unforced Cole-Hopf heat evolution (Dirichlet-on-u BC only). "
            "Default: cole_hopf_exact when --method is cole_hopf or "
            "cole_hopf_circuit; sine otherwise."
        ),
    )
    parser.add_argument(
        "--ic-center", type=float, default=0.5,
        help="Gaussian IC centre x0 (only used with --ic gaussian).",
    )
    parser.add_argument(
        "--ic-sigma", type=float, default=0.1,
        help="Gaussian IC width sigma (only used with --ic gaussian).",
    )
    parser.add_argument(
        "--ic-cole-hopf-coeffs", type=str, default="1.0,0.3",
        help=(
            "Comma-separated Cole-Hopf cosine-sum coefficients "
            "a_0,a_1,a_2,...  Only used when --ic cole_hopf_exact.  Must "
            "satisfy a_0 > sum(|a_n|) for n>=1 (positivity of phi)."
        ),
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
        "--no-classical-reference", dest="classical_reference",
        action="store_false", default=True,
        help=(
            "Skip the classical reference run entirely (no FTCS "
            "solve, no error-vs-classical metrics, no speedup ratio).  "
            "Default: classical reference is run."
        ),
    )
    parser.add_argument(
        "--no-analytic-reference", dest="analytic_reference",
        action="store_false", default=True,
        help=(
            "When --ic cole_hopf_exact, by default the closed-form "
            "analytic solution u(x,t) is used as the classical reference "
            "(replacing FTCS, which is itself approximate).  Pass "
            "this flag to fall back to the FTCS reference even "
            "with the analytic IC.  No effect for other --ic choices."
        ),
    )
    parser.add_argument(
        "--ref-points", type=int, default=200,
        help=(
            "Minimum grid points for the classical FTCS reference, "
            "decoupled from the quantum grid (N=2^q).  The reference runs "
            "on the smallest BC-consistent superset of the quantum nodes "
            "with at least this many points, then is subsampled back to "
            "the quantum nodes for pointwise error scoring (no "
            "interpolation error).  Default: 200.  No effect on the "
            "analytic Cole-Hopf reference (exact at any resolution)."
        ),
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
            "cole_hopf_circuit", "lbm", "qlbm_circuit", "direct_lcu",
        ],
        help=(
            "Evolution method.  Classical: shift (FTCS), tebd, "
            "cole_hopf, lbm (D1Q3 -- ignores --shots).  Hybrid: "
            "quantum_exact, quantum_circuit, mps, qlbm_circuit.  "
            "Pure-quantum: cole_hopf_circuit.  See OVERVIEW §2 for "
            "the full classification."
        ),
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

    # Measure-and-reprepare (segmented) evolution
    parser.add_argument(
        "--evolution-mode", type=str, default="single",
        choices=["single", "measure_reprepare"],
        help="Evolution mode (cole_hopf_circuit shots only)",
    )
    parser.add_argument(
        "--segment-size", type=int, default=10,
        help="Steps per segment (measure_reprepare mode only)",
    )
    parser.add_argument(
        "--metric-transpile-timeout", type=float, default=60.0,
        help=(
            "Per-circuit wall-time cap (s) on the isolated basis-transpile "
            "used ONLY to report depth/gate counts (qlbm + cole_hopf "
            "circuit).  Never affects execution/results -- exceeding it "
            "just records metrics unavailable.  0 = uncapped (let it run as "
            "long as needed; good for a dedicated metrics pass)."
        ),
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
        "--auto-cadence", action="store_true",
        help=(
            "Auto-pick --segment-size (nearest divisor of the computed "
            "n_steps) and --save-every (= segment-size), so segmented "
            "evolution stays aligned at any q.  Overrides both flags."
        ),
    )
    parser.add_argument(
        "--noshow", action="store_true",
        help="Suppress plots",
    )

    args = parser.parse_args()

    # Resolve method-dependent IC default.  Cole-Hopf methods default to
    # the analytic-IC family so the closed-form u(x,t) reference is
    # automatic; everything else defaults to sine.
    if args.ic is None:
        args.ic = (
            "cole_hopf_exact"
            if args.method in ("cole_hopf", "cole_hopf_circuit")
            else "sine"
        )

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
    if args.ic != "gaussian":
        _g_defaults = {"ic_center": 0.5, "ic_sigma": 0.1}
        for _k, _v in _g_defaults.items():
            if getattr(args, _k) != _v:
                print(
                    f"[burgers] WARNING: --{_k.replace('_', '-')} is ignored "
                    f"for --ic {args.ic} (only applies to --ic gaussian)",
                    file=sys.stderr, flush=True,
                )

    ch_coeffs: np.ndarray | None = None
    if args.ic == "cole_hopf_exact":
        if args.bc != "dirichlet":
            raise ValueError(
                "--ic cole_hopf_exact requires --bc dirichlet "
                "(the cosine basis pairs with Neumann-on-phi, which "
                "maps to Dirichlet-on-u under Cole-Hopf)."
            )
        if args.source != "none":
            raise ValueError(
                "--ic cole_hopf_exact requires --source none (forced "
                "evolution couples the cosine modes and breaks the "
                "closed-form analytic reference)."
            )
        try:
            ch_coeffs = np.array(
                [float(c) for c in args.ic_cole_hopf_coeffs.split(",")],
                dtype=float,
            )
        except ValueError as e:
            raise ValueError(
                f"--ic-cole-hopf-coeffs must be a comma-separated list "
                f"of floats; got {args.ic_cole_hopf_coeffs!r}"
            ) from e
        validate_cole_hopf_coeffs(ch_coeffs)
    u0 = construct_ic(x, args, nu, ch_coeffs)
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

    # Auto cadence: derive aligned segment-size/save-every from n_steps so
    # measure_reprepare never trips the "n_steps % segment_size" check.
    if args.auto_cadence:
        args.segment_size = _nearest_divisor(n_steps, _PREFERRED_SEGMENT)
        args.save_every = args.segment_size
        print(
            f"[burgers] auto-cadence: segment-size={args.segment_size} "
            f"save-every={args.save_every} (n_steps={n_steps})",
            file=sys.stderr, flush=True,
        )

    print(
        f"[burgers] q={q} N={N} nu={nu:.1e} cfl={args.cfl} "
        f"dt={dt:.4e} n_steps={n_steps} method={args.method}",
        file=sys.stderr, flush=True,
    )

    # Reference trajectory.  Three paths:
    #   1. --no-classical-reference: skip altogether.
    #   2. --ic cole_hopf_exact AND analytic_reference: closed-form
    #      u(x, t) from the Cole-Hopf analytic family (free, exact).
    #   3. Otherwise: classical FTCS solve.
    if not args.classical_reference:
        sols_classical = None
        t_classical = None
        print(
            "[burgers] reference trajectory skipped "
            "(--no-classical-reference)",
            file=sys.stderr, flush=True,
        )
    elif (
        args.ic == "cole_hopf_exact"
        and args.analytic_reference
        and ch_coeffs is not None
    ):
        print(
            f"[burgers] building analytic Cole-Hopf reference "
            f"(coeffs={ch_coeffs.tolist()}) ...",
            file=sys.stderr, flush=True,
        )
        t0 = time.time()
        sols_classical = [
            analytic_solution_cole_hopf(x, k * dt, ch_coeffs, nu, L_box=1.0)
            for k in range(n_steps + 1)
        ]
        t_classical = time.time() - t0
        print(
            f"[burgers] analytic reference done {t_classical:.4f}s",
            file=sys.stderr, flush=True,
        )
    else:
        # Resolved FTCS reference: run on a refined grid (>= --ref-points)
        # whose nodes are an exact superset of the quantum grid, with the
        # IC re-evaluated at that resolution, then subsample back to the
        # quantum nodes.  Decouples reference accuracy from N=2^q.
        x_ref, take = make_reference_grid(
            x, bc=args.bc, min_points=args.ref_points,
        )
        u0_ref = construct_ic(x_ref, args, nu, ch_coeffs)
        print(
            f"[burgers] running classical FTCS baseline on "
            f"{len(x_ref)} pts (subsampled to N={N}) ...",
            file=sys.stderr, flush=True,
        )
        t0 = time.time()
        sols_classical = solve_burgers_subsampled(
            u0_ref, x_ref, take, nu, dt, n_steps,
            source_fn=source_fn, bc=args.bc,
        )
        t_classical = time.time() - t0
        print(
            f"[burgers] classical done {t_classical:.2f}s",
            file=sys.stderr, flush=True,
        )

    # Build solverfw config and grid
    grid = Grid1D.from_qubits(q, bc=args.bc)
    fw_config = BurgersConfig(
        q=q, nu=nu, cfl=args.cfl,
        dt=dt, n_steps=n_steps, bc=args.bc,
        method=args.method,
        ic=args.ic, source=args.source,
        ic_modes=args.ic_modes,
        ic_seed=args.ic_seed,
        ic_alpha=args.ic_alpha,
        ic_center=args.ic_center,
        ic_sigma=args.ic_sigma,
        ic_cole_hopf_coeffs=(
            args.ic_cole_hopf_coeffs if args.ic == "cole_hopf_exact"
            else None
        ),
        classical_reference=args.classical_reference,
        analytic_reference=args.analytic_reference,
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
        segment_size=args.segment_size,
        metric_transpile_timeout=args.metric_transpile_timeout,
        phi_modes=args.phi_modes,
        taylor_order=args.lcu_taylor_order,
        readout=args.readout,
        save_every=args.save_every,
        shock_pct=shock_pct,
    )

    # Run selected method via solverfw
    print(f"[burgers] running method={args.method} ...", file=sys.stderr, flush=True)
    t0 = time.time()
    sols_method, step_metrics, genuine_steps = run_simulation_fw(
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

    # LBM-family solvers run on a coarser native cadence (dt_lbm = dx);
    # the caller-grid solutions are nearest-neighbor fills.  Persist and
    # score ONLY the genuinely-computed steps so the stored series is
    # honest about the method's true temporal resolution (no duplicated
    # frames inflating it to the caller step count).
    if genuine_steps is not None:
        save_steps = sorted(set(genuine_steps))

    if sols_classical is not None:
        errors = {}
        for step in save_steps:
            u_m = sols_method[step]
            if not np.all(np.isfinite(u_m)):
                continue
            errors[step] = compute_error(u_m, sols_classical[step])

        final_error = errors.get(n_steps, float("nan"))
        max_error = max(errors.values()) if errors else float("nan")

        print(
            f"[burgers] final_error={final_error:.4e} "
            f"max_error={max_error:.4e}",
            file=sys.stderr, flush=True,
        )
    else:
        final_error = float("nan")
        max_error = float("nan")

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
