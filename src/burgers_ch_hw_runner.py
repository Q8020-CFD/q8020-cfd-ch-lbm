"""Cole-Hopf hardware runner (F12).

Standalone driver that executes the existing Cole-Hopf measure-reprepare
segment loop on a simulator (for dry-run testing) or real IBM Quantum
hardware (Heron-class, e.g. Boston), with TREX measurement mitigation and
dynamical decoupling.  Reuses run_cole_hopf_circuit_simulation unchanged;
the only solver-side hook is the opt-in allow_hardware kwarg.

The segments are intrinsically serial (segment k+1 is built from segment
k's measured counts), so the run is N serial QPU invocations.  On hardware
they are wrapped in one held Session so the device is queued once.

Sim-testable end to end with no IBM credentials:
    python burgers_ch_hw_runner.py --case smooth --target sim --dry-run
    python burgers_ch_hw_runner.py --case smooth --target sim \\
        --backend-name fake_sherbrooke --measure-mitigation

See docs/archive/F12-HARDWARE-RUNNER-SPEC.md.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Local src modules (burgers_*).  The package layout puts these alongside
# this file; the solver runs them the same way.
from burgers_classical import initial_condition_sine, solve_burgers
from burgers_cole_hopf import cole_hopf_inverse
from burgers_cole_hopf_circuit import run_cole_hopf_circuit_simulation

from q8020_cfd_qutil.backend import get_backend
from q8020_cfd_qutil.circuit import transpile_circuit


# ---------------------------------------------------------------------------
# Cases: parameters mirrored verbatim from the q4 hardware-probe TOMLs
# (q8020_burgers_ab_shock_q4.toml, q8020_burgers_ab_show_q4_ch_smooth.toml).
# n_segments = n_steps / segment_size = the number of serial QPU calls.
# ---------------------------------------------------------------------------
CASES: dict[str, dict[str, Any]] = {
    "shock": {
        "q": 4, "nu": 0.03, "ic_amplitude": 0.4, "cfl": 0.1,
        "n_steps": 60, "segment_size": 5, "save_every": 10,
        "bc": "periodic", "propagator": "qft-diagonal",
        "bond_dim": 8, "phi_modes": 8,
    },
    "smooth": {
        "q": 4, "nu": 0.08, "ic_amplitude": 0.3, "cfl": 0.1,
        "n_steps": 30, "segment_size": 5, "save_every": 10,
        "bc": "periodic", "propagator": "qft-diagonal",
        "bond_dim": 8, "phi_modes": 8,
    },
}


def build_inputs(case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray,
                                                 float, float]:
    """Construct (u0, x, dt, nu) exactly as burgers_solver does for a
    sine IC on a periodic grid."""
    q = case["q"]
    N = 2 ** q
    x = np.linspace(0, 1, N, endpoint=False)  # periodic grid
    dx = x[1] - x[0]
    dt = case["cfl"] * dx
    u0 = initial_condition_sine(x) * case["ic_amplitude"]
    return u0, x, dt, case["nu"]


def build_sampler_options(measure_mitigation: bool,
                          dynamical_decoupling: bool) -> dict | None:
    """Nested SamplerV2 options dict for TREX + DD, or None if both off."""
    opts: dict[str, Any] = {}
    if measure_mitigation:
        # TREX: twirled readout-error extinction.
        opts["twirling"] = {"enable_measure": True}
    if dynamical_decoupling:
        opts["dynamical_decoupling"] = {
            "enable": True, "sequence_type": "XpXm",
        }
    return opts or None


def open_session(backend: Any, target: str, use_session: bool):
    """Return a qiskit_ibm_runtime Session bound to backend, or None.

    Sessions apply to the SamplerV2 path (target=hardware and target=local,
    the credential-free FakeBackendV2 local-testing mode).  The plain sim
    path runs through AerSimulator, which ignores sessions -> None.
    Returns a context-manager-or-None.
    """
    if target not in ("hardware", "local") or not use_session:
        return None
    from qiskit_ibm_runtime import Session
    return Session(backend=backend)


def dry_run_transpile(case: dict[str, Any], backend: Any,
                      optimization_level: int, seed: int) -> dict[str, Any]:
    """Build the first segment's circuit (via the SAME build_segment_circuit
    the solver loop uses) and transpile it against the resolved backend to
    report the true (heavy-hex) CX/depth.  No shots.

    Reusing build_segment_circuit guarantees the reported cost matches what
    actually executes -- no duplicated prep/heat/compose logic to drift."""
    from burgers_cole_hopf import (
        cole_hopf_forward, log_phi_to_normalized_psi,
        cole_hopf_forward_centered, _should_center,
    )
    from burgers_cole_hopf_circuit import (
        build_segment_circuit, permute_to_encoding,
    )
    from burgers_mps import normalize_state

    u0, x, dt, nu = build_inputs(case)
    dx = x[1] - x[0]
    q = case["q"]
    L_box = float((2 ** q) * dx)
    seg = case["segment_size"]

    # Forward CH transform -> psi0, mirroring run_cole_hopf_circuit_simulation
    # (centering policy must match so segment 0 is identical).
    if _should_center(u0, dx, nu):
        log_phi, _ = cole_hopf_forward_centered(u0, dx, nu)
        psi0 = log_phi_to_normalized_psi(log_phi)
    else:
        phi0 = cole_hopf_forward(u0, dx, nu)
        psi0 = phi0 / float(np.linalg.norm(phi0))
    psi0 = permute_to_encoding(psi0, q, "binary")
    psi_current, _ = normalize_state(psi0)

    raw_qc, total_q, _n_bond, _n_heat_anc = build_segment_circuit(
        psi_current, q, nu, dt, seg, L_box, case["bc"], case["propagator"],
        bond_dim=case["bond_dim"], encoding="binary",
    )

    qc_t, info = transpile_circuit(raw_qc, backend,
                                   optimization_level=optimization_level,
                                   seed_transpiler=seed)
    ops = qc_t.count_ops()
    cx = ops.get("cx", 0) + ops.get("ecr", 0) + ops.get("cz", 0)
    return {
        "segment_qubits": total_q,
        "raw_depth": raw_qc.depth(),
        "transpiled_depth": qc_t.depth(),
        "transpiled_2q_gates": int(cx),
        "transpiled_ops": {k: int(v) for k, v in ops.items()},
        "n_segments": case["n_steps"] // seg,
        "transpile_info": info,
    }


def compute_reference(u0: np.ndarray, x: np.ndarray, nu: float,
                      dt: float, n_steps: int, bc: str) -> list[np.ndarray]:
    """FTCS classical reference, per-step [u0..u_{n_steps}] on the grid."""
    return solve_burgers(u0, x, nu, dt, n_steps, bc=bc)


def write_movie_series(parent: Path, method: str, x: np.ndarray,
                       sol_by_step: dict[int, np.ndarray], nu: float,
                       dt: float, n_steps: int, bc: str) -> None:
    """Write one method's per-step snapshots as a standalone case dir under
    *parent*, in the exact fragment shape plot_method_compare.py consumes
    (a 'case' fragment carrying method + a 'solution_steps'/'grid' artifact).

    Each series lives in its own subdir so the movie postproc, pointed at
    *parent* via --sweep-dir, discovers them as separate case dirs."""
    from q8020_cfd_metautil.meta_fragment import (
        make_experiment_meta, make_case_meta, write_experiment,
        write_case, write_artifacts,
    )
    cd = parent / method
    cd.mkdir(parents=True, exist_ok=True)
    write_experiment(cd, make_experiment_meta(name=f"ch_hw_{method}"))
    # _source='solver' so _extract_case_params picks this dict; 'method' is
    # the series key the movie styles/labels by.
    write_case(cd, make_case_meta(
        name="burgers_cole_hopf_hw", method=method,
        q=int(np.log2(len(x))), nu=nu, dt=dt, n_steps=n_steps, bc=bc,
        _source="solver",
    ))
    write_artifacts(cd, {
        "grid": x.tolist(),
        "solution_steps": {
            str(s): np.asarray(v).real.tolist()
            for s, v in sol_by_step.items()
        },
    })


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    nb = np.linalg.norm(b)
    if nb < 1e-15:
        return float("nan")
    return float(np.linalg.norm(a - b) / nb)


def json_safe(obj: Any) -> Any:
    """Coerce a nested structure to JSON-serializable form.

    job.metrics() / usage payloads contain datetime and other non-JSON
    objects; the metautil fragment writer has no default serializer, so
    sanitize before storing.  Dicts/lists recurse; primitives pass
    through; anything else is stringified.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return str(obj)


def write_metadata(outdir: Path, case_name: str, case: dict[str, Any],
                   args: argparse.Namespace, backend: Any,
                   metrics: list[dict[str, Any]], u_final: np.ndarray,
                   ref_l2: float, dry: dict | None,
                   wall_s: float) -> None:
    """Write the q8020 metadata bundle via metautil writers so output is
    sweep/harvest/postproc compatible."""
    from q8020_cfd_metautil.meta_fragment import (
        make_experiment_meta, make_case_meta, make_code_meta,
        make_backend_meta, write_experiment, write_case, write_code,
        write_backend, write_results, write_analysis,
    )

    exp = make_experiment_meta(name=f"ch_hw_{case_name}")
    write_experiment(outdir, exp)
    write_case(outdir, make_case_meta(
        name="burgers_cole_hopf_hw", case=case_name, **case,
        target=args.target, shots=args.shots,
        measure_mitigation=args.measure_mitigation,
        dynamical_decoupling=args.dynamical_decoupling,
    ))
    write_code(outdir, make_code_meta(
        algorithm="cole_hopf_circuit",
        entry_point="burgers_ch_hw_runner.py",
        run_args=vars(args),
    ))
    if backend is not None and args.target in ("hardware", "local"):
        try:
            write_backend(outdir, make_backend_meta(backend))
        except Exception as e:
            print(f"[ch_hw] backend meta skipped: {e}", file=sys.stderr)

    # Real-hardware only: harvest the per-segment execution-time metrics +
    # calibration snapshot via the qutil builtin (get_job_result), keyed on
    # the job_ids we already captured.  This is the authoritative billed-
    # usage + calibration record (calibration can drift across the serial
    # chain's queue waits), and it uses the purpose-built async helper
    # rather than re-deriving anything here.  Best-effort: never fail the
    # run on a harvest hiccup.
    if args.target == "hardware":
        try:
            from q8020_cfd_qutil.job import get_job_result
            harvest = []
            for rec in seg_records:
                jid = rec.get("job_id")
                if not jid:
                    continue
                res = get_job_result(jid, token=args.token,
                                     channel=args.channel,
                                     instance=args.instance)
                harvest.append({
                    "segment_idx": rec.get("segment_idx"),
                    "job_id": jid,
                    "status": res.get("status"),
                    "metrics": json_safe(res.get("metrics")),
                    "backend_name": res.get("backend_name"),
                    "backend": json_safe(res.get("backend")),
                })
            if harvest:
                write_results(outdir, {
                    "job_harvest": harvest,
                    "_source": "burgers_ch_hw_runner.get_job_result",
                }, index=1)
        except Exception as e:
            print(f"[ch_hw] job harvest skipped: {e}", file=sys.stderr)

    # Per-segment job_ids + post-selection for audit/reconstruction.
    seg_records = []
    total_quantum_seconds = 0.0
    any_quantum_seconds = False
    for m in metrics:
        ex = m.get("execute", {})
        qs = ex.get("quantum_seconds")
        if qs is not None:
            any_quantum_seconds = True
            total_quantum_seconds += float(qs)
        seg_records.append({
            "segment_idx": m.get("segment_idx"),
            "step": m.get("step"),
            "p_success": m.get("p_success"),
            "n_kept": m.get("n_kept"),
            "circuit_depth": m.get("circuit_depth"),
            "gate_counts": m.get("gate_counts"),
            # Honesty labels: hardware-cost stats are exact only for the
            # segment that produced them; reused as an estimate otherwise
            # (None/False when skipped under a held Session -- see dry-run).
            "circuit_metrics_available": m.get("circuit_metrics_available"),
            "circuit_metrics_from_segment": m.get(
                "circuit_metrics_from_segment"),
            "circuit_metrics_exact": m.get("circuit_metrics_exact"),
            "job_id": ex.get("job_id"),
            "execution_time_s": m.get("execution_time_s"),
            # Billed QPU usage (real hardware only; None on sim/local).
            "quantum_seconds": qs,
            "job_metrics": json_safe(ex.get("job_metrics")),
        })
    write_results(outdir, {
        "u_final_method": u_final.tolist(),
        "segments": seg_records,
        "_source": "burgers_ch_hw_runner",
    })
    write_analysis(outdir, {
        "case": case_name,
        "target": args.target,
        "backend_name": args.backend_name,
        "n_segments": case["n_steps"] // case["segment_size"],
        "shots_per_segment": args.shots,
        "rel_l2_vs_ftcs": ref_l2,
        "measure_mitigation": args.measure_mitigation,
        "dynamical_decoupling": args.dynamical_decoupling,
        "dry_run_transpile": dry,
        "wall_time_s": wall_s,
        # Billed QPU time summed across the serial segments (real hardware
        # only; None when no segment reported usage -- sim/local).
        "total_quantum_seconds": (
            total_quantum_seconds if any_quantum_seconds else None
        ),
        "_source": "burgers_ch_hw_runner",
    })


def main() -> int:
    p = argparse.ArgumentParser(
        description="Cole-Hopf measure-reprepare hardware runner (F12)")
    p.add_argument("--case", choices=sorted(CASES), required=True)
    p.add_argument("--target", choices=["sim", "local", "hardware"],
                   default="sim",
                   help="sim = AerSimulator (optional fake-backend noise). "
                        "local = SamplerV2(mode=FakeBackendV2) local-testing "
                        "mode: exercises the EXACT hardware code path "
                        "(SamplerV2 + Session + TREX/DD) with NO credentials. "
                        "hardware = real IBM Quantum via QiskitRuntimeService.")
    p.add_argument("--backend-name", default=None,
                   help="IBM backend (hardware) or fake-backend name (sim/"
                        "local noise), e.g. ibm_brisbane / manila.")
    p.add_argument("--shots", type=int, default=150000)
    p.add_argument("--segments", type=int, default=None,
                   help="Override segment count (debug); default = case "
                        "n_steps/segment_size.")
    p.add_argument("--optimization-level", type=int, default=None,
                   help="Default 3 for hardware, 1 for sim.")
    p.add_argument("--dry-run", action="store_true",
                   help="Transpile + report CX/depth, then stop (no shots).")
    p.add_argument("--no-session", action="store_true",
                   help="Disable the held Session (N separate jobs).")
    p.add_argument("--measure-mitigation", choices=["auto", "on", "off"],
                   default="auto",
                   help="TREX measurement mitigation. auto (default) = on "
                        "for the SamplerV2 paths (hardware/local), off for "
                        "Aer sim.")
    p.add_argument("--dynamical-decoupling", choices=["auto", "on", "off"],
                   default="auto",
                   help="Dynamical decoupling. auto (default) = on for "
                        "hardware/local, off for Aer sim.")
    p.add_argument("--outdir", default="~/q8020")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--token", default=None)
    p.add_argument("--channel", default="ibm_cloud")
    p.add_argument("--instance", default=None)
    args = p.parse_args()

    # Resolve tri-state mitigation flags to bools.  Only the SamplerV2
    # paths (hardware, local) honor mitigation options; the Aer sim path
    # ignores them, so 'auto' is off there and an explicit 'on' for sim is
    # rejected (it would only mislabel the run -- see metadata below).
    sampler_path = args.target in ("hardware", "local")
    if args.measure_mitigation == "on" and not sampler_path:
        p.error("--measure-mitigation on is only valid for --target "
                "hardware/local (Aer sim ignores mitigation options).")
    if args.dynamical_decoupling == "on" and not sampler_path:
        p.error("--dynamical-decoupling on is only valid for --target "
                "hardware/local (Aer sim ignores mitigation options).")
    measure_mitigation = (
        args.measure_mitigation == "on"
        or (args.measure_mitigation == "auto" and sampler_path)
    )
    dynamical_decoupling = (
        args.dynamical_decoupling == "on"
        or (args.dynamical_decoupling == "auto" and sampler_path)
    )
    # Stamp the resolved booleans back so metadata reflects what actually
    # ran, not the raw auto/on/off request.
    args.measure_mitigation = measure_mitigation
    args.dynamical_decoupling = dynamical_decoupling
    if args.optimization_level is None:
        args.optimization_level = 3 if sampler_path else 1

    case = dict(CASES[args.case])
    if args.segments is not None:
        case["n_steps"] = args.segments * case["segment_size"]
    n_segments = case["n_steps"] // case["segment_size"]

    print(f"[ch_hw] case={args.case} target={args.target} "
          f"backend={args.backend_name} segments={n_segments} "
          f"(serial QPU invocations) shots/seg={args.shots} "
          f"mitigation={args.measure_mitigation} dd={args.dynamical_decoupling}",
          file=sys.stderr, flush=True)

    # Build backend.
    if args.target == "hardware":
        backend = get_backend(name=args.backend_name, backend_type="hardware",
                              token=args.token, channel=args.channel,
                              instance=args.instance)
    elif args.target == "local":
        # FakeBackendV2 directly (not wrapped in Aer): routes through
        # SamplerV2 local-testing mode, the credential-free hardware proxy.
        # Needs >= 10 qubits for the q4 circuit; default to a 27q V2 fake.
        # Topology fidelity is secondary here (this validates the code
        # path, not the error budget -- use --target hardware --dry-run
        # for the true Boston heavy-hex CX/depth).
        name = args.backend_name or "cairo"
        backend = get_backend(name=name, backend_type="fake")
        if args.measure_mitigation or args.dynamical_decoupling:
            print("[ch_hw] NOTE: --target local validates the SamplerV2 + "
                  "Session + options PLUMBING, but qiskit local-testing "
                  "mode IGNORES TREX/DD numerically (options have no effect). "
                  "To see mitigation's numerical benefit, run on real "
                  "hardware, or use --target sim with a noise model to study "
                  "the noisy floor.", file=sys.stderr, flush=True)
    else:
        # sim: optional fake-backend noise via name; None = ideal.
        backend = get_backend(name=args.backend_name, backend_type="sim")

    # Pre-flight transpile (heavy-hex when hardware).  Run it when it is
    # actually useful: for --dry-run, and for the SamplerV2 paths where it
    # is the authoritative segment CX/depth (the in-loop metric transpile
    # is skipped under a held Session).  On plain Aer sim the in-loop metric
    # stats already report honest CX, so skip the redundant transpile.
    dry = None
    if args.dry_run or sampler_path:
        dry = dry_run_transpile(case, backend, args.optimization_level,
                                args.seed)
        print(f"[ch_hw] transpiled: {dry['transpiled_2q_gates']} 2q gates, "
              f"depth {dry['transpiled_depth']}, {dry['segment_qubits']} "
              f"qubits x {n_segments} segments",
              file=sys.stderr, flush=True)
    if args.dry_run:
        print("[ch_hw] --dry-run: stopping before execution.",
              file=sys.stderr, flush=True)
        import json
        print(json.dumps(dry, indent=2, default=str))
        return 0

    sampler_options = build_sampler_options(
        args.measure_mitigation, args.dynamical_decoupling)

    u0, x, dt, nu = build_inputs(case)
    dx = x[1] - x[0]

    # A held Session is mandatory on real hardware (N serial segments must
    # share one queue slot); the solver enforces this too, but fail early
    # here with a clear message before any backend work.
    if args.target == "hardware" and args.no_session:
        p.error("--no-session is not allowed with --target hardware: the "
                "serial segments would each open a separate queued job.")

    t0 = time.time()
    session_cm = open_session(backend, args.target, not args.no_session)

    # backend_type drives the CH-internal hardware guard.  Only the real
    # hardware target should set "hardware" (with allow_hardware to opt
    # past the v1 sim-only guard).  The local SamplerV2 proxy keeps
    # backend_type="sim" so no other hardware-only paths are triggered;
    # the SamplerV2 branch is selected automatically by the non-Aer
    # backend object in execute_circuit_counts.
    ch_backend_type = "hardware" if args.target == "hardware" else "sim"

    def _run() -> tuple[list[np.ndarray], list[dict[str, Any]]]:
        return run_cole_hopf_circuit_simulation(
            u0, x, nu, dt, case["n_steps"],
            bc=case["bc"], propagator=case["propagator"],
            snapshot_interval=case["save_every"],
            shots=args.shots, bond_dim=case["bond_dim"],
            encoding="binary", backend=backend,
            backend_type=ch_backend_type,
            backend_name=args.backend_name,
            optimization_level=args.optimization_level, seed=args.seed,
            evolution_mode="measure_reprepare",
            segment_size=case["segment_size"], phi_modes=case["phi_modes"],
            allow_hardware=(args.target == "hardware"),
            session=session_cm, sampler_options=sampler_options,
        )

    if session_cm is not None:
        with session_cm:
            solutions, metrics = _run()
    else:
        solutions, metrics = _run()
    wall_s = time.time() - t0

    n_steps = case["n_steps"]
    u_final = solutions[n_steps]
    ref_steps = compute_reference(u0, x, nu, dt, n_steps, case["bc"])
    ref_l2 = rel_l2(np.asarray(u_final), np.asarray(ref_steps[-1]))
    print(f"[ch_hw] done in {wall_s:.1f}s  rel-L2 vs FTCS = {ref_l2:.4f}",
          file=sys.stderr, flush=True)

    outdir = Path(os.path.expanduser(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)
    write_metadata(outdir, args.case, case, args, backend, metrics,
                   np.asarray(u_final), ref_l2, dry, wall_s)

    # Movie series: write the quantum result and the FTCS reference as two
    # standalone case dirs under <outdir>/method_compare so the
    # plot_method_compare.py postproc can render the evolution GIF.
    # `solutions` is step-indexed (NaN-fill at non-snapshot steps); keep
    # only the finite snapshots the solver actually stored, plus step 0.
    mc_dir = outdir / "method_compare"
    q_steps = {0: np.asarray(solutions[0])}
    for s in range(1, n_steps + 1):
        v = np.asarray(solutions[s])
        if np.all(np.isfinite(v)):
            q_steps[s] = v
    write_movie_series(mc_dir, "cole_hopf_circuit", x, q_steps,
                       nu, dt, n_steps, case["bc"])
    ref_by_step = {s: np.asarray(ref_steps[s]) for s in range(n_steps + 1)}
    write_movie_series(mc_dir, "ftcs_reference", x, ref_by_step,
                       nu, dt, n_steps, case["bc"])
    print(f"[ch_hw] metadata + movie series written to {outdir}",
          file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
