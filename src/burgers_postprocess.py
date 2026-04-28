"""PostProcessor for the Burgers equation solver.

Collects per-step metrics (circuit depth, gate counts, timings) and
writes metadata fragments via q8020_cfd_metautil at finalization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from q8020_cfd_metautil.solverfw import PostProcessor


class BurgersPostProcessor(PostProcessor):
    """Collect step metrics and write output fragments.

    Parameters
    ----------
    classical_solutions : list[np.ndarray]
        Solution snapshots from the classical baseline solver.
    source_fn : callable or None
        Source function g(x, t) used in the simulation.
    x : np.ndarray
        Grid coordinates (for artifact output).
    output_dir : Path or str
        Directory for metadata fragments.
    experiment_id : str or None
        Experiment identifier for the sweeper.
    save_steps : list[int]
        Which steps to save solution snapshots for.
    """

    def __init__(
        self,
        classical_solutions: list[np.ndarray],
        source_fn=None,
        x: np.ndarray | None = None,
        output_dir: Path | str = ".",
        experiment_id: str | None = None,
        save_steps: list[int] | None = None,
    ) -> None:
        self.classical_solutions = classical_solutions
        self.source_fn = source_fn
        self.x = x
        self.output_dir = Path(output_dir)
        self.experiment_id = experiment_id
        self.save_steps = save_steps or [0]

        self.step_metrics: list[dict[str, Any]] = []
        self.solutions: list[np.ndarray] = []
        self._latest_step = 0

    # ------------------------------------------------------------------
    # PostProcessor interface
    # ------------------------------------------------------------------

    def on_step(self, step, t, state, metrics):
        """Record per-step metrics."""
        if metrics:
            self.step_metrics.append(metrics)
        self._latest_step = step

    def finalize(self, config, elapsed_s):
        """Called at the end of the main loop (no-op here; use
        ``write_fragments`` for full output)."""
        pass

    # ------------------------------------------------------------------
    # Full fragment output
    # ------------------------------------------------------------------

    def write_fragments(
        self,
        config,
        u0: np.ndarray,
        solutions: list[np.ndarray],
        t_classical: float,
        t_method: float,
        args,
    ) -> dict[str, Any]:
        """Write all metadata fragments and return JSON summary.

        Parameters
        ----------
        config : BurgersConfig
        u0 : np.ndarray
            Initial condition.
        solutions : list[np.ndarray]
            All solution snapshots from the method run.
        t_classical : float
            Wall-clock time for the classical baseline.
        t_method : float
            Wall-clock time for the method.
        args : argparse.Namespace
            CLI arguments (for experiment_id, backend, etc.).

        Returns
        -------
        dict
            JSON-serialisable summary for the sweeper.
        """
        from burgers_trotter import compute_error
        from q8020_cfd_metautil.meta_fragment import (
            make_case_meta,
            write_analysis,
            write_artifacts,
            write_case,
            write_results,
        )

        n_steps = config.n_steps
        x = self.x
        save_steps = self.save_steps

        # Compute errors
        errors = {}
        for step in save_steps:
            u_m = solutions[step]
            if not np.all(np.isfinite(u_m)):
                continue
            errors[step] = compute_error(
                u_m, self.classical_solutions[step],
            )

        final_error = errors.get(n_steps, float("nan"))
        max_error = max(errors.values()) if errors else float("nan")

        outdir = self.output_dir
        exp_id = self.experiment_id

        # Case fragment
        case_data = make_case_meta(
            name="burgers_1d",
            q=config.q, N=1 << config.q,
            nu=config.nu,
            dx=float(x[1] - x[0]) if x is not None else None,
            dt=float(config.dt) if config.dt else None,
            cfl=config.cfl, n_steps=n_steps,
            shock_pct=config.shock_pct,
            ic=config.ic,
            ic_modes=(
                config.ic_modes if config.ic == "multimode" else None
            ),
            ic_seed=(
                config.ic_seed if config.ic == "multimode" else None
            ),
            ic_alpha=(
                config.ic_alpha if config.ic == "multimode" else None
            ),
            source=config.source, method=config.method,
            trotter_order=config.trotter_order,
            trotter_reps=config.trotter_reps,
            bond_dim=config.bond_dim,
            mps_threshold=config.mps_threshold,
            shots=config.shots,
            backend_name=config.backend_name,
            t1=config.t1, t2=config.t2,
            bc=config.bc,
            sign_recovery=config.sign_recovery,
            backend_type=config.backend_type,
            coupling_map=config.coupling_map,
            optimization_level=config.optimization_level,
            seed=config.seed,
        )
        write_case(outdir, case_data, experiment_id=exp_id)

        # Results fragment
        results_data = {
            "u_initial": u0.tolist(),
            "u_final_classical": np.array(
                self.classical_solutions[-1],
            ).real.tolist(),
            "u_final_method": np.array(
                solutions[-1],
            ).real.tolist(),
            "errors_by_step": {
                str(k): v for k, v in errors.items()
            },
            "final_error": final_error,
            "max_error": max_error,
        }
        write_results(outdir, results_data, experiment_id=exp_id)

        # Analysis fragment
        analysis_data = {
            "final_error_epsilon": final_error,
            "max_error_epsilon": max_error,
            "classical_wall_time_s": t_classical,
            "method_wall_time_s": t_method,
            "speedup_ratio": t_classical / max(t_method, 1e-9),
            "n_pauli_terms": None,
            "shots": config.shots,
            "t1": getattr(config, "t1", None),
            "t2": getattr(config, "t2", None),
            "backend_name": config.backend_name,
            "backend_type": config.backend_type,
            "coupling_map": config.coupling_map,
            "optimization_level": config.optimization_level,
            "seed": config.seed,
        }
        if config.method != "shift" and x is not None:
            from burgers_nonlinear import build_evolution_hamiltonian

            g0 = (
                self.source_fn(x, 0)
                if self.source_fn is not None
                else None
            )
            dx = float(x[1] - x[0])
            dt = float(config.dt) if config.dt else 0.0
            H = build_evolution_hamiltonian(u0, dx, dt, config.nu, g0)
            analysis_data["n_pauli_terms"] = len(H)

        step_metrics = self.step_metrics
        if step_metrics:
            n = len(step_metrics)
            analysis_data["total_pauli_decomp_time"] = sum(
                m.get("pauli_decomp_time_s", 0) for m in step_metrics
            )
            analysis_data["total_circuit_construction_time"] = sum(
                m.get("circuit_construction_time_s", 0)
                for m in step_metrics
            )
            analysis_data["total_transpilation_time"] = sum(
                m.get("transpilation_time_s", 0)
                for m in step_metrics
            )
            analysis_data["total_execution_time"] = sum(
                m.get("execution_time_s", 0) for m in step_metrics
            )
            depths = [
                m["circuit_depth"]
                for m in step_metrics
                if "circuit_depth" in m
            ]
            analysis_data["avg_circuit_depth"] = (
                sum(depths) / len(depths) if depths else None
            )
            gate_totals = [
                sum(m["gate_counts"].values())
                for m in step_metrics
                if "gate_counts" in m
            ]
            analysis_data["avg_gate_count"] = (
                sum(gate_totals) / len(gate_totals)
                if gate_totals
                else None
            )
            analysis_data["avg_cx_gates"] = (
                sum(
                    m["gate_counts"].get("cx", 0)
                    for m in step_metrics
                    if "gate_counts" in m
                )
                / n
            )
            analysis_data["n_qubits"] = (
                step_metrics[0].get("n_qubits")
                if step_metrics
                else None
            )
            analysis_data["sign_recovery"] = step_metrics[0].get(
                "sign_recovery", "none",
            )
            analysis_data["per_step_metrics"] = step_metrics
        write_analysis(outdir, analysis_data, experiment_id=exp_id)

        # Artifacts fragment
        artifacts_data = {
            "grid": x.tolist() if x is not None else [],
            "solution_steps": {
                str(s): np.array(solutions[s]).real.tolist()
                for s in save_steps
            },
        }
        write_artifacts(outdir, artifacts_data, experiment_id=exp_id)

        # JSON summary
        summary = {
            "experiment_id": exp_id,
            "algorithm": f"burgers_{config.method}",
            "q": config.q,
            "N": 1 << config.q,
            "nu": config.nu,
            "n_steps": n_steps,
            "shock_pct": config.shock_pct,
            "ic": config.ic,
            "ic_modes": (
                config.ic_modes
                if config.ic == "multimode"
                else None
            ),
            "ic_seed": (
                config.ic_seed
                if config.ic == "multimode"
                else None
            ),
            "ic_alpha": (
                config.ic_alpha
                if config.ic == "multimode"
                else None
            ),
            "method": config.method,
            "bc": config.bc,
            "trotter_order": config.trotter_order,
            "trotter_reps": config.trotter_reps,
            "shots": config.shots,
            "t1": getattr(config, "t1", None),
            "t2": getattr(config, "t2", None),
            "backend_name": config.backend_name,
            "final_error": final_error,
            "max_error": max_error,
            "classical_time_s": t_classical,
            "method_time_s": t_method,
            "exit_code": 0,
        }
        return summary
