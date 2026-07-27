"""PostProcessor for the Burgers equation solver.

Collects per-step metrics (circuit depth, gate counts, timings) and
writes metadata fragments via q8020_cfd_metautil at finalization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from q8020_cfd_metautil.meta_fragment import (
    make_case_meta,
    write_analysis,
    write_artifacts,
    write_case,
    write_results,
)
from q8020_cfd_metautil.solverfw import PostProcessor

from lib_fd import compute_error


class BurgersPostProcessor(PostProcessor):
    """Collect step metrics and write output fragments.

    Parameters
    ----------
    classical_solutions : list[np.ndarray] or None
        Solution snapshots from the classical baseline solver, or None
        when ``--no-classical-reference`` is used.  When None, error
        metrics and ``u_final_classical`` are emitted as null/NaN and
        ``speedup_ratio`` is omitted.
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
        classical_solutions: list[np.ndarray] | None,
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

    # ------------------------------------------------------------------
    # PostProcessor interface
    # ------------------------------------------------------------------

    def on_step(self, step, t, state, metrics):
        """Record per-step metrics."""
        if metrics:
            self.step_metrics.append(metrics)

    def finalize(self, config, elapsed_s):
        """Called at the end of the main loop (no-op here; use
        ``write_fragments`` for full output)."""

    # ------------------------------------------------------------------
    # Full fragment output
    # ------------------------------------------------------------------

    def write_fragments(
        self,
        config,
        u0: np.ndarray,
        solutions: list[np.ndarray],
        t_classical: float | None,
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
        t_classical : float | None
            Wall-clock time for the classical baseline (None if skipped).
        t_method : float
            Wall-clock time for the method.
        args : argparse.Namespace
            CLI arguments (for experiment_id, backend, etc.).

        Returns
        -------
        dict
            JSON-serialisable summary for the sweeper.
        """
        n_steps = config.n_steps
        x = self.x
        save_steps = self.save_steps

        # Compute errors (only when a classical reference was run)
        errors: dict[int, float] = {}
        if self.classical_solutions is not None:
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
            ic_center=(
                config.ic_center if config.ic == "gaussian" else None
            ),
            ic_sigma=(
                config.ic_sigma if config.ic == "gaussian" else None
            ),
            ic_cole_hopf_coeffs=(
                config.ic_cole_hopf_coeffs
                if config.ic == "cole_hopf_exact"
                else None
            ),
            classical_reference=config.classical_reference,
            analytic_reference=(
                config.analytic_reference
                if config.ic == "cole_hopf_exact"
                else None
            ),
            source=config.source, method=config.method,
            trotter_order=config.trotter_order,
            bond_dim=config.bond_dim,
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
            "u_final_classical": (
                np.real(np.array(self.classical_solutions[-1])).tolist()
                if self.classical_solutions is not None
                else None
            ),
            "u_final_method": np.real(
                np.array(solutions[-1]),
            ).tolist(),
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
            "speedup_ratio": (
                t_classical / max(t_method, 1e-9)
                if t_classical is not None
                else None
            ),
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
        # n_pauli_terms only describes methods that emit it as a per-step
        # metric from their own run (see lib_fd for the shift method).
        # Recomputing it here for other methods is meaningless (ftcs/lbm have no
        # circuit; cole_hopf/qlbm build different circuits) and costs a dense
        # 4^q-basis solve that blows up at q>=8.  Take the value from the method
        # when it reported one; otherwise leave it null.
        if step_metrics_npt := next(
            (
                m["n_pauli_terms"]
                for m in self.step_metrics
                if m.get("n_pauli_terms") is not None
            ),
            None,
        ):
            analysis_data["n_pauli_terms"] = step_metrics_npt

        step_metrics = self.step_metrics
        if step_metrics:
            analysis_data["total_pauli_decomp_time"] = sum(
                m.get("pauli_decomp_time_s", 0) for m in step_metrics
            )
            analysis_data["total_circuit_construction_time"] = sum(
                m.get("circuit_construction_time_s", 0)
                for m in step_metrics
            )
            # Per-step circuit stats may be flat (lib_fd:
            # circuit_depth/gate_counts/n_qubits) or nested under
            # transpile.{after,wall_time} (cole_hopf_circuit / qlbm_circuit
            # via the qutil transpile helper).  Normalise both shapes.
            def _circ(m):
                after = (m.get("transpile") or {}).get("after") or {}
                depth = m.get("circuit_depth", after.get("depth"))
                gates = m.get("gate_counts", after.get("gate_counts"))
                nq = m.get("n_qubits", after.get("num_qubits"))
                return depth, gates, nq

            def _ttime(m):
                return m.get(
                    "transpilation_time_s",
                    (m.get("transpile") or {}).get("wall_time", 0),
                )

            def _etime(m):
                return m.get(
                    "execution_time_s",
                    (m.get("execute") or {}).get("wall_time", 0),
                )

            analysis_data["total_transpilation_time"] = sum(
                _ttime(m) for m in step_metrics
            )
            analysis_data["total_execution_time"] = sum(
                _etime(m) for m in step_metrics
            )
            # qlbm-specific costs broken out so the total is fully
            # accounted (zero for methods that don't use them).
            analysis_data["total_sign_recovery_time"] = sum(
                m.get("sign_recovery_time_s", 0) for m in step_metrics
            )
            analysis_data["total_metric_transpile_time"] = sum(
                m.get("metric_transpile_time_s", 0) for m in step_metrics
            )
            # Total circuit->backend submissions (qlbm's Hadamard
            # recovery fires many per step; CH is one per segment/snap).
            analysis_data["n_circuits_executed"] = sum(
                m.get("n_circuits", 0) for m in step_metrics
            )
            circ = [_circ(m) for m in step_metrics]
            depths = [d for d, _, _ in circ if d is not None]
            analysis_data["avg_circuit_depth"] = (
                sum(depths) / len(depths) if depths else None
            )
            gate_dicts = [g for _, g, _ in circ if g]
            gate_totals = [sum(g.values()) for g in gate_dicts]
            analysis_data["avg_gate_count"] = (
                sum(gate_totals) / len(gate_totals)
                if gate_totals
                else None
            )
            analysis_data["avg_cx_gates"] = (
                sum(g.get("cx", 0) for g in gate_dicts) / len(gate_dicts)
                if gate_dicts
                else None
            )
            # Sign-recovery (qlbm Hadamard-test) circuit cost, reported
            # per bin circuit and as the honest total across all bins
            # (n_bins per step = n_circuits - 1, the main step circuit).
            sr_depths = [
                m["sign_recovery_depth_per_circuit"] for m in step_metrics
                if m.get("sign_recovery_depth_per_circuit") is not None
            ]
            analysis_data["avg_sign_recovery_depth_per_circuit"] = (
                sum(sr_depths) / len(sr_depths) if sr_depths else None
            )
            sr_cx = [
                m["sign_recovery_gate_counts_per_circuit"].get("cx", 0)
                for m in step_metrics
                if m.get("sign_recovery_gate_counts_per_circuit")
            ]
            analysis_data["avg_sign_recovery_cx_per_circuit"] = (
                sum(sr_cx) / len(sr_cx) if sr_cx else None
            )
            analysis_data["total_sign_recovery_cx"] = sum(
                m["sign_recovery_gate_counts_per_circuit"].get("cx", 0)
                * max(m.get("n_circuits", 1) - 1, 0)
                for m in step_metrics
                if m.get("sign_recovery_gate_counts_per_circuit")
            )
            nqs = [nq for _, _, nq in circ if nq is not None]
            analysis_data["n_qubits"] = nqs[0] if nqs else None
            # Coverage: how many iterations actually yielded circuit
            # stats (e.g. a crash-isolated transpile may have degraded).
            analysis_data["circuit_metrics_n_available"] = sum(
                1 for d, _, _ in circ if d is not None
            )
            analysis_data["circuit_metrics_n_total"] = len(circ)
            analysis_data["sign_recovery"] = step_metrics[0].get(
                "sign_recovery", "none",
            )
            analysis_data["per_step_metrics"] = step_metrics
        write_analysis(outdir, analysis_data, experiment_id=exp_id)

        # Artifacts fragment
        artifacts_data = {
            "grid": x.tolist() if x is not None else [],
            "solution_steps": {
                str(s): np.real(np.array(solutions[s])).tolist()
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
            "ic_center": (
                config.ic_center if config.ic == "gaussian" else None
            ),
            "ic_sigma": (
                config.ic_sigma if config.ic == "gaussian" else None
            ),
            "ic_cole_hopf_coeffs": (
                config.ic_cole_hopf_coeffs
                if config.ic == "cole_hopf_exact"
                else None
            ),
            "classical_reference": config.classical_reference,
            "analytic_reference": (
                config.analytic_reference
                if config.ic == "cole_hopf_exact"
                else None
            ),
            "method": config.method,
            "bc": config.bc,
            "trotter_order": config.trotter_order,
            "qalb_collision_trotter_reps": getattr(
                config, "qalb_collision_trotter_reps", 0),
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
