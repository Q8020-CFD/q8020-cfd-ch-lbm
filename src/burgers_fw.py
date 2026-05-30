"""solverfw bindings for the Burgers equation solver.

Provides BurgersConfig, ShiftFD spatial operator, and TimeIntegrator
wrappers for every existing evolution method. These are thin adapters
around the existing step functions in burgers_trotter.py -- the physics
code is unchanged.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from q8020_cfd_metautil.solverfw import (
    DenseState,
    Grid1D,
    PostProcessor,
    SolverConfig,
    SpatialOperator,
    TimeIntegrator,
)

from burgers_nonlinear import compute_rhs_shift


# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------


@dataclass
class BurgersConfig(SolverConfig):
    """Burgers-specific solver configuration."""

    q: int = 3
    nu: float = 1e-4
    ic: str = "sine"
    ic_modes: int = 6
    ic_seed: int = 42
    ic_alpha: float = 1.0
    ic_center: float = 0.5
    ic_sigma: float = 0.1
    ic_cole_hopf_coeffs: str | None = None
    source: str = "sine"
    classical_reference: bool = True
    analytic_reference: bool = True

    # Method-specific params
    trotter_order: int = 1
    trotter_reps: int = 1
    bond_dim: int | None = None
    mps_threshold: float = 0.0
    shots: int = 0
    backend_name: str | None = None
    backend_type: str = "sim"
    coupling_map: str = "default"
    optimization_level: int = 1
    seed: int | None = None
    t1: float | None = None
    t2: float | None = None
    sign_recovery: str = "none"
    splitting: str = "lie"
    propagator: str = "qft-diagonal"
    encoding: str = "binary"
    evolution_mode: str = "single"
    chunk_size: int = 10
    phi_modes: int = 0
    taylor_order: int = 4
    readout: str = "direct"
    shock_pct: float | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "equation": "burgers_1d",
            "q": self.q,
            "N": 1 << self.q,
            "nu": self.nu,
            "cfl": self.cfl,
            "bc": self.bc,
            "method": self.method,
            "ic": self.ic,
            "source": self.source,
            "n_steps": self.n_steps,
        }


# -----------------------------------------------------------------------
# Spatial operator
# -----------------------------------------------------------------------


class ShiftFD(SpatialOperator):
    """Central-difference RHS using shift operators S+/S-."""

    def compute_rhs(self, state, grid, config, t=0.0):
        u = state.to_dense()
        g = None
        if hasattr(config, "_source_fn") and config._source_fn is not None:
            g = config._source_fn(grid.xc, t)
        return compute_rhs_shift(u, grid.dx, config.nu, g, bc=grid.bc)


# -----------------------------------------------------------------------
# Time integrators -- thin wrappers around existing step functions
# -----------------------------------------------------------------------


class ShiftEulerIntegrator(TimeIntegrator):
    """Classical forward-Euler with shift-operator FD."""

    def step(self, state, spatial_op, grid, config, dt, t=0.0):
        from burgers_trotter import shift_euler_step

        u = state.to_dense()
        g = None
        if hasattr(config, "_source_fn") and config._source_fn:
            g = config._source_fn(grid.xc, t)
        u_new, metrics = shift_euler_step(
            u, grid.dx, dt, config.nu, g, bc=grid.bc,
        )
        return DenseState(u_new), metrics


class QuantumExactIntegrator(TimeIntegrator):
    """Pauli decomposition + exact matrix exponential."""

    def step(self, state, spatial_op, grid, config, dt, t=0.0):
        from burgers_trotter import quantum_exact_step

        u = state.to_dense()
        g = None
        if hasattr(config, "_source_fn") and config._source_fn:
            g = config._source_fn(grid.xc, t)
        u_new, metrics = quantum_exact_step(
            u, grid.dx, dt, config.nu, g, bc=grid.bc,
        )
        return DenseState(u_new), metrics


class QuantumCircuitIntegrator(TimeIntegrator):
    """Pauli decomposition + Trotterised circuit."""

    def __init__(self, backend: Any = None) -> None:
        self.backend = backend

    def step(self, state, spatial_op, grid, config, dt, t=0.0):
        from burgers_trotter import (
            dual_rail_quantum_step,
            quantum_circuit_step,
        )

        u = state.to_dense()
        g = None
        if hasattr(config, "_source_fn") and config._source_fn:
            g = config._source_fn(grid.xc, t)

        if config.sign_recovery == "dual_rail":
            u_new, metrics = dual_rail_quantum_step(
                u, grid.dx, dt, config.nu, g,
                config.trotter_order, config.trotter_reps,
                shots=config.shots, backend=self.backend,
                t1=config.t1, t2=config.t2,
                backend_name=config.backend_name, bc=grid.bc,
            )
        else:
            u_new, metrics = quantum_circuit_step(
                u, grid.dx, dt, config.nu, g,
                config.trotter_order, config.trotter_reps,
                shots=config.shots, backend=self.backend,
                t1=config.t1, t2=config.t2,
                backend_name=config.backend_name, bc=grid.bc,
                sign_recovery=config.sign_recovery,
            )
        return DenseState(u_new), metrics


class MPSIntegrator(TimeIntegrator):
    """MPS state-prep + exact Hamiltonian evolution."""

    def __init__(self, backend: Any = None) -> None:
        self.backend = backend

    def step(self, state, spatial_op, grid, config, dt, t=0.0):
        from burgers_trotter import mps_step

        u = state.to_dense()
        g = None
        if hasattr(config, "_source_fn") and config._source_fn:
            g = config._source_fn(grid.xc, t)
        u_new, metrics = mps_step(
            u, grid.dx, dt, config.nu, g,
            bond_dim=config.bond_dim,
            threshold=config.mps_threshold,
            shots=config.shots, bc=grid.bc,
            sign_recovery=config.sign_recovery,
            backend=self.backend,
        )
        return DenseState(u_new), metrics


class TEBDCircuitIntegrator(TimeIntegrator):
    """TEBD circuit: MPS state-prep + W-II gate layer."""

    def __init__(self, backend: Any = None) -> None:
        self.backend = backend

    def step(self, state, spatial_op, grid, config, dt, t=0.0):
        from burgers_trotter import tebd_circuit_step

        u = state.to_dense()
        g = None
        if hasattr(config, "_source_fn") and config._source_fn:
            g = config._source_fn(grid.xc, t)
        u_new, metrics = tebd_circuit_step(
            u, grid.xc, dt, config.nu, g,
            bond_dim=config.bond_dim,
            threshold=config.mps_threshold,
            shots=config.shots, backend=self.backend,
            sign_recovery=config.sign_recovery, bc=grid.bc,
            splitting=config.splitting,
            readout=getattr(config, "readout", "direct"),
        )
        return DenseState(u_new), metrics


# -----------------------------------------------------------------------
# Delegating integrators -- methods that own their own multi-step loop.
#
# TEBD, cole_hopf, and cole_hopf_circuit run their own internal loops
# (they need MPS state carried across steps, or a pre-built propagator).
# We wrap them so MainLoop calls step() once with the *full* n_steps,
# and the integrator returns the final state + all collected metrics.
#
# MainLoop treats n_steps=1 for these; the real step count is inside.
# -----------------------------------------------------------------------


class _DelegatingIntegrator(TimeIntegrator):
    """Base for methods that own their own multi-step loop.

    Subclasses implement _run_all() and return (solutions, metrics).
    MainLoop calls step() once; the integrator runs all n_steps
    internally.
    """

    def step(self, state, spatial_op, grid, config, dt, t=0.0):
        sols, mets = self._run_all(state, grid, config, dt)
        # Store full solution list in metrics so the caller can
        # retrieve it (MainLoop will get the final state from us).
        final = DenseState(sols[-1])
        return final, {
            "_delegated_solutions": sols,
            "_delegated_metrics": mets,
        }

    def _run_all(self, state, grid, config, dt):
        raise NotImplementedError


class TEBDIntegrator(_DelegatingIntegrator):
    """TEBD: dense H -> expm -> MPO -> apply to MPS (multi-step)."""

    def _run_all(self, state, grid, config, dt):
        from burgers_tebd import run_tebd_simulation
        from burgers_classical import source_term_sine

        u0 = state.to_dense()
        source_fn = None
        if hasattr(config, "_source_fn"):
            source_fn = config._source_fn
        return run_tebd_simulation(
            u0, grid.xc, config.nu, dt, config.n_steps,
            source_fn=source_fn, bc=grid.bc,
            max_bond=config.bond_dim,
            cutoff=config.mps_threshold or 1e-14,
        )


class ColeHopfIntegrator(_DelegatingIntegrator):
    """Cole-Hopf linearisation (multi-step)."""

    def _run_all(self, state, grid, config, dt):
        from burgers_cole_hopf import run_cole_hopf_simulation

        u0 = state.to_dense()
        source_fn = None
        if hasattr(config, "_source_fn"):
            source_fn = config._source_fn
        return run_cole_hopf_simulation(
            u0, grid.xc, config.nu, dt, config.n_steps, bc=grid.bc,
            max_bond=config.bond_dim,
            cutoff=config.mps_threshold or 1e-14,
        )


class LBMIntegrator(_DelegatingIntegrator):
    """Classical D1Q3 LBM (multi-step, owns its loop)."""

    def _run_all(self, state, grid, config, dt):
        from burgers_lbm import run_lbm_simulation

        u0 = state.to_dense()
        source_fn = None
        if hasattr(config, "_source_fn"):
            source_fn = config._source_fn
        return run_lbm_simulation(
            u0, grid.xc, config.nu, dt, config.n_steps, bc=grid.bc,
            source_fn=source_fn,
        )


class QLBMCircuitIntegrator(_DelegatingIntegrator):
    """QLBM circuit (multi-step, owns its loop)."""

    def __init__(self, backend: Any = None) -> None:
        self.backend = backend

    def _run_all(self, state, grid, config, dt):
        from burgers_qlbm_circuit import run_qlbm_circuit_simulation

        u0 = state.to_dense()
        source_fn = None
        if hasattr(config, "_source_fn"):
            source_fn = config._source_fn
        return run_qlbm_circuit_simulation(
            u0, grid.xc, config.nu, dt, config.n_steps, bc=grid.bc,
            source_fn=source_fn,
            shots=config.shots,
            backend=self.backend,
            sign_recovery=config.sign_recovery,
            optimization_level=config.optimization_level,
            seed=config.seed,
        )


class ColeHopfCircuitIntegrator(_DelegatingIntegrator):
    """Cole-Hopf circuit (multi-step, owns its loop)."""

    def __init__(self, backend: Any = None) -> None:
        self.backend = backend

    def _run_all(self, state, grid, config, dt):
        from burgers_cole_hopf_circuit import (
            run_cole_hopf_circuit_simulation,
        )

        u0 = state.to_dense()
        source_fn = None
        if hasattr(config, "_source_fn"):
            source_fn = config._source_fn

        if config.shots > 0 and self.backend is None:
            from q8020_cfd_qutil.backend import get_backend
            self.backend = get_backend(
                name=config.backend_name,
                backend_type=config.backend_type,
                t1=config.t1, t2=config.t2,
                coupling_map=config.coupling_map,
            )

        return run_cole_hopf_circuit_simulation(
            u0, grid.xc, config.nu, dt, config.n_steps, bc=grid.bc,
            propagator=config.propagator, shots=config.shots,
            snapshot_interval=max(1, config.save_every),
            bond_dim=config.bond_dim, encoding=config.encoding,
            backend=self.backend,
            backend_type=config.backend_type,
            backend_name=config.backend_name,
            optimization_level=config.optimization_level,
            seed=config.seed,
            source_fn=source_fn,
            evolution_mode=config.evolution_mode,
            chunk_size=config.chunk_size,
            phi_modes=config.phi_modes,
            taylor_order=config.taylor_order,
            readout=getattr(config, "readout", "direct"),
        )


# -----------------------------------------------------------------------
# Integrator registry
# -----------------------------------------------------------------------

# Methods that delegate their own multi-step loop.
_DELEGATING_METHODS = {
    "tebd", "cole_hopf", "cole_hopf_circuit", "lbm", "qlbm_circuit",
}


def make_integrator(
    config: BurgersConfig,
) -> TimeIntegrator:
    """Build the right TimeIntegrator for config.method."""
    m = config.method

    if m == "shift":
        return ShiftEulerIntegrator()

    if m == "quantum_exact":
        return QuantumExactIntegrator()

    # Pre-build backend once for circuit methods.
    backend = None
    if m in ("quantum_circuit", "mps", "tebd_circuit"):
        try:
            from q8020_cfd_qutil.backend import get_backend
            backend = get_backend(
                name=config.backend_name,
                backend_type="sim",
                t1=config.t1, t2=config.t2,
            )
        except ImportError:
            pass

    if m == "quantum_circuit":
        return QuantumCircuitIntegrator(backend)
    if m == "mps":
        return MPSIntegrator(backend)
    if m == "tebd_circuit":
        return TEBDCircuitIntegrator(backend)
    if m == "tebd":
        return TEBDIntegrator()
    if m == "cole_hopf":
        return ColeHopfIntegrator()
    if m == "cole_hopf_circuit":
        return ColeHopfCircuitIntegrator()
    if m == "lbm":
        return LBMIntegrator()
    if m == "qlbm_circuit":
        return QLBMCircuitIntegrator(backend)

    raise ValueError(f"Unknown method: {m}")


# -----------------------------------------------------------------------
# Top-level run_simulation (drop-in replacement)
# -----------------------------------------------------------------------


def run_simulation_fw(
    config: BurgersConfig,
    grid: Grid1D,
    u0: np.ndarray,
    source_fn=None,
) -> tuple[list[np.ndarray], list[dict] | None]:
    """Run a Burgers simulation using the solverfw MainLoop.

    Drop-in replacement for burgers_trotter.run_simulation().
    Returns (solutions, step_metrics) with the same contract.
    """
    from q8020_cfd_metautil.solverfw import MainLoop

    # Attach source function to config for integrators to use.
    config._source_fn = source_fn  # type: ignore[attr-defined]

    state = DenseState(u0.copy())
    spatial_op = ShiftFD()
    integrator = make_integrator(config)

    # Delegating methods run their own loop internally.
    if config.method in _DELEGATING_METHODS:
        _, result_metrics = integrator.step(
            state, spatial_op, grid, config, config.dt or 0.0,
        )
        sols = result_metrics.get("_delegated_solutions", [u0])
        mets = result_metrics.get("_delegated_metrics")
        return sols, mets

    # Standard per-step methods: use MainLoop.
    loop = MainLoop()
    solutions, all_metrics = loop.run(
        config, grid, state, spatial_op, integrator,
    )
    return solutions, all_metrics
