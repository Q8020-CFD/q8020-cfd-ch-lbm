"""solverfw bindings for the Burgers equation solver.

Provides BurgersConfig, ShiftFD spatial operator, and TimeIntegrator
wrappers for every existing evolution method. These are thin adapters
around the existing step functions in lib_fd.py -- the physics
code is unchanged.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from q8020_cfd_metautil.solverfw import (
    DenseState,
    Grid1D,
    MainLoop,
    SolverConfig,
    SpatialOperator,
    TimeIntegrator,
)
from q8020_backend_utils.ibm.backend import get_backend

from lib_cole_hopf_circuit import run_cole_hopf_circuit_simulation
from lib_fd import compute_rhs_shift, shift_euler_step
from lib_lbm import run_lbm_simulation
from lib_qalb_circuit import run_qalb_simulation

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
    bond_dim: int | None = None
    use_mps_prep: bool = True
    shots: int = 0
    backend_name: str | None = None
    backend_type: str = "sim"
    coupling_map: str = "default"
    optimization_level: int = 1
    seed: int | None = None
    t1: float | None = None
    t2: float | None = None
    sign_recovery: str = "none"
    evolution_mode: str = "single"
    segment_size: int = 10
    # Collapse the per-step QFT/diagonal/QFT^-1 layers within each
    # read-out-free stretch into one layer for the stretch's total time.
    # Default ON: exact for the diagonal heat operator, so results are
    # identical to the per-step form (to fp/sampling noise) while cutting
    # circuit depth, gate count, and transpile/build time.  --no-collapse
    # restores the per-step path for A/B validation.
    collapse: bool = True
    max_total_qubits: int | None = None
    fock_qubits: int = 3
    qalb_collision_trotter_reps: int = 0
    metric_transpile_timeout: float = 60.0
    phi_modes: int = 0
    shock_pct: float | None = None

    # Hardware execution (F12 port).  Booleans are the RESOLVED values
    # (the solver maps the auto/on/off CLI tri-states before building
    # the config), so metadata reflects what actually ran.
    use_session: bool = False
    measure_mitigation: bool = False
    dynamical_decoupling: bool = False
    initial_layout: list[int] | None = None
    ibm_token: str | None = None
    ibm_channel: str = "ibm_cloud"
    ibm_instance: str | None = None

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
# Hardware execution helpers (F12 port)
# -----------------------------------------------------------------------


def resolve_hw_execution(
    backend_type: str,
    evolution_mode: str,
    session: str = "auto",
    measure_mitigation: str = "auto",
    dynamical_decoupling: str = "auto",
    initial_layout: str | None = None,
) -> dict[str, Any]:
    """Resolve the auto/on/off hardware CLI tri-states to concrete values.

    `backend_type="fake"` routes through SamplerV2 local-testing mode
    (the credential-free proxy for the exact hardware code path), so an
    explicit `on` is legal there; plain Aer sim ignores sessions and
    sampler options, so `on` is rejected rather than silently
    mislabelling the run.  `auto` enables everything only for real
    hardware.

    Returns dict(use_session, measure_mitigation, dynamical_decoupling,
    initial_layout).  Raises ValueError on invalid combinations.
    """
    sampler_path = backend_type in ("hardware", "fake")
    auto_hw = backend_type == "hardware"

    def _tri(name: str, value: str) -> bool:
        if value == "on" and not sampler_path:
            raise ValueError(
                f"--{name} on is only valid for --backend-type "
                "hardware/fake (plain Aer sim ignores SamplerV2 "
                "options; the run would be mislabelled)."
            )
        return value == "on" or (value == "auto" and auto_hw)

    use_session = _tri("session", session)
    mit = _tri("measure-mitigation", measure_mitigation)
    dd = _tri("dynamical-decoupling", dynamical_decoupling)

    # Session + mitigation only exist on the measure-reprepare SamplerV2
    # path.  Single mode's hardware branch is one async submit_job that
    # routes through neither -- a held Session would sit idle and the
    # mitigation flags would be recorded as on while running unmitigated.
    # Disable (with a note) so metadata reflects what actually runs.
    if evolution_mode != "measure_reprepare" and (use_session or mit or dd):
        print(
            "[burgers] NOTE: session/mitigation disabled: only "
            "--evolution-mode measure_reprepare routes jobs through the "
            "SamplerV2 + Session path (single mode is one async "
            "submission).",
            file=sys.stderr, flush=True,
        )
        use_session = mit = dd = False

    layout = None
    if initial_layout:
        layout = [int(t) for t in initial_layout.split(",")]

    return {
        "use_session": use_session,
        "measure_mitigation": mit,
        "dynamical_decoupling": dd,
        "initial_layout": layout,
    }


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


# -----------------------------------------------------------------------
# Spatial operator
# -----------------------------------------------------------------------


class ShiftFD(SpatialOperator):
    """Central-difference RHS using shift operators S+/S-."""

    def compute_rhs(self, state, grid, config, t=0.0):
        grid = cast(Grid1D, grid)
        config = cast(BurgersConfig, config)
        u = state.to_dense()
        g = None
        source_fn = getattr(config, "_source_fn", None)
        if source_fn is not None:
            g = source_fn(grid.xc, t)
        return compute_rhs_shift(u, grid.dx, config.nu, g, bc=grid.bc)


# -----------------------------------------------------------------------
# Time integrators -- thin wrappers around existing step functions
# -----------------------------------------------------------------------


class ShiftEulerIntegrator(TimeIntegrator):
    """Classical forward-Euler with shift-operator FD."""

    def step(self, state, spatial_op, grid, config, dt, t=0.0):
        grid = cast(Grid1D, grid)
        config = cast(BurgersConfig, config)
        u = state.to_dense()
        g = None
        source_fn = getattr(config, "_source_fn", None)
        if source_fn:
            g = source_fn(grid.xc, t)
        u_new, metrics = shift_euler_step(
            u, grid.dx, dt, config.nu, g, bc=grid.bc,
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
        result = self._run_all(state, grid, config, dt)
        # LBM-family solvers also return a genuine-step set (3-tuple);
        # the rest return (solutions, metrics).
        if len(result) == 3:
            sols, mets, genuine_steps = result
        else:
            sols, mets = result
            genuine_steps = None
        # Store full solution list in metrics so the caller can
        # retrieve it (MainLoop will get the final state from us).
        final = DenseState(sols[-1])
        return final, {
            "_delegated_solutions": sols,
            "_delegated_metrics": mets,
            "_delegated_genuine_steps": genuine_steps,
        }

    def _run_all(self, state, grid, config, dt):
        raise NotImplementedError


class LBMIntegrator(_DelegatingIntegrator):
    """Classical D1Q3 LBM (multi-step, owns its loop)."""

    def _run_all(self, state, grid, config, dt):
        u0 = state.to_dense()
        source_fn = None
        if hasattr(config, "_source_fn"):
            source_fn = config._source_fn
        return run_lbm_simulation(
            u0, grid.xc, config.nu, dt, config.n_steps, bc=grid.bc,
            source_fn=source_fn,
        )


class QALBIntegrator(_DelegatingIntegrator):
    """Pure-quantum QALB (Phase 2, #27): state-independent App C
    finite-position collision (built once, applied per site) + streaming.
    """

    def __init__(self, backend: Any = None) -> None:
        self.backend = backend

    def _run_all(self, state, grid, config, dt):
        # QALB reads out <q-hat> directly (eigenvalues span +/-, so the
        # estimate is inherently signed) -- there is no magnitude/sign split
        # for a Hadamard test to recover.  Reject any non-none sign-recovery
        # rather than silently ignoring it (the flag is meaningful only for
        # the magnitude-readout methods, e.g. direct_lcu).
        if getattr(config, "sign_recovery", "none") not in ("none", None):
            raise ValueError(
                f"--sign-recovery={config.sign_recovery!r} is not supported "
                "for method 'qlbm_circuit' (QALB): its readout estimates "
                "<q-hat>, which is already signed, so sign recovery is a "
                "no-op.  Use --sign-recovery none (the default)."
            )

        u0 = state.to_dense()
        source_fn = getattr(config, "_source_fn", None)
        return run_qalb_simulation(
            u0, grid.xc, config.nu, dt, config.n_steps, bc=grid.bc,
            source_fn=source_fn,
            shots=config.shots,
            backend=self.backend,
            qc=getattr(config, "fock_qubits", 3),
            seed=config.seed,
            trotter_reps=getattr(config, "qalb_collision_trotter_reps", 0),
            trotter_order=config.trotter_order,
        )


class ColeHopfCircuitIntegrator(_DelegatingIntegrator):
    """Cole-Hopf circuit (multi-step, owns its loop)."""

    def __init__(self, backend: Any = None) -> None:
        self.backend = backend

    def _run_all(self, state, grid, config, dt):
        u0 = state.to_dense()
        source_fn = None
        if hasattr(config, "_source_fn"):
            source_fn = config._source_fn

        if config.shots > 0 and self.backend is None:
            self.backend = get_backend(
                name=config.backend_name,
                backend_type=config.backend_type,
                t1=config.t1, t2=config.t2,
                coupling_map=config.coupling_map,
                token=config.ibm_token,
                channel=config.ibm_channel,
                instance=config.ibm_instance,
            )
        # Expose the resolved backend so the solver can write the
        # backend-calibration fragment (hardware/fake provenance).
        config._resolved_backend = self.backend  # type: ignore[attr-defined]

        sampler_options = build_sampler_options(
            config.measure_mitigation, config.dynamical_decoupling,
        )
        # One held Session per case: the measure-reprepare segments are
        # serial QPU jobs, and a Session lets them share a single queue
        # slot.  Correctness never depends on it -- each segment is
        # re-prepared classically from the prior segment's counts, so
        # session-less job mode (IBM open plan) yields identical
        # results, just N queue waits.
        session_cm = None
        if (
            config.use_session
            and config.shots > 0
            and config.backend_type in ("hardware", "fake")
        ):
            from qiskit_ibm_runtime import Session
            session_cm = Session(backend=self.backend)

        def _run():
            return run_cole_hopf_circuit_simulation(
                u0, grid.xc, config.nu, dt, config.n_steps, bc=grid.bc,
                shots=config.shots,
                snapshot_interval=max(1, config.save_every),
                bond_dim=config.bond_dim,
                use_mps_prep=config.use_mps_prep,
                backend=self.backend,
                backend_type=config.backend_type,
                backend_name=config.backend_name,
                optimization_level=config.optimization_level,
                seed=config.seed,
                source_fn=source_fn,
                evolution_mode=config.evolution_mode,
                segment_size=config.segment_size,
                collapse=config.collapse,
                max_total_qubits=config.max_total_qubits,
                phi_modes=config.phi_modes,
                metric_transpile_timeout=config.metric_transpile_timeout,
                allow_hardware=(config.backend_type == "hardware"),
                session=session_cm,
                sampler_options=sampler_options,
                initial_layout=config.initial_layout,
            )

        if session_cm is not None:
            with session_cm:
                return _run()
        return _run()


# -----------------------------------------------------------------------
# Integrator registry
# -----------------------------------------------------------------------

# Methods that delegate their own multi-step loop.
_DELEGATING_METHODS = {
    "cole_hopf_circuit", "lbm", "qlbm_circuit",
}


def make_integrator(
    config: BurgersConfig,
) -> TimeIntegrator:
    """Build the right TimeIntegrator for config.method."""
    m = config.method

    if m == "shift":
        return ShiftEulerIntegrator()

    if m == "cole_hopf_circuit":
        return ColeHopfCircuitIntegrator()
    if m == "lbm":
        return LBMIntegrator()
    # `qlbm_circuit` is the pure-quantum QALB (Phase 2).
    if m == "qlbm_circuit":
        return QALBIntegrator(None)

    raise ValueError(f"Unknown method: {m}")


# -----------------------------------------------------------------------
# Top-level run_simulation (drop-in replacement)
# -----------------------------------------------------------------------


def run_simulation_fw(
    config: BurgersConfig,
    grid: Grid1D,
    u0: np.ndarray,
    source_fn=None,
) -> tuple[list[np.ndarray], list[dict] | None, list[int] | None]:
    """Run a Burgers simulation using the solverfw MainLoop.

    Returns (solutions, step_metrics, genuine_steps).  genuine_steps is
    the caller-step indices that are genuinely computed (LBM family,
    coarser than the caller grid) or None when every step is genuine.
    """
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
        genuine = result_metrics.get("_delegated_genuine_steps")
        return sols, mets, genuine

    # Standard per-step methods: use MainLoop.
    loop = MainLoop()
    solutions, all_metrics = loop.run(
        config, grid, state, spatial_op, integrator,
    )
    return solutions, all_metrics, None
