"""solverfw bindings for the Burgers equation solver.

Provides BurgersConfig, ShiftFD spatial operator, and TimeIntegrator
wrappers for every existing evolution method. These are thin adapters
around the existing step functions in lib_fd.py -- the physics
code is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from q8020_cfd_metautil.solverfw import (
    ContainerIntegrator,
    ContainerResult,
    DenseState,
    ForwardEuler,
    Grid1D,
    MainLoop,
    ProblemTransform,
    SolverConfig,
    SpatialOperator,
    TimeIntegrator,
)
from q8020_backend_utils.ibm.backend import get_backend

from lib_cole_hopf_circuit import run_cole_hopf_circuit_simulation
from lib_fd import compute_rhs_shift
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
    fock_qubits: int = 3
    qalb_collision_trotter_reps: int = 0
    metric_transpile_timeout: float = 60.0
    phi_modes: int = 0
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


# The `shift` method is plain forward-Euler on ShiftFD.compute_rhs
# (u_new = u + dt*rhs), which is exactly the framework's ForwardEuler. We
# use it directly rather than a bespoke integrator, so ShiftFD.compute_rhs
# is the single source of the stepping physics (it was previously dead --
# the old ShiftEulerIntegrator called shift_euler_step directly and never
# touched the SpatialOperator seam).


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

        # Honor the CLI backend config (--backend/--backend-type/--t1/--t2):
        # build the backend from config when shots are requested, matching
        # ColeHopfCircuitIntegrator. Previously QALB was constructed with
        # backend=None and always fell back to a bare AerSimulator(), so the
        # backend flags were silently dropped for qlbm_circuit.
        if config.shots > 0 and self.backend is None:
            self.backend = get_backend(
                name=config.backend_name,
                backend_type=config.backend_type,
                t1=config.t1, t2=config.t2,
                coupling_map=config.coupling_map,
            )

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


class ColeHopfTransform(ProblemTransform):
    """Named Cole-Hopf change-of-variables (u <-> phi <-> psi).

    The container applies it internally (its inverse is entangled with the
    circuit readout, so it cannot be a MainLoop-level bracket), but naming
    it as a ProblemTransform makes it a recorded slot in code.chain rather
    than logic buried inside the delegating step.
    """

    def __init__(self, phi_modes: int = 0) -> None:
        self.phi_modes = phi_modes

    def forward(self, u, grid, config):
        from lib_cole_hopf import cole_hopf_forward
        phi = cole_hopf_forward(u, grid.dx, config.nu)
        return phi, {"nu": config.nu, "dx": grid.dx}

    def inverse(self, v, carry, grid, config):
        from lib_cole_hopf import cole_hopf_inverse
        return cole_hopf_inverse(
            v, carry["dx"], carry["nu"], bc=grid.bc,
        )


class ColeHopfCircuitIntegrator(ContainerIntegrator):
    """Cole-Hopf circuit: a container that owns its multi-step loop and its
    named sub-plugins (transform / encode / backend / readout).

    Promoted from the delegating idiom to a ContainerIntegrator (SPEC v2b
    §2b.6): MainLoop drives it via run_all(); its internals are recorded in
    code.chain instead of smuggled through sentinel metric keys.
    """

    def __init__(self, backend: Any = None) -> None:
        self.backend = backend
        self.transform = ColeHopfTransform()

    def run_all(self, state, spatial_op, grid, config) -> ContainerResult:
        u0 = state.to_dense()
        source_fn = getattr(config, "_source_fn", None)
        self.transform = ColeHopfTransform(phi_modes=config.phi_modes)

        if config.shots > 0 and self.backend is None:
            self.backend = get_backend(
                name=config.backend_name,
                backend_type=config.backend_type,
                t1=config.t1, t2=config.t2,
                coupling_map=config.coupling_map,
            )

        sols, mets = run_cole_hopf_circuit_simulation(
            u0, grid.xc, config.nu, config.dt or 0.0, config.n_steps,
            bc=grid.bc,
            shots=config.shots,
            snapshot_interval=max(1, config.save_every),
            bond_dim=config.bond_dim,
            backend=self.backend,
            backend_type=config.backend_type,
            backend_name=config.backend_name,
            optimization_level=config.optimization_level,
            seed=config.seed,
            source_fn=source_fn,
            evolution_mode=config.evolution_mode,
            segment_size=config.segment_size,
            phi_modes=config.phi_modes,
            readout=getattr(config, "readout", "direct"),
            metric_transpile_timeout=config.metric_transpile_timeout,
        )
        return ContainerResult(solutions=sols, step_metrics=mets)

    def code_chain(self, config) -> dict:
        """The recorded plugin chain for this run (SPEC v2b §2b.3)."""
        from q8020_cfd_metautil.meta_fragment import chain_entry, make_chain
        readout_plugin = (
            "statevector_project" if config.shots == 0
            else "counts_marginalize"
        )
        return make_chain(
            transform=chain_entry(
                "transform", "cole_hopf", {"phi_modes": config.phi_modes},
            ),
            encode=chain_entry(
                "encode", "mps_staircase", {"bond_dim": config.bond_dim},
            ),
            solve=chain_entry(
                "solve", "cole_hopf_circuit", kind="container",
                knobs={"evolution_mode": config.evolution_mode},
            ),
            readout=chain_entry(
                "readout", readout_plugin, {"shots": config.shots},
            ),
        )


# -----------------------------------------------------------------------
# Integrator registry
# -----------------------------------------------------------------------

# Methods still on the legacy delegating idiom (own their loop, bypass
# MainLoop via the sentinel-key shim). cole_hopf_circuit graduated to a
# ContainerIntegrator (SPEC v2b) and is driven by MainLoop's container
# branch; lbm / qlbm_circuit migrate incrementally (they need the
# genuine_steps remap the container branch does not yet thread).
_DELEGATING_METHODS = {
    "lbm", "qlbm_circuit",
}


def make_integrator(
    config: BurgersConfig,
) -> TimeIntegrator:
    """Build the right TimeIntegrator for config.method."""
    m = config.method

    if m == "shift":
        return ForwardEuler()

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

    # Legacy delegating methods (lbm / qlbm_circuit): run their own loop
    # via the sentinel-key shim until they migrate to ContainerIntegrator.
    if config.method in _DELEGATING_METHODS:
        _, result_metrics = integrator.step(
            state, spatial_op, grid, config, config.dt or 0.0,
        )
        sols = result_metrics.get("_delegated_solutions", [u0])
        mets = result_metrics.get("_delegated_metrics")
        genuine = result_metrics.get("_delegated_genuine_steps")
        return sols, mets, genuine

    # Per-step methods and ContainerIntegrators both go through MainLoop
    # (it detects a ContainerIntegrator and drives run_all() once).
    loop = MainLoop()
    solutions, all_metrics = loop.run(
        config, grid, state, spatial_op, integrator,
    )
    return solutions, all_metrics, None
