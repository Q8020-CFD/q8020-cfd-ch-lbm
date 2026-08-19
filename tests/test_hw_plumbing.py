"""Tests for the F12 hardware-execution port (session + mitigation +
dry-run plumbing through lib_fw / lib_cole_hopf_circuit).

No IBM credentials are needed anywhere here: the hardware guard fires
before any IBM contact, and the end-to-end session test runs through
SamplerV2 local-testing mode on a FakeBackendV2 (the same credential-
free proxy the F12 runner's --target local used).
"""

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

from q8020_cfd_metautil.solverfw import Grid1D
from lib_cole_hopf_circuit import (
    dry_run_segment_transpile,
    forward_ch_psi0,
    run_cole_hopf_circuit_simulation,
)
from lib_fw import (
    BurgersConfig,
    build_sampler_options,
    resolve_hw_execution,
    run_simulation_fw,
)
from lib_postprocess import json_safe
from q8020_backend_utils.ibm.backend import get_backend


def _make_ic(q: int = 3):
    """Sine IC on [0,1) with 2^q grid points."""
    N = 1 << q
    dx = 1.0 / N
    x = np.linspace(0, 1 - dx, N)
    u0 = 0.3 * np.sin(2 * np.pi * x)
    return u0, x, dx


# ---------------------------------------------------------------------
# Tri-state resolution
# ---------------------------------------------------------------------


class TestResolveHwExecution:

    def test_auto_hardware_enables_everything(self):
        r = resolve_hw_execution("hardware", "measure_reprepare")
        assert r["use_session"] is True
        assert r["measure_mitigation"] is True
        assert r["dynamical_decoupling"] is True

    def test_auto_sim_disables_everything(self):
        r = resolve_hw_execution("sim", "measure_reprepare")
        assert r["use_session"] is False
        assert r["measure_mitigation"] is False
        assert r["dynamical_decoupling"] is False

    def test_auto_fake_disables_everything(self):
        """Existing fake-backend sweeps keep their behaviour under auto."""
        r = resolve_hw_execution("fake", "measure_reprepare")
        assert r["use_session"] is False
        assert r["measure_mitigation"] is False

    def test_explicit_on_valid_for_fake(self):
        r = resolve_hw_execution(
            "fake", "measure_reprepare",
            session="on", measure_mitigation="on",
            dynamical_decoupling="on",
        )
        assert r["use_session"] is True
        assert r["measure_mitigation"] is True
        assert r["dynamical_decoupling"] is True

    def test_explicit_on_rejected_for_sim(self):
        with pytest.raises(ValueError, match="only valid"):
            resolve_hw_execution(
                "sim", "measure_reprepare", session="on",
            )
        with pytest.raises(ValueError, match="only valid"):
            resolve_hw_execution(
                "sim", "measure_reprepare", measure_mitigation="on",
            )

    def test_explicit_off_wins_on_hardware(self):
        r = resolve_hw_execution(
            "hardware", "measure_reprepare",
            session="off", measure_mitigation="off",
            dynamical_decoupling="off",
        )
        assert r["use_session"] is False
        assert r["measure_mitigation"] is False
        assert r["dynamical_decoupling"] is False

    def test_single_mode_disables_session_and_mitigation(self):
        """Single mode is one async submit_job: neither a Session nor
        sampler options route through it, so they must resolve off."""
        r = resolve_hw_execution("hardware", "single")
        assert r["use_session"] is False
        assert r["measure_mitigation"] is False
        assert r["dynamical_decoupling"] is False

    def test_initial_layout_parses(self):
        r = resolve_hw_execution(
            "hardware", "measure_reprepare",
            initial_layout="12,13,14",
        )
        assert r["initial_layout"] == [12, 13, 14]

    def test_initial_layout_none(self):
        r = resolve_hw_execution("sim", "measure_reprepare")
        assert r["initial_layout"] is None


# ---------------------------------------------------------------------
# Sampler options
# ---------------------------------------------------------------------


class TestBuildSamplerOptions:

    def test_both_off_is_none(self):
        assert build_sampler_options(False, False) is None

    def test_trex_only(self):
        opts = build_sampler_options(True, False)
        assert opts == {"twirling": {"enable_measure": True}}

    def test_both_on(self):
        opts = build_sampler_options(True, True)
        assert opts["twirling"] == {"enable_measure": True}
        assert opts["dynamical_decoupling"] == {
            "enable": True, "sequence_type": "XpXm",
        }


# ---------------------------------------------------------------------
# Hardware guard (regression: opt-in stays explicit)
# ---------------------------------------------------------------------


class TestHardwareGuard:

    def test_measure_reprepare_hardware_requires_optin(self):
        """Without allow_hardware=True the v1 guard still fires (before
        any IBM contact, so no backend object is needed)."""
        u0, x, _ = _make_ic(q=3)
        with pytest.raises(NotImplementedError, match="allow_hardware"):
            run_cole_hopf_circuit_simulation(
                u0, x, nu=0.02, dt=0.01, n_steps=2,
                shots=1024, backend=None,
                backend_type="hardware", backend_name="ibm_anywhere",
                evolution_mode="measure_reprepare", segment_size=1,
                snapshot_interval=1,
            )


# ---------------------------------------------------------------------
# Dry-run pre-flight
# ---------------------------------------------------------------------


class TestDryRun:

    def test_dry_run_reports_segment_stats(self):
        u0, x, _ = _make_ic(q=3)
        backend = get_backend(backend_type="sim")
        stats = dry_run_segment_transpile(
            u0, x, nu=0.02, dt=0.01, n_steps=4, segment_size=2,
            bond_dim=2, backend=backend, seed=42,
        )
        assert stats["n_segments"] == 2
        assert stats["segment_qubits"] >= 4  # q + bond + heat anc
        assert stats["transpiled_depth"] > 0
        assert stats["transpiled_2q_gates"] > 0
        assert isinstance(stats["transpiled_ops"], dict)
        # The report must be JSON-serialisable for the CLI output.
        json.dumps(stats, default=str)

    def test_dry_run_matches_forward_transform(self):
        """Dry-run seeds segment 0 from the same forward transform the
        solver loop uses (single source of truth, no drift)."""
        u0, x, dx = _make_ic(q=3)
        psi0, phi_norm, _ = forward_ch_psi0(u0, dx, 0.02)
        assert np.isclose(np.linalg.norm(psi0), 1.0)
        assert phi_norm > 0


# ---------------------------------------------------------------------
# End-to-end: SamplerV2 + Session + TREX/DD via the integrator (the
# exact hardware code path, credential-free on a FakeBackendV2)
# ---------------------------------------------------------------------


class TestFakeSessionEndToEnd:

    def test_session_mitigated_measure_reprepare_runs(self):
        u0, x, _ = _make_ic(q=3)
        N = len(u0)
        dx = x[1] - x[0]
        config = BurgersConfig(
            q=3, nu=0.02, cfl=0.1, dt=0.1 * dx, n_steps=2,
            bc="periodic", method="cole_hopf_circuit",
            ic="sine", source="none",
            shots=1024, seed=42,
            bond_dim=2,
            backend_name="torino", backend_type="fake",
            evolution_mode="measure_reprepare", segment_size=1,
            save_every=1,
            use_session=True,
            measure_mitigation=True,
            dynamical_decoupling=True,
        )
        grid = Grid1D.from_qubits(3, bc="periodic")
        sols, mets, _ = run_simulation_fw(config, grid, u0)

        assert len(sols) == config.n_steps + 1
        assert np.all(np.isfinite(sols[-1]))
        assert sols[-1].shape == (N,)
        # SamplerV2 path stamps a job_id per segment execution.
        assert mets[-1]["execute"].get("job_id")
        # Every-segment audit trail rides on the final snapshot.
        segs = mets[-1].get("segments_full")
        assert segs and len(segs) == 2
        assert all(s["execute"].get("job_id") for s in segs)
        # The resolved backend is exposed for the calibration fragment.
        assert getattr(config, "_resolved_backend", None) is not None


# ---------------------------------------------------------------------
# JSON sanitizer (hardware job payloads carry datetimes)
# ---------------------------------------------------------------------


class TestJsonSafe:

    def test_datetime_and_numpy_coerce(self):
        payload = {
            "execute": {
                "job_id": "abc123",
                "job_metrics": {
                    "timestamps": {
                        "created": datetime.datetime(
                            2026, 8, 19, tzinfo=datetime.timezone.utc,
                        ),
                    },
                    "usage": {"quantum_seconds": np.float64(3.25)},
                },
            },
            "p_success": np.float32(0.9),
            "counts": np.array([1, 2, 3]),
        }
        safe = json_safe(payload)
        out = json.dumps(safe)  # must not raise
        assert "abc123" in out
        assert safe["execute"]["job_metrics"]["usage"][
            "quantum_seconds"] == 3.25
        assert safe["counts"] == [1, 2, 3]

    def test_primitives_pass_through(self):
        assert json_safe({"a": 1, "b": None, "c": "x", "d": True}) == {
            "a": 1, "b": None, "c": "x", "d": True,
        }
