"""Tests for cole_hopf_circuit shots/backend plumbing (§7b of SPEC-shots-backend)."""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from lib_cole_hopf_circuit import run_cole_hopf_circuit_simulation
from q8020_cfd_qutil.backend import get_backend


def _make_ic(q: int = 3):
    """Sine IC on [0,1) with 2^q grid points."""
    N = 1 << q
    dx = 1.0 / N
    x = np.linspace(0, 1 - dx, N)
    u0 = np.sin(2 * np.pi * x)
    return u0, x, dx


class TestColeHopfShotsIdeal:

    def test_cole_hopf_shots_ideal_sim_no_regression(self):
        """Ideal-Aer shots path returns finite results."""
        u0, x, _ = _make_ic(q=3)
        backend = get_backend(backend_type="sim")
        sols, mets = run_cole_hopf_circuit_simulation(
            u0, x, nu=0.01, dt=0.001, n_steps=2,
            shots=2048,
            backend=backend, seed=42,
        )
        assert len(sols) == 3
        assert np.all(np.isfinite(sols[-1]))
        assert mets[-1]["p_success"] > 0.5

    def test_cole_hopf_shots_noisy_finite(self):
        """Noisy sim (t1/t2) returns finite results (qft-diagonal)."""
        u0, x, _ = _make_ic(q=4)
        backend = get_backend(backend_type="sim", t1=50, t2=70)
        sols, mets = run_cole_hopf_circuit_simulation(
            u0, x, nu=0.01, dt=0.001, n_steps=2,
            shots=2048,
            backend=backend, seed=42,
        )
        assert np.all(np.isfinite(sols[-1]))
        assert mets[-1]["p_success"] > 0.1

    def test_seed_reproducibility_end_to_end(self):
        """Same seed -> bit-identical phi_hat."""
        u0, x, _ = _make_ic(q=3)
        backend = get_backend(backend_type="sim")
        kwargs: dict[str, Any] = dict(
            nu=0.01, dt=0.001, n_steps=2,
            shots=2048,
            backend=backend, seed=42,
        )
        s1, _ = run_cole_hopf_circuit_simulation(u0, x, **kwargs)
        s2, _ = run_cole_hopf_circuit_simulation(u0, x, **kwargs)
        assert np.array_equal(s1[-1], s2[-1])

    def test_metrics_carry_transpile_info(self):
        """Metrics include transpile and execute sub-dicts."""
        u0, x, _ = _make_ic(q=3)
        backend = get_backend(backend_type="sim")
        _, mets = run_cole_hopf_circuit_simulation(
            u0, x, nu=0.01, dt=0.001, n_steps=2,
            shots=2048,
            backend=backend, seed=42,
        )
        m = mets[-1]
        assert "transpile" in m
        assert "after" in m["transpile"]
        assert "depth" in m["transpile"]["after"]
        assert "execute" in m
        assert "shots_executed" in m["execute"]
