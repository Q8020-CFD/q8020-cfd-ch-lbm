"""Source-induced potential V(x,t) for forced Burgers via Cole-Hopf.

For du/dt + u*du/dx = nu*d2u/dx2 + g(x,t), the Cole-Hopf substitution
u = -2*nu*d(ln phi)/dx yields dphi/dt = nu*d2phi/dx2 - V(x,t)*phi
with V_x = +g/(2*nu).  Returned V is gauge-fixed to spatial mean-zero
so that phi does not accumulate multiplicative blow-up/decay.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def potential_from_source(
    source_fn: Callable[[np.ndarray, float], np.ndarray] | None,
    x: np.ndarray,
    t: float,
    nu: float,
    bc: str = "periodic",
) -> np.ndarray | None:
    """Return V(x, t) on the grid; None if source_fn is None.

    Integration is trapezoidal on the grid x.  The gauge constant is
    chosen so V has zero spatial mean (irrelevant to u via Cole-Hopf).
    """
    if source_fn is None:
        return None
    g = source_fn(x, t)
    dx = x[1] - x[0]
    # antiderivative G(x) = int_0^x g(x') dx' via trapezoid rule
    G = np.concatenate(([0.0], np.cumsum((g[:-1] + g[1:]) * dx / 2)))
    V_raw = G / (2.0 * nu)
    return V_raw - V_raw.mean()
