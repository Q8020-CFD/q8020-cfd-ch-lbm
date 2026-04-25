"""State-encoding permutations for the Cole-Hopf circuit pipeline.

Supports binary (default, no-op) and Gray code encodings.
Gray code maps adjacent grid indices to Hamming-1 qubit states,
making the tridiagonal Laplacian nearest-neighbor on the qubit chain.
"""

from __future__ import annotations

import numpy as np


VALID_ENCODINGS = ("binary", "gray")


def encoding_permutation(q: int, encoding: str) -> np.ndarray:
    """Return permutation array of length N = 2^q.

    perm[i] = encoded qubit-state index for grid index i.
    For gray: perm[i] = i XOR (i >> 1).
    """
    if encoding not in VALID_ENCODINGS:
        raise ValueError(
            f"Unknown encoding {encoding!r}; "
            f"expected one of {VALID_ENCODINGS}"
        )
    N = 1 << q
    idx = np.arange(N)
    if encoding == "binary":
        return idx.copy()
    # Gray code: i -> i ^ (i >> 1)
    return idx ^ (idx >> 1)


def permute_to_encoding(
    v: np.ndarray, q: int, encoding: str,
) -> np.ndarray:
    """Permute grid-indexed vector to encoded-index order.

    v[i] (grid index i) -> w[perm[i]] = v[i].
    """
    if encoding == "binary":
        return v.copy()
    perm = encoding_permutation(q, encoding)
    w = np.empty_like(v)
    w[perm] = v
    return w


def permute_from_encoding(
    w: np.ndarray, q: int, encoding: str,
) -> np.ndarray:
    """Permute encoded-index vector back to grid-index order.

    v[i] = w[perm[i]].
    """
    if encoding == "binary":
        return w.copy()
    perm = encoding_permutation(q, encoding)
    return w[perm]


def permute_operator(
    L: np.ndarray, q: int, encoding: str,
) -> np.ndarray:
    """Permute operator from grid basis to encoded basis.

    L_enc = P L P^T  where P is the permutation matrix for the
    encoding.  Equivalent to L_enc[a,b] = L[inv(a), inv(b)].
    """
    if encoding == "binary":
        return L
    perm = encoding_permutation(q, encoding)
    inv_perm = np.argsort(perm)
    return L[np.ix_(inv_perm, inv_perm)]
