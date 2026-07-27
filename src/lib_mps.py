"""MPS state preparation for quantum encoding of classical fields.

Implements the MPS decomposition from Gopalakrishnan Meena et al.
AIAA-2026 §III.B.1 (Eqs. 5-6) and the MPS-to-circuit conversion
following Ran 2020 (Ref [27]).

Given a classical vector u of length N = K^q (K=2 for qubits), the state
is decomposed into q site tensors A^k via iterated SVD, then each site
tensor is embedded into a unitary gate for circuit preparation.
"""

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate


def normalize_state(u: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalize a classical vector to unit L2 norm for quantum encoding.

    Returns (normalized_vector, norm).
    """
    norm = np.linalg.norm(u)
    if norm < 1e-15:
        raise ValueError("Cannot normalize a zero vector.")
    return u / norm, norm


def classical_to_mps(
    u: np.ndarray,
    bond_dim: int | None = None,
    threshold: float = 0.0,
    canonical: str = "left",
) -> list[np.ndarray]:
    """Decompose a state vector into MPS site tensors via iterated SVD.

    For N = 2^q, produces q tensors:
      A[0]: shape (1, 2, d1)       — left boundary
      A[k]: shape (dk, 2, dk+1)    — bulk sites
      A[q-1]: shape (d_{q-1}, 2, 1) — right boundary

    Parameters
    ----------
    u : state vector of length N = 2^q (must be normalized)
    bond_dim : max bond dimension. None = full rank (no truncation).
    threshold : discard singular values below this.
    canonical : "left" or "right". Left-canonical: sweep left-to-right,
        each tensor satisfies sum_j A^{j dagger} A^j = I_{d_right}.
        Right-canonical: sweep right-to-left, each tensor satisfies
        sum_j A^j A^{j dagger} = I_{d_left}.

    Returns
    -------
    List of q site tensors.
    """
    N = len(u)
    q = int(np.log2(N))
    if 2**q != N:
        raise ValueError(f"N={N} is not a power of 2.")

    if canonical == "left":
        return _mps_left_canonical(u, q, bond_dim, threshold)
    else:
        return _mps_right_canonical(u, q, bond_dim, threshold)


def _truncate_svd(
    S: np.ndarray,
    bond_dim: int | None,
    threshold: float,
) -> int:
    """Determine truncation rank from singular values."""
    d = len(S)
    if threshold > 0:
        keep = int(np.sum(S > threshold))
        d = max(keep, 1)
    if bond_dim is not None:
        d = min(d, bond_dim)
    return d


def _mps_left_canonical(
    u: np.ndarray, q: int, bond_dim: int | None, threshold: float
) -> list[np.ndarray]:
    """Left-canonical MPS: sweep left to right."""
    psi = u.copy().astype(complex).reshape([2] * q)
    tensors: list[np.ndarray] = []
    left_dim = 1

    for k in range(q - 1):
        right_size = 2 ** (q - k - 1)
        mat = psi.reshape(left_dim * 2, right_size)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)

        d = _truncate_svd(S, bond_dim, threshold)
        U = U[:, :d]
        S = S[:d]
        Vh = Vh[:d, :]

        tensors.append(U.reshape(left_dim, 2, d))
        psi = np.diag(S) @ Vh
        left_dim = d

    tensors.append(psi.reshape(left_dim, 2, 1))
    return tensors


def _mps_right_canonical(
    u: np.ndarray, q: int, bond_dim: int | None, threshold: float
) -> list[np.ndarray]:
    """Right-canonical MPS: sweep right to left.

    Each tensor satisfies sum_j A^j A^{j dagger} = I_{d_left}, making
    the map V: |alpha> -> sum_{j,beta} A[alpha,j,beta] |j,beta> an
    isometry (V^H V = I_{d_left}).
    """
    psi = u.copy().astype(complex).reshape([2] * q)
    tensors: list[np.ndarray] = [None] * q  # type: ignore
    right_dim = 1

    for k in range(q - 1, 0, -1):
        left_size = 2**k
        mat = psi.reshape(left_size, right_dim * 2)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)

        d = _truncate_svd(S, bond_dim, threshold)
        U = U[:, :d]
        S = S[:d]
        Vh = Vh[:d, :]

        # Site tensor from Vh: shape (d, 2 * right_dim) -> (d, 2, right_dim)
        tensors[k] = Vh.reshape(d, 2, right_dim)
        psi = U @ np.diag(S)
        right_dim = d

    # Leftmost tensor
    tensors[0] = psi.reshape(1, 2, right_dim)
    return tensors


def site_tensor_to_unitary(
    A: np.ndarray,
    n_bond_qubits: int,
) -> np.ndarray:
    """Embed a right-canonical MPS site tensor into a unitary.

    A has shape (d_left, 2, d_right). For right-canonical MPS, the map
    V: |alpha> -> sum_{j,beta} A[alpha,j,beta] |j,beta> is an isometry
    (V^H V = I_{d_left}).

    The unitary acts on 1 physical qubit + n_bond_qubits bond qubits.
    Qiskit ordering: [phys, bond_0, ...] with phys as LSB.
    Local index = phys_val + 2 * bond_val.

    Column 2*alpha (phys_in=0, bond_in=alpha) gets the isometry column.
    """
    d_left, K, d_right = A.shape
    n_gate = 1 + n_bond_qubits
    dim = 2**n_gate

    # Build isometry V: dim x d_left
    V = np.zeros((dim, d_left), dtype=complex)
    for alpha in range(d_left):
        for jk in range(K):
            for beta in range(d_right):
                V[jk + 2 * beta, alpha] = A[alpha, jk, beta]

    iso_indices = [2 * alpha for alpha in range(d_left)]
    free_indices = [j for j in range(dim) if j not in iso_indices]

    # Ran 2020 (Ref [27]) describes this unitary completion for exact
    # right-canonical MPS, where V is guaranteed to be a proper isometry
    # (V^H V = I_{d_left}).  However, the leftmost tensor (_mps_right_canonical
    # line "tensors[0] = psi.reshape(...)") carries the accumulated singular
    # values from truncation and ||V[:,0]|| = S_0 < 1 when bond_dim is
    # truncated — violating the isometry condition and making the direct
    # column placement non-unitary.
    #
    # QR-decomposing V gives a proper isometry Q whose column space equals
    # range(V).  For exact MPS (no truncation) Q = V up to trivial phases,
    # so behavior is identical to the original construction.  For truncated
    # MPS, Q[:,alpha] is the normalized leading singular direction of the
    # truncated state — the physically correct thing to prepare on a quantum
    # circuit, since quantum state preparation always requires a unit-norm
    # input.  This is an extension beyond Ran 2020, which does not address
    # the truncated case.
    Q, R = np.linalg.qr(V, mode="reduced")  # Q: (dim, d_left), orthonormal cols
    # QR has sign ambiguity; enforce positive diagonal of R so that Q
    # columns align with the isometry columns and global phases are consistent.
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs

    # Full SVD of Q gives an orthonormal complement (null(Q^H)) for the
    # free columns, guaranteeing a valid unitary without Gram-Schmidt.
    U_svd, _, _ = np.linalg.svd(Q, full_matrices=True)
    null_cols = U_svd[:, d_left:]  # shape (dim, dim - d_left)

    U = np.zeros((dim, dim), dtype=complex)
    for alpha in range(d_left):
        U[:, iso_indices[alpha]] = Q[:, alpha]
    for i, j in enumerate(free_indices):
        U[:, j] = null_cols[:, i]

    return U


def mps_to_circuit(
    tensors: list[np.ndarray],
) -> QuantumCircuit:
    """Convert MPS site tensors to a quantum circuit (Ran 2020).

    Layout: q physical qubits (0..q-1) + n_bond bond qubits (q..q+n-1).
    Each site tensor becomes a unitary on [phys_k, bond_0, ..., bond_{n-1}].
    Applied left to right (site 0 first). All qubits start in |0>.

    After the circuit, the bond register returns to |0> (for exact MPS),
    so the physical qubits hold the target state.
    """
    q = len(tensors)

    max_bond = 1
    for t in tensors:
        max_bond = max(max_bond, t.shape[0], t.shape[2])
    n_bond = int(np.ceil(np.log2(max(max_bond, 2))))

    n_total = q + n_bond
    qc = QuantumCircuit(n_total, name="MPS_prep")

    bond_qubits = list(range(q, q + n_bond))

    for k in range(q):
        A_k = tensors[k]
        U = site_tensor_to_unitary(A_k, n_bond)
        n_gate = 1 + n_bond
        # Site 0 = MSB of grid index -> qubit q-1 (Qiskit MSB)
        phys_qubit = q - 1 - k
        target = [phys_qubit] + bond_qubits
        gate = UnitaryGate(U, label=f"A_{k}")
        qc.append(gate, target[:n_gate])

    return qc

