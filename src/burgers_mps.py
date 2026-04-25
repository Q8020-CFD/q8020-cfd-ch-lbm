"""MPS state preparation for quantum encoding of classical fields.

Implements the MPS decomposition from Murali et al. AIAA 2026 (Eq. 5-6)
and the MPS-to-circuit conversion following Ran 2020 (Ref [27]).

Given a classical vector u of length N = K^q (K=2 for qubits), the state
is decomposed into q site tensors A^k via iterated SVD, then each site
tensor is embedded into a unitary gate for circuit preparation.
"""

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector


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


def reconstruct_from_mps(tensors: list[np.ndarray]) -> np.ndarray:
    """Contract MPS site tensors back into a full state vector.

    Contracts left to right: result[x1,x2,...,xq] = sum_j A^1 A^2 ... A^q
    """
    q = len(tensors)
    # Start with the first tensor: shape (1, 2, d1)
    result = tensors[0]  # (1, 2, d1)

    for k in range(1, q):
        # result: (1, 2^k, d_k)
        # tensors[k]: (d_k, 2, d_{k+1})
        # Contract over the bond dimension
        d_left = result.shape[-1]
        phys_left = result.shape[1]
        A_k = tensors[k]  # (d_k, 2, d_{k+1})

        # result reshaped: (phys_left, d_left)
        # multiply: (phys_left, d_left) x (d_left, 2 * d_{k+1})
        mat_left = result.reshape(-1, d_left)
        mat_right = A_k.reshape(d_left, -1)
        contracted = mat_left @ mat_right
        # Now shape: (1 * phys_left, 2 * d_{k+1})
        # which is (1, 2^(k+1), d_{k+1}) when reshaped
        d_right = A_k.shape[2]
        result = contracted.reshape(1, phys_left * 2, d_right)

    # Final shape: (1, N, 1) -> (N,)
    return result.reshape(-1)


def mps_bond_dimensions(tensors: list[np.ndarray]) -> list[int]:
    """Return the bond dimension at each cut of the MPS."""
    return [t.shape[2] for t in tensors[:-1]]


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


def mps_fidelity(
    u: np.ndarray,
    bond_dim: int | None = None,
    threshold: float = 0.0,
) -> tuple[float, float, list[int]]:
    """Compute MPS encoding fidelity for a given bond dimension.

    Returns (max_error, fidelity, bond_dims) where:
    - max_error = max |u_orig - u_reconstructed| (on normalized vectors)
    - fidelity = |<u_orig|u_reconstructed>|^2
    - bond_dims = bond dimension at each cut
    """
    u_norm, _ = normalize_state(u)
    tensors = classical_to_mps(u_norm, bond_dim=bond_dim, threshold=threshold)
    u_recon = reconstruct_from_mps(tensors)

    max_err = float(np.max(np.abs(u_norm - u_recon)))
    fidelity = float(np.abs(np.vdot(u_norm, u_recon)) ** 2)
    bonds = mps_bond_dimensions(tensors)

    return max_err, fidelity, bonds


def extract_physical_state(
    sv: Statevector,
    n_phys: int,
    n_bond: int,
) -> np.ndarray:
    """Extract physical-qubit amplitudes where bond qubits are all |0>.

    Qiskit little-endian: index = q0 + 2*q1 + 4*q2 + ...
    Physical qubits are 0..n_phys-1 (low bits).
    Bond qubits are n_phys..n_phys+n_bond-1 (high bits).

    Bond=|0> means the high bits are zero, so the relevant indices
    are simply 0..2^n_phys - 1.
    """
    N = 2**n_phys
    amplitudes = np.array(sv.data)
    return amplitudes[:N]


def mps_circuit_fidelity(
    u: np.ndarray,
    bond_dim: int | None = None,
    threshold: float = 0.0,
) -> tuple[float, float]:
    """Prepare the MPS circuit, simulate, compare to target state.

    Returns (max_amplitude_error, state_fidelity).
    """
    u_norm, _ = normalize_state(u)
    # Circuit needs right-canonical MPS for isometry condition
    tensors = classical_to_mps(
        u_norm, bond_dim=bond_dim, threshold=threshold, canonical="right"
    )
    qc = mps_to_circuit(tensors)

    q = len(tensors)
    n_bond = qc.num_qubits - q

    sv = Statevector.from_instruction(qc)
    u_circuit = extract_physical_state(sv, q, n_bond)

    max_err = float(np.max(np.abs(u_norm - u_circuit)))
    fidelity = float(np.abs(np.vdot(u_norm, u_circuit)) ** 2)

    return max_err, fidelity


if __name__ == "__main__":  # smoke test only — not the solver entry point
    import matplotlib.pyplot as plt

    # Test with sine wave on N=8
    q = 3
    N = 2**q
    x = np.linspace(0, 1, N, endpoint=False)
    u = np.sin(2.0 * np.pi * x)

    print(f"=== MPS Decomposition Test (N={N}, q={q}) ===")
    print(f"u = {u}")

    u_norm, norm = normalize_state(u)
    print(f"||u|| = {norm:.6f}")

    for bd in [None, 2, 4]:
        label = "full" if bd is None else str(bd)
        max_err, fid, bonds = mps_fidelity(u, bond_dim=bd)
        print(
            f"  bond_dim={label:>4s}: max_err={max_err:.2e}, "
            f"fidelity={fid:.10f}, bonds={bonds}"
        )

    print()

    # Test circuit preparation
    print("=== Circuit Fidelity Test ===")
    for bd in [None, 2]:
        label = "full" if bd is None else str(bd)
        max_err, fid = mps_circuit_fidelity(u, bond_dim=bd)
        print(
            f"  bond_dim={label:>4s}: max_err={max_err:.2e}, "
            f"fidelity={fid:.10f}"
        )

    # Convergence plot (like Fig. 7)
    print()
    print("=== Convergence vs Threshold (N=256) ===")
    N_big = 256
    x_big = np.linspace(0, 1, N_big, endpoint=False)
    u_big = np.sin(2.0 * np.pi * x_big)

    thresholds = [0.5, 0.05, 0.001, 5e-5, 1e-6, 1e-8, 0.0]
    errors = []
    for th in thresholds:
        max_err, fid, bonds = mps_fidelity(u_big, threshold=th)
        max_bd = max(bonds) if bonds else 0
        errors.append(max_err)
        print(
            f"  threshold={th:.1e}: max_err={max_err:.2e}, "
            f"max_bond={max_bd}"
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    u_norm_big, _ = normalize_state(u_big)
    for th in [0.5, 0.05, 0.001, 5e-5, 1e-6]:
        tensors = classical_to_mps(u_norm_big, threshold=th)
        u_recon = reconstruct_from_mps(tensors)
        # Unnormalize for plotting
        ax.plot(
            x_big,
            u_recon.real * np.linalg.norm(u_big),
            "--",
            label=f"th = {th}",
        )
    ax.plot(x_big, u_big, "k-", linewidth=2, label="Ground truth")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("u (m/s)")
    ax.set_title("MPS representation — sine wave IC")
    ax.legend()
    plt.tight_layout()
    plt.savefig("/tmp/mps_convergence.png", dpi=150)
    print(f"\nSaved convergence plot to /tmp/mps_convergence.png")
