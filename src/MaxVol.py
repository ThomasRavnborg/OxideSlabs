import numpy as np
from scipy.linalg import lu, solve_triangular
from time import time
#from tqdm import tqdm
#from pynep.calculate import NEP
from calorine.calculators import CPUNEP as NEP
#from src.asiIO import save_asi, load_asi

# Avoid large value since GPUMD use float
def find_inverse(m):
    return np.linalg.pinv(m, rcond=1e-8)

def maxvol(A, e, k):
    """Compute the maximal-volume submatrix for given tall matrix.

    Args:
        A (np.ndarray): tall matrix of the shape [n, r] (n > r).
        e (float): accuracy parameter (should be >= 1). If the parameter is
            equal to 1, then the maximum number of iterations will be performed
            until true convergence is achieved. If the value is greater than
            one, the algorithm will complete its work faster, but the accuracy
            will be slightly lower (in most cases, the optimal value is within
            the range of 1.01 - 1.1).
        k (int): maximum number of iterations (should be >= 1).

    Returns:
        (np.ndarray, np.ndarray): the row numbers I containing the maximal
        volume submatrix in the form of 1D array of length r and coefficient
        matrix B in the form of 2D array of shape [n, r], such that
        A = B A[I, :] and A (A[I, :])^{-1} = B.

    Note:
        The description of the basic implementation of this algorithm is
        presented in the work: Goreinov S., Oseledets, I., Savostyanov, D.,
        Tyrtyshnikov, E., Zamarashkin, N. "How to find a good submatrix".
        Matrix Methods: Theory, Algorithms And Applications: Dedicated to the Memory of Gene Golub (2010): 247-256.

    """
    n, r = A.shape

    if n <= r:
        raise ValueError('Input matrix should be "tall"')

    P, L, U = lu(A, check_finite=False)
    I = P[:, :r].argmax(axis=0)
    Q = solve_triangular(U, A.T, trans=1, check_finite=False)
    B = solve_triangular(
        L[:r, :], Q, trans=1, check_finite=False, unit_diagonal=True, lower=True
    ).T

    t0 = time()
    for iter in range(k):
        i, j = np.divmod(np.abs(B).argmax(), r)
        E = np.abs(B[i, j])
        if E <= e:
            v = iter / (time() - t0)
            print(f"Maxvol Speed: {int(v)} iters/s")
            break

        I[j] = i

        bj = B[:, j]
        bi = B[i, :].copy()
        bi[j] -= 1.0

        B -= np.outer(bj, bi / B[i, j])

    return I


def calculate_maxvol(
    A,
    struct_index,
    gamma_tol=1.001,
    maxvol_iter=1000,
    batch_size=None,
    n_refinement=10,
):
    # one batch
    if batch_size is None:
        selected = maxvol(A, gamma_tol, maxvol_iter)
        return A[selected], struct_index[selected]

    # multiple batches
    batch_num = np.ceil(len(A) / batch_size)
    batch_splits_indices = np.array_split(
        np.arange(len(A)),
        batch_num,
    )

    # stage 1 - cumulative maxvol
    A_selected = None
    struct_index_selected = None
    for i, ind in enumerate(batch_splits_indices):
        # first batch
        if A_selected is None:
            A_joint = A[ind]
            struct_index_joint = struct_index[ind]
        # other batches
        else:
            A_joint = np.vstack([A_selected, A[ind]])
            struct_index_joint = np.hstack([struct_index_selected, struct_index[ind]])

        selected = maxvol(A_joint, gamma_tol, maxvol_iter)
        if A_selected is None:
            l = 0
        else:
            l = len(A_selected)
        A_selected = A_joint[selected]
        struct_index_selected = struct_index_joint[selected]
        n_add = (selected >= l).sum()
        print(f"Batch {i}: adding {n_add} envs. ")

    # stage 2 - refinement
    for ii in range(n_refinement):
        # check max gamma, if small enough, no need to refine
        inv = find_inverse(A_selected)
        gamma = np.abs(A_selected @ inv)
        large_gamma = gamma > gamma_tol
        print(
            f"Refinement round {ii}: {large_gamma.sum()} envs out of active set. Max gamma = {np.max(gamma)}"
        )
        if np.max(gamma) < gamma_tol:
            print("Refinement done.")
            return A_selected, struct_index_selected

        A_joint = np.vstack([A_selected, A[large_gamma]])
        struct_index_joint = np.hstack(
            [struct_index_selected, struct_index[large_gamma]]
        )
        selected = maxvol(A_joint, gamma_tol, maxvol_iter)
        A_selected = A_joint[selected]
        struct_index_selected = struct_index_joint[selected]
