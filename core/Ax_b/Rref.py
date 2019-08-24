import numpy as np
from core.Ax_b.LU import LU_decompose
from core.Ax_b.Inverse import inverse


def rref(mat):
    """
    Desc: Calculate the reduced row echelon form matrix by Gaussian elimination.
    Parameters:
      mat: A matrix prepared to become simplest row echelon form matrix.
    Return: A simplest row echelon form matrix.
    """

    # (1)get LU decomposition
    lu = LU_decompose(mat)
    upper = lu['upper']

    # (2)from upper triangular to simplest row echelon matrix
    def row_subtract(matrix, nrow, nonZero_idx):

        E = np.identity(matrix.shape[0])
        multiplier = matrix[:nrow, [nonZero_idx]] / matrix[nrow, nonZero_idx]
        E[:nrow, [nrow]] = - multiplier
        matrix = E @ matrix
        elementary.append(E)

        return matrix
    pivot_row, pivot_col = upper.shape
    elementary = []
    try:
        # If the upper triangular is non-singular, then it can be reduced to a identity matrix.
        inv = inverse(upper)
        ref = upper.copy()
        for i in range(1, pivot_row):
            idx = np.where(upper[i, :] != 0)[0]
            first_idx = idx[0]
            ref = row_subtract(ref, i, first_idx)

    except Exception:
        # (3) let each non-zero pivot to be 1.
        for i in range(pivot_row):
            idx = np.where(upper[i, :] != 0)[0]
            if idx.shape[0] == 0:
                # All of this row in upper are zero
                continue
            else:
                first_idx = idx[0]
                E = np.identity(upper.shape[0])
                E[i, i] = 1/upper[i, first_idx]
                upper = E @ upper
                elementary.append(E)

        # (4) Use elimination to get the simplest row echelon form matrix
        ref = upper.copy()
        for i in range(1, pivot_row):
            idx = np.where(upper[i, :] != 0)[0]
            if idx.shape[0] == 0:
                continue
            else:
                first_idx = idx[0]
                ref = row_subtract(ref, i, first_idx)

    return {"rref": ref, 'elementary': elementary}

