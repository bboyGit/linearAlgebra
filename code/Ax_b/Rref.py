import numpy as np
from code.Ax_b.LU import LU_decompose
from code.Ax_b.Inverse import inverse


def rref(mat):
    # args:
    #   mat: A matrix prepared to become simplest row echelon form matrix.
    # return: A simplest row echelon form matrix.

    # (1)get LU decomposition
    lu = LU_decompose(mat)
    upper = lu['upper']

    # (2)from upper triangular to simplest row echelon matrix
    try:
        # If the upper triangular is non-singular, then it can be reduced to a identity matrix.
        inv = inverse(upper)
        ref = np.identity(mat.shape[0])

    except:
        elementary = []
        pivot_row, pivot_col = upper.shape
        pivot_row -= 1
        pivot_col -= 1
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
        def row_subtract(matrix, nrow, nonZero_idx):

            E = np.identity(matrix.shape[0])
            multiplier = matrix[:nrow, [nonZero_idx]] / matrix[nrow, nonZero_idx]
            E[:nrow, [nrow]] = - multiplier
            matrix = E @ matrix
            elementary.append(E)

            return matrix

        ref = upper.copy()
        for i in range(1, pivot_row):
            idx = np.where(upper[i, :] != 0)[0]
            if idx.shape[0] == 0:
                continue
            else:
                first_idx = idx[0]
                ref = row_subtract(ref, i, first_idx)

    return {"rref": ref, 'elementary': elementary}

if __name__ == "__main__":
    mat = np.array([[1, 3, 3, 2], [2, 6, 9, 7], [-1, -3, 3, 4]])
    r = rref(mat)
