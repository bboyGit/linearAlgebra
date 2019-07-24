import numpy as np
from core.Ax_b.LU import LU_decompose
from core.Ax_b.Inverse import inverse

def rref(mat):
    # args:
    #   mat: A matrix prepared to become simplest row echelon form matrix.
    # return: A simplest row echelon form matrix.

    # (1)get LU decomposition
    lu = LU_decompose(mat)
    upper = lu['upper']

    # (2)from upper triangular to simplest_row_echelon_matrix
    try:
        # If the upper triangular is non-singular, then it can be reduced to a identity matrix.
        inv = inverse(upper)
        ref = np.identity(mat.shape[0])
    except:
        pivot_row, pivot_col = upper.shape
        pivot_row -= 1
        pivot_col -= 1
        # (3) let each non-zero pivot to be 1.
        for i in range(pivot_row):
            idx = np.where(upper[i, :] != 0)[0]
            if idx.shape[0] == 0:
                continue
            else:
                first_idx = idx[0]
                upper[i, :] = upper[i, :]/upper[i, first_idx]


        # (4) Use elimination to get the simplest row echelon form matrix
        def row_subtract(matrix, pivot_row):

            this_row = matrix[pivot_row, :]
            last_nonZero_idx = np.where(this_row != 0)[0][-1]
            E = np.identity(matrix.shape[0])
            multiplier = matrix[:pivot_row, [last_nonZero_idx]] / matrix[pivot_row, last_nonZero_idx]
            E[:pivot_row, [pivot_row]] = - multiplier
            matrix = E @ matrix

            return matrix

        ref = upper.copy()
        while True:

            if (np.abs(ref[pivot_row, :]) < 10**(-10)).all():
                pivot_row -= 1
            else:
                # find the non-zero column in this row
                # get multiplier and subtract
                ref = row_subtract(ref, pivot_row)
                pivot_row -= 1
                pivot_col -= 1

            if pivot_row == 0 or pivot_col == 0:
                break

    return ref


if __name__ == '__main__':

    mat = np.array([[1, 3, 3, 2], [2, 6, 9, 7], [-1, -3, 3, 4]])
    ref = rref(mat)
