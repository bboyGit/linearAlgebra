import numpy as np
from core.Ax_b.LU import LU_decompose

def rank(mat):
    # args:
    #   mat: A matrix
    # return: the rank of matrix

    lu = LU_decompose(mat)
    upper = lu['upper']
    nrow, ncol = upper.shape
    row_bool = [False if (upper[i, :] == 0).all() else True for i in range(nrow)]
    u = upper[row_bool, :]
    result = u.shape[0]

    return result

if __name__ == "__main__":
    mat = np.array([[1, 3, 3, 2], [2, 6, 9, 7], [-1, -3, 3, 4]])
    r = rank(mat)
