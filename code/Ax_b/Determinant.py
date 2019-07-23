import numpy as np
from core.Ax_b.LDU import LDU_decompose

def det(mat):
    # args:
    #   mat: A square matrix
    # return: The determinant of matrix mat.

    nrow, ncol = mat.shape
    if nrow != ncol:
        raise Exception('mat is not a square matrix.')
    ldu = LDU_decompose(mat)
    d = ldu['diag']
    Det = d.diagonal().cumprod()[-1]

    return Det


if __name__ == '__main__':
    mat = np.array([[1, 3, 1], [3, 1, 4], [1, -4, 5]])
    print(det(mat))