import numpy as np
from core.Ax_b.LU import LU_decompose

def det(mat):
    # args:
    #   mat: A square matrix
    # return: The determinant of matrix mat.

    nrow, ncol = mat.shape
    if nrow != ncol:
        raise Exception('mat is not a square matrix.')
    lu = LU_decompose(mat)
    u = lu['upper']
    num_exchange = lu['num_exchange']
    sign = (-1)**num_exchange
    Det = u.diagonal().cumprod()[-1] * sign

    return Det


if __name__ == '__main__':
    mat1 = np.array([[0, 1, 2], [1, 1, 3], [2, 0, 5]])
    print(det(mat1))
