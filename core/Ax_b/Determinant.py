import numpy as np
from core.Ax_b.LU import LU_decompose

def det(mat):
    """
    Desc: Calculate determinant by Gaussian elimination.
          Det = (-1)^t * (a1a2...an), where a1 to an are pivots of U(from LU decomposition)
          and t is the number of row exchange while LU decomposition.
    Parameters:
      mat: A square matrix
    Return: The determinant of matrix mat.
    """

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
