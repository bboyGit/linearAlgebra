import numpy as np
from core.Ax_b.Determinant import det

def isPositiveDef(mat):
    """
    Desc: Check whether the matix are positive definite
    Parameters:
      mat: A matrix
    Return: A bool
    """
    nrow, ncol = mat.shape
    if nrow != ncol:
        raise Exception("mat must be a square matrix")

    for i in range(nrow):
        if i == 0:
            det_i = mat[0, 0]
        else:
            sub_mat = mat[:(i + 1), :(i + 1)]
            det_i = det(sub_mat)
        if det_i <= 0:
            return False

    return True

if __name__ == "__main__":
    mat = np.array([[1, 2, 3], [2, 1, 5], [0, 3, 6]])
    isPositiveDef(mat)