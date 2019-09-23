
import numpy as np
from core.Ax_b.Determinant import det

def cofactor(mat, i, j):
    """
    Desc: Compute the cofactor(代数余子式) Cij of a matrix
    Parameters:
      mat: A square matrix
      i: A integer representing index of row
      j: A integer representing index of column
    Return: An int
    """
    mat = mat.copy()
    sign = (-1)**(i + j)
    mat1 = np.concatenate((mat[:i, :j], mat[:i, (j+1):]), axis=1)
    mat2 = np.concatenate((mat[(i+1):, :j], mat[(i+1):, (j+1):]), axis=1)
    sub_mat = np.concatenate((mat1, mat2), axis=0)
    Mij = det(sub_mat)           # minor (子行列式)
    Cij = sign * Mij

    return Cij

def cofactorMatrix(mat):
    """
    Desc: derive cofactor matrix(伴随矩阵)
    Parameters:
      mat:  A square matrix
    """
    nrow, ncol = mat.shape
    if nrow != ncol:
        raise Exception('mat must be square matrix')
    cofactorMat = np.zeros([nrow, ncol])

    for i in range(nrow):
        for j in range(ncol):
            cofactorMat[i, j] = cofactor(mat, i, j)

    return cofactorMat

if __name__ == '__main__':
    mat = np.array([[1, 2, 3], [0, 4, 5], [1, 0, 6]])
    cofactorMatrix(mat)
