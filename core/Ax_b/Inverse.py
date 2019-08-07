import numpy as np
from core.Ax_b.LU import LU_decompose
from warnings import warn

def inverse(mat):
    # args:
    #   mat: A matrix
    # return: The inverse of mat

    # (1) Catch exceptions and initialize variables
    if mat.ndim != 2:
        raise Exception('The dimension of array must be 2')
    nrow, ncol = mat.shape
    if nrow == ncol == 1:
        warn('The mat has only 1 element')
        return 1/mat

    if nrow != ncol:
        raise Exception('The input matrix must be a square matrix')

    lu = LU_decompose(mat, get_elementary=True)
    upper = lu['upper']
    elementary = lu['elementary']
    diag = upper.diagonal()
    if any(diag == 0):
        raise Exception('Zero pivot exists, so the matrix is singular.')

    # (2) Get inverse matrix of upper
    upper_new = upper.copy()
    for i in range(nrow - 1, 0, -1):
        multiplier = upper[:i, i]/upper[i, i]
        E = np.identity(mat.shape[0])
        E[:i, i] = - multiplier
        upper_new = E @ upper_new
        elementary.append(E)

    E = np.diag(1/upper_new.diagonal())
    elementary.append(E)

    # (3) Get inverse matrix of mat
    inv = elementary[0]
    for j in range(1, len(elementary)):
        inv = elementary[j] @ inv

    return inv


