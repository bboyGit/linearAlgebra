import numpy as np
from warnings import warn

def hessenberg(mat):
    """
    Desc: Use Household transformation to transfer any square matrix into a Hessenberg one.
    Parameters:
     mat: A square matrix.
    Return: A Hessenberg matrix
    """
    # (1) Deal exception
    m = mat.copy()
    nrow, ncol = mat.shape
    if nrow != ncol:
        raise Exception('The input matrix must be square')
    n = nrow

    zero_idx = np.where(mat > 10**(-10))
    row, col = zero_idx
    if all(col >= row):
        warn("mat is a triangular matrix")
        return mat

    # (2) Use householder transformation to transfer mat into a hessenberg
    for i in range(n - 2):
        len_x = nrow - (i + 1)
        x = m[(i + 1):, i].reshape(len_x, 1)
        z = np.array([[1] + [0] * (len_x - 1)]).T
        norm_x = np.sqrt(x.T @ x)
        v = x + z * norm_x[0]
        H = np.identity(len_x) - 2 * (v @ v.T)/(v.T @ v)
        U = np.identity(nrow)
        U[(i + 1):, (i + 1):] = H
        m = U @ m @ U                    # Change the original matrix

    return m
