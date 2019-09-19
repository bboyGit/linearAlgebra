import numpy as np
import pandas as pd
from core.Ax_bx.qr import qr
from core.Ax_b.nullSpace import getNullSpace

def eigen(mat):
    """
    Desc: Calculate eigenvalues and eigen vectors
    Parameters:
      mat: The given square matrix
    Return: A tuple with 2 elements where the first one is an 1d array containing eigenvalues and
            the second one is a 2d array whose columns are eigen vectors
    """
    # (1) Deal exceptions
    nrow, ncol = mat.shape
    if nrow != ncol:
        raise Exception("The given matrix must be square.")

    # (2) Calculate eigenvalues
    eigval = qr(mat, shift=True, step=30, hess=True, tol=10**(-10))

    # (3) Calculate eigen vectors
    eigvec = []
    for eigv in eigval:
        m = mat - np.identity(nrow) * eigv
        space = getNullSpace(m)
        nullspace = space['null_space'].values
        nullspace = nullspace/np.sqrt(nullspace.T @ nullspace)
        eigvec.append(nullspace)
    eigvec = np.concatenate(eigvec, axis=1)

    # (4) Tidy result
    result = (eigval, eigvec)

    return result

if __name__ == "__main__":
    mat = np.array([[1, 2, 3], [1, 3, 3], [1, 2, 4]])
    eig = eigen(mat)
    mat1 = np.array([[1, 2, 3, 9], [0, 1, 4, 5], [0, 0, 1, 4], [0, 0, 0, 10]])
    eig1 = eigen(mat1)


