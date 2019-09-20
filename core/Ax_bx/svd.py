
import numpy as np
from core.Ax_bx.eigen import eigen

def svd(mat):
    """
    Desc: Do singular value decomposition
    Parameters:
      mat: An arbitrary matrix
    Return: A tuple with 3 matrix
    """
    # (1) Compute the wanted eigenvalues and eigen vectors
    nrow, ncol = mat.shape
    A1 = mat @ mat.T
    A2 = mat.T @ mat
    u = eigen(A1)[1]
    eigvalue, v = eigen(A2)
    singular = np.sqrt(eigvalue)

    # (2) format singular value matrix
    singular = np.diag(singular)
    if ncol > nrow:
        singular = singular[:nrow, :]
    elif ncol < nrow:
        singular = np.concatenate([singular, np.zeros(nrow - ncol)])

    result = u, singular, v

    return result

if __name__ == "__main__":
    mat = np.array([[-1, 1, 0],
                    [0, -1, 1]])
    result = svd(mat)