
import numpy as np
from core.Ax_bx.eigen import eigen

def pca(x, k):
    """
    Desc: Execute principal component analysis to achieve dimensionality reduction
    Parameters:
      x: The input 2D array
      k: An int representing the object dimension
    Return: The principal component y and explained variance ratio
    """
    # (1) Make mean of x equal to zero
    x = x - x.mean(axis=0)

    # (2) Get eigenvalues and eigen vectors of x.T @ x
    eigval, eigvec = eigen(x.T @ x)
    eigval = np.diag(eigval)

    # (3) Get the matrix p where y = x @ p
    p = eigvec[:, :k]

    # (4) Get the explained variance
    explained_var = eigval[:k, :k]
    explained_var_ratio = explained_var.sum()/eigval.sum()

    # (5) Get the principal component
    comp = x @ p

    return comp, explained_var_ratio

if __name__ == "__main__":
    mat = np.array([[-1, 1, 0],
                    [0, -1, 1]])
    pca(mat, 2)