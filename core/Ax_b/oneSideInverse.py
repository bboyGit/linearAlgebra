import numpy as np
from core.Ax_b.Rank import rank
from core.Ax_b.Inverse import inverse
from warnings import warn

def one_side_inverse(mat):
    # args:
    #   mat: A matrix
    # return: A dict with one piar-wise key-value. Key indicating which side the inverse is and value is the inverse matrix.

    try:
        inv = inverse(mat)
        warn("The matrix mat has two-side inverse")
        return {"two_side": inv}

    except Exception:
        r = rank(mat)
        nrow, ncol = mat.shape
        if r < min(nrow, ncol):
            raise Exception("Matrix mat are neither row full rank nor col full rank. One-side inverse doesn't exist.")

        elif r == nrow and r < ncol:
            # Best right-inverse: A' @ inverse(AA'). We can see, if AA' is non-singular, then AA' @ inverse(AA') = I.
            tt = mat @ mat.T
            inv = mat.T @ inverse(tt)
            return {'right': inv}

        elif r == ncol and r < nrow:
            # Best left-inverse: inverse(A'A) @ A'. We can see, if A'A is non-singular, then inverse(A'A) @ A'A = I.
            tt = mat.T @ mat
            inv = inverse(tt) @ mat.T
            return {'left': inv}
