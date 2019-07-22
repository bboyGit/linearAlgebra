import numpy as np
from core.Ax_b.LU import LU_decompose

def LDU_decompose(mat):

    # (1) get the basic LU decomposition
    lu = LU_decompose(mat, only_upper=False)
    upper = lu['upper']
    lower = lu['lower']

    # (2) transform upper triangle matrix into a diagonal matrix and a upper triangle
    # regular: In a row, if pivot is 0, insert 1 into the corresponding entry, else, insert the pivot.
    nrow, ncol = upper.shape
    if nrow <= ncol:
        D = np.array([d if d else 1 for d in upper.diagonal()])
    else:
        D1 = [d if d else 1 for d in upper.diagonal()]
        D1.extend([1] * (nrow - ncol))
        D = np.array(D1)

    diag = np.diag(D)
    upper_new = np.multiply(upper, np.array([1/D]).T)
    upper_new = upper_new.round(4)

    return {'lower': lower, 'diag': diag, 'upper_new': upper_new}
