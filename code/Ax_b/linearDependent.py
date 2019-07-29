import numpy as np
from code.Ax_b.nullSpace import getNullSpace

def linearDependent(mat):
    # Desc: check whether the coluumns of mat are linearly independent.
    # return: A bool, True indicating linear independent.
    nrow, ncol = mat.shape
    if ncol > nrow:
        return False
    else:
        col_null = getNullSpace(mat)
        nullpace = np.array(col_null['null_space'])
        cond = nullpace == np.array([[0] * ncol]).T
        if cond.all():
            return True
        else:
            return False

