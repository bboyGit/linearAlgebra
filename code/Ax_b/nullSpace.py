import numpy as np
import pandas as pd
from core.Ax_b.Rref import rref

def getNullSpace(mat):
    # args:
    #   mat: A matrix
    # return: A matrix whose columns are the basis of null space of mat

    # (1) get the simplest reduced row echelon form matrix
    ref = rref(mat)

    # (2) find column space and null space of ref
    nrow, ncol = ref.shape
    if nrow == ncol and (ref == np.identity(ref.shape[0])).all():
        colspace = ref
        nullspace = np.array([[0] * ref.shape[0]]).T

    else:
        col_bool = []
        df = pd.DataFrame(ref)
        for col in df:
            the_col = df[col]
            cond1 = set(the_col.unique()) == set([1,0])
            cond2 = the_col[the_col].shape[0] == 1
            if col == 0 and cond1 and cond2:
                col_bool.append(True)
            elif col != 0 and cond1 and cond2:
                equal_or_not = pd.Series([(the_col == df[i]).all() for i in range(col)])
                col_bool.append(False) if equal_or_not.any() else col_bool.append(True)
            else:
                col_bool.append(False)

        col_bool = pd.Series(col_bool)
        colspace = ref[:, col_bool]
        pivot_col = list(col_bool[col_bool].index)
        free_col = list(col_bool[~col_bool].index)

        for sp_idx in range(len(free_col)):
            pivot_free1 = [1 if i in pivot_col or i == free_col[sp_idx] else 0 for i in range(df.shape[1])]
            make_zero = np.array([pivot_free1]).T
            r = ref @ make_zero




    return nullspace


if __name__ == "__main__":

    nullspace = getNullSpace(mat)