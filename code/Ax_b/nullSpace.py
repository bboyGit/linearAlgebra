import numpy as np
import pandas as pd
from core.Ax_b.Rref import rref

def getNullSpace(mat):
    # args:
    #   mat: A matrix
    # return: A matrix whose columns are the basis of null space of mat

    # (1) Get the simplest reduced row echelon form matrix
    ref = rref(mat)

    # (2) Find column space and null space of ref
    nrow, ncol = ref.shape
    if nrow == ncol and (ref == np.identity(ref.shape[0])).all():
        colspace = ref
        nullspace = np.array([[0] * ref.shape[0]]).T

    else:
        col_bool = []
        df = pd.DataFrame(ref)
        # (2.1) Get the columns index of pivot_variable and free_variable
        for col in df:
            the_col = df[col]
            cond1 = set(the_col.unique()) == set([1, 0])
            cond2 = the_col[the_col == 1].shape[0] == 1
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

        # (2.2) Iter each free variable to set it to 1 and other free variables to 0.

        for sp_idx in range(len(free_col)):
            pivot_free1 = [1 if i in pivot_col or i == free_col[sp_idx] else 0 for i in range(df.shape[1])]
            make_zero = np.array([pivot_free1])
            r = np.multiply(ref, make_zero)
            drop_all_zero_row = [False if (r[i, :] == 0).all() else True for i in range(r.shape[0])]
            r = r[drop_all_zero_row, :]

            free_solution = {}
            for idx, value in enumerate(free_col):
                if idx == sp_idx:
                    free_solution[value] = 1
                else:
                    free_solution[value] = 0
            free_solution = pd.DataFrame(free_solution, index=['solution' + str(sp_idx + 1)])

            pivot_solution = {}
            for j in range(r.shape[0]):
                this_row = r[j, :]
                free_var_col_nonzero = free_col[sp_idx]
                free = this_row[free_var_col_nonzero]
                pivot = this_row[this_row != 0][0]
                pivot_idx = np.where(this_row != 0)[0][0]
                pivot_solution[pivot_idx] = - free/pivot
            pivot_solution = pd.DataFrame(pivot_solution, index=['solution' + str(sp_idx + 1)])

            special_solution = pd.concat([pivot_solution, free_solution], axis=1)
            special_solution = special_solution.reindex(columns=range(special_solution.shape[1]))

            if 'nullspace' not in locals():
                nullspace = special_solution.T
            else:
                nullspace = pd.concat([nullspace, special_solution.T], axis=1)

    return {'null_space': nullspace, 'column_space': colspace}
