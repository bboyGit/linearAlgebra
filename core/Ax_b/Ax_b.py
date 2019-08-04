import numpy as np
import pandas as pd
from code.Ax_b.LU import LU_decompose
from code.Ax_b.nullSpace import getNullSpace
from code.Ax_b.Rref import rref
from code.Ax_b.Inverse import inverse
from warnings import warn

def ax_b(mat, b):
    # args:
    #   mat: A matrix
    #   b: A column vector
    # return: If no solution: Nones. Else if unique solution: columns vector. Else if infinite solution: dict.

    # (1) Deal with Exception and try if mat has unique solution
    if mat.shape[0] != b.shape[0]:
        raise Exception("The number of rows of mat must equal to that of b")
    # (2) Check if it has unique solution
    try:
        inv = inverse(mat)
        # Unique solution
        solution = inv @ b
    except Exception:
        # There's no unique solution
        # (3) Get homogeneous general solution
        col_null = getNullSpace(mat)
        general_solution = col_null['null_space']
        pivot_col = col_null['pivot_idx']
        free_col = col_null['free_idx']

        # (4) Get v: where ref @ x = v and mat @ x = b (cause E @ mat = ref, E @ b = v).
        result = rref(mat)
        ref = result['rref']
        elementary2 = result['elementary']
        lu = LU_decompose(mat, get_elementary=True)
        elementary1 = lu['elementary']
        elementary = elementary1 + elementary2
        for i in range(len(elementary)):
            if 'E' not in locals():
                E = elementary[i]
            else:
                E = elementary[i] @ E
        v = E @ b

        # (5) Judge whether this linear system has solution or not
        all_zero = [(ref[i, :] == 0).all() for i in range(ref.shape[0])]
        all_zero_idx = np.where(all_zero)[0]
        if (v[all_zero_idx, :] != 0).any():
            # No solution
            solution = None
            warn("This linear system has no solution")
        else:
            # Infinite solution
            # (6) Find special_solution
            expand = np.concatenate([ref, v], axis=1)
            special_solution = {}
            for i in range(ref.shape[0]):
                the_row = ref[i, :]
                nonzero = np.where(the_row != 0)[0]
                if nonzero.shape[0] == 0:
                    continue
                else:
                    nonzero_col = nonzero[0]
                    special_solution[nonzero_col] = v[i, 0]
            for j in free_col:
                special_solution[j] = 0
            special_solution = pd.DataFrame(special_solution, index=['sp_solution'])
            special_solution = special_solution.reindex(columns=range(special_solution.shape[1]))
            special_solution = special_solution.T

            solution = {"special_solution": special_solution, "general_solution": general_solution}

    return solution
