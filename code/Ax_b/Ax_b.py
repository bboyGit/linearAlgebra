import numpy as np
import pandas as pd
from code.Ax_b.LU import LU_decompose
from code.Ax_b.nullSpace import getNullSpace
from code.Ax_b.Rref import rref
from code.Ax_b.Inverse import inverse

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
        solution = inv @ b
    except Exception:

        # (3) Get homogeneous general solution
        col_null = getNullSpace(mat)
        general_solution = col_null['null_space']
        pivot_col = col_null['pivot_idx']
        free_col = col_null['free_idx']

        # (4) Get v: where ref @ x = v and mat @ x = b
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
        all_zero = [(ref[i, :] == 0).all() for i in range(len(ref))]
        all_zero_idx = np.where(all_zero)[0]
        if (v[all_zero_idx, :] != 0).any():
            solution = None
            Warning("This linear system has no solution")
        else:
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

if __name__ == '__main__':
    mat = np.array([[1, 3, 3, 2], [2, 6, 9, 7], [-1, -3, 3, 4]])
    y = np.array([[1, 5, 5]]).T
    solution = ax_b(mat, y)
