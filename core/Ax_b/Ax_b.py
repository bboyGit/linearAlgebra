import numpy as np
import pandas as pd
from core.Ax_b.LU import LU_decompose
from core.Ax_b.nullSpace import getNullSpace
from core.Ax_b.Rref import rref
from core.Ax_b.Inverse import inverse
from warnings import warn

def ax_b(mat, b):
    """
    Desc: Solve Ax=b.
    Parameters:
      mat: A matrix
      b: A column vector
    Return:
        If no solution: return Nones.
        Else if unique solution: return a columns vector.
        Else if infinite solution: return a dict containing 2 dataframes, those are special solution and general solution.
    """

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
            warn("This linear system has no solution.")
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

if __name__ == '__main__':
    mat = np.array([[1, 3, 3, 2], [2, 6, 9, 7], [-1, -3, 3, 4]])
    y1 = np.array([[1, 5, 5]]).T
    y2 = np.array([[1, 5, 6]]).T
    solution1 = ax_b(mat, y1)
    solution2 = ax_b(mat, y2)

    mat1 = np.array([[1, 2, 3], [1, 3, 3], [1, 2, 4]])
    ax_b(mat1, np.array([[3, 9, 1]]).T)
