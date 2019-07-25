import numpy as np
from core.Ax_b.LU import LU_decompose
from core.Ax_b.nullSpace import getNullSpace
from core.Ax_b.Rref import rref
from core.Ax_b.Inverse import inverse
# The general special solution is not right, correct it !

def ax_b(mat, b):
    # args:
    #   mat: A matrix
    #   b: A column vector
    # return: If no solution: Nones. Else if unique solution: columns vector. Else if infinite solution: dict.

    # (1) Deal with Exception and try if mat has unique solution
    if mat.shape[0] != b.shape[0]:
        raise Exception("The number of rows of mat must equal to that of b")
    try:
        inv = inverse(mat)
        solution = inv @ b
    except Exception:

        # (2) Get homogeneous general solution
        col_null = getNullSpace(mat)
        general_solution = col_null['null_space']

        # (3) Get the may_special_solution v: where ref @ x = v and mat @ x = b
        expand_mat = np.concatenate([mat, b], axis=1)
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
        may_special_solution = E @ b

        # (4) Judge whether this linear system has solution or not
        all_zero = [(ref[i, :] == 0).all() for i in range(len(ref))]
        all_zero_idx = np.where(all_zero)[0]
        if (may_special_solution[all_zero_idx, :] != 0).any():
            solution = None
            Warning("This linear system has no solution")
        else:
            special_solution = may_special_solution
            solution = {"special_solution": special_solution, "general_solution": general_solution}

    return solution

if __name__ == '__main__':
    mat = np.array([[1, 3, 3, 2], [2, 6, 9, 7], [-1, -3, 3, 4]])
    y = np.array([[1, 5, 5]]).T
    solution = ax_b(mat, y)