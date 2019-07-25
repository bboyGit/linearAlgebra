import numpy as np
from code.Ax_b.nullSpace import getNullSpace

def rank(mat):
    # args:
    #   mat: A matrix
    # return: the rank of matrix

    col_null = getNullSpace(mat)
    colspace = col_null['column_space']
    result = colspace.shape[1]

    return result

if __name__ == "__main__":
    mat = np.array([[1, 3, 3, 2], [2, 6, 9, 7], [-1, -3, 3, 4]])
    r = rank(mat)
