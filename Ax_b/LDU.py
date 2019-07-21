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

if __name__ == '__main__':
    # print('Input is n by n')
    # mat1 = np.array([[2, 4, 5], [2, 3, 1], [4, 7, 0]])
    # ldu1 = LDU_decompose(mat1)
    # print(ldu1, '\n')
    #
    # print('Input is m by n where m < n')
    # mat2 = np.array([[3, 8, 9, 7], [11, 2, 4, 7], [9, 0, 3, 4]])
    # ldu2 = LDU_decompose(mat2)
    # print(ldu2, '\n')
    #
    # print('Input is n by m where m < n')
    # mat3 = mat2.T
    # ldu3 = LDU_decompose(mat3)
    # print(ldu3)

    from numpy.random import randint,rand
    mat = []
    for i in range(10**4):
        print(i)
        mat.append(rand(randint(2, 10, size=1)[0], randint(1, 10, size=1)[0]))
        lu = LDU_decompose(mat[i])