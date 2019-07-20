import numpy as np

def LU_decompose(mat, only_upper=False):
    # Description: This function will achieve the LU decomposition of a given matrix.
    # args:
    #   mat: The matrix prepared to be transformed to upper triangle.
    #   vectorization: A bool, if it is True, then we use left-multiply matrix to realize row exchange of matrix.
    # return: The upper triangle matrix

    if not isinstance(mat, type(np.array([1,2]))):
        raise Exception('mat must be an array')

    upper = mat.copy()
    upper = upper.astype(float)
    nrow, ncol = mat.shape
    pivot_row, pivot_col = 0, 0
    L = []

    def row_exchange(pivot_row, pivot_col, matrix, only_upper):

        if not only_upper:
            pivots = matrix[pivot_row + 1:, [pivot_col]]
            nonZero1th_idx = np.where(pivots != 0)[0][0] + 1
            E = np.identity(matrix.shape[0])
            E[[pivot_row, pivot_row + nonZero1th_idx], :] = E[[pivot_row + nonZero1th_idx, pivot_row], :]
            matrix = E @ matrix
            L.append(E.T)
        else:
            pivots = matrix[pivot_row + 1:, [pivot_col]]
            nonZero1th_idx = np.where(pivots != 0)[0][0] + 1
            matrix[[pivot_row, pivot_row + nonZero1th_idx], :] = matrix[[pivot_row + nonZero1th_idx, pivot_row], :]

        return matrix

    def row_subtract(pivot_row, pivot_col, matrix, only_upper):

        if not only_upper:
            Elementary = np.identity(matrix.shape[0])
            multiplier = matrix[pivot_row + 1:, [pivot_col]] / matrix[pivot_row, pivot_col]
            Elementary[pivot_row+1:, [pivot_col]] = - multiplier
            matrix = Elementary @ matrix
            inverse_elementary = Elementary
            inverse_elementary[pivot_row+1:, [pivot_col]] = multiplier
            L.append(inverse_elementary)
        else:
            multiplier = matrix[pivot_row + 1:, [pivot_col]] / matrix[pivot_row, pivot_col]
            l = multiplier.shape[0]
            subtract1 = np.concatenate([matrix[[pivot_row], :]] * l)
            subtract = np.multiply(subtract1, multiplier)
            matrix[pivot_row + 1:, ] = matrix[pivot_row + 1:, ] - subtract

        return matrix

    while True:

        if pivot_row >= nrow - 1 or pivot_col > ncol - 1:
            break

        if all(upper[:, [pivot_col]] == 0):
            pivot_col += 1
        elif upper[pivot_row, pivot_col] != 0:
            upper = row_subtract(pivot_row, pivot_col, upper, only_upper=only_upper)
            pivot_row += 1
            pivot_col += 1
        else:
            upper = row_exchange(pivot_row, pivot_col, upper, only_upper=only_upper)
            upper = row_subtract(pivot_row, pivot_col, upper, only_upper=only_upper)
            pivot_row += 1
            pivot_col += 1

    if only_upper:
        return upper
    else:
        lower = L[0]
        for i in range(1, len(L)):
            lower = lower @ L[i]

        return {'upper': upper, 'lower': lower}


if __name__ == '__main__':

    from numpy.random import randint
    print('The first test')
    mat1 = randint(10**5, size=(3, 1))
    upper1 = LU_decompose(mat1, only_upper=True)
    lu1 = LU_decompose(mat1)

    print('The second test')
    mat2 = np.array([[2, 3, 1], [4, 7, 5], [0, -2, 2]])
    upper2 = LU_decompose(mat2, only_upper=True)
    lu2 = LU_decompose(mat2)

    print('The third test')
    mat3 = randint(10, size=(randint(1000, size=1)[0], randint(1000, size=1)[0]))
    upper3 = LU_decompose(mat3, only_upper=True)
    lu3 = LU_decompose(mat3)
