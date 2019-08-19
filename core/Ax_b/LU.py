import numpy as np

def LU_decompose(mat, only_upper=False, get_elementary=False):
    # Description: This function will achieve the LU decomposition of a given matrix.
    # args:
    #   mat: The matrix prepared to be transformed to upper triangle.
    #   only_upper: A bool, if it is False, then we use left-multiply matrix to realize row exchange of matrix.
    #   get_elementary: A bool, if it's True, we'll also return the elementary matrix.
    # return: The upper triangle matrix

    # (1) Catch exceptions and initialize variables
    if not isinstance(mat, type(np.array([1, 2]))):
        raise Exception('mat must be an array')
    if mat.ndim != 2:
        raise Exception('The dimension of array must be 2')
    if mat.shape[0] < 2:
        raise Exception('The number of row must larger than 1')
    if (mat == 0).all():
        raise Exception('zero matrix is forbidden')
    upper = mat.copy()
    upper = upper.astype(float)
    nrow, ncol = mat.shape
    pivot_row, pivot_col = 0, 0
    L, elementary = [], []
    num_exchange = [0]

    # (2) Do the Gaussian elimination
    def row_exchange(pivot_row, pivot_col, matrix, only_upper):

        num_exchange[0] += 1
        if not only_upper:
            pivots = matrix[pivot_row + 1:, [pivot_col]]
            nonZero1th_idx = np.where(pivots != 0)[0][0] + 1
            permutation = np.identity(matrix.shape[0])
            permutation[[pivot_row, pivot_row + nonZero1th_idx], :] = permutation[[pivot_row + nonZero1th_idx, pivot_row], :]
            matrix = permutation @ matrix
            elementary.append(permutation)
            L.append(permutation.T)
        else:
            pivots = matrix[pivot_row + 1:, [pivot_col]]
            nonZero1th_idx = np.where(pivots != 0)[0][0] + 1
            matrix[[pivot_row, pivot_row + nonZero1th_idx], :] = matrix[[pivot_row + nonZero1th_idx, pivot_row], :]

        return matrix

    def row_subtract(pivot_row, pivot_col, matrix, only_upper):

        if not only_upper:
            E = np.identity(matrix.shape[0])
            multiplier = matrix[pivot_row + 1:, [pivot_col]] / matrix[pivot_row, pivot_col]
            E[pivot_row+1:, [pivot_row]] = - multiplier
            matrix = E @ matrix
            elementary.append(E)
            inverse_elementary = E.copy()
            inverse_elementary[pivot_row+1:, [pivot_row]] = multiplier
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

        if (np.abs(upper[pivot_row:, [pivot_col]]) < 10**(-10)).all():
            pivot_col += 1
        elif np.abs(upper[pivot_row, pivot_col]) > 10**(-10):
            upper = row_subtract(pivot_row, pivot_col, upper, only_upper=only_upper)
            pivot_row += 1
            pivot_col += 1
        else:
            upper = row_exchange(pivot_row, pivot_col, upper, only_upper=only_upper)
            upper = row_subtract(pivot_row, pivot_col, upper, only_upper=only_upper)
            pivot_row += 1
            pivot_col += 1

    # (3) Tidy result
    upper = upper.round(4)
    if only_upper:
        return upper
    else:
        lower = L[0]
        for i in range(1, len(L)):
            lower = lower @ L[i]
        lower = lower.round(4)
        if get_elementary:
            return {'upper': upper, 'lower': lower, 'elementary': elementary, 'num_exchange': num_exchange[0]}
        else:
            return {'upper': upper, 'lower': lower, 'num_exchange': num_exchange[0]}
