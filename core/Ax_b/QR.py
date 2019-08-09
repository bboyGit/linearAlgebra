import numpy as np
from core.Ax_b.Rank import rank

def qr_decompose(mat):
    # Desc: This function achieve qr decomposition by Gram-schmidt process.
    # Args:
    #   mat: The matrix to do qr decomposition
    # Return: The orthonormal matrix Q and corresponding upper triangle matrix R.

    # (1) Deal exceptions
    if not isinstance(mat, type(np.array([0]))):
        raise Exception('mat must be an array')
    if mat.ndim != 2:
        raise Exception("Dimension of mat must be 2")

    nrow, ncol = mat.shape
    r = rank(mat)
    if r < ncol:
        raise Exception("Columns of mat are not linearly independent.")

    # (2) Do Gram-schmidt process to obtain Q
    Q = []
    for i in range(ncol):
        col = mat[:, [i]]
        if i == 0:
            q = col/np.sqrt(np.dot(col.T, col))
        else:
            qi = np.concatenate(Q, axis=1)
            proj = qi @ qi.T
            orthogonal = col - proj @ col
            q = orthogonal/np.sqrt(np.dot(orthogonal.T, orthogonal))
        Q.append(q)

    Q = np.concatenate(Q, axis=1)
    # (3) Obtain R by mat and Q
    R = []
    for j in range(ncol):
        q_j = Q[:, :(j+1)].T
        zeros = np.zeros([ncol - 1 - j, nrow])
        r_j = np.concatenate([q_j, zeros], axis=0) @ mat[:, [j]]
        R.append(r_j)
    R = np.concatenate(R, axis=1)

    # (4) Tidy result
    result = {'q': Q, 'r': R}

    return result

if __name__ == '__main__':
    A = np.array([[1, 0, 1], [1, 0, 0], [2, 1, 0]])
    qr = qr_decompose(A)
    print('Q.T @ Q :\n', qr['q'].T @ qr['q'])
    print('R:\n', qr['r'])
