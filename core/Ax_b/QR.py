import numpy as np
from core.Ax_b.orthogonal import ortho

def qr_decompose(mat):
    """
    Desc: This function achieve qr decomposition by Gram-schmidt process.
    Parameters:
     mat: The matrix to do qr decomposition
    return: The orthonormal matrix Q and corresponding upper triangle matrix R.
    """

    # (1) Do Gram-schmidt process to obtain Q
    nrow, ncol = mat.shape
    Q = ortho(mat, unit=True)

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
