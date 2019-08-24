import numpy as np
from core.Ax_b.Rank import rank
from core.Ax_b.Inverse import inverse

def ortho(mat, unit):
    # Args:
    #   mat: A matrix
    #   unit: A bool indicating whether to scale each column to unit or not

    # (1) Deal exceptions
    if not isinstance(mat, type(np.array([0]))):
        raise Exception('mat must be an array')
    if mat.ndim != 2:
        raise Exception("Dimension of mat must be 2")

    nrow, ncol = mat.shape
    r = rank(mat)
    # if r < ncol:
    #     raise Exception("Columns of mat are not linearly independent.")

    # (2) Do Gram-schmidt process to obtain Q
    Q = []
    for i in range(ncol):
        col = mat[:, [i]]
        if i == 0:
            q = col
        else:
            qi = np.concatenate(Q, axis=1)
            proj = qi @ inverse(qi.T @ qi) @ qi.T
            orthogonal = col - proj @ col
            q = orthogonal.round(10)

        if unit and np.abs(q).sum() > 10**(-10):
            q = q/np.sqrt(np.dot(q.T, q))

        Q.append(q)
    Q = np.concatenate(Q, axis=1)

    return Q

if __name__ == '__main__':
    # A = np.array([[1, 0, 1], [1, 0, 0], [2, 1, 0]])
    # ortho(A, unit=False)
    mat = np.array([[1, 0, -1 / 3], [0, 1, 2 / 3], [-1, 1, 1]])
    ortho(mat, unit=True)