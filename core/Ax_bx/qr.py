import numpy as np
from core.Ax_b.QR import qr_decompose
from core.Ax_bx.hessenberg import hessenberg

def qr(mat, shift, step, hess=True):
    """
    Desc: Use QR algorithm to calculate eigenvalues of a given square matrix
    Parameters:
      mat: A squre matrix
      shift: A bool indicating whether to use shifted QR method or not
      step: An int indicating the number of iteration
      hess: An bool indicating whether we transfer the given matrix into a hessenberg or not
    Return: An array contaion eigenvalues
    """
    a = mat.copy()
    n = a.shape[0]
    q1 = np.identity(n)

    if hess:
        a = hessenberg(a)

    if not shift:
        # The unshifted qr algorithm
        for i in range(step):
            qr = qr_decompose(a)
            q = qr['q']
            r = qr['r']
            a = r @ q
            q1 = q1 @ q
        result = a.diagonal()
    else:
        # The shifted qr algorithm
        pass

    return result

if __name__ == "__main__":
    mat = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 2]])
    qr(mat, shift=False, step=30, hess=True)
