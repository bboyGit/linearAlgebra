import numpy as np
from core.Ax_b.QR import qr_decompose
from core.Ax_bx.hessenberg import hessenberg

def qr(mat, shift, step, hess=True, tol=10**(-8)):
    """
    Desc: Use QR algorithm to calculate eigenvalues of a given square matrix
    Parameters:
      mat: A squre matrix
      shift: A bool indicating whether to use shifted QR method or not
      step: An int indicating the number of iteration
      hess: An bool indicating whether we transfer the given matrix into a hessenberg or not
      tol: A float indicating the accuracy
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
        tot_count = 500
        result = []
        while n > 1:
            count = 0
            while count < tot_count and max(np.abs(a[n - 1, :(n-1)])) > tol:
                const = a[n - 1, n - 1]
                shift = const * np.identity(n)
                qr = qr_decompose(a - shift)
                q = qr['q']
                r = qr['r']
                a = r @ q + shift
            if count < tot_count:
                lam = a[n - 1, n - 1]
                result.append(lam)
                n -= 1
                a = a[:n, :n]
            else:
                sub = a[(n - 2):, (n - 2):]
                desc = (sub[0, 0] + sub[1, 1])**2 - 4 * (sub[0, 0] * sub[1, 1] - sub[0, 1] * sub[1, 0])
                lam1 = (sub[0, 0] + sub[1, 1] + np.sqrt(desc))/2
                lam2 = (sub[0, 0] + sub[1, 1] - np.sqrt(desc))/2
                result.extend([lam1, lam2])
                n -= 2
                a = a[:n, :n]
        result.append(a[0, 0])
        result = np.array(result)
    result = result[~np.isnan(result)]

    return result

if __name__ == "__main__":
    mat = np.array([[1, 2, 3, 6],
                    [2, 4, 5, 0],
                    [0, 3, 5, 2],
                    [12, 0, 1.4, 5]])
    eigval = qr(mat, shift=False, step=50, hess=True)
    eigv = qr(mat, shift=True, step=30, hess=True, tol=10**(-10))
