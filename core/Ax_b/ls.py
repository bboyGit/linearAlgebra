import numpy as np
from core.Ax_b.Inverse import inverse
from core.Ax_b.Rank import rank

def ols(x, y, const=True):
    # Args:
    #   x: A matrix contain explanatory variables
    #   y: A column vector contain dependent variable
    #   const: A bool, indicating whether we add constant or not. Default True, which means add constant.
    # Return: A dict contain fitted values, coefficients and errors.

    # (1) Handle Exceptions
    type_array = type(np.array([[0]]))
    if not isinstance(x, type_array) or not isinstance(y, type_array):
        raise TypeError("Both x and y must be array.")
    if x.ndim != 2 or y.ndim != 2:
        raise Exception("Dimension of x and y must be 2.")
    if x.shape[0] != y.shape[0]:
        raise Exception("The number of observations of x and y must be the same.")

    # (2) Calculate coefficients, errors and fitted values.
    if const:
        c = np.array([[1] * x.shape[0]]).T
        x = np.concatenate([c, x], axis=1)
    if rank(x) < x.shape[1]:
        raise Exception("Columns of x are linearly dependent.")

    mat1 = inverse(x.T @ x) @ x.T
    beta = mat1 @ y
    y_hat = x @ beta
    error = y - y_hat

    # (3) Tidy results
    result = {'coef': beta, 'fitted_value': y_hat, 'resid': error}

    return result

def wls(x, y, const=True, weight=None):
    # Args:
    #   x: A matrix contain explanatory variables
    #   y: A column vector contain dependent variable
    #   const: A bool, indicating whether we add constant or not. Default True, which means add constant.
    #   weight: An 1 dimensional array. Default None.
    # Return: A dict contain fitted values, coefficients, errors, weight, transfer x and transfer y..

    # (1) Handle Exceptions
    type_array = type(np.array([[0]]))
    if not isinstance(x, type_array) or not isinstance(y, type_array):
        raise TypeError("Both x and y must be array.")
    if x.ndim != 2 or y.ndim != 2:
        raise Exception("Dimension of x and y must be 2.")
    if x.shape[0] != y.shape[0]:
        raise Exception("The number of observations of x and y must be the same.")

    # (2) Deal weight
    if weight == None:
        # 用户不填权重时我们默认用残差绝对值的倒数作为权重
        ols_fit = ols(x, y, const)
        error= np.abs(ols_fit['resid'][:, 0])
        weight = np.diag(error)
        weight = np.sqrt(weight)

    elif isinstance(weight, type_array):
        if (weight < 0).any():
            raise Exception("Negative weight is forbidden")
        elif weight.ndim != 1:
            raise Exception("The dimension of weight must be 1")
        else:
            weight = np.diag(weight)
            weight = np.sqrt(weight)
    else:
        raise TypeError("Type of weight must be array")

    # (3) Calculate coef, fitted value and error
    if const:
        c = np.array([[1] * x.shape[0]]).T
        x = np.concatenate([c, x], axis=1)
    if rank(x) < x.shape[1]:
        raise Exception("Columns of x are linearly dependent.")

    X = weight @ x
    Y = weight @ y
    mat1 = inverse(X.T @ X) @ X.T
    beta = mat1 @ Y
    Y_hat = X @ beta
    error = Y - Y_hat

    # (3) Tidy results
    result = {'coef': beta, 'fitted_value': Y_hat,
              'resid': error, 'weight': weight.diagonal(),
              'transfer_x': X, 'transfer_y': Y}

    return result

if __name__ == '__main__':
    y = np.array([[2, 1, 4, 3, 5]]).T
    x = np.array([[1, 2.5, 3, 5, 4]]).T
    fit = ols(x, y, const=False)
    wfit = wls(x, y, const=False)
