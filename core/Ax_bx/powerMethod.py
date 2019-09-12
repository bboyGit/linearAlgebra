import numpy as np
from core.Ax_b.Inverse import inverse

def max_eigenvalue(mat, initial, iter_step):
    """
    Desc: Solve the largest abs eigen value of a matrix by power method
    Parameters:
      mat: A given 2-D array
      initial: A column vector for the initial guess.
      iter_step: A int indicating the number of iteration
    Return: max eigenvalue and the corresponding eigen-vector.
    """

    # (1) Get the eigen vector which is corresponding to max eigen value
    mat = mat.copy()
    pre = initial.copy()
    for i in range(iter_step):
        after = mat @ pre
        alpha = after[0]
        after = after/alpha
        pre = after
    eigen_vector = after.copy()
    eigen_vector = eigen_vector/eigen_vector[0, 0]

    # (2) Get the max eigen value by least square
    eigen_value = alpha
    result = {'eigen_value': eigen_value[0], 'eigen_vector': eigen_vector}

    return result

def power_method(mat, initial, step, iter_step):
    '''
    Desc: Get all the eigen value and eigen vector of a given matrix
    Parameters:
      mat: The given matrix
      initial: The first guess vector.
      step: A int determining how many divisions we'd like to divide the potential range of all eigen values.
    Return: An array of eigenvalues
    '''

    # (1) find the max and min eigen value of given matrix
    max_eig = max_eigenvalue(mat, initial, iter_step)
    max_eigv = max_eig['eigen_value']

    # (2) Determine the range of eigen values of the given matrix
    eig_range = np.linspace(-np.abs(max_eigv), np.abs(max_eigv), step + 2)
    eig_range = eig_range[1:-1]

    # (3) Iterate eig_range and get the eigen value closest to each point
    I = np.identity(mat.shape[0])
    eigen_values = []
    for s in eig_range:
        mats = mat - s * I
        inv_mats = inverse(mats)
        invmats_eig = max_eigenvalue(inv_mats, initial, iter_step)
        invmats_eigv = invmats_eig['eigen_value']
        mats_eigv = 1/invmats_eigv + s
        eigen_values.append(mats_eigv)
    eigen_values = np.array(eigen_values).round(3)          # 注意：这是一个不稳定的地方，虽然随着eigen_range被分得越来越细，理论上我们可以得到所有的特征值。
    eigen_values = np.unique(eigen_values)                  # 但这些特征值可能有相同的，但是我们无法完美地识别，因为你不知道从哪一个精度上round可以把所有重复值变成真的相同大小的值。这是幂迭代的一大弱点。
                                                            # 另一大弱点是当且仅当初始向量不与最大特征值对应的特征向量正交时才能找到最大特征值及其特征向量。
                                                            # 最后一大弱点是初始向量还必须在特征空间中。
    return eigen_values


if __name__ == '__main__':
    mat = np.array([[1, 0, -1/3], [0, 1, 2/3], [-1, 1, 1]])
    init = np.array([[1, 1, 1]]).T
    eigen = max_eigenvalue(mat, init, iter_step=50)
    eigenv = power_method(mat, init, step=50, iter_step=100)
