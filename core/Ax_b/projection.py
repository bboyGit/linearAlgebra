import numpy as np
from core.Ax_b.Rank import rank
from core.Ax_b.Inverse import inverse

def project_mat(mat):

    r = rank(mat)
    if r != mat.shape[1]:
        raise Exception("The columns of mat is not linear independent")

    proj = mat @ inverse(mat.T @ mat) @ mat.T

    return proj

def projection(v, obj_space):
    # Args:
    #   v: A column vector
    #   obj_space: A matrix whose column vectors are the basis of a space
    # Return: A column vector in space obj_space

    # (1) Deal exception and Get the projection matrix of obj_space
    proj = project_mat(obj_space)
    try:
        v.shape[1]
    except IndexError:
        return Exception('The columns vector v should be 2 dimensional')

    nrow = v.shape[0]
    if nrow != obj_space.shape[0]:
        raise Exception("v and obj_space must have the same number of rows")

    # (2) Get the projection from v to space obj_space
    v_hat = proj @ v

    return v_hat

if __name__ == '__main__':
    y = np.array([[1, 2, 7]]).T
    A = np.array([[1, 1], [1, -1], [-2, 4]])
    project_mat(A)
    projection(y, A)
