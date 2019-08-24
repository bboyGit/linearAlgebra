from core.Ax_b.Rank import rank
from core.Ax_b.Inverse import inverse

def project_mat(mat):
    """
    Desc: Generate projection matrix of given matrix.
    parameters:
     mat: A matrix 
    return: The projection matrix. 
    """
    r = rank(mat)
    if r != mat.shape[1]:
        raise Exception("The columns of mat is not linear independent")
    inv = inverse(mat.T @ mat)
    proj = mat @ inv @ mat.T

    return proj

def projection(v, obj_space):
    """
    Desc: Calculate projection of a given vector to a given space.
    Parameters:
      v: A column vector
      obj_space: A matrix whose column vectors are the objective space.
    Return: A column vector in space obj_space
    """

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

