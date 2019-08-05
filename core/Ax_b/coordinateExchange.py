import numpy as np
from core.Ax_b.Rank import rank
from core.Ax_b.oneSideInverse import one_side_inverse

def coord_exchange(coordinate, current_basis, object_basis):
    # Desc: Get coordinate in object_basis in the given coordinate of current_basis. Note that the object_basis
    #       and current_basis are the same basis of a space.
    # Args:
    #   coordinate: An array represent the coordinate in the basis of current_basis.
    #   current_basis: An array whose columns represent the current basis.
    #   object_basis: An array whose columns represent the object basis.
    # Return: An array represent the coordinate in the basis of object basis.

    # (1) Deal Exception
    rank_current = rank(current_basis)
    rank_object = rank(object_basis)
    mat = np.concatenate([current_basis, object_basis], axis=1)
    rank_both = rank(mat)
    if current_basis.shape != object_basis.shape:
        raise Exception("The number of columns and rows in current_basis and object_basis must be equal.")
    if coordinate.shape[0] != object_basis.shape[1]:
        raise Exception("The number of rows of coordinate must equal to columns of object_basis")
    if rank_object != object_basis.shape[1]:
        raise Exception("Columns of object_basis is linear dependent and it's not a basis")
    if rank_current != current_basis.shape[1]:
        raise Exception("Columns of current_basis is linear dependent and it's not a basis")
    if rank_both > rank_current:
        raise Exception("current_basis and object_basis aren't in the same space")

    # (2) Get basis-change-matrix
    inv = one_side_inverse(object_basis)
    try:
        inv = inv['two_side']
    except KeyError:
        inv = inv['left']
    basis_change = inv @ current_basis

    # (3) Get coordinate of object_basis from coord
    object_coord = basis_change @ coordinate

    return object_coord

if __name__ == "__main__":
    v = np.array([[2, 3]]).T
    current = np.identity(2)
    obj = np.array([[1, 0], [1, 1]]).T
    coor = coord_exchange(v, current, obj)
    print(coor)

    cu = np.array([[1, 0, 0], [0, 1, 0]]).T
    ob = np.array([[1, 0, 0], [1, 1, 0]]).T
    print(coord_exchange(v, cu, ob))
