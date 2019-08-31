import numpy as np
from numba import jit

@jit
def hess_from_coords(coords,cutoff=15.,gamma=1.):
    dof = coords.size
    n_atoms = coords.shape[0]
    assert np.isclose(dof / 3.,n_atoms)
    cutoff2 = cutoff * cutoff
    g = gamma # TODO: function for distance dependence / scalar map value
    hessian = np.zeros((dof, dof), np.float32)
    for i in range(n_atoms):
        res_i3 = i*3
        res_i33 = res_i3+3
        i_p1 = i+1
        i2j_all = coords[i_p1:, :] - coords[i]
        for j, dist2 in enumerate((i2j_all ** 2).sum(1)):
            if dist2 > cutoff2:
                continue
            i2j = i2j_all[j]
            j += i_p1
            #g = gamma(dist2, i, j)
            res_j3 = j*3
            res_j33 = res_j3+3
            super_element = np.outer(i2j, i2j) * (- g / dist2)
            hessian[res_i3:res_i33, res_j3:res_j33] = super_element
            hessian[res_j3:res_j33, res_i3:res_i33] = super_element
            hessian[res_i3:res_i33, res_i3:res_i33] = \
                hessian[res_i3:res_i33, res_i3:res_i33] - super_element
            hessian[res_j3:res_j33, res_j3:res_j33] = \
                hessian[res_j3:res_j33, res_j3:res_j33] - super_element
    return(hessian)