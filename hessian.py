import numpy as np
import scipy
from numba import jit

@jit
def hess_from_coords(coords,cutoff=15.,gamma=1.,sparse=True):
    '''
    sparse: encoding the hessian as sparse is very important since the final matrix will be sparse. suitable for very large matrices since the memory won't swamp with a large sparse matrix
    '''
    assert coords.shape[1] == 3, 'coords shape is not Nx3'
    dof = coords.size
    n_atoms = coords.shape[0]
    #assert np.isclose(dof / 3.,n_atoms)
    cutoff2 = cutoff * cutoff
    g = gamma # TODO: function for distance dependence / scalar map value
    hess = np.zeros((dof, dof), np.float32)
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
            #TODO unravel and return values, cols and rows
            #TODO: try to vectorize local transpose symetry
            hess[res_i3:res_i33, res_j3:res_j33] = super_element
            hess[res_j3:res_j33, res_i3:res_i33] = super_element
            # TODO: test vectorized summing up rows/cols outside of j loop
            hess[res_i3:res_i33, res_i3:res_i33] = \
                hess[res_i3:res_i33, res_i3:res_i33] - super_element
            hess[res_j3:res_j33, res_j3:res_j33] = \
                hess[res_j3:res_j33, res_j3:res_j33] - super_element
    return(hess)

@jit
def hess_from_coords_coo_denseblockdiag(coords,cutoff=15.,gamma=1.):
    '''
    sparse: encoding the hessian as sparse is very important since the final matrix will be sparse. suitable for very large matrices since the memory won't swamp with a large sparse matrix
    '''
    assert coords.shape[1] == 3, 'coords shape is not Nx3'
    dof = coords.size
    n_atoms = coords.shape[0]
    #assert np.isclose(dof / 3.,n_atoms)
    cutoff2 = cutoff * cutoff
    g = gamma # TODO: function for distance dependence / scalar map value
    hess_data, hess_rows, hess_cols = [],[],[]
    hess_diagonals=np.zeros((n_atoms,3,3))
    for i in range(n_atoms):
        res_i3 = i*3
        res_i33 = res_i3+3
        i_p1 = i+1
        i2j_all = coords[i_p1:, :] - coords[i]
        for j, dist2 in enumerate((i2j_all ** 2).sum(1)):
            if dist2 > cutoff2: # TODO: precompute with kdtree, fill in array instead of extent lists, since will know how large the lists are at the start
                continue
            i2j = i2j_all[j]
            j += i_p1
            #g = gamma(dist2, i, j)
            res_j3 = j*3
            res_j33 = res_j3+3
            super_element = np.outer(i2j, i2j) * (- g / dist2)
            # append to data, rows, cols
            super_element_list=super_element.flatten()
            hess_data.extend(super_element_list)
            hess_rows.extend([res_i3,res_i3,res_i3,res_i3+1,res_i3+1,res_i3+1,res_i3+2,res_i3+2,res_i3+2])
            hess_cols.extend([res_j3,res_j3+1,res_j3+2,res_j3,res_j3+1,res_j3+2,res_j3,res_j3+1,res_j3+2])
            #TODO: transpose this later?
            hess_data.extend(super_element_list)
            hess_rows.extend([res_j3,res_j3,res_j3,res_j3+1,res_j3+1,res_j3+1,res_j3+2,res_j3+2,res_j3+2])
            hess_cols.extend([res_i3,res_i3+1,res_i3+2,res_i3,res_i3+1,res_i3+2,res_i3,res_i3+1,res_i3+2])

            hess_diagonals[i]-=super_element
            hess_diagonals[j]-=super_element


    return(hess_data, hess_rows, hess_cols,hess_diagonals)

def hess_from_coords_csr(coords,**kwargs):
    '''
    returns csr hessian (save tie since already sparse)
    '''
    dof = coords.size
    hess_data, hess_rows, hess_cols,hess_diagonals = hess_from_coords_coo_denseblockdiag(coords,**kwargs)
    coo=scipy.sparse.coo_matrix((hess_data,(hess_rows,hess_cols)), shape=(dof,dof))
    bd = scipy.sparse.block_diag(hess_diagonals)
    hess_csr = bd + coo # sparse, converts to csr
    return(hess_csr)

