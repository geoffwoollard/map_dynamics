import numpy as np
import scipy
from numba import jit

@jit(nopython=True)
def hess_from_coords(coords,cutoff=15.,gamma=1.,sparse=True,kdtree_indeces=None):
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
    if kdtree_indeces is not None:
        n_pairs=kdtree_indeces.shape[0]
        for idx in range(n_pairs):
            i,j = kdtree_indeces[idx]
            res_i3 = i*3
            res_i33 = res_i3+3
            res_j3 = j*3
            res_j33 = res_j3+3
            i2j = coords[j] - coords[i]
            dist2 = np.dot(i2j, i2j)
            #TODO: write helper function so not in code twice
            super_element = np.outer(i2j, i2j) * -g / dist2
            #TODO unravel and return values, cols and rows
            #TODO: try to vectorize local transpose symetry
            hess[res_i3:res_i33, res_j3:res_j33] = super_element
            hess[res_j3:res_j33, res_i3:res_i33] = super_element
            # TODO: test vectorized summing up rows/cols outside of j loop
            hess[res_i3:res_i33, res_i3:res_i33] = \
                hess[res_i3:res_i33, res_i3:res_i33] - super_element
            hess[res_j3:res_j33, res_j3:res_j33] = \
                hess[res_j3:res_j33, res_j3:res_j33] - super_element


    else:
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


def hess_from_coords_coo_fixedarray(coords,cutoff=15.,gamma=1., sparsity=0.07):    
    '''
    sparse: encoding the hessian as sparse is very important since the final matrix will be sparse. suitable for very large matrices since the memory won't swamp with a large sparse matrix
    '''
    assert coords.shape[1] == 3, 'coords shape is not Nx3'
    dof = coords.size
    n_atoms = coords.shape[0]
    cutoff2 = cutoff * cutoff
    g = gamma # TODO: function for distance dependence / scalar map value
    upper_bound_size = int(9*dof*dof*sparsity)
    hess_data, hess_rows, hess_cols = np.empty(upper_bound_size), np.empty(upper_bound_size, dtype=np.int32), np.empty(upper_bound_size,dtype=np.int32)
    hess_diagonals=np.zeros((n_atoms,3,3))
    counter=0
    for i in range(n_atoms):
        res_i3 = i*3
        i_p1 = i+1
        i2j_all = coords[i_p1:, :] - coords[i]
        for j, dist2 in enumerate((i2j_all ** 2).sum(1)):
            if dist2 > cutoff2: # TODO: precompute with kdtree, fill in array instead of extent lists, since will know how large the lists are at the start
                continue
            i2j = i2j_all[j]
            j += i_p1
            res_j3 = j*3
#             xx,xy,xz = x*x,x*y,x*z
#             yy,yz = y*y,y*z
#             zz=z*z
#             super_element_list=[xx,xy,xz,xy,yy,yz,xz,yz,zz]
#             super_element = np.array(super_element_list).reshape(3,3)
            
            
            super_element=np.outer(i2j, i2j) * -g / dist2
            super_element_list = super_element.flatten() # TODO: faster if do explicitly, only 3x3 so can write out all 9 entries, but loose verctorization
            # write data, rows, cols
            hess_data[counter:counter+9] = super_element_list
            hess_rows[counter:counter+9] = [res_i3,res_i3,res_i3,res_i3+1,res_i3+1,res_i3+1,res_i3+2,res_i3+2,res_i3+2]
            hess_cols[counter:counter+9] = [res_j3,res_j3+1,res_j3+2,res_j3,res_j3+1,res_j3+2,res_j3,res_j3+1,res_j3+2]
            counter+=9
            #TODO: transpose this later?
            hess_data[counter:counter+9] = super_element_list
            hess_rows[counter:counter+9] = [res_j3,res_j3,res_j3,res_j3+1,res_j3+1,res_j3+1,res_j3+2,res_j3+2,res_j3+2]
            hess_cols[counter:counter+9] = [res_i3,res_i3+1,res_i3+2,res_i3,res_i3+1,res_i3+2,res_i3,res_i3+1,res_i3+2]

            hess_diagonals[i]-=super_element
            hess_diagonals[j]-=super_element
            counter+=9
    if hess_rows[:counter].max() < dof or hess_cols[:counter].max() < dof: 
        # if needed, put zeros explicitly on diag, so will get back full sparse matrix incase whole col/row are empty
        res_i3 = dof-3
        res_j3 = res_i3
        hess_data[counter:counter+9]=0 # will add to diagonal later, so keep zero
        hess_rows[counter:counter+9] = [res_i3,res_i3,res_i3,res_i3+1,res_i3+1,res_i3+1,res_i3+2,res_i3+2,res_i3+2]
        hess_cols[counter:counter+9] = [res_j3,res_j3+1,res_j3+2,res_j3,res_j3+1,res_j3+2,res_j3,res_j3+1,res_j3+2]
        counter+=9
        
    hess_data[:counter]

    return(hess_data[:counter], hess_rows[:counter], hess_cols[:counter],hess_diagonals)

#@jit(nopython=True)
def hess_from_coords_coo_kdtree(coords,gamma=1., sparsity=0.07,kdtree_indeces=None):    
    '''
    #cutoff: not used since pairs precomputed in kdtree_indeces. needed to use in wrapper with other functions that take this arg
    sparse: encoding the hessian as sparse is very important since the final matrix will be sparse. suitable for very large matrices since the memory won't swamp with a large sparse matrix
    '''
    assert coords.shape[1] == 3, 'coords shape is not Nx3'
    dof = coords.size
    n_atoms = coords.shape[0]
    #assert np.isclose(dof / 3.,n_atoms)
    g = gamma # TODO: function for distance dependence / scalar map value
    hess_diagonals=np.zeros((n_atoms,3,3))
    n_pairs = kdtree_indeces.shape[0]
    sparse_size = 9*(2*n_pairs+1)
    hess_data, hess_rows, hess_cols = np.empty(sparse_size), np.empty(sparse_size, dtype=np.int32), np.empty(sparse_size,dtype=np.int32)

    counter=0
    for idx in range(n_pairs):
        i,j = kdtree_indeces[idx]
        res_i3 = i*3
        res_j3 = j*3
        i2j = coords[j] - coords[i]
        dist2 = np.dot(i2j, i2j)
        super_element = np.outer(i2j, i2j) * -g / dist2
        super_element_list = super_element.flatten()
        
        # write data, rows, cols
        hess_data[counter:counter+9] = super_element_list
        hess_rows[counter:counter+9] = [res_i3,res_i3,res_i3,res_i3+1,res_i3+1,res_i3+1,res_i3+2,res_i3+2,res_i3+2]
        hess_cols[counter:counter+9] = [res_j3,res_j3+1,res_j3+2,res_j3,res_j3+1,res_j3+2,res_j3,res_j3+1,res_j3+2]
        counter+=9
        
        #TODO: transpose this later?
        hess_data[counter:counter+9] = super_element_list
        hess_rows[counter:counter+9] = [res_j3,res_j3,res_j3,res_j3+1,res_j3+1,res_j3+1,res_j3+2,res_j3+2,res_j3+2]
        hess_cols[counter:counter+9] = [res_i3,res_i3+1,res_i3+2,res_i3,res_i3+1,res_i3+2,res_i3,res_i3+1,res_i3+2]
        counter+=9
        
        hess_diagonals[i]-=super_element
        hess_diagonals[j]-=super_element
        
    hess_diagonals = hess_diagonals 
    
    # put zeros explicitly on diag, so will get back full sparse matrix incase whole col/row are empty
    res_i3 = dof-3
    res_j3 = res_i3
    hess_data[counter:counter+9]=0 # will add to diagonal later, so keep zero
    hess_rows[counter:counter+9] = [res_i3,res_i3,res_i3,res_i3+1,res_i3+1,res_i3+1,res_i3+2,res_i3+2,res_i3+2]
    hess_cols[counter:counter+9] = [res_j3,res_j3+1,res_j3+2,res_j3,res_j3+1,res_j3+2,res_j3,res_j3+1,res_j3+2]
    counter+=9
    assert counter == sparse_size
        

    return(hess_data, hess_rows, hess_cols,hess_diagonals)

def kdtree_idx_from_coords(coords,cutoff=15.,method='cKDTree'):
    if method == 'cKDTree':
        kdt = scipy.spatial.cKDTree(coords)
        kdtree_indeces = np.array(list(kdt.query_pairs(cutoff)))
    elif method == 'KDTree':
        kdt = scipy.spatial.KDTree(coords)
        kdtree_indeces = np.array(list(kdt.query_pairs(cutoff)))
    return(kdtree_indeces)

def hess_from_coords_csr(coords,method=None,**kwargs):
    if method == 'fixed':
        hess_data, hess_rows, hess_cols,hess_diagonals = jit(hess_from_coords_coo_fixedarray,nopython=True)(coords, **kwargs)
    elif method == 'kdtree':
        kdtree_indeces=kdtree_idx_from_coords(coords,method='cKDTree',**kwargs)
        hess_data, hess_rows, hess_cols,hess_diagonals = jit(hess_from_coords_coo_kdtree,nopython=True)(coords,kdtree_indeces=kdtree_indeces,**kwargs)
    else:
        hess_data, hess_rows, hess_cols,hess_diagonals = jit(hess_from_coords_coo_denseblockdiag,nopython=True)(coords,**kwargs)
    coo=scipy.sparse.coo_matrix((hess_data,(hess_rows,hess_cols)))
    bd = scipy.sparse.block_diag(hess_diagonals)
    hess_csr = bd + coo
    return(hess_csr)



