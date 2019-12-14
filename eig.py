import pycuda.gpuarray as gpuarray
import numpy as np
import scipy

def gpu_setup():
    import pycuda.autoinit
    from skcuda import linalg
    linalg.init() # TODO go inside function?

def hess_to_eig_gpu(hess,nmodes=20, shift=6):
    assert max(hess.shape) < 50000, 'RAM maxes out ~ 20000'
    hess = np.array(hess, np.float32, order='F')
    hess_gpu = gpuarray.to_gpu(hess)
    vec_gpu, val_gpu = linalg.eig(hess_gpu, 'N', 'V')
    vec,val = vec_gpu.get().T, val_gpu.get()
    vec = vec[:,shift:shift+nmodes]
    val = val[shift:shift+nmodes]
    return(vec,val)

def blk_hess_to_eig(hess,project,nmodes=20,shift=6):
    scipy.linalg.eigh
    vals,blk_vecs = scipy.linalg.eigh(
        hess,
        eigvals=(0,nmodes+shift-1)
        )
    vecs = np.dot(project,blk_vecs[:,shift:])
    vals=vals[shift:]
    return(vecs,vals)

def sparse_hess_to_eig(hess_s,nmodes=20,shift=6,which='SM'):
    vals, vecs = scipy.sparse.linalg.eigsh(hess_s,k=nmodes+shift,which=which) # because h_blk_s symmetric
    # TODO: remove zeros once figure out if before or after
    return(vals[shift:], vecs[:,shift:])