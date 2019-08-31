import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import numpy as np
from skcuda import linalg

linalg.init() # TODO go inside function?

def hess_to_eig(hess,nmodes=20, shift=6):
  #TODO add nmodes and shift
  assert max(hess.shape) < 20000
  hess = np.array(hess, np.float32, order='F')
  hess_gpu = gpuarray.to_gpu(hess)
  vec_gpu, val_gpu = linalg.eig(hess_gpu, 'N', 'V')
  vec,val = vec_gpu.get().T, val_gpu.get()
  vec = vec[:,shift:shift+nmodes]
  val = val[shift:shift+nmodes]
  return(vec,val)