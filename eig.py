import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import numpy as np
from skcuda import linalg

linalg.init() # TODO go inside function?

def hess_to_eig(hess,nmodes=None, shift=None):
  #TODO add nmodes and shift
  assert max(hess.shape) < 20000
  hess = np.array(hess, np.float32, order='F')
  hess_gpu = gpuarray.to_gpu(hess)
  vr_gpu, w_gpu = linalg.eig(hess_gpu, 'N', 'V')
  return(vr_gpu.get().T, w_gpu.get())