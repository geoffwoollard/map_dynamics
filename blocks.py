# based on https://github.com/prody/ProDy/blob/e66e29f8d5368d40ce51b89eb05fa3d91fd18f77/prody/dynamics/rtb.py

import numpy as np
import scipy
import scipy.sparse

class Increment(object):

    def __init__(self, s=0):

        self._i = s

    def __call__(self, i=1):

        self._i += i
        return self._i


def calc_projection_setup(coords, blks, natoms):
    '''
    array (natoms x 3 ) coords: xyz coordinates
    array blks: block indeces, with arbitrary integers
    int natoms: number of atoms

    return 
        array (3 x n) coords: transpose of coords
        array blks: array of block indeces, starting at 1 and ascending by 1
        array (n x nb6) project: 
        int natoms
        int nblocks: number of blocks
        int nb6: size of projection matrix. calculated in this function. six times number of blocks, minus the blocks of size 1
        int maxsize: the size of the longest block
    '''

    #natoms = self._n_atoms

    if natoms != len(blks):
        raise ValueError('len(blocks) must match number of atoms')

    #LOGGER.timeit('_rtb')
    from collections import defaultdict
    i = Increment()
    d = defaultdict(i)
    blks = np.array([d[b] for b in blks], dtype='int32')

    try:
        from collections import Counter
    except ImportError:
        counter = defaultdict(int)
        for b in blks:
            counter[b] += 1
    else:
        counter = Counter(blks)

    nblocks = len(counter)
    maxsize = 1
    nones = 0
    while counter:
        _, size = counter.popitem()
        if size == 1:
            nones += 1
        if size > maxsize:
            maxsize = size
    print('System has {0} blocks largest with {1} of {2} units.'
                .format(nblocks, maxsize, natoms))
    nb6 = nblocks * 6 - nones * 3

    coords = coords.T.astype(float, order='C')

    project = np.zeros((natoms * 3, nb6), float)

    return(coords, blks, project, natoms, nblocks, nb6, maxsize)

def get_projection(coords, blks, project, natoms, nblocks, nb6, maxsize):
    from blocksmodule import calc_projection # TODO: from .blocksmodule
    calc_projection(coords, blks, project, natoms, nblocks, nb6, maxsize) # returns project filled
    return(project)

def nan_to_zero(arr):
    nans = np.isnan(arr)
    nans_sum=nans.sum()
    if nans_sum > 0:
        arr[nans] = 0
        print('set %i nans in projection matrix to zero'%nans_sum)
    return(arr)

def do_projection(hessian,coords, blks, project, natoms, nblocks, nb6, maxsize):
    '''
    TODO: debug
    '''
    project = get_projection(coords, blks, project, natoms, nblocks, nb6, maxsize)
    project = nan_to_zero(project)
    hessian_block = np.linalg.multi_dot([project.T,hessian,project])
    return(hessian_block,project)

def make_sparse(arr,sparse_encoding=scipy.sparse.csr_matrix):
    '''func scipy.sparse.lil_matrix etc
    '''
    return(sparse_encoding(arr))

def sparse_dot(hessian_anm_s,project_s):
    '''ms instead of 2min for 27k x 27k with 0.5 % filled hessian_anm'''
    hessian_blk_s = project_s.T @ hessian_anm_s @ project_s
    return(hessian_blk_s)

def do_sparse_dot_wrapper(hessian_anm,project,**kwargs):
    hessian_anm_s = make_sparse(hessian_anm,**kwargs)
    project_s = make_sparse(project,**kwargs)
    hessian_blk_s = sparse_dot(hessian_anm_s,project_s)
    hessian_blk_s_d = hessian_blk_s.dense()
    return(hessian_blk_s_d)


