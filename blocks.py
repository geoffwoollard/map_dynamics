# based on https://github.com/prody/ProDy/blob/e66e29f8d5368d40ce51b89eb05fa3d91fd18f77/prody/dynamics/rtb.py

import numpy as np

class Increment(object):

    def __init__(self, s=0):

        self._i = s

    def __call__(self, i=1):

        self._i += i
        return self._i


def calc_projection_setup(coords, blks, natoms):
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

    # hessian = self._hessian
    # TODO: remove hessian into another function
    project = np.zeros((natoms * 3, nb6), float)

    return(coords, blks, project, natoms, nblocks, nb6, maxsize)

def do_projection(hessian,coords, blks, project, natoms, nblocks, nb6, maxsize):

    from blocksmodule import calc_projection # TODO: from .blocksmodule

    calc_projection(coords, blks, project, natoms, nblocks, nb6, maxsize)

    hessian_block = project.T.dot(hessian).dot(project)
    return(hessian_block,project)
