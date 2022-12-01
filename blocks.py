# based on https://github.com/prody/ProDy/blob/e66e29f8d5368d40ce51b89eb05fa3d91fd18f77/prody/dynamics/rtb.py

import numpy as np
import scipy
import scipy.sparse
# TODO: cupy functions


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
        array (3*natoms x nb6) project: 
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

def calc_projection_py(coords, blks, project, natoms, nblocks, nb6, maxsize,method='csr'):
    '''
    blocksmodule:calc_projection implemented in python
    # TODO optimize by 
        vectorizing more
        benchmarking svd vs eigh
        sparse output 
        hsize check redundant
        IC no need for copy
        HH vs PP same thing
    '''
    bmx = maxsize
    bdim = nb6
    nblx = nblocks
    natm = natoms

    if 18*bmx*nblx > 12*natm:
        hsize=12*natm
    else:
        hsize=18*bmx*nblx

    hh_idx = np.zeros((hsize,2))
    hh_x = np.zeros(hsize)
    pp_idx = np.zeros_like(hh_idx,dtype=np.int32)
    pp_x = np.zeros_like(hh_x,dtype=np.float64)
    elm,pp_idx,pp_x = dblock_projections2_py(coords,blks,pp_idx,pp_x,natm,nblx,bmx)
    project = copy_prj_ofst_py(project,elm,pp_idx,pp_x,bdim,method=method)
    return(project)
 

def dblock_projections2_py(coords,blks,pp_idx,pp_x,natm,nblx,bmx):
    nres=natm
    #/* INITIALIZE BLOCK ARRAYS */
    elm = 0;
    #X = np.zeros((bmx,3))# dmatrix(1, bmx, 1, 3);
    IDX = np.zeros(bmx,dtype=np.int32)#ivector(1, bmx);
    #CM = np.zeros(3)#dvector(1, 3);
    #I = np.zeros((3,3))#dmatrix(1, 3, 1, 3);
    #W = np.zeros(3)#dvector(1, 3);
    #A = np.zeros((3,3))#dmatrix(1, 3, 1, 3);
    ISQT = np.zeros((3,3))#dmatrix(1, 3, 1, 3);

    # cycle through blocks
    # TODO: verify works for blocks of size 1 (zero entries, averages, etc)
    for b in range(1,nblx+1):
        # clear matrices
        CM = np.zeros(3)
        I = np.zeros((3,3))
        X = np.zeros((bmx,3))
        
        # store values for current block
        nbp=0
        for i in range(nres):
            if blks[i] == b:
                nbp+=1
                IDX[nbp-1]=i 
                X[nbp-1]=coords[:,i]
                CM = CM+coords[:,i]
        assert nbp != 2, 'need 3+ points for a rigid body. eigen decomposition fails for only 2 points'
        
        # translate block centre of mass to origin
        CM = CM/nbp
        X=X-CM
        
        if nbp > 1: # condition on nbp > 1 because do not use I, W, A, ISQT for blocks of one
            # calculate inertia tensor
            for k in range(nbp):
                dd=0
        #         df = X[k]
                dd=(X[k]*X[k]).sum() # or np.linalg.norm(X[k])**2
                for i in range(3):
                    I[i,i] +=  dd - X[k,i]*X[k,i]
                    for j in range(i+1,3):
                        I[i,j] -= X[k,i]*X[k,j]
                        I[j,i] = I[i,j]
            
            # diagonalize inertia tensor
            # blocksmodule : dsvd (a, w, v) --> (U W Vt) in http://numerical.recipes/webnotes/nr3web2.pdf --> (u,s,vh) in np.linalg.svd
        #     u, s, vh = np.linalg.svd(IC, full_matrices=False)#,hermitian=True) # s sorted descending
        #     W = s 
        #     A = vh.T # transpose corresponds to column eigenvectors in scipy.linalg.eigh

            W,A = scipy.linalg.eigh(I) # same as svd, check which faster on 3x3 matrices
            cp = np.cross(A[:,0],A[:,1]) 
            if np.dot(cp,A[:,2]) < 0:
                A[:,2] = - A[:,2]
            
            # find its square root
            for i in range(3):
                for j in range(3):
                    dd=0
                    for k in range(3):
                        dd+=A[i,k]*A[j,k]/np.sqrt(W[k]) # divide by zero!
                    ISQT[i,j]=dd

        # update pp with the rigid motions of the block
        tr=nbp**-.5
        
        for i in range(nbp):
            
            # /* TRANSLATIONS: 3*(IDX[i]-1)+1 = x-COORDINATE OF RESIDUE IDX[i]; 6*(b-1)+1 = x-COORDINATE OF BLOCK b */
            # ie x,y,z coord have j=0,1,2
            for j in range(3):
                pp_idx[elm,0] = 3*(IDX[i])+j # IDX contains indeces that start at 0
                pp_idx[elm,1] = 6*(b-1)+j # blocks start at 1 
                pp_x[elm] = tr
                elm+=1

            # rotations
            # ie 3x3 rotation in project for residue i is cross product of ISQT and X (np.cross(ISQT,X[i])) TODO: try to vectorize
            if nbp > 1: # WARNING: bad values for nbp=2 because near zero eigenvector and blows up when divide by it
                for ii in range(3): # ii is column in project
                    for jj in range(3): ## jj is row in project
                        if jj == 0:
                            aa=2
                            bb=3
                        elif jj==1:
                            aa=3
                            bb=1
                        else:
                            aa=1
                            bb=2
                        dd = ISQT[ii][aa-1]*X[i][bb-1] - ISQT[ii][bb-1]*X[i][aa-1] # classic cross product indeces
                        pp_idx[elm,0] = 3*(IDX[i])+jj # row
                        pp_idx[elm,1] = 6*(b-1)+3+ii # column
                        pp_x[elm] = dd
                        elm += 1
                        # pp_idx[elm-1] is the last item in pp_idx[elm-1], zeros from pp_idx[elm] onwards
    return(elm,pp_idx[:elm],pp_x[:elm])
        
def copy_prj_ofst_py(project,elm,pp_idx,pp_x,bdim,method):
    '''
    project: empty 2d array
    '''
    mx = pp_idx[:,1].max()+1
    I1 = np.zeros(mx,np.int32)
    I2 = np.zeros(mx,np.int32)
    for i in range(elm):
        I1[pp_idx[i,1]] = pp_idx[i,1]+1 # have to start indexing at one bc check for zero in code right after. subtract one later once transformed
    j=0
    for i in range(mx):
        if I1[i] != 0: j+= 1
        I2[i]=j
    I2 -= 1 # correct from before
    if method == 'reshape':
        assert False #TODO: problem with zero values not being zero. atol 1e-2 not 1e-3 (not good enough)
        project_1d = project.flatten()
        for i in range(elm):
             if pp_x[i] != 0:
                    project_1d[bdim*pp_idx[i,0] + I2[pp_idx[i,1]]] = pp_x[i] # no minus one since zero indexed
        project = project_1d.reshape(project.shape)
    elif method == 'csr':
        proj_csr = scipy.sparse.csr_matrix((pp_x, (pp_idx[:,0], I2[pp_idx[:,1]])))
        project = proj_csr
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


