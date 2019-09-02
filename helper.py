import numpy as np

# TODO: error in no values going to middie (coordinates with zero, rounding error?)
def coord_to_index(coord,offset,dx):
    index = np.floor((coord - offset)/dx).astype(np.int32)
    return(index)


def coords_to_map(coords,vals,offset,dx,N):
    '''
    offset: side/2
    dx: pixel step size. side/float(int(side / (side/nsamples)))
    N: number of pixels along an axis.
    '''
    coarse_map = np.zeros((N,N,N))
    for coord,val in zip(coords,vals):
        x,y,z = coord_to_index(coord,offset=offset,dx=dx).tolist()
        try:
        	coarse_map[x,y,z] = val
        except Exception as e:
        	print(e,coord) # for when coord on edge
    return(coarse_map)