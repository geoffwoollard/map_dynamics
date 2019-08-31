import numpy as np

# TODO: error in no values going to middie (coordinates with zero, rounding error?)
def coord_to_index(coord,offset,dx):
    index = np.rint((coord - offset)/dx).astype(int)
    return(index)


def coords_to_map(coords,vals,offset = -128,dx=2,N=128):
    coarse_map = np.zeros((N,N,N))
    for coord,val in zip(coords,vals):
        x,y,z = coord_to_index(coord,offset=offset,dx=dx).tolist()
        try:
        	coarse_map[x,y,z] = val
        except Exception as e:
        	print(e,coord) # for when coord on edge
    return(coarse_map)