import numpy as np
import scipy as sp
import scipy.signal
# Game of Life Transitions, for reference
#---------------------------------------------
#         | Neighbours =                      |
#---------------------------------------------|
# Current | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
#---------------------------------------------|
#    1    | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 
#    0    | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
#---------------------------------------------

def transition(state, neighbour_density, alpha=2.5):
    #return (num_neighbours==3) or (state==1 and num_neighbours==2)
    probability = np.exp( -alpha * (8*neighbour_density - 3 + 0.5*state)**2 / (1 + 3*state))
    return np.random.binomial(n=1,p=probability)

def gaussian_kernel(n=5, sd=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(n-1)/2., (n-1)/2., n)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sd))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

#neighbour_kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])/8
neighbour_kernel = gaussian_kernel()
neighbour_kernel[2][2] = 0
neighbour_kernel /= np.sum(neighbour_kernel)
def evolve(matrix):
    neighbour_counts = sp.signal.convolve2d(matrix, neighbour_kernel, mode = 'same', boundary = 'wrap')
    pairs = zip(matrix.flatten(), neighbour_counts.flatten())
    return np.array(list(map(lambda t : transition(*t), pairs))).reshape(matrix.shape)
