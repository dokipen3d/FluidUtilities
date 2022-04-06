from types import DynamicClassAttribute
import matplotlib.pyplot as plt
import tensorly as tl
import numpy as np
import sys
import math
from numba import jit

from tensorly.decomposition import parafac

# fill width and not print e notation
np.set_printoptions(threshold=sys.maxsize, linewidth=100000, suppress=True)
np.core.arrayprint._line_width = 4000



# how wide? corresponds to number of jacobi iterations
filterWidth = 64

# padding
filterWidth += 2

filterWidthDiv2 = int((filterWidth)/2)
x0 = np.zeros((filterWidth,filterWidth,filterWidth))
x1 = np.zeros((filterWidth,filterWidth,filterWidth))
b = np.zeros((filterWidth,filterWidth,filterWidth))

b[filterWidthDiv2,filterWidthDiv2,filterWidthDiv2] = 1
#print(filter)

iters = filterWidthDiv2

@jit(nopython = True)
def loop(x0, x1, b):
    for iter in range(0,iters):
        for k in range(1, filterWidth-1):
            for j in range(1, filterWidth-1):
                for i in range(1, filterWidth-1):
                    x0[i,j,k] = (x1[i-1,j,k] + x1[i+1,j,k] + x1[i,j-1,k] + x1[i,j+1,k] + x1[i,j,k-1] + x1[i,j,k+1] + b[i,j,k])/6.0
        x0, x1 = x1, x0
    return x1

xFinal = loop(x0, x1, b) 



filterWidthDiv2F = math.ceil((filterWidth)/2)


####
# parafac_power vs symmetric
weights, factors = tl.decomposition.symmetric_parafac_power_iteration(xFinal, rank=2, n_repeat=10, n_iteration=10, verbose=False)
print(factors[filterWidthDiv2F:filterWidthDiv2F+16].transpose())
print(weights)
