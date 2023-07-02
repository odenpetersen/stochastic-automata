#!/usr/bin/env python3
from automata import evolve
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create a random 10x10 matrix
size = 300
matrix = np.random.binomial(n=1, p=0.3, size=size*size).reshape((size,size))

lookahead = 5
matrices = [matrix]
for _ in range(lookahead):
    matrix = evolve(matrix)
    matrices.append(matrix)

# Create a figure and axis
fig, ax = plt.subplots()

# Create an empty imshow plot
im = ax.imshow(matrix, cmap='hot', animated=True)

decay = 0.9
weights = decay**np.array(list(range(lookahead+1)))
weights /= sum(weights)
# Update function for the animation
def update(frame):
    global matrix,matrices
    # Update the matrix values with new random values
    matrix = evolve(matrix)
    matrices = matrices[1:]
    matrices.append(matrix)
    
    # Update the imshow plot with the new matrix
    matrices_array = np.array(matrices)
    smoothed = np.tensordot(weights, matrices_array, axes=(0,0))
    im.set_array(smoothed)

    matrix = matrices[-1]
    
    # Return the updated artists
    return im,

# Create the animation
animation = FuncAnimation(fig, update, interval=100)

# Show the animation
plt.show()

