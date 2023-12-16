
import numpy as np

from typing import *

def pagerank_algorithm(adjacency_matrix : np.ndarray, 
                       damping_factor   : float, *,
                       epsilon          : Optional[ float ] = 0.01,
                       max_iterations   : Optional[ int   ] =   -1) -> np.ndarray:
    # obtaining number of vertices in graph
    num_vertices = len(adjacency_matrix)
    # initialize pagerank vector with ones, then normalize it (summation equals to 1)
    pagerank = np.ones(shape = (num_vertices, ), dtype = np.float32) / num_vertices
    # store number of children nodes (outgoing links) for each node
    num_children = np.sum(adjacency_matrix, axis = 1)
    # force min-number of children to 1 => for matrix ops, does not affect result because of mask
    num_children = np.maximum(num_children, np.ones_like(num_children))
    # execute algorithm for maximum number of timesteps
    while (max_iterations != 0):
        # initialize pagerank vector at new timestep (t+1) with zeros
        pagerank_tmp = np.zeros_like(pagerank)
        # iterate through each vertex
        for vertex in range(num_vertices):
            # calculate pagerank value based on random-surfer pagerank equation
            pagerank_tmp[vertex] += (
                damping_factor / num_vertices + 
                (1 - damping_factor) * np.sum(pagerank / num_children * adjacency_matrix[..., vertex])
            )
        # calculate value difference between timesteps
        delta = np.sum(np.sqrt(np.square(pagerank - pagerank_tmp)))
        # update pagerank for next timestep
        pagerank = pagerank_tmp
        # early stopping on minuscule change
        if (delta < epsilon):
            break    
        # decrement maximum number of remaining iterations
        max_iterations -= 1
    # normalize pagerank values
    pagerank = pagerank / np.sum(pagerank)
    return pagerank