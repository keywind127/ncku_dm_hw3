import numpy as np

from typing import *

def hits_algorithm(adjacency_matrix : np.ndarray, *,
                   epsilon          : Optional[ float ] = 0.01,
                   max_iterations   : Optional[ int   ] =   -1) -> Tuple[ np.ndarray, np.ndarray ]:
    # obtaining number of vertices
    num_vertices = len(adjacency_matrix)
    # initialize hub vector with ones
    hubness = np.ones(shape = (num_vertices, ), dtype = np.float32)
    # initialize authority vector with ones
    authority = np.ones(shape = (num_vertices, ), dtype = np.float32)
    # execute for `max_iterations` number of time steps
    while (max_iterations != 0):
        # initialize hub (t+1)
        hubness_tmp = np.zeros_like(hubness)
        # initailize authority (t+1)
        authority_tmp = np.zeros_like(authority)
        # iterate through each vertex
        for vertex in range(num_vertices):
            # calculate vertex hub score with children authority scores
            hubness_tmp[vertex] += np.matmul(authority, adjacency_matrix[vertex].T)
            # calculate vertex authority score with parent hub scores
            authority_tmp[vertex] += np.matmul(hubness, adjacency_matrix[..., vertex].T)
        # normalize authority 
        authority_tmp = authority_tmp / np.sum(authority_tmp)
        # normalize hub 
        hubness_tmp = hubness_tmp / np.sum(hubness_tmp)
        # calculate delta, hub & authority difference between timesteps
        delta = (
            np.sum(np.sqrt(np.square(hubness_tmp   -   hubness))) + 
            np.sum(np.sqrt(np.square(authority_tmp - authority)))
        )
        # update authority for next timestep
        authority = authority_tmp
        # update hub for next timestep
        hubness = hubness_tmp
        # perform early stopping on minuscule change 
        if (delta < epsilon):
            break
        # decrement maximum number of remaining timesteps
        max_iterations -= 1
    # return tuple of authority and hub scores
    return (authority, hubness)

if (__name__ == "__main__"):

    edges = [
        [ 1, 2 ], [ 2, 3 ], [ 3, 4 ], [ 4, 5 ], [ 5, 6 ]
    ]

    (adjacency_matrix, mapping) = __import__("utilities").edges_to_adjacency_matrix(edges)

    authority, hubness = hits_algorithm(adjacency_matrix, max_iterations = 2)

    print("ATH:", authority)

    print("HUB:", hubness)