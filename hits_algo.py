import numpy as np

from typing import *

def hits_algorithm(adjacency_matrix : np.ndarray, *,
                   epsilon          : Optional[ float ] = 0.01,
                   max_iterations   : Optional[ int   ] =   -1) -> Tuple[ np.ndarray, np.ndarray ]:

    num_vertices = len(adjacency_matrix)

    hubness = np.ones(shape = (num_vertices, ), dtype = np.float32)

    authority = np.ones(shape = (num_vertices, ), dtype = np.float32)

    while (max_iterations != 0):

        hubness_tmp = np.zeros_like(hubness)

        authority_tmp = np.zeros_like(authority)

        for vertex in range(num_vertices):

            hubness_tmp[vertex] += np.matmul(authority, adjacency_matrix[vertex].T)

            authority_tmp[vertex] += np.matmul(hubness, adjacency_matrix[..., vertex].T)

        authority_tmp = authority_tmp / np.sum(authority_tmp)

        hubness_tmp = hubness_tmp / np.sum(hubness_tmp)

        delta = (
            np.sum(np.sqrt(np.square(hubness_tmp   -   hubness))) + 
            np.sum(np.sqrt(np.square(authority_tmp - authority)))
        )

        authority = authority_tmp

        hubness = hubness_tmp

        if (delta < epsilon):
            break

        max_iterations -= 1

    return (authority, hubness)

if (__name__ == "__main__"):

    edges = [
        [ 1, 2 ], [ 2, 3 ], [ 3, 4 ], [ 4, 5 ], [ 5, 6 ]
    ]

    (adjacency_matrix, mapping) = __import__("utilities").edges_to_adjacency_matrix(edges)

    authority, hubness = hits_algorithm(adjacency_matrix, max_iterations = 2)

    print("ATH:", authority)

    print("HUB:", hubness)