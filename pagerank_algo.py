
import numpy as np

from typing import *

def pagerank_algorithm(adjacency_matrix : np.ndarray, 
                       damping_factor   : float, *,
                       epsilon          : Optional[ float ] = 0.01,
                       max_iterations   : Optional[ int   ] =   -1) -> np.ndarray:

    num_vertices = len(adjacency_matrix)

    pagerank = np.ones(shape = (num_vertices, ), dtype = np.float32) / num_vertices

    num_children = np.sum(adjacency_matrix, axis = 1)

    num_children = np.maximum(num_children, np.ones_like(num_children))

    while (max_iterations != 0):

        pagerank_tmp = np.zeros_like(pagerank)

        for vertex in range(num_vertices):

            pagerank_tmp[vertex] += damping_factor / num_vertices + (1 - damping_factor) * np.sum(pagerank / num_children * adjacency_matrix[..., vertex])

        delta = np.sum(np.sqrt(np.square(pagerank - pagerank_tmp)))

        pagerank = pagerank_tmp

        if (delta < epsilon):
            break

        max_iterations -= 1

    pagerank = pagerank / np.sum(pagerank)

    return pagerank