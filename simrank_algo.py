import numpy as np

from typing import *

import itertools


def simrank_algorithm(adjacency_matrix : np.ndarray, 
                      decay_factor     : float, *, 
                      max_iterations   : Optional[ int ] = 3) -> np.ndarray:

    simrank = np.zeros_like(adjacency_matrix, dtype = np.float32)

    num_vertices = len(adjacency_matrix)

    for vertex in range(num_vertices):
        simrank[vertex][vertex] = 1

    num_parent = np.sum(adjacency_matrix, axis = 1)

    def compute_simrank(vertex_a : int, vertex_b : int) -> float:

        nonlocal simrank, simrank_tmp, num_parent, num_vertices, decay_factor, adjacency_matrix

        if (vertex_a == vertex_b):
            return 1

        num_parent_a = num_parent[vertex_a]

        num_parent_b = num_parent[vertex_b]

        if ((num_parent_a == 0) or (num_parent_b == 0)):
            return 0

        rescale_factor = decay_factor / num_parent_a / num_parent_b

        counter = 0

        for parent_vert_a, parent_vert_b in itertools.product(range(num_vertices), repeat = 2):

            if (
                (adjacency_matrix[parent_vert_a][vertex_a] == 0) or 
                (adjacency_matrix[parent_vert_b][vertex_b] == 0)
            ):
                continue

            counter += simrank[parent_vert_a][parent_vert_b]

        return counter * rescale_factor

    for _ in range(max_iterations):

        simrank_tmp = np.zeros_like(simrank)

        for s_vertex, e_vertex in itertools.product(range(num_vertices), repeat = 2):

            simrank_tmp[s_vertex][e_vertex] = compute_simrank(s_vertex, e_vertex)

        simrank = simrank_tmp.copy()

    return simrank