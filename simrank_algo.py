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

    num_parent = np.sum(adjacency_matrix, axis = 0)

    def find_parents() -> Dict[ int, Set[ int ] ]:

        nonlocal num_vertices, adjacency_matrix

        parents = dict()

        for e_vert in range(num_vertices):
            for s_vert in range(num_vertices):
                if (adjacency_matrix[s_vert][e_vert] == 1):
                    parents[e_vert] = parents.get(e_vert, set()).union({ s_vert })

        return parents
    
    parents = find_parents()

    def compute_simrank(vertex_a : int, vertex_b : int) -> float:

        nonlocal simrank, simrank_tmp, parents, num_parent, num_vertices, decay_factor, adjacency_matrix

        if (vertex_a == vertex_b):
            return 1

        num_parent_a = num_parent[vertex_a]

        num_parent_b = num_parent[vertex_b]

        if ((num_parent_a == 0) or (num_parent_b == 0)):
            return 0

        rescale_factor = decay_factor / num_parent_a / num_parent_b

        counter = 0

        # for parent_vert_a, parent_vert_b in itertools.product(range(num_vertices), repeat = 2):

        #     if (
        #         (adjacency_matrix[parent_vert_a][vertex_a] == 0) or 
        #         (adjacency_matrix[parent_vert_b][vertex_b] == 0)
        #     ):
        #         continue

        for parent_vert_a in parents[vertex_a]:
            for parent_vert_b in parents[vertex_b]:
                counter += simrank[parent_vert_a][parent_vert_b]

        return counter * rescale_factor

    for _ in range(max_iterations):

        simrank_tmp = np.zeros_like(simrank)

        for s_vertex, e_vertex in itertools.product(range(num_vertices), repeat = 2):

            simrank_tmp[s_vertex][e_vertex] = compute_simrank(s_vertex, e_vertex)

        simrank = simrank_tmp.copy()

    return simrank