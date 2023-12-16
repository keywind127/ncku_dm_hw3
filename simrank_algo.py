import numpy as np

from typing import *

import itertools


def simrank_algorithm(adjacency_matrix : np.ndarray, 
                      decay_factor     : float, *, 
                      max_iterations   : Optional[ int ] = 3) -> np.ndarray:
    # initialize simrank matrix with zeros
    simrank = np.zeros_like(adjacency_matrix, dtype = np.float32)
    # obtaining number of vertices
    num_vertices = len(adjacency_matrix)
    # initialize identity matrix
    for vertex in range(num_vertices):
        simrank[vertex][vertex] = 1
    # pre-calculate number of parents for each vertex
    num_parent = np.sum(adjacency_matrix, axis = 0)
    def find_parents() -> Dict[ int, Set[ int ] ]:
        nonlocal num_vertices, adjacency_matrix
        parents = dict()
        # look for parent nodes in adjacency matrix
        for e_vert in range(num_vertices):
            for s_vert in range(num_vertices):
                if (adjacency_matrix[s_vert][e_vert] == 1):
                    parents[e_vert] = parents.get(e_vert, set()).union({ s_vert })
        return parents   
    # record all parent nodes for each node
    parents = find_parents()
    def compute_simrank(vertex_a : int, vertex_b : int) -> float:
        nonlocal simrank, simrank_tmp, parents, num_parent, num_vertices, decay_factor, adjacency_matrix
        # same vertex => 1
        if (vertex_a == vertex_b):
            return 1
        # number of parents of vertex a
        num_parent_a = num_parent[vertex_a]
        # number of parents of vertex b
        num_parent_b = num_parent[vertex_b]
        # product of both parent numbers equals to 0 => 0
        if ((num_parent_a == 0) or (num_parent_b == 0)):
            return 0
        # decay factor divided by product of both parent numbers
        rescale_factor = decay_factor / num_parent_a / num_parent_b
        counter = 0
        # aggregate simrank of all parent vertex combinations
        for parent_vert_a in parents[vertex_a]:
            for parent_vert_b in parents[vertex_b]:
                counter += simrank[parent_vert_a][parent_vert_b]
        # 
        return counter * rescale_factor
    for _ in range(max_iterations):
        # initialize simrank for next timestep (t+1) with zeros
        simrank_tmp = np.zeros_like(simrank)
        # calculate simrank value for each (a,b) pair
        for s_vertex, e_vertex in itertools.product(range(num_vertices), repeat = 2):
            simrank_tmp[s_vertex][e_vertex] = compute_simrank(s_vertex, e_vertex)
        # update simrank for next timestep
        simrank = simrank_tmp.copy()
    return simrank