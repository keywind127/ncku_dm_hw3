import numpy as np

from typing import *

def edges_to_adjacency_matrix(edges : List[ List[ int ] ]) -> Tuple[ np.ndarray, Dict[ int, int ] ]:

    assert isinstance(edges, list)

    vertices = np.unique(edges).astype(np.int32)

    mapping = {  vert : indx for indx, vert in enumerate(vertices)  }

    num_vertices = len(vertices)

    adjacency_matrix = np.zeros(shape = (num_vertices, num_vertices), dtype = np.int32)

    for s_vert, e_vert in edges:
        adjacency_matrix[mapping[s_vert]][mapping[e_vert]] = 1

    return (adjacency_matrix, mapping)

def load_edges(filename : str) -> List[ List[ int ] ]:

    assert isinstance(filename, str)

    return [
        list(map(int, edge_str.replace(" ", "").split(","))) for edge_str in 
            filter("".__ne__, open(filename, mode = "r", encoding = "utf-8").readlines())
    ]

def bidirectional_adjacency_matrix(adjacency_matrix : np.ndarray) -> np.ndarray:

    return adjacency_matrix | adjacency_matrix.T

if (__name__ == "__main__"):

    edges = load_edges("./test_data/graph1.txt")

    print(edges)

    adj_mat, mapping = edges_to_adjacency_matrix(edges)

    print(adj_mat)

    bid_adj_mat = bidirectional_adjacency_matrix(adj_mat)

    print(bid_adj_mat)