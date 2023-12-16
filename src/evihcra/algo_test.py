from utilities import edges_to_adjacency_matrix, load_edges

from pagerank_algo import pagerank_algorithm

from simrank_algo import simrank_algorithm

from hits_algo import hits_algorithm

import numpy as np

import os


# >> CONSTANTS 

max_iterations = 30

damping_factor = 0.10

decay_factor   = 0.70

epsilon        = 1e-8

# << CONSTANTS

if (__name__ == "__main__"):

    filename = "data/graph_1.txt"

    edges = load_edges(filename, csv_format = True)

    adj_mat, mapping = edges_to_adjacency_matrix(edges)

    pagerank = pagerank_algorithm(adj_mat, damping_factor, epsilon = epsilon, max_iterations = max_iterations)

    print(f"PR (Before): {pagerank}")

    # (auth, hub) = hits_algorithm(adj_mat, epsilon = epsilon, max_iterations = max_iterations)

    # print("ATH:", np.round(auth, 3))

    # print("HUB:", np.round(hub, 3))

    filename = "data/graph_3_mod.txt"

    edges = load_edges(filename, csv_format = True)

    adj_mat, mapping = edges_to_adjacency_matrix(edges)

    pagerank = pagerank_algorithm(adj_mat, damping_factor, epsilon = epsilon, max_iterations = max_iterations)

    print(f"PR  (After): {pagerank}")

    # (auth, hub) = hits_algorithm(adj_mat, epsilon = epsilon, max_iterations = max_iterations)

    # print("ATH:", np.round(auth, 3))

    # print("HUB:", np.round(hub, 3))

    # increase hub: add more outgoing connections
    
    # increase ath: add more incoming connections