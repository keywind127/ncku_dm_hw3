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

    simrank_exception = {
        "graph_6", "ibm-5000"
    }

    input_data_graph = [
        "data/graph_1.txt",
        "data/graph_2.txt",
        "data/graph_3.txt",
        "data/graph_4.txt",
        "data/graph_5.txt",
        "data/graph_6.txt"
    ]

    input_data_ibm = "data/ibm-5000.txt"

    graph_names = [ input_data_ibm ] + input_data_graph

    edges_list = [ load_edges(input_data_ibm, csv_format = False), *[
        load_edges(input_data_g, csv_format = True) for input_data_g in input_data_graph
    ]]

    output_folder = "results"

    os.makedirs(output_folder, exist_ok = True)

    for idx, edges in enumerate(edges_list):

        filename = graph_names[idx]

        basename = os.path.basename(os.path.splitext(filename)[0])

        current_folder = os.path.join(output_folder, basename)

        os.makedirs(current_folder, exist_ok = True)

        (adjacency_matrix, mapping) = edges_to_adjacency_matrix(edges)

        pagerank = pagerank_algorithm(
            adjacency_matrix, 
            damping_factor = damping_factor, 
            max_iterations = max_iterations,
            epsilon        = epsilon
        )

        print("PGR:",  __import__("numpy").round(pagerank, 3))

        print("Num Nodes:", pagerank.__len__())

        np.savetxt(f"{current_folder}/{basename}_PageRank.txt", pagerank, fmt = "%1.3f")

        (authority, hubness) = hits_algorithm(
            adjacency_matrix, 
            max_iterations = max_iterations,
            epsilon        = epsilon
        )

        np.savetxt(f"{current_folder}/{basename}_HITS_authority.txt", authority, fmt = "%1.3f")

        np.savetxt(f"{current_folder}/{basename}_HITS_hub.txt", hubness, fmt = "%1.3f")

        print("ATH:", __import__("numpy").round(authority, 3))

        print("HUB:", __import__("numpy").round(hubness, 3))

        print("Num Nodes:", authority.__len__(), hubness.__len__())

        if (basename in simrank_exception):
            continue

        simrank = simrank_algorithm(
            adjacency_matrix, 
            decay_factor   = decay_factor, 
            max_iterations = max_iterations
        )

        np.savetxt(f"{current_folder}/{basename}_SimRank.txt", simrank, fmt = "%1.3f")

        print("SIM:", simrank)

        simrank = np.round(simrank, 3)

        print(np.sum(simrank == 0.021))

        print(adjacency_matrix.__len__())