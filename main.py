from utilities import edges_to_adjacency_matrix, load_edges

from pagerank_algo import pagerank_algorithm

from simrank_algo import simrank_algorithm

from hits_algo import hits_algorithm


# >> CONSTANTS 

max_iterations = 30

damping_factor = 0.10

decay_factor   = 0.90

epsilon        = 1e-8

# << CONSTANTS


if (__name__ == "__main__"):

    edges = load_edges("./data/graph_5.txt")

    (adjacency_matrix, mapping) = edges_to_adjacency_matrix(edges)

    # pagerank = pagerank_algorithm(
    #     adjacency_matrix, 
    #     damping_factor = damping_factor, 
    #     max_iterations = max_iterations,
    #     epsilon        = epsilon
    # )

    # print("PGR:",  __import__("numpy").round(pagerank, 3))

    # (authority, hubness) = hits_algorithm(
    #     adjacency_matrix, 
    #     max_iterations = max_iterations,
    #     epsilon        = epsilon
    # )

    # print("ATH:", __import__("numpy").round(authority, 3))

    # print("HUB:", __import__("numpy").round(hubness, 3))

    simrank = simrank_algorithm(
        adjacency_matrix, 
        decay_factor   = 0.60, 
        max_iterations = max_iterations
    )

    print("SIM:", simrank)