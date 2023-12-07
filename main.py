from utilities import edges_to_adjacency_matrix, load_edges

from pagerank_algo import pagerank_algorithm

from hits_algo import hits_algorithm

# >> CONSTANTS 

max_iterations = 30

damping_factor = 0.15

epsilon        = 1e-8

# << CONSTANTS


if (__name__ == "__main__"):

    edges = load_edges("./data/graph_3.txt")

    (adjacency_matrix, mapping) = edges_to_adjacency_matrix(edges)

    pagerank = pagerank_algorithm(
        adjacency_matrix, 
        damping_factor = damping_factor, 
        max_iterations = max_iterations,
        epsilon        = epsilon
    )

    print("PGR:", pagerank)

    (authority, hubness) = hits_algorithm(
        adjacency_matrix, 
        max_iterations = max_iterations,
        epsilon        = epsilon
    )

    print("ATH:", authority)

    print("HUB:", hubness)