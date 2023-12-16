from utilities import edges_to_adjacency_matrix, load_edges

from pagerank_algo import pagerank_algorithm

from simrank_algo import simrank_algorithm

from hits_algo import hits_algorithm

from utils import StopWatch

import numpy as np

import os

from matplotlib import pyplot as plt

# >> CONSTANTS 

max_iterations = 30

# damping_factor = 0.99 # 0.10

# decay_factor   = 0.70

epsilon        = 1e-8

# << CONSTANTS

if (__name__ == "__main__"):

    graph_label = "2"

    filename = f"data/graph_{graph_label}.txt"

    thresholds = [ 0.01, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99 ] #[ 0.01, 0.50, 0.99 ] # [ 0.01, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99 ]

    x_label_name = "Damping Factor"

    y_label_name = "PageRank Value"

    graph_title = f"DF-PR (Graph-{graph_label})"




    edges = load_edges(filename, csv_format = True)

    adj_mat, mapping = edges_to_adjacency_matrix(edges)

    node_names = [  f"node{idx}" for idx in range(1, len(adj_mat) + 1)  ]

    result = []

    for damping_factor in thresholds:

        pagerank = pagerank_algorithm(adj_mat, damping_factor, epsilon = epsilon, max_iterations = max_iterations)

        # pagerank = simrank_algorithm(adj_mat, damping_factor, max_iterations = max_iterations)

        result.append(pagerank)

        print(f"SR (DF={damping_factor:.2f}): \n{np.round(pagerank, 4)}")

    result = np.stack(result)

    # print(result)

    columns = result.shape[1]

    linestyles = [
        "--", "-.", ":"
    ]

    markers = [
        "+", "o", "x"
    ]

    for col in range(columns):
        plt.plot(result[:, col], marker = markers[col % len(markers)], linestyle = linestyles[col % len(linestyles)])

    plt.xticks(range(len(thresholds)), thresholds)

    plt.xlabel(x_label_name)

    plt.ylabel(y_label_name)

    plt.title(graph_title)

    plt.legend(node_names)

    plt.savefig(os.path.join(os.path.dirname(__file__), "plot_imgs", f"{graph_title}.png"))

    plt.show()