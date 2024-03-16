import matplotlib.pyplot as plt
import time
import networkx as nx
import numpy as np
from algos.Dijkstra import dijkstra_for_adjacency_matrix
from algos.Bellman import BellmanFord
from algos.FloydWarshall import floydWarshall_adjacency_matrix
from algos.johnson import JohnsonAlgorithm
import random

if __name__ == "__main__":
    nodes_range = range(10, 1000, 10)
    dijkstra_execution_times_adjacency_matrix = []
    bellman_execution_times_adjacency_matrix = []
    floydwarshall_execution_times_adjacency_matrix = []
    johnson_execution_times_adjacency_matrix = []

    for node in nodes_range:
        print(node)
        # Generate a multigraph using erdos_renyi_graph
        G = nx.MultiGraph()
        G.add_nodes_from(range(node))
        for edge in nx.erdos_renyi_graph(node, 0.5).edges():
            weight = random.uniform(0.1, 1.0)
            G.add_edge(edge[0], edge[1], weight=weight)

        nodes = list(G.nodes())
        num_nodes = len(nodes)
        adj_matrix = np.zeros((num_nodes, num_nodes))
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if G.has_edge(node_i, node_j):
                    # Assuming you want the sum of weights for multigraphs
                    adj_matrix[i, j] = sum(data['weight'] for data in G[node_i][node_j].values())

        start_node = 0
        graph = G
        V = G.number_of_nodes()

        # dijkstra's algorithm
        start_time = time.time()
        dist = dijkstra_for_adjacency_matrix(start_node, node, adj_matrix, node-1)
        end_time = time.time()
        dijkstra_execution_time = end_time - start_time
        dijkstra_execution_times_adjacency_matrix.append(dijkstra_execution_time)


        # bellman's algorithm
        start_time = time.time()
        dist = BellmanFord(0, node-1, V, G)
        end_time = time.time()
        t = end_time - start_time
        bellman_execution_times_adjacency_matrix.append(t)

        # foldy warshall

        new_matrix = np.where(np.eye(len(adj_matrix), dtype=bool), 0, adj_matrix).astype(float)
        new_matrix[new_matrix == 0] = np.inf
        np.fill_diagonal(new_matrix, 0)

        start_time = time.time()
        dist = floydWarshall_adjacency_matrix(new_matrix, node, 0, node-1)
        end_time = time.time()
        t = end_time - start_time
        floydwarshall_execution_times_adjacency_matrix.append(t)

        # jonshon's algorithm

        start_time = time.time()
        dist = JohnsonAlgorithm(adj_matrix, 0, node-1)
        end_time = time.time()
        t = end_time - start_time
        johnson_execution_times_adjacency_matrix.append(t)

    plt.plot(nodes_range, dijkstra_execution_times_adjacency_matrix, label="Dijkstra", marker='o')
    plt.plot(nodes_range, bellman_execution_times_adjacency_matrix, label="Bellman", marker='o')
    plt.plot(nodes_range, floydwarshall_execution_times_adjacency_matrix, label="FoldyWarshall", marker='o')
    plt.plot(nodes_range, johnson_execution_times_adjacency_matrix, label="Johnson", marker='o')
    plt.title("Multigraph")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Execution Time (seconds)")
    plt.legend()
    plt.show()
