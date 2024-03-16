import matplotlib.pyplot as plt
import time
import networkx as nx
import numpy as np
from algos.Bellman import BellmanFord
from algos.FloydWarshall import floydWarshall_adjacency_matrix
from algos.johnson import JohnsonAlgorithm
import random


def generate_weighted_undirected_acyclic_graph(num_nodes, seed=None):
    random.seed(seed)
    edges = []
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            edges.append((i, j))
    random.shuffle(edges)
    for edge in edges:
        i, j = edge
        if not nx.has_path(G, j, i):
            w = random.uniform(0.1, 1.0)
            G.add_edge(i, j, weight=w)
            if nx.is_directed_acyclic_graph(nx.DiGraph(G)):
                continue
            else:
                G.remove_edge(i, j)
    return G


def graph_generate(num_nodes, probability, weight_range=(-1.0, 1.0)):
    G = generate_weighted_undirected_acyclic_graph(num_nodes)
    for edge in G.edges():
        if random.random() < probability:
            weight = random.uniform(weight_range[0], 0.0)  # Ensure negative weight
            G[edge[0]][edge[1]]['weight'] = weight

    return G


if __name__ == "__main__":
    nodes_range = range(100, 2000, 100)
    dijkstra_execution_times_adjacency_matrix = []
    bellman_execution_times_adjacency_matrix = []
    folydwarshall_execution_times_adjacency_matrix = []
    johnson_execution_times_adjacency_matrix = []

    for node in nodes_range:
        print(node)
        G = graph_generate(node, 0.6, weight_range=(-0.1, 0.1))

        nodes = list(G.nodes())
        num_nodes = len(nodes)
        adj_matrix = np.zeros((num_nodes, num_nodes))
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if G.has_edge(node_i, node_j):
                    adj_matrix[i, j] = G[node_i][node_j]['weight']

        start_node = 0
        graph = G
        V = G.number_of_nodes()

        print("bellman starts")

        start_time = time.time()
        dist = BellmanFord(start_node, 9, V, graph)
        end_time = time.time()
        t = end_time - start_time
        bellman_execution_times_adjacency_matrix.append(t)

        print("foldy warshall starts")

        start_time = time.time()
        dist = floydWarshall_adjacency_matrix(adj_matrix, V, start_node, 9)
        end_time = time.time()
        t = end_time - start_time
        folydwarshall_execution_times_adjacency_matrix.append(t)

        print("johnson starts")

        start_time = time.time()
        dist = JohnsonAlgorithm(adj_matrix, start_node, 9)
        end_time = time.time()
        t = end_time - start_time
        johnson_execution_times_adjacency_matrix.append(t)

    plt.plot(nodes_range, bellman_execution_times_adjacency_matrix, label="Bellman", marker='o')
    #plt.plot(nodes_range, folydwarshall_execution_times_adjacency_matrix, label="FoldyWarshall", marker='o')
    plt.plot(nodes_range, johnson_execution_times_adjacency_matrix, label="Johnson", marker='o')
    plt.title("Negative weights Time vs. Number of Nodes")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Execution Time (seconds)")
    plt.legend()
    plt.show()
