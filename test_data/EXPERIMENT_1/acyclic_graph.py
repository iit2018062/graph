import time
import numpy as np
from algos.Dijkstra import dijkstra_for_adjacency_matrix
from algos.Bellman import BellmanFord_matrix_graph
from algos.FloydWarshall import floydWarshall_adjacency_matrix
from algos.johnson import JohnsonAlgorithm
import random
import matplotlib.pyplot as plt

random.seed(42)
import random
import networkx as nx


def generate_weighted_undirected_acyclic_graph(num_nodes, seed=None):
    random.seed(seed)
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = []
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


def generate_graph_with_negative_weights(num_nodes, probability, weight_range=(-1.0, 1.0)):
    G = generate_weighted_undirected_acyclic_graph(num_nodes)
    for edge in G.edges():
        if random.random() < probability:
            weight = random.uniform(weight_range[0], 0.0)
            G[edge[0]][edge[1]]['weight'] = weight

    return G


def heuristic_function(u, v):
    return abs(u - v)


if __name__ == "__main__":
    nodes_range = range(10, 100, 10)
    dijkstra_execution_times_adjacency_matrix = []
    bellman_execution_times_adjacency_matrix = []
    folydwarshall_execution_times_adjacency_matrix = []
    johnson_execution_times_adjacency_matrix = []
    for node in nodes_range:
        print(node)
        G = generate_weighted_undirected_acyclic_graph(node, 0.5)
        nodes = list(G.nodes())

        num_nodes = len(nodes)
        adj_matrix = np.zeros((num_nodes, num_nodes))
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if G.has_edge(node_i, node_j):
                    adj_matrix[i, j] = G[node_i][node_j]['weight']

        graph = G
        V = G.number_of_nodes()

        # dijkstra's algorithm

        start_time = time.time()
        dist = dijkstra_for_adjacency_matrix(0, node, adj_matrix, node - 1)
        end_time = time.time()
        dijkstra_execution_time = end_time - start_time
        dijkstra_execution_times_adjacency_matrix.append(dijkstra_execution_time)

        # bellman's algorithm
        # making edges array
        edges = []
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix[i])):
                if adj_matrix[i][j] != 0:
                    edges.append([i, j, adj_matrix[i][j]])

        start_time = time.time()
        dist = BellmanFord_matrix_graph(edges, 0, node, node - 1)
        end_time = time.time()
        t = end_time - start_time
        bellman_execution_times_adjacency_matrix.append(t)

        # johnson's algorithm

        start_time = time.time()
        dist = JohnsonAlgorithm(adj_matrix, 0, node - 1)
        end_time = time.time()
        t = end_time - start_time
        johnson_execution_times_adjacency_matrix.append(t)

        # foldy warshall
        # making the matrix so that it has inf instead of 0 for foldywarshall

        new_matrix = np.where(np.eye(len(adj_matrix), dtype=bool), 0, adj_matrix).astype(float)
        new_matrix[new_matrix == 0] = np.inf
        np.fill_diagonal(new_matrix, 0)

        start_time = time.time()
        dist = floydWarshall_adjacency_matrix(adj_matrix, node, 0, node - 1)
        end_time = time.time()
        t = end_time - start_time
        folydwarshall_execution_times_adjacency_matrix.append(t)

    plt.plot(nodes_range, dijkstra_execution_times_adjacency_matrix, label="Dijkstra", marker='o')
    plt.plot(nodes_range, bellman_execution_times_adjacency_matrix, label="Bellman", marker='o')
    plt.plot(nodes_range, folydwarshall_execution_times_adjacency_matrix, label="FoldyWarshall", marker='o')
    plt.plot(nodes_range, johnson_execution_times_adjacency_matrix, label="Johnson", marker='o')
    plt.title("Acyclic graph")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Execution Time (seconds)")
    plt.legend()
    plt.show()
