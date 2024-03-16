import time
import random
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np

MAX_INT = float('Inf')


def BellmanFord(src, dest, V, graph):
    distance = [float("Inf")] * V
    distance[src] = 0
    for _ in range(V - 1):
        for edge in graph.edges(data=True):

            a, b, data = edge
            w = data['weight']
            if distance[a] != float("Inf") and distance[a] + w < distance[b]:
                distance[b] = distance[a] + w

    for u, v, data in graph.edges(data=True):
        w = data['weight']
        if distance[u] != float("Inf") and distance[u] + w < distance[v]:
            print("negative cycle detected")
            return
    return distance[dest]


MAX_INT = float('inf')


def BellmanFord_matrix_graph(edges, src, num_vertices, dest):
    distance = [MAX_INT] * num_vertices
    distance[src] = 0
    for i in range(num_vertices - 1):
        for (src, des, weight) in edges:
            if distance[src] != MAX_INT and distance[src] + weight < distance[des]:
                distance[des] = distance[src] + weight

    for (src, des, weight) in edges:
        if distance[src] != MAX_INT and distance[src] + weight < distance[des]:
            return None

    return distance[dest]


if __name__ == "__main__":
    nodes_range = range(10, 100, 10)
    bellman_ford_execution_times_adjacency_list = []
    bellman_ford_execution_times_adjacency_matrix = []
    for node in nodes_range:
        print(node)
        G = nx.erdos_renyi_graph(node, 0.5)
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = random.uniform(-0.1, 1.0)
        start_node = 0
        end_node = 9
        graph = G
        V = G.number_of_nodes()
        nodes = list(G.nodes())
        num_nodes = len(nodes)
        adj_matrix = np.zeros((num_nodes, num_nodes))
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if G.has_edge(node_i, node_j):
                    adj_matrix[i, j] = G[node_i][node_j]['weight']
        edges = []
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix[i])):
                if adj_matrix[i][j] != 0:
                    edges.append([i, j, adj_matrix[i][j]])

        start_time = time.time()
        print(BellmanFord_matrix_graph(edges, 0, num_nodes, node - 1))
        end_time = time.time()
        t = end_time - start_time
        bellman_ford_execution_times_adjacency_matrix.append(t)
    plt.plot(nodes_range, bellman_ford_execution_times_adjacency_matrix, label="adjacency matrix", marker='o')
    plt.title("Algorithm Execution Time vs. Number of Nodes")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Execution Time (seconds)")
    plt.legend()
    plt.show()
