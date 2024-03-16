import matplotlib.pyplot as plt
import time
import networkx as nx
import random
import numpy as np

INF = float('inf')


def floydwarshall_adjancey_list(graph):
    V = len(graph)
    dist = [[INF] * V for _ in range(V)]
    for i in range(V):
        for j in range(V):
            if i == j:
                dist[i][j] = 0
            elif graph.has_edge(i, j):
                dist[i][j] = graph[i][j]['weight']

    for k in range(V):
        for i in range(V):
            for j in range(V):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist


def shortest_distance_between(graph, src, dest):
    dist_matrix = floydwarshall_adjancey_list(graph)
    return dist_matrix[src][dest]


def floydWarshall_adjacency_matrix(graph, V, src, dest):
    dist = [row[:] for row in graph]

    for k in range(V):
        for i in range(V):
            for j in range(V):
                if dist[i][k] != float('inf') and dist[k][j] != float('inf'):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    # Check for negative cycles
    for i in range(V):
        if dist[i][i] < 0:
            return float('-inf')  # Negative cycle detected

    shortest_distance = dist[src][dest]
    return shortest_distance

if __name__ == "__main__":
    nodes_range = range(10, 20, 10)
    folydWarshall_execution_times_adjacency_list = []
    folydWarshall_execution_times_adjacency_matrix = []
    for node in nodes_range:
        print(node)
        G = nx.erdos_renyi_graph(node, 0.5)
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = random.uniform(0.1, 1.0)

        nodes = list(G.nodes())
        num_nodes = len(nodes)
        INF = float('inf')
        adj_matrix = np.full((num_nodes, num_nodes), INF)
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i == j:
                    adj_matrix[i, j] = 0
                if G.has_edge(node_i, node_j):
                    adj_matrix[i, j] = G[node_i][node_j]['weight']
        print(adj_matrix)
        start_node = 0
        end_node = 9
        graph = G
        V = G.number_of_nodes()
        start_time = time.time()
        dist = floydWarshall_adjacency_matrix(adj_matrix, node,0,9)
        end_time = time.time()
        t = end_time - start_time
        folydWarshall_execution_times_adjacency_matrix.append(t)
    print("hello")
    plt.plot(nodes_range, folydWarshall_execution_times_adjacency_matrix, label="Foldy Warshall", marker='o')
    plt.title("Algorithm Execution Time vs. Number of Nodes")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Execution Time (seconds)")
    plt.legend()
    plt.show()
