import matplotlib.pyplot as plt
import time
import networkx as nx
import random
import numpy as np
import sys
import heapq


def dijkstra_for_adjacency_matrix(start, V, graph, end):
    distance = [sys.maxsize] * V
    distance[start] = 0
    q = [(0, start)]
    while q:
        curr_dist, u = heapq.heappop(q)
        if end == u:
            return distance[u]
        if distance[u] < curr_dist:
            continue
        for v in range(V):
            if graph[u][v] > 0 and distance[v] > distance[u] + graph[u][v]:
                distance[v] = distance[u] + graph[u][v]
                heapq.heappush(q, (distance[v], v))
    return distance[end]


if __name__ == "__main__":
    nodes_range = range(10, 100, 10)
    dijkstra_execution_times_adjacency_matrix = []
    for node in nodes_range:
        print(node)
        G = nx.erdos_renyi_graph(node, 0.5)
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = random.uniform(0.1, 1.0)

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
        start_time = time.time()
        dist = dijkstra_for_adjacency_matrix(start_node, node, adj_matrix, node - 1)
        end_time = time.time()
        dijkstra_execution_time = end_time - start_time
        dijkstra_execution_times_adjacency_matrix.append(dijkstra_execution_time)
    plt.plot(nodes_range, dijkstra_execution_times_adjacency_matrix, label="adjacency matrix", marker='o')
    plt.title("Algorithm Execution Time vs. Number of Nodes")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Execution Time (seconds)")
    plt.legend()
    plt.show()
