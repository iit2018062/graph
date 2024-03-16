import matplotlib.pyplot as plt
import time
import networkx as nx
import numpy as np
from algos.Dijkstra import dijkstra_for_adjacency_matrix
from algos.Bellman import  BellmanFord
from algos.FloydWarshall import floydWarshall_adjacency_matrix
from algos.johnson import JohnsonAlgorithm
from algos.bfs import bfs_shortest_path

if __name__ == "__main__":
    nodes_range = range(100, 1000, 100)
    dijkstra_execution_times_adjacency_matrix = []
    bellman_execution_times_adjacency_matrix = []
    floydwarshall_execution_times_adjacency_matrix = []
    johnson_execution_times_adjacency_matrix = []
    bfs_times = []
    for node in nodes_range:
        print(node)
        G = nx.erdos_renyi_graph(node, 0.6)
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

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

        # dijkstra's algorithm
        start_time = time.time()
        dist = dijkstra_for_adjacency_matrix(start_node, node, adj_matrix, 9)
        end_time = time.time()
        dijkstra_execution_time = end_time - start_time
        dijkstra_execution_times_adjacency_matrix.append(dijkstra_execution_time)

        # bfs's algorithm

        start_time = time.time()
        dist = bfs_shortest_path(adj_matrix, start_node, 9)
        end_time = time.time()
        t = end_time - start_time
        bfs_times.append(t)

        # bellman ford algorithm

        start_time = time.time()
        dist = BellmanFord(start_node, 9, node, graph)
        end_time = time.time()
        t = end_time - start_time
        bellman_execution_times_adjacency_matrix.append(t)

        # foldy warshall algorithm
        new_matrix = np.where(np.eye(len(adj_matrix), dtype=bool), 0, adj_matrix).astype(float)
        new_matrix[new_matrix == 0] = np.inf
        np.fill_diagonal(new_matrix, 0)

        start_time = time.time()
        dist = floydWarshall_adjacency_matrix(new_matrix, node, start_node, 9)
        end_time = time.time()
        t = end_time - start_time
        floydwarshall_execution_times_adjacency_matrix.append(t)

        # johnson's algorithm

        start_time = time.time()
        dist = JohnsonAlgorithm(adj_matrix, start_node, 9)
        end_time = time.time()
        t = end_time - start_time
        johnson_execution_times_adjacency_matrix.append(t)



    plt.plot(nodes_range, dijkstra_execution_times_adjacency_matrix, label="Dijkstra", marker='o')
    plt.plot(nodes_range, bellman_execution_times_adjacency_matrix, label="Bellman", marker='o')
    plt.plot(nodes_range, floydwarshall_execution_times_adjacency_matrix, label="FlodyWarshall", marker='o')
    plt.plot(nodes_range, johnson_execution_times_adjacency_matrix, label="Johnson", marker='o')
    plt.plot(nodes_range, bfs_times, label="BFS", marker='o')
    plt.title("unweighted graph")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Execution Time (seconds)")
    plt.legend()
    plt.show()
