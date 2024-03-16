import matplotlib.pyplot as plt
import time
import networkx as nx
import numpy as np
from queue import PriorityQueue

MAX_INT = float('Inf')

MAX_INT = float('inf')


def Dijkstra(graph, modifiedGraph, src, dest):
    num_vertices = len(graph)
    dist = [MAX_INT] * num_vertices
    dist[src] = 0
    priority_queue = PriorityQueue()
    priority_queue.put((0, src))

    while not priority_queue.empty():
        (cur_dist, cur_vertex) = priority_queue.get()
        if cur_vertex == dest:
            return cur_dist
        for next_vertex in range(num_vertices):
            if (dist[next_vertex] > cur_dist + modifiedGraph[cur_vertex][next_vertex] and
                    graph[cur_vertex][next_vertex] != 0):
                dist[next_vertex] = cur_dist + modifiedGraph[cur_vertex][next_vertex]
                priority_queue.put((dist[next_vertex], next_vertex))

    return MAX_INT



def BellmanFord(edges, graph, num_vertices):
    dist = [MAX_INT] * (num_vertices + 1)
    dist[num_vertices] = 0

    for i in range(num_vertices):
        edges.append([num_vertices, i, 0])

    for i in range(num_vertices):
        for (src, des, weight) in edges:
            if dist[src] != MAX_INT and dist[src] + weight < dist[des]:
                dist[des] = dist[src] + weight

    # Check for negative cycles
    for (src, des, weight) in edges:
        if dist[src] != MAX_INT and dist[src] + weight < dist[des]:
            print("negative cycle")
            return dist[0:num_vertices], True  # Negative cycle detected

    return dist[0:num_vertices], False  # No negative cycle found


def JohnsonAlgorithm(graph, src, dest):
    edges = []
    for i in range(len(graph)):
        for j in range(len(graph[i])):

            if graph[i][j] != 0:
                edges.append([i, j, graph[i][j]])
    modifyWeights, flag = BellmanFord(edges, graph, len(graph))
    if(flag):
        return MAX_INT
    modifiedGraph = [[0 for x in range(len(graph))] for y in
                     range(len(graph))]
    for i in range(len(graph)):
        for j in range(len(graph[i])):

            if graph[i][j] != 0:
                modifiedGraph[i][j] = (graph[i][j] +
                                       modifyWeights[i] - modifyWeights[j])
    return Dijkstra(graph, modifiedGraph, src, dest)

import random

def generate_random_graph_adjacency_matrix(num_vertices, density, max_weight):
    # Initialize an empty adjacency matrix
    graph = [[0] * num_vertices for _ in range(num_vertices)]

    # Calculate the number of edges based on density
    max_edges = (num_vertices * (num_vertices - 1)) // 2
    num_edges = int(density * max_edges)

    # Generate random edges with random weights
    for _ in range(num_edges):
        src = random.randint(0, num_vertices - 1)
        des = random.randint(0, num_vertices - 1)
        while src == des or graph[src][des] != 0:  # Ensure no self-loops or duplicate edges
            des = random.randint(0, num_vertices - 1)
        weight = random.randint(-max_weight, max_weight)  # Allow negative weights
        graph[src][des] = weight

    return graph

def has_negative_cycle(graph):
    num_vertices = len(graph)
    dist = [float('inf')] * num_vertices
    dist[0] = 0

    # Relax all edges V-1 times
    for _ in range(num_vertices - 1):
        for src in range(num_vertices):
            for des in range(num_vertices):
                if graph[src][des] != 0:
                    dist[des] = min(dist[des], dist[src] + graph[src][des])

    # Check for negative cycles
    for src in range(num_vertices):
        for des in range(num_vertices):
            if graph[src][des] != 0 and dist[des] > dist[src] + graph[src][des]:
                return True  # Negative cycle detected

    return False



# Driver Code
if __name__ == "__main__":
    nodes_range = range(10, 100, 10)
    dijkstra_execution_times_adjacency_list = []
    dijkstra_execution_times_adjacency_matrix = []
    for node in nodes_range:
        random_graph = generate_random_graph_adjacency_matrix(node, 0.5, 10)
        while has_negative_cycle(random_graph):
            random_graph = generate_random_graph_adjacency_matrix(node, 0.5, 10)

        G = nx.erdos_renyi_graph(node, 0.2)
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = random.uniform(-1, 1)

        nodes = list(G.nodes())
        num_nodes = len(nodes)
        adj_matrix = np.zeros((num_nodes, num_nodes))
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if G.has_edge(node_i, node_j):
                    adj_matrix[i, j] = G[node_i][node_j]['weight']

        start_node = 0
        end_node = 9
        graph = G
        V = G.number_of_nodes()

        start_time = time.time()
        dist = JohnsonAlgorithm(random_graph,0,9)
        end_time = time.time()
        dijkstra_execution_time = end_time - start_time
        dijkstra_execution_times_adjacency_matrix.append(dijkstra_execution_time)
    plt.plot(nodes_range, dijkstra_execution_times_adjacency_matrix, label="johnson", marker='o')
    plt.title("Algorithm Execution Time vs. Number of Nodes")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Execution Time (seconds)")
    plt.legend()
    plt.show()
