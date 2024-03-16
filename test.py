from algos.Dijkstra import dijkstra_for_adjacency_matrix
from algos.Bellman import BellmanFord_matrix_graph
from algos.FloydWarshall import floydWarshall_adjacency_matrix
from algos.johnson import JohnsonAlgorithm
import random
import numpy as np

random.seed(42)

if __name__ == "__main__":
    g = [
        [0, 4, 0, 0, 0, 0, 0, 8, 0],
        [4, 0, 8, 0, 0, 0, 0, 11, 0],
        [0, 8, 0, 7, 0, 4, 0, 0, 2],
        [0, 0, 7, 0, 9, 14, 0, 0, 0],
        [0, 0, 0, 9, 0, 10, 0, 0, 0],
        [0, 0, 4, 14, 10, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 1, 6],
        [8, 11, 0, 0, 0, 0, 1, 0, 7],
        [0, 0, 2, 0, 0, 0, 6, 7, 0]
    ]

    # Dijkstra's algorithm
    dist_dijkstra = dijkstra_for_adjacency_matrix(0, 9, g, 8)
    print("Dijkstra: Vertex Distance from Source")
    print(dist_dijkstra)

    edges = []
    for i in range(len(g)):
        for j in range(len(g[i])):
            if g[i][j] != 0:
                edges.append([i, j, g[i][j]])
    dist = BellmanFord_matrix_graph(edges, 0, 9, 8)
    print("bellman")
    print(dist)
    # Johnson's algorithm
    dist_johnson = JohnsonAlgorithm(g,0,8)
    print("johnson")
    print(dist_johnson)
    # Floyd-Warshall algorithm
    new_matrix = np.where(np.eye(len(g), dtype=bool), 0, g).astype(float)
    new_matrix[new_matrix == 0] = np.inf
    np.fill_diagonal(new_matrix, 0)
    #print(new_matrix)
    dist_floyd_warshall = floydWarshall_adjacency_matrix(new_matrix, 9, 0, 8)
    print("Floyd-Warshall: Vertex Distance from Source")
    print(dist_floyd_warshall)




