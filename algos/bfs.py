from collections import deque
import math
import heapq
import numpy as np
import time
import matplotlib.pyplot as plt


def bfs_shortest_path(graph, source, target):
    visited = [False] * len(graph)
    V = len(graph)
    q = deque([(source, [source])])
    while q:
        vertex, path = q.popleft()
        if vertex == target:
            return path
        if not visited[vertex]:
            visited[vertex] = True
            for neighbor, connected in enumerate(graph[vertex]):
                if connected and not visited[neighbor]:
                    q.append((neighbor, path + [neighbor]))

    return None  # If no path found


def main():
    nodes_range = range(10, 100, 10)
    bfs_times = []
    for node in nodes_range:
        print("*****")
        print(node)
        print("***")
        random_matrix = np.random.randint(0, 2, size=(node, node))
        start_time = time.time()
        bfs_shortest_path(random_matrix, 0, 10)
        end_time = time.time()
        t = end_time - start_time
        bfs_times.append(t)
    plt.plot(nodes_range, bfs_times, label="bfs", marker='o')
    plt.title("Algorithm Execution Time vs. Number of Nodes")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Execution Time (seconds)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
