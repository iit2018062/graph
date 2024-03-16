import networkx as nx
import time
import matplotlib.pyplot as plt
import random


def heuristic_function(u, v):
    return abs(u - v)


def generate_weighted_cyclic_graph(num_nodes, seed=None):
    random.seed(seed)
    graph = nx.cycle_graph(num_nodes)
    for i in range(num_nodes):
        j = (i + 1) % num_nodes
        weight = random.uniform(0.1, 1.0)
        graph[i][j]['weight'] = weight
    return graph


def algorithms():
    nodes_range = range(10, 100, 10)
    a_star_execution_times = []
    a_star_heuristic_execution_times = []
    dijkstra_execution_times = []
    bellman_ford_execution_times = []
    floyd_warshall_execution_times = []
    johnson_execution_times = []

    a_star_execution_times_dense = []
    a_star_heuristic_execution_times_dense = []
    dijkstra_execution_times_dense = []
    bellman_ford_execution_times_dense = []
    floyd_warshall_execution_times_dense = []
    johnson_execution_times_dense = []

    a_star_execution_times_cyclic = []
    a_star_heuristic_execution_times_cyclic = []
    dijkstra_execution_times_cyclic = []
    bellman_ford_execution_times_cyclic = []
    floyd_warshall_execution_times_cyclic = []
    johnson_execution_times_cyclic = []

    a_star_execution_times_multi_graph = []
    a_star_heuristic_execution_times_multi_graph = []
    dijkstra_execution_times_multi_graph = []
    bellman_ford_execution_times_multi_graph = []
    floyd_warshall_execution_times_multi_graph = []
    johnson_execution_times_multi_graph = []

    for num_nodes in nodes_range:
        print(num_nodes)
        # sparse graph
        graph = nx.fast_gnp_random_graph(num_nodes, 0.2, directed=True)
        graph.add_edge(0, num_nodes - 1)
        for u, v in graph.edges():
            graph[u][v]['weight'] = random.uniform(0.1, 1.0)

        # dense graph
        G = nx.complete_graph(num_nodes)
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = random.uniform(0.1, 1.0)

        # cyclic graph
        cyclic_graph = generate_weighted_cyclic_graph(num_nodes)

        # multigraph
        multi_graph = nx.MultiGraph()
        multi_graph.add_nodes_from(range(num_nodes))
        for edge in nx.erdos_renyi_graph(num_nodes, 0.5).edges():
            weight = random.uniform(0.1, 1.0)
            multi_graph.add_edge(edge[0], edge[1], weight=weight)
        # sparse graph
        start_time = time.time()
        nx.astar_path_length(graph, 0, num_nodes - 1, heuristic=None, weight='weight')
        end_time = time.time()
        a_star_execution_times.append(end_time - start_time)
        # dense graph
        start_time = time.time()
        nx.astar_path_length(G, 0, num_nodes - 1, heuristic=None, weight='weight')
        end_time = time.time()
        a_star_execution_times_dense.append(end_time - start_time)
        # cyclic graph
        start_time = time.time()
        nx.astar_path_length(cyclic_graph, 0, num_nodes - 1, heuristic=None, weight='weight')
        end_time = time.time()
        a_star_execution_times_cyclic.append(end_time - start_time)
        # multigraph graph
        start_time = time.time()
        nx.astar_path_length(multi_graph, 0, num_nodes - 1, heuristic=None, weight='weight')
        end_time = time.time()
        a_star_execution_times_multi_graph.append(end_time - start_time)

        # sparse graph
        start_time = time.time()
        nx.astar_path_length(graph, 0, num_nodes - 1, heuristic=heuristic_function, weight='weight')
        end_time = time.time()
        a_star_heuristic_execution_times.append(end_time - start_time)
        # dense graph
        start_time = time.time()
        nx.astar_path_length(G, 0, num_nodes - 1, heuristic=heuristic_function, weight='weight')
        end_time = time.time()
        a_star_heuristic_execution_times_dense.append(end_time - start_time)
        # cyclic graph
        start_time = time.time()
        nx.astar_path_length(cyclic_graph, 0, num_nodes - 1, heuristic=heuristic_function, weight='weight')
        end_time = time.time()
        a_star_heuristic_execution_times_cyclic.append(end_time - start_time)
        # multigraph
        start_time = time.time()
        nx.astar_path_length(multi_graph, 0, num_nodes - 1, heuristic=heuristic_function, weight='weight')
        end_time = time.time()
        a_star_heuristic_execution_times_multi_graph.append(end_time - start_time)

        # sparse graph
        start_time = time.time()
        nx.single_source_dijkstra_path_length(graph, 0, weight='weight')
        end_time = time.time()
        dijkstra_execution_times.append(end_time - start_time)

        # dense graph
        start_time = time.time()
        nx.single_source_dijkstra_path_length(G, 0, weight='weight')
        end_time = time.time()
        dijkstra_execution_times_dense.append(end_time - start_time)

        # multigraph
        start_time = time.time()
        nx.single_source_dijkstra_path_length(multi_graph, 0, weight='weight')
        end_time = time.time()
        dijkstra_execution_times_multi_graph.append(end_time - start_time)

        # cyclic graph
        start_time = time.time()
        nx.single_source_dijkstra_path_length(cyclic_graph, 0, weight='weight')
        end_time = time.time()
        dijkstra_execution_times_cyclic.append(end_time - start_time)

        # sparse graph

        start_time = time.time()
        nx.bellman_ford_path_length(graph, 0, num_nodes - 1, weight='weight')
        end_time = time.time()
        bellman_ford_execution_times.append(end_time - start_time)

        # dense graph
        start_time = time.time()
        nx.bellman_ford_path_length(G, 0, num_nodes - 1, weight='weight')
        end_time = time.time()
        bellman_ford_execution_times_dense.append(end_time - start_time)

        # multigraph
        start_time = time.time()
        nx.bellman_ford_path_length(multi_graph, 0, num_nodes - 1, weight='weight')
        end_time = time.time()
        bellman_ford_execution_times_multi_graph.append(end_time - start_time)

        # cyclic graph
        start_time = time.time()
        nx.bellman_ford_path_length(cyclic_graph, 0, num_nodes - 1, weight='weight')
        end_time = time.time()
        bellman_ford_execution_times_cyclic.append(end_time - start_time)

        # sparse graph
        start_time = time.time()
        nx.floyd_warshall_numpy(graph, weight='weight')
        end_time = time.time()
        floyd_warshall_execution_times.append(end_time - start_time)

        # dense graph
        start_time = time.time()
        nx.floyd_warshall_numpy(G, weight='weight')
        end_time = time.time()
        floyd_warshall_execution_times_dense.append(end_time - start_time)

        # multigraph
        start_time = time.time()
        nx.floyd_warshall_numpy(multi_graph, weight='weight')
        end_time = time.time()
        floyd_warshall_execution_times_multi_graph.append(end_time - start_time)

        # cyclic graph
        start_time = time.time()
        nx.floyd_warshall_numpy(cyclic_graph, weight='weight')
        end_time = time.time()
        floyd_warshall_execution_times_cyclic.append(end_time - start_time)

        # sparse graph
        start_time = time.time()
        nx.johnson(graph, weight='weight')
        end_time = time.time()
        johnson_execution_times.append(end_time - start_time)

        # dense graph
        start_time = time.time()
        nx.johnson(G, weight='weight')
        end_time = time.time()
        johnson_execution_times_dense.append(end_time - start_time)

        # multigraph
        start_time = time.time()
        nx.johnson(multi_graph, weight='weight')
        end_time = time.time()
        johnson_execution_times_multi_graph.append(end_time - start_time)

        # cyclic graph
        start_time = time.time()
        nx.johnson(cyclic_graph, weight='weight')
        end_time = time.time()
        johnson_execution_times_cyclic.append(end_time - start_time)

    # Plot the results for sparse graph
    plt.plot(nodes_range, a_star_execution_times, label="A* (without heuristic)")
    plt.plot(nodes_range, a_star_heuristic_execution_times, label="A* (with heuristic)")
    plt.plot(nodes_range, dijkstra_execution_times, label="Dijkstra")
    plt.plot(nodes_range, bellman_ford_execution_times, label="Bellman-Ford")
    plt.plot(nodes_range, floyd_warshall_execution_times, label="Floyd-Warshall")
    plt.plot(nodes_range, johnson_execution_times, label="Johnson")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Execution Time (seconds)")
    plt.title("sparse graph")
    plt.legend()
    plt.show()
    # Plot the results for dense graph
    plt.plot(nodes_range, a_star_execution_times_dense, label="A* (without heuristic)")
    plt.plot(nodes_range, a_star_heuristic_execution_times_dense, label="A* (with heuristic)")
    plt.plot(nodes_range, dijkstra_execution_times_dense, label="Dijkstra")
    plt.plot(nodes_range, bellman_ford_execution_times_dense, label="Bellman-Ford")
    plt.plot(nodes_range, floyd_warshall_execution_times_dense, label="Floyd-Warshall")
    plt.plot(nodes_range, johnson_execution_times_dense, label="Johnson")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Execution Time (seconds)")
    plt.title("dense graph")
    plt.legend()
    plt.show()
    # plot the results for multigraph
    plt.plot(nodes_range, a_star_execution_times_multi_graph, label="A* (without heuristic)")
    plt.plot(nodes_range, a_star_heuristic_execution_times_multi_graph, label="A* (with heuristic)")
    plt.plot(nodes_range, dijkstra_execution_times_multi_graph, label="Dijkstra")
    plt.plot(nodes_range, bellman_ford_execution_times_multi_graph, label="Bellman-Ford")
    plt.plot(nodes_range, floyd_warshall_execution_times_multi_graph, label="Floyd-Warshall")
    plt.plot(nodes_range, johnson_execution_times_multi_graph, label="Johnson")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Execution Time (seconds)")
    plt.title("multi graph")
    plt.legend()
    plt.show()
    # cyclic graph
    plt.plot(nodes_range, a_star_execution_times_cyclic, label="A* (without heuristic)")
    plt.plot(nodes_range, a_star_heuristic_execution_times_cyclic, label="A* (with heuristic)")
    plt.plot(nodes_range, dijkstra_execution_times_cyclic, label="Dijkstra")
    plt.plot(nodes_range, bellman_ford_execution_times_cyclic, label="Bellman-Ford")
    plt.plot(nodes_range, floyd_warshall_execution_times_cyclic, label="Floyd-Warshall")
    plt.plot(nodes_range, johnson_execution_times_cyclic, label="Johnson")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Execution Time (seconds)")
    plt.title("cyclic graph")
    plt.legend()
    plt.show()




algorithms()
