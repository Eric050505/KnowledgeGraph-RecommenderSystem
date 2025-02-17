import argparse
import networkx as nx
import numpy as np


def read_social_network(path):
    with open(path, 'r') as f:
        n, m = map(int, f.readline().strip().split())
        G = nx.DiGraph()
        for _ in range(m):
            u, v, p1, p2 = map(float, f.readline().strip().split())
            G.add_edge(int(u), int(v), p1=p1, p2=p2)

    return G, n, m


def read_seed_sets(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    k1, k2 = map(int, lines[0].strip().split())
    I1 = [int(lines[i + 1].strip()) for i in range(k1)]
    I2 = [int(lines[k1 + i + 1].strip()) for i in range(k2)]

    return I1, I2


def cascade(G, seed_set, attr):
    activated_nodes = set(seed_set)
    explored_nodes = set(seed_set)
    frontier_nodes = set(seed_set)

    while frontier_nodes:
        next_nodes = set()
        for curr_node in frontier_nodes:
            neighbors = list(G.neighbors(curr_node))
            probs = np.random.rand(len(neighbors))
            activation_probs = np.array([G[curr_node][neighbor][attr] for neighbor in neighbors])
            activated_neighbors = np.where(probs < activation_probs)[0]
            explored_nodes = explored_nodes.union(set(neighbors))

            for i in activated_neighbors:
                neighbor = neighbors[i]
                if neighbor not in activated_nodes:
                    next_nodes.add(neighbor)
                    activated_nodes.add(neighbor)

        frontier_nodes = next_nodes

    return explored_nodes, activated_nodes


def compute_objective_value(G, I1, I2, S1, S2, iteration=2000):
    U1 = I1 + S1
    U2 = I2 + S2
    total = 0
    total_len = len(G.nodes)
    for i in range(iteration):
        exposure_nodes_c1 = cascade(G, U1, 'p1')[0]
        exposure_nodes_c2 = cascade(G, U2, 'p2')[0]
        symmetric_diff = (exposure_nodes_c1 - exposure_nodes_c2).union(exposure_nodes_c2 - exposure_nodes_c1)
        objective_value = total_len - len(symmetric_diff)
        total += objective_value
    return total / iteration


def main():
    parser = argparse.ArgumentParser(description='Evaluator for social network diffusion model')
    parser.add_argument('-n', '--network', required=True, help='Path to the social network file')
    parser.add_argument('-i', '--initial', required=True, help='Path to the initial seed set file')
    parser.add_argument('-b', '--balanced', required=True, help='Path to the balanced seed set file')
    parser.add_argument('-k', '--budget', type=int, required=True, help='Budget for the seed sets')
    parser.add_argument('-o', '--output', required=True, help='Path to the output file for the objective value')

    args = parser.parse_args()

    G, n, m = read_social_network(args.network)
    I1, I2 = read_seed_sets(args.initial)
    S1, S2 = read_seed_sets(args.balanced)

    objective_value = compute_objective_value(G, I1, I2, S1, S2)

    with open(args.output, 'w') as f:
        f.write(f'{objective_value}\n')


if __name__ == '__main__':
    main()
