import argparse
import random
import numpy as np
from Evaluator import read_social_network, read_seed_sets, cascade


def eva_node(G, attr, card_node, explored_nodes, activated_nodes):
    frontier_nodes = {card_node}
    explored_nodes.add(card_node)
    activated_nodes.add(card_node)

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


def greedy_heur_search(G, I1, I2, k, iteration):
    S1 = []
    S2 = []
    U1 = I1
    U2 = I2
    h1 = {v: 0 for v in G if v not in U1}
    h2 = {v: 0 for v in G if v not in U2}

    while len(S1) + len(S2) < k:
        for i in range(iteration):
            e1, a1 = cascade(G, U1, 'p1')
            e2, a2 = cascade(G, U2, 'p2')
            fai = -len((e1 - e2).union(e2 - e1))
            for v in h1:
                if v not in a1 and v not in U1:
                    e1_v = eva_node(G, 'p1', v, e1.copy(), a1.copy())[0]
                    e1_v = e1.union(e1_v)
                    h1[v] += -len((e1_v - e2).union(e2 - e1_v)) - fai

            for v in h2:
                if v not in a2 and v not in U2:
                    e2_v = eva_node(G, 'p2', v, e2.copy(), a2.copy())[0]
                    e2_v = e2.union(e2_v)
                    h2[v] += -len((e2_v - e1).union(e1 - e2_v)) - fai

        v1 = max(h1, key=h1.get)
        v2 = max(h2, key=h2.get)

        if h1[v1] > h2[v2]:
            S1.append(v1)
            U1.append(v1)
        else:
            S2.append(v2)
            U2.append(v2)

        avg_h1 = np.mean(list(h1.values()))
        avg_h2 = np.mean(list(h2.values()))

        if len(h1) > k:
            h1 = {v: h1[v] for v in h1 if h1[v] >= avg_h1 and v not in U1}
        else:
            h1 = {v: h1[v] for v in h1 if v not in U1}
            while len(h1) < (k - 1):
                random_node = random.choice([v for v in G if v not in U1 and v not in h1])
                h1[random_node] = 0

        if len(h2) > k:
            h2 = {v: h2[v] for v in h2 if h2[v] >= avg_h2 and v not in U2}
        else:
            h2 = {v: h2[v] for v in h2 if v not in U2}
            while len(h2) < (k - 1):
                random_node = random.choice([v for v in G if v not in U2 and v not in h2])
                h2[random_node] = 0

    return S1, S2


def main():
    parser = argparse.ArgumentParser(description='Evaluator for social network diffusion model')
    parser.add_argument('-n', '--network', required=True)
    parser.add_argument('-i', '--initial', required=True)
    parser.add_argument('-b', '--balanced', required=True)
    parser.add_argument('-k', '--budget', type=int, required=True)
    args = parser.parse_args()

    G, n, m = read_social_network(args.network)
    I1, I2 = read_seed_sets(args.initial)
    k = int(args.budget)

    if n > 30000:
        iteration = 5
    elif n > 5000:
        iteration = 10
    elif n > 1000:
        iteration = 20
    else:
        iteration = 150

    S1, S2 = greedy_heur_search(G, I1, I2, k, iteration)

    with open(args.balanced, 'w') as f:
        f.write(f'{len(S1)} {len(S2)}\n')
        for i in range(len(S1)):
            f.write(f'{S1[i]}\n')
        for i in range(len(S2)):
            f.write(f'{S2[i]}\n')


if __name__ == '__main__':
    main()
