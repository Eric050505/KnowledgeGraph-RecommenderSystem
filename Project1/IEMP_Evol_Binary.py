import argparse
import random
import time

import numpy as np

from Evaluator import read_social_network, read_seed_sets, cascade, compute_objective_value

global len_G


def fitness(G, individual, k, len_I, iteration=10):
    avg = 0
    U1 = None
    U2 = None
    for _ in range(iteration):
        U1 = np.arange(len_G)[individual[:len_G]]
        U2 = np.arange(len_G, len(individual))[individual[len_G:]] - len_G
        e1 = cascade(G, U1, 'p1')[0]
        e2 = cascade(G, U2, 'p2')[0]
        sym = (e1 - e2).union(e2 - e1)
        avg += len_G - len(sym)
    if len(U1) + len(U2) > k + len_I:
        avg = -avg
    return avg / iteration


def generate_initial_population(G, I1, I2, initial_pop_size, k):
    population = []
    individual_temp = np.zeros(2 * len(G), dtype=bool)
    valid_nodes_temp = [v for v in G if v not in I1 and v not in I2]
    for v in I1:
        individual_temp[v] = True
    for v in I2:
        individual_temp[len(G) + v] = True

    for _ in range(initial_pop_size):
        individual = individual_temp.copy()
        valid_nodes = valid_nodes_temp.copy()
        remaining_budget = k
        while remaining_budget > 0:
            v = random.choice(valid_nodes)
            if random.random() < 0.5:
                individual[v] = True
            else:
                individual[len(G) + v] = True
            remaining_budget -= 1
            valid_nodes.remove(v)
        population.append(individual)

    return population


def binary_tournament_selection(population, pop_fitness):
    individual1, individual2 = random.sample(range(len(population)), 2)

    return population[individual1] if pop_fitness[individual1] > pop_fitness[individual2] else population[individual2]


def two_points_crossover(parent1, parent2):
    point1, point2 = random.sample(range(len(parent1)), 2)
    point1, point2 = sorted([point1, point2])
    offspring1 = np.copy(parent1)
    offspring2 = np.copy(parent2)
    for i in range(point1, point2):
        offspring1[i] = parent2[i]
        offspring2[i] = parent1[i]

    return offspring1, offspring2


def bit_flip_mutation(offspring, mutation_rate, I1, I2, k):
    len_S = 0
    for i in range(len(offspring)):
        if offspring[i]:
            if (i < len_G and i not in I1) or (i >= len_G and i - len_G not in I2):
                len_S += 1

    for i in offspring:
        if i not in I1 and i not in I2:
            if random.random() < mutation_rate:
                if offspring[i] is False and len_S < k:
                    offspring[i] = True
                    len_S += 1
                elif offspring[i] is True:
                    offspring[i] = False
                    len_S -= 1

    return offspring


def generate_new_solutions(population, pop_fitness, mutation_rate, I1, I2, k):
    new_population = []
    while len(new_population) < len(population):
        parent1 = binary_tournament_selection(population, pop_fitness)
        parent2 = binary_tournament_selection(population, pop_fitness)

        offspring1, offspring2 = two_points_crossover(parent1, parent2)
        offspring1 = bit_flip_mutation(offspring1, mutation_rate, I1, I2, k)
        offspring2 = bit_flip_mutation(offspring2, mutation_rate, I1, I2, k)

        new_population.append(offspring1)
        if len(new_population) < len(population):
            new_population.append(offspring2)

    return new_population


def genetic_algorithm(G, I1, I2, k, initial_pop_size, max_generation, mutation_rate):
    len_I = len(I1) + len(I2)
    population = generate_initial_population(G, I1, I2, initial_pop_size, k)
    pop_fitness = [fitness(G, individual, k, len_I) for individual in population]
    curr_generation = 0
    best_individual = None
    while curr_generation < max_generation:
        new_population = generate_new_solutions(population, pop_fitness, mutation_rate, I1, I2, k)
        new_pop_fitness = [fitness(G, individual, k, len_I) for individual in new_population]
        population = new_population
        pop_fitness = new_pop_fitness
        curr_generation += 1
        best_individual = population[np.argmax(pop_fitness)]

    return best_individual


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Evaluator for social network diffusion model')
    parser.add_argument('-n', '--network', required=True)
    parser.add_argument('-i', '--initial', required=True)
    parser.add_argument('-b', '--balanced', required=True)
    parser.add_argument('-k', '--budget', type=int, required=True)
    args = parser.parse_args()

    G, n, m = read_social_network(args.network)
    I1, I2 = read_seed_sets(args.initial)
    k = int(args.budget)
    global len_G
    len_G = len(G)

    if n > 30000:
        max_generation = 5
    elif n > 5000:
        max_generation = 10
    elif n > 1000:
        max_generation = 20
    else:
        max_generation = 100

    best_individual = genetic_algorithm(G, I1, I2,
                                        k=k, initial_pop_size=20, max_generation=max_generation,
                                        mutation_rate=0.1)

    U1 = np.arange(len_G)[best_individual[:len_G]]
    U2 = np.arange(len_G, len(best_individual))[best_individual[len_G:]] - len_G
    S1 = [v for v in U1 if v not in I1]
    S2 = [v for v in U2 if v not in I2]
    with open(args.balanced, 'w') as f:
        f.write(f'{len(S1)} {len(S2)}\n')
        for i in range(len(S1)):
            f.write(f'{S1[i]}\n')
        for i in range(len(S2)):
            f.write(f'{S2[i]}\n')

    object_value = compute_objective_value(G, I1, I2, S1, S2, 20)
    print(time.time() - start_time)
    print(object_value)


if __name__ == '__main__':
    main()
