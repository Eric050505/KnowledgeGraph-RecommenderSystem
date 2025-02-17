import random
import time

import numpy as np
import argparse

from Evaluator import read_social_network, read_seed_sets, cascade, compute_objective_value


def fitness(G, individual, I1, I2, iteration=10):
    avg = 0
    U1 = I1 + individual[0]
    U2 = I2 + individual[1]
    for _ in range(iteration):
        e1 = cascade(G, U1, 'p1')[0]
        e2 = cascade(G, U2, 'p2')[0]
        sym = (e1 - e2).union(e2 - e1)
        avg += len(G) - len(sym)
    return avg / iteration


def generate_initial_population(G, I1, I2, initial_pop_size, k):
    new_population = []
    for _ in range(initial_pop_size):
        individual = [list(), list()]
        while len(individual[0]) + len(individual[1]) < k:
            random_node = random.choice(list(G.nodes()))
            if random.random() < 0.5 and random_node not in I1 and random_node not in individual[0]:
                individual[0].append(random_node)
            elif random.random() >= 0.5 and random_node not in I2 and random_node not in individual[0]:
                individual[1].append(random_node)
        new_population.append(individual)
    return new_population


def binary_tournament_selection(population, pop_fitness):
    individual1, individual2 = random.sample(range(len(population)), 2)

    return population[individual1] if pop_fitness[individual1] > pop_fitness[individual2] else population[individual2]


def two_points_crossover(parent1, parent2):
    offspring1 = [sublist[:] for sublist in parent1]
    offspring2 = [sublist[:] for sublist in parent2]

    for i in range(len(parent1)):
        len_min = min(len(parent1[i]), len(parent2[i]))
        if len_min > 1:
            point1, point2 = sorted(random.sample(range(len_min), 2))
            offspring1[i][point1:point2] = parent2[i][point1:point2]
            offspring2[i][point1:point2] = parent1[i][point1:point2]

    return offspring1, offspring2


def mutation(G, offspring, mutation_rate, I1, I2):
    for i in range(len(offspring)):
        curr_I = I1 if i == 0 else I2
        for j in range(len(offspring[i])):
            if random.random() < mutation_rate:
                random_node = random.choice(list(G.nodes()))
                while random_node in curr_I:
                    random_node = random.choice(list(G.nodes()))
                offspring[i][j] = random_node


def generate_new_solutions(G, population, pop_fitness, mutation_rate, I1, I2):
    new_population = []
    while len(new_population) < len(population):
        parent1 = binary_tournament_selection(population, pop_fitness)
        parent2 = binary_tournament_selection(population, pop_fitness)

        offspring1, offspring2 = two_points_crossover(parent1, parent2)
        mutation(G, offspring1, mutation_rate, I1, I2)
        mutation(G, offspring2, mutation_rate, I1, I2)

        new_population.append(offspring1)
        if len(new_population) < len(population):
            new_population.append(offspring2)

    return new_population


def genetic_algorithm(G, I1, I2, k, initial_pop_size, max_generation, mutation_rate):
    population = generate_initial_population(G, I1, I2, initial_pop_size, k)
    pop_fitness = [fitness(G, individual, I1, I2) for individual in population]
    curr_generation = 0
    best_individual = None
    while curr_generation < max_generation:
        new_population = generate_new_solutions(G, population, pop_fitness, mutation_rate, I1, I2)
        new_pop_fitness = [fitness(G, individual, I1, I2) for individual in new_population]
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

    if n > 10000:
        max_generation = 40
    elif n > 5000:
        max_generation = 45
    elif n > 1000:
        max_generation = 50
    else:
        max_generation = 100

    best_individual = genetic_algorithm(G, I1, I2,
                                        k=k, initial_pop_size=20, max_generation=max_generation,
                                        mutation_rate=0.1)

    S1 = best_individual[0]
    S2 = best_individual[1]
    with open(args.balanced, 'w') as f:
        f.write(f'{len(S1)} {len(S2)}\n')
        for i in range(len(S1)):
            f.write(f'{S1[i]}\n')
        for i in range(len(S2)):
            f.write(f'{S2[i]}\n')

    object_value = compute_objective_value(G, I1, I2, S1, S2, 20)
    print(object_value)
    print(time.time() - start_time)


if __name__ == '__main__':
    main()
