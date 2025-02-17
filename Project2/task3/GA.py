import numpy as np
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from deap import base, creator, tools, algorithms
from tqdm import tqdm

import util

"""
class GA:
    def __init__(self, population_size=100, individual_size=256, num_true=30, generations=100, crossover_rate=0.7,
                 mutation_rate=0.01):
        self.population_size = population_size
        self.individual_size = individual_size
        self.num_true = num_true
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population = [self.generate_individual() for _ in range(population_size)]
        self.X = util.load_data('X.pkl')
        self.y = util.load_data('y.pkl')

    def generate_individual(self):
        individual = np.zeros(self.individual_size, dtype=bool)
        idx = random.sample(range(self.individual_size), self.num_true)
        individual[idx] = True
        return individual

    def fitness(self, individual):
        mask_code = individual.astype(int)
        X = self.X.copy()
        X = X * mask_code
        X = X[:, np.any(X != 0, axis=0)]
        X_train, X_test, y_train, y_test = train_test_split(X, self.y, test_size=0.2, random_state=42)
        svm = SVC(kernel='rbf', gamma='scale', C=1.0, max_iter=100)
        svm.fit(X_train, y_train)
        return accuracy_score(y_test, svm.predict(X_test))

    def tournament_selection(self, k=3):
        selected = random.sample(self.population, k)
        selected.sort(key=self.fitness(), reverse=True)
        return selected[0]

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, self.individual_size - 1)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            return child1, child2
        return parent1, parent2

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            mutation_point = random.randint(0, self.individual_size - 1)
            individual[mutation_point] = not individual[mutation_point]
        return individual

    def run(self):
        for generation in range(self.generations):
            self.population.sort(key=self.fitness, reverse=True)

            best_individual = self.population[0]
            best_fitness = self.fitness(best_individual)
            print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")
            new_population = []
            new_population.append(self.population[0])

            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()

                child1, child2 = self.crossover(parent1, parent2)

                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            self.population = new_population
"""

X = util.load_data('X.pkl')
y = util.load_data('y.pkl')

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def create_individual():
    individual = np.zeros(30, dtype=int)
    idx = random.sample(range(256), 30)
    individual[idx] = True
    return individual


def evaluate(individual):
    mask_code = np.array(individual, dtype=int)
    if sum(mask_code) <= 30:
        X_temp = X.copy()
        X_temp = X_temp * mask_code
        X_temp = X_temp[:, np.any(X_temp != 0, axis=0)]
        X_train, X_test, y_train, y_test = train_test_split(X_temp, y, test_size=0.2, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        return accuracy_score(y_test, rf.predict(X_test)),
    else:
        return -1,


def mut_boolean(individual, indpb=0.2):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = not individual[i]
    return individual,


def cx_boolean(ind1, ind2):
    size = len(ind1)
    cxpoint = random.randint(1, size - 1)
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]
    return ind1, ind2


toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", cx_boolean)
toolbox.register("mutate", mut_boolean, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

population_size = 30
generations = 100
cx_prob = 0.7
mut_prob = 0.2

population = toolbox.population(n=population_size)

for gen in tqdm(range(generations), desc="Generations", unit="gen"):
    for ind in tqdm(population, desc="Evaluating individuals", leave=True, ncols=100, dynamic_ncols=True):
        if not ind.fitness.valid:
            ind.fitness.values = toolbox.evaluate(ind)

    offspring = toolbox.select(population, len(population))

    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cx_prob:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < mut_prob:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    for ind in offspring:
        if not ind.fitness.valid:
            ind.fitness.values = toolbox.evaluate(ind)

    population[:] = offspring

    fits = [ind.fitness.values[0] for ind in population]
    best_fitness = max(fits)
    best_individual = population[fits.index(best_fitness)]
    print(f"Generation {gen}: Best Fitness = {best_fitness}")
    print(f"Generation {gen}: Best Individual = {best_individual}")

best_individual = tools.selBest(population, 1)[0]
print(f"Best individual: {best_individual}, Fitness: {best_individual.fitness.values[0]}")
