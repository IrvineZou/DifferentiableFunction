from utils import *
import copy
import numpy as np

class GeneticAlgorithm:
    def __init__(
        self,
        env,
        topological_order,
        population_size=80,
        generations=100,
        mutation_rate=0.1,
        crossover_rate=0.8
    ):
        self.env = env
        self.topological_order = topological_order
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.action_space = env.action_space
        self.population = self.initialize_population()
        self.best_individual = None
        self.best_fitness = float('-inf')
        self.fitness_history = []

        self.min_time = np.inf
        self.sample = 0
        self.results = {}
        self.fitness_cache = {}

        # Extract action space parameters
        self.nvec = self.action_space.nvec  # Array of action sizes for each dimension

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = {}
            for node in self.topological_order:
                action = self.action_space.sample()
                individual[node] = action
            population.append(individual)
        return population

    def fitness(self, individual):
        # Use cached fitness if available
        individual_key = tuple((node, tuple(individual[node])) for node in self.topological_order)
        if individual_key in self.fitness_cache:
            return self.fitness_cache[individual_key]

        total_reward = 0
        obs = self.env.reset(True)
        done = False
        pdb.set_trace()
        for node in self.topological_order:
            action = individual[node]
            obs, reward, done, _, _ = self.env.step2(action,False)
            total_reward += reward

            if done:
                #if self.env.time < self.min_time:
                #    self.min_time = self.env.time
                #self.results[self.sample] = self.min_time
                self.sample += 1
                break  # Exit the loop if done

        # Cache the fitness score
        pdb.set_trace()
        self.fitness_cache[individual_key] = total_reward
        return total_reward

    def selection(self, fitness_scores):
        tournament_size = 3
        selected = []
        for _ in range(self.population_size):
            participants = random.sample(list(zip(self.population, fitness_scores)), tournament_size)
            best = max(participants, key=lambda ind_fitness: ind_fitness[1])
            selected.append(copy.deepcopy(best[0]))
        return selected

    def crossover(self, parent1, parent2):
        child1 = {}
        child2 = {}
        for node in self.topological_order:
            if random.random() < self.crossover_rate:
                # Swap actions between parents
                child1[node] = parent2[node]
                child2[node] = parent1[node]
            else:
                child1[node] = parent1[node]
                child2[node] = parent2[node]
        return child1, child2

    def mutate(self, individual):
        for node in self.topological_order:
            if random.random() < self.mutation_rate:
                action = individual[node].copy()
                # Randomly select an index in the action vector to mutate
                idx = random.randint(0, len(self.nvec) - 1)
                # Randomly select a new value for that index
                action[idx] = random.randint(0, self.nvec[idx] - 1)
                individual[node] = action
        return individual

    def evolve(self):
        for generation in range(self.generations):
            # Evaluate fitness for the current population
            fitness_scores = [self.fitness(individual) for individual in self.population]

            max_fitness = max(fitness_scores)
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            self.fitness_history.append(avg_fitness)

            best_idx = fitness_scores.index(max_fitness)
            if max_fitness > self.best_fitness:
                self.best_fitness = max_fitness
                self.best_individual = copy.deepcopy(self.population[best_idx])

            # Selection
            selected = self.selection(fitness_scores)
            print("generation {}:fitness score:{}".format(generation,max_fitness))
            # Crossover and Mutation
            next_generation = []
            for i in range(0, self.population_size, 2):
                parent1 = selected[i]
                parent2 = selected[(i + 1) % self.population_size]
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                next_generation.extend([child1, child2])

            self.population = next_generation[:self.population_size]

