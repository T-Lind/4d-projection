import numpy as np
from tqdm import tqdm


def fitness(solution):
    central_sphere = np.array([0, 0, 0, 0, 0])
    count = 0
    for sphere in solution:
        if np.linalg.norm(sphere - central_sphere) == 2:
            count += 1
    return count


def generate_initial_population(size, num_spheres):
    population = []
    for _ in range(size):
        solution = [np.random.randn(5) for _ in range(num_spheres)]
        population.append(solution)
    return population


def crossover(parent1, parent2):
    # Simple one-point crossover
    crossover_point = np.random.randint(len(parent1))
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate(solution, mutation_rate=0.01):
    for sphere in solution:
        if np.random.rand() < mutation_rate:
            sphere += np.random.randn(5) * 0.1


def genetic_algorithm(pop_size, num_spheres, generations):
    population = generate_initial_population(pop_size, num_spheres)
    for generation in tqdm(range(generations)):
        population = sorted(population, key=fitness, reverse=True)
        new_population = population[:pop_size // 2]
        while len(new_population) < pop_size:
            indices = np.random.choice(pop_size // 4, 2, replace=False)
            parent1, parent2 = population[indices[0]], population[indices[1]]
            child1, child2 = crossover(parent1, parent2)
            mutate(child1)
            mutate(child2)
            new_population.extend([child1, child2])
        population = new_population
    best_solution = max(population, key=fitness)
    return best_solution


best_solution = genetic_algorithm(100, 120, 50000)
print("Best solution has", fitness(best_solution), "spheres touching the central sphere.")
