import numpy as np
import pandas as pd
# from pymoo.core.problem import Problem
# from pymoo.optimize import minimize
# from pymoo.algorithms.soo.nonconvex.ga import GA
# from pymoo.termination import get_termination
# from pymoo.operators.crossover.ux import UniformCrossover
# from pymoo.operators.mutation.inversion import InversionMutation

# #function to calculate the total cost of a route
# def calculate_total_cost(route, cost_matrix):
#     #total cost starts at 0
#     total_cost = 0

#     #go through all values of the route, retrieve their cost from the cost matrix and add to the total cost
#     for i in range(len(route) - 1):
#         total_cost += cost_matrix[route[i], route[i+1]]
#     total_cost += cost_matrix[route[-1], route[0]]

#     #return the cost
#     return total_cost

# def evaluate(X, cost_matrix):
#     return np.array([calculate_total_cost(route, cost_matrix) for route in X])

# #define the problem to use pymoo's GA
# def create_problem(cost_matrix, n_cities):
#     problem = Problem(
#         n_var=n_cities,
#         n_obj=1,
#         xl=0,
#         xu=n_cities-1,
#         type_var=int,
#         evaluation=lambda X, out: out.__setitem__("F", evaluate(X, cost_matrix))
#     )
#     return problem

# def solve_tsp(cost_matrix):
#     n_cities = 50
#     problem = create_problem(cost_matrix, n_cities)

#     algorithm = GA(
#         pop_size=40,  # Population size
#         crossover=UniformCrossover(),  # Ordered crossover
#         mutation=InversionMutation(prob=0.1),  # Inversion mutation
#         eliminate_duplicates=True
#     )
#     termination = get_termination('n_gen', 250)
#     res = minimize(problem, algorithm, termination=termination, seed=1, verbose=True)

#     return res.X, res.F

#function to load the cost and time functions of the transport methods
def load_matrix(filename):
    #read csv with the filename
    df = pd.read_csv(filename, index_col=0)

    #replace the dashes with insanely high number, so that these options are immediately discarted
    df.replace("-", 0, inplace=True)
    
    #convert from pandas dataframe to a numpy array (easier to work with)
    return df.values

#calculate total cost of a given route (fitness)
def tsp_total_cost(route, cost_matrix):
    total_cost = 0
    for i in range(len(route) - 1):
        total_cost += float(cost_matrix[route[i], route[i + 1]])
    total_cost += float(cost_matrix[route[-1], route[0]])
    return total_cost

#generate an initial population (heuristic)
def generate_population(pop_size, n_cities):
    population = [np.random.permutation(n_cities) for _ in range(pop_size)]
    population[pop_size-1] = [0, 25, 6, 24, 12, 26, 10, 35, 34, 49, 40, 33, 17, 21, 20, 11, 13, 1, 39, 31, 46, 41, 32, 23, 18, 29, 15, 9, 30, 4, 27, 42, 44, 43, 14, 47, 28, 37, 48, 22, 45, 19, 38, 8, 16, 7, 5, 36, 2, 3] #49
    return population

#evaluate fitness of population
def evaluate_population(population, cost_matrix):
    fitness = [tsp_total_cost(route, cost_matrix) for route in population]
    return np.array(fitness)


#tournament selection
def tournament_selection(population, fitness, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        participants = np.random.choice(len(population), tournament_size)
        best = participants[np.argmin(fitness[participants])]
        selected.append(population[best])
    return selected

#ordered crossover
def ordered_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(np.random.choice(range(size), 2, replace=False))
    child = [-1] * size
    child[start:end+1] = parent1[start:end+1]

    pointer = 0
    for i in range(size):
        if parent2[i] not in child:
            while child[pointer] != -1:
                pointer += 1
            child[pointer] = parent2[i]
    
    return np.array(child)

#swap mutation
def swap_mutation(route, mutation_rate=0.1):
    if np.random.rand() < mutation_rate:
        idx1, idx2 = np.random.choice(len(route), 2, replace=False)
        route[idx1], route[idx2] = route[idx2], route[idx1]
    return route

#create children
def create_offspring(parents, crossover_rate=0.9, mutation_rate=0.1):
    offspring = []
    for i in range(0, len(parents), 2):
        if np.random.rand() < crossover_rate:
            child1 = ordered_crossover(parents[i], parents[i+1])
            child2 = ordered_crossover(parents[i+1], parents[i])
        else:
            child1, child2 = parents[i], parents[i+1]
        
        offspring.append(swap_mutation(child1, mutation_rate))
        offspring.append(swap_mutation(child2, mutation_rate))
    
    return offspring

#GA
def genetic_algorithm(cost_matrix, n_cities=50, pop_size=40, n_generations=250, tournament_size=3, crossover_rate=0.9, mutation_rate=0.1):
    # Step 1: Generate the initial population
    population = generate_population(pop_size, n_cities)
    
    # Step 2: Evaluate the initial population
    fitness = evaluate_population(population, cost_matrix)
    
    # Step 3: Evolution loop
    for generation in range(n_generations):
        # Step 4: Select parents using tournament selection
        parents = tournament_selection(population, fitness, tournament_size)
        
        # Step 5: Create offspring using crossover and mutation
        offspring = create_offspring(parents, crossover_rate, mutation_rate)
        
        # Step 6: Evaluate the new offspring
        fitness_offspring = evaluate_population(offspring, cost_matrix)
        
        # Step 7: Combine population and offspring
        combined_population = population + offspring
        combined_fitness = np.concatenate((fitness, fitness_offspring))
        
        # Step 8: Select the best individuals for the next generation
        best_indices = np.argsort(combined_fitness)[:pop_size]
        population = [combined_population[i] for i in best_indices]
        fitness = combined_fitness[best_indices]
        
        # Optionally, print the best fitness every generation
        print(f"Generation {generation + 1}, Best Fitness: {fitness[0]}")
    
    # Return the best solution and its fitness
    best_solution = population[0]
    best_fitness = fitness[0]
    
    return best_solution, best_fitness

#read the xy matrix
xy = pd.read_csv("Data_Sets/xy.csv")
#bus matrices
cost_bus = load_matrix("Data_Sets/costbus.csv")
time_bus = load_matrix("Data_Sets/timebus.csv")
#plane matrices
cost_plane = load_matrix("Data_Sets/costplane.csv")
time_plane = load_matrix("Data_Sets/timeplane.csv")
#train matrices
cost_train = load_matrix("Data_Sets/costtrain.csv")
time_train = load_matrix("Data_Sets/timetrain.csv")

# Example: Solve for 10 cities using plane costs
best_route, best_cost = genetic_algorithm(cost_plane[:50, :50])

# Print the best route and cost found
print("Best Route (Plane):", best_route)
print("Best Total Cost (Plane):", best_cost)


###Cost
#Bus

