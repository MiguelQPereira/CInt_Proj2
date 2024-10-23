import numpy as np
import pandas as pd

#function to load the cost and time functions of the transport methods
def load_matrix(filename):
    #read csv with the filename
    df = pd.read_csv(filename, index_col=0)

    #replace the dashes with insanely high number, so that these options are immediately discarted

    if (filename[10:14] == "cost"):
        df.replace("-", 1e6, inplace=True)
        for col_name in df.columns:
            df[col_name] = df[col_name].astype(float)  # Convert the column to float
    elif (filename[10:14] == "time"):
        df.replace("-", "200h00", inplace=True)
        for col_index in range(len(df.columns)):  # Iterate through each column index
            for row_index in range(len(df)):  # Iterate through each row index
                value = df.iat[row_index, col_index]  # Access the value using iat
                hours, minutes = map(int, value.split('h'))
                df.iat[row_index, col_index] = hours*60 + minutes

    
    #convert from pandas dataframe to a numpy array (easier to work with)
    return df.values

def trim_cost_matrix(cost_matrix):
    indexes = []
    stationNumber = 0
    #iterate over columns
    for col in range(cost_matrix.shape[1]):
        #iterate over rows
        for row in range(cost_matrix.shape[0]):
            #get the value and see if its not 1e6, if its not then add 1 to stationNumber
            value = cost_matrix[row, col]
            if value < 1e6:
                stationNumber =  stationNumber + 1
        if stationNumber < 5:
            indexes.append(col)
        stationNumber = 0
    cost_matrix = np.delete(cost_matrix, indexes, axis=1)
    cost_matrix = np.delete(cost_matrix, indexes, axis=0)
    return len(cost_matrix), cost_matrix

def trim_time_matrix(cost_matrix):
    indexes = []
    stationNumber = 0
    #iterate over columns
    for col in range(cost_matrix.shape[1]):
        #iterate over rows
        for row in range(cost_matrix.shape[0]):
            #get the value and see if its not 1e6, if its not then add 1 to stationNumber
            value = cost_matrix[row, col]
            if value < 12000:
                stationNumber += 1
        if stationNumber < 5:
            indexes.append(col)
        stationNumber = 0
    cost_matrix = np.delete(cost_matrix, indexes, axis=1)
    cost_matrix = np.delete(cost_matrix, indexes, axis=0)
    return len(cost_matrix), cost_matrix

#calculate total cost of a given route (fitness)
def total_cost(route, cost_matrix):
    total_cost = 0
    for i in range(len(route) - 1):
        total_cost += float(cost_matrix[route[i], route[i + 1]])
    total_cost += float(cost_matrix[route[-1], route[0]])
    return total_cost

#generate an initial population (heuristic)
def generate_population(pop_size, n_cities):
    population = [np.random.permutation(n_cities) for _ in range(pop_size)]
    return population

#evaluate fitness of population
def evaluate_population(population, matrix):
    fitness = [total_cost(route, matrix) for route in population]
    return np.array(fitness)

def three_transport_evaluate_population(population, matrices):
    fitness = [three_transport_total_cost(route, matrix) for route in population]
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

def single_transport_optimization(matrix1, type, pop_size, n_generations, tournament_size, crossover_rate, mutation_rate):
#first we delete the collumns that have nothing in them (this is done for the singular transport problems, this should not happen on the three transport problem)
    if type == "cost":
        n_cities, matrix1 = trim_cost_matrix(matrix1)
    elif type == "time":
        n_cities, matrix1 = trim_time_matrix(matrix1)

    population = generate_population(pop_size, n_cities)
    
    fitness = evaluate_population(population, matrix1)
    
    # Step 3: Evolution loop
    for generation in range(n_generations):
        # Step 4: Select parents using tournament selection
        parents = tournament_selection(population, fitness, tournament_size)
        
        # Step 5: Create offspring using crossover and mutation
        offspring = create_offspring(parents, crossover_rate, mutation_rate)
        
        # Step 6: Evaluate the new offspring
        fitness_offspring = evaluate_population(offspring, matrix1)
        
        # Step 7: Combine population and offspring
        combined_population = population + offspring
        combined_fitness = np.concatenate((fitness, fitness_offspring))
        
        # Step 8: Select the best individuals for the next generation
        best_indices = np.argsort(combined_fitness)[:pop_size]
        population = [combined_population[i] for i in best_indices]
        fitness = combined_fitness[best_indices]
        
        # Optionally, print the best fitness every generation
        print(f"Generation {generation + 1}, Best Fitness: {fitness[0]}")

    return population[0], fitness[0]

def three_transport_optimization(matrix1, matrix2, matrix3, type, pop_size, n_generations, tournament_size, crossover_rate, mutation_rate):
    
    matrices = [matrix1, matrix2, matrix3]    
    n_cities = []
    cities = 0

    for i in range(3):
        if type == "cost":
            cities, matrices[i] = trim_cost_matrix(matrices[i])
            n_cities.append(cities)
        elif type == "time":
        
            cities, matrices[i] = trim_time_matrix(matrices[i])
            n_cities.append(cities)
        population = generate_population(pop_size, n_cities)
    
    
    
    fitness = three_transport_evaluate_population(population, matrices)
    
    # Step 3: Evolution loop
    for generation in range(n_generations):
        # Step 4: Select parents using tournament selection
        parents = tournament_selection(population, fitness, tournament_size)
        
        # Step 5: Create offspring using crossover and mutation
        offspring = create_offspring(parents, crossover_rate, mutation_rate)
        
        # Step 6: Evaluate the new offspring
        fitness_offspring = evaluate_population(offspring, matrix1)
        
        # Step 7: Combine population and offspring
        combined_population = population + offspring
        combined_fitness = np.concatenate((fitness, fitness_offspring))
        
        # Step 8: Select the best individuals for the next generation
        best_indices = np.argsort(combined_fitness)[:pop_size]
        population = [combined_population[i] for i in best_indices]
        fitness = combined_fitness[best_indices]
        
        # Optionally, print the best fitness every generation
        print(f"Generation {generation + 1}, Best Fitness: {fitness[0]}")


#GA
def single_objective_genetic_algorithm(matrix1, matrix2, matrix3, type, pop_size=40, n_generations=250, tournament_size=3, crossover_rate=0.9, mutation_rate=0.1):

    if isinstance(matrix1, np.ndarray) == 0:
        print("Error loading matrix1")
        exit(0)
    elif (isinstance(matrix1, np.ndarray) != 0) & (isinstance(matrix2, np.ndarray) == 0) & (isinstance(matrix3, np.ndarray) == 0):
        best_solution, best_fitness = single_transport_optimization(matrix1, type, pop_size, n_generations, tournament_size, crossover_rate, mutation_rate)
    elif (isinstance(matrix1, np.ndarray) != 0) & (isinstance(matrix2, np.ndarray) != 0) & (isinstance(matrix2, np.ndarray) != 0):
        three_transport_optimization(matrix1, matrix2, matrix3, type, pop_size, n_generations, tournament_size, crossover_rate, mutation_rate)
    else:
        print("invalid matrix composition. Send only matrix 1 for single transport optimization or all 3 for 3 transport optimization")
        exit(0)
    
    return best_solution, best_fitness

#read the xy matrix1
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
print(len(cost_plane))
best_route_plane, best_cost_plane = single_objective_genetic_algorithm(time_plane, 0, 0, "time")
# Print the best route and cost found
print("Best Route (Plane):", best_route_plane)
print("Best Total Cost (Plane):", best_cost_plane)


###Cost
#Bus

