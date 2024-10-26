import pandas as pd
import numpy as np
import time
import random

#function to load the cost and time functions of the transport methods
def loadMatrix(filename):
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
                
    return df.values

#trims matrixes to remove unwanted cities
def trimMatrix(matrix, type):
    indexes = []
    n_stations = 0
    min_stations = 5

    #change maxValue based on type
    if type == "cost":
        max_value = 1e6
    elif type == "time":
        max_value = 12e3
    else:
        print("invalid type, please use a valid type")
        exit(1)

    #iterate over columns
    for col in range(matrix.shape[1]):
        #iterate over rows
        for row in range(matrix.shape[0]):
            #get the value and see if its lower than maxValue, if it is then add 1 to n_stations
            value = matrix[row, col]
            if value < max_value:
                n_stations += 1
        #if a given city has less than minStations stations with valid times, it gets deleted
        if n_stations < min_stations:
            indexes.append(col)
        n_stations = 0
    matrix = np.delete(matrix, indexes, axis=1)
    matrix = np.delete(matrix, indexes, axis=0)
    return len(matrix), matrix

#generate population based on population size and number of cities
def generatePopulation(pop_size, n_cities):
    population = [np.random.permutation(n_cities) for _ in range(pop_size)]
    return population

#calculate the route cost (fitness of a route using the cost matrix)
def routeCost(route, matrix):
    route_cost = 0

    #sum cost values between the cities in the route
    for i in range(len(route) - 1):
        route_cost += float(matrix[route[i], route[i + 1]])
    route_cost += float(matrix[route[-1], route[0]])

    #return the total route cost
    return route_cost

#evaluate fitness of a population
def evaluatePopulation(matrix, population):
    fitness = [routeCost(route, matrix) for route in population]
    return np.array(fitness)

def tournamentSelection(population, fitness, tournament_size=2):
    selected = []
    participants = [0, 0]
    tournament_length = int(len(population)/2)

    for _ in range(tournament_length):
        participants = np.random.choice(len(population), tournament_size)
        while participants[0] == participants[1]:
            participants = np.random.choice(len(population), tournament_size)
        best = participants[np.argmin(fitness[participants])]
        selected.append(population[best])
        population = np.delete(population, participants, axis=0)

    return selected

#crossover parents with uniform crossover
def uniformCrossover(parents, popLen, probability):
    parentSize = len(parents[0])
    offsprings = np.zeros((popLen, parentSize), dtype=int)

    #create kids
    for i in range(0, int(popLen/2), 2):
        for j in range(parentSize):
            randomNumber = random.random()
            if probability >= randomNumber:
                offsprings[i][j] = parents[i][j]
                offsprings[i+1][j] = parents[i+1][j]
            else:
                offsprings[i][j] = parents[i+1][j]
                offsprings[i+1][j] = parents[i][j]
    
    return offsprings  

def orderCrossover1(parents, pop_len):
    
    parents_size = len(parents[0])
    offsprings = np.full((int(pop_len/2), parents_size), 1e6, dtype=int)
    
    #generate lower bound and upper bound so that a random segment of half the parent's length is selected
    lb = int(random.randint(0, int(parents_size / 2)))
    ub = int(lb + parents_size / 2)
    
    #iterate for every child
    for i in range(0, int(pop_len/2)):
        #select parents
        parent1 = parents[i]
        parent2 = parents[(i + 1) % len(parents)]

        #place parent1 values in child
        offsprings[i, lb:ub] = parent1[lb:ub]
        
        #this will be used to know what elements from each parent are aleady in the child
        existing_elements1 = set(offsprings[i, lb:ub])
        
        #now fill the rest of the child with the other parent's values
        pos = 0
        for j in range(parents_size):
            #skip the part that has been filled already
            if pos == lb:  
                pos = ub
            #put second parent's values in child while verifying that they arent already in the child
            if parent2[j] not in existing_elements1:
                offsprings[i, pos] = parent2[j]
                pos += 1

    return offsprings

def orderCrossover2(parents, pop_len):
    
    parents_size = len(parents[0])
    offsprings = np.full((pop_len, parents_size), 1e6, dtype=int)
    
    #generate lower bound and upper bound so that a random segment of half the parent's length is selected
    lb = int(random.randint(0, int(parents_size / 2)))
    ub = int(lb + parents_size / 2)
    
    #iterate for every child
    for i in range(0, int(pop_len/2)):
        #select parents
        parent1 = parents[i]
        parent2 = parents[(i + 1) % len(parents)]

        #place parent1 values in child
        offsprings[(i*2), lb:ub] = parent1[lb:ub]
        #place parent2 values in next child
        offsprings[(i*2)+1, lb:ub] = parent2[lb:ub]
        
        #this will be used to know what elements from each parent are aleady in the child
        existing_elements1 = set(offsprings[(i*2), lb:ub])
        existing_elements2 = set(offsprings[(i*2)+1, lb:ub])
        
        #now fill the rest of the child with the other parent's values
        pos = 0
        for j in range(parents_size):
            #skip the part that has been filled already
            if pos == lb:  
                pos = ub
            #put second parent's values in child while verifying that they arent already in the child
            if parent2[j] not in existing_elements1:
                offsprings[(i*2), pos] = parent2[j]
                pos += 1
        pos = 0
        for j in range(parents_size):
            #skip the part that has been filled already
            if pos == lb:  
                pos = ub
            #put second parent's values in child while verifying that they arent already in the child
            if parent1[j] not in existing_elements2:
                offsprings[(i*2)+1, pos] = parent1[j]
                pos += 1

    return offsprings

def swapMutation(offsprings, probability):
    pop_size = len(offsprings)
    offspring_size = len(offsprings[0])

    #iterate for every child
    for i in range(pop_size):
        random_number = random.random()
        for j in range(0, offspring_size, 2):
            if random_number > probability:
                break
            temp = offsprings[i][j]
            offsprings[i][j] = offsprings[i][(j+1) % offspring_size]
            offsprings[i][(j+1) % offspring_size] = temp

    return offsprings

#GA for single transport SOO  
def SingleTransportOptimization(matrix, type, pop_size, n_generations):
    
    #first we discard unwanted cities (cities with low number of stations)
    n_cities, matrix = trimMatrix(matrix, type)

    #now we generate a population with the trimmed matrix
    population = generatePopulation(pop_size, n_cities)

    #evaluate the fitness of the starting population
    fitness = evaluatePopulation(matrix, population)

    #now we begin the genetic algorithm loop
    for generation in range(n_generations):
        #select the parents
        parents = tournamentSelection(population, fitness)

        #create offspring
        offsprings = orderCrossover1(parents, len(population))

        #apply mutation to offspring
        offsprings = swapMutation(offsprings, probability=0.5)

        #evaluate offspring fitness
        offspring_fitness = evaluatePopulation(matrix, offsprings)

        #combine parents and kids
        combined_population = np.concatenate((population, offsprings))
        combined_fitness = np.concatenate((fitness, offspring_fitness))

        #get the best individuals, best individual is stored in population[0] and fitness[0]
        best_indices = np.argsort(combined_fitness)[:pop_size]
        population = [combined_population[i] for i in best_indices]
        fitness = combined_fitness[best_indices]

        #print(f"Generation {generation + 1}, Best Fitness: {fitness[0]}")

    return population[0], fitness[0]
    
#calculate the route cost based on the three matrices, alwats choosing the smallest value out of the three
def ThreeTransportRouteCost(route, matrices):
    route_cost = 0

    #for each part of the route, we calculate the three fitnesses from the matrices, then see which is the smallest and use that
    for i in range(len(route) - 1):
        #get the three costs
        costs = [float(matrices[j][route[i], route[i + 1]]) for j in range(3)]
        #sum the lowest cost to the route_cost
        route_cost += min(costs)
    #repeat for the first value
    costs = [float(matrices[j][route[-1], route[0]]) for j in range(3)]
    route_cost += min(costs)

    return route_cost

#evaluate fitness of a population based on the three transports
def ThreeTransportEvaluatePopulation(matrices, population):
    fitness = [ThreeTransportRouteCost(route, matrices) for route in population]
    return np.array(fitness)

#GA for three transport SOO  
def ThreeTransportOptimization(matrix1, matrix2, matrix3, type, pop_size, n_generations):

    matrices = np.array([matrix1, matrix2, matrix3])
    n_cities = 50

    #now we generate a population
    population = generatePopulation(pop_size, n_cities)

    #evaluate the fitness of the three starting populations at the same time
    fitness = ThreeTransportEvaluatePopulation(matrices, population)

    #now we begin the genetic algorithm loop
    for generation in range(n_generations):
        #select the parents
        parents = tournamentSelection(population, fitness)

        #create offspring
        offsprings = orderCrossover1(parents, len(population))

        #evaluate offspring fitness
        offspring_fitness = ThreeTransportEvaluatePopulation(matrices, offsprings)

        #combine parents and kids
        combined_population = np.concatenate((population, offsprings))
        combined_fitness = np.concatenate((fitness, offspring_fitness))

        #get the best individuals, best individual is stored in population[0] and fitness[0]
        best_indices = np.argsort(combined_fitness)[:pop_size]
        population = [combined_population[i] for i in best_indices]
        fitness = combined_fitness[best_indices]

        #print(f"Generation {generation + 1}, Best Fitness: {fitness[0]}")

    return population[0], fitness[0]

#given a time or cost matrix or set of 3 matrices, use a genetic algorithm to find an optimal route (minimize time or cost)
def SingleObjectiveGeneticAlgorithm(matrix1, matrix2, matrix3, type, pop_size=50, n_generations=2000):
    if isinstance(matrix1, np.ndarray) == 0:
        print("Error loading matrix1")
        exit(1)
    elif (isinstance(matrix1, np.ndarray) != 0) & (isinstance(matrix2, np.ndarray) == 0) & (isinstance(matrix3, np.ndarray) == 0):
        best_solution, best_fitness = SingleTransportOptimization(matrix1, type, pop_size, n_generations)
    elif (isinstance(matrix1, np.ndarray) != 0) & (isinstance(matrix2, np.ndarray) != 0) & (isinstance(matrix2, np.ndarray) != 0):
        best_solution, best_fitness = ThreeTransportOptimization(matrix1, matrix2, matrix3, type, pop_size, n_generations)
    else:
        print("invalid matrix composition. Send only matrix 1 for single transport optimization or all 3 for 3 transport optimization")
        exit(1)

    return best_solution, best_fitness