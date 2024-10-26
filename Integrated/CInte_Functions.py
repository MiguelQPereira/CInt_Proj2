import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
random.seed(42) 
np.random.seed(42)

#function to load the cost and time functions of the transport methods
def loadMatrix(filename, obj):
    #read csv with the filename
    df = pd.read_csv(filename, index_col=0)

    #replace the dashes with insanely high number, so that these options are immediately discarted

    if (obj == "cost"):
        df.replace("-", 1e6, inplace=True)
        for col_name in df.columns:
            df[col_name] = df[col_name].astype(float)  # Convert the column to float
    elif (obj == "time"):
        df.replace("-", "200", inplace=True)
    elif (obj == "not applicable"):
        #return the dataframe without the country
        df = pd.read_csv(filename)
        return df.drop(columns=df.columns[1])
                
    return df.values

#trims matrixes to remove unwanted cities
def trimMatrix(matrix, n_cities, type):
    indexes = []

    #change maxValue based on type
    if type == "cost":
        max_value = 1e6
    elif type == "time":
        max_value = 2e2
    else:
        print("invalid type, please use a valid type")
        exit(1)

    #iterate over columns
    for col in range(matrix.shape[1]):
        empty = True
        #iterate over rows
        for row in range(matrix.shape[0]):
            value = float(matrix[row, col])
            if value < max_value:
                #if any value is less than max, then its not an empty city
                empty = False
        
        #after checking all of a city's values, see if its emtpy, if it is then place its index in the indexes variable
        if empty == True:
            indexes.append(col)

    #delete empty cities from the matrix
    matrix = np.delete(matrix, indexes, axis=1)
    matrix = np.delete(matrix, indexes, axis=0)

    #if the matrix has less cities than the number of cities specified, give error
    if len(matrix) < n_cities:
        print(f"error, matrix does not have {n_cities} non-empty cities, please lower the number of cities (-s n_cities)")
        exit(1)
    #if the matrix still has more cities than specified by the user, delete the last 
    elif len(matrix) > n_cities:
        n_columns = len(matrix) - n_cities
        matrix = matrix[:, :-n_columns]
        
    return matrix   

def createTSPMap(xy, route, n_cities):
    if n_cities < 50:
        xy = xy.iloc[:-(50-n_cities)]
        
    plt.figure(figsize=(12, 8))
    m = Basemap(projection='mill', llcrnrlat=30, urcrnrlat=70, llcrnrlon=-15, urcrnrlon=45, resolution='l')  
    m.drawcoastlines()
    m.drawcountries()

    x, y = m(xy['longitude'].values, xy['latitude'].values)
    m.scatter(x, y, s=50, color='red', marker='o', zorder=5)

    for i in range(len(route) - 1):
        # Get the indices of the two cities to connect
        start_idx = route[i]
        end_idx = route[i + 1]
        
        # Get the coordinates for these cities
        x_start, y_start = x[start_idx], y[start_idx]
        x_end, y_end = x[end_idx], y[end_idx]
        
        # Plot the line
        m.plot([x_start, x_end], [y_start, y_end], linestyle='--', color='blue', linewidth=2, zorder=4)

    m.plot([x[route[0]], x[route[-1]]], [y[route[0]], y[route[-1]]], linestyle='--', color='blue', linewidth=2, zorder=4)

    # Annotate cities with names
    for i, city in enumerate(xy['city']):
        plt.text(x[i], y[i], city, fontsize=12, ha='right')

    plt.title("TSP Graph")
    plt.show()

#generate population based on population size and number of cities
def generatePopulation(n_cities, pop_size):
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

def orderCrossover(parents, pop_len):
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
        existing_elements = set(offsprings[i, lb:ub])
        
        #now fill the rest of the child with the other parent's values
        pos = 0
        for j in range(parents_size):
            #skip the part that has been filled already
            if pos == lb:  
                pos = ub
            #put second parent's values in child while verifying that they arent already in the child
            if parent2[j] not in existing_elements:
                offsprings[i, pos] = parent2[j]
                pos += 1

    return offsprings

#GA for single transport SOO  
def SingleTransportOptimization(matrix, type, transport, n_cities, pop_size, n_generations):
            
    #first we discard unwanted cities (cities with low number of stations)
    matrix = trimMatrix(matrix, n_cities, type)

    #now we generate a population with the trimmed matrix
    population = generatePopulation(n_cities, pop_size)

    #evaluate the fitness of the starting population
    fitness = evaluatePopulation(matrix, population)

    #now we begin the genetic algorithm loop
    for generation in range(n_generations):
        #select the parents
        parents = tournamentSelection(population, fitness)

        #create offspring
        offsprings = orderCrossover(parents, len(population))

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
def ThreeTransportOptimization(matrix1, matrix2, matrix3, n_cities, pop_size, n_generations):

    #remove the last cities of each matrix according to n_cities
    if n_cities < 50:
        matrix1 = matrix1[:, :-(50-n_cities)]
        matrix2 = matrix2[:, :-(50-n_cities)]
        matrix3 = matrix3[:, :-(50-n_cities)]

    #group all 3 matrices
    matrices = np.array([matrix1, matrix2, matrix3])

    #now we generate a population
    population = generatePopulation(n_cities, pop_size)

    #evaluate the fitness of the three starting populations at the same time
    fitness = ThreeTransportEvaluatePopulation(matrices, population)

    #now we begin the genetic algorithm loop
    for generation in range(n_generations):
        #select the parents
        parents = tournamentSelection(population, fitness)

        #create offspring
        offsprings = orderCrossover(parents, len(population))

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
def SingleObjectiveGeneticAlgorithm(matrix1, matrix2, matrix3, xy, type, transport, n_cities, pop_size=50, n_generations=250):
    if isinstance(matrix1, np.ndarray) == 0:
        print("Error loading matrix1")
        exit(1)
    elif (isinstance(matrix1, np.ndarray) != 0) & (isinstance(matrix2, np.ndarray) == 0) & (isinstance(matrix3, np.ndarray) == 0):
        best_solution, best_fitness = SingleTransportOptimization(matrix1, type, transport, n_cities, pop_size, n_generations)
    elif (isinstance(matrix1, np.ndarray) != 0) & (isinstance(matrix2, np.ndarray) != 0) & (isinstance(matrix2, np.ndarray) != 0):
        best_solution, best_fitness = ThreeTransportOptimization(matrix1, matrix2, matrix3, n_cities, pop_size, n_generations)
    else:
        print("invalid matrix composition. Send only matrix 1 for single transport optimization or all 3 for 3 transport optimization")
        exit(1)
 
    #report the results in terminal
    if type == "cost":
        if transport == "bus":
            print(f"Bus final route ({n_cities} cities): {best_solution}")
            print(f"Bus final cost: €{round(best_fitness, 2)}")
        elif transport == "train":
            print(f"Train final route ({n_cities} cities): {best_solution}")
            print(f"Train final cost: €{round(best_fitness, 2)}")
        elif transport == "plane":
            print(f"Plane final route ({n_cities} cities): {best_solution}")
            print(f"Plane final cost: €{round(best_fitness, 2)}")
        elif transport == "all":
            print(f"All transports final route ({n_cities} cities): {best_solution}")
            print(f"All transports final cost: €{round(best_fitness, 2)}")
    elif type == "time":
        if transport == "bus":
            print(f"Bus final route ({n_cities} cities): {best_solution}")
            print(f"Bus final time: {int(best_fitness)}h")
        elif transport == "train":
            print(f"Train final route ({n_cities} cities): {best_solution}")
            print(f"Train final time: {int(best_fitness)}h")
        elif transport == "plane":
            print(f"Plane final route ({n_cities} cities): {best_solution}")
            print(f"Plane final time: {int(best_fitness)}h")
        elif transport == "all":
            print(f"All transports final route ({n_cities} cities): {best_solution}")
            print(f"All transports final time: {int(best_fitness)}h")
    else:
        print(f"invalid type {type}, please select a valid type (cost, time)")
        exit(1)

    print("#####################################\n")

    #plot the map
    createTSPMap(xy, best_solution, n_cities)

    return best_solution, best_fitness


##############################################################################
##############################################################################
##############################################################################

def multiEvaluationSingle(matrix1,matrix2, population):
    fitness1 = [routeCost(route, matrix1) for route in population]
    fitness2 = [routeCost(route, matrix2) for route in population]
    fitness = [[f1, f2] for f1, f2 in zip(fitness1, fitness2)]
    return fitness

def multiThreeTransportEvaluatePopulation(matrices1, matrices2, population):
    fitness1 = [ThreeTransportRouteCost(route, matrices1) for route in population]
    fitness2 = [ThreeTransportRouteCost(route, matrices2) for route in population]
    fitness = [[f1, f2] for f1, f2 in zip(fitness1, fitness2)]
    return fitness

def pareto_fronts(population, fitness_scores):
    fronts = []
    remaining_indices = set(range(len(fitness_scores)))

    while remaining_indices:
        front = []
        for i in remaining_indices:
            score = np.array(fitness_scores[i])
            dominated = False
            for j in remaining_indices:
                if i == j:
                    continue  # Skip comparing the individual with itself
                other_score = np.array(fitness_scores[j])
                # Check if `other_score` dominates `score`
                if np.all(other_score <= score) and np.any(other_score < score):
                    dominated = True
                    break
            if not dominated:
                front.append(i)

        fronts.append(front)
        remaining_indices -= set(front)  # Remove individuals in the current front from remaining
    return fronts

def findExtremities(front, scores):
    min_idx = front[0]
    max_idx = front[0]
    for idx in front:
        if scores[idx][1] < scores[min_idx][1]:  # Update min based on the score's second objective
            min_idx = idx
        if scores[idx][1] > scores[max_idx][1]:  # Update max based on the score's second objective
            max_idx = idx

    return [min_idx, max_idx]

def computeCuboids (front, scores):
    combined = list(zip([scores[idx][0] for idx in front], [scores[idx][1] for idx in front], front))

    # Sort combined based on the first vector (vector1)
    combined.sort(key=lambda x: x[0])  # Sort by the first element of the tuples

    # Unzip the sorted combined list back into separate vectors
    sorted_cost, sorted_time, sorted_front = zip(*combined)

    sorted_front = list(sorted_front)
    sorted_cost = list(sorted_cost)
    sorted_time = list(sorted_time)
    cuboide = []
    for i in range(len(sorted_front)):
        if i == 0 or i == len(sorted_front) - 1:
            cuboide.append(1000000)  # Append boundary value for edges
        else:
            cuboide.append((sorted_cost[i+1] - sorted_cost[i-1]) * (sorted_time[i+1] - sorted_time[i-1]))
        
    return cuboide

def nsga_ii_selection(population, fitness_scores, cities): 
    """
    Implements NSGA-II selection for multi-objective optimization.
    """
    n_parents = cities
    fronts = pareto_fronts(population, fitness_scores)
    selected_indices = []

    for front in fronts:
        if len(front) <= n_parents - len(selected_indices):
            selected_indices.extend(front)
            break

        extremes = findExtremities(front, fitness_scores)
        
        if n_parents - len(selected_indices) == 1:
            selected_indices.append(extremes[random.randint(0, 1)])
        elif n_parents - len(selected_indices) == 2:
            selected_indices.extend(extremes)
        else:
            missing_elements = n_parents - len(selected_indices)
            cuboids = computeCuboids(front, fitness_scores)
            combined = list(zip(cuboids, [fitness_scores[idx][0] for idx in front], [fitness_scores[idx][1] for idx in front], front))

            # Sort combined based on the first vector (vector1)
            combined.sort(key=lambda x: x[0], reverse=True)  # Sort by the first element of the tuples

            # Unzip the sorted combined list back into separate vectors
            sorted_cuboids, sorted_cost, sorted_time, sorted_front = zip(*combined)
            sorted_cuboids = list(sorted_cuboids)
            sorted_front = list(sorted_front)
            sorted_cost = list(sorted_cost)
            sorted_time = list(sorted_time)

            selected_indices.extend(sorted_front[0:missing_elements])

    selected_population = [population[i] for i in selected_indices]
    return selected_population

def SingleTransportMultiOptimization(matrix1, matrix2, cities, n_generations):
    cost1 = trimMatrix(matrix1, cities, 'cost')
    cost2 = trimMatrix(matrix2, cities, 'time')
    
    population = generatePopulation(cities, 50)
    pop_size = 50

    # Evaluate the fitness of the starting population
    fitness = multiEvaluationSingle(cost1, cost2, population)
    for generation in range(n_generations):
        # Select the parents
        parents = nsga_ii_selection(population, fitness, cities//2)
        # Create offspring
        offsprings = orderCrossover(parents, len(parents))

        # Evaluate offspring fitness
        offspring_fitness = multiEvaluationSingle(cost1, cost2, offsprings)
        # Combine parents and offspring
        combined_population = population.copy()
        combined_population.extend(offsprings)
        combined_fitness = fitness.copy()
        combined_fitness.extend(offspring_fitness)

        # Get the best individuals
        # Get the best individuals using NSGA-II selection
        population = nsga_ii_selection(combined_population, combined_fitness, cities)
        
        # Recompute fitness for the new population
        fitness = multiEvaluationSingle(cost1, cost2, population)
        

        print(f"Generation {generation + 1}, Best Fitness: {fitness[0]}")
    
    return population[0], fitness[0]

def ThreeTransportMultiOptimization(matrix1, matrix2, matrix3, matrix4, matrix5, matrix6, cities, n_generations):
    #remove the last cities of each matrix according to n_cities
    if cities < 50:
        matrix1 = matrix1[:, :-(50-cities)]
        matrix2 = matrix2[:, :-(50-cities)]
        matrix3 = matrix3[:, :-(50-cities)]
        matrix4 = matrix4[:, :-(50-cities)]
        matrix5 = matrix5[:, :-(50-cities)]
        matrix6 = matrix6[:, :-(50-cities)]

    #group all 3 matrices
    matrices1 = [matrix1, matrix3, matrix5]
    matrices2 = [matrix2, matrix4, matrix6]

    #now we generate a population
    population = generatePopulation(cities, 50)

    #evaluate the fitness of the three starting populations at the same time
    fitness = multiThreeTransportEvaluatePopulation(matrices1, matrices2, population)

    for generation in range(n_generations):
        # Select the parents
        parents = nsga_ii_selection(population, fitness, cities//2)
        # Create offspring
        offsprings = orderCrossover(parents, len(parents))

        # Evaluate offspring fitness
        offspring_fitness = multiThreeTransportEvaluatePopulation(matrices1, matrices2, offsprings)
        # Combine parents and offspring
        combined_population = population.copy()
        combined_population.extend(offsprings)
        combined_fitness = fitness.copy()
        combined_fitness.extend(offspring_fitness)

        # Get the best individuals
        # Get the best individuals using NSGA-II selection
        population = nsga_ii_selection(combined_population, combined_fitness, cities)
        
        # Recompute fitness for the new population
        fitness = multiThreeTransportEvaluatePopulation(matrices1, matrices2, population)

    return population[0], fitness[0]

def MultiObjectiveGeneticAlgorithm(matrix1, matrix2, matrix3, matrix4, matrix5, matrix6, pop_size=50, n_generations=250):
    if isinstance(matrix1, np.ndarray) == 0:
        print("Error loading matrix1")
        exit(1)
    elif (isinstance(matrix1, np.ndarray) != 0) & (isinstance(matrix2, np.ndarray) != 0) & (isinstance(matrix3, np.ndarray) == 0) & (isinstance(matrix4, np.ndarray) == 0) & (isinstance(matrix5, np.ndarray) == 0)& (isinstance(matrix6, np.ndarray) == 0):
        best_solution, best_fitness = SingleTransportMultiOptimization(matrix1, matrix2, pop_size, n_generations)
    elif (isinstance(matrix1, np.ndarray) != 0) & (isinstance(matrix2, np.ndarray) != 0) & (isinstance(matrix3, np.ndarray) != 0) & (isinstance(matrix4, np.ndarray) != 0) & (isinstance(matrix5, np.ndarray) != 0)& (isinstance(matrix6, np.ndarray) != 0):
        best_solution, best_fitness = ThreeTransportMultiOptimization(matrix1, matrix2, matrix3, matrix4, matrix5, matrix6, pop_size, n_generations)
    else:
        print("invalid matrix composition. Send matrix 2 for single transport optimization or all 6 for 3 transport optimization")
        exit(1) 

    print("Final route:", best_solution)
    print("Final objective:", best_fitness)

    return best_solution, best_fitness