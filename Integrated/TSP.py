import sys
import os

import CInte_Functions as ci


# Initialize default values
PROBLEM = 'all'  # Defines wich problem to solve
DATA = '' # Defines the directory of the Data_set
SIZE = 0 # Defines the size of the problems to solve; if 0 solve for 10, 30 and 50;
OBJECTIVE = "all" # Defines to wich objectives the problem will be solved
TRANSPORT = "all" # Defines the means of transportation available

# kip the script name (sys.argv[0])
args = sys.argv[1:]
i = 0
while i < len(args):
    arg = args[i]
    
    # Check for flags or named arguments
    if arg in ("-p", "--problem"):
        i += 1
        if (args[i] in ("all", "single", "multi")):
            PROBLEM = args[i]
        else:
            print("Error: Expected \"all\", \"cost\" or \"time\" after -o or --objective")
            sys.exit(1)
    elif arg in ("-d", "--data"):
        i += 1  # Move to the next item in the list for the value
        if os.path.isdir(args[i]):
            DATA = str(args[i])
        else:
            print(f"Error: '{args[i]}' is not a valid folder path.")
    elif arg in ("-s", "--size"):
        i += 1
        if i < len(args) and args[i].isdigit():  # Check if there's a next item and if it's a number
            SIZE = int(args[i])
        else:
            print("Error: Expected an integer after -s or --size")
            sys.exit(1)
    elif arg in ("-o", "-objective"):
        i += 1
        if (args[i] in ("all", "cost", "time")):
            OBJECTIVE = args[i]
        else:
            print("Error: Expected \"all\", \"cost\" or \"time\" after -o or --objective")
            sys.exit(1)
    elif arg in ("-t", "-transport"):
        i += 1
        if (args[i] in ("all", "bus", "train", "plane")):
            TRANSPORT = args[i]
        else:
            print("Error: Expected \"all\", \"bus\", \"train\" or \"plane\" after -o or --objective")
            sys.exit(1)
    else:
        print(f"Unknown argument: {arg}")
        sys.exit(1)
    
    i += 1

#load Data

#xy coordinates matrix
xy = ci.loadMatrix(f"{DATA}/xy.csv", "not applicable")

#bus matrices
cost_bus = ci.loadMatrix(f"{DATA}/costbus.csv", 'cost')
time_bus = ci.loadMatrix(f"{DATA}/timebus.csv", 'time')
#plane matrices
cost_plane = ci.loadMatrix(f"{DATA}/costplane.csv", 'cost')
time_plane = ci.loadMatrix(f"{DATA}/timeplane.csv", 'time')
#train matrices
cost_train = ci.loadMatrix(f"{DATA}/costtrain.csv", 'cost')
time_train = ci.loadMatrix(f"{DATA}/timetrain.csv", 'time')

if (PROBLEM == 'all'):
    problem = ['single', 'multi']
else:
    problem = [PROBLEM]
if (SIZE == 0):
    size = [10, 30, 50]
else:
    size = [SIZE]
if (OBJECTIVE == 'all' and PROBLEM != 'multi'):
    objective = ['cost', 'time']
else:
    objective = [OBJECTIVE]
if (TRANSPORT == 'all'):
    transport = ['all']
else:
    transport = [TRANSPORT]

# Start solving the problems
for p in problem:
    for s in size:
        for o in objective:
            for t in transport:
                if p == 'single':
                    if o == 'cost':
                        if t == 'bus':
                            ci.SingleObjectiveGeneticAlgorithm(cost_bus, 0, 0, xy, o, "bus", s)
                        elif t == 'train':
                            ci.SingleObjectiveGeneticAlgorithm(cost_train, 0, 0, xy, o, "train", s)
                        elif t == 'plane':
                            ci.SingleObjectiveGeneticAlgorithm(cost_plane, 0, 0, xy, o, "plane", s)
                        else:
                            ci.SingleObjectiveGeneticAlgorithm(cost_plane, cost_train, cost_bus, xy, o, "all", s)
                    elif o == 'time':
                        if t == 'bus':
                            ci.SingleObjectiveGeneticAlgorithm(time_bus, 0, 0, xy, o, "bus", s)
                        elif t == 'train':
                            ci.SingleObjectiveGeneticAlgorithm(time_train, 0, 0, xy, o, "train", s)
                        elif t == 'plane':
                            ci.SingleObjectiveGeneticAlgorithm(time_plane, 0, 0, xy, o, "plane", s)
                        else:
                            ci.SingleObjectiveGeneticAlgorithm(time_plane, cost_train, cost_bus, xy, o, "all", s)
                elif (p == 'multi'):
                    if t == "bus":
                        ci.MultiObjectiveGeneticAlgorithm(cost_bus, time_bus, 0, 0, 0, 0, s)
                    elif t == 'train':
                        ci.MultiObjectiveGeneticAlgorithm(cost_train, time_train, 0, 0, 0, 0, s)
                    elif t == 'plane':
                        ci.MultiObjectiveGeneticAlgorithm(cost_plane, time_plane, 0, 0, 0, 0, s)
                    else:
                        ci.MultiObjectiveGeneticAlgorithm(cost_bus, time_bus, cost_train, time_train, cost_plane, time_plane, s)