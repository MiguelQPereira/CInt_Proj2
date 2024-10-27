# CInt_Project2
Second project of Applied Computational Intelligence


Grupo 7:
Miguel Quina Pereira (103254)
Lucas Bargas e Leiradella (103566)

How to use the program:

single objective optimization:

-p or --porblem: 
Description: 
Type of problem (single or multi optimization)
Possible Values:
"single" or "multi"

-o or --objective:
Description:
What variable should be minimized (use only on single optimizaiton)
Possible values:
"cost", "time" or "all"
default = "all"

-d or --data:
Description:
path to a folder where all matrixes can be found (all cost and time datasets and xy, etc.)
Possible values:
Any path to a valid directory
default = ''

-s or --size: 
Description: 
number of cities for the problem (the data matrix must have at least the specified number of non empty cities)
Possible values:
any integer in [1; 50]
default value = 0 

-t or --transport:
Description: 
Select which kind of transport to use in the optimization
Possible values:
"bus", "train", "plane" or "all"
default = "all"


example uses:

single objective time plane with 30 cities
> python TSP.py -p single -s 30 -d ../Data_Sets/ -t plane -o time

single objective all transports with 50 cities
> python TSP.py -p single -s 50 -d ../Data_Sets/ -t all -o time

multi objective train with 50 cities:
> python TSP.py -p multi -s 50 -d ../Data_Sets/ -t train

multi objective all transports with 30 cities:
> python TSP.py -p multi -s 30 -d ../Data_Sets/ -t all

when a single objective optimization is done, an image of the map with the best route called "{transport}_{objective}" (example: train_time.png) will be saved to the directory and another image called "convergence_curves_{objective}" (example: convergence_curves_time.png). The terminal will write the best route and fitness 3 times before giving and average and STD.

when a multi objective optimization is done, the route maps of the 2 extremes will be saved as, for example: train_multi_extreme1.png and train_multi_extreme2.png. Another image called pareto_front.png will also be saved displaying the 1st and 2nd fronts.