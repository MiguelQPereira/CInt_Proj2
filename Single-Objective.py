#from CInte_Functions import load_matrix
import CInte_Functions as ci

#bus matrices
cost_bus = ci.loadMatrix("Data_Sets/costbus.csv")
time_bus = ci.loadMatrix("Data_Sets/timebus.csv")
#plane matrices
cost_plane = ci.loadMatrix("Data_Sets/costplane.csv")
time_plane = ci.loadMatrix("Data_Sets/timeplane.csv")
#train matrices
cost_train = ci.loadMatrix("Data_Sets/costtrain.csv")
time_train = ci.loadMatrix("Data_Sets/timetrain.csv")


############## Single Objective ##############

###########one transport
###costs
cost_plane_route, cost_plane_fitness = ci.SingleObjectiveGeneticAlgorithm(cost_plane, 0, 0, "cost")

print("Final route (plane):", cost_plane_route)
print("Final cost (plane):", cost_plane_fitness)

cost_train_route, cost_train_fitness = ci.SingleObjectiveGeneticAlgorithm(cost_train, 0, 0, "cost")

print("Final route (train):", cost_train_route)
print("Final cost (train):", cost_train_fitness)

cost_bus_route, cost_bus_fitness = ci.SingleObjectiveGeneticAlgorithm(cost_bus, 0, 0, "cost")

print("Final route (bus):", cost_bus_route)
print("Final cost (bus):", cost_bus_fitness)

###times
time_plane_route, time_plane_fitness = ci.SingleObjectiveGeneticAlgorithm(time_plane, 0, 0, "time")

print("Final route (plane):", time_plane_route)
print("Final time (plane):", time_plane_fitness)

time_train_route, time_train_fitness = ci.SingleObjectiveGeneticAlgorithm(time_train, 0, 0, "time")

print("Final route (train):", time_train_route)
print("Final time (train):", time_train_fitness)

time_bus_route, time_bus_fitness = ci.SingleObjectiveGeneticAlgorithm(time_bus, 0, 0, "time")

print("Final route (bus):", time_bus_route)
print("Final time (bus):", time_bus_fitness)

###########three transport
###costs
cost_three_transport_route, cost_three_transport_fitness = ci.SingleObjectiveGeneticAlgorithm(cost_plane, cost_train, cost_bus, "cost")

print("Final route (three transport):", cost_three_transport_route)
print("Final cost (three transport):", cost_three_transport_fitness)

time_three_transport_route, time_three_transport_fitness = ci.SingleObjectiveGeneticAlgorithm(time_plane, time_train, time_bus, "time")

print("Final route (three transport):", time_three_transport_route)
print("Final time (three transport):", time_three_transport_fitness)
