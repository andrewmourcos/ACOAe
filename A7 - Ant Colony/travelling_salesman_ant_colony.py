import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math

import threading
import random
random.seed(0)


class TSPGraph():
    def __init__(self, city_coordinates):
        self.city_coordinates = city_coordinates
        self.num_cities = len(self.city_coordinates)

        self.distances = np.zeros( (self.num_cities, self.num_cities) )
        self.compute_distances()
    
    def compute_distances(self):
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                self.distances[i][j] = self.distance(i, j)

    def distance(self, cityA, cityB):
        """ Compute distance b/w any two cities """
        if (self.num_cities <= cityA < 0) or (self.num_cities <= cityB < 0):
            raise ValueError("Trying to find distance from city {} to city {}, but there are only {} cities!".format(cityA, cityB, self.num_cities))
        diff_vector = self.city_coordinates[cityA] - self.city_coordinates[cityB]
        return math.sqrt(diff_vector[0]**2 + diff_vector[1]**2)

    def check_solution(self, path):
        cost = 0
        prev_node = path[0]
        for node in path:
            cost += self.distance(prev_node, node)
            prev_node = node
        return cost


class AntColonyTSP():
    def __init__(self, city_graph: TSPGraph, population_size, diffusion_rate, evaporation_rate, num_iters):
        # Algorithm parameters
        self.population_size = population_size
        self.diffusion_rate = diffusion_rate
        self.evaporation_rate = evaporation_rate
        self.initial_pheromone = 10
        self.num_iters = num_iters
        # Environment parameters
        self.pheromones = np.full((city_graph.num_cities, city_graph.num_cities), self.initial_pheromone)
        self.city_graph = city_graph
        # Ant thread tracker
        self.ants = [Ants() for _ in range(self.population_size) ]
        self.ant_threads = [None for _ in range(self.population_size)]

        # Probability Look-up-Table to update on each iteration (based on pheromone update)
        self.probability_LUT = np.zeros( (self.city_graph.num_cities, self.city_graph.num_cities) )

        # Solution Tracker
        self.best_solution = []
        self.best_fitness = 10**10
        self.best_fitness_per_epoch = []
        self.average_fitness_per_epoch = []
        self.best_solution_per_epoch = []

    def update_probabilities_LUT(self):
        alpha = 1
        beta = 1
        tau_over_d = np.divide( np.linalg.matrix_power(self.pheromones, alpha), np.linalg.matrix_power(self.city_graph.distances, beta)+0.0000001 )
        
        for i in range(self.city_graph.num_cities):
            for j in range(self.city_graph.num_cities):
                self.probability_LUT[i][j] = tau_over_d[i][j] / np.sum(tau_over_d[i])

    def update_pheromones(self):
        # Evaporate
        self.pheromones = self.pheromones * self.evaporation_rate 

        # Using best solution from current iteration AND best overall solution to diffuse pheromones (similar to Stutzle and Hoos)
        prev_node = self.best_solution_per_epoch[0]
        for node in self.best_solution_per_epoch:
            self.pheromones[node][prev_node] += self.diffusion_rate * (self.pheromones[node][prev_node] / self.evaporation_rate)
            prev_node = node
        
        # prev_node = self.best_solution[0]
        # for node in self.best_solution:
        #     self.pheromones[node][prev_node] += self.diffusion_rate * (self.pheromones[node][prev_node] / self.evaporation_rate)
        #     prev_node = node

        

    def run(self):
        # Iterate
        for iter in range(self.num_iters):
            self.update_probabilities_LUT()
            
            # Instantiate ant population
            for i, ant in enumerate(self.ants):
                ant_thread = threading.Thread(target=ant.roundTrip, args=(random.randint(0, self.city_graph.num_cities-1),
                                                            self.city_graph.num_cities,
                                                            np.copy(self.probability_LUT),            # Ugly workaround to make thread-safe (numpy arrays can't be shared by threads w/o proper practice)
                                                            np.copy(self.city_graph.distances)))
                self.ant_threads[i] = ant_thread
                ant_thread.start()

            # Wait for all ants to complete
            fitnesses = []
            best_epoch_fitness = 10**10
            for i, ant in enumerate(self.ants):
                self.ant_threads[i].join()

                if ant.cost < best_epoch_fitness:
                    best_epoch_fitness = ant.cost
                    self.best_solution_per_epoch = ant.visited
                    if ant.cost < self.best_fitness:
                        self.best_fitness = ant.cost
                        self.best_solution = ant.visited
                fitnesses.append(ant.cost)
            self.best_fitness_per_epoch.append(best_epoch_fitness)
            self.average_fitness_per_epoch.append( sum(fitnesses)/len(fitnesses) )

            # Update pheromones
            self.update_pheromones()
            print(iter)
            # if iter % 30 == 1:
            #     plt.imshow(self.pheromones)
            #     plt.colorbar()
            #     plt.show()
        

import time
class Ants( AntColonyTSP ):
    def __init__(self):
        self.visited = []
        self.cost = 0

    def roundTrip(self, start_city, total_city_num, probability_LUT, distances):
        # Reset internal variables
        self.total_city_num = total_city_num
        self.visited = [start_city]
        self.cost = 0

        while len(self.visited) < self.total_city_num:
            curr_city = self.visited[-1]
            # Consider probability for each possible path
            probabilities = probability_LUT[curr_city]
            for v in self.visited:
                probabilities[v] = 0 # Eliminate possibility of taking a previously-seen path

            min_probabilities = np.min(probabilities)
            max_probabilities = np.max(probabilities)
            normalized_probabilities = (probabilities - min_probabilities)/(max_probabilities-min_probabilities+1E-10)
            normalized_probabilities /= normalized_probabilities.sum()
            
            # Roulette-wheel selection
            next_city = np.random.choice( self.total_city_num, p=normalized_probabilities )
            # Update self
            self.visited.append(next_city)
            self.cost += distances[curr_city][next_city]
        
        self.visited.append(start_city)
        self.cost += distances[self.visited[-1]][start_city]

if __name__ == "__main__":
    # Load Travelling Salesman Graph from Excel
    city_coordinates = pd.read_excel("A7 - Ant Colony/Assignment7-city coordinates.xlsx", index_col=0).to_numpy()
    
    city_graph = TSPGraph(city_coordinates) 
    # print( "distance: ", city_graph.distance(0, 5) )

    ACO = AntColonyTSP(city_graph=city_graph, population_size=20, diffusion_rate=0.2, evaporation_rate=0.3, num_iters=100)
    # Run ACO
    ACO.run()

    print("Best round-trip distance: ", ACO.best_fitness)
    print("Best path: ", ACO.best_solution)

    true_best = [1,28,6,12,9,5,26,29,3,2,20,10,4,15,18,17,14,22,11,19,25,7,23,27,8,24,16,13,21,1]
    print("Optimal solution of bays29: ", city_graph.check_solution([a-1 for a in true_best]) )

    print("yeet: ", city_graph.check_solution( [2, 28, 1, 20, 4, 25, 8, 5, 11, 27, 0, 23, 7, 26, 15, 22, 6, 24, 18, 12, 19, 9, 3, 14, 10, 21, 16, 17, 13] ) )

    plt.plot(ACO.average_fitness_per_epoch, label="AVG population fitness per epoch")
    plt.plot(ACO.best_fitness_per_epoch, label="Best population fitness per epoch")
    plt.legend()
    plt.xlabel("Epoch No.")
    plt.ylabel("Round trip distance")
    plt.show()

    pass