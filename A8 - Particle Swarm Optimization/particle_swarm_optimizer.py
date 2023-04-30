"""
Using Particle Swarm Optimization to search the 2D Camelback function
"""

import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(2)

class Particle():
    def __init__(self, initial_position):
        self.position = initial_position
        self.curr_fitness = 100
        self.velocity = [0, 0]
        self.personal_best_position = initial_position
        self.personal_best_value = 100

    def update_velocity(self, W:float, C1:float, C2:float, global_best_position, max_speed):
        r1, r2 = random.random(), random.random()
        nextV_x = W*self.velocity[0] + C1*r1*(self.personal_best_position[0] - self.position[0]) + C2*r2*(global_best_position[0] - self.position[0])
        nextV_y = W*self.velocity[1] + C1*r1*(self.personal_best_position[1] - self.position[1]) + C2*r2*(global_best_position[1] - self.position[1])
        if nextV_x > max_speed:
            nextV_x = max_speed
        if nextV_y > max_speed:
            nextV_y = max_speed

        self.velocity = [nextV_x, nextV_y]

    def update_position(self, func):
        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]
        curr_score = func.evaluate(self.position[0], self.position[1])
        self.curr_fitness = curr_score
        if curr_score < self.personal_best_value:
            self.personal_best_value = curr_score
            self.personal_best_position = self.position

class ParticleSwarmOptimizer():
    def __init__(self, func, population_size, inertia, p_attraction, g_attraction, v_max, max_iters):
        self.function = func # Function to optimize
        # Algorithm Parameters
        self.pop_size = population_size
        self.inertia = inertia
        self.p_attraction = p_attraction
        self.g_attraction = g_attraction
        self.v_max = v_max
        self.max_iters = max_iters
        # Particle Tracker
        self.particles = []
        # Result Trackers
        self.g_best = 0
        self.g_best_position = [0, 0]
        self.avg_fitness_per_epoch = []
        self.best_fitness_per_epoch = []

    def init_particles(self):
        for _ in range(self.pop_size):
            starting_pos = [random.randrange(-50, 50)/10 for _ in range(2)] # Generate random X and Y positions
            particle = Particle(starting_pos)
            self.particles.append(particle)

    def run(self):
        # Initialize swarm with random starting points
        self.init_particles()
        plt.ion()
        for i in range(self.max_iters):
            epoch_best_fitnesses = []
            particle_positions = []
            particle_fitnesses = []
            for particle in self.particles:
                particle.update_velocity(self.inertia, self.p_attraction, self.g_attraction, self.g_best_position, self.v_max)
                particle.update_position(self.function)
                
                particle_positions.append(particle.position)
                particle_fitnesses.append(particle.curr_fitness)
                epoch_best_fitnesses.append( (particle.personal_best_value, particle.personal_best_position) )
            
            plt.clf()
            plt.scatter(*zip(*particle_positions), c=[j for j in range(len(particle_positions))], cmap=plt.get_cmap('tab20b'))
            plt.scatter(self.g_best_position[0], self.g_best_position[1], c="Red")
            plt.xlim([-5,5])
            plt.ylim([-5,5])
            plt.show()
            plt.pause(0.1)

            # Update global best
            best_epoch_fitness = min(epoch_best_fitnesses,key=lambda item:item[0])
            if best_epoch_fitness[0] < self.g_best:
                self.g_best = best_epoch_fitness[0]
                self.g_best_position = best_epoch_fitness[1]

            self.avg_fitness_per_epoch.append( sum(particle_fitnesses)/len(particle_fitnesses) )
            self.best_fitness_per_epoch.append( best_epoch_fitness[0] )
        plt.ioff()
        plt.close()


class CamelBackFunction():
    def __init__(self, range2d: int):
        self.dims = 2
        self.range = range2d # Defines a range of +/- range2d in X and Y directions

    def evaluate(self, x, y):
        return (4 - 2.1*x**2 + (1/3)*x**4)*x**2 + x*y + (-4 + 4*y**2)*y**2


if __name__ == "__main__":
    CB = CamelBackFunction(range2d=5)
    PS = ParticleSwarmOptimizer(func=CB, population_size=30, inertia=0.9, p_attraction=0.1, g_attraction=0.1, v_max=3, max_iters=50)
    PS.run()

    print("Best solution: ", PS.g_best_position)
    print('Fitness of solution: ', PS.g_best)

    plt.plot(PS.avg_fitness_per_epoch, marker="o")
    plt.title("AVG population fitness per epoch")
    plt.xlabel("Epoch No.")
    plt.ylabel("Particle Fitness")
    plt.show()

    plt.plot(PS.best_fitness_per_epoch, label="Best population fitness per epoch", marker="o")
    plt.title("Best population fitness per epoch")
    plt.xlabel("Epoch No.")
    plt.ylabel("Particle Fitness")
    plt.show()



