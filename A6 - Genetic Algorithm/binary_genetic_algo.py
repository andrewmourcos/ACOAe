import random
import numpy as np

random.seed(2)

class GeneticAlgorithm():

    def __init__(self, func, precision, pop_size, generations):
        self.function = func
        self.precision = precision
        self.generations = generations

        self.num_bits = int(np.ceil( np.log2( (2*self.function.range * 10**self.precision) + 1 ) ))
        self.pop_size = pop_size

        self.initial_population = np.random.choice(a=[0,1], size=(self.pop_size, self.num_bits*2))

        print("Starting GA with:\n\t-Bits:{}\n\t-Pop size:{}\n\t-Precision:{}".format(self.num_bits,
            self.pop_size, self.generations))

        self.best_score = 0
        self.best_chromosome = 0
        self.average_pop_fitness = 0
        self.avg_fitness = []
        self.best_fitness = []

        self.run()
        

    def fitness(self, chromosome):
        x, y = chromosome[:14], chromosome[14:]
        x = self.binary2real(x)/1000 - 5 
        y = self.binary2real(y)/1000 - 5
        f = -1 * self.function.evaluate(x, y)
        # print(x, y, f)
        return f

    def evaluate(self, population):
        """ Apply fitness function to whole population """
        scores = []
        for p in population:
            s = self.fitness(p)
            scores.append(s)
            if s > self.best_score:
                self.best_score = s
                self.best_chromosome = p

        self.average_pop_fitness = sum(scores)/len(scores)
        return scores

    def select(self, population, scores):
        min_score = np.min(scores)
        max_score = np.max(scores)
        normalized_scores = (scores - min_score)/(np.max(scores)-min_score)
        normalized_scores /= normalized_scores.sum()

        # Roulette wheel parent selections
        selections = []
        for _ in range(self.pop_size // 2):
            p1_idx = np.random.choice( self.pop_size, p=normalized_scores )
            p2_idx = np.random.choice( self.pop_size, p=normalized_scores )
            selections.append( (p1_idx,p2_idx) )
        return selections

    def crossover(self, parents, population):
        """ Performs single-point crossover
            parents: list of index pairs to crossover from population array
            population: list of all current chromosomes 
        """
        children = []
        for parent_pair in parents:
            cross_idx = random.randrange(1,self.num_bits)
            parentA, parentB = population[parent_pair[0]], population[parent_pair[1]]

            childA = np.concatenate( (parentA[:cross_idx], parentB[cross_idx:]) )
            childB = np.concatenate( (parentB[:cross_idx], parentA[cross_idx:]) )
            
            children.append(childA)
            children.append(childB)

        return np.array(children)
    
    def mutate(self, population):
        # Random bit-flip mutation
        for i in range(len(population)):
            p = random.uniform(0, 1)
            if p > 0.5:
                child = population[i]
                rand_bit = random.randint(0, len(child)-1)
                population[i][rand_bit] ^= 1

    def run(self):
        population = self.initial_population
        
        for i in range(self.generations):
            # Evaluate/rank fitness of individuals
            scores = self.evaluate(population)
            # Select parents
            parents = self.select(population, scores)
            # Apply crossover
            population = self.crossover(parents, population)
            # Apply mutation
            self.mutate(population)
            print("iteration", i, "Best chromosome:", self.best_chromosome, "Best value:", self.best_score, "avg score:", self.average_pop_fitness)

            # x, y = self.best_chromosome[:14], self.best_chromosome[14:]
            # x = self.binary2real(x)/1000 - 5 
            # y = self.binary2real(y)/1000 - 5
            # print("Best, X, Y: ", x, y)

            self.avg_fitness.append(self.average_pop_fitness)
            self.best_fitness.append(self.best_score)
        print("Done!")


    def binary2real(self, arr):
        return int("".join(arr.astype(str)), 2 )



class CamelBackFunction():
    def __init__(self, range2d: int):
        self.dims = 2
        self.range = range2d # Defines a range of +/- range2d in X and Y directions

    def evaluate(self, x, y):
        return (4 - 2.1*x**2 + (1/3)*x**4)*x**2 + x*y + (-4 + 4*y**2)*y**2


if __name__ == "__main__":
    CB = CamelBackFunction(range2d=5)
    GA = GeneticAlgorithm(func=CB, precision=3, pop_size=10, generations=100)

    avg = np.asarray(GA.avg_fitness)
    np.savetxt("avg.csv", avg, delimiter=",")

    best = np.asarray(GA.best_fitness)
    np.savetxt("best.csv", best, delimiter=",")



