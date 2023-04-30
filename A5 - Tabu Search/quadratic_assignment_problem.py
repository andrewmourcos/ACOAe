""""
Using Tabu Search to solve a quadratic assignment problem
"""
import random
import numpy as np
import pandas as pd

class QAP():
    def __init__(self, distance, flow, initial_state):
        """
        Solutions will be represented as arrays of size N
        """
        # Publics:
        self.curr_solution = initial_state
        self.size = len(initial_state)

        # Privates:
        self.distance = distance
        self.flow = flow

    
    def neighbourhood(self, randK=-1):
        """ Gathers all neighbouring solutions by carrying out all combinations of 2-element swaps
            Returns neighbours sorted by evaluation function

            randK -> randomly pick top K neighbours. If less than 1, returns ALL neighbours (sorted).
        """
        def swap_copy(original_list, idxA, idxB):
            l = original_list[:]
            l[idxA], l[idxB] = l[idxB], l[idxA]
            return l
        
        neighbours = []
        for i in range(self.size):
            for j in range(self.size):
                if i == j: continue
                swap = (i, j)
                neighbour = swap_copy(self.curr_solution, i, j)
                score = self.evaluate(neighbour)
                
                neighbours.append( [swap, neighbour, score] )

        if randK >= 1:
            neighbours = random.sample(neighbours, randK)

        neighbours = sorted(neighbours, key=lambda x: x[-1], reverse=False)
        return neighbours

    def evaluate(self, solution):
        """ Uses flow and distance matrix to evaluate the cost of a solution """
        score = 0
        for i in range(self.size):
            for j in range(self.size):
                distance = self.distance[i][j]
                flow = self.flow[solution[i]][solution[j]]
                score += flow*distance
        return score
   
class TabuSearch():
    """
        Applies Tabu Search to the "Scheme" (ex: Linear/Quadratic Assignment Problem object).
        Currently only supports 2D tabu structures (good for swapping problems. Ex: NQueens, LAP, QAP)
    """
    def __init__(self, Scheme, tenure, max_iters, max_neighbour_size, target_score=1285, aspiration=False, frequency_penalty=False):
        self.Scheme = Scheme
        self.tabu_list = np.zeros((Scheme.size, Scheme.size))
        self.aspiration = aspiration
        self.tenure = tenure
        self.max_neighbours = max_neighbour_size
        self.max_iters = max_iters
        self.target_score = target_score
        self.frequency_penalty = frequency_penalty

        self.best_solution = Scheme.curr_solution
        self.best_score = Scheme.evaluate(Scheme.curr_solution)
        self.num_iters = 0

        self.runTS()

    def runTS(self):
        # print('Starting')
        iter = 0
        while iter < self.max_iters:
            # Get candidates
            if self.max_neighbours >= 1:
                candidates = self.Scheme.neighbourhood(self.max_neighbours)
            candidates = self.Scheme.neighbourhood()

            # Choose best admissable next solution
            next_solution = self.Scheme.curr_solution
            next_score = self.Scheme.evaluate(self.Scheme.curr_solution)
            next_move = (0,0)

            for i, solution_struct in enumerate(candidates):
                move, solution, score = solution_struct
                tabu_value = self.tabu_list[move[0]][move[1]]

                if tabu_value == 0:
                    if self.frequency_penalty:
                        # Apply frequency penalty, if current option worse than next in sorted list, we should pass
                        pen_score = score - self.tabu_list[next_move[1]][next_move[0]]
                        if i < len(candidates):
                            if candidates[i+1][-1] >= pen_score:
                                next_solution = solution
                                next_move = move
                                next_score = score
                                break
                    else:
                        next_solution = solution
                        next_move = move
                        next_score = score
                        break
                else:
                    # Check for aspiration criteria
                    if self.aspiration and score < self.best_score:
                        next_solution = solution
                        next_move = move
                        next_score = score
                        break
            
            # print(next_solution, next_score, next_move)

            # Update Scheme and best tracker
            if next_score <= self.best_score:
                self.best_score = next_score
                self.best_solution = next_solution
            self.Scheme.curr_solution = next_solution

            # Terminate
            if self.best_score <= self.target_score:
                break

            # Update tabu list
            for i, row in enumerate(self.tabu_list):
                for j, cell in enumerate(row):
                    if cell > 0:
                        self.tabu_list[i][j] -= 1
            self.tabu_list[next_move[0]][next_move[1]] = self.tenure
            if self.frequency_penalty:
                self.tabu_list[next_move[1]][next_move[0]] += 1

            
            iter += 1

        self.num_iters = iter


if __name__ == "__main__":
    distance_matrix = pd.read_excel("A5\A5-Distance.xlsx", index_col=0).to_numpy()
    flow_matrix = pd.read_excel("A5\A5-Flow.xlsx", index_col=0).to_numpy()
    
    initial_state = [i for i in range(20)]
    max_iters = 500

    print("Using T=14, max_iters=500, starting at [0,1,2,...,19]")
    Q = QAP(distance_matrix, flow_matrix, initial_state)
    TS = TabuSearch(Scheme=Q, tenure=14, max_iters=max_iters, max_neighbour_size=-1, target_score=2570, aspiration=False)
    print("Best score {}".format(TS.best_score))
    print("Best solution {}".format(TS.best_solution))
    print("Stoppd at {}".format(TS.num_iters))
    print()

    # Experiment 1: Try two other initial states
    random.seed(0)
    for i in range(2):
        initial_state_c = initial_state[:]
        random.shuffle(initial_state_c)
        Q = QAP(distance_matrix, flow_matrix, initial_state_c)
        print("Using T=14, max_iters=500, starting state={}".format(initial_state_c))
        TS = TabuSearch(Scheme=Q, tenure=14, max_iters=max_iters, max_neighbour_size=-1, target_score=2570, aspiration=False)
        print("Best score {}".format(TS.best_score))
        print("Best solution {}".format(TS.best_solution))
        print("Stoppd at {}".format(TS.num_iters))
    print()

    # Experiment 2: Try one higher, one lower tabu tenure
    print("Using T=10, max_iters=500, starting state={}".format(initial_state))
    Q = QAP(distance_matrix, flow_matrix, initial_state)
    TS = TabuSearch(Scheme=Q, tenure=10, max_iters=max_iters, max_neighbour_size=-1, target_score=2570, aspiration=False)
    print("Best score {}".format(TS.best_score))
    print("Best solution {}".format(TS.best_solution))
    print("Stoppd at {}".format(TS.num_iters))
    print("Using T=20, max_iters=500, starting state={}".format(initial_state))
    Q = QAP(distance_matrix, flow_matrix, initial_state)
    TS = TabuSearch(Scheme=Q, tenure=20, max_iters=max_iters, max_neighbour_size=-1, target_score=2570, aspiration=False)
    print("Best score {}".format(TS.best_score))
    print("Best solution {}".format(TS.best_solution))
    print("Stoppd at {}".format(TS.num_iters))
    print()

    # Experiment 3: Enable aspirations for "best solution so far"
    print("Using T=14, max_iters=500, starting state={}, ASPIRATIONS ENABLED".format(initial_state))
    Q = QAP(distance_matrix, flow_matrix, initial_state)
    TS = TabuSearch(Scheme=Q, tenure=14, max_iters=max_iters, max_neighbour_size=-1, target_score=2570, aspiration=True)
    print("Best score {}".format(TS.best_score))
    print("Best solution {}".format(TS.best_solution))
    print("Stoppd at {}".format(TS.num_iters))
    print()

    # Experiment 4: Only use random 50% sample of neighbourhood
    print("Using T=14, max_iters=500, , starting state={}, only checking 50% neighbours (95/190) combos".format(initial_state))
    Q = QAP(distance_matrix, flow_matrix, initial_state)
    TS = TabuSearch(Scheme=Q, tenure=14, max_iters=max_iters, max_neighbour_size=95, target_score=2570, aspiration=False)
    print("Best score {}".format(TS.best_score))
    print("Best solution {}".format(TS.best_solution))
    print("Stoppd at {}".format(TS.num_iters))
    print()

    # Experiment 5: Enable frequency list
    print("Using T=14, max_iters={500}, starting at [0,1,2,...,19], using frequency in TABU structure")
    Q = QAP(distance_matrix, flow_matrix, initial_state)
    TS = TabuSearch(Scheme=Q, tenure=14, max_iters=max_iters, max_neighbour_size=-1, target_score=2570, aspiration=False, frequency_penalty=True)
    print("Best score {}".format(TS.best_score))
    print("Best solution {}".format(TS.best_solution))
    print("Stoppd at {}".format(TS.num_iters))
    print()






    
