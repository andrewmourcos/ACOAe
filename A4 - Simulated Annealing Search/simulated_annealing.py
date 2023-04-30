from typing import Callable
import numpy as np
import pandas as pd

class SimulatedAnnealing():
    def __init__(self, function_handle:Callable,
                       neighbour_generator_handle:Callable,
                       dims:int,
                       starting_location:np.array,
                       step_size:float, 
                       initial_temperature:float, 
                       alpha:float,
                       schedule:str,
                       optim_type:str = "min"):
        
        if schedule not in ["geometric", "linear", "slow"]:
            raise ValueError("Invalid annealing schedule: {}".format(schedule))
        if optim_type not in ["max", "min"]:
            raise ValueError("Invalid optimization type: {}".format(optim_type))

        self.function_handle = function_handle
        self.neighbour_generator_handle = neighbour_generator_handle

        self.schedule = schedule
        self.dims = dims
        self.current_state = starting_location
        self.step_size = step_size
        self.temperature = initial_temperature
        self.alpha = alpha
        self.optim_type = optim_type

        self.current_cost = self.function_handle(self.current_state)
        self.num_iters = 0
        self.SA()

    def step_temperature(self):
        if self.schedule == "linear":
            self.temperature = self.temperature - self.alpha
        elif self.schedule == "geometric":
            self.temperature = self.temperature * self.alpha
        elif self.schedule == "slow":
            self.temperature = self.temperature / (1+self.temperature*self.alpha)
    

    def step(self) -> int:
        """ Performs a single step in the SA Algo """
        self.num_iters += 1
        # Generate random next move
        next_state = self.neighbour_generator_handle(self.dims, self.step_size, self.current_state)
        # Evaluate function
        next_cost = self.function_handle(next_state)
        
        # Compute acceptance probability
        if ((self.current_cost - next_cost > 0 and self.optim_type == "min") or
            (self.current_cost - next_cost <= 0 and self.optim_type == "max")):
            P = 1
        else:
            # Note: 1E-10 to avoid divide by zero error
            P = np.exp(-np.abs(self.current_cost-next_cost)/(self.temperature+1E-10))
                
        # Move or terminate
        if P >= np.random.rand() and self.temperature > 0:
            self.current_cost = next_cost
            self.current_state = next_state
        else:
            return 0 # Tell SA algo to terminate

        # Update temperature
        self.step_temperature()

        return 1 # Tell SA algo that step was completed succesfully

    def SA(self):
        while self.step():
            pass


def easom_function(x:np.array):
    return -np.cos(x[0])*np.cos(x[1])*np.exp(-((x[0]-np.pi)**2+(x[1]-np.pi)**2));

def generate_neighbour(dims, step_size, current_state) -> np.array:
    # Pick random neighbour within a certain step distance
    random_step = (np.random.rand(dims) - 0.5) * 2 * step_size
    random_neighbour = current_state + random_step
    # If neighbour is out of bounds, step in the opposite direction instead
    for i in range(dims):
        if not 100 > random_neighbour[i] > -100:
            random_neighbour[i] -= 2*random_step[i]

    return random_neighbour


if __name__ == "__main__":
    print("Running SA algo with many different options... This may take a couple minutes.")
    # Pandas Dataframe is used to save results from all experiments needed for the report
    df = pd.DataFrame(columns=["Starting Point", "Ending Point", "Ending Value", "No. Iterations", "Init Temperature", "Annealing Schedule"])

    # Experiments for parts B-D
    for j, init_temp in enumerate([100, 60, 80, 120, 140]):
        np.random.seed(0)
        for i in range(10):
            starting_point = (np.random.rand(2) - 0.5)*100
            schedule="linear"
            optim = SimulatedAnnealing(function_handle=easom_function,
                            neighbour_generator_handle=generate_neighbour,
                            dims=2, 
                            starting_location=starting_point,
                            step_size=5,
                            initial_temperature=init_temp,
                            alpha=0.001,
                            schedule=schedule)
            df.loc[i+(j*10)] = [starting_point.round(decimals=4), optim.current_state.round(decimals=4), optim.current_cost, optim.num_iters, init_temp, schedule]
 
    # Experiment for part E
    np.random.seed(0)
    for i in range(10):
        init_temp=100
        starting_point = (np.random.rand(2) - 0.5)*100
        schedule="geometric"
        optim = SimulatedAnnealing(function_handle=easom_function,
                        neighbour_generator_handle=generate_neighbour,
                        dims=2, 
                        starting_location=starting_point,
                        step_size=5,
                        initial_temperature=init_temp,
                        alpha=0.99999,
                        schedule=schedule)
        df.loc[i+(50)] = [starting_point.round(decimals=4), optim.current_state.round(decimals=4), optim.current_cost, optim.num_iters, init_temp, schedule]

    print(df.round(decimals=4))
