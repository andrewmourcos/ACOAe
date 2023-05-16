# ACOAe
Adaptive and Cooperative Optimization Algorithm Experiments implemented in Python.

## Simulated Annealing
<a href="https://github.com/andrewmourcos/ACOAe/blob/main/A4%20-%20Simulated%20Annealing%20Search/simulated_annealing.py">Implementation.</a>

This trajectory-based algorithm extends typical hill-climbing optimizers by introducing the idea of an acceptance function. As the algorithm evaluates nearby candidate states, it uses a probabilistic acceptance function to choose whether to move to the next state or not - in particular this was used:

$$
  P_{accept}=\begin{cases}
    1, & \text{if better state}.\\
    e^{-\text{(how much worse)}/T}, & \text{if worse state}.
  \end{cases}
$$

Where T is the "annealing temperature" (experimented with a fixed, geometrically decreasing, and linearly decreasing T). The idea behind allowing worse states is to allow the optimizer the chance to escape local minima and promote exploration.

## Tabu Search
<a href="https://github.com/andrewmourcos/ACOAe/blob/main/A5%20-%20Tabu%20Search/quadratic_assignment_problem.py">Implementation.</a>

This is another trajectory-based optimization method, although its approach is quite different from simulated annealing. It operates on the principle of avoiding repeated moves by tracking them in a "tabu structure" for some predefined tenure. The optimizer evaluates all neighbouring solution candidates, but is restricted from accepting solutions that are in the tabu structure (with some exceptions, referred to as "aspirations"). In the code snippet, I ran a few experiments to optimize for a quadratic assignment problem. In general, it performed well but it's a bit tricky to find suitable hyperparameters for different problems.

## Genetic Algorithm
<a href="https://github.com/andrewmourcos/ACOAe/blob/main/A6%20-%20Genetic%20Algorithm/binary_genetic_algo.py">Implementation.</a>

This was a pretty neat evolutionary algorithm. In essence, a population of candidate solutions are generated and evaluated based on their "fitness". The best candidates become "parents" to child solutions (made via cross-over of two parents with randomized mutations) and the cycle restarts. Ideally, every generation is better than the last by promoting the most fit individuals and their offspring.

## Ant Colony Optimization
<a href="https://github.com/andrewmourcos/ACOAe/blob/main/A7%20-%20Ant%20Colony/travelling_salesman_ant_colony.py">Implementation.</a>

Here, I implemented an ant colony algorithm to optimize a 29-city traveling salesman problem. I used a pseudo-random proportional rule for transitions (as proposed by Gambardella & Dorigo) and an online delayed pheromone update rule - hoping to balance exploration and exploitation of the search algorithm. A nice thing about this implementation is that ant populations could be spawned in parallel - so I used the opportunity as multi-threading practice.
With the hyperparameters shown in the test script, we achieve a minimum round-trip distance of 8845.61km after less than 100 iterations. Note that the optimal solution had a trip distance of 8782.60km - so pretty close.

## Particle Swarm Optimization
<a href="https://github.com/andrewmourcos/ACOAe/blob/main/A8%20-%20Particle%20Swarm%20Optimization/particle_swarm_optimizer.py">Implementation.</a>

This was probably my favourite adaptive/cooperative algorithm to implement - it's both easy to code and can provide some very satisfying animations as shown below.
Here, particles are initialized to random locations in the search space. Each record local and global "bests" to which surrounding particles are attracted in hopes of converging to a global optimum (candidate exploitation). An inertial term is added to the velocity update equations for particle motion to encourage exploration and avoid convergence to local optima.
