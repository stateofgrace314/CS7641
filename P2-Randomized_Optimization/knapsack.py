"""
Definition of the knapsack optimization problem and some functions to help with analysis
"""
import mlrose_hiive as mlrose
from RandOptBase import RandOptBase
from numpy.random import randint, seed
import numpy as np

def runKnapsack():
    """
    Run the knapsack optimization problem using the randomized optimization techniques needed
    """
    print("** Starting Evaluation of Knapsack **")
    seed(0)
    weights = randint(1, 21, size=50)
    values = randint(1, 21, size=50)
    objective = mlrose.Knapsack(weights=weights, values=values)
    problem = mlrose.DiscreteOpt(length=50, fitness_fn=objective, maximize=True, max_val=2)
    problem.set_mimic_fast_mode(True)

    # setup the optimizers
    randOptBase = RandOptBase(verbose=True)
    randOptBase.SetProblem(problem, "Knapsack")
    randOptBase.SetRHC({"iters": 2500, "restarts": 9})
    randOptBase.SetSA({"iters": 2500, "decay_rate": 0.2, "param_range": np.arange(0.02, 1.01, 0.02)})
    randOptBase.SetGA({"iters": 500, "param_range": np.arange(200, 2001, 200)})
    randOptBase.SetMIMIC({"iters": 200, "param_range": np.arange(200, 2001, 200)})

    # run the gridsearch and evaluate the optimizers
    randOptBase.EvaluateOpts()


if __name__ == "__main__":
    runKnapsack()
