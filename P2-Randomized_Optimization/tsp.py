"""
Definition of the traveling salesperson optimization problem
"""
import mlrose_hiive as mlrose
from RandOptBase import RandOptBase
import numpy as np
from numpy.random import randint, seed

def runTSP():
    """
    Run the Traveling Salesperson optimization problem using the randomized optimization techniques
    """
    print("** Starting Evaluation of Traveling Salesperson **")
    length = 20
    coords_list = []
    seed(0)
    while len(coords_list) < length:
        coord = (randint(0, 21), randint(0, 21))
        if coord not in coords_list:
            coords_list.append(coord)
    objective = mlrose.TravellingSales(coords=coords_list)
    problem = mlrose.TSPOpt(length=length, fitness_fn=objective, maximize=False)
    problem.set_mimic_fast_mode(True)

    # setup the optimizers
    randOptBase = RandOptBase(verbose=True)
    randOptBase.SetProblem(problem, "TSP")
    randOptBase.SetRHC({"iters": 2500, "restarts": 4})
    randOptBase.SetSA({"iters": 2500, "decay_rate": 0.5, "param_range": np.arange(0.05, 1.01, 0.05)})
    randOptBase.SetGA({"iters": 400, "param_range": np.arange(200, 2001, 200)})
    randOptBase.SetMIMIC({"iters": 200, "param_range": np.arange(200, 2001, 200)})

    # run the gridsearch and evaluate the optimizers
    randOptBase.EvaluateOpts()


if __name__ == "__main__":
    runTSP()
