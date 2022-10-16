"""
Definition of the one max optimization problem and some functions to help with analysis
"""
import mlrose_hiive as mlrose
from RandOptBase import RandOptBase
import numpy as np

def runOneMax():
    """
    Run the one max optimization problem using the randomized optimization techniques needed
    """
    print("** Starting Evaluation of One Max **")
    objective = mlrose.OneMax()
    problem = mlrose.DiscreteOpt(length=100, fitness_fn=objective, maximize=True, max_val=2)
    problem.set_mimic_fast_mode(True)

    # setup the optimizers
    randOptBase = RandOptBase(verbose=True)
    randOptBase.SetProblem(problem, "One_Max")
    randOptBase.SetRHC({"iters": 2000, "restarts": 4})
    randOptBase.SetSA({"iters": 2000, "decay_rate": 0.2, "param_range": np.arange(0.05, 1.01, 0.05)})
    randOptBase.SetGA({"iters": 250, "param_range": np.arange(200, 2001, 200)})
    randOptBase.SetMIMIC({"iters": 100, "param_range": np.arange(200, 2001, 200)})

    # run the gridsearch and evaluate the optimizers
    randOptBase.EvaluateOpts()


if __name__ == "__main__":
    runOneMax()
