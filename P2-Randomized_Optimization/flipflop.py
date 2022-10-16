"""
Definition of the flip flop optimization problem and some functions to help with analysis
"""
import mlrose_hiive as mlrose
from RandOptBase import RandOptBase
import numpy as np

def runFlipFlop():
    """
    Run the Flip Flop optimization problem using the randomized optimization techniques needed
    """
    print("** Starting Evaluation of Flip Flop **")
    objective = mlrose.FlipFlop()
    problem = mlrose.DiscreteOpt(length=100, fitness_fn=objective, maximize=True, max_val=2)
    problem.set_mimic_fast_mode(True)

    # setup the optimizers
    randOptBase = RandOptBase(verbose=True)
    randOptBase.SetProblem(problem, "Flip_Flop")
    randOptBase.SetRHC({"iters": 2500, "restarts": 4})
    randOptBase.SetSA({"iters": 2500, "param_range": np.arange(0.05, 2.01, 0.05)})
    randOptBase.SetGA({"iters": 500, "param_range": np.arange(200, 2001, 200)})
    randOptBase.SetMIMIC({"iters": 100, "param_range": np.arange(200, 2001, 200)})

    # run the gridsearch and evaluate the optimizers
    randOptBase.EvaluateOpts()


if __name__ == "__main__":
    runFlipFlop()
