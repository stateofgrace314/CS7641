"""
Definition of the four peaks optimization problem
"""
import mlrose_hiive as mlrose
from RandOptBase import RandOptBase
import numpy as np

def runFourPeaks():
    """
    Run the four peaks optimization problem using the randomized optimization techniques needed
    """
    print("** Starting Evaluation of Four Peaks **")
    objective = mlrose.FourPeaks(t_pct=0.1)
    problem = mlrose.DiscreteOpt(length=100, fitness_fn=objective, maximize=True, max_val=2)
    problem.set_mimic_fast_mode(True)

    # setup the optimizers
    randOptBase = RandOptBase(verbose=True)
    randOptBase.SetProblem(problem, "Four_Peaks")
    randOptBase.SetRHC({"iters": 6000, "restarts": 9})
    randOptBase.SetSA({"iters": 6000, "decay_rate": 0.02, "param_range": np.arange(0.002, 0.1, 0.002)})
    randOptBase.SetGA({"iters": 500, "param_range": np.arange(200, 2001, 200)})
    randOptBase.SetMIMIC({"iters": 250, "param_range": np.arange(200, 2001, 200)})

    # run the gridsearch and evaluate the optimizers
    randOptBase.EvaluateOpts()


if __name__ == "__main__":
    runFourPeaks()
