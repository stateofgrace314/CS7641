"""
Definition of the four peaks optimization problem
"""
import mlrose_hiive as mlrose
from RandOptBase import RandOptBase

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

    # run the gridsearch and evaluate the optimizers
    randOptBase.EvaluateOpts()


if __name__ == "__main__":
    runFourPeaks()
