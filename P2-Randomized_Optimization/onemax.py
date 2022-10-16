"""
Definition of the one max optimization problem and some functions to help with analysis
"""
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
from RandOptBase import RandOptBase

def runOneMax():
    """
    Run the one max optimization problem using the randomized optimization techniques needed
    """
    print("** Starting Evaluation of One Max **")
    objective = mlrose.OneMax()
    problem = mlrose.DiscreteOpt(length=100, fitness_fn=objective, maximize=True, max_val=2)

    # setup the optimizers
    randOptBase = RandOptBase(verbose=True)
    randOptBase.SetProblem(problem, "One_Max")

    # run the gridsearch and evaluate the optimizers
    randOptBase.EvaluateOpts()


if __name__ == "__main__":
    runOneMax()
