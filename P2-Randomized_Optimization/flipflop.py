"""
Definition of the flip flop optimization problem and some functions to help with analysis
"""
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
from RandOptBase import RandOptBase

def runFlipFlop():
    """
    Run the Flip Flop optimization problem using the randomized optimization techniques needed
    """
    print("** Starting Evaluation of Flip Flop **")
    objective = mlrose.FlipFlop()
    problem = mlrose.DiscreteOpt(length=100, fitness_fn=objective, maximize=True, max_val=2)

    # setup the optimizers
    randOptBase = RandOptBase(verbose=True)
    randOptBase.SetProblem(problem, "Flip_Flop")

    # run the gridsearch and evaluate the optimizers
    randOptBase.EvaluateOpts()


if __name__ == "__main__":
    runFlipFlop()
