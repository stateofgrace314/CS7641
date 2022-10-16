"""
Definition of the knapsack optimization problem and some functions to help with analysis
"""
import mlrose_hiive as mlrose
from RandOptBase import RandOptBase
from numpy.random import randint, seed

def runKnapsack():
    """
    Run the knapsack optimization problem using the randomized optimization techniques needed
    """
    print("** Starting Evaluation of Knapsack **")
    seed(0)
    weights = randint(1, 21, size=25)
    values = randint(1, 21, size=25)
    objective = mlrose.Knapsack(weights=weights, values=values)
    problem = mlrose.DiscreteOpt(length=25, fitness_fn=objective, maximize=True, max_val=2)
    problem.set_mimic_fast_mode(True)

    # setup the optimizers
    randOptBase = RandOptBase(verbose=True)
    randOptBase.SetProblem(problem, "Knapsack")

    # run the gridsearch and evaluate the optimizers
    randOptBase.EvaluateOpts()


if __name__ == "__main__":
    runKnapsack()
