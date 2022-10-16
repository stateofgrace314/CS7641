"""
Definition of the traveling salesperson optimization problem
"""
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
from RandOptBase import RandOptBase
from numpy.random import randint

def runTSP():
    """
    Run the Traveling Salesperson optimization problem using the randomized optimization techniques needed
    """
    print("** Starting Evaluation of Traveling Salesperson **")
    length = 20
    coords_list = []
    while len(coords_list) < length:
        coord = (randint(0, 21), randint(0, 21))
        if coord not in coords_list:
            coords_list.append(coord)
    objective = mlrose.TravellingSales(coords=coords_list)
    problem = mlrose.TSPOpt(length=length, fitness_fn=objective, maximize=False)

    # setup the optimizers
    randOptBase = RandOptBase(verbose=True)
    randOptBase.SetProblem(problem, "TSP")

    # run the gridsearch and evaluate the optimizers
    randOptBase.EvaluateOpts()


if __name__ == "__main__":
    runTSP()