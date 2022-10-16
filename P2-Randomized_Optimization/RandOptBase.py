"""
The base model for all randomized optimizers being used in GATech ML, CS7641
Since mlrose has very easy to use optimization models, this class is mostly just the default
parameters, as well as some helper functions
"""
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
from copy import deepcopy
from time import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = str(Path(__file__).parent.resolve()) + "/plots"

class RandOptBase:
    """
    RandOptBase primarily stores some defaults and has some helper functions for analysis
    """

    def __init__(self, verbose=False) -> None:
        self.verbose = verbose
        # number of iterations in grid search
        self.random_state = 0
        # Randomized Hill Climbing
        self.rhc = {"name": "Randomized Hill Climbing",
                    "algorithm": mlrose.random_hill_climb,
                    "params": {"restarts": 0,
                               "max_attempts": 10000,
                               "max_iters": 10000,
                               "curve": True,
                               "random_state": self.random_state},
                    "plotter": self._plotRHC}
        # Simulated Annealing
        decay = mlrose.ExpDecay(init_temp=100, exp_const=0.3, min_temp=0.001)
        self.sa = {"name": "Simulated Annealing",
                   "algorithm": mlrose.simulated_annealing,
                   "params": {"schedule": decay,
                              "max_attempts": 10000,
                              "max_iters": 10000,
                              "curve": True,
                              "random_state": self.random_state},
                   "plotter": self._plotSA}
        # Genetic Algorithm
        self.ga = {"name": "Genetic Algorithm",
                   "algorithm": mlrose.genetic_alg,
                   "params": {"pop_size": 1000,
                              "max_attempts": 500,
                              "max_iters": 500,
                              "curve": True,
                              "random_state": self.random_state},
                   "plotter": self._plotGA}
        # MIMIC
        self.mimic = {"name": "MIMIC",
                      "algorithm": mlrose.mimic,
                      "params": {"pop_size": 1000,
                                 "max_attempts": 100,
                                 "max_iters": 100,
                                 "curve": True,
                                 "random_state": self.random_state},
                      "plotter": self._plotMIMIC}
        # name of the problem
        self.problemName = ""
        # quick list of all optimization algorithms
        self._algs = (self.rhc, self.sa, self.ga, self.mimic)

    def __repr__(self):
        return "RandOptBase"

    def SetProblem(self, problem, problemName):
        """
        Adds the mlrose problem to be optimized to the param list for each algorithm
        """
        self.problemName = problemName
        for alg in self._algs:
            alg["params"]["problem"] = deepcopy(problem)

    def EvaluateOpts(self):
        """
        Runs evaluation for each optimizer over the problem given to it and plots the results.
        """
        plt.figure()
        for alg in self._algs:
            name = alg["name"]
            algorithm = alg["algorithm"]
            params = alg["params"]
            plotter = alg["plotter"]
            if self.verbose:
                print(f"Running evaluation for {name}")
            start = time()
            best_state, best_fitness, fitness_curve = algorithm(**params)
            runtime = time() - start
            if self.verbose:
                print(f"Completed optimization for {name} in {runtime} seconds")
                print(f"    Best score = {best_fitness}")
            plotter(best_state, best_fitness, fitness_curve, runtime)
        plt.title(f"{self.problemName} - Objective Score vs Iteration")
        plt.legend(loc="best")
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.savefig(f"{OUTPUT_DIR}/{self.problemName}-ObjectiveVsIteration.png")
        plt.close()
        return

    def _plotRHC(self, best_state, best_fitness, fitness_curve, runtime):
        """
        Run Evaluation for the Random Hill Climbing Algorithm
        """
        x = np.arange(1, self.rhc["params"]["max_iters"] + 1)
        # mean = np.mean(fitness_curve, axis=0)
        mean = fitness_curve
        # std = np.std(fitness_curve, axis=0)
        plot = plt.plot(x, mean, label="RHC")
        # plt.fill_between(x, mean - std, mean + std, alpha=0.1, color=plot[0].get_color())

    def _plotSA(self, best_state, best_fitness, fitness_curve, runtime):
        """
        Run Evaluation for the Simulated Annealing Algorithm
        """
        x = np.arange(1, self.sa["params"]["max_iters"] + 1)
        # mean = np.mean(fitness_curve, axis=0)
        mean = fitness_curve
        # std = np.std(fitness_curve, axis=0)
        plot = plt.plot(x, mean, label="SA")
        # plt.fill_between(x, mean - std, mean + std, alpha=0.1, color=plot[0].get_color())

    def _plotGA(self, best_state, best_fitness, fitness_curve, runtime):
        """
        Run Evaluation for the Genetic Algorithm
        """
        x = np.arange(1, self.ga["params"]["max_iters"] + 1)
        # mean = np.mean(fitness_curve, axis=0)
        mean = fitness_curve
        # std = np.std(fitness_curve, axis=0)
        plot = plt.plot(x, mean, label="GA")
        # plt.fill_between(x, mean - std, mean + std, alpha=0.1, color=plot[0].get_color())

    def _plotMIMIC(self, best_state, best_fitness, fitness_curve, runtime):
        """
        Run Evaluation for the MIMIC Algorithm
        """
        x = np.arange(1, self.mimic["params"]["max_iters"] + 1)
        # mean = np.mean(fitness_curve, axis=0)
        mean = fitness_curve
        # std = np.std(fitness_curve, axis=0)
        plot = plt.plot(x, mean, label="MIMIC")
        # plt.fill_between(x, mean - std, mean + std, alpha=0.1, color=plot[0].get_color())
