"""
The base model for all randomized optimizers being used in GATech ML, CS7641
Since mlrose has very easy to use optimization models, this class is mostly just the default
parameters, as well as some helper functions
"""
from copy import deepcopy
from time import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import mlrose_hiive as mlrose

OUTPUT_DIR = str(Path(__file__).parent.resolve()) + "/plots"

class RandOptBase:
    """
    RandOptBase primarily stores some defaults and has some helper functions for analysis
    """

    def __init__(self, verbose=False) -> None:
        self.verbose = verbose
        # number of iterations in grid search
        self.random_state = 0
        # Set the default states and parameters for each algorithm
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

    def SetRHCIters(self, num_iters):
        """
        Set the max iteration parameter for the Random Hill Climbing Algorithm
        """
        self._set_iters(self.rhc, num_iters)

    def SetSAIters(self, num_iters):
        """
        Set the max iteration parameter for the Simulated Annealing Algorithm
        """
        self._set_iters(self.sa, num_iters)

    def SetGAIters(self, num_iters):
        """
        Set the max iteration parameter for the Genetic Algorithm
        """
        self._set_iters(self.ga, num_iters)

    def SetMIMICIters(self, num_iters):
        """
        Set the max iteration parameter for the MIMIC Algorithm
        """
        self._set_iters(self.mimic, num_iters)

    @staticmethod
    def _set_iters(algorithm, num_iters):
        """
        Helper function to set the number of iterations for the input algorithm
        """
        algorithm["params"]["max_attempts"] = num_iters
        algorithm["params"]["max_iters"] = num_iters

    def SetRHCRestarts(self, num_restarts):
        """
        Set the restarts parameter for the Random Hill Climbing Algorithm
        """
        self.rhc["params"]["restarts"] = num_restarts

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
                print(f"    Running evaluation for {name}")
            # initialize storage for outputs
            best_states, best_fitnesses, fitness_curves, runtimes = [], [], [], []
            # loop through random states to get avg and mean for each
            for random_state in range(5):
                params["random_state"] = random_state
                start = time()
                state, fitness, curve = algorithm(**params)
                runtime = time() - start
                best_states.append(state)
                best_fitnesses.append(fitness)
                fitness_curves.append(curve)
                runtimes.append(time() - start)
            plotter(fitness_curves)
            avg_peak_time = self._time_to_peak(fitness_curves, runtimes)
            # print results to terminal
            if self.verbose:
                runtime = sum(runtimes)/len(runtimes)
                print(f"    Optimization for {name} completed in avg time of {runtime} seconds")
                print(f"    Average time to reach optimal value = {avg_peak_time}")
                print(f"    Best score = {max(best_fitnesses)}")
        # finalize plot
        plt.title(f"{self.problemName} - Objective Score vs Iteration")
        plt.legend(loc="best")
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.savefig(f"{OUTPUT_DIR}/{self.problemName}-ObjectiveVsIteration.png")
        plt.close()
        return

    @staticmethod
    def _time_to_peak(curves, runtimes):
        """
        Helper function to compute the time it took the algorithm to reach the optimal solution
        """
        times_to_peak = []
        for curve, runtime in curves, runtimes:
            optimal_value = max(curve)
            it_time = runtime/len(curve)
            iters = 0
            for i in curve:
                iters += 1
                if i >= optimal_value:
                    break
            times_to_peak.append(iters * it_time)
        return sum(times_to_peak)/len(times_to_peak)

    def _plotRHC(self, fitness_curves):
        """
        Run Evaluation for the Random Hill Climbing Algorithm
        """
        x = np.arange(1, self.rhc["params"]["max_iters"] * (self.rhc["params"]["restarts"] + 1) + 1)
        mean = np.mean(fitness_curves, axis=0)
        std = np.std(fitness_curves, axis=0)
        plot = plt.plot(x, mean, label="RHC")
        plt.fill_between(x, mean - std, mean + std, alpha=0.1, color=plot[0].get_color())

    def _plotSA(self, fitness_curves):
        """
        Run Evaluation for the Simulated Annealing Algorithm
        """
        x = np.arange(1, self.sa["params"]["max_iters"] + 1)
        mean = np.mean(fitness_curves, axis=0)
        std = np.std(fitness_curves, axis=0)
        plot = plt.plot(x, mean, label="SA")
        plt.fill_between(x, mean - std, mean + std, alpha=0.1, color=plot[0].get_color())

    def _plotGA(self, fitness_curves):
        """
        Run Evaluation for the Genetic Algorithm
        """
        x = np.arange(1, self.ga["params"]["max_iters"] + 1)
        mean = np.mean(fitness_curves, axis=0)
        std = np.std(fitness_curves, axis=0)
        plot = plt.plot(x, mean, label="GA")
        plt.fill_between(x, mean - std, mean + std, alpha=0.1, color=plot[0].get_color())

    def _plotMIMIC(self, fitness_curves):
        """
        Run Evaluation for the MIMIC Algorithm
        """
        x = np.arange(1, self.mimic["params"]["max_iters"] + 1)
        mean = np.mean(fitness_curves, axis=0)
        std = np.std(fitness_curves, axis=0)
        plot = plt.plot(x, mean, label="MIMIC")
        plt.fill_between(x, mean - std, mean + std, alpha=0.1, color=plot[0].get_color())
