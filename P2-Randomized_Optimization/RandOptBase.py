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
        self.rhc = {"name": "Random_Hill_Climbing",
                    "algorithm": mlrose.random_hill_climb,
                    "params": {"restarts": 0,
                               "max_attempts": 10000,
                               "max_iters": 10000,
                               "curve": True,
                               "random_state": self.random_state},
                    "plotter": self._plotRHC,
                    "setter": self.SetRHC,
                    "param_to_iterate": "restarts",
                    "param_range": np.arange(0, 50, 2)}
        # Simulated Annealing
        decay = mlrose.ExpDecay(init_temp=100, exp_const=0.3, min_temp=0.001)
        self.sa = {"name": "Simulated_Annealing",
                   "algorithm": mlrose.simulated_annealing,
                   "params": {"schedule": decay,
                              "max_attempts": 10000,
                              "max_iters": 10000,
                              "curve": True,
                              "random_state": self.random_state},
                   "plotter": self._plotSA,
                   "setter": self.SetSA,
                   "param_to_iterate": "decay_rate",
                   "param_range": np.arange(0.05, 2.01, 0.05)}
        # Genetic Algorithm
        self.ga = {"name": "Genetic_Algorithm",
                   "algorithm": mlrose.genetic_alg,
                   "params": {"pop_size": 1000,
                              "max_attempts": 500,
                              "max_iters": 500,
                              "curve": True,
                              "random_state": self.random_state},
                   "plotter": self._plotGA,
                   "setter": self.SetGA,
                   "param_to_iterate": "pop_size",
                   "param_range": np.arange(200, 2001, 200)}
        # MIMIC
        self.mimic = {"name": "MIMIC",
                      "algorithm": mlrose.mimic,
                      "params": {"pop_size": 1000,
                                 "max_attempts": 100,
                                 "max_iters": 100,
                                 "curve": True,
                                 "random_state": self.random_state},
                      "plotter": self._plotMIMIC,
                      "setter": self.SetMIMIC,
                      "param_to_iterate": "pop_size",
                      "param_range": np.arange(200, 2001, 200)}
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

    def SetRHC(self, params_dict):
        """
        Setter function for Random Hill Climbing
        """
        for k, v in params_dict.items():
            if "iters" in k:
                self.rhc["params"]["max_attempts"] = v
                self.rhc["params"]["max_iters"] = v
            elif k.lower() == "restarts":
                self.rhc["params"]["restarts"] = v
            elif k.lower() == "random_state":
                self.rhc["params"]["random_state"] = v
            elif k.lower() == "param_range":
                self.rhc["param_range"] = v
            else:
                print(f"WARNING: Unknown or illegal parameter change attempted on RHS: {k} = {v}")

    def SetSA(self, params_dict):
        """
        Setter function for Simulated Annealing
        """
        for k, v in params_dict.items():
            if "iters" in k:
                self.sa["params"]["max_attempts"] = v
                self.sa["params"]["max_iters"] = v
            elif k.lower() == "random_state":
                self.sa["params"]["random_state"] = v
            elif k.lower() == "param_range":
                self.sa["param_range"] = v
            elif k.lower() == "decay_rate":
                decay = mlrose.ExpDecay(init_temp=100, exp_const=v, min_temp=0.001)
                self.sa["params"]["schedule"] = decay
            else:
                print(f"WARNING: Unknown or illegal parameter change attempted on SA: {k} = {v}")

    def SetGA(self, params_dict):
        """
        Setter function for Genetic Algorithm
        """
        for k, v in params_dict.items():
            if "iters" in k:
                self.ga["params"]["max_attempts"] = v
                self.ga["params"]["max_iters"] = v
            elif k.lower() == "random_state":
                self.ga["params"]["random_state"] = v
            elif k.lower() == "param_range":
                self.ga["param_range"] = v
            elif k.lower() == "pop_size":
                self.ga["params"]["pop_size"] = v
            else:
                print(f"WARNING: Unknown or illegal parameter change attempted on GA: {k} = {v}")

    def SetMIMIC(self, params_dict):
        """
        Setter function for MIMIC
        """
        for k, v in params_dict.items():
            if "iters" in k:
                self.mimic["params"]["max_attempts"] = v
                self.mimic["params"]["max_iters"] = v
            elif k.lower() == "random_state":
                self.mimic["params"]["random_state"] = v
            elif k.lower() == "param_range":
                self.mimic["param_range"] = v
            elif k.lower() == "pop_size":
                self.mimic["params"]["pop_size"] = v
            else:
                print(f"WARNING: Unknown or illegal parameter change attempted on MIMIC: {k} = {v}")

    def EvaluateOpts(self):
        """
        Runs evaluation for each optimizer over the problem given to it and plots the results.
        """
        plt.figure()
        param_it_results = {}
        for alg in self._algs:
            name = alg["name"]
            algorithm = alg["algorithm"]
            params = alg["params"]
            plotter = alg["plotter"]
            if self.verbose:
                print(f"  Running evaluation for {name}")
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
            plotter(fitness_curves, param_it_results)
            avg_peak_time = self._time_to_peak(fitness_curves, runtimes)
            # print results to terminal
            if self.verbose:
                runtime = sum(runtimes)/len(runtimes)
                print(f"    {name} completed in avg time of {runtime} seconds")
                print(f"    Average time to reach optimal value = {avg_peak_time}")
                print(f"    Best score = {max(best_fitnesses)}")
        # finalize performace plot
        plt.title(f"{self.problemName} - Objective Score vs Iteration")
        plt.legend(loc="best")
        plt.grid()
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.savefig(f"{OUTPUT_DIR}/{self.problemName}-ObjectiveVsIteration.png")
        plt.close()
        # plot the parameter iteration data
        for alg in self._algs:
            alg_name = alg["name"]
            param = alg["param_to_iterate"]
            results = param_it_results[alg_name]
            x = results["x"]
            runtimes = results["runtimes"]
            best_fitnesses = results["best_fitnesses"]
            # fitness_curves = param_it_results[alg_name]["curves"]
            fig, axes = plt.subplots(2, 1)
            fig.suptitle(f"{self.problemName} - {alg_name} Runtime and Peak Score vs {param}")
            fig.set_size_inches(4,6)
            fig.set_dpi(90)
            axes[0].plot(x, runtimes)
            axes[0].grid()
            axes[0].set_xlabel(param)
            axes[0].set_ylabel("Runtime (s)")
            axes[1].plot(x, best_fitnesses)
            axes[1].grid()
            axes[1].set_xlabel(param)
            axes[1].set_ylabel("Fitness Score")
            # save and close
            plt.savefig(f"{OUTPUT_DIR}/{self.problemName}-{alg_name}_Over_{param}.png")
            plt.close()
        return

    @staticmethod
    def _time_to_peak(curves, runtimes):
        """
        Helper function to compute the time it took the algorithm to reach the optimal solution
        """
        times_to_peak = []
        for i in range(len(curves)):
            curve = curves[i][:,0]
            runtime = runtimes[i]
            optimal_value = max(curve)
            it_time = runtime/len(curve)
            iters = 0
            for i in curve:
                iters += 1
                if i >= optimal_value:
                    break
            times_to_peak.append(iters * it_time)
        return sum(times_to_peak)/len(times_to_peak)

    @staticmethod
    def _iterate_params(algorithm, results):
        """
        Helper function to iterate over parameters and store results
        """
        fitnesses, curves, runtimes = [], [], []
        algorithm["setter"]({"random_state": 0})
        for p in algorithm["param_range"]:
            algorithm["setter"]({algorithm["param_to_iterate"]: p})
            start = time()
            *_, fitness, curve = algorithm["algorithm"](**algorithm["params"])
            runtimes.append(time() - start)
            fitnesses.append(fitness)
            curves.append(curve[:,0])
        results[algorithm["name"]] = {"x": algorithm["param_range"],
                                      "runtimes": runtimes,
                                      "best_fitnesses": fitnesses,
                                      "fitness_curves": curves}

    def _plotRHC(self, fitness_curves, param_it_results):
        """
        Run Evaluation for the Random Hill Climbing Algorithm
        """
        x = np.arange(1, self.rhc["params"]["max_iters"] + 1)
        mean = np.mean(fitness_curves, axis=0)[:,0]
        std = np.std(fitness_curves, axis=0)[:,0]
        plot = plt.plot(x, mean, label="RHC")
        plt.fill_between(x, mean - std, mean + std, alpha=0.1, color=plot[0].get_color())

        # iterate over random restart values
        self._iterate_params(self.rhc, param_it_results)

    def _plotSA(self, fitness_curves, param_it_results):
        """
        Run Evaluation for the Simulated Annealing Algorithm
        """
        x = np.arange(1, self.sa["params"]["max_iters"] + 1)
        mean = np.mean(fitness_curves, axis=0)[:,0]
        std = np.std(fitness_curves, axis=0)[:,0]
        plot = plt.plot(x, mean, label="SA")
        plt.fill_between(x, mean - std, mean + std, alpha=0.1, color=plot[0].get_color())

        # iterate over parameter values
        self._iterate_params(self.sa, param_it_results)

    def _plotGA(self, fitness_curves, param_it_results):
        """
        Run Evaluation for the Genetic Algorithm
        """
        x = np.arange(1, self.ga["params"]["max_iters"] + 1)
        mean = np.mean(fitness_curves, axis=0)[:,0]
        std = np.std(fitness_curves, axis=0)[:,0]
        plot = plt.plot(x, mean, label="GA")
        plt.fill_between(x, mean - std, mean + std, alpha=0.1, color=plot[0].get_color())

        # iterate over parameter values
        self._iterate_params(self.ga, param_it_results)

    def _plotMIMIC(self, fitness_curves, param_it_results):
        """
        Run Evaluation for the MIMIC Algorithm
        """
        x = np.arange(1, self.mimic["params"]["max_iters"] + 1)
        mean = np.mean(fitness_curves, axis=0)[:,0]
        std = np.std(fitness_curves, axis=0)[:,0]
        plot = plt.plot(x, mean, label="MIMIC")
        plt.fill_between(x, mean - std, mean + std, alpha=0.1, color=plot[0].get_color())

        # iterate over parameter values
        self._iterate_params(self.mimic, param_it_results)
