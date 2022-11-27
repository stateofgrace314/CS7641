"""
The functions used to perform the MDP analysis
"""
from hiive.mdptoolbox.mdp import ValueIteration, PolicyIteration, QLearning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pathlib
import itertools

# suppress pandas warning
pd.options.mode.chained_assignment = None
# set seed
np.random.seed(0)

# Constants
OUTDIR = f"{pathlib.Path(__file__).parent.resolve()}/plots"
COLUMNS = ("gamma", "epsilon", "time", "iterations", "reward", "average_steps", "steps_stddev",
           "success_pct", "policy", "mean_rewards", "max_rewards", "error")


def RunVI(T, R, gammas, epsilons, max_iterations=100000, verbose=True):
    """
    Runs value iteration on the input MDP
    """
    data = pd.DataFrame(0.0, index=np.arange(len(gammas)*len(epsilons)), columns=COLUMNS)
    data = data.astype({"mean_rewards": "object", "max_rewards": "object", "error": "object",
                        "policy": "object"})
    # print out a table of values
    if verbose:
        print("*** Value Iteration ***")
        print("Gamma,\tEps,\tTime,\tIter,\tReward")
        print("___________________________________________")

    t0 = time.time()
    n = 0
    for g in gammas:
        for e in epsilons:
            model = ValueIteration(T, R, gamma=g, epsilon=e, max_iter=max_iterations)
            results = model.run()
            t = results[-1]["Time"]
            iters = results[-1]["Iteration"]
            maxR = results[-1]["Max V"]
            # loop through runs and store each
            max_rewards, mean_rewards, errors = [], [], []
            for run in results:
                max_rewards.append(run["Max V"])
                mean_rewards.append(run["Mean V"])
                errors.append(run["Error"])
            # store all data
            data["gamma"][n] = g
            data["epsilon"][n] = e
            data["time"][n] = t
            data["iterations"][n] = iters
            data["reward"][n] = maxR
            data["mean_rewards"][n] = tuple(mean_rewards)
            data["max_rewards"][n] = tuple(max_rewards)
            data["error"][n] = tuple(errors)
            data["policy"][n] = model.policy
            # output table
            if verbose:
                print(f"{g:.3f},\t{e:.0E},\t{t:.2f},\t{iters},\t{maxR:0.3f}")
            n += 1
    if verbose:
        print(f"Total Runtime: {time.time()-t0:.2f}")

    # fill NaN values with 0s
    data.fillna(0, inplace=True)
    return data

def RunPI(T, R, gammas, max_iterations=100000, verbose=True):
    """
    Run policy iteration on the MDP
    """
    data = pd.DataFrame(0.0, index=np.arange(len(gammas)), columns=COLUMNS)
    data = data.astype({"mean_rewards": "object", "max_rewards": "object", "error": "object",
                        "policy": "object"})
    # print out a table of values
    if verbose:
        print("*** Policy Iteration ***")
        print("Gamma,\tTime,\tIter,\tReward")
        print("___________________________________")

    t0 = time.time()
    n = 0
    for g in gammas:
        model = PolicyIteration(T, R, gamma=g, max_iter=max_iterations, eval_type="iterative")
        results = model.run()
        t = results[-1]["Time"]
        iters = results[-1]["Iteration"]
        maxR = results[-1]["Max V"]
        # loop through runs and store each
        max_rewards, mean_rewards, errors = [], [], []
        for run in results:
            max_rewards.append(run["Max V"])
            mean_rewards.append(run["Mean V"])
            errors.append(run["Error"])
        # store all data
        data["gamma"][n] = g
        data["time"][n] = t
        data["iterations"][n] = iters
        data["reward"][n] = maxR
        data["mean_rewards"][n] = tuple(mean_rewards)
        data["max_rewards"][n] = tuple(max_rewards)
        data["error"][n] = tuple(errors)
        data["policy"][n] = model.policy
        # output table
        if verbose:
            print(f"{g:.3f},\t{t:.2f},\t{iters},\t{maxR:0.3f}")
        n += 1
    if verbose:
        print(f"Total Runtime: {time.time()-t0:.2f}")

    # fill NaN values with 0s
    data.fillna(0, inplace=True)
    return data

def RunQ(T, R, gammas, alphas, alpha_decays=[0.99], epsilon_decays=[0.99], n_iterations=[10000000],
         verbose=True):
    """
    Run the Qlearner on the MDP
    """
    columns = ["gamma", "alpha", "alpha_decay", "epsilon_decay", "iterations", "time", "reward",
               "average_steps", "steps_stddev", "success_pct", "policy", "mean_rewards",
               "max_rewards", "error"]
    n_tests = len(gammas)*len(alphas)*len(alpha_decays)*len(epsilon_decays)*len(n_iterations)
    data = pd.DataFrame(0.0, index=np.arange(n_tests), columns=columns)
    data = data.astype({"mean_rewards": "object", "max_rewards": "object", "error": "object",
                        "policy": "object"})
    # print out a table of values
    if verbose:
        print("*** Value Iteration ***")
        print("Gamma,\tAlpha,\tTime,\tIter,\tReward")
        print("___________________________________________")

    t0 = time.time()
    n = 0
    paramlist = [gammas, alphas, alpha_decays, epsilon_decays, n_iterations]
    params = list(itertools.product(*paramlist))
    for g, a, a_decay, e_decay, n_it in params:
        if verbose:
            print(f"Test Num {n+1}/{n_tests}")
            print(f"Gamma: {g:.3f},\tAlpha: {a:.2f},\tAlpha Decay: {a_decay:.3f},\tEps Decay: "
                  f"{e_decay:0.3f},\tIterations: {n_it}")

        model = QLearning(T, R, gamma=g, alpha=a, alpha_decay=a_decay, epsilon_decay=e_decay,
                          n_iter=n_it)
        results = model.run()
        t = results[-1]["Time"]
        iters = results[-1]["Iteration"]
        maxR = results[-1]["Max V"]

        max_rewards, mean_rewards, errors = [], [], []
        for run in results:
            max_rewards.append(run["Max V"])
            mean_rewards.append(run["Mean V"])
            errors.append(run["Error"])
        # store all data
        data["gamma"][n] = g
        data["alpha"][n] = a
        data["alpha_decay"][n] = a_decay
        data["epsilon_decay"][n] = e_decay
        data["time"][n] = t
        data["iterations"][n] = iters
        data["reward"][n] = maxR
        data["mean_rewards"][n] = tuple(mean_rewards)
        data["max_rewards"][n] = tuple(max_rewards)
        data["error"][n] = tuple(errors)
        data["policy"][n] = model.policy
        # output table
        if verbose:
            print(f"{g:.3f},\t{a:.3f},\t{t:.2f},\t{iters},\t{maxR:0.3f}")
        n += 1
    if verbose:
        print(f"Total Runtime: {time.time()-t0:.2f}")

    # fill NaN values with 0s
    data.fillna(0, inplace=True)
    return data

def plot_QL(df, dependent, independent, problem, title=None, logscale=False):
    """
    Plot the Qlearner plots
    """
    x = np.unique(df[dependent])
    y = [df.loc[df[dependent] == i][independent].mean() for i in x]

    plt.figure(figsize=(6,4))
    plt.plot(x, y, "o-")
    plt.title(title, fontsize=12)
    plt.xlabel(dependent)
    plt.ylabel(independent)
    plt.grid(True)
    if logscale:
        plt.xscale("log")
    plt.tight_layout()
    title=f"{problem}{independent}vs{dependent}"
    plt.savefig(f"{OUTDIR}/{title}Ql.png", dpi=400)
