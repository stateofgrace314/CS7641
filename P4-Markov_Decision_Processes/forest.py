"""
Function for running the Forest Management problem and analysis
"""

from hiive.mdptoolbox.example import forest
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from utils import OUTDIR, RunVI, RunPI, RunQ, plot_QL

def plot_forest(policy, model=""):
    """
    Plots the forest policy
    """
    colors = { 0: "g", 1: "k"}
    title = f"Forest Management Optimal Policy - {model}\n(Black=Cut, Green=Wait)"

    plt.figure(figsize=(9,3))
    plt.title(title, fontsize=12, weight="bold")

    # represent each data point as a vertical colored line
    for i in range(len(policy)):
        color = colors[policy[i]]
        plt.axvline(x=i, color=color, lw=2)
    plt.axis("tight")
    plt.yticks([])
    plt.xlabel("Year")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/ForestOptimalPolicy{model}.png", dpi=400)

def runForest(verbose=True):
    """
    Runs the RL analysis on the forest management problem
    """
    # init the forest managment problem
    T,R = forest(S=625)

    ### Value Iteration ###

    # Paramater ranges
    gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999]
    epsilons = [1e-1, 1e-2, 1e-3, 1e-5, 1e-7, 1e-10, 1e-12]

    # run the analysis
    vi_data  = RunVI(T, R, gammas, epsilons, verbose)

    # plot reward vs gamma
    vi_data.plot(x="gamma", y="reward", title="Forest Management Reward vs. Gamma")
    plt.grid()
    plt.savefig(f"{OUTDIR}/ForestRewardvsGammaVI.png")
    # plot reward vs iterations
    vi_data.sort_values("iterations", inplace=True)
    vi_data.plot(x="iterations", y="reward", title="Forest Management Reward vs. Iterations")
    plt.grid(True)
    plt.savefig(f"{OUTDIR}/ForestRewardvsIterationVI.png")

    # get the best one, plot it and save the data
    bestRun = vi_data["reward"].argmax()
    bestPolicy = vi_data["policy"][bestRun]
    bestR = vi_data["reward"].max()
    bestG = vi_data["gamma"][bestRun]
    bestE = vi_data["epsilon"][bestRun]
    plot_forest(bestPolicy, "VI")
    vi_data.to_csv(f"{OUTDIR}/ForestDataVI.csv")
    if verbose:
        print(f"Best Result:\n\tReward = {bestR:.2f}\n\tGamma = {bestG:.3f}\n\tEps= {bestE:.2E}")

    ### Policy Iteration ###

    # parameter range
    gammas   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999]

    # run the analysis
    pi_data  = RunPI(T, R, gammas, verbose)

    # plot reward vs gamma
    pi_data.plot(x="gamma", y="reward", title="Rewards vs. Gamma")
    plt.grid()
    plt.savefig(f"{OUTDIR}/ForestRewardvsGammaPI.png")
    # plot reward vs iteration
    pi_data.sort_values("iterations", inplace=True)
    pi_data.plot(x="iterations", y="reward", title="Rewards vs. Iterations")
    plt.grid()
    plt.savefig(f"{OUTDIR}/ForestRewardvsIterationPI.png")

    # get the best one, plot it and save the data
    bestRun = pi_data["reward"].argmax()
    bestPolicy = pi_data["policy"][bestRun]
    bestR = pi_data["reward"].max()
    bestG = pi_data["gamma"][bestRun]
    plot_forest(bestPolicy, "PI")
    vi_data.to_csv(f"{OUTDIR}/ForestDataPI.csv")
    if verbose:
        print(f"Best Result:\n\tReward = {bestR:.2f}\n\tGamma = {bestG:.3f}")

    ### Q learning
    gammas         = [0.8, 0.9, 0.99]
    alphas         = [0.01, 0.1, 0.2]
    alpha_decays   = [0.9, 0.999]
    epsilon_decays = [0.9, 0.999]
    iterations     = [1e5, 1e6, 1e7]

    # to run again, uncommment below (takes a long time)
    # ql_data  = RunQ(T, R, gammas, alphas, alpha_decays=alpha_decays, epsilon_decays=epsilon_decays, n_iterations=iterations, showResults=False)
    # # write all results to csv file
    # csvFile = f"{OUTDIR}/QL_results_forest.csv"
    # ql_data.to_csv(csvFile)

    # Read in Q-Learning data
    ql_data = pd.read_csv(f"{OUTDIR}/ForestDataQl.csv")

    # check which hyperparameters made most impact
    slice = ["gamma", "alpha", "alpha_decay", "epsilon_decay", "iterations", "reward", "time"]
    df = ql_data[slice]
    ql_corr = df.corr()
    sns.set(style="white")
    fig, ax = plt.subplots(figsize=(8,7))
    ax.set_title("Correlation Matrix of Q-Learning Parameters", fontsize=20)
    mask = np.triu(np.ones_like(ql_corr, dtype=np.bool))
    cmap = sns.diverging_palette(255, 0, as_cmap=True)
    sns.heatmap(ql_corr, mask=mask, cmap=cmap, square=True, linewidths=0.5, cbar_kws={"shrink":.75})
    plt.savefig(f"{OUTDIR}/ForestQlParamCorrelation.png")

    # Plots
    plot_QL(df, "iterations", "time", "Forest", title="Mean Runtime vs. Num Iterations", logscale=True)
    plot_QL(df, "iterations", "reward", "Forest", title="Mean Reward vs. Num Iterations", logscale=True)
    plot_QL(df, "alpha_decay", "reward", "Forest", title="Mean Reward vs. Alpha Decay")
    plot_QL(df, "gamma", "reward", "Forest", title="Mean Reward vs. Gamma")

    # Get the best one
    bestRun = ql_data["reward"].argmax()
    best_policy = ql_data["policy"][bestRun]
    bestR = ql_data["reward"][bestRun]
    bestG = ql_data["gamma"][bestRun]
    bestA = ql_data["alpha"][bestRun]
    bestAD = ql_data["alpha_decay"][bestRun]
    bestED = ql_data["epsilon_decay"][bestRun]
    bestIt = ql_data["iterations"][bestRun]

    # plot the policy (convert if necessary)
    if isinstance(best_policy, str):
        best_policy = best_policy.strip("{").strip("}")
        best_policy = best_policy.strip("(").strip(")")
        best_policy = [int(i) for i in best_policy.split(",")]
    plot_forest(best_policy, "Ql")

    if verbose:
        print(f"Best Result:\n\tReward = {bestR:.2f}\n\tGamma = {bestG:.3f}\n\tAlpha = {bestA:.3f}"
              f"\n\tAlpha Decay = {bestAD:.3f}\n\tEps Decay = {bestED:.3f}\n\tIterations = {bestIt}")
