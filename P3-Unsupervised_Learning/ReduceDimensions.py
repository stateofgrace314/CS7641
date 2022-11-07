"""
Run dimensionality reduction methods on input datasets and produce necessary plots for analysis
"""

from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.random_projection import GaussianRandomProjection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, explained_variance_score
from yellowbrick.features import ParallelCoordinates
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
from DataUtils import LoadData

OUT_DIR = f"{Path(__file__).parent.resolve()}/plots"

def getKurtosisScore(model, X):
    """
    Returns a score using kurtosis
    """
    result = pd.DataFrame(model.fit_transform(X))
    return result.kurtosis()

# PCA - Principal Component Analysis
def runPCA(X, feature_names, dataset_name):
    """
    Run PCA over the input dataset
    """
    # init the dimensionality reduction model
    dr_model = PCA(random_state=0)
    model_name = "PCA"

    # Loop through number of components/features
    n_components = np.arange(1, len(feature_names) + 1)
    scores = []
    for n in n_components:
        dr_model.n_components = n
        scores.append(np.mean(cross_val_score(dr_model, X)))

    # get the best number of components based on variance
    dr_model = PCA(random_state=0).fit(X)
    evr = np.cumsum(dr_model.explained_variance_ratio_)
    # "best" number of components is 2 more than the first to pass 98%
    evr_zero = evr-evr.min()
    evr_norm = evr_zero/evr_zero.max()
    n_components_best = np.argmax(evr_norm >= 0.99) + 1
    best_score = scores[n_components_best - 1]
    print(f"{model_name}: best n_components by CV for {dataset_name} = {n_components_best}")

    # start the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    title = f"{model_name}_Analysis-{dataset_name}"
    filename = f"{OUT_DIR}/{title}.png"
    fig.suptitle(title)

    # plot 1 - CV over n_components
    ax1.plot(n_components, scores, "b", label="Cross Validation Scores")
    ax1.axvline(n_components_best, color="k", label=f"{model_name} Best CV score: {best_score:.4f}",
                linestyle="--")
    ax1.set_ylabel("CV score")
    ax1.legend()
    ax1.set_title("Cross Validation Score by num Components")
    y_min = min(scores)
    y_max = max(scores)
    diff = y_max - y_min
    y_ticks = np.linspace(y_min - 0.1*diff, y_max + 0.1*diff, 6)
    ax1.set_yticks(y_ticks)

    # plot 2 - Variance Ratio
    ax2.plot(n_components, evr, label="Explained Variance Ratio")
    ax2.axvline(n_components_best, color="k", label=f"Best num components: {n_components_best}",
                linestyle="--")
    ax2.set_ylabel("variance (%)")
    ax2.legend()
    ax2.set_title(f"{dataset_name}: Variance by num Components")
    y_min = min(evr)
    y_max = max(evr)
    diff = y_max - y_min
    y_ticks = np.linspace(y_min - 0.1*diff, y_max + 0.1*diff, 6)
    ax2.set_yticks(y_ticks)

    # global plot settings
    plt.xticks(n_components)
    plt.xlabel("num components")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    # return the best number
    return n_components_best

# ICA - Independent component Analysis
def runICA(X, feature_names, dataset_name):
    """
    Run ICA over the input dataset
    """
    # init the dimensionality reduction model
    model_name = "ICA"
    dr_model = FastICA(random_state=0)

    # Loop through number of components/features
    n_components = np.arange(1, len(feature_names) + 1)
    scores = []
    for n in n_components:
        dr_model.n_components = n
        scores.append(np.mean(getKurtosisScore(dr_model, X)))

    # get the best number of components based on variance
    dr_model = FastICA(random_state=0).fit(X)
    n_components_best = n_components[np.argmax(scores)]
    best_score = scores[n_components_best - 1]

    # start the plot
    fig, ax1 = plt.subplots(1, 1, sharex=True)
    title = f"{model_name}_Analysis-{dataset_name}"
    filename = f"{OUT_DIR}/{title}.png"
    fig.suptitle(title)

    # plot 1 - CV over n_components
    ax1.plot(n_components, scores, "b", label="Kurtosis")
    ax1.axvline(n_components_best, color="k", label=f"{model_name} Best kurtosis score: {best_score:.4f}",
                linestyle="--")
    ax1.set_ylabel("Kurtosis")
    ax1.legend()
    ax1.set_title("Kurtosis by num Components")
    y_min = min(scores)
    y_max = max(scores)
    diff = y_max - y_min
    y_ticks = np.linspace(y_min - 0.1*diff, y_max + 0.1*diff, 6)
    ax1.set_yticks(y_ticks)

    # global plot settings
    plt.xticks(n_components)
    plt.xlabel("num components")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    # return the best number
    return n_components_best


# RP - Randomized Projections
def runRP(X, feature_names, dataset_name):
    """
    Run Randomized projections on dataset
    """
    # init the dimensionality reduction model
    model_name = "RP"
    dr_model = GaussianRandomProjection(random_state=0)

    # Loop through number of components/features
    n_components = np.arange(1, len(feature_names) + 1)
    scores = []
    for n in n_components:
        dr_model.n_components = n
        reduced = dr_model.fit_transform(X)
        A = np.linalg.pinv(dr_model.components_.T)
        reconstructed = np.dot(reduced, A)
        err = mean_squared_error(X, reconstructed)
        scores.append(err)

    # get the best number of components manually
    if "heart" in dataset_name:
        n_components_best = 10
    elif "water" in dataset_name:
        n_components_best = 6
    best_score = scores[n_components_best - 1]

    # start the plot
    fig, ax1 = plt.subplots(1, 1, sharex=True)
    title = f"{model_name}_Analysis-{dataset_name}"
    filename = f"{OUT_DIR}/{title}.png"
    fig.suptitle(title)

    # plot 1 - CV over n_components
    ax1.plot(n_components, scores, "b", label="RMSE")
    ax1.axvline(n_components_best, color="k", label=f"{model_name} Min RMSE: {best_score:.4f}",
                linestyle="--")
    ax1.set_ylabel("RMSE")
    ax1.legend()
    ax1.set_title("Randomized Projection RMSE by num Components")
    y_min = min(scores)
    y_max = max(scores)
    diff = y_max - y_min
    y_ticks = np.linspace(y_min - 0.1*diff, y_max + 0.1*diff, 8)
    ax1.set_yticks(y_ticks)

    # global plot settings
    plt.xticks(n_components)
    plt.xlabel("num components")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    # return the best number
    return n_components_best

def runNMF(X, feature_names, dataset_name):
    """
    Run NMF on dataset
    """
    # init the dimensionality reduction model
    dr_model = NMF(random_state=0)
    model_name = "NMF"

    # Loop through number of components/features
    n_components = np.arange(1, len(feature_names) + 1)
    scores = []
    for n in n_components:
        dr_model.n_components = n
        dr_model.fit(X)
        y_pred = dr_model.inverse_transform(dr_model.transform(X))
        scores.append(explained_variance_score(X, y_pred))

    # get the best number of components
    scores = np.array(scores)
    evr_zero = scores-scores.min()
    evr_norm = evr_zero/evr_zero.max()
    n_components_best = np.argmax(evr_norm >= 0.85) + 1
    best_score = scores[n_components_best - 1]

    # start the plot
    fig, ax1 = plt.subplots(1, 1, sharex=True)
    title = f"{model_name}_Analysis-{dataset_name}"
    filename = f"{OUT_DIR}/{title}.png"
    fig.suptitle(title)

    # plot 1 - CV over n_components
    ax1.plot(n_components, scores, "b", label="Explained Variance")
    ax1.axvline(n_components_best, color="k", label=f"{model_name} Best EVR: {best_score:.4f}",
                linestyle="--")
    ax1.set_ylabel("Variance (%)")
    ax1.legend()
    ax1.set_title("Explained Variance by num Components")
    y_min = min(scores)
    y_max = max(scores)
    diff = y_max - y_min
    y_ticks = np.linspace(y_min - 0.1*diff, y_max + 0.1*diff, 8)
    ax1.set_yticks(y_ticks)

    # global plot settings
    plt.xticks(n_components)
    plt.xlabel("num components")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    # return the best number
    return n_components_best
