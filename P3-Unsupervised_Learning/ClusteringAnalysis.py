"""
Implementation of experiment 1 - Run clustering using KMeans and Expectation Maximization
Analysis done using Silhouette and Elbow graphs
"""

from sklearn.base import ClusterMixin
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, homogeneity_score, silhouette_score
from yellowbrick.cluster import KElbowVisualizer
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from DataUtils import LoadData

OUT_DIR = f"{Path(__file__).parent.resolve()}/plots"

class GMYBWrapper(GaussianMixture, ClusterMixin):
    """
    New class which is basically a wrapper wround the sklearn GaussianMixture class to make it
    compatible with Yellowbricks visualizer functions.
    Wrapper idea taken from Chris Vandevelde:
        https://stackoverflow.com/questions/58648969/the-supplied-model-is-not-a-clustering-estimator-in-yellowbrick
    """

    def fit(self, X):
        super().fit(X)
        self.labels_ = self.predict(X)
        return self

    def get_params(self, **kwargs):
        output = super().get_params(**kwargs)
        output["n_clusters"] = output.get("n_components", None)
        return output

    def set_params(self, **kwargs):
        kwargs["n_components"] = kwargs.pop("n_clusters", None)
        return super().set_params(**kwargs)

def runClusteringAnalysis(X, Y, dataset_name, metric="distortion"):
    """
    Runs analysis on the input dataset over a range of cluster values.
    Includes elbow graph and silhouette, homogeneity, and adjusted rand score
    """
    # loop through the two clustering models
    for func, model_name in [(KMeans, "KMeans"), (GMYBWrapper, "EM")]:
        name = f"{model_name}_{dataset_name}"
        model = func()
        clusters = (2,20)
        elbowViz = KElbowVisualizer(model, k=clusters, metric=metric, force_model=True)

        # Plot the elbow curve
        plt.figure()
        filename = f"{OUT_DIR}/Elbow-{name}.png"
        elbowViz.fit(X)
        elbowViz.show(outpath=filename)
        elbow = elbowViz.elbow_value_
        print(f"found elbow at {elbow}")
        plt.close()

        # plot the homogeneity, silhouette, and adjusted random score
        clusters = np.arange(2,21)
        xticks = np.arange(2,21,2)
        sil_scores = []
        adj_ran_scores = []
        homo_scores = []

        for cluster in clusters:
            model = KMeans(random_state=0, n_clusters=cluster)
            model.fit(X)
            y_pred = model.predict(X)
            ars = adjusted_rand_score(Y, y_pred)
            homo = homogeneity_score(Y, y_pred)
            sil = silhouette_score(X, y_pred)
            adj_ran_scores.append(ars)
            homo_scores.append(homo)
            sil_scores.append(sil)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        title = f"ClusterAnalysis-{name}"
        filename = f"{OUT_DIR}/{title}.png"
        fig.suptitle(title)
        # silhouette
        y_min = min(sil_scores)
        y_max = max(sil_scores)
        diff = y_max - y_min
        y_ticks = np.linspace(y_min - 0.1*diff, y_max + 0.1*diff, 6)
        ax1.set_ylabel("Score", color="b")
        ax1.plot(clusters, sil_scores, "b", label="Silhouette Coef")
        ax1.set_yticks(y_ticks)
        ax1.axvline(x=elbow, color="k", linestyle="--")
        ax1.tick_params(axis="y", labelcolor="b")
        ax1.legend()
        # adj rand score and homogeneity
        y_min = min(adj_ran_scores + homo_scores)
        y_max = max(adj_ran_scores + homo_scores)
        diff = y_max - y_min
        y_ticks = np.linspace(y_min - 0.1*diff, y_max + 0.1*diff, 6)
        ax2.set_ylabel("Score", color="darkred")
        ax2.plot(clusters, adj_ran_scores, "firebrick", label="AdjRandScore")
        ax2.plot(clusters, homo_scores, "orangered", label="Homogeneity")
        ax2.set_yticks(y_ticks)
        ax2.axvline(x=elbow, color="k", linestyle="--")
        ax2.tick_params(axis="y", labelcolor="firebrick")
        ax2.legend()
        # global params
        plt.xticks(xticks)
        plt.xlabel("num clusters")
        plt.grid(visible=True, which="both", axis="x")
        plt.savefig(filename)
        plt.close()

if __name__ == "__main__":
    data = LoadData("water_potability", 1)
    x_set = data[0]
    y_set = data[2]
    runClusteringAnalysis(x_set, y_set, "water_potability")
