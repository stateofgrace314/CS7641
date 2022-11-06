"""
Main script to run all the experiments generating plots and data for P3
"""

from DataUtils import LoadData
from ClusteringAnalysis import runClusteringAnalysis


for dataset in ["water_potability", "heart"]:
    # use the entire dataset, no test set
    data = LoadData(dataset, 1)
    x_set = data[0]
    y_set = data[2]
    # shorten name for water dataset
    if dataset == "water_potability":
        dataset = "water"

    # Experiment 1
    runClusteringAnalysis(x_set, y_set, f"{dataset}_base")
