"""
Main script to run all the experiments generating plots and data for P3
"""

from DataUtils import LoadData
from ClusteringAnalysis import runClusteringAnalysis
from ReduceDimensions import runPCA, runICA, runRP, runNMF
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.random_projection import GaussianRandomProjection
from NeuralNetwork import NeuralNetModel


def printNNResults(model):
    """
    helper function to print metrics for NN model
    """
    print(f"{str(model)} {dataset} results:\n\tBest cv accuracy = {model.cvAccuracy}\n" +
        f"\tTest Accuracy = {model.finalAccuracy}\n\tTrain Time = {model.avgTrainTime}\n" +
        f"\tPredict Time = {model.avgPredictTime}\n\tModel Params: {model.optimal_parameters}\n")

for dataset in ["water_potability", "heart"]:
    # use the entire dataset, no test set
    data = LoadData(dataset, 0.2)
    x_train = data[0]
    x_test = data[1]
    y_train = data[2]
    y_test = data[3]
    feature_names = data[4]
    # shorten name for water dataset
    if dataset == "water_potability":
        dataset = "water"

    # Experiment 1
    runClusteringAnalysis(x_train, y_train, f"{dataset}_base")

    # Experiment 2
    n_pca = runPCA(x_train, feature_names, dataset)
    n_ica = runICA(x_train, feature_names, dataset)
    n_rp = runRP(x_train, feature_names, dataset)
    n_nmf = runNMF(x_train, feature_names, dataset)

    # Experiment 3
    x_train_pca = PCA(n_components=n_pca, random_state=0).fit_transform(x_train)
    x_test_pca = PCA(n_components=n_pca, random_state=0).fit_transform(x_test)
    runClusteringAnalysis(x_train_pca, y_train, f"{dataset}_PCA")
    x_train_ica = FastICA(n_components=n_ica, random_state=0).fit_transform(x_train)
    x_test_ica = FastICA(n_components=n_ica, random_state=0).fit_transform(x_test)
    runClusteringAnalysis(x_train_ica, y_train, f"{dataset}_ICA")
    x_train_rp = GaussianRandomProjection(n_components=n_rp, random_state=0).fit_transform(x_train)
    x_test_rp = GaussianRandomProjection(n_components=n_rp, random_state=0).fit_transform(x_test)
    runClusteringAnalysis(x_train_rp, y_train, f"{dataset}_RP")
    x_train_nmf = NMF(n_components=n_nmf, random_state=0).fit_transform(x_train)
    x_test_nmf = NMF(n_components=n_nmf, random_state=0).fit_transform(x_test)
    runClusteringAnalysis(x_train_nmf, y_train, f"{dataset}_NMF")

    # Experiment 4
    #base
    nn_base = NeuralNetModel()
    nn_base.ImportData(x_train, y_train, x_test, y_test, f"{dataset}_base")
    nn_base.OptimizeParameters()
    nn_base.GenerateValidationCurve()
    nn_base.GenerateLearningCurve()
    printNNResults(nn_base)
    #PCA
    nn_pca = NeuralNetModel()
    nn_pca.ImportData(x_train_pca, y_train, x_test_pca, y_test, f"{dataset}_PCA")
    nn_pca.OptimizeParameters()
    nn_pca.GenerateValidationCurve()
    nn_pca.GenerateLearningCurve()
    printNNResults(nn_pca)
    #ICA
    nn_ica = NeuralNetModel()
    nn_ica.ImportData(x_train_ica, y_train, x_test_ica, y_test, f"{dataset}_ICA")
    nn_ica.OptimizeParameters()
    nn_ica.GenerateValidationCurve()
    nn_ica.GenerateLearningCurve()
    printNNResults(nn_ica)
    #RP
    nn_rp = NeuralNetModel()
    nn_rp.ImportData(x_train_rp, y_train, x_test_rp, y_test, f"{dataset}_RP")
    nn_rp.OptimizeParameters()
    nn_rp.GenerateValidationCurve()
    nn_rp.GenerateLearningCurve()
    printNNResults(nn_rp)
    #NMF
    nn_nmf = NeuralNetModel()
    nn_nmf.ImportData(x_train_nmf, y_train, x_test_nmf, y_test, f"{dataset}_NMF")
    nn_nmf.OptimizeParameters()
    nn_nmf.GenerateValidationCurve()
    nn_nmf.GenerateLearningCurve()
    printNNResults(nn_nmf)
