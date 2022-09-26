"""
Script which iterates over ranges of parameters for each classifier, saving the data as csv files
which can be read in and parsed to make plots and other analysis.
"""

from DecisionTree import DTModel
from Boosting import BoostingModel
from NeuralNetwork import NeuralNetModel
from SVM import SVMModel
from KNN import KNNModel
from DataUtils import LoadData
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# verbosity
verbose=False
datasets = ["water_potability", "heart"]

for dataset in datasets:
    x_train, x_test, y_train, y_test = LoadData(dataset)

    # for storing optimal results
    results = {}

    # init the models and read the data
    dt = DTModel(verbose=verbose)
    dt.ImportData(x_train, y_train, x_test, y_test, dataset_name=dataset)
    boost = BoostingModel(verbose=verbose)
    boost.ImportData(x_train, y_train, x_test, y_test, dataset_name=dataset)
    knn = KNNModel(verbose=verbose)
    knn.ImportData(x_train, y_train, x_test, y_test, dataset_name=dataset)
    svm = SVMModel(verbose=verbose)
    svm.ImportData(x_train, y_train, x_test, y_test, dataset_name=dataset)
    nn = NeuralNetModel(verbose=verbose)
    nn.ImportData(x_train, y_train, x_test, y_test, dataset_name=dataset)

    # loop through each model
    # dt, boost, knn, nn, svm
    models = (dt, boost, knn, nn, svm)
    for model in models:
        # Optimize the model
        model.OptimizeParameters()
        # Create the Validation Curve
        model.GenerateValidationCurve()
        # Create the learning Curve
        model.GenerateLearningCurve()
        # Print out the Best
        print(f"{str(model)} {dataset} results:\n\tBest cv accuracy = {model.cvAccuracy}\n" +
            f"\tTest Accuracy = {model.finalAccuracy}\n\tTrain Time = {model.avgTrainTime}\n\t" +
            f"Predict Time = {model.avgPredictTime}\n\tModel Params: {model.optimal_parameters}")

        # for plots. Use Test Accuracy
        results[str(model)] = {}
        results[str(model)]["CV Accuracy"] = model.cvAccuracy
        results[str(model)]["Accuracy"] = model.finalAccuracy
        results[str(model)]["Train Time"] = model.avgTrainTime
        results[str(model)]["Predict Time"] = model.avgPredictTime

    # plot the comparison table
    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(9, 4)
    fig.set_dpi(120)
    fig.suptitle(f"Test Accuracy and Times of Optimized Model ({dataset})")
    x = ["DT", "Boost", "KNN", "NeuralNet", "SVM"]
    y = [results[str(m)]["Accuracy"] for m in models]
    axes[0].bar(x, y)
    axes[0].grid()
    axes[0].set_title("Test Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticklabels(x, rotation = 45)
    y = [results[str(m)]["Train Time"] for m in models]
    axes[1].bar(x, y)
    axes[1].grid()
    axes[1].set_title("Training Time (s)")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xticklabels(x, rotation = 45)
    y = [results[str(m)]["Predict Time"] for m in models]
    axes[2].bar(x, y)
    axes[2].grid()
    axes[2].set_title("Prediction Time (s)")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_xticklabels(x, rotation = 45)
    plt.savefig(f"outputs/PerformanceComparison_{dataset}.png")
    plt.close()



print("#\n#\n#\nCOMPLETE!!!\n#\n#\n#")
