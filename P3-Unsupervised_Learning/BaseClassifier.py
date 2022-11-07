"""
The base model for all classifiers being used in GATech ML, CS7641
This class is to be used as the basic API for all classifiers using sklearn as the
foundation for all methods being used.
"""

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import validation_curve, learning_curve, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from copy import deepcopy
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

class BaseClassifier:
    """
    BaseClassifier is essentially just an API definition for more specific models
    """
    # some default parameters
    n_jobs = 12
    random_state = 0
    scoring = "accuracy"
    cv = 5

    def __init__(self, verbose=False) -> None:
        # init the classifier - needs to be replaced by child classes
        self.pipeline = Pipeline([("scaler", None), ("estimator", None)])
        self.classifier = None
        self.dataset = ""
        # init attributes
        self.verbose=verbose
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.LCTitle = ""
        self.VCTitle = ""
        self.prefix = ""
        self.parameters_to_tune = []
        self.grid = {}
        self.optimal_parameters = {}
        self.avgTrainTime = 0
        self.avgPredictTime = 0
        self.avgTrainScore = 0
        self.avgTestScore = 0
        self.cvAccuracy = 0
        self.finalAccuracy = 0
        # some progress flags
        self.dataImported = False
        self.optimized = False

    def __repr__(self):
        return "BaseClassifier"

    def ImportData(self, x_train, y_train, x_test, y_test, dataset_name):
        """
        Import data into the model. The data imported must already be split into test and training
        sets, and have any pre-processing steps complete.

        Parameters
        ----------
        x_train (pd.DataFrame):
        y_train (pd.DataFrame):
        x_test (pd.DataFrame):
        y_test (pd.DataFrame):

        Returns
        -------
        None
        """
        self.x_train = deepcopy(x_train)
        self.y_train = deepcopy(y_train)
        self.x_test = deepcopy(x_test)
        self.y_test = deepcopy(y_test)
        self.dataset = dataset_name
        self.dataImported = True

    def OptimizeParameters(self):
        """
        Loops through some pre-determined parameter ranges to find the set of parameters that has
        the best cross-validation score
        """
        cv_results = GridSearchCV(clone(self.pipeline), self.grid, cv=self.cv, refit=True,
            verbose=self.verbose, scoring=self.scoring, n_jobs=self.n_jobs, return_train_score=True)
        cv_results.fit(self.x_train, self.y_train)

        # store important information for processing results
        columns = [f"param_{p}" for p in self.parameters_to_tune]
        columns += ["mean_fit_time", "mean_score_time", "mean_train_score", "mean_test_score",
            "std_fit_time", "std_score_time", "std_test_score", "std_train_score" ]
        out = pd.DataFrame(cv_results.cv_results_, columns=columns)
        out.to_csv(f"plots/{self.prefix}ParameterOptimization_{self.dataset}.csv")

        # get the best model based on min delta between train and test
        self.classifier = clone(cv_results.best_estimator_.named_steps["estimator"])
        self.optimized = True
        for p in self.parameters_to_tune:
            self.optimal_parameters[p] = cv_results.best_params_[f"estimator__{p}"]
        self.cvAccuracy = cv_results.best_score_
        self.avgTrainTime = cv_results.cv_results_["mean_fit_time"][cv_results.best_index_]
        self.avgPredictTime = cv_results.cv_results_["mean_score_time"][cv_results.best_index_]
        self.Train()
        self.finalAccuracy = self.PredictAndMeasure()

    def GenerateValidationCurve(self):
        """
        Generate and plot the Validation curve for the model
        """
        # get some information on the parameters
        num_params = len(self.parameters_to_tune)

        # setup the plot
        fig, axes = plt.subplots(num_params, 1)
        fig.set_size_inches(9, 9)
        fig.set_dpi(120)
        fig.suptitle(self.VCTitle + f"\n({self.dataset})")

        # Get the VC data for each parameter and add it to the plot
        i = 0
        for p, p_range in self.grid.items():
            # only plot the value if it's an estimator parameter
            if "estimator__" not in p:
                continue
            train_scores, test_scores = validation_curve(clone(self.pipeline), self.x_train,
                self.y_train, scoring=self.scoring, cv=self.cv, param_name=p,
                param_range=p_range, n_jobs=self.n_jobs)
            train_scores_avg = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_avg = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            # plot the values
            x_labels = []
            for param in p_range:
                if isinstance(param, tuple):
                    x_labels.append(str(param))
                else:
                    x_labels.append(param)
            axes[i].plot(x_labels, train_scores_avg, "b", x_labels, test_scores_avg, "r")
            axes[i].fill_between(x_labels, train_scores_avg - train_scores_std,
                                 train_scores_avg + train_scores_std, alpha=0.2, color="blue", lw=2)
            axes[i].fill_between(x_labels, test_scores_avg - test_scores_std,
                                 test_scores_avg + test_scores_std, alpha=0.2, color="red", lw=2)
            axes[i].grid()
            axes[i].set_xlabel(f"{p[11:]} value")
            axes[i].set_ylabel("Accuracy Score")
            axes[i].set_xticks(x_labels)
            axes[i].legend(["Training set", "Test set"])
            # iterate
            i += 1
        # save and close
        plt.savefig(f"plots/{self.prefix}ValidationCurve_{self.dataset}.png", bbox_inches="tight")
        plt.close()


    def GenerateLearningCurve(self):
        """
        Generates a Learning Curve
        """
        train_sizes, train_scores, test_scores, fit_times, score_times = \
            learning_curve(Pipeline([("scaler", None), ("estimator", clone(self.classifier))]),
                self.x_train, self.y_train, train_sizes=np.linspace(0.1, 1.0, 9),
                scoring=self.scoring, random_state=self.random_state, return_times=True)
        # get avg and std for plots
        train_scores_avg = train_scores.mean(axis=1)
        test_scores_avg = test_scores.mean(axis=1)
        fit_times_avg = fit_times.mean(axis=1)
        score_times_avg = score_times.mean(axis=1)
        train_scores_std = train_scores.std(axis=1)
        test_scores_std = test_scores.std(axis=1)
        fit_times_std = fit_times.std(axis=1)
        score_times_std = score_times.std(axis=1)
        # make the plot
        fig, axes = plt.subplots(2, 1)
        fig.set_size_inches(9, 9)
        fig.set_dpi(120)
        fig.suptitle(self.LCTitle + f" ({self.dataset})")
        # plot 1, the accuracy
        axes[0].plot(train_sizes, train_scores_avg, "b",
                     train_sizes, test_scores_avg, "r")
        axes[0].fill_between(train_sizes, train_scores_avg - train_scores_std,
            train_scores_avg + train_scores_std, alpha=0.2, color="blue", lw=2)
        axes[0].fill_between(train_sizes, test_scores_avg - test_scores_std,
            test_scores_avg + test_scores_std, alpha=0.2, color="red", lw=2)
        axes[0].grid()
        axes[0].set_xlabel("Size of Training Data")
        axes[0].set_ylabel("Accuracy Score")
        axes[0].legend(["training set", "test set"])
        # plot 2, times
        axes[1].plot(train_sizes, fit_times_avg, "b",
                     train_sizes, score_times_avg, "r")
        axes[1].fill_between(train_sizes, fit_times_avg - fit_times_std,
            fit_times_avg + fit_times_std, alpha=0.2, color="blue", lw=2)
        axes[1].fill_between(train_sizes, score_times_avg - score_times_std,
            score_times_avg + score_times_std, alpha=0.2, color="red", lw=2)
        axes[1].grid()
        axes[1].set_xlabel("Size of Training Data")
        axes[1].set_ylabel("Time (s)")
        axes[1].legend(["Average Fit Time", "Average Score Time"])
        # save and close
        plt.savefig(f"plots/{self.prefix}LearningCurve_{self.dataset}.png", bbox_inches="tight")
        plt.close()

    def Train(self):
        """
        Train the data. Requires a dataset to have already been imported.

        Returns
        -------
        None
        """
        self.classifier.fit(self.x_train, self.y_train)
        return

    def Predict(self, x_test=None):
        """
        Predict the labels for the test set (or input data)

        Parameters
        ----------
        x_test(pd.DataFrame):   The test data to classify. If None, use the stored testing dataset

        Returns
        -------
        (array):    The predicted classes/values.
        """
        if x_test is None:
            x_test = self.x_test
        return self.classifier.predict(x_test)

    def CalculateAccuracy(self, y_pred, y_test=None):
        """
        Calculate the accuracy of the model predictions

        Parameters
        ----------
        y_pred(array):  An array of the predictions made by the model
        y_test(array):  An array of the actual values. If None, use the stored test set

        Returns
        -------
        (float):    The accuracy in percentage.
        """
        if y_test is None:
            y_test = self.y_test
        accuracy = accuracy_score(y_test,y_pred)*100
        if self.verbose:
            report = f"Model parameters: {self.classifier}\n"
            report += f"Accuracy against test set: {accuracy}\n"
            report += classification_report(y_test, y_pred)
            print(report)
        return accuracy

    def PredictAndMeasure(self, x_test=None, y_test=None):
        """
        Runs prediction on the test data, then compares it with the actual values to get an accuracy
        percentage.

        Parameters
        ----------
        x_test(pd.DataFrame):   The test data to classify. If None, use the stored testing dataset

        Returns
        -------
        (float):    The accuracy of the classifier in percent.
        """
        if x_test is None:
            x_test = self.x_test
        y_pred = self.Predict(x_test)

        if y_test is None:
            y_test = self.y_test
        return self.CalculateAccuracy(y_pred, y_test)
