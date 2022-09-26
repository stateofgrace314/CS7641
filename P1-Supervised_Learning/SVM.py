"""
Method to train and run a SVM model on an input dataset
Implemented primarily using SKlearn SVM
"""

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from BaseClassifier import BaseClassifier
import numpy as np

class SVMModel(BaseClassifier):
    """
    SVM Classifier class
    """
    # Parameters that won't change
    gamma = "scale"
    tol = 0.001
    cache_size = 1000

    # Parameters that that need to be optimized
    # C
    # degree
    # kernel

    # other parameters I might play around with
    shrinking = True
    max_iter = -1
    decision_function_shape = "ovr"

    def __init__(self, verbose=False) -> None:
        # Initialize from super
        super(SVMModel, self).__init__(verbose=verbose)
        # init the classifier
        self.classifier = SVC(
            gamma=self.gamma,
            tol=self.tol,
            cache_size=self.cache_size,
            random_state=self.random_state,
            shrinking=self.shrinking,
            max_iter=self.max_iter,
            decision_function_shape=self.decision_function_shape,
            verbose=self.verbose)
        self.pipeline = Pipeline([("scaler", None), ("estimator", clone(self.classifier))])
        self.LCTitle = "Support Vector Machine Learning Curve"
        self.VCTitle = "Support Vector Machine Validation Curve"
        self.prefix = "SVM_"
        self.parameters_to_tune = ["C", "degree", "kernel"]
        self.grid = {
            "scaler": [StandardScaler()],
            "estimator": [clone(self.classifier)],
            "estimator__C": np.arange(1, 21, 2),
            "estimator__degree": np.arange(1, 6),
            "estimator__kernel": ("linear", "poly", "rbf", "sigmoid")
        }
        self.optimal_parameters = {
            "C": 0,
            "degree": 0,
            "kernel": ""
        }

    def __repr__(self):
        return "SVM"

# Driver code
def main():
    """
    Main function if file is being run independently
    """

    return

# Calling main function
if __name__=="__main__":
    main()
