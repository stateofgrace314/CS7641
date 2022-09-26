"""
Method to train and run a KNN model on an input dataset
Implemented primarily using SKlearn KNeighborsClassifier
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from BaseClassifier import BaseClassifier
import numpy as np

class KNNModel(BaseClassifier):
    """
    KNN Classifier class
    """
    # parameters to keep constant
    weights = "uniform"
    algorithm = "auto"
    metric = "minkowski"
    leaf_size = 30

    # Parameters that need to be optimized
    # n_neighbors
    # p

    def __init__(self, verbose=False) -> None:
        # Initialize from super
        super(KNNModel, self).__init__(verbose=verbose)
        # init the classifier
        self.classifier = KNeighborsClassifier(
            weights=self.weights,
            algorithm=self.algorithm,
            metric=self.metric,
            n_jobs=self.n_jobs)
        self.pipeline = Pipeline([("scaler", None), ("estimator", clone(self.classifier))])
        self.LCTitle = "K Nearest Neighbor Learning Curve"
        self.VCTitle = "K Nearest Neighbor Validation Curve"
        self.prefix = "KNN_"
        self.parameters_to_tune = ["n_neighbors", "p"]
        self.grid = {
            "scaler": [StandardScaler()],
            "estimator": [clone(self.classifier)],
            "estimator__n_neighbors": np.arange(1, 26),
            "estimator__p": np.arange(1, 6)
        }
        self.optimal_parameters = {
            "n_neighbors": 0,
            "p": 0
        }

    def __repr__(self):
        return "KNN"

# Driver code
def main():
    """
    Main function if file is being run independently
    """

    return

# Calling main function
if __name__=="__main__":
    main()

