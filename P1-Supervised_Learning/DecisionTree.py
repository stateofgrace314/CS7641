"""
Method to train and run a Decision Tree (with pruning) on an input dataset
inspired by: https://www.geeksforgeeks.org/decision-tree-implementation-python/amp/
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from BaseClassifier import BaseClassifier
import numpy as np

class DTModel(BaseClassifier):
    """
    Decision Tree class
    """
    # Parameters that won't change
    criterion = "entropy"
    splitter = "best"
    min_samples_split = 2
    max_leaf_nodes = None

    # Parameters that need to be optimized
    # max_depth
    # min_samples_leaf
    # ccp_alpha

    # Other parameters I might play around with
    min_weight_fraction_leaf = 0
    max_features = None
    min_impurity_decrease = 0

    def __init__(self, verbose=False) -> None:
        # Initialize from super
        super(DTModel, self).__init__(verbose=verbose)
        # init the classifier with the fixed values
        self.classifier = DecisionTreeClassifier(
            criterion=self.criterion,
            splitter=self.splitter,
            min_samples_split=self.min_samples_split,
            max_leaf_nodes=self.max_leaf_nodes,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            min_impurity_decrease=self.min_impurity_decrease,
            random_state=self.random_state)
        self.pipeline = Pipeline([("scaler", None), ("estimator", clone(self.classifier))])
        self.LCTitle = "Decision Tree Learning Curve"
        self.VCTitle = "Decision Tree Validation Curve"
        self.prefix = "DT_"
        self.parameters_to_tune = ["max_depth", "min_samples_leaf", "ccp_alpha"]
        self.grid = {
            "scaler": [StandardScaler()],
            "estimator": [clone(self.classifier)],
            "estimator__max_depth": np.arange(1, 11),
            "estimator__min_samples_leaf": np.arange(1, 51, 2),
            "estimator__ccp_alpha": np.arange(0.0, 0.02001, 0.005)
        }
        self.optimal_parameters = {
            "max_depth": 0,
            "min_samples_leaf": 0,
            "ccp_alpha": 0
        }

    def __repr__(self):
        return "DecisionTree"

# Driver code
def main():
    """
    Main function if file is being run independently
    """

    return

# Calling main function
if __name__=="__main__":
    main()
