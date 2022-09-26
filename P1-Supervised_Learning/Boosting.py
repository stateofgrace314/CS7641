"""
Method to train and run a Boosting model on an input dataset
Implemented primarily using SKlearn GradientBoostingClassifier
"""

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from BaseClassifier import BaseClassifier
import numpy as np

class BoostingModel(BaseClassifier):
    """
    Boosting Classifier class
    """
    # Parameters that won't change
    criterion = "entropy"
    max_depth = 3

    # Parameters that that need to be optimized
    # n_estimators
    # learning_rate

    # other parameters I might play around with
    algorithm="SAMME.R"

    def __init__(self, verbose=False) -> None:
        # Initialize from super
        super(BoostingModel, self).__init__(verbose=verbose)
        # init the classifier with the fixed values
        self.classifier = AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(criterion=self.criterion,
                                                  max_depth=self.max_depth),
            algorithm=self.algorithm,
            random_state=self.random_state)
        self.pipeline = Pipeline([("scaler", None), ("estimator", clone(self.classifier))])
        self.LCTitle = "AdaBoost Learning Curve"
        self.VCTitle = "AdaBoost Validation Curve"
        self.prefix = "Boost_"
        self.parameters_to_tune = ["learning_rate", "n_estimators"]
        self.grid = {
            "scaler": [StandardScaler()],
            "estimator": [clone(self.classifier)],
            "estimator__learning_rate": np.arange(0.1, 1.0, 0.1),
            "estimator__n_estimators": np.arange(2, 51, 2)
        }
        self.optimal_parameters = {
            "learning_rate": 0,
            "n_estimators": 0
        }

    def __repr__(self):
        return "AdaBoost"

# Driver code
def main():
    """
    Main function if file is being run independently
    """

    return

# Calling main function
if __name__=="__main__":
    main()
