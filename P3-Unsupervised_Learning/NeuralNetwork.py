"""
Method to train and run a NeuralNet model on an input dataset
Implemented primarily using SKlearn MLPClassifier
"""

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from BaseClassifier import BaseClassifier
from matplotlib import pyplot as plt

class NeuralNetModel(BaseClassifier):
    """
    NeuralNet Classifier class
    """
    # Parameters that won't change
    solver = "adam"
    max_iter = 10000
    tol = 0.00001
    early_stopping = False
    validation_fraction = 0.1
    n_iter_no_change = 20
    activation = "relu"

    # Parameters that need to be optimized
    # hidden_layer_sizes
    # alpha
    # learning_rate_init

    # other parameters I might play around with
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-08

    def __init__(self, verbose=False) -> None:
        # Initialize from super
        super(NeuralNetModel, self).__init__(verbose=verbose)
        # init the pipeline
        self.classifier = MLPClassifier(random_state=self.random_state,
                                        solver=self.solver,
                                        max_iter=self.max_iter,
                                        n_iter_no_change=self.n_iter_no_change,
                                        validation_fraction=self.validation_fraction,
                                        activation=self.activation)
        self.pipeline = Pipeline([("scaler", StandardScaler()), ("estimator", clone(self.classifier))])
        self.LCTitle = "Neural Network Learning Curve"
        self.VCTitle = "Neural Network Validation Curve"
        self.prefix = "NN_"
        self.parameters_to_tune = ["learning_rate_init", "hidden_layer_sizes", "alpha"]
        self.grid = {
            "scaler": [StandardScaler()],
            "estimator": [clone(self.classifier)],
            "estimator__learning_rate_init": [0.0001, 0.0005, 0.001, 0.005],
            "estimator__hidden_layer_sizes": [(2,), (3,), (5,), (10,), (20,), (100,),
                                              (2, 2), (3, 3), (10, 10), (20, 20)],
            "estimator__alpha": [0.0001, 0.0005, 0.001, 0.005]
        }
        self.optimal_parameters = {
            "hidden_layer_sizes": 0,
            "alpha": 0,
            "learning_rate_init": 0
        }

    def __repr__(self):
        return "NeuralNetwork"

    def OptimizeParameters(self):
        """
        For Neural Nets, after optimization is done, also generate the loss plot
        """
        super().OptimizeParameters()
        classifier = clone(self.classifier)
        classifier.fit(self.x_train, self.y_train)
        loss = classifier.loss_curve_
        fig = plt.figure()
        fig.set_size_inches(4,4)
        fig.set_dpi(120)
        plt.plot(loss, label = "loss", color="red")
        plt.title(f"Neural Net Loss Curve ({self.dataset})")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.savefig(f"plots/NN_LossGraph_{self.dataset}.png", bbox_inches="tight")

# Driver code
def main():
    """
    Main function if file is being run independently
    """

    return

# Calling main function
if __name__=="__main__":
    main()
