"""
Function to optimize neural network weights using RO algorithms
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from mlrose_hiive.neural import NeuralNetwork
from mlrose_hiive import ExpDecay
from time import time

def runNNOptimizers():
    """
    Function to optimize neural network weights using mlrose algorithms
    """
    print("Running NN using random optimizers")
    # load the data
    data_file = "../datasets/water_quality/water_potability.csv"
    data = pd.read_csv(data_file).dropna()
    X =  data.iloc[:,:-1]
    Y = data.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    # neural net params
    hidden_layer = (10,)
    random_state = 0
    max_iter = 1000
    alpha = 0.00002
    activation = "relu"

    # setup standard neural net for comparison
    std_net = Pipeline([('normalizer', StandardScaler()),
                        ('clf', MLPClassifier(hidden_layer_sizes=hidden_layer,
                                              activation="relu",
                                              alpha=alpha,
                                              random_state=random_state,
                                              max_iter=max_iter))])
    # setup RhC
    rhc_net = NeuralNetwork(hidden_nodes=hidden_layer,
                            activation=activation,
                            algorithm="random_hill_climb",
                            max_iters=2500,
                            max_attempts=2500,
                            restarts=19,
                            bias=True,
                            learning_rate=0.001,
                            is_classifier=True,
                            early_stopping=True,
                            random_state=random_state,
                            curve=True)
    # setup SA
    decay_schdule = ExpDecay(init_temp=100, min_temp=0.001, exp_const=0.5)
    sa_net = NeuralNetwork(hidden_nodes=hidden_layer,
                           activation=activation,
                           algorithm="simulated_annealing",
                           max_iters=5000,
                           max_attempts=5000,
                           bias=True,
                           learning_rate=0.001,
                           schedule=decay_schdule,
                           is_classifier=True,
                           early_stopping=True,
                           random_state=random_state,
                           curve=True)
    # setup GA
    ga_net = NeuralNetwork(hidden_nodes=hidden_layer,
                           activation=activation,
                           algorithm="genetic_alg",
                           max_iters=500,
                           max_attempts=500,
                           pop_size=200,
                           mutation_prob=0.1,
                           bias=True,
                           learning_rate=0.001,
                           is_classifier=True,
                           early_stopping=True,
                           random_state=random_state,
                           curve=True)
    # output containers
    scores = {}
    train_times = {}
    query_times = {}
    iters = {}
    y_pred = {}

    # Train and time each NN
    # run standard
    print("    Training standard neural network")
    start = time()
    std_net.fit(x_train, y_train)
    train_times["STD"] = time() - start
    iters["STD"] = std_net._final_estimator.n_iter_
    # run rhc
    print("    Training RHC neural network")
    start = time()
    rhc_net.fit(x_train, y_train)
    iters["RHC"] = rhc_net.fitness_curve.shape[0]
    train_times["RHC"] = time() - start
    # run sa
    print("    Training SA neural network")
    start = time()
    sa_net.fit(x_train, y_train)
    iters["SA"] = sa_net.fitness_curve.shape[0]
    train_times["SA"] = time() - start
    # run ga
    print("    Training GA neural network")
    start = time()
    ga_net.fit(x_train, y_train)
    iters["GA"] = ga_net.fitness_curve.shape[0]
    train_times["GA"] = time() - start
    # Query and time each NN
    # run std
    print("    Querying standard neural network")
    start = time()
    y_pred = std_net.predict(x_test)
    query_times["STD"] = time() - start
    scores["STD"] = accuracy_score(y_test, y_pred)
    # run rhc
    print("    Querying RHC neural network")
    start = time()
    y_pred = rhc_net.predict(x_test)
    query_times["RHC"] = time() - start
    scores["RHC"] = accuracy_score(y_test, y_pred)
    # run sa
    print("    Querying SA neural network")
    start = time()
    y_pred = sa_net.predict(x_test)
    query_times["SA"] = time() - start
    scores["SA"] = accuracy_score(y_test, y_pred)
    # run ga
    print("    Querying GA neural network")
    start = time()
    y_pred = ga_net.predict(x_test)
    query_times["GA"] = time() - start
    scores["GA"] = accuracy_score(y_test, y_pred)

    # print results
    print("Neural Network test complete")
    print(f"Accuracy scores: {scores}")
    print(f"Train Times: {train_times}")
    print(f"Query Times: {query_times}")
    print(f"Iterations: {iters}")

if __name__ == "__main__":
    runNNOptimizers()
