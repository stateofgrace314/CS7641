"""
File contains a series of utility functions to support use of the datasets
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import normalize
import pathlib
import os

DATA_DIR = str(pathlib.Path(__file__).parent.parent.resolve()) + "/datasets"

def LoadData(dataset_name:str, test_size:float=0.2):
    """
    Simple function to import a dataset as a pandas dataframe

    Parameters
    ----------
    dataset_name(str):  The name of the dataset to load
    test_size(float):       The proportion of the data which will be put in the test set.

    Returns
    -------
    (pd.DataFrame): The dataset as a Pandas DataFrame
    """
    # find the dataset
    dataset_path = ""
    for dirpath, *_, filenames in os.walk(DATA_DIR):
        for filename in [f for f in filenames]:
            if filename.lower() == dataset_name.lower() + ".csv":
                dataset_path = os.path.join(dirpath, filename)

    # import the csv to a DataFrame and split into features (X) and labels (Y)
    data = pd.read_csv(dataset_path, header=0)
    X = data[data.columns[:-1]].copy()
    Y = data[data.columns[-1]].copy()

    # drop if 2+ missing features in a row
    n_features = X.shape[1]
    thresh = n_features - 1
    X.dropna(thresh=thresh, axis=0, inplace=True)
    indeces = X.index
    Y = Y[indeces]
    X.reset_index(inplace=True)
    X.drop(['index'], axis=1, inplace=True)

    # Split the dataset into train and test
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=100)

    # fill in missing values using knn imputing
    # imputer = KNNImputer(missing_values=np.nan, add_indicator=True)
    imputer = KNNImputer(missing_values=np.nan, add_indicator=False)
    x_train = imputer.fit_transform(x_train)
    x_test = imputer.fit_transform(x_test)

    # normalize numbers (but not the missing indicators)
    # features = x_train[:,:n_features]
    # indicators = x_train[:,n_features:]
    # x_train = np.hstack((normalize(features, axis=0), indicators))
    # features = x_test[:,:n_features]
    # indicators = x_test[:,n_features:]
    # x_test = np.hstack((normalize(features, axis=0), indicators))

    # normalize everything
    # x_train = normalize(x_train, axis=0)
    # x_test = normalize(x_test, axis=0)

    # get the feature names
    feature_names = list(data.columns)[:-1]

    return x_train, x_test, y_train, y_test, feature_names

if __name__ == "__main__":
    out = LoadData("water_potability")

    print("out")
