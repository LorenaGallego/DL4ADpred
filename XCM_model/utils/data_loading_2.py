import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, KFold
import itertools
import pandas as pd
from sklearn.model_selection import train_test_split


def transform_labels(y_train, y_test):
    """
    Transform label to min equal zero and continuous
    For example if we have [1,3,4] ---> [0,1,2]


    Parameters
    ----------
    y_train: array
        Labels of the train set

    y_test: array
        Labels of the test set


    Returns
    -------
    new_y_train: array
        Transformed y_train array

    new_y_test: array
        Transformed y_test array
    """

    # Initiate the encoder
    encoder = LabelEncoder()

    # Concatenate train and test to fit
    y_train_test = np.concatenate((y_train, y_test), axis=0)

    # Fit the encoder
    encoder.fit(y_train_test.ravel())

    # Transform to min zero and continuous labels
    new_y_train_test = encoder.transform(y_train_test.ravel())

    # Resplit the train and test
    new_y_train = new_y_train_test[0 : len(y_train)]
    new_y_test = new_y_train_test[len(y_train) :]

    return new_y_train, new_y_test


def import_data(dataset, fase, log=print):
    """
    Load and preprocess train and test sets


    Parameters
    ----------
    dataset: string
        Name of the dataset


    Returns
    -------
    X_train: array
        Train set without labels

    y_train: array
        Labels of the train set encoded

    X_test: array
        Test set without labels

    y_test: array
        Labels of the test set encoded

    y_train_nonencoded: array
        Labels of the train set non-encoded

    y_test_nonencoded: array
        Labels of the test set non-encoded
    """

    X_train_folds = []
    X_test_folds = []
    y_train_folds = []
    y_test_folds = []
    y_train_nonencoded_folds = []
    y_test_nonencoded_folds = []

    #load x, y and IDs datasets
    X = np.load("./datasets/" + dataset + "/X_" + fase + ".npy")
    y = np.load("./datasets/" + dataset + "/y_" + fase + ".npy")
    IDs = np.load("./datasets/" + dataset + "/IDs_" + fase + ".npy")

    #to make different folds, we have to divide the different signals by patients
    id_unique, id_counts = np.unique(IDs, return_counts=True)

    X_patients = [[] for i in range(len(id_unique))]
    y_patients = [[] for i in range(len(id_unique))] 

    d = {'index': list(IDs), 'x_data': list(X), 'y_data': list(y)}
    df = pd.DataFrame(d)

    for i, value in enumerate(id_unique):
        X_patients[i] = np.array(df['x_data'][df['index'] == value])
        y_patients[i] = np.array(df['y_data'][df['index'] == value])

    y_unique = [np.unique(i)[0] for i in y_patients] #para transformar la lista de arrays de los y_patients en una unica lista 

    cv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    for fold, (train_index, test_index) in enumerate(cv.split(np.zeros(len(X_patients)), y_unique)):
        print(test_index)
        x_train, x_test = np.array(X_patients)[train_index], np.array(X_patients)[test_index]
        y_train, y_test = np.array(y_patients)[train_index], np.array(y_patients)[test_index]

        
        #to undo patient lists 
        X_train, X_test = np.array(list(itertools.chain(*x_train))), np.array(list(itertools.chain(*x_test)))
        y_train, y_test = np.array(list(itertools.chain(*y_train))), np.array(list(itertools.chain(*y_test)))
        

        # Transform to continuous labels
        y_train, y_test = transform_labels(y_train, y_test)
        y_train_nonencoded, y_test_nonencoded = y_train, y_test

        # One hot encoding of the labels
        enc = OneHotEncoder()
        enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
        y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
        y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

        # Reshape data to match 2D convolution filters input shape
        X_train = np.reshape(
            np.array(X_train),
            (X_train.shape[0], X_train.shape[2], X_train.shape[1], 1),
            order="C",
        )
        X_test = np.reshape(
            np.array(X_test),
            (X_test.shape[0], X_test.shape[2], X_test.shape[1], 1),
            order="C",
        )

        X_train_folds.append(X_train)
        X_test_folds.append(X_test)
        y_train_folds.append(y_train)
        y_test_folds.append(y_test)
        y_train_nonencoded_folds.append(y_train_nonencoded)
        y_test_nonencoded_folds.append(y_test_nonencoded)


        log("\nDataset" + " " + dataset + " " + "loaded")
        log("Training set size: {0}".format(len(X_train)))
        log("Testing set size: {0}".format(len(X_test)))

    return X_train_folds, y_train_folds, X_test_folds, y_test_folds, y_train_nonencoded_folds, y_test_nonencoded_folds
