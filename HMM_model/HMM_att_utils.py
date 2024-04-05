import numpy as np
import scipy.sparse as sp
import sklearn
import sklearn.metrics
import torch
import pandas as pd
import random
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, KFold
import itertools
import pickle
import random
from pyhhmm.gaussian import GaussianHMM
import pyhhmm.utils as hu
from HMM_features import *
import os

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def loaddata(filename):
    df = pd.read_csv(filename, header=None, delimiter=",")
    a = np.array(df.as_matrix())
    return a

def HMM_train(n_states, n_emissions, c_features, outModelFile_Fold, n_init = 1, n_iterations = 100):

    try: 
        hu.load_model(outModelFile_Fold)
    except:
        # instantiate a MultinomialHMM object
        trained_hmm = GaussianHMM(
            # number of hidden states
            n_states=n_states,
            # number of distinct emissions
            n_emissions=n_emissions,
            # can be 'diagonal', 'full', 'spherical', 'tied'
            covariance_type='diagonal',
            verbose=True
        )
        '''
        trained_hmm.pi = np.array([0.5, 0.3, 0.1, 0.1])

        trained_hmm.A = np.array(
            [
                [0.7, 0.1, 0.1, 0.1],
                [0.2, 0.5, 0.2, 0.1],
                [0.1, 0.3, 0.5, 0.1],
                [0.2, 0.1, 0.1, 0.6],
            ])

       # trained_hmm.means = np.array([[0.0, 0.0], [0.0, 11.0], [9.0, 10.0], [11.0, -1.0]])
        # the covariance of each component - shape depends `covariance_type`
        #             (n_states, )                          if 'spherical',
        #             (n_states, n_emissions)               if 'diagonal',
        #             (n_states, n_emissions, n_emissions)  if 'full'
        #             (n_emissions, n_emissions)            if 'tied'
        trained_hmm.covars = 0.5 * np.ones((4, 3))  # diagonal covariance matrix

        '''

        # reinitialise the parameters and see if we can re-learn them
        trained_hmm, log_likelihoods = trained_hmm.train(
            c_features,
            n_init=n_init,     # number of initialisations to perform
            n_iter=n_iterations,   # maximum number of iterations to run
            conv_thresh=0.001,  # what percentage of change in the log-likelihood between iterations is considered convergence
            conv_iter=5,  # for how many iterations does it have to hold
            # whether to plot the evolution of the log-likelihood over the iterations
            plot_log_likelihood=True,
            # set to True if want to train until maximum number of iterations is reached
            ignore_conv_crit=False,
            no_init=False,  # set to True if the model parameters shouldn't be re-initialised befor training; in this case they have to be set manually first, otherwise errors occur
        )

        with open(outModelFile_Fold, 'wb') as f:
            pickle.dump(trained_hmm, f)

        return trained_hmm


def output2(trained_hmm, c_features_train, c_features_test):
    print(hu.pretty_print_hmm(trained_hmm, hmm_type='Gaussian'))
    
    #prediction of the HMM probabilities for each set
    pred_train =trained_hmm.predict_proba(c_features_train)#, algorithm='viterbi')
    pred_test =trained_hmm.predict_proba(c_features_test)#, algorithm='viterbi')
    
    X_train = np.array(pred_train)
    X_test = np.array(pred_test)
    
    print(X_train.shape)
    print(X_test.shape)

    return X_train, X_test

def load_features(filepath, fold, data):
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            features = pickle.load(f)
    else:
        print('Extracting features for fold {}'.format(fold))
        features = concatenate_features(data)
        with open(filepath, 'wb') as f:
            pickle.dump(features, f)
    return features

def load_raw_ts(path, dataset,fase, tensor_format=True,n_states= 5, n_emissions = 4, out = './out/trainer_HMM_feat2_st5_model2_REM'):
    path = path + "raw/" + dataset + "/"

    ts_folds = []
    labels_folds = []
    idx_train_folds = []
    idx_val_folds = []
    idx_test_folds = []
    nclass_folds = []

    #load x, y and IDs datasets
    
    X = np.load('Datasets/MTS-UEA/PSG/X_REM.npy')
    y = np.load('Datasets/MTS-UEA/PSG/y_REM.npy')
    IDs = np.load('Datasets/MTS-UEA/PSG/IDs_REM.npy')

    #to make different folds, we have to divide the different signals by patients
    id_unique, id_counts = np.unique(IDs, return_counts=True)

    X_patients = [[] for i in range(len(id_unique))]
    y_patients = [[] for i in range(len(id_unique))] 

    d = {'index': list(IDs), 'x_data': list(X), 'y_data': list(y)}
    df = pd.DataFrame(d)

    for i, value in enumerate(id_unique):
        X_patients[i] = np.array(df['x_data'][df['index'] == value])
        y_patients[i] = np.array(df['y_data'][df['index'] == value])

    y_unique = [np.unique(i)[0] for i in y_patients] #to transform the list of arrays of the y_patients into a single list 


    cv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    for fold, (train_index, test_index) in enumerate(cv.split(np.zeros(len(X_patients)), y_unique)):
        x_train, x_test = np.array(X_patients)[train_index], np.array(X_patients)[test_index]
        y_train, y_test = np.array(y_patients)[train_index], np.array(y_patients)[test_index]
        #to undo patient lists 
        x_train, x_test = np.array(list(itertools.chain(*x_train))), np.array(list(itertools.chain(*x_test)))
        y_train, y_test = np.array(list(itertools.chain(*y_train))), np.array(list(itertools.chain(*y_test)))
        

        #feature extraction
        filepath_train = './features/hjorth_mean_train3_REM_fold{}'.format(fold)
        filepath_test= './features/hjorth_mean_test3_REM_fold{}'.format(fold)
        
        x_train = load_features(filepath_train, fold, x_train)
        x_test = load_features(filepath_test, fold, x_test)

        #train the HMM model 
        outModelFile_Fold = out + '_fold{}'.format(fold)
        trained_hmm = HMM_train(n_states, n_emissions, x_train, outModelFile_Fold, n_init = 1, n_iterations = 100)
        trained_hmm = hu.load_model(outModelFile_Fold)
        x_train, x_test = output2(trained_hmm, x_train, x_test) #to model the signals into probabilities with the trained HMM model


        ts = np.concatenate((x_train, x_test), axis=0)
        ts = np.transpose(ts, axes=(0, 2, 1))
        labels = np.concatenate((y_train, y_test), axis=0)
        nclass = int(np.amax(labels)) + 1

        train_size = y_train.shape[0]

        total_size = labels.shape[0]
        idx_train = range(train_size)
        idx_val = range(train_size, total_size)
        idx_test = range(train_size, total_size)

        if tensor_format:
            # features = torch.FloatTensor(np.array(features))
            ts = torch.FloatTensor(np.array(ts))
            labels = torch.LongTensor(labels)

            idx_train = torch.LongTensor(idx_train)
            idx_val = torch.LongTensor(idx_val)
            idx_test = torch.LongTensor(idx_test)

            ts_folds.append(ts)
            labels_folds.append(labels)
            idx_train_folds.append(idx_train)
            idx_val_folds.append(idx_val)
            idx_test_folds.append(idx_test)

    return ts_folds, labels_folds, idx_train_folds, idx_val_folds, idx_test_folds, nclass


def normalize(mx):
    """Row-normalize sparse matrix"""
    row_sums = mx.sum(axis=1)
    mx = mx.astype('float32')
    row_sums_inverse = 1 / row_sums
    f = mx.multiply(row_sums_inverse)
    return sp.csr_matrix(f).astype('float32')


def accuracy(output, labels):
    preds = output.max(1)[1].cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy_score = (sklearn.metrics.accuracy_score(labels, preds))

    return accuracy_score

def f1_score(output, labels):
    preds = output.max(1)[1].cpu().numpy()
    labels = labels.cpu().numpy()
    f1 = (sklearn.metrics.f1_score(labels, preds, average = 'weighted'))
	
    return f1

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def output_conv_size(in_size, kernel_size, stride, padding):

    output = int((in_size - kernel_size + 2 * padding) / stride) + 1

    return output

def dump_embedding(proto_embed, sample_embed, labels, dump_file='./plot/embeddings.txt'):
    proto_embed = proto_embed.cpu().detach().numpy()
    sample_embed = sample_embed.cpu().detach().numpy()
    embed = np.concatenate((proto_embed, sample_embed), axis=0)

    nclass = proto_embed.shape[0]
    labels = np.concatenate((np.asarray([i for i in range(nclass)]),
                             labels.squeeze().cpu().detach().numpy()), axis=0)

    with open(dump_file, 'wb') as f:
        for i in range(len(embed)):
            label = str(labels[i])
            line = label + "," + ",".join(["%.4f" % j for j in embed[i].tolist()])
            f.write(line + '\n')
