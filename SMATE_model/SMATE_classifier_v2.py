import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

from utils.UEA_utils import *

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, KFold
from sklearn.model_selection import train_test_split

from SMATE_model2 import SMATE

import time, math, json
import numpy as np
import pandas as pd
import random as rd
import argparse, configparser
import itertools


import tensorflow as tf
import tensorflow.compat.v1.keras.backend as K

from tensorflow.core.protobuf import rewriter_config_pb2

K.clear_session()
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
off = rewriter_config_pb2.RewriterConfig.OFF
config.graph_options.rewrite_options.arithmetic_optimization = off
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)



# add parameters

def get_parameters():
        parser = argparse.ArgumentParser(description="SMATE for Semi-Supervised MTS Learning")
        parser.add_argument('--config', type=str, default='utils/param.conf',
                            help='the parameter configuration file')
        parser.add_argument('--enable_cuda', type=bool, default='True',
                            help='enable CUDA, default as True')
        parser.add_argument('--sup_ratio', type=float, default=1.0,
                            help='the supervised ratio, default as 1.0')
        parser.add_argument('--epochs', type=int, default=1000,
                            help='epochs, default as 1000')
        parser.add_argument('--rep_main', type=str, default='Datasets/MTS-UEA/',
                            help='the root path of datasets')
        parser.add_argument('--ds_name', type=str, default='CharacterTrajectories',
                            help='the UEA MTS dataset name')
        parser.add_argument('--outModelFile', type=str, default='./out_new_ss/best_model_N3_sk_01',
                            help='save the model weights')
        
        args = parser.parse_args()
        config = configparser.ConfigParser()
        config.read(args.config)
        print('Training configs: {}'.format(args))
        return args, config

def prepare_data(rep_main, ds_name, sup_ratio, norm):
        rep_ds_train = rep_main + ds_name + "/output_train/"
        rep_ds_test = rep_main + ds_name + "/output_test/"
        meta_csv = "meta_data.csv"  # the meta data of training/testing set
        rep_output = rep_ds_train + "out/"  # output results, e.g., training loss, models
        os.system("mkdir -p " + rep_output)

        dataset = get_UEA_dataset(rep_ds_train, rep_ds_test, meta_csv, sup_ratio, norm)

        x_train = dataset['X_train']
        x_test = dataset['X_test']
        x_sup = dataset['X_sup']  # 3-D Array: N * T * M
        x_unsup = dataset['X_unsup']

        # Bacis Dataset Information and Model Configurations
        train_size = x_train.shape[0]
        T = x_train.shape[1]
        M = x_train.shape[2]
        n_class = dataset['n_classes']
        label_size = x_sup.shape[0]
        unlabel_size = x_unsup.shape[0]
        print("n_train is", train_size, "; n_test =", x_test.shape[0], "; T =", T, "; M =", M,
              "; n_class =",n_class)
        return dataset, T, M, n_class, label_size, unlabel_size

if __name__ == "__main__":
    args, config = get_parameters()

    n_epochs = args.epochs
    rep_main = args.rep_main
    ds_name = args.ds_name
    sup_ratio = args.sup_ratio

    # read the parameters for each dataset
    ds_config = config[ds_name]

    kernels = json.loads(ds_config['kernels'])
    p_ratio = float(ds_config['p_ratio'])
    d_prime_ratio = float(ds_config['d_prime_ratio'])
    norm = json.loads(ds_config['normalisation'].lower())
    patience = int(ds_config['patience'])
    val_ratio = float(ds_config['val_ratio'])
    
    #load x, y and IDs datasets
    X = np.load('Datasets/MTS-UEA/' + ds_name + '/X_N3.npy')
    y = np.load('Datasets/MTS-UEA/' + ds_name + '/y_N3.npy')
    IDs = np.load('Datasets/MTS-UEA/' + ds_name + '/IDs_N3.npy')

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

    #5-fold stratiffied cross validation. First we divide into train and test and later the train set is divided in train and validation
    cv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    for fold, (train_index, test_index) in enumerate(cv.split(np.zeros(len(X_patients)), y_unique)):
        print(test_index)
        x_train, x_test = np.array(X_patients)[train_index], np.array(X_patients)[test_index]
        y_train, y_test = np.array(y_patients)[train_index], np.array(y_patients)[test_index]

        y_unique = [np.unique(i)[0] for i in y_train]

        train_index, val_index = train_test_split(train_index, test_size=0.2, random_state=0, stratify=y_unique)

        print(train_index, val_index)

        x_train, x_val = np.array(X_patients)[train_index], np.array(X_patients)[val_index]
        y_train, y_val = np.array(y_patients)[train_index], np.array(y_patients)[val_index]

        #to undo patient lists created to make train, val and test division 
        x_train, x_test = np.array(list(itertools.chain(*x_train))), np.array(list(itertools.chain(*x_test)))
        y_train, y_test = np.array(list(itertools.chain(*y_train))), np.array(list(itertools.chain(*y_test)))
        x_val, y_val= np.array(list(itertools.chain(*x_val))), np.array(list(itertools.chain(*y_val)))
        

        if sup_ratio==1.0:    
            x_unsup = np.array([])
            y_unsup = np.array([])
            x_sup = x_train
            y_sup = y_train
            
        else:
            
            #Split training dataset into sup and unsup depending on sup_ratio
            sss = StratifiedShuffleSplit(n_splits=1, train_size=sup_ratio, random_state=0)
            for sup_index, unsup_index in sss.split(np.zeros(x_train.shape[0]), y_train):
                x_sup, x_unsup = x_train[sup_index,:,:], x_train[unsup_index,:,:]
                y_sup, y_unsup = np.array(y_train)[sup_index], np.array(y_train)[unsup_index]
                print("X train shape", x_train.shape)
                print("X sup shape", x_sup.shape)
                print("X unsup shape", x_unsup.shape)
            
        T = x_train.shape[1]
        M = x_train.shape[2]
        n_class = 2
        label_size = x_sup.shape[0]
        unlabel_size = x_unsup.shape[0]

        '''=================================================== Model Training ========================================================'''
        # Build SMATE model
        outModelFile_Fold = args.outModelFile + '_fold{}'.format(fold) #to save each fold in a different file
        smate = SMATE(T, M, n_class, label_size, unlabel_size,
                    y_sup, sup_ratio, p_ratio, d_prime_ratio, kernels, outModelFile_Fold)

        smate.build_model()
        # smate.model.summary()
        # Train SMATE model
        t1 = time.time()
        #smate.model.load_weights(args.outModelFile)
               
        current_epoch =0
        print('==== Training ====')
        smate.fit(n_epochs, x_train, x_sup, x_unsup, val_ratio, patience, current_epoch, x_val, y_val)
        print("training time is {}".format(time.time() - t1))
        smate.model.load_weights(outModelFile_Fold)
        
        # =================================================== Model Testing ========================================================
        print('======= Fold {} ========'.format(fold))
        # Test SMATE model on both supervised and semi-supervised classification
        smate.predict(x_train, y_train, x_test, y_test,  x_val, y_val)
        #smate.predict_ssl(x_sup, y_sup, x_unsup, y_unsup, x_test, y_test) #uncomment for semi-supervised prediction  

        '''
        #UNCOMMENT FOR T-SNE VISUALIZATION
        name_original = 'original_graph_REM_db_fold2_H'
        name_encoder = 'encoder_graph_REM_db_fold2_H'
        title = '-N3-'
        smate.tSNE_visual(x_test,y_test,name_original,name_encoder,title)
        '''
    
       
    
    K.clear_session()




       