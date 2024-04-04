from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"

import math
import sys
import time
import argparse

import torch.optim as optim
from models_2 import TapNet
from utils_2 import *
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score

parser = argparse.ArgumentParser()

# dataset settings
parser.add_argument('--data_path', type=str, default="./data/",
                    help='the path of data.')
parser.add_argument('--dataset', type=str, default="PSG", #PSG
                    help='time series dataset.')
parser.add_argument('--fase', type=str, default="REM", #PSG
                    help='Sleep stage dataset')
parser.add_argument('--use_muse', action='store_true', default=True,
                    help='whether to use the raw data. Default:False')

# cuda settings
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

# Training parameter settings
parser.add_argument('--epochs', type=int, default=3000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-5,
                    help='Initial learning rate. default:[0.00001]')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters). default: 5e-3')
parser.add_argument('--stop_thres', type=float, default=1e-9,
                    help='The stop threshold for the training error. If the difference between training losses '
                         'between epoches are less than the threshold, the training will be stopped. Default:1e-9')

# Model parameters


parser.add_argument('--use_cnn', type=boolean_string, default=True,
                    help='whether to use CNN for feature extraction. Default:False')
parser.add_argument('--use_lstm', type=boolean_string, default=True,
                    help='whether to use LSTM for feature extraction. Default:False')
parser.add_argument('--use_rp', type=boolean_string, default=True,
                    help='Whether to use random projection')
parser.add_argument('--rp_params', type=str, default='-1,3',
                    help='Parameters for random projection: number of random projection, '
                         'sub-dimension for each random projection')
parser.add_argument('--use_metric', action='store_true', default=False,
                    help='whether to use the metric learning for class representation. Default:False')
parser.add_argument('--metric_param', type=float, default=0.01,
                    help='Metric parameter for prototype distances between classes. Default:0.000001')
parser.add_argument('--use_ss', action='store_true', default=True,
                    help='Use semi-supervised learning.')
parser.add_argument('--sup_ratio', type=float, default=0.5,
                    help='Supervised ratio for labeled data in training set')
parser.add_argument('--filters', type=str, default="256,256,128",
                    help='filters used for convolutional network. Default:256,256,128')
parser.add_argument('--kernels', type=str, default="8,5,3",
                    help='kernels used for convolutional network. Default:8,5,3')
parser.add_argument('--dilation', type=int, default=1,
                    help='the dilation used for the first convolutional layer. '
                         'If set to -1, use the automatic number. Default:-1')
parser.add_argument('--layers', type=str, default="500,300",
                    help='layer settings of mapping function. [Default]: 500,300')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout rate (1 - keep probability). Default:0.5')
parser.add_argument('--lstm_dim', type=int, default=128,
                    help='Dimension of LSTM Embedding.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
args.sparse = True
args.layers = [int(l) for l in args.layers.split(",")]
args.kernels = [int(l) for l in args.kernels.split(",")]
args.filters = [int(l) for l in args.filters.split(",")]
args.rp_params = [float(l) for l in args.rp_params.split(",")]

if not args.use_lstm and not args.use_cnn:
    print("Must specify one encoding method: --use_lstm or --use_cnn")
    print("Program Exiting.")
    exit(-1)

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))


print("Loading dataset", args.dataset, "...")
# Model and optimizer
model_type = "TapNet" 

# training function
def train():
    loss_list = [sys.maxsize]
    test_best_possible, best_so_far = 0.0, sys.maxsize
    for epoch in range(args.epochs):
        print('Epoch:' , epoch)
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output, proto_dist, indices_train, indices_val, indices_test = model(input)

        #loss_train = F.cross_entropy(output[idx_train], torch.squeeze(labels[idx_train]))
        loss_train = F.cross_entropy(output[:128], torch.squeeze(labels[indices_train]))
        if args.use_metric:
            loss_train = loss_train - args.metric_param * proto_dist

        
        if abs(loss_train.item() - loss_list[-1]) < args.stop_thres \
                or loss_train.item() > loss_list[-1]:

                break
        else:
            loss_list.append(loss_train.item())
        

        #acc_train = accuracy(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[:128], labels[indices_train])
        f1_train = f1_score(output[:128], labels[indices_train])
        print('Train accuracy:', acc_train)
        print('Train f1 score', f1_train)
        loss_train.backward()
        optimizer.step()

        loss_val = F.cross_entropy(output[128:192], torch.squeeze(labels[indices_val]))
        acc_val = accuracy(output[128:192], labels[indices_val])
        f1 = f1_score(output[128:192], labels[indices_val])

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.8f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t),
              'f1_val_score: {:.4f}'.format(f1.item()))

        if acc_val.item() > test_best_possible:
            test_best_possible = acc_val.item()
        if best_so_far > loss_train.item():
            best_so_far = loss_train.item()
            test_acc = acc_val.item()
    print("test_acc: " + str(test_acc))
    print("best possible: " + str(test_best_possible))

# test function
def test(fold):
    output, proto_dist,indices_train, indices_val, indices_test = model(input)
    loss_test = F.cross_entropy(output[192:], torch.squeeze(labels[indices_test]))
    if args.use_metric:
        loss_test = loss_test - args.metric_param * proto_dist

    acc_test = accuracy(output[192:], labels[indices_test])
    f1 = f1_score(output[192:], labels[indices_test])
    
    print(labels[indices_test].detach().cpu().numpy())
    print(output[192:][:,1].detach().cpu().numpy())
    
    
    fpr, tpr, th = roc_curve(labels[indices_test].detach().cpu().numpy(), output[192:][:,1].detach().cpu().numpy(), pos_label=3)
    print(fpr.shape)
    print(tpr.shape)
    #print(th)
    np.save('TruePR_fold{}_sup_0'.format(fold) + args.fase + '.npy', tpr)
    np.save('FalsePR_fold{}_sup_0'.format(fold) + args.fase + '.npy', fpr)
    
    print(args.dataset, "Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()),
          "f1_score: {:.4f}".format(f1.item()))
    

if model_type == "TapNet":

    features_fold, labels_fold, idx_train_fold, idx_val_fold, idx_test_fold, nclass_fold = load_raw_ts(args.data_path, dataset=args.dataset, fase=args.fase)

    n_folds = 5

    for i in range(n_folds):
        features = features_fold[i]
        labels = labels_fold[i]
        idx_train = idx_train_fold[i]
        idx_val = idx_val_fold[i]
        idx_test = idx_test_fold[i]
        nclass = nclass_fold

        # update random permutation parameter
        if args.rp_params[0] < 0:
            dim = features.shape[1]
            args.rp_params = [3, math.floor(dim / (3 / 2))]
        else:
            dim = features.shape[1]
            args.rp_params[1] = math.floor(dim / args.rp_params[1])
        
        args.rp_params = [int(l) for l in args.rp_params]
        print("rp_params:", args.rp_params)

        # update dilation parameter
        if args.dilation == -1:
            args.dilation = math.floor(features.shape[2] / 64)

        print("Data shape:", features.size())
        model = TapNet(nfeat=features.shape[1],
                    len_ts=features.shape[2],
                    layers=args.layers,
                    nclass=nclass,
                    dropout=args.dropout,
                    use_lstm=args.use_lstm,
                    use_cnn=args.use_cnn,
                    filters=args.filters,
                    dilation=args.dilation,
                    kernels=args.kernels,
                    use_ss=args.use_ss,
                    sup_ratio=args.sup_ratio, 
                    use_metric=args.use_metric,
                    use_rp=args.use_rp,
                    rp_params=args.rp_params,
                    lstm_dim=args.lstm_dim
                    )
    
        # cuda
        if args.cuda:
            #model = nn.DataParallel(model) #Used when you have more than one GPU. Sometimes work but not stable
            model.cuda()
            features, labels, idx_train = features.cuda(), labels.cuda(), idx_train.cuda()
        input = (features, labels, idx_train, idx_val, idx_test)

        # init the optimizer
        optimizer = optim.Adam(model.parameters(),
                            lr=args.lr, weight_decay=args.wd)
        '''
        idx_sup_list = []
        for i in range(nclass):
            idx = (labels[idx_train].squeeze() == i).nonzero().squeeze(1)
            ## define the (un-)supervised portion in class i
            n_sup_i = int(idx.shape[0] * args.sup_ratio)
            idx_sup_i = idx[:n_sup_i]
            idx_unsup_i = idx[n_sup_i:]
            idx_sup_list.append(idx_sup_i)

        idx_sup = torch.cat(idx_sup_list, dim=0)
'''
        # Train model
        t_total = time.time()
        print('======== Sup ratio {} ========'.format(args.sup_ratio))
        print('======Fold {} ========'.format(i))
        train()
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        test(fold = i)
