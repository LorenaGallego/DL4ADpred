import torch
import torch.nn as nn
import torch.nn.functional as F
from HMM_att_utils import euclidean_dist, normalize, output_conv_size, dump_embedding
import numpy as np
import random

class ATT_model(nn.Module):

    def __init__(self, nfeat, len_ts, nclass, filters, kernels, dilation, layers, rp_params,
                 use_att=True, sup_ratio=0):
        super(ATT_model, self).__init__()
        self.nclass = nclass

        # parameters for random projection
        self.rp_group, self.rp_dim = rp_params

        #random dimension permutation
        paddings = [0, 0, 0]

        self.conv_1_models = nn.ModuleList()
        self.idx = []
        for i in range(self.rp_group):
            self.conv_1_models.append(nn.Conv1d(self.rp_dim, filters[0], kernel_size=kernels[0], dilation=dilation, stride=1, padding=paddings[0]))
            self.idx.append(np.random.permutation(nfeat)[0: self.rp_dim])

        
        self.conv_bn_1 = nn.BatchNorm1d(filters[0])

        self.conv_2 = nn.Conv1d(filters[0], filters[1], kernel_size=kernels[1], stride=1, padding=paddings[1])

        self.conv_bn_2 = nn.BatchNorm1d(filters[1])

        self.conv_3 = nn.Conv1d(filters[1], filters[2], kernel_size=kernels[2], stride=1, padding=paddings[2])

        self.conv_bn_3 = nn.BatchNorm1d(filters[2])

        # compute the size of input for fully connected layers
        fc_input = 0
        conv_size = len_ts

        for i in range(len(filters)):
            conv_size = output_conv_size(conv_size, kernels[i], 1, paddings[i])
        fc_input += conv_size
        #* filters[-1]
    
        #because of the random projection 
        fc_input = self.rp_group * filters[2] 

        # Representation mapping function
        layers = [fc_input] + layers
        print("Layers", layers)
        self.mapping = nn.Sequential()
        for i in range(len(layers) - 2):
            self.mapping.add_module("fc_" + str(i), nn.Linear(layers[i], layers[i + 1]))
            self.mapping.add_module("bn_" + str(i), nn.BatchNorm1d(layers[i + 1]))
            self.mapping.add_module("relu_" + str(i), nn.LeakyReLU())

        # add last layer
        self.mapping.add_module("fc_" + str(len(layers) - 2), nn.Linear(layers[-2], layers[-1]))
        if len(layers) == 2:  # if only one layer, add batch normalization
            self.mapping.add_module("bn_" + str(len(layers) - 2), nn.BatchNorm1d(layers[-1]))

        # Attention
        att_dim, semi_att_dim = 128, 128
        self.use_att = use_att
        if self.use_att:
            self.att_models = nn.ModuleList()
            for _ in range(nclass):

                att_model = nn.Sequential(
                    nn.Linear(layers[-1], att_dim),
                    nn.Tanh(),
                    nn.Linear(att_dim, 1)
                )
                self.att_models.append(att_model)

        
        self.sup_ratio = sup_ratio # 0.0 for unsupervised learning
        self.semi_att = nn.Sequential(
            nn.Linear(layers[-1], semi_att_dim),
            nn.Tanh(),
            nn.Linear(semi_att_dim, self.nclass)
        )

    def forward(self, input):
        x, labels, idx_train, idx_val, idx_test = input  # x is N * L, where L is the time-series feature dimension
        rand = random.sample(range(0, idx_train.size(0)), 128)
        rand_val = random.sample(range(idx_train.size(0), x.size(0)), 64)
        rand_test = random.sample(range(idx_train.size(0), x.size(0)), 64)
        print(len(rand), len(rand_val), len(rand_test), 'shapes arrays')
        #res_lt = [rand_val[i] + idx_train.size(0) for i in range(len(rand_val))]
        #res_lt2 = [rand_test[i] + idx_train.size(0) for i in range(len(rand_test))]
        x = torch.cat((x[rand,:,:],x[rand_val,:,:],x[rand_test,:,:] ),dim=0)
        idx_train = idx_train[rand]

        ## define the labeled and unlabeled portions in dataset 
       # n_sup = int(sup_ratio * self.n_train)

        # Covolutional Network with random projection
        # input ts: # N * C * L
        for i in range(len(self.conv_1_models)):
            #x_conv = x
            x_conv = self.conv_1_models[i](x[:, self.idx[i], :])

            x_conv = self.conv_bn_1(x_conv)
            x_conv = F.leaky_relu(x_conv)

            x_conv = self.conv_2(x_conv)
            x_conv = self.conv_bn_2(x_conv)
            x_conv = F.leaky_relu(x_conv)

            x_conv = self.conv_3(x_conv)
            x_conv = self.conv_bn_3(x_conv)
            x_conv = F.leaky_relu(x_conv)

            x_conv = torch.mean(x_conv, 2)

            if i == 0:
                x_conv_sum = x_conv
            else:
                x_conv_sum = torch.cat([x_conv_sum, x_conv], dim=1)

        x_conv = x_conv_sum

        x = x_conv
        
        # linear mapping to low-dimensional space
        x = self.mapping(x)
        
        # generate the class protocal with dimension C * D (nclass * dim)
        proto_list = []
        idx_unsup_list = []
        for i in range(self.nclass):
            idx = (labels[idx_train].squeeze() == i).nonzero().squeeze(1)
            n_sup_i = int(idx.shape[0] * self.sup_ratio)
            idx_sup_i = idx[:n_sup_i]
            idx_unsup_i = idx[n_sup_i:]
            idx_unsup_list.append(idx_unsup_i)
            print(len(idx_unsup_i), '``````````````')

            if self.use_att:
                #A = self.att_models[i](x[idx_train][idx])  # N_k * 1
                A = self.att_models[i](x[idx_sup_i])
                A = torch.transpose(A, 1, 0)  # 1 * N_k
                A = F.softmax(A, dim=1)  # softmax over N_k

                #class_repr = torch.mm(A, x[idx_train][idx]) # 1 * L
                class_repr = torch.mm(A, x[idx_sup_i])
                class_repr = torch.transpose(class_repr, 1, 0)  # L * 1
            else:  # if do not use attention, simply use the mean of training samples with the same labels.
                #class_repr = x[idx_train][idx].mean(0)  # L * 1
                class_repr = x[idx_sup_i].mean(0)
                #print(class_repr.size)
            proto_list.append(class_repr.view(1, -1))
            #print(proto_list)
        x_proto = torch.cat(proto_list, dim=0)
        
        # prototype distance
        proto_dists = euclidean_dist(x_proto, x_proto)
        num_proto_pairs = int(self.nclass * (self.nclass - 1) / 2)
        proto_dist = torch.sum(proto_dists) / num_proto_pairs


        ## apply the unsupervised portion to adjust the class prototype
        idx_unsup = torch.cat(idx_unsup_list, dim=0)
        semi_A = self.semi_att(x[idx_unsup])  # N_test * c
        print(semi_A.shape)
        semi_A = torch.transpose(semi_A, 1, 0)  # c * N_test
        semi_A = F.softmax(semi_A, dim=1)  # softmax over N_test
        x_proto_test = torch.mm(semi_A, x[idx_unsup])  # c * L
        x_proto = (x_proto + x_proto_test) / 2

        dists = euclidean_dist(x, x_proto)
        print(dists.shape, 'dist')
        
        return -dists, proto_dist, rand, rand_val, rand_test

