import torch
import numpy as np
import pandas as pd
from torchvision.datasets import MNIST as mnist
from torchvision.datasets import CIFAR10 as cifar10

from BaseNetworks import BaseNetMNIST, BaseNetCIFAR, BaseNetMushroom, BaseNetMiniboone
import TS_Updates as ts

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def normalize(train_feat, test_feat=None):
    train_feat = (train_feat - np.mean(train_feat))/np.std(train_feat)
    test_feat = (test_feat - np.mean(test_feat))/np.std(test_feat)
    
    return train_feat, test_feat


def loadMNISTData():
    train = mnist('../data',download=True, train=True)
    test = mnist('../data',download=True, train=False)
    
    train_data, test_data = train.data.numpy(), test.data.numpy()
    train_labels, test_labels = train.targets.numpy(), test.targets.numpy()
    
    yr_train = train_labels.astype('int64')
    yr_test = test_labels.astype('int64')
    
    xr_train,xr_test = normalize(train_data, test_data)
    
    xr_train = xr_train.reshape((-1,1,28,28))
    xr_test = xr_test.reshape((-1,1,28,28))
    
    return xr_train,xr_test,yr_train,yr_test


def loadCIFARData():
    train = cifar10('../data',download=True, train=True)
    test = cifar10('../data',download=True, train=False)
    
    train_data, test_data = train.data, test.data
    train_labels, test_labels = np.array(train.targets), np.array(test.targets)
    
    yr_train = train_labels.astype('int64')
    yr_test = test_labels.astype('int64')
    
    xr_train,xr_test = normalize(train_data, test_data)
    
    xr_train = xr_train.reshape((-1,3,32,32))
    xr_test = xr_test.reshape((-1,3,32,32))
    
    return xr_train,xr_test,yr_train,yr_test


def loadMushroomData():
    
    all_data = pd.read_csv('mushrooms.csv').to_numpy()[1:]
    N = len(all_data)
    
    
    for col in range(all_data.shape[1]):
        unique_vals = list(np.unique(all_data[:,col]))
        num_classes = range(len(unique_vals))
        
        for i, val in enumerate(all_data[:,col]):
            all_data[i,col] = num_classes[unique_vals.index(val)]
    
    
    np.random.shuffle(all_data)
    
    
    
    all_features = all_data[:,1:]
    all_labels = all_data[:,:1]
    
    train_data = all_features[:int(N*0.9)]
    test_data = all_features[int(N*0.9):]
    
    train_labels = all_labels[:int(N*0.9)]
    test_labels = all_labels[int(N*0.9):]
    
    yr_train = train_labels.astype('int64').ravel()
    yr_test = test_labels.astype('int64').ravel()
    
    #xr_train,xr_test = normalize(train_data, test_data)
    xr_train,xr_test = train_data.astype('int64'), test_data.astype('int64')
    
    return xr_train,xr_test,yr_train,yr_test


def loadMinibooneData(process=False):
    
    if process:
        with open('MiniBooNE_PID.txt','r') as f:
            data = f.read().split('\n')
        line1= data[0].split()
        n1, n2 = int(line1[0]), int(line1[1])
        labels1, labels2 = np.repeat(0, n1), np.repeat(1, n2)
        labels = np.hstack((labels1, labels2)).reshape(-1,1)
        
        all_data = np.array([[float(s) for s in d.split()] for d in data[1:-1]])
        all_data = np.hstack((all_data,labels))
        
        df = pd.DataFrame(all_data)
        df.to_csv('miniboone.csv')

    
    all_data = pd.read_csv('miniboone.csv').to_numpy()[1:,1:]
    N = len(all_data)
    
    np.random.shuffle(all_data)
    
    all_features = all_data[:,:-1]
    all_labels = all_data[:,-1:]
    
    train_data = all_features[:int(N*0.8)]
    test_data = all_features[int(N*0.8):]
    
    train_labels = all_labels[:int(N*0.8)]
    test_labels = all_labels[int(N*0.8):]
    
    yr_train = train_labels.astype('int64').ravel()
    yr_test = test_labels.astype('int64').ravel()
    
    #xr_train,xr_test = normalize(train_data, test_data)
    xr_train,xr_test = train_data.astype('int64'), test_data.astype('int64')
    
    return xr_train,xr_test,yr_train,yr_test


'''
This main code is similar to that for RSO, with a few changes. I include full
instruction below.



In order to switch the data used, simply alter the name of the loadBLANKData function
as well as the BaseNetBLANK class to be in the form of BLANK = {MNIST, CIFAR,
                                                                Mushroom, Miniboone}

To alter the type of TS used, training parameters, and any other model/training
arguments, adjust the args dictionary as follows:
    - net, X, y, X_t, y_t: These are the network and data, respectively. Usually,
                           there is no need to alter these. 
    - cycles: Number of training cycles to use.
    - bs: Batch size of each cycle.
    - binary: Determines whether to use regular updates (False), or binary 
              updates (True).
    - scale_offset: Point to break up linear annealing schedule. If 0, no break, 
                    and if -1 it breaks in the middle to go down to 1/10, then 
                    back up. If -2, it goes down to 1/100th after going to 1/10.
    - f_length: Forgetting length (0 turns forgetting off).
    - sample_type: Determines whether to use Beta TS (0) or Gaussian TS (1).
    - save_name: The name of the file to save results to.

If you wish to use ternary/binary RSO, you must also change ternweight_flag to
1 (ternary) or 2 (binary) in the network object instantiation.
'''


X_train,X_test,y_train,y_test = loadMushroomData()

feature_dim = X_train.shape[-1]
out_dim = len(np.unique(y_train))


TSNet = BaseNetMushroom(True,ternweight_flag=0).to(device)


args = {'net':TSNet,
        'X':X_train,
        'y':y_train,
        'X_t':X_test,
        'y_t':y_test,
        'cycles':50,
        'bs':2500,
        'binary':False,
        'scale_offset':0,
        'f_length':0,
        'sample_type':0,
        'save_name':'ts_beta_mushroom.txt'}



loss_vals, acc_vals = ts.trainTS(**args)










