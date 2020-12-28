import torch
from torch import nn
import numpy as np
from copy import deepcopy, copy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from BaseNetworks import BaseNetMNIST, BaseNetCIFAR, BaseNetMushroom, BaseNetMiniboone


def toNumpy(arr):
    if torch.cuda.is_available():
        return arr.cpu().detach().numpy()
    else:
        return arr.numpy()

def getLoss(net, criterion, X, y):
        lower_output = net.forward(X)
        loss_low = criterion(lower_output, y.long())
        return loss_low



class BaseNetStandardMNIST(BaseNetMNIST):
    def __init__(self,RSO_flag,triweight_flag=False):
        super().__init__(RSO_flag,triweight_flag)
    
    def forward(self, x):
        return self.ff(x)

class BaseNetStandardCIFAR(BaseNetCIFAR):
    def __init__(self,RSO_flag,triweight_flag=False):
        super().__init__(RSO_flag,triweight_flag)
    
    def forward(self, x):
        return self.ff(x)


class BaseNetStandardMushroom(BaseNetMushroom):
    def __init__(self,RSO_flag,triweight_flag=0):
        super().__init__(RSO_flag,triweight_flag)
    
    def forward(self, x):
        return self.ff(x)


class BaseNetStandardMiniboone(BaseNetMiniboone):
    def __init__(self,RSO_flag,triweight_flag=0):
        super().__init__(RSO_flag,triweight_flag)
    
    def forward(self, x):
        return self.ff(x)


def trainStandard(net, X, y, X_t, y_t, epochs=10, bs=64):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=0.01)
    
    input_length = X.shape[0]
    
    loss_vals, acc_vals = [], []
    for i in range(epochs):
        num_batches = int(input_length/bs)
        selections = np.random.choice(range(0,input_length,bs), num_batches)
        net.train()
        for j,batch_sel in enumerate(selections):
            optimizer.zero_grad()
            
            # The input size may not be divisible by the batch size
            end_ind=batch_sel+bs
            if end_ind>input_length:
                end_ind=input_length
            
            # Cut batches out of data
            X_train = torch.FloatTensor(X[batch_sel:end_ind].copy()).to(device)
            y_train = torch.LongTensor(y[batch_sel:end_ind].copy()).to(device)
            
            output = net(X_train)
            #output = torch.argmax(output, dim=-1)
            
            loss = criterion(output, y_train)
            loss_num = toNumpy(loss)
            loss.backward()
            optimizer.step()
            
        X_test = torch.FloatTensor(X_t.copy()).to(device)
        y_test = torch.LongTensor(y_t.copy()).to(device)
        
        net.eval()
        y_hat = toNumpy(torch.argmax(net(X_test),dim=-1))
        
        total_correct = np.array([1 if ground == obs else 0 for ground, obs in zip(y_test, y_hat)])
        acc = round(np.mean(total_correct),3)
        print ('\nCurrent Accuracy:',acc)
        print ('\nCurrent Loss:',loss_num)
        
        loss_vals.append(loss_num)
        acc_vals.append(acc)
            
    return loss_vals, acc_vals
