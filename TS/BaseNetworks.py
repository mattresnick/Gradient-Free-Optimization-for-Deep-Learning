import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class BaseNetMNIST(nn.Module):
    def __init__(self,RSO_flag, ternweight_flag=0):
        super().__init__()
        
        
        self.layers = nn.ModuleList([nn.Conv2d(1,16,3,1),
                                     nn.Conv2d(16,16,3,1),
                                     nn.Conv2d(16,16,3,1),
                                     nn.Conv2d(16,16,3,1),
                                     nn.Conv2d(16,16,3,1),
                                     nn.Conv2d(16,16,3,1),
                                     nn.Linear(16,10)])
        
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        
        
        self.bn_layers = nn.ModuleList([nn.BatchNorm2d(16,affine=False),
                                        nn.BatchNorm2d(16,affine=False),
                                        nn.BatchNorm2d(16,affine=False),
                                        nn.BatchNorm2d(16,affine=False),
                                        nn.BatchNorm2d(16,affine=False),
                                        nn.BatchNorm2d(16,affine=False)])
        
        self.L = len(self.layers)
        
        self.pool = nn.MaxPool2d(2)
        
        if RSO_flag:
            for layer in self.layers:
                layer.weight.requires_grad=False
                layer.bias.requires_grad=False
        
        if ternweight_flag!=0:
            #self.ID = lambda x: torch.sign(x)
            self.ID = lambda x: nn.Identity()(x)
            
            for layer in self.layers:
                
                if ternweight_flag==2:
                    layer.weight = nn.Parameter(torch.sign(layer.weight))
                    layer.bias = nn.Parameter(torch.sign(layer.bias))
                
                else:
                
                    init_weights = np.random.choice([-1,0,1], 
                                                size=layer.weight.shape, 
                                                replace=True)
                    layer.weight = nn.Parameter(torch.Tensor(init_weights).float())
                    
                    init_biases = np.random.choice([-1,0,1], 
                                                size=layer.bias.shape, 
                                                replace=True)
                    layer.bias = nn.Parameter(torch.Tensor(init_biases).float())
                
        else:
            self.ID = lambda x: nn.Identity()(x)
    
    def ff(self, x):
        
        for l, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.bn_layers[l](x)
            x = nn.ReLU()(x)
            x = self.ID(x)
            if l==1:
                x = self.pool(x)
            elif l==5:
                x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(x.size(0), -1)
        x = nn.Softmax(dim=-1)(self.layers[-1](x))
        return x
    
    def forward(self, x):
        with torch.no_grad():
            return self.ff(x)










class BaseNetCIFAR(nn.Module):
    def __init__(self,RSO_flag, ternweight_flag=0):
        super().__init__()
        
        
        self.layers = nn.ModuleList([nn.Conv2d(3,16,3,1),
                                     nn.Conv2d(16,16,3,1),
                                     nn.Conv2d(16,32,3,1),
                                     nn.Conv2d(32,32,3,1),
                                     nn.Conv2d(32,32,3,1),
                                     nn.Conv2d(32,32,3,1),
                                     nn.Linear(32,10)])
        
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        
        
        self.bn_layers = nn.ModuleList([nn.BatchNorm2d(16,affine=False),
                                        nn.BatchNorm2d(16,affine=False),
                                        nn.BatchNorm2d(32,affine=False),
                                        nn.BatchNorm2d(32,affine=False),
                                        nn.BatchNorm2d(32,affine=False),
                                        nn.BatchNorm2d(32,affine=False)])
        
        self.L = len(self.layers)
        
        self.pool = nn.MaxPool2d(2)
        
        if RSO_flag:
            for layer in self.layers:
                layer.weight.requires_grad=False
                layer.bias.requires_grad=False
        
        if ternweight_flag!=0:
            #self.ID = lambda x: torch.sign(x)
            self.ID = lambda x: nn.Identity()(x)
            
            for layer in self.layers:
                
                if ternweight_flag==2:
                    layer.weight = nn.Parameter(torch.sign(layer.weight))
                    layer.bias = nn.Parameter(torch.sign(layer.bias))
                
                else:
                
                    init_weights = np.random.choice([-1,0,1], 
                                                size=layer.weight.shape, 
                                                replace=True)
                    layer.weight = nn.Parameter(torch.Tensor(init_weights).float())
                    
                    init_biases = np.random.choice([-1,0,1], 
                                                size=layer.bias.shape, 
                                                replace=True)
                    layer.bias = nn.Parameter(torch.Tensor(init_biases).float())
                
        else:
            self.ID = lambda x: nn.Identity()(x)
    
    def ff(self, x):
        
        for l, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.bn_layers[l](x)
            x = nn.ReLU()(x)
            x = self.ID(x)
            if l==1:
                x = self.pool(x)
            elif l==5:
                x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(x.size(0), -1)
        x = nn.Softmax(dim=-1)(self.layers[-1](x))
        return x
    
    def forward(self, x):
        with torch.no_grad():
            return self.ff(x)





class BaseNetMushroom(nn.Module):
    def __init__(self,RSO_flag, ternweight_flag=0):
        super().__init__()
        
        
        self.layers = nn.ModuleList([nn.Linear(22,100),
                                     nn.Linear(100,50),
                                     nn.Linear(50,50),
                                     nn.Linear(50,2)])
        
        
        
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(100,affine=False),
                                        nn.BatchNorm1d(50,affine=False),
                                        nn.BatchNorm1d(50,affine=False)])
        
        
        if RSO_flag:
            for layer in self.layers:
                layer.weight.requires_grad=False
                layer.bias.requires_grad=False
        
        if ternweight_flag!=0:
            #self.ID = lambda x: torch.sign(x)
            self.ID = lambda x: nn.Identity()(x)
            
            for layer in self.layers:
                
                if ternweight_flag==2:
                    layer.weight = nn.Parameter(torch.sign(layer.weight))
                    layer.bias = nn.Parameter(torch.sign(layer.bias))
                
                else:
                
                    init_weights = np.random.choice([-1,0,1], 
                                                size=layer.weight.shape, 
                                                replace=True)
                    layer.weight = nn.Parameter(torch.Tensor(init_weights).float())
                    
                    init_biases = np.random.choice([-1,0,1], 
                                                size=layer.bias.shape, 
                                                replace=True)
                    layer.bias = nn.Parameter(torch.Tensor(init_biases).float())
                
        else:
            self.ID = lambda x: nn.Identity()(x)
    
    def ff(self, x):
        
        for l, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.bn_layers[l](x)
            x = nn.ReLU()(x)
            x = self.ID(x)
        #x = x.view(x.size(0), -1)
        x = nn.Softmax(dim=-1)(self.layers[-1](x))
        return x
    
    def forward(self, x):
        with torch.no_grad():
            return self.ff(x)




class BaseNetMiniboone(nn.Module):
    def __init__(self,RSO_flag, ternweight_flag=0):
        super().__init__()
        
        
        self.layers = nn.ModuleList([nn.Linear(50,100),
                                     nn.Linear(100,50),
                                     nn.Linear(50,50),
                                     nn.Linear(50,2)])
        
        
        
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(100,affine=False),
                                        nn.BatchNorm1d(50,affine=False),
                                        nn.BatchNorm1d(50,affine=False)])
        
        
        if RSO_flag:
            for layer in self.layers:
                layer.weight.requires_grad=False
                layer.bias.requires_grad=False
        
        if ternweight_flag!=0:
            #self.ID = lambda x: torch.sign(x)
            self.ID = lambda x: nn.Identity()(x)
            
            for layer in self.layers:
                
                if ternweight_flag==2:
                    layer.weight = nn.Parameter(torch.sign(layer.weight))
                    layer.bias = nn.Parameter(torch.sign(layer.bias))
                
                else:
                
                    init_weights = np.random.choice([-1,0,1], 
                                                size=layer.weight.shape, 
                                                replace=True)
                    layer.weight = nn.Parameter(torch.Tensor(init_weights).float())
                    
                    init_biases = np.random.choice([-1,0,1], 
                                                size=layer.bias.shape, 
                                                replace=True)
                    layer.bias = nn.Parameter(torch.Tensor(init_biases).float())
                
        else:
            self.ID = lambda x: nn.Identity()(x)
    
    def ff(self, x):
        
        for l, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.bn_layers[l](x)
            x = nn.ReLU()(x)
            x = self.ID(x)
        #x = x.view(x.size(0), -1)
        x = nn.Softmax(dim=-1)(self.layers[-1](x))
        return x
    
    def forward(self, x):
        with torch.no_grad():
            return self.ff(x)
