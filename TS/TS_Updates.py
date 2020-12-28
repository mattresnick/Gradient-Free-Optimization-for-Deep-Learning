import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy, copy
from sklearn.metrics import roc_auc_score as AUC

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def toNumpy(arr):
    if torch.cuda.is_available():
        return arr.cpu().detach().numpy()
    else:
        return arr.numpy()


# Depricated now that I use BaseNetworks, but I leave it here for concise
# application if needed.
class BaseNetTS(nn.Module):
    def __init__(self,TS_flag, ternweight_flag=False):
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
        
        
        self.bn_layers = nn.ModuleList([nn.BatchNorm2d(16),
                                        nn.BatchNorm2d(16),
                                        nn.BatchNorm2d(16),
                                        nn.BatchNorm2d(16),
                                        nn.BatchNorm2d(16),
                                        nn.BatchNorm2d(16)])
        
        self.L = len(self.layers)
        
        self.pool = nn.MaxPool2d(2)
        
        if TS_flag:
            for layer in self.layers:
                layer.weight.requires_grad=False
                layer.bias.requires_grad=False
        
        if ternweight_flag:
            #self.ID = lambda x: torch.sign(x)
            self.ID = lambda x: nn.Identity()(x)
            
            for layer in self.layers:
                layer.weight = nn.Parameter(torch.sign(layer.weight))
                layer.bias = nn.Parameter(torch.sign(layer.bias))
                
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
                #x = torch.max(x.view(x.size(0), x.size(1), -1), dim=2)
        x = x.view(x.size(0), -1)
        x = nn.Softmax(dim=-1)(self.layers[-1](x))
        return x
    
    def forward(self, x):
        with torch.no_grad():
            return self.ff(x)






def getLoss(net, criterion, X, y):
        lower_output = net.forward(X)
        loss_low = criterion(lower_output, y.long())
        
        return loss_low




class TSUpdateGaussian():
    
    def __init__(self,criterion,schedule,delta_type):
        '''
        Parameters:
            - criterion (nn loss object): Loss function for given network.
            - schedule (array-like): Array of annealing scale factors per epoch.
            - delta_type (integer): Choice of sampling for weight updates. The
                                    choices are 0) normal, 1) uniform, 2) fixed
                                    at stddev of layer.
        '''
        
        self.crit = criterion
        self.schedule = schedule
        
        if delta_type==0:
            self.sample = lambda d: np.random.normal(0,d)
        elif delta_type==1:
            self.sample = lambda d: np.random.uniform(-2*d,2*d)
        else:
            self.sample = lambda d: d
    
    def forward(self, net, X, y, s, belief, bias_belief):
        '''
        Parameters:
            - net (nn.Module object): Given network for updating.
            - X (torch tensor): Minibatch of data.
            - y (torch tensor): Labels for minibatch.
            - s (integer): Current epoch number for annealing.
        '''
        with torch.no_grad():
            current_loss = toNumpy(getLoss(net, self.crit, X, y))
            
            for i,layer in enumerate(net.layers):
                original_shape = layer.weight.shape
                
                delta=toNumpy(layer.weight.std())/self.schedule[s]
                w_step = self.sample(delta)
                
                flat_params = toNumpy(layer.weight.view(-1,1))
                
                for j, w in enumerate(flat_params):
                    
                    # Initialize statistics if they've not yet been seen.
                    if belief[i][j][1]==0:
                        belief[i][j][0]=0
                        belief[i][j][1]=1
                        belief[i][j][2]=1
                        belief[i][j][3]=1
                        belief[i][j][4]=0 # Total loss diffs
                        belief[i][j][5]=0 # running SSE total
                    
                    mu=belief[i][j][0]
                    kappa=belief[i][j][1]
                    alpha=belief[i][j][2]
                    beta=belief[i][j][3]
                    xT = belief[i][j][4]
                    sse = belief[i][j][5]
                    
                    g_sample = np.random.gamma(alpha, 1/beta)
                    
                    if kappa==0 or g_sample==0:
                        dist = np.random.normal(mu,0)
                    else:
                        dist = np.random.normal(mu,(1/(kappa*g_sample)))
                    
                    sample = bool(np.round(dist))
                    
                    if sample:
                        # Create copies of the nework to test alternative parameters.
                        down_net = deepcopy(net)
                        up_net = deepcopy(net)
                        
                        
                        # First replace the given parameter with a lower value.
                        down_params = down_net.layers[i].weight.view(-1,1)
                        down_params[j] -= w_step
                        down_net.layers[i].weight = nn.Parameter(down_params.view(original_shape))
                        down_net.layers[i].weight.requires_grad=False
                        
                        
                        # Then replace the given parameter with a higher value.
                        up_params = up_net.layers[i].weight.view(-1,1)
                        up_params[j] += w_step
                        up_net.layers[i].weight = nn.Parameter(up_params.view(original_shape))
                        up_net.layers[i].weight.requires_grad=False
                        
                        
                        # Then get losses and check for reduction.
                        nets_to_check = [down_net, up_net]
                        losses = [toNumpy(getLoss(n, self.crit, X, y)) for n in nets_to_check]
                        
                        # If one of the new losses was better, keep the parameter update.
                        minlosses = np.min(losses)
                        R = 0
                        
                        if minlosses<current_loss:
                            R = current_loss-minlosses
                            current_loss = minlosses.copy()
                            net = nets_to_check[np.argmin(losses)]
                        
                        # Sampling parameter updates.
                        x_bar = (xT + R)/kappa
                        belief[i][j][4]+=R
                        
                        belief[i][j][5] = sse + (R-x_bar)**2
                        
                        belief[i][j][0] = (kappa*mu+x_bar)/(kappa+1)
                        belief[i][j][1] = kappa  +1
                        belief[i][j][2] = alpha + 1/2
                        belief[i][j][3] = beta + (1/2)*belief[i][j][5] + \
                                          (kappa/(kappa+1))*((x_bar-mu)**2/2)
                        
                        
                        del down_net
                        del up_net
                    
                    printstring = 'Layer #'+str(i+1)+': '+str(j)+\
                        '/'+ \
                    str(flat_params.shape[0]) +\
                        '. Sum weights: '+str(np.sum(toNumpy(net.layers[i].weight)))+\
                        '. Loss: '+str(np.round(current_loss,4))
                    print('\r{:100}'.format(printstring),end='')
                
                print()
                
                for k, b in enumerate(layer.bias):
                    
                    # Initialize statistics if they've not yet been seen.
                    if bias_belief[i][k][1]==0:
                        bias_belief[i][k][0]=0
                        bias_belief[i][k][1]=1
                        bias_belief[i][k][2]=1
                        bias_belief[i][k][3]=1
                        bias_belief[i][k][4]=0 # Total loss diffs
                        bias_belief[i][k][5]=0 # running SSE total
                    
                    mu=bias_belief[i][k][0]
                    kappa=bias_belief[i][k][1]
                    alpha=bias_belief[i][k][2]
                    beta=bias_belief[i][k][3]
                    xT = bias_belief[i][k][4]
                    sse = bias_belief[i][k][5]
                    
                    g_sample = np.random.gamma(alpha, 1/beta)
                    
                    if kappa==0 or g_sample==0:
                        dist = np.random.normal(mu,0)
                    else:
                        dist = np.random.normal(mu,(1/(kappa*g_sample)))
                    sample = bool(np.round(dist))
                    
                    if sample:
                        # Create copies of the nework to test alternative parameters.
                        down_net = deepcopy(net)
                        up_net = deepcopy(net)
                        
                        
                        # First replace the given parameter with a lower value.
                        down_params = down_net.layers[i].bias
                        down_params[k] -= w_step
                        down_net.layers[i].bias = nn.Parameter(down_params)
                        down_net.layers[i].bias.requires_grad=False
                        
                        
                        # Then replace the given parameter with a higher value.
                        up_params = up_net.layers[i].bias
                        up_params[k] += w_step
                        up_net.layers[i].bias = nn.Parameter(up_params)
                        up_net.layers[i].bias.requires_grad=False
                        
                        
                        # Then get losses and check for reduction.
                        nets_to_check = [down_net, up_net]
                        losses = [toNumpy(getLoss(n, self.crit, X, y)) for n in nets_to_check]
                        
                        # If one of the new losses was better, keep the parameter update.
                        minlosses = np.min(losses)
                        R = 0
                        
                        if minlosses<current_loss:
                            R = current_loss-minlosses
                            current_loss = minlosses.copy()
                            net = nets_to_check[np.argmin(losses)]
                        
                        # Sampling parameter updates.
                        x_bar = (xT + R)/kappa
                        bias_belief[i][k][4]+=R
                        
                        bias_belief[i][k][5] = sse + (R-x_bar)**2
                        
                        bias_belief[i][k][0] = (kappa*mu+x_bar)/(kappa+1)
                        bias_belief[i][k][1] = kappa + 1
                        bias_belief[i][k][2] = alpha + 1/2
                        bias_belief[i][k][3] = beta + (1/2)*bias_belief[i][k][5] + \
                                          (kappa/(kappa+1))*((x_bar-mu)**2/2)
                        
                        
                        del down_net
                        del up_net
                    
                    printstring = 'Layer #'+str(i+1)+': '+str(k)+\
                        '/'+ \
                    str(flat_params.shape[0]) +\
                        '. Sum biases: '+str(np.sum(toNumpy(net.layers[i].bias)))+\
                        '. Loss: '+str(np.round(current_loss,4))
                    print('\r{:100}'.format(printstring),end='')
        
        return net, current_loss, belief, bias_belief




class TSUpdateGaussianBinary():
    
    def __init__(self,criterion,schedule):
        '''
        Parameters:
            - criterion (nn loss object): Loss function for given network.
            - schedule (array-like): Array of annealing scale factors per epoch.
        '''
        
        self.crit = criterion
        self.schedule = schedule
    
    def forward(self, net, X, y, s, belief, bias_belief):
        '''
        Parameters:
            - net (nn.Module object): Given network for updating.
            - X (torch tensor): Minibatch of data.
            - y (torch tensor): Labels for minibatch.
            - s (integer): Current epoch number for annealing.
        '''
        with torch.no_grad():
            current_loss = toNumpy(getLoss(net, self.crit, X, y))
            
            for i,layer in enumerate(net.layers):
                original_shape = layer.weight.shape
                
                flat_params = toNumpy(layer.weight.view(-1,1))
                
                for j, w in enumerate(flat_params):
                    
                    # Initialize statistics if they've not yet been seen.
                    if belief[i][j][1]==0:
                        belief[i][j][0]=0
                        belief[i][j][1]=1
                        belief[i][j][2]=1
                        belief[i][j][3]=1
                        belief[i][j][4]=0 # Total loss diffs
                        belief[i][j][5]=0 # running SSE total
                    
                    mu=belief[i][j][0]
                    kappa=belief[i][j][1]
                    alpha=belief[i][j][2]
                    beta=belief[i][j][3]
                    xT = belief[i][j][4]
                    sse = belief[i][j][5]
                    
                    g_sample = np.random.gamma(alpha, beta)
                    
                    if kappa==0 or g_sample==0:
                        dist = np.random.normal(mu,0)
                    else:
                        dist = np.random.normal(mu,(1/(kappa*g_sample)))
                    sample = bool(np.round(dist))
                    
                    if sample:
                        # Create copies of the nework to test alternative parameters.
                        down_net = deepcopy(net)
                        
                        
                        # First replace the given parameter with a lower value.
                        down_params = down_net.layers[i].weight.view(-1,1)
                        down_params[j] *= -1
                        down_net.layers[i].weight = nn.Parameter(down_params.view(original_shape))
                        down_net.layers[i].weight.requires_grad=False
                        
                        
                        loss = toNumpy(getLoss(down_net, self.crit, X, y))
                        R = 0
                        
                        if loss<current_loss:
                            R = current_loss-loss
                            current_loss = loss.copy()
                            net = down_net
                            
                        # Sampling parameter updates.
                        x_bar = (xT + R)/kappa
                        belief[i][j][4]+=R
                        
                        belief[i][j][5] = sse + (R-x_bar)**2
                        
                        belief[i][j][0] = (kappa*mu+x_bar)/(kappa+1)
                        belief[i][j][1] = kappa  +1
                        belief[i][j][2] = alpha + 1/2
                        belief[i][j][3] = beta + (1/2)*belief[i][j][5] + \
                                          (kappa/(kappa+1))*((x_bar-mu)**2/2)
                        
                        
                        del down_net
                        
                    
                    printstring = 'Layer #'+str(i+1)+': '+str(j)+\
                        '/'+ \
                    str(flat_params.shape[0]) +\
                        '. Sum weights: '+str(np.sum(toNumpy(net.layers[i].weight)))+\
                        '. Loss: '+str(np.round(current_loss,4))
                    print('\r{:100}'.format(printstring),end='')
                
                print()
                
                for k, b in enumerate(layer.bias):
                    
                    # Initialize statistics if they've not yet been seen.
                    if bias_belief[i][k][1]==0:
                        bias_belief[i][k][0]=0
                        bias_belief[i][k][1]=1
                        bias_belief[i][k][2]=1
                        bias_belief[i][k][3]=1
                        bias_belief[i][k][4]=0 # Total loss diffs
                        bias_belief[i][k][5]=0 # running SSE total
                    
                    mu=bias_belief[i][k][0]
                    kappa=bias_belief[i][k][1]
                    alpha=bias_belief[i][k][2]
                    beta=bias_belief[i][k][3]
                    xT = bias_belief[i][k][4]
                    sse = bias_belief[i][k][5]
                    
                    g_sample = np.random.gamma(alpha, beta)
                    
                    if kappa==0 or g_sample==0:
                        dist = np.random.normal(mu,0)
                    else:
                        dist = np.random.normal(mu,(1/(kappa*g_sample)))
                    
                    sample = bool(np.round(dist))
                    
                    if sample:
                        if b==-1:
                            updates=[0,1]
                        elif b==0:
                            updates=[-1,1]
                        elif b==1:
                            updates=[-1,0]
                        
                        # Create copies of the nework to test alternative parameters.
                        down_net = deepcopy(net)
                        up_net = deepcopy(net)
                        
                        
                        # First replace the given parameter with a lower value.
                        down_params = down_net.layers[i].bias
                        down_params[k] = updates[0]
                        down_net.layers[i].bias = nn.Parameter(down_params)
                        down_net.layers[i].bias.requires_grad=False
                        
                        
                        # Then replace the given parameter with a higher value.
                        up_params = up_net.layers[i].bias
                        up_params[k] = updates[1]
                        up_net.layers[i].bias = nn.Parameter(up_params)
                        up_net.layers[i].bias.requires_grad=False
                        
                        
                        # Then get losses and check for reduction.
                        nets_to_check = [down_net, up_net]
                        losses = [toNumpy(getLoss(n, self.crit, X, y)) for n in nets_to_check]
                        
                        # If one of the new losses was better, keep the parameter update.
                        minlosses = np.min(losses)
                        R = 0
                        
                        if minlosses<current_loss:
                            R = current_loss-minlosses
                            current_loss = minlosses.copy()
                            net = nets_to_check[np.argmin(losses)]
                        
                        # Sampling parameter updates.
                        x_bar = (xT + R)/kappa
                        bias_belief[i][k][4]+=R
                        
                        bias_belief[i][k][5] = sse + (R-x_bar)**2
                        
                        bias_belief[i][k][0] = (kappa*mu+x_bar)/(kappa+1)
                        bias_belief[i][k][1] = kappa + 1
                        bias_belief[i][k][2] = alpha + 1/2
                        bias_belief[i][k][3] = beta + (1/2)*bias_belief[i][k][5] + \
                                          (kappa/(kappa+1))*((x_bar-mu)**2/2)
                        
                        
                        del down_net
                        del up_net
                    
                    printstring = 'Layer #'+str(i+1)+': '+str(k)+\
                        '/'+ \
                    str(flat_params.shape[0]) +\
                        '. Sum biases: '+str(np.sum(toNumpy(net.layers[i].bias)))+\
                        '. Loss: '+str(np.round(current_loss,4))
                    print('\r{:100}'.format(printstring),end='')
        
        return net, current_loss, belief, bias_belief















class TSUpdateBernoulli():
    
    def __init__(self,criterion,schedule,delta_type,f_length=0):
        '''
        Parameters:
            - criterion (nn loss object): Loss function for given network.
            - schedule (array-like): Array of annealing scale factors per epoch.
            - delta_type (integer): Choice of sampling for weight updates. The
                                    choices are 0) normal, 1) uniform, 2) fixed
                                    at stddev of layer.
        '''
        
        self.crit = criterion
        self.schedule = schedule
        self.f_length = f_length
        self.use_f = f_length>0
        
        if delta_type==0:
            self.sample = lambda d: np.random.normal(0,d)
        elif delta_type==1:
            self.sample = lambda d: np.random.uniform(-2*d,2*d)
        else:
            self.sample = lambda d: d
    
    def forward(self, net, X, y, s, belief, bias_belief):
        '''
        Parameters:
            - net (nn.Module object): Given network for updating.
            - X (torch tensor): Minibatch of data.
            - y (torch tensor): Labels for minibatch.
            - s (integer): Current epoch number for annealing.
        '''
        with torch.no_grad():
            current_loss = toNumpy(getLoss(net, self.crit, X, y))
            
            for i,layer in enumerate(net.layers):
                original_shape = layer.weight.shape
                
                delta=toNumpy(layer.weight.std())/self.schedule[s]
                w_step = self.sample(delta)
                
                flat_params = toNumpy(layer.weight.view(-1,1))
                
                for j, w in enumerate(flat_params):
                    
                    S = [1 for s in belief[i][j] if s=='a']
                    F = [1 for s in belief[i][j] if s=='b']
                    alpha=sum(S)+1
                    beta=sum(F)+1
                    
                    sample = bool(np.round(np.random.beta(alpha, beta)))
                    
                    if sample:
                        # Create copies of the nework to test alternative parameters.
                        down_net = deepcopy(net)
                        up_net = deepcopy(net)
                        
                        
                        # First replace the given parameter with a lower value.
                        down_params = down_net.layers[i].weight.view(-1,1)
                        down_params[j] -= w_step
                        down_net.layers[i].weight = nn.Parameter(down_params.view(original_shape))
                        down_net.layers[i].weight.requires_grad=False
                        
                        
                        # Then replace the given parameter with a higher value.
                        up_params = up_net.layers[i].weight.view(-1,1)
                        up_params[j] += w_step
                        up_net.layers[i].weight = nn.Parameter(up_params.view(original_shape))
                        up_net.layers[i].weight.requires_grad=False
                        
                        
                        # Then get losses and check for reduction.
                        nets_to_check = [down_net, up_net]
                        losses = [toNumpy(getLoss(n, self.crit, X, y)) for n in nets_to_check]
                        
                        # If one of the new losses was better, keep the parameter update.
                        minlosses = np.min(losses)
                        
                        if minlosses<current_loss:
                            current_loss = minlosses.copy()
                            net = nets_to_check[np.argmin(losses)]
                            belief[i][j] = np.hstack((belief[i][j], np.array(['a'])))
                            
                        else:
                            belief[i][j] = np.hstack((belief[i][j], np.array(['b'])))
                        
                        # Forgetting.
                        if len(belief[i][j])>self.f_length and self.use_f:
                                belief[i][j] = belief[i][j][1:]
                        
                        del down_net
                        del up_net
                    
                    printstring = 'Layer #'+str(i+1)+': '+str(j)+\
                        '/'+ \
                    str(flat_params.shape[0]) +\
                        '. Sum weights: '+str(np.sum(toNumpy(net.layers[i].weight)))+\
                        '. Loss: '+str(np.round(current_loss,4))
                    print('\r{:100}'.format(printstring),end='')
                    
                
                
                print ()
                
                for k, b in enumerate(layer.bias):
                    
                    S = [1 for s in bias_belief[i][k] if s=='a']
                    F = [1 for s in bias_belief[i][k] if s=='b']
                    alpha=sum(S)+1
                    beta=sum(F)+1
                    
                    sample = bool(np.round(np.random.beta(alpha, beta)))
                    
                    if sample:
                        # Create copies of the nework to test alternative parameters.
                        down_net = deepcopy(net)
                        up_net = deepcopy(net)
                        
                        
                        # First replace the given parameter with a lower value.
                        down_params = down_net.layers[i].bias
                        down_params[k] -= w_step
                        down_net.layers[i].bias = nn.Parameter(down_params)
                        down_net.layers[i].bias.requires_grad=False
                        
                        
                        # Then replace the given parameter with a higher value.
                        up_params = up_net.layers[i].bias
                        up_params[k] += w_step
                        up_net.layers[i].bias = nn.Parameter(up_params)
                        up_net.layers[i].bias.requires_grad=False
                        
                        
                        # Then get losses and check for reduction.
                        nets_to_check = [down_net, up_net]
                        losses = [toNumpy(getLoss(n, self.crit, X, y)) for n in nets_to_check]
                        
                        # If one of the new losses was better, keep the parameter update.
                        minlosses = np.min(losses)
                        
                        if minlosses<current_loss:
                            current_loss = minlosses.copy()
                            net = nets_to_check[np.argmin(losses)]
                            bias_belief[i][k] = np.hstack((bias_belief[i][k], np.array(['a'])))
                            
                        else:
                            bias_belief[i][k] = np.hstack((bias_belief[i][k], np.array(['b'])))
                        
                        # Forgetting.
                        if len(bias_belief[i][k])>self.f_length and self.use_f:
                                bias_belief[i][k] = bias_belief[i][k][1:]
                        
                        del down_net
                        del up_net
                    
                    printstring = 'Layer #'+str(i+1)+': '+str(k)+\
                        '/'+ \
                    str(flat_params.shape[0]) +\
                        '. Sum weights: '+str(np.sum(toNumpy(net.layers[i].weight)))+\
                        '. Loss: '+str(np.round(current_loss,4))
                    print('\r{:100}'.format(printstring),end='')
        
        return net, current_loss, belief, bias_belief





class TSUpdateBernoulliBinary():
    
    def __init__(self,criterion,schedule,f_length=0):
        '''
        Parameters:
            - criterion (nn loss object): Loss function for given network.
            - schedule (array-like): Array of annealing scale factors per epoch.
        '''
        
        self.crit = criterion
        self.schedule = schedule
        self.f_length = f_length
        self.use_f = f_length>0
        
    
    def forward(self, net, X, y, s, belief, bias_belief):
        '''
        Parameters:
            - net (nn.Module object): Given network for updating.
            - X (torch tensor): Minibatch of data.
            - y (torch tensor): Labels for minibatch.
            - s (integer): Current epoch number for annealing.
        '''
        with torch.no_grad():
            current_loss = toNumpy(getLoss(net, self.crit, X, y))
            
            for i,layer in enumerate(net.layers):
                original_shape = layer.weight.shape
                
                flat_params = toNumpy(layer.weight.view(-1,1))
                
                for j, w in enumerate(flat_params):
                    
                    S = [1 for s in belief[i][j] if s=='a']
                    F = [1 for s in belief[i][j] if s=='b']
                    alpha=sum(S)+1
                    beta=sum(F)+1
                    
                    sample = bool(np.round(np.random.beta(alpha, beta)))
                    
                    if sample:
                        # Create copies of the nework to test alternative parameters.
                        down_net = deepcopy(net)
                        
                        
                        
                        # First replace the given parameter with a lower value.
                        down_params = down_net.layers[i].weight.view(-1,1)
                        down_params[j] *= -1
                        down_net.layers[i].weight = nn.Parameter(down_params.view(original_shape))
                        down_net.layers[i].weight.requires_grad=False
                    
                    
                    
                        # Then get losses and check for reduction.
                        loss = toNumpy(getLoss(down_net, self.crit, X, y))
                        
                        
                        if loss<current_loss:
                            current_loss = loss.copy()
                            net = down_net
                            belief[i][j] = np.hstack((belief[i][j], np.array(['a'])))
                            
                        else:
                            belief[i][j] = np.hstack((belief[i][j], np.array(['b'])))
                        
                        # Forgetting.
                        if len(belief[i][j])>self.f_length and self.use_f:
                                belief[i][j] = belief[i][j][1:]
                        
                        del down_net
                    
                    printstring = 'Layer #'+str(i+1)+': '+str(j)+\
                        '/'+ \
                    str(flat_params.shape[0]) +\
                        '. Sum weights: '+str(np.sum(toNumpy(net.layers[i].weight)))+\
                        '. Loss: '+str(np.round(current_loss,4))
                    print('\r{:100}'.format(printstring),end='')
                    
                
                
                print ()
                
                for k, b in enumerate(layer.bias):
                    
                    S = [1 for s in bias_belief[i][k] if s=='a']
                    F = [1 for s in bias_belief[i][k] if s=='b']
                    alpha=sum(S)+1
                    beta=sum(F)+1
                    
                    sample = bool(np.round(np.random.beta(alpha, beta)))
                    
                    if sample:
                        
                        if b==-1:
                            updates=[0,1]
                        elif b==0:
                            updates=[-1,1]
                        elif b==1:
                            updates=[-1,0]
                        
                        # Create copies of the nework to test alternative parameters.
                        down_net = deepcopy(net)
                        up_net = deepcopy(net)
                        
                        
                       # First replace the given parameter with a lower value.
                        down_params = down_net.layers[i].bias
                        down_params[k] = updates[0]
                        down_net.layers[i].bias = nn.Parameter(down_params)
                        down_net.layers[i].bias.requires_grad=False
                        
                        
                        # Then replace the given parameter with a higher value.
                        up_params = up_net.layers[i].bias
                        up_params[k] = updates[1]
                        up_net.layers[i].bias = nn.Parameter(up_params)
                        up_net.layers[i].bias.requires_grad=False
                        
                        
                        # Then get losses and check for reduction.
                        nets_to_check = [down_net, up_net]
                        losses = [toNumpy(getLoss(n, self.crit, X, y)) for n in nets_to_check]
                        
                        # If one of the new losses was better, keep the parameter update.
                        minlosses = np.min(losses)
                        
                        if minlosses<current_loss:
                            current_loss = minlosses.copy()
                            net = nets_to_check[np.argmin(losses)]
                            bias_belief[i][k] = np.hstack((bias_belief[i][k], np.array(['a'])))
                            
                        else:
                            bias_belief[i][k] = np.hstack((bias_belief[i][k], np.array(['b'])))
                        
                        # Forgetting.
                        if len(bias_belief[i][k])>self.f_length and self.use_f:
                                bias_belief[i][k] = bias_belief[i][k][1:]
                        
                        del down_net
                        del up_net
                    
                    printstring = 'Layer #'+str(i+1)+': '+str(k)+\
                        '/'+ \
                    str(flat_params.shape[0]) +\
                        '. Sum weights: '+str(np.sum(toNumpy(net.layers[i].weight)))+\
                        '. Loss: '+str(np.round(current_loss,4))
                    print('\r{:100}'.format(printstring),end='')
        
        return net, current_loss, belief, bias_belief








# Create empty belief structures.
def makeBeliefStructures(net, sample_type):
    
    # Beta sampling.
    if sample_type==0:
        belief, bias_belief = [], []
        for l in net.layers:
            w_container, b_container=[],[]
            weight_num = l.weight.flatten().shape[0]
            bias_num = l.bias.shape[0]
            
            for w in range(weight_num):
                w_container.append([])
                
            for b in range(bias_num):
                b_container.append([])
            
            belief.append(w_container)
            bias_belief.append(b_container)
        
        return np.array(belief), np.array(bias_belief)
    
    # Gaussian sampling.
    else:
        belief, bias_belief = [], []
        for l in net.layers:
            w_container, b_container=[],[]
            weight_num = l.weight.flatten().shape[0]
            bias_num = l.bias.shape[0]
            
            for w in range(weight_num):
                w_container.append(np.zeros(6))
                
            for b in range(bias_num):
                b_container.append(np.zeros(6))
            
            belief.append(w_container)
            bias_belief.append(b_container)
            
        return np.array(belief), np.array(bias_belief)



# Training function for network via Thompson Sampling.
def trainTS(net, X, y, X_t, y_t, cycles=5, bs=1000, binary=False, delta_type=0,
            scale_offset=0, f_length=0, sample_type=0, save_name=''):
    '''
    Training function using Thompson Sampling updates.
    
    Parameters:
        - net (nn.Module object): Network model to be trained.
        - X (ndarray): Training data.
        - y (ndarray): Training data labels.
        - X_t (ndarray): Testing/validation data.
        - y_t (ndarray): Testing/validation data labels.
        - cycles (integer): Number of training cycles to perform.
        - bs (integer): Number of samples per batch, per cycle.
        - binary (boolean): Flag variable to indicate ternary/binary model 
                            parameters.
        - delta_type (integer): Flag variable to indicate which type of update 
                                value to use. Generally this will just be 0,
                                normally sampled updates.
        - scale_offset (integer): Point to break up linear annealing schedule.
                                  If 0, no break, and if -1 it breaks in the
                                  middle to go down to 1/10, then back up. If
                                  -2, it goes down to 1/100th after going to 1/10.
        - f_length (integer): Forgetting length. If 0, no forgetting. If 
                              positive, sets the memory length for recording
                              observations. If negative, uses wipe forgetting,
                              and resets the belief after that many positive
                              cycles.
        - sample_type (integer): Flag variable that indicates type of Thompson
                                 Sampling to use. 0 is Beta, 1 is Gaussian.
        - save_name (string): Name of file to save loss and accuracy results to.
    
    Returns:
        - loss_vals (list): Recorded loss values during training.
        - acc_vals (list): Recorded test accuracy values during training.
    '''
    
    
    # Anonymous function to count correct predictions.
    count_correct = lambda y_t,y_h: np.array([1 if obs==pred else 0 for obs,pred in zip(y_t, y_h)])
    
    # Make an annealing schedule for weight updates.
    if scale_offset==0:
        # Linear Scale
        schedule_scale = np.linspace(1,.1,cycles)
    elif scale_offset==-1:
        # Up and down scale. 
        scale1 = np.linspace(1,.1,cycles//2)
        scale2 = np.linspace(.1,1,cycles//2)
        schedule_scale = np.hstack((scale1,scale2[1:]))
    elif scale_offset==-2:
        # Down and down further scale. 
        scale1 = np.linspace(1,.1,cycles)
        scale2 = np.linspace(.1,.01,10)
        schedule_scale = np.hstack((scale1,scale2[1:]))
        cycles+=10
    else:
        # Piecewise linear scale. 
        scale1 = np.linspace(1,.5,cycles-scale_offset)
        scale2 = np.linspace(.5,.1,scale_offset+1)
        schedule_scale = np.hstack((scale1,scale2[1:]))
    
    
    criterion = nn.CrossEntropyLoss()
    input_length = X.shape[0]
    
    # Beta sampling.
    if sample_type==0:
        if binary:
            updater = TSUpdateBernoulliBinary(criterion,schedule_scale,f_length)
        else:
            updater = TSUpdateBernoulli(criterion,schedule_scale,delta_type,f_length)
    
    # Gaussian sampling.
    else:
        if binary:
            updater = TSUpdateGaussianBinary(criterion,schedule_scale)
        else:
            updater = TSUpdateGaussian(criterion,schedule_scale,delta_type)
    
    belief, bias_belief = makeBeliefStructures(net, sample_type)
    
    loss_vals, auc_vals = [], []
    for i in range(cycles):
        selections = np.random.choice(range(0,input_length), bs)
        
        
        # Cut batches out of training data.
        X_train = torch.FloatTensor(X[selections].copy()).to(device)
        y_train = torch.FloatTensor(y[selections].copy()).to(device)
        
        # Ready testing data too.
        X_test = torch.FloatTensor(X_t.copy()).to(device)
        
        # Get results before training begins.
        if i==0:
            loss = np.round(toNumpy(getLoss(net, criterion, X_train, y_train)),4)
            y_hat = toNumpy(torch.argmax(net(X_test),dim=-1))
            #acc = round(np.mean(count_correct(y_test,y_hat)),3)
            auc = AUC(y_t,y_hat)
            with open(save_name,'a') as f:
                f.write(str(auc)+','+str(loss)+'\n')
        
        # Reset belief structures if using wipe forgetting.
        if f_length<0 and i==np.abs(f_length):
            belief, bias_belief = makeBeliefStructures(net, sample_type)
        
        net, loss, belief, bias_belief = updater.forward(net, X_train, y_train, i, belief, bias_belief)
        y_hat = toNumpy(torch.argmax(net(X_test),dim=-1))
        
        #acc = round(np.mean(count_correct(y_test,y_hat)),3)
        auc = AUC(y_t,y_hat)
        print ('\nCurrent AUC:'+str(auc)+'. Cycle #'+str(i+1)+'/'+str(cycles))
        
        with open(save_name,'a') as f:
            f.write(str(auc)+','+str(loss)+'\n')
        
        loss_vals.append(loss)
        auc_vals.append(auc)
        
        del X_train
        del y_train
        del X_test
            
    return loss_vals, auc_vals
    
    

