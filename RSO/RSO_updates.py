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
    


# Outdated for the current implementation, but I leave it here for concise 
# application if needed.
class BaseNetRSO(nn.Module):
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
        
        
        self.bn_layers = nn.ModuleList([nn.BatchNorm2d(16),
                                        nn.BatchNorm2d(16),
                                        nn.BatchNorm2d(16),
                                        nn.BatchNorm2d(16),
                                        nn.BatchNorm2d(16),
                                        nn.BatchNorm2d(16)])
        
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




def getLoss(net, criterion, X, y):
        lower_output = net.forward(X)
        #lower_output = torch.argmax(lower_output, dim=-1).float()
        loss_low = criterion(lower_output, y.long())
        
        return loss_low



# RSO without steps. Just two binary values.
class RSOUpdateBinary():
    def __init__(self,criterion,schedule):
        self.crit = criterion
        self.schedule = schedule
    
    def forward(self, net, X, y, s):
        with torch.no_grad():
            current_loss = toNumpy(getLoss(net, self.crit, X, y))
            
            for i,layer in enumerate(net.layers):
                original_shape = layer.weight.shape
                
                
                flat_params = toNumpy(layer.weight.view(-1,1))
                
                for j, w in enumerate(flat_params):
                    
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
                    
                    del down_net
                    
                    printstring = 'Layer #'+str(i+1)+': '+str(j)+\
                        '/'+ \
                    str(flat_params.shape[0]) +\
                        '. Sum weights: '+str(np.sum(toNumpy(net.layers[i].weight)))+\
                        '. Loss: '+str(np.round(current_loss,4))
                    print('\r{:100}'.format(printstring),end='')
                
                print ()
                
                for k, b in enumerate(layer.bias):
                    
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
                    
                    del down_net
                    del up_net
                    
                    printstring = 'Layer #'+str(i+1)+': '+str(k)+\
                        '/'+ \
                    str(net.layers[i].bias.shape[0]) +\
                        '. Sum biases: '+str(np.sum(toNumpy(net.layers[i].bias)))+\
                        '. Loss: '+str(np.round(current_loss,4))
                    print('\r{:100}'.format(printstring),end='')
                    
            return net, np.round(current_loss,4)





# RSO without steps. Just three ternary values.
class RSOUpdateTern():
    def __init__(self,criterion,schedule):
        self.crit = criterion
        self.schedule = schedule
    
    def forward(self, net, X, y, s):
        with torch.no_grad():
            current_loss = toNumpy(getLoss(net, self.crit, X, y))
            
            for i,layer in enumerate(net.layers):
                original_shape = layer.weight.shape
                
                
                flat_params = toNumpy(layer.weight.view(-1,1))
                
                for j, w in enumerate(flat_params):
                    
                    if w==-1:
                        updates=[0,1]
                    elif w==0:
                        updates=[-1,1]
                    elif w==1:
                        updates=[-1,0]
                    
                    
                    # Create copies of the nework to test alternative parameters.
                    down_net = deepcopy(net)
                    up_net = deepcopy(net)
                    
                    
                    # First replace the given parameter with a lower value.
                    down_params = down_net.layers[i].weight.view(-1,1)
                    down_params[j] = updates[0]
                    down_net.layers[i].weight = nn.Parameter(down_params.view(original_shape))
                    down_net.layers[i].weight.requires_grad=False
                    
                    
                    # Then replace the given parameter with a higher value.
                    up_params = up_net.layers[i].weight.view(-1,1)
                    up_params[j] = updates[1]
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
                    
                    del down_net
                    del up_net
                    
                    printstring = 'Layer #'+str(i+1)+': '+str(k)+\
                        '/'+ \
                    str(net.layers[i].bias.shape[0]) +\
                        '. Sum biases: '+str(np.sum(toNumpy(net.layers[i].bias)))+\
                        '. Loss: '+str(np.round(current_loss,4))
                    print('\r{:100}'.format(printstring),end='')
                    
            return net, np.round(current_loss,4)







# RSO update by iterating through each parameter of the layers independent of shape.
class RSOUpdateShapeFree():
    
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
    
    def forward(self, net, X, y, s):
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
                    
                    del down_net
                    del up_net
                    
                    printstring = 'Layer #'+str(i+1)+': '+str(k)+\
                        '/'+ \
                    str(net.layers[i].bias.shape[0]) +\
                        '. Sum biases: '+str(np.sum(toNumpy(net.layers[i].bias)))+\
                        '. Loss: '+str(np.round(current_loss,4))
                    print('\r{:100}'.format(printstring),end='')
            
            '''
            for i,layer in enumerate(net.bn_layers):
                original_shape = layer.weight.shape
                
                
                flat_params = toNumpy(layer.weight.view(-1,1))
                
                for j, w in enumerate(flat_params):
                    
                    
                    # Create copies of the nework to test alternative parameters.
                    down_net = deepcopy(net)
                    up_net = deepcopy(net)
                    
                    
                    # First replace the given parameter with a lower value.
                    down_params = down_net.bn_layers[i].weight.view(-1,1)
                    down_params[j] -= w_step
                    down_net.bn_layers[i].weight = nn.Parameter(down_params.view(original_shape))
                    down_net.bn_layers[i].weight.requires_grad=False
                    
                    
                    # Then replace the given parameter with a higher value.
                    up_params = up_net.bn_layers[i].weight.view(-1,1)
                    up_params[j] += w_step
                    up_net.bn_layers[i].weight = nn.Parameter(up_params.view(original_shape))
                    up_net.bn_layers[i].weight.requires_grad=False
                    
                    
                    # Then get losses and check for reduction.
                    nets_to_check = [down_net, up_net]
                    losses = [toNumpy(getLoss(n, self.crit, X, y)) for n in nets_to_check]
                    
                    
                    # If one of the new losses was better, keep the parameter update.
                    minlosses = np.min(losses)
                    if minlosses<current_loss:
                        current_loss = minlosses.copy()
                        net = nets_to_check[np.argmin(losses)]
                    
                    del down_net
                    del up_net
                    
                    printstring = 'Layer #'+str(i+1)+': '+str(j)+\
                        '/'+ \
                    str(flat_params.shape[0]) +\
                        '. Sum weights: '+str(np.sum(toNumpy(net.bn_layers[i].weight)))+\
                        '. Loss: '+str(np.round(current_loss,4))
                    print('\r{:100}'.format(printstring),end='')
                
                print ()
                
                for k, b in enumerate(layer.bias):
                    
                    # Create copies of the nework to test alternative parameters.
                    down_net = deepcopy(net)
                    up_net = deepcopy(net)
                    
                    
                    # First replace the given parameter with a lower value.
                    down_params = down_net.bn_layers[i].bias
                    down_params[k] -= w_step
                    down_net.bn_layers[i].bias = nn.Parameter(down_params)
                    down_net.bn_layers[i].bias.requires_grad=False
                    
                    
                    # Then replace the given parameter with a higher value.
                    up_params = up_net.bn_layers[i].bias
                    up_params[k] += w_step
                    up_net.bn_layers[i].bias = nn.Parameter(up_params)
                    up_net.bn_layers[i].bias.requires_grad=False
                    
                    
                    # Then get losses and check for reduction.
                    nets_to_check = [down_net, up_net]
                    losses = [toNumpy(getLoss(n, self.crit, X, y)) for n in nets_to_check]
                    
                    
                    # If one of the new losses was better, keep the parameter update.
                    minlosses = np.min(losses)
                    if minlosses<current_loss:
                        current_loss = minlosses.copy()
                        net = nets_to_check[np.argmin(losses)]
                    
                    del down_net
                    del up_net
                    
                    printstring = 'Layer #'+str(i+1)+': '+str(k)+\
                        '/'+ \
                    str(net.bn_layers[i].bias.shape[0]) +\
                        '. Sum biases: '+str(np.sum(toNumpy(net.bn_layers[i].bias)))+\
                        '. Loss: '+str(np.round(current_loss,4))
                    print('\r{:100}'.format(printstring),end='')
            
            '''
            return net, np.round(current_loss,4)






# Training function for network via Random Search Optimization.
def trainRSO(net, X, y, X_t, y_t, cycles=50, bs=5000, tern=0, delta_type=0,
             alt_scaling=50,save_name=''):
    
    # Anonymous function to count correct predictions.
    count_correct = lambda y_t,y_h: np.array([1 if obs==pred else 0 for obs,pred in zip(y_t, y_h)])
    
    # Make an annealing schedule for weight updates.
    schedule_scale = np.linspace(1,.1,alt_scaling)
    criterion = nn.CrossEntropyLoss()
    
    # Determine use of ternary version or not.
    if tern==0:
        updater = RSOUpdateShapeFree(criterion,schedule_scale,delta_type)
    elif tern==1:
        updater = RSOUpdateTern(criterion,schedule_scale)
    else:
        updater = RSOUpdateBinary(criterion,schedule_scale)
    
    input_length = X.shape[0]
    
    loss_vals, auc_vals = [], []
    for i in range(cycles):
        selections = np.random.choice(range(0,input_length), bs)
        
        
        # Cut batches out of training data.
        X_train = torch.FloatTensor(X[selections].copy()).to(device)
        y_train = torch.FloatTensor(y[selections].copy()).to(device)
        
        # Ready testing data too.
        X_test = torch.FloatTensor(X_t.copy()).to(device)
        
        net.eval()
        # Get results before training begins.
        if i==0:
            loss = np.round(toNumpy(getLoss(net, criterion, X_train, y_train)),4)
            y_hat = toNumpy(torch.argmax(net(X_test),dim=-1))
            #acc = round(np.mean(count_correct(y_test,y_hat)),3)
            auc = AUC(y_t,y_hat)
            with open(save_name,'a') as f:
                f.write(str(auc)+','+str(loss)+'\n')
        
        net.train()
        net, loss = updater.forward(net, X_train, y_train, i)
        net.eval()
        
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



