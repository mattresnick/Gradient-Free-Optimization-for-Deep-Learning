import matplotlib.pyplot as plt
import numpy as np
import os

plotfile_dir = './plot_results/'

def loadData(filename):
    with open(filename,'r') as f:
        data = f.read().split('\n')[:-1]
    data = [[float(d.split(',')[0]),float(d.split(',')[1])] for d in data]
    data = np.array(data)
    return data


mse_normal_data = loadData(plotfile_dir+'rso_save_normal_mse.txt')
steps = len(mse_normal_data)
mse_normal_data = mse_normal_data[[i for i in range(steps) if i%12==0]]


ce_normal_data = loadData(plotfile_dir+'rso_save_normal_ce.txt')
uniform_data = loadData(plotfile_dir+'rso_save_uniform_ce.txt')
fixed_data  = loadData(plotfile_dir+'rso_save_fixed_ce.txt')



# Plot MSE vs CE losses.
fig, ax = plt.subplots()

x = range(len(ce_normal_data[:,0]))

ax.plot(x,mse_normal_data[:,1],color='blue',label='Loss: MSE')
ax.plot(x,ce_normal_data[:,1],color='red',label='Loss: CE')
ax.set(xlabel='Cycle',ylabel='Loss Value')

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

fig.savefig('mse_vs_ce_loss',dpi=200)


# Plot MSE vs CE test accuracies.
fig, ax = plt.subplots()

ax.plot(x,mse_normal_data[:,0],color='blue',label='Loss: MSE')
ax.plot(x,ce_normal_data[:,0],color='red',label='Loss: CE')
ax.set(xlabel='Cycle',ylabel='Test Accuracy')

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

fig.savefig('mse_vs_ce_accuracy',dpi=200)






standard_data  = loadData(plotfile_dir+'standard_save.txt')

# Plot different sampling method losses.
fig, ax = plt.subplots()

colors=plt.cm.rainbow(np.linspace(0,1,4))
ax.plot(x,standard_data[:,1],color=colors[3],label='Standard Updates (SGD)')
ax.plot(x,ce_normal_data[:,1],color=colors[0],label='RSO (Normal)')
ax.plot(x,uniform_data[:,1],color='grey',label='RSO (Uniform)')
ax.plot(x,fixed_data[:,1],color=colors[2],label='RSO (Fixed)')
ax.set(xlabel='Cycle',ylabel='Loss Value')

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

fig.savefig('ce_loss',dpi=200)



# Plot different sampling method test accuracies.
fig, ax = plt.subplots()

ax.plot(x,standard_data[:,0],color=colors[3],label='Standard Updates (SGD)')
ax.plot(x,ce_normal_data[:,0],color=colors[0],label='RSO (Normal)')
ax.plot(x,uniform_data[:,0],color='grey',label='RSO (Uniform)')
ax.plot(x,fixed_data[:,0],color=colors[2],label='RSO (Fixed)')
ax.set(xlabel='Cycle',ylabel='Test Accuracy')

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

fig.savefig('ce_accuracy',dpi=200)






