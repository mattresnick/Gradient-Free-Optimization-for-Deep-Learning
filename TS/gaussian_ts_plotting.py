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



beta_data = loadData(plotfile_dir+'ts_save_beta.txt')[:6]
gaussian_data  = loadData(plotfile_dir+'ts_save_gaussian.txt')[:6]
rso_data = loadData(plotfile_dir+'rso_save_normal_ce.txt')[:6]

x = range(len(beta_data[:,0]))
colors=plt.cm.rainbow(np.linspace(0,1,4))

# Plot losses for the different Thompson Sampling methods and RSO.
fig, ax = plt.subplots()


ax.plot(x,beta_data[:,1],color='grey',label='Beta Sampling')
ax.plot(x,gaussian_data[:,1],color=colors[2],label='Gaussian Sampling')
ax.plot(x,rso_data[:,1],color=colors[3],label='Standard RSO')

ax.set(xlabel='Cycle',ylabel='Loss Value (CE)')

plt.grid(True)
ax.legend()
plt.tight_layout()
plt.show()

#fig.savefig('gaussian_ts_loss',dpi=200)


# Plot accuracies for the different Thompson Sampling methods and RSO.
fig, ax = plt.subplots()

ax.plot(x,beta_data[:,0],color='grey',label='Beta Sampling')
ax.plot(x,gaussian_data[:,0],color=colors[2],label='Gaussian Sampling')
ax.plot(x,rso_data[:,0],color=colors[3],label='Standard RSO')

ax.set(xlabel='Cycle',ylabel='Test Accuracy')

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#fig.savefig('gaussian_ts_acc',dpi=200)




gaussian_data  = loadData(plotfile_dir+'ts_save_gaussian.txt')
x1 = range(len(gaussian_data[:,0]))

binary_data  = loadData(plotfile_dir+'ts_gaussian_binary.txt')
x2 = range(len(binary_data[:,0]))


# Plot losses for the different Thompson Sampling methods and RSO.
fig, ax = plt.subplots()

ax.plot(x1,gaussian_data[:,1],color='blue',label='Gaussian TS')
ax.plot(x2,binary_data[:,1],color='red',label='Binary Gaussian TS')

ax.set(xlabel='Cycle',ylabel='Loss Value (CE)')

plt.grid(True)
ax.legend()
plt.tight_layout()
plt.show()

fig.savefig('binary_gaussian_ts_loss_b',dpi=200)


# Plot accuracies for the different Thompson Sampling methods and RSO.
fig, ax = plt.subplots()

ax.plot(x1,gaussian_data[:,0],color='blue',label='Gaussian TS')
ax.plot(x2,binary_data[:,0],color='red',label='Binary Gaussian TS')

ax.set(xlabel='Cycle',ylabel='Test Accuracy')

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

fig.savefig('binary_gaussian_ts_acc_b',dpi=200)
