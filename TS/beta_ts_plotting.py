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



beta_data = loadData(plotfile_dir+'ts_save_beta.txt')
x1 = range(len(beta_data[:,0]))

offset_data  = loadData(plotfile_dir+'ts_save_beta_scaleoffset.txt')
x2 = range(len(offset_data[:,0]))

upanddown_data = loadData(plotfile_dir+'ts_save_beta_upanddownscale.txt')
x3 = range(len(upanddown_data[:,0]))

colors=plt.cm.rainbow(np.linspace(0,1,4))

# Plot losses for binary and ternary RSO.
fig, ax = plt.subplots()


ax.plot(x1,beta_data[:,1],color=colors[0],label='Linear Schedule')
ax.plot(x2,offset_data[:,1],color='black',label='Piecewise Linear Schedule')
ax.plot(x3,upanddown_data[:,1],color=colors[2],label='Up and Down Schedule')

ax.set(xlabel='Cycle',ylabel='Loss Value (CE)')

plt.grid(True)
ax.legend()
plt.tight_layout()
plt.show()

#fig.savefig('beta_ts_loss',dpi=200)


# Plot test accuracies for binary and ternary RSO.
fig, ax = plt.subplots()

ax.plot(x1,beta_data[:,0],color=colors[0],label='Linear Schedule')
ax.plot(x2,offset_data[:,0],color='black',label='Piecewise Linear Schedule')
ax.plot(x3,upanddown_data[:,0],color=colors[2],label='Up and Down Schedule')

ax.set(xlabel='Cycle',ylabel='Test Accuracy')

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#fig.savefig('beta_ts_acc',dpi=200)










binary_data  = loadData(plotfile_dir+'ts_save_beta_binary.txt')
x4 = range(len(binary_data[:,0]))

binary_forgetting_data = loadData(plotfile_dir+'ts_save_beta_binary_forgetting3.txt')
x5 = range(len(binary_forgetting_data[:,0]))

colors=plt.cm.rainbow(np.linspace(0,1,4))

# Plot losses for binary and ternary RSO.
fig, ax = plt.subplots()


ax.plot(x1,beta_data[:,1],color=colors[0],label='Beta TS')
ax.plot(x4,binary_data[:,1],color='black',label='Binary Beta TS')
ax.plot(x5,binary_forgetting_data[:,1],color=colors[2],label='Binary Beta TS (w/ Forgetting Length 3)')

ax.set(xlabel='Cycle',ylabel='Loss Value (CE)')

plt.grid(True)
ax.legend()
plt.tight_layout()
plt.show()

fig.savefig('binary_beta_ts_loss',dpi=200)


# Plot test accuracies for binary and ternary RSO.
fig, ax = plt.subplots()

ax.plot(x1,beta_data[:,0],color=colors[0],label='Beta TS')
ax.plot(x4,binary_data[:,0],color='black',label='Binary Beta TS')
ax.plot(x5,binary_forgetting_data[:,0],color=colors[2],label='Binary Beta TS (w/ Forgetting Length 3)')

ax.set(xlabel='Cycle',ylabel='Test Accuracy')

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

fig.savefig('binary_beta_ts_acc',dpi=200)



