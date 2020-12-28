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
forgetting5_data = loadData(plotfile_dir+'ts_save_beta_forgetting5.txt')
forgetting3_data = loadData(plotfile_dir+'ts_save_beta_forgetting3.txt')

x = range(len(beta_data[:,0]))


colors=plt.cm.rainbow(np.linspace(0,1,4))

# Plot losses for the different forgetting methods.
fig, ax = plt.subplots()


ax.plot(x,beta_data[:,1],color=colors[0],label='Without Forgetting')
ax.plot(x,forgetting5_data[:,1],color='grey',label='Forgetting Length 5')
ax.plot(x,forgetting3_data[:,1],color=colors[2],label='Forgetting Length 3')

ax.set(xlabel='Cycle',ylabel='Loss Value (CE)')

plt.grid(True)
ax.legend()
plt.tight_layout()
plt.show()

fig.savefig('forgetting_loss',dpi=200)


# Plot accuracies for the different forgetting methods.
fig, ax = plt.subplots()

ax.plot(x,beta_data[:,0],color=colors[0],label='Without Forgetting')
ax.plot(x,forgetting5_data[:,0],color='grey',label='Forgetting Length 5')
ax.plot(x,forgetting3_data[:,0],color=colors[2],label='Forgetting Length 3')

ax.set(xlabel='Cycle',ylabel='Test Accuracy')

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

fig.savefig('forgetting_acc',dpi=200)



