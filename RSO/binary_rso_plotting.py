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



mse_tern_data = loadData(plotfile_dir+'rso_save_tern_mse.txt')
x1 = range(len(mse_tern_data[:,0]))

tern_w0_data  = loadData(plotfile_dir+'rso_save_tern_w0.txt')
x2 = range(len(tern_w0_data[:,0]))

tern_n0_data = loadData(plotfile_dir+'rso_save_tern_no0.txt')
x3 = range(len(tern_n0_data[:,0]))

colors=plt.cm.rainbow(np.linspace(0,1,4))

# Plot losses for binary and ternary RSO.
fig, ax = plt.subplots()

ax2 = ax.twinx() 
ax2.set_ylabel('Loss Value (MSE)')

line1=ax2.plot(x1,mse_tern_data[:,1],color='grey',label='Ternary RSO (MSE)')
line2=ax.plot(x2,tern_w0_data[:,1],color=colors[2],label='Ternary RSO (CE)')
line3=ax.plot(x3,tern_n0_data[:,1],color=colors[3],label='Binary RSO (CE)')

ax.set(xlabel='Cycle',ylabel='Loss Value (CE)')

labels = [l.get_label() for l in line1+line2+line3]
ax.legend(line1+line2+line3, labels)

plt.grid(True)
plt.tight_layout()
plt.show()

#fig.savefig('tern_loss_plot',dpi=200)


# Plot test accuracies for binary and ternary RSO.
fig, ax = plt.subplots()

ax.plot(x1,mse_tern_data[:,0],color='grey',label='Ternary RSO (MSE)')
ax.plot(x2,tern_w0_data[:,0],color=colors[2],label='Ternary RSO (CE)')
ax.plot(x3,tern_n0_data[:,0],color=colors[3],label='Binary RSO (CE)')

ax.set(xlabel='Cycle',ylabel='Test Accuracy')

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#fig.savefig('tern_acc_plot',dpi=200)






bn_data = loadData(plotfile_dir+'rso_save_binary_bn.txt')
x4 = range(len(mse_tern_data[:,0]))

nobinact_data  = loadData(plotfile_dir+'rso_save_binary_nonbinaryact.txt')
x5 = range(len(tern_w0_data[:,0]))

colors=plt.cm.rainbow(np.linspace(0,1,4))

# Plot losses for binary RSO versions.
fig, ax = plt.subplots()


ax.plot(x3,tern_n0_data[:,1],color='grey',label='Base Binary RSO')
ax.plot(x4,nobinact_data[:,1],color=colors[2],label='Real Activations')
ax.plot(x5,bn_data[:,1],color=colors[3],label='Real Activations + BN')

ax.set(xlabel='Cycle',ylabel='Loss Value (CE)')

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

fig.savefig('bin_loss_plot',dpi=200)


# Plot test accuracies for binary RSO versions.
fig, ax = plt.subplots()

ax.plot(x3,tern_n0_data[:,0],color='grey',label='Base Binary RSO')
ax.plot(x4,nobinact_data[:,0],color=colors[2],label='Real Activations')
ax.plot(x5,bn_data[:,0],color=colors[3],label='Real Activations + BN')

ax.set(xlabel='Cycle',ylabel='Test Accuracy')

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

fig.savefig('bin_acc_plot',dpi=200)















