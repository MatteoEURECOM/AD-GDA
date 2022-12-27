import numpy as np
import matplotlib.pyplot as plt


def reorder(array,order):
    reorder=np.zeros(array.shape)
    for j in range(0,MCReps):
        for i in range(0,nodes):
            reorder[j,order[j,i],:]=array[j,i,:]
    return reorder
def movingaverage(interval, window_size):
    for i in range(0,window_size-1):
        interval=np.append(interval,interval[-1])
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'valid')

#PLOT COOS7 Experiments
plt.rc('font', family='serif', serif='Computer Modern Roman', size=14)
plt.rc('text', usetex=True)
plt.rcParams.update({'figure.autolayout': True})
plt.figure(figsize=(5,5))

#SIMULATION PARAMETERS
n_params=408727
iter=5000
MCReps=3
nodes=10
np.random.seed(1)
plac=np.tile(np.arange(0,nodes),(MCReps,1))
[np.random.shuffle(x) for x in plac]

#AD-GDA 4 bit quantization
bit_param=4  #bits for each NN param
bit_dual=32  #bits for the dual variable (UNCOMPRESSED)
max_deg=4    #Maximum Degree
bit_per_it=(n_params*bit_param+nodes*bit_dual)*max_deg/(10e6)  #bits sent by the busiest node in the network
data = np.load('ADGDA/CNN_2DTORUS_0.8_ROBUST_Quantization_0.01.npy', allow_pickle=True)
net_loss=np.stack([np.array(b) for b in data[1]])
net_loss=reorder(net_loss,plac)
mean_net_loss= np.mean(np.min(net_loss,axis=1),axis=0)
std_net_loss= np.std(np.min(net_loss,axis=1),axis=0)
y=movingaverage(mean_net_loss,15)
x=bit_per_it*np.arange(0,iter,5)
ci=movingaverage(std_net_loss/np.sqrt(MCReps),15)
plt.plot(bit_per_it*np.arange(0,iter,5),y,color='tab:green',linewidth=2,linestyle='-',label='AD-GDA 4 bit Quant.',zorder=4)
plt.fill_between(bit_per_it*np.arange(0,iter,5), y-ci, y+ci,color='tab:green',alpha=0.25,zorder=4)


#DR-DSGD
bit_param=32  #bits for each NN param
bit_dual=0  #bits for the dual variable (UNCOMPRESSED)
max_deg=4    #Maximum Degree
bit_per_it=(n_params*bit_param+nodes*bit_dual)*max_deg/(10e6)  #bits sent by the busiest node in the network
data = np.load('DR-DSGD/CNN_2DTORUS_DR-DSGD.npy', allow_pickle=True)
net_loss=np.stack([np.array(b) for b in data[1]])
net_loss=reorder(net_loss,plac)
mean_net_loss= np.mean(np.min(net_loss,axis=1),axis=0)
std_net_loss= np.std(np.min(net_loss,axis=1),axis=0)
y=movingaverage(mean_net_loss,20)
x=bit_per_it*np.arange(0,iter,5)
ci=movingaverage(std_net_loss/np.sqrt(MCReps),20)
plt.plot(bit_per_it*np.arange(0,iter,5),y,color='tab:blue',linestyle='-.',linewidth=2,label='DR-DSGD')
plt.fill_between(bit_per_it*np.arange(0,iter,5), y-ci, y+ci,color='tab:blue',alpha=0.25)

#CHOCO-SGD
bit_param=4  #bits for each NN param
bit_dual=0  #bits for the dual variable (UNCOMPRESSED)
max_deg=4    #Maximum Degree
bit_per_it=(n_params*bit_param+nodes*bit_dual)*max_deg/(10e6)  #bits sent by the busiest node in the network
data = np.load('ADGDA/CNN_2DTORUS_0.8_NOT_ROBUST_Quantization_1.0.npy', allow_pickle=True)
net_loss=np.stack([np.array(b) for b in data[1]])
mean_net_loss= np.mean(np.min(net_loss,axis=1),axis=0)
std_net_loss= np.std(np.min(net_loss,axis=1),axis=0)
y=movingaverage(mean_net_loss,15)
x=bit_per_it*np.arange(0,iter,5)
ci=movingaverage(std_net_loss/np.sqrt(MCReps),15)
plt.plot(bit_per_it*np.arange(0,iter,5),y,color='black',linestyle=':',linewidth=2,label='CHOCO-SGD 4 bit Quant.',zorder=5)
plt.fill_between(bit_per_it*np.arange(0,iter,5), y-ci, y+ci,color='black',alpha=0.25,zorder=5)


#DRFA
bit_param=32  #bits for each NN param
bit_dual=0  #bits for the dual variable (UNCOMPRESSED)
max_deg=5    #Maximum Degree
n_local_iteration=10
bit_per_it=(n_params*bit_param+nodes*bit_dual)*max_deg*2/(10e6*n_local_iteration)  #bits sent by the busiest node in the network, factor 2 to account for random sketching of DRFA
data = np.load('DRFA/DRFA_ROBUST.npy', allow_pickle=True)
net_loss=data
mean_net_loss= np.mean(np.min(net_loss,axis=1),axis=0)
std_net_loss= np.std(np.min(net_loss,axis=1),axis=0)
y=movingaverage(mean_net_loss,50)
x=bit_per_it*np.arange(0,iter,5)
ci=movingaverage(std_net_loss/np.sqrt(MCReps),50)
plt.plot(bit_per_it*np.arange(0,iter,10),y,color='tab:red',marker='o',linewidth=2,markersize=5,label='DRFA',markevery=20)
plt.fill_between(bit_per_it*np.arange(0,iter,10), y-ci, y+ci,color='tab:red',alpha=0.3)

plt.legend()
plt.xlim(0,8000)
plt.ylim(0.3,0.76)
plt.grid()
plt.ylabel('Accuracy')
plt.xlabel('MB')
plt.savefig("COOS7_Comm.pdf")
plt.show()
