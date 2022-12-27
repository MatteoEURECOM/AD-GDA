import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy as np
import importlib
import torch
import os
import tensorflow as tf
from  sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist,fashion_mnist,cifar10
import scipy
def connectivity_matrix(nodes,type):
    '''
    Creates connectivity matrix
    :param nodes: number of nodes
    :param type: topology type
    :return: uniformly weighted connectivity matrix
    '''
    G=np.eye(nodes)
    if (type == 'RING'):
        G = np.zeros((nodes, nodes))
        for i in range(0, nodes):
            G[i, (i - 1) % nodes] = 1. / 3.
            G[i, i % nodes] = 1. / 3.
            G[i, (i + 1) % nodes] = 1. / 3.
    if (type == 'MESH'):
        G = np.ones((nodes, nodes)) / nodes
    if (type == '2DTORUS'):
        G = np.zeros((nodes, nodes))
        for i in range(0, nodes):
            G[i, (i - 2) % nodes] = 1
            G[i, (i - 1) % nodes] = 1.
            G[i, (i + 1) % nodes] = 1.
            G[i, (i + 2) % nodes] = 1.
    return G
def show_graph_with_labels(adjacency_matrix):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    plt.figure(figsize=(5, 5))
    values=['steelblue','seagreen','seagreen','seagreen','seagreen','steelblue','seagreen','seagreen','seagreen','seagreen']
    nx.draw(gr, node_size=800, with_labels=False,linewidths=2,width=2,node_color=values,edgecolors='black')
    plt.savefig('COOS7_network.png',dpi=600)
    plt.show()

show_graph_with_labels(connectivity_matrix(10,'2DTORUS'))


MCReps=3
nodes=10
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

color=['steelblue','gray','seagreen']
CI=False
plt.rc('font', family='serif', serif='Computer Modern Roman', size=13)
plt.rc('text', usetex=True)
plt.rcParams.update({'figure.autolayout': True})
plt.figure(figsize=(6,4.5))
np.random.seed(1)
plac=np.tile(np.arange(0,nodes),(MCReps,1))
[np.random.shuffle(x) for x in plac]
iter=5000
MCReps=3
bit_per_it=1
data = np.load('ADGDA/CNN_2DTORUS_0.8_ROBUST_Quantization_0.01.npy', allow_pickle=True)
loss=reorder(np.stack([np.array(b) for b in data[1]]),plac)
mic_1=np.mean(loss[:,8:10,-1],axis=1)
print('Mic 1 mean: '+str(np.mean(mic_1))+'  Mic 2 std: '+str(np.std(mic_1)))
mic_2=np.mean(loss[:,0:8,-1],axis=1)
print('Mic 1 mean: '+str(np.mean(mic_2))+'  Mic 2 std: '+str(np.std(mic_2)))
avg_mic=np.mean(np.vstack([mic_1,mic_2]),axis=0)
print('Avg. mean: '+str(np.mean(avg_mic))+'  Avg. std: '+str(np.std(avg_mic)))
net_loss=np.mean(loss,axis=0)
x=bit_per_it*np.arange(0,iter,5)
mic_1= movingaverage(np.mean(net_loss[8:10,:],axis=0),15)
ci_mic_1=movingaverage(np.std(net_loss[8:10,:],axis=0)/np.sqrt(MCReps),15)
mic_2= movingaverage(np.mean(net_loss[0:8,:],axis=0),15)
ci_mic_2=movingaverage(np.std(net_loss[0:8,:],axis=0)/np.sqrt(MCReps),15)
mic_avg= movingaverage(np.mean([mic_1,mic_2],axis=0),15)
avg=np.mean([np.mean(loss[:,8:10,:],axis=1),np.mean(loss[:,0:8,:],axis=1)],axis=0)
ci_mic_avg=movingaverage(np.std(avg,axis=0)/np.sqrt(MCReps),15)




data = np.load('ADGDA/CNN_2DTORUS_0.8_NOT_ROBUST_Quantization_1.0.npy', allow_pickle=True)
loss=reorder(np.stack([np.array(b) for b in data[1]]),plac)
mic_1_c=np.mean(loss[:,8:10,-1],axis=1)
print('Mic 1 mean: '+str(np.mean(mic_1_c))+'  Mic 2 std: '+str(np.std(mic_1_c)))
mic_2_c=np.mean(loss[:,0:8,-1],axis=1)
print('Mic 1 mean: '+str(np.mean(mic_2_c))+'  Mic 2 std: '+str(np.std(mic_2_c)))
avg_mic_c=np.mean(np.vstack([mic_1_c,mic_2_c]),axis=0)
print('Avg. mean: '+str(np.mean(avg_mic_c))+'  Avg. std: '+str(np.std(avg_mic_c)))
net_loss=np.mean(loss,axis=0)
x=bit_per_it*np.arange(0,iter,5)
c_mic_1= movingaverage(np.mean(net_loss[8:10,:],axis=0),15)
c_ci_mic_1=movingaverage(np.std(net_loss[8:10,:],axis=0)/np.sqrt(MCReps),15)
c_mic_2= movingaverage(np.mean(net_loss[0:8,:],axis=0),15)
c_ci_mic_2=movingaverage(np.std(net_loss[0:8,:],axis=0)/np.sqrt(MCReps),15)
c_mic_avg= movingaverage(np.mean([c_mic_1,c_mic_2],axis=0),15)
avg=np.mean([np.mean(loss[:,8:10,:],axis=1),np.mean(loss[:,0:8,:],axis=1)],axis=0)
c_ci_mic_avg=movingaverage(np.std(avg,axis=0)/np.sqrt(MCReps),15)

plt.plot(bit_per_it*np.arange(0,iter,5),mic_1,color=color[0],linewidth=2,linestyle='-',label=r'AD-GDA Microscope 1')
plt.plot(bit_per_it*np.arange(0,iter,5),c_mic_1,color=color[0],linewidth=3,linestyle='--',label=r' Microscope 1')
plt.plot(bit_per_it*np.arange(0,iter,5),mic_avg,color=color[1],linewidth=2,linestyle='-',label=r'AD-GDA Average')
plt.plot(bit_per_it*np.arange(0,iter,5),c_mic_avg,color=color[1],linewidth=3,linestyle='--',label=r' Average')
plt.plot(bit_per_it*np.arange(0,iter,5),mic_2,color=color[2],linewidth=2,linestyle='-',label=r'AD-GDA 4 Microscope 2')
plt.plot(bit_per_it*np.arange(0,iter,5),c_mic_2,color=color[2],linewidth=3,linestyle='--',label=r' Microscope 2')

if CI:
    plt.fill_between(bit_per_it*np.arange(0,iter,5), mic_1-ci_mic_1, mic_1+ci_mic_1,color=color[0],alpha=0.3)
    plt.fill_between(bit_per_it * np.arange(0, iter, 5), c_mic_1 - c_ci_mic_1, c_mic_1 + c_ci_mic_1, color=color[0], alpha=0.3)
    plt.fill_between(bit_per_it * np.arange(0, iter, 5), c_mic_avg - c_ci_mic_avg, c_mic_avg + c_ci_mic_avg, color=color[1], alpha=0.3)
    plt.fill_between(bit_per_it * np.arange(0, iter, 5), mic_avg - ci_mic_avg, mic_avg + ci_mic_avg, color=color[1], alpha=0.3)
    plt.fill_between(bit_per_it * np.arange(0, iter, 5), mic_2 - ci_mic_2, mic_2 + ci_mic_2, color=color[2], alpha=0.3)
    plt.fill_between(bit_per_it * np.arange(0, iter, 5), c_mic_2 - c_ci_mic_2, c_mic_2 + c_ci_mic_2, color=color[2], alpha=0.3)


plt.legend(loc='lower right')
plt.xlim([0,5000])
plt.ylim([0.4,0.89])
plt.grid()
plt.ylabel('Accuracy')
plt.xlabel('Iteration')
plt.savefig("COOS7_Example.pdf")
plt.show()
