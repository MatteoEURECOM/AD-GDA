import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance
import torch
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import pickle
pickle.HIGHEST_PROTOCOL = 4

def change_contrast(img,factor):
    '''
    Change the contrast of the image
    :param img: number of data partitions, either 5 or 10
    :param factor: contrast factor > 0
    :return: image with changed contrast (lower contrast: factor<1) (Higher contrast: factor>1)
    '''
    factor = float(factor)
    return(np.clip(np.power(np.clip(128 + factor * (img * 255 - 128), 0, 255),1.1), 0, 255).astype(np.uint8))/255.

def loadCOOS7(nodes):
    '''
    Partition the Fashion_mnist dataset according to labels in 5 or 10 shards
    :param nodes: number of data partitions, either 5 or 10
    :return: class-wise partitioned dataset
    '''
    n_tr=1000
    n_te = 1000
    mic_1 = h5py.File('COOS7_v1.1_test3.hdf5', 'r')
    im_1= mic_1['data'][:]
    l_1 = mic_1['labels'][:]
    mic_2 = h5py.File('COOS7_v1.1_test4.hdf5', 'r')
    im_2 = mic_2['data'][:]
    l_2 = mic_2['labels'][:]
    im_1= im_1 / np.max(im_1)
    im_1 = np.rollaxis(im_1, 1, 4)
    im_2= im_2 / np.max(im_2)
    im_2= np.rollaxis(im_2, 1, 4)
    num_classes = 7
    x_shape = im_1.shape[1:]
    if nodes == 5:
        train_X = [np.squeeze(im_2[i*n_tr:(i +1)*n_tr]).astype(np.float16) for i in range(0, nodes-1)]
        train_X.append(np.squeeze(im_1[0:n_tr]).astype(np.float16))
        test_X = [np.squeeze(im_2[i*n_te:(i +1)*n_te]).astype(np.float16) for i in range(nodes, 2*nodes-1)]
        test_X.append(np.squeeze(im_1[n_tr:2*n_tr]).astype(np.float16))
        train_Y = [np.squeeze(l_2[i*n_tr:(i +1)*n_tr]) for i in range(0, nodes-1)]
        train_Y.append(np.squeeze(l_1[0:n_tr]))
        test_Y = [np.squeeze(l_2[i*n_te:(i +1)*n_te]) for i in range(nodes, 2*nodes-1)]
        test_Y.append(np.squeeze(l_1[n_te:2*n_te]))
    elif nodes == 10:
        train_X = [np.squeeze(im_2[i * n_tr:(i + 1) * n_tr]).astype(np.float16) for i in range(0, nodes - 2)]
        train_X.append(np.squeeze(im_1[0:n_tr]).astype(np.float16))
        train_X.append(np.squeeze(im_1[n_tr:2*n_tr]).astype(np.float16))
        test_X = [np.squeeze(im_2[i * n_te:(i + 1) * n_te]).astype(np.float16) for i in range(nodes, 2 * nodes - 2)]
        test_X.append(np.squeeze(im_1[2*n_tr:3 * n_tr]).astype(np.float16))
        test_X.append(np.squeeze(im_1[3 * n_tr:4 * n_tr]).astype(np.float16))
        train_Y = [np.squeeze(l_2[i * n_tr:(i + 1) * n_tr]) for i in range(0, nodes - 2)]
        train_Y.append(np.squeeze(l_1[0:n_tr]))
        train_Y.append(np.squeeze(l_1[n_tr:2*n_tr]))
        test_Y = [np.squeeze(l_2[i * n_te:(i + 1) * n_te]) for i in range(nodes, 2 * nodes - 2)]
        test_Y.append(np.squeeze(l_1[2*n_te:3* n_te]))
        test_Y.append(np.squeeze(l_1[3 * n_te:4 * n_te]))
    train_Y = [tf.one_hot(train_Y[i], depth=num_classes).numpy() for i in range(0, nodes)]
    test_Y = [tf.one_hot(test_Y[i], depth=num_classes).numpy() for i in range(0, nodes)]
    fracs=[x.shape[0] for x in train_X]
    fracs=[f/np.sum(fracs) for f in fracs]
    df = pd.DataFrame({'train_X': train_X, 'train_Y': train_Y})
    df.to_hdf('train.h5', key='df', mode='w')
    df = pd.DataFrame({'test_X': test_X, 'test_Y': test_Y})
    df.to_hdf('test.h5', key='df', mode='w')
    return train_X, train_Y, test_X , test_Y,  x_shape, num_classes, fracs

def euclidean_proj_simplex(v, s=1):
    '''
    Projection operator over the s-simplex
    From:
    Large-scale Multiclass Support Vector Machine Training via Euclidean Projection onto the Simplex
    Mathieu Blondel, Akinori Fujino, and Naonori Ueda.
    ICPR 2014.
    http://www.mblondel.org/publications/mblondel-icpr2014.pdf

    :param v: vector that has to be sent over the simplex
    :param s: magnitude of the simplex
    :return: projected vector
    '''
    n, = v.shape
    if v.sum() == s and np.alltrue(v >= 0):
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    theta = float(cssv[rho] - s) / rho
    w = (v - theta).clip(min=0)
    return w

def topKsparsification(x,perc=1,random=False):
    '''
    Sparsification operator
    :param x: vector to sparsify
    :param perc: percentage of components to save
    :param random: if True random sparsification is applied, otherwise choose largest components
    :return: sparsified vector
    '''
    w_size=[weight.size for weight in x]
    if random:
        indices=np.random.choice(np.sum(w_size), int(np.sum(w_size)*(1.0-perc)), replace=False)
        start_interval=0
        for i in range(0,len(x)):
            curr_ind=indices[np.where(np.logical_and(indices>=start_interval, indices<(start_interval+w_size[i])))[0]]-start_interval
            flat_x=x[i].flatten()
            flat_x[curr_ind]=0
            x[i] = flat_x.reshape(x[i].shape)
            start_interval=start_interval+w_size[i]
    else:
        x_flat=np.hstack([w.flatten() for w in x])
        th=np.sort(np.abs(x_flat))
        #ind=np.argsort(np.abs(x_flat))
        #x_flat[ind[0:int(np.sum(w_size)*(1-perc))]]=0
        th=th[int(np.sum(w_size)*(1-perc))]
        ind=0
        for i in range(0,len(x)):
            flat_x=x[i].flatten()
            flat_x[np.where(np.abs(flat_x)<th)]=0
            x[i] = flat_x.reshape(x[i].shape)
            #size=x[i].flatten().size
            #x[i] = x_flat[ind:ind+size].reshape(x[i].shape)
            #ind=ind+size
        #x_flat = np.hstack([w.flatten() for w in x])
    return x

def quantization(x,b=32):
    '''
    Random quantization operator
    :param x: vector to quantize
    :param b: number of bit levels to use
    :return: randomly quantized vector
    '''
    s=2.0**b
    w_size=[np.sum(weight!=0) for weight in x]
    d=np.sum(w_size)
    tau=1.0+min(d/s**2,np.sqrt(d)/s)
    x_flat=np.hstack([w.flatten() for w in x])
    norm=np.sqrt(np.sum(np.power((x_flat),2)))
    for i in range(0,len(x)):
        flat_x=x[i].flatten()
        sign=2.0*((flat_x>0)-0.5)
        quantized=(sign*norm/(tau*s)*np.ceil(s*np.abs(flat_x/norm)+np.random.rand(len(flat_x))))
        quantized[flat_x==0]=0
        x[i] = quantized.reshape(x[i].shape)
    return x


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
            G[i, (i - 2) % nodes] = 1. / 5.
            G[i, (i - 1) % nodes] = 1. / 5.
            G[i, i % nodes] = 1. / 5.
            G[i, (i + 1) % nodes] = 1. / 5.
            G[i, (i + 2) % nodes] = 1. / 5.
    return G

def euclidean_proj_simplex_t(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    v=torch.from_numpy(v)
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and (v >= 0).all():
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = torch.flip(torch.sort(v)[0],dims=(0,))
    cssv = torch.cumsum(u,dim=0)
    # get the number of > 0 components of the optimal solution
    non_zero_vector = torch.nonzero(u * torch.arange(1, n+1) > (cssv - s), as_tuple=False)
    if len(non_zero_vector) == 0:
        rho=0.0
    else:
        rho = non_zero_vector[-1].squeeze()
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clamp(min=0)
    return w
