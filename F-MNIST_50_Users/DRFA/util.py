import numpy as np
import tensorflow as tf
import torch


def loadFASHION_MNIST(nodes):
    '''
    Partition the Fashion_mnist dataset according to labels in 5 or 10 shards
    :param nodes: number of data partitions, either 5 or 10
    :return: class-wise partitioned dataset
    '''
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train / 255., x_test / 255.
    num_classes = 10
    x_shape = x_train.shape[1:]
    if nodes == 5:
        train_X = [np.squeeze(x_train[np.argwhere((y_train == 2 * i) + (y_train == 2 * i + 1))]) for i in range(0, nodes)]
        test_X = [np.squeeze(x_test[np.argwhere((y_test == 2 * i) + (y_test == 2 * i + 1))]) for i in range(0, nodes)]
        test_Y = [np.squeeze(y_test[np.argwhere((y_test == 2 * i) + (y_test == 2 * i + 1))]) for i in range(0, nodes)]
        train_Y = [np.squeeze(y_train[np.argwhere((y_train == 2 * i) + (y_train == 2 * i + 1))]) for i in range(0, nodes)]
    elif nodes == 10:
        train_X = [np.squeeze(x_train[np.argwhere(y_train == i)]) for i in range(0, nodes)]
        test_X = [np.squeeze(x_test[np.argwhere(y_test == i)]) for i in range(0, nodes)]
        test_Y = [np.squeeze(y_test[np.argwhere(y_test == i)]) for i in range(0, nodes)]
        train_Y = [np.squeeze(y_train[np.argwhere(y_train == i)]) for i in range(0, nodes)]
    elif nodes == 50:
        train_X = np.vstack([np.split(np.squeeze(x_train[np.argwhere(y_train == i)]), 5) for i in range(0, 10)])
        test_X = [np.squeeze(x_test[np.argwhere(y_test == i)])for i in range(0, 10)]
        test_X = [item for item in test_X for i in range(5)]
        test_Y = [np.squeeze(y_test[np.argwhere(y_test == i)]) for i in range(0, 10)]
        test_Y = [item for item in test_Y for i in range(5)]
        train_Y = np.vstack([np.split(np.squeeze(y_train[np.argwhere(y_train == i)]), 5) for i in range(0, 10)])
    train_Y = [tf.one_hot(train_Y[i], depth=num_classes).numpy() for i in range(0, nodes)]
    test_Y = [tf.one_hot(test_Y[i], depth=num_classes).numpy() for i in range(0, nodes)]
    fracs=[x.shape[0] for x in train_X]
    fracs=[f/np.sum(fracs) for f in fracs]
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
        th=th[int(np.sum(w_size)*(1-perc))]
        for i in range(0,len(x)):
            flat_x=x[i].flatten()
            flat_x[np.where(np.abs(flat_x)<th)]=0
            x[i] = flat_x.reshape(x[i].shape)
    return x

def quantization(x,b=32):
    '''
    Random quantization operator
    :param x: vector to quantize
    :param b: number of bit levels to use
    :return: randomly quantized vector
    '''
    s=2.0**b
    w_size=[weight.size for weight in x]
    d=np.sum(w_size)
    tau=1.0+min(d/s**2,np.sqrt(d)/s)
    x_flat=np.hstack([w.flatten() for w in x])
    norm=np.sqrt(np.sum(np.power((x_flat),2)))
    for i in range(0,len(x)):
        flat_x=x[i].flatten()
        sign=2.0*((flat_x>0)-0.5)
        quantized=(sign*norm/(tau*s)*np.ceil(s*np.abs(flat_x/norm)+np.random.rand(len(flat_x))))
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
            G[i, (i - 1) % nodes] = 1 / 3
            G[i, i % nodes] = 1 / 3
            G[i, (i + 1) % nodes] = 1 / 3
    if (type == 'MESH'):
        G = np.ones((nodes, nodes)) / nodes
    if (type == '2DTORUS'):
        G = np.zeros((nodes, nodes))
        for i in range(0, nodes):
            G[i, (i - 2) % nodes] = 1 / 5
            G[i, (i - 1) % nodes] = 1 / 5
            G[i, i % nodes] = 1 / 5
            G[i, (i + 1) % nodes] = 1 / 5
            G[i, (i + 2) % nodes] = 1 / 5
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
