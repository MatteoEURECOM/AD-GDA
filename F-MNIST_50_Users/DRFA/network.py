import numpy as np
from util import euclidean_proj_simplex_t
import util
import tensorflow as tf
from node import Node
import argparse

parser = argparse.ArgumentParser(description='DRFA Params')
parser.add_argument('-d', '--dataset', default='FMNIST', choices=['FMNIST','CIFAR'], help='dataset')
parser.add_argument('-m', '--model', default='FullyConnected', choices=['FullyConnected','Logistic','CNN'], help='Dataset name.')
parser.add_argument('-b', '--batchsize', default=50,help='batch size')
parser.add_argument('-n','--nodes', default=50,  choices=[10,5], help='number of network nodes')
parser.add_argument('-mode','--mode', default='ROBUST',  choices=['ROBUST','NOT_ROBUST'], help='type of training')
parser.add_argument('-T','--iter', default=500, help='number of iterations')
parser.add_argument('-lT','--local_iter', default=10, help='number of local iterations')
parser.add_argument('-r','--MCReps', default=5, help='number of Monte Carlo repetitions')
parser.add_argument('-pdev','--part_dev', default=25, help='number of participating devices each round')
parser.add_argument('-etax','--eta_x', default=1/50., help='primal variable step size')
parser.add_argument('-etay','--eta_y', default=8e-3, help='dual variable step size')
parser.add_argument('-mu','--mu', default=0.01, help='regularization parameter')
parser.add_argument('-sched', '--lr_sched', default='Geom', choices=['Geom','Step'], help='learning rate scheduler')
args = parser.parse_args()
d = vars(args)
if(d['mode']=='NOT_ROBUST'):
    d['eta_y']=0
'''Loading the dataset'''
if(d['dataset']=='FMNIST'):
    tr_x, tr_y, te_x , te_y,  d['x_shape'], d['num_classes'], fracs= util.loadFASHION_MNIST(d['nodes'])
elif(d['dataset']=='CIFAR'):
    tr_x, tr_y, te_x, te_y, d['x_shape'], d['num_classes'], fracs = util.loadCIFAR10_CONTRAST(d['nodes'])
'''Fix seed for reproducibility purposes'''
np.random.seed(0)
tf.random.set_seed(0)
'''Logging Data'''
log,LOG,log_Net=[],[],[]
'''Training '''
for rep in range(0,d['MCReps']):
    print('Repetition: '+str(rep))
    node_list=[Node(i,fracs[i],[tr_x[i],tr_y[i]],[te_x[i],te_y[i]],d) for i in range(0,d['nodes'])]
    init=node_list[0].get_model_params()
    [node.initialize(init) for node in node_list] #initialization
    accuracy=[]
    lambdas=np.ones(d['nodes'])/d['nodes']
    eta_x=d['eta_x']
    for it in range(0,d['iter']):
        if(d['lr_sched']=='Geom'):
            '''Geometrically decaying step size'''
            geom_decay = (0.98)**(it*d['local_iter']/10.)
            eta_x=d['eta_x']*geom_decay
        elif(d['lr_sched']=='Step'):
            '''Step decaying step size'''
            if (it > 0 and it*d['local_iter'] % 2000 == 0):
                eta_x = eta_x / 2.0
        lambdas=lambdas/np.sum(lambdas)
        if(d['mode']=='ROBUST'):
            selected=np.random.choice(range(0,d['nodes']),d['part_dev'],replace=False,p=lambdas)
        else:
            selected=np.random.choice(range(0,d['nodes']),d['part_dev'],replace=False)
        msg=[node_list[i].local_train(eta_x) for i in selected]  #Local Computation
        weights=[m[0] for m in msg]
        weights_sampled = [m[1] for m in msg]
        weights=([[w[w_ind] for w in weights] for w_ind in range(0,len(weights[0]))])
        weights_sampled = ([[w[w_ind] for w in weights_sampled] for w_ind in range(0, len(weights_sampled[0]))])
        W=np.ones(d['part_dev'])/d['part_dev']
        avg_w=[np.average(weights[w_ind],axis=0,weights=W) for w_ind in range(0,len(weights))]
        avg_w_sampled=[np.average(weights_sampled[w_ind],axis=0,weights=W) for w_ind in range(0,len(weights_sampled))]
        u_selected = np.random.choice(range(0, d['nodes']), d['nodes'], replace=False)
        local_loss=[node_list[i].local_test_lambda(avg_w_sampled) for i in u_selected]
        node_ind = 0
        for i in u_selected:
            lambdas[i]=lambdas[i]+d['nodes']*d['eta_y']*d['local_iter']*local_loss[node_ind]/d['part_dev']
            node_ind=node_ind+1
        lambdas=euclidean_proj_simplex_t(lambdas)
        lambdas =lambdas.numpy()
        zero=lambdas<1e-5
        lambdas[zero]=1e-5
        [node.set_model_params(avg_w) for node in node_list]
        if(it>0 and it%100==0):
            print('--------------Iteration: '+str(it)+' --------------')
            print('Loss: '+str([round(accuracy[len(accuracy)-d['nodes']+i],2) for i in range(0,d['nodes'])]))
            print('Min Loss:'+str(np.min([round(accuracy[len(accuracy)-d['nodes']+i],2) for i in range(0,d['nodes'])])))
            print('Lambda:  '+str(np.round(lambdas,4)))
        node_ind=0
        if (it % 1 == 0):
            for node in node_list:
                accuracy.append(1-node.local_test())
                node_ind=node_ind+1
    accuracy=[accuracy[i:(np.asarray(accuracy)).shape[0]:d['nodes']] for i in range(0,d['nodes'])]
    log.append(np.asarray(accuracy))
np.save(str(d['dataset'])+'DRFA_ROBUST.npy', log, allow_pickle=True)
