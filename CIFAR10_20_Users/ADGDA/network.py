import numpy as np
import matplotlib.pyplot as plt
import util
import tensorflow as tf
from node import Node
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='AGDA Params')
    parser.add_argument('-m', '--model', default='CNN', choices=['CNN'], help='Model architecture')
    parser.add_argument('-b', '--batchsize', default=32,type=int,help='batch size')
    parser.add_argument('-n','--nodes', default=20, type=int, choices=[20], help='number of network nodes')
    parser.add_argument('-mode','--mode', default='ROBUST',  choices=['ROBUST','NOT_ROBUST'], help='type of training')
    parser.add_argument('-T','--iter', default=5000,  type=int,help='number of iterations')
    parser.add_argument('-r','--MCReps', default=3,  type=int,help='number of Monte Carlo repetitions')
    parser.add_argument('-g','--gamma', default=0.7, type=float, help='consensus step size')
    parser.add_argument('-cs','--scheme', default='Sparsification',choices=['Sparsification','Quantization','Combination'], help='type of compression operator')
    parser.add_argument('-s','--save', default=1, type=float, help='percentage of components that the sparsificication operator saves') #[1,0.5,0.25]
    parser.add_argument('-bit','--bits', default=4,type=int, help='bits used for the random quantization')
    parser.add_argument('-etax','--eta_x', default=1., type=float, help='primal variable step size')
    parser.add_argument('-etay','--eta_y', default=0.01, type=float, help='primal variable step size')
    parser.add_argument('-mu','--mu', default=0.001,  type=float,help='regularization parameter')
    parser.add_argument('-t','--topology', default='2DTORUS', choices=['RING','STAR','2DTORUS','MESH'], help='network topology')
    args = parser.parse_args()
    d = vars(args)
    return d

def network(d):
    print('Called with args:')
    print(d)
    if(d['mode']=='NOT_ROBUST'):
        d['eta_y']=0.
    '''Loading the dataset'''
    tr_x, tr_y, te_x, te_y, d['x_shape'], d['num_classes'], fracs = util.loadCIFAR10_CONTRAST(d['nodes'])
    '''Shuffling of the data placement'''
    np.random.seed(1)
    tf.random.set_seed(1)
    plac=np.tile(np.arange(0,d['nodes']),(d['MCReps'],1))
    [np.random.shuffle(x) for x in plac]
    G=util.connectivity_matrix(d['nodes'],d['topology'])
    '''Fix seed for reproducibility purposes'''
    np.random.seed(1)
    tf.random.set_seed(1)
    '''Logging Data'''
    log=[]
    LOG=[]
    log_Net=[]
    '''Training '''
    for rep in range(0,d['MCReps']):
        print('Repetition: '+str(rep))
        p_ind=plac[rep,:]
        node_list=[Node(i,fracs[p_ind[i]],[tr_x[(p_ind[i])%d['nodes']],tr_y[(p_ind[i])%d['nodes']]],[te_x[(p_ind[i])%d['nodes']],te_y[(p_ind[i])%d['nodes']]],d) for i in range(0,d['nodes'])]
        init=node_list[0].get_model_params()
        [node.initialize(init) for node in node_list] #initialization
        accuracy=[]
        accuracy_Net=[]
        for it in range(0,d['iter']):
            [node.gradient_step() for node in node_list]    #Local computation
            msgs=[node.get_msg() for node in node_list]     #Messages to send
            q=([[m[0][w_ind] for m in msgs] for w_ind in range(0,len(msgs[0][0]))])   #List of Numpy
            lambdas=[m[1] for m in msgs]    #List of Lambdas
            weights=[node.get_model_params() for node in node_list]  #Get Local weights to compute network averaged model
            weights=([[w[w_ind] for w in weights] for w_ind in range(0,len(weights[0]))])
            if(it%100==0 and it>0) :
                print('--------------Iteration: '+str(it)+' --------------mu: '+str(d['mu']))
                print('Local Losses: '+str([round(accuracy[len(accuracy)-d['nodes']+i],2) for i in range(0,d['nodes'])]))
                print('Network Loss: '+str([round(accuracy_Net[len(accuracy_Net)-d['nodes']+i],2) for i in range(0,d['nodes'])]))
                print('Min Loss: ' + str(np.min([round(accuracy_Net[len(accuracy_Net) - d['nodes'] + i], 2) for i in range(0, d['nodes'])])))
                print('Lambdas:      ' + str([round(lambdas[i][i], 4) for i in range(0, d['nodes'])]))
            node_ind=0
            for node in node_list:
                if(it%5==0):
                    '''Local testing to log performance during training'''
                    node.set_model_params([np.average(weights[w_ind],axis=0,weights=np.ones(d['nodes'])/d['nodes']) for w_ind in range(0,len(weights))])
                    accuracy_Net.append(1-node.local_test())
                    node.set_model_params([np.average(weights[w_ind],axis=0,weights=np.eye(d['nodes'])[node_ind,:]) for w_ind in range(0,len(weights))])
                    accuracy.append(1-node.local_test())
                '''Message exchange'''
                avgd_q=[np.average(q[w_ind],axis=0,weights=G[node_ind,:]) for w_ind in range(0,len(q))]
                own_q=[np.average(q[w_ind],axis=0,weights=np.eye(d['nodes'])[node_ind,:]) for w_ind in range(0,len(q))]
                avgd_lambda=np.average(lambdas,axis=0,weights=G[node_ind,:])
                '''Local Variables Update'''
                node.udate_params(avgd_lambda,avgd_q,own_q)
                node_ind=node_ind+1
        accuracy=[accuracy[i:(np.asarray(accuracy)).shape[0]:d['nodes']] for i in range(0,d['nodes'])]
        accuracy_Net=[accuracy_Net[i:(np.asarray(accuracy_Net)).shape[0]:d['nodes']] for i in  range(0,d['nodes'])]
        log.append(np.asarray(accuracy))
        log_Net.append(np.asarray(accuracy_Net))
    LOG=[log,log_Net]
    np.save(d['model']+'_'+d['topology']+'_'+str(d['gamma'])+'_'+str(d['mode'])+'_'+str(d['scheme'])+'_'+str(d['mu'])+'.npy', LOG, allow_pickle=True)


if __name__ == '__main__':
    d = parse_args()
    network(d)
