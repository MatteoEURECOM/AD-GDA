import numpy as np
from operator import add
import tensorflow as tf
from models import CNN
from util import quantization, topKsparsification, euclidean_proj_simplex

class Node(object):
    """
    Class implementing the network node
    """
    def __init__(self,id,frac,train_data,test_data,d):
        self.cid = id
        if(d['model']=='Logistic'):
            x_shape= np.prod(d['x_shape'])
            train_data[0], test_data[0] = train_data[0].reshape([-1, x_shape]), test_data[0].reshape([-1, x_shape])
            self.model= Logistic(x_shape,d['num_classes'])
        elif(d['model']=='FullyConnected'):
            x_shape= np.prod(d['x_shape'])
            train_data[0], test_data[0] = train_data[0].reshape([-1, x_shape]), test_data[0].reshape([-1,x_shape])
            self.model= fullyConnected(x_shape,d['num_classes'])
        elif (d['model'] == 'CNN'):
            self.model = CNN(d['x_shape'], d['num_classes'])
        self.batch_size= d['batchsize']
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=d['eta_x'])
        self.loss_fn=  lambda y_true, y_pred: tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true , y_pred, from_logits=False))
        self.lambdas,self.frac= np.ones(d['nodes'])/d['nodes'],frac
        self.s,self.theta_hat=[],[]
        self.etax,self.mu=d['eta_x'],d['mu']
        self.iteration=0

    def initialize(self,init):
        '''Initialized model params and set public variables to zero'''
        zeroed=[w*0 for w in init]
        self.s=zeroed.copy()
        self.theta_hat=zeroed.copy()
        self.model.set_weights(init.copy())

    def get_model_params(self):
        """Get model parameters"""
        return self.model.get_weights()

    def set_model_params(self, model_params_dict):
        """Set model parameters"""
        self.model.set_weights(model_params_dict)

    def udate_params(self,theta_hat_up):
        '''Updates local variables'''
        self.model.set_weights(theta_hat_up)

    def set_lamda(self,new):
        '''Updates dual variable lambda'''
        self.lambdas= new.copy()

    def update_s(self,update):
        '''Updates public variable s'''
        self.s=list( map(add, update, self.s) )

    def update_theta_hat(self,update):
        '''Updates public variable theta_hat'''
        self.theta_hat=list( map(add, update,self.theta_hat))


    def gradient_step(self):
        """Query oracle and update primal and dual variables"""
        ind=np.arange(self.iteration*self.batch_size,(self.iteration+1)*self.batch_size)%(self.train_data[0].shape[0])
        geom_decay=(0.98)**(self.iteration/10.)
        with tf.GradientTape() as tape:
            preds=self.model(self.train_data[0][ind])
            loss_value=self.loss_fn(self.train_data[1][ind], preds)
        # Query Oracle
        theta_grad = tape.gradient(loss_value, self.model.trainable_variables)
        # Update Variables
        h=(np.exp(loss_value/self.mu)/self.mu)
        print(geom_decay*self.etax*h)
        self.optimizer.learning_rate.assign(geom_decay*self.etax*h)
        self.optimizer.apply_gradients(zip(theta_grad, self.model.trainable_variables))
        self.iteration=self.iteration+1

    def get_msg(self):
        return self.model.get_weights()

    def local_test(self):
        """Test current model on local eval data"""
        preds=np.argmax(self.model(self.test_data[0]),axis=1)
        return np.mean(np.argmax(self.test_data[1],axis=1)!= preds)
