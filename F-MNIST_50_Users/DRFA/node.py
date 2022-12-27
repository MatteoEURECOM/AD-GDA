import numpy as np
from operator import add
import tensorflow as tf
from models import Logistic, fullyConnected,CNN
from util import euclidean_proj_simplex

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
        print(np.sum([np.prod(v.shape) for v in  self.model.get_weights()]))
        self.batch_size= d['batchsize']
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=d['eta_x'])
        self.loss_fn=  lambda y_true, y_pred: tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true , y_pred, from_logits=False))
        self.lambdas,self.frac= np.ones(d['nodes'])/d['nodes'],frac
        self.s,self.theta_hat=[],[]
        self.etax,self.mu=d['eta_x'],d['mu']
        self.local_iterations=d['local_iter']
        self.iteration=0

    def initialize(self,init):
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

    def udate_params(self,new_lambda,s_up,theta_hat_up):
        self.lambdas= new_lambda.copy()
        self.s=list( map(add, s_up, self.s) )
        self.theta_hat=list( map(add, theta_hat_up,self.theta_hat) )

    def local_train(self,eta_x):
        """Query oracle and update primal and dual variables"""
        self.etax=eta_x
        ind_ret=np.random.choice(np.arange(0,self.local_iterations),1)
        for loc_iter in range(0,self.local_iterations):
            ind=np.random.choice(np.arange(0,self.train_data[0].shape[0]),self.batch_size,replace=False)
            with tf.GradientTape() as tape:
                preds=self.model(self.train_data[0][ind])
                loss_value=self.loss_fn(self.train_data[1][ind], preds)
            # Query Oracle
            theta_grad = tape.gradient(loss_value, self.model.trainable_variables)
            # Update Variables
            #print(self.etax)
            self.optimizer.learning_rate.assign(self.etax)
            self.optimizer.apply_gradients(zip(theta_grad, self.model.trainable_variables))
            if(loc_iter==ind_ret):
                to_ret= self.model.get_weights()
            self.iteration=self.iteration+1
        return self.model.get_weights(),to_ret

    def local_test_lambda(self,model):
        """Test current model on local eval data"""
        old=self.model.get_weights()
        self.model.set_weights(model)
        ind = np.random.choice(np.arange(0, self.train_data[0].shape[0]), self.batch_size, replace=False)
        preds = self.model(self.train_data[0][ind])
        loss_hard=np.mean(np.argmax(self.train_data[1][ind], axis=1) != np.argmax(preds,axis=1))
        loss_value = self.loss_fn(self.train_data[1][ind], preds)
        self.model.set_weights(old)
        return float(loss_hard)


    def local_test(self):
        """Test current model on local eval data"""
        preds=np.argmax(self.model(self.test_data[0]),axis=1)
        return np.mean(np.argmax(self.test_data[1],axis=1)!= preds)