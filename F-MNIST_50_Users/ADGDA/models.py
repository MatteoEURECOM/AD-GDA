from tensorflow.keras.layers import  Dense,Input
from tensorflow.keras.models import Model
import keras

def Logistic(shape,num_classes):
    '''
    :param shape: Input Shape
    :param num_classes: Number of classes
    :return: Logistic model
    '''
    input=Input(shape=[shape])
    predictions=Dense(num_classes, activation='softmax',kernel_initializer = keras.initializers.Ones(),bias_initializer= keras.initializers.Zeros())(input)
    architecture = Model(inputs=input, outputs=predictions)
    return architecture

def fullyConnected(shape,num_classes):
    '''
    :param shape: Input Shape
    :param num_classes: Number of classes
    :return: Fully Connected Model
    '''
    input=Input(shape=[shape])
    hidden1= Dense(25, activation='relu')(input)
    predictions=Dense(num_classes, activation='softmax')(hidden1)
    architecture = Model(inputs=input, outputs=predictions)
    return architecture

