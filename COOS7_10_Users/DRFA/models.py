from tensorflow.keras.layers import  Dense,Input,Conv2D,AveragePooling2D,Flatten,GRU
from tensorflow.keras.models import Model
import keras
import tensorflow as tf
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

def CNN(shape,num_classes):
    '''
    :param shape: Input Shape
    :param num_classes: Number of classes
    :return: Convolutional Neural Network
    '''
    input=Input(shape=shape)
    Cnn1= Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform')(input)
    Avg1=AveragePooling2D()(Cnn1)
    Cnn2= Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform')(Avg1)
    Avg2=AveragePooling2D()(Cnn2)
    Flat=Flatten()(Avg2)
    Dense1=Dense(64, activation='relu')(Flat)
    Dense2=Dense(32, activation='relu')(Dense1)
    predictions=Dense(num_classes, activation='softmax')(Dense2)
    architecture = Model(inputs=input, outputs=predictions)
    return architecture

