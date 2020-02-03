'''
Created on Fri Jul 26 14:18:35 2019

@author: usi
'''

from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os


batch_size = 128
epochs = 10
num_classes = 10

subtract_pixel_mean = True

n = 3
version = 1

depth = (n * 6) + 2


model_type = 'ResNet%dv%d' * (depth, version)

def lr_schedule(epoch):
    lr = 0.001
    if epoch > 180:
        lr *= 0.0005
    elif epoch > 160:
        lr *= 0.001
    elif epoch > 120:
        lr *= 0.01
    elif epoch > 80:
        lr *= 0.1
    return lr


def resnet_layer(inputs, \
    num_filters=16, \
    kernel_size=3, \
    strides=1, \
    activation='relu', \
    batch_normalization=True, \
    conv_first=True):
    conv = Conv2D(num_filters, kernel_size=\
        kernel_size, strides=\
        strides, padding=\
        'same', kernel_initializer=\
        'he_normal', kernel_regularizer=\
        l2(0.0001))
    
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10):
    
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)
    
    
    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    
    
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if (stack > 0 and res_block == 0):
                strides = 2
            
            y = resnet_layer(inputs=x, num_filters=\
                num_filters, strides=\
                strides)
            
            y = resnet_layer(inputs=y, num_filters=\
                num_filters, activation=\
                None)
            if (stack > 0 and res_block == 0):
                
                x = resnet_layer(inputs=x, num_filters=\
                    num_filters, kernel_size=\
                    1, strides=\
                    strides, activation=\
                    None, batch_normalization=\
                    False)
            
            x = keras.layers.add([x, y])
            
            x = Activation('relu')(x)
        num_filters *= 2
    
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, activation=\
        'softmax', kernel_initializer=\
        'he_normal')(y)
    
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_model(x_train, y_train, x_test, y_test, model_name, layer=-1):
    
    input_shape = x_train.shape[1:]
    
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
    
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    model = resnet_v1(input_shape=input_shape, depth=depth)
    model.compile(loss='categorical_crossentropy', optimizer=\
        Adam(lr=lr_schedule(0)), metrics=\
        ['accuracy'])
    
    lr_scheduler = LearningRateScheduler(lr_schedule)
    
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=\
        0, patience=\
        5, min_lr=\
        5e-07)
    
    callbacks = [lr_reducer, lr_scheduler]
    model.fit(x_train, y_train, batch_size=\
        batch_size, epochs=\
        epochs, validation_data=\
        (x_test, y_test), shuffle=\
        True, callbacks=\
        callbacks)
    
    
    scores = model.evaluate(x_test, y_test, verbose=1)
    model.save(model_name)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    
    return (scores[0], scores[1])