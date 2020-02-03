from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten

from batch_generator import Generator
from utils import INPUT_SHAPE, batch_generator, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS
from keras import backend as K
from PIL import Image
import numpy as np

def build_model(args):
    '''
    Modified NVIDIA model
    '''
    model = Sequential()
    model.add(Lambda((lambda x: ((x / 127.5) - 1.0)), input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 3)))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    return model


def train_model(x_train, x_valid, y_train, y_valid, model_name, args):
    '''
    Train the model
    '''
    model = build_model(args)
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))
    train_generator = Generator(x_train, y_train, True, args)
    validation_generator = Generator(x_valid, y_valid, False, args)
    
    model.fit_generator(train_generator, validation_data=\
        validation_generator, epochs=\
        args.nb_epoch, use_multiprocessing=\
        False, max_queue_size=\
        10, workers=\
        4)
    
    model.save(model_name)