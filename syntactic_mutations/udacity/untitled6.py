#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 17:14:21 2019

@author: usi
"""

import pandas as pd
import os
import numpy as np
import argparse
from udacity_train import train_model
from sklearn.model_selection import train_test_split
from keras import backend as K

def get_args(dataset = '../datasets/dataset5'):
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default=dataset)
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=100)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=100)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=256)
    #parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='false')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()
    return args

def load_data(args, rs):
    tracks  = ["track1"]
    drive = ["normal"]
    x = None
    y = None
    for track in tracks:
        for drive_style in drive:
            
            path = os.path.join("/Users/usi/Documents/Pahlava/repos/repos/datasets/dataset5/", track, drive_style, 'driving_log.csv')
            data_df = pd.read_csv(path)
            if x is None:
                x = data_df[['center', 'left', 'right']].values
                y = data_df['steering'].values
            else:
                x = np.concatenate((x, data_df[['center', 'left', 'right']].values), axis=0)
                y = np.concatenate((y, data_df['steering'].values), axis=0)
        
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=rs)
    return x_train, x_valid, y_train, y_valid

f = open("errors.txt","w+")
errNum = 0
datasetPath = "/Users/usi/Documents/Pahlava/repos/repos/datasets/dataset5/"
args = get_args(datasetPath)
num = 20
    
try:
    import mutant1
    for i in range(0, num):
            print("Mutant " + str(1) + ", Run" + str(i))
            x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
            model_name = "udacity_mutant1_" + str(i) + ".h5"
            mutant1.train_model(x_train, x_valid, y_train, y_valid, model_name)
            K.clear_session()
except Exception as e:
    print(str(e))
    f.write("for " + str(1) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1

f.write("errNum:" + str(errNum))
f.close()
print("errNum:" + str(errNum))