#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 17:14:21 2019

@author: usi
"""

import pandas as pd
import os
import argparse
import numpy
from sklearn.model_selection import train_test_split
from keras import backend as K
from udacity_train import train_model

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
            
            path = os.path.join("/home/ubuntu/dataset5/", track, drive_style, 'driving_log.csv')
            data_df = pd.read_csv(path)
            if x is None:
                x = data_df[['center', 'left', 'right']].values
                y = data_df['steering'].values
            else:
                x = numpy.concatenate((x, data_df[['center', 'left', 'right']].values), axis=0)
                y = numpy.concatenate((y, data_df['steering'].values), axis=0)
        
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=rs)
    return x_train, x_valid, y_train, y_valid

f = open("errors.txt","w+")
errNum = 0
datasetPath = "/home/ubuntu/dataset5/"
args = get_args(datasetPath)
num = 20 

"""
try:
    import mutant1
    for i in range(0, num):
        print("Mutant " + str(1) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant1_" + str(i) + ".h5"
        mutant1.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    print(str(e))
    f.write("for " + str(1) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant2
    for i in range(0, num):
        print("Mutant " + str(2) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant2_" + str(i) + ".h5"
        mutant2.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(2) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant3
    for i in range(0, num):
        print("Mutant " + str(3) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant3_" + str(i) + ".h5"
        mutant3.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(3) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant4
    for i in range(0, num):
        print("Mutant " + str(4) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant4_" + str(i) + ".h5"
        mutant4.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(4) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant5
    for i in range(0, num):
        print("Mutant " + str(5) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant5_" + str(i) + ".h5"
        mutant5.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(5) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant6
    for i in range(0, num):
        print("Mutant " + str(6) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant6_" + str(i) + ".h5"
        mutant6.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(6) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant7
    for i in range(0, num):
        print("Mutant " + str(7) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant7_" + str(i) + ".h5"
        mutant7.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(7) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1
"""

try:
    import mutant8
    for i in range(0, num):
        print("Mutant " + str(8) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant8_" + str(i) + ".h5"
        mutant8.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    print(str(e))
    f.write("for " + str(8) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant9
    for i in range(1, num):
        print("Mutant " + str(9) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant9_" + str(i) + ".h5"
        mutant9.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    print(str(e))
    f.write("for " + str(9) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant10
    for i in range(1, num):
        print("Mutant " + str(10) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant10_" + str(i) + ".h5"
        mutant10.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    print(str(e))
    f.write("for " + str(10) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant11
    for i in range(1, num):
        print("Mutant " + str(11) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant11_" + str(i) + ".h5"
        mutant11.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(11) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant12
    for i in range(1, num):
        print("Mutant " + str(12) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant12_" + str(i) + ".h5"
        mutant12.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(12) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant13
    for i in range(1, num):
        print("Mutant " + str(13) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant13_" + str(i) + ".h5"
        mutant13.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(13) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant14
    for i in range(1, num):
        print("Mutant " + str(14) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant14_" + str(i) + ".h5"
        mutant14.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(14) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant15
    for i in range(1, num):
        print("Mutant " + str(15) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant15_" + str(i) + ".h5"
        mutant15.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(15) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant16
    for i in range(1, num):
        print("Mutant " + str(16) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant16_" + str(i) + ".h5"
        mutant16.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(16) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant17
    for i in range(1, num):
        print("Mutant " + str(17) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant17_" + str(i) + ".h5"
        mutant17.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(17) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant18
    for i in range(1, num):
        print("Mutant " + str(18) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant18_" + str(i) + ".h5"
        mutant18.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(18) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant19
    for i in range(1, num):
        print("Mutant " + str(19) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant19_" + str(i) + ".h5"
        mutant19.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(19) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant20
    for i in range(1, num):
        print("Mutant " + str(20) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant20_" + str(i) + ".h5"
        mutant20.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(20) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant21
    for i in range(1, num):
        print("Mutant " + str(21) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant21_" + str(i) + ".h5"
        mutant21.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(21) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant22
    for i in range(1, num):
        print("Mutant " + str(22) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant22_" + str(i) + ".h5"
        mutant22.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(22) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant23
    for i in range(1, num):
        print("Mutant " + str(23) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant23_" + str(i) + ".h5"
        mutant23.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(23) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant24
    for i in range(1, num):
        print("Mutant " + str(24) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant24_" + str(i) + ".h5"
        mutant24.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(24) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant25
    for i in range(1, num):
        print("Mutant " + str(25) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant25_" + str(i) + ".h5"
        mutant25.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(25) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant26
    for i in range(1, num):
        print("Mutant " + str(26) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant26_" + str(i) + ".h5"
        mutant26.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(26) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant27
    for i in range(1, num):
        print("Mutant " + str(27) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant27_" + str(i) + ".h5"
        mutant27.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(27) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant28
    for i in range(1, num):
        print("Mutant " + str(28) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant28_" + str(i) + ".h5"
        mutant28.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(28) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant29
    for i in range(1, num):
        print("Mutant " + str(29) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant29_" + str(i) + ".h5"
        mutant29.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(29) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant30
    for i in range(1, num):
        print("Mutant " + str(30) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant30_" + str(i) + ".h5"
        mutant30.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(30) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant31
    for i in range(1, num):
        print("Mutant " + str(31) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant31_" + str(i) + ".h5"
        mutant31.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(31) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant32
    for i in range(1, num):
        print("Mutant " + str(32) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant32_" + str(i) + ".h5"
        mutant32.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(32) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant33
    for i in range(1, num):
        print("Mutant " + str(33) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant33_" + str(i) + ".h5"
        mutant33.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(33) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant34
    for i in range(1, num):
        print("Mutant " + str(34) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant34_" + str(i) + ".h5"
        mutant34.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(34) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant35
    for i in range(1, num):
        print("Mutant " + str(35) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant35_" + str(i) + ".h5"
        mutant35.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(35) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant36
    for i in range(1, num):
        print("Mutant " + str(36) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant36_" + str(i) + ".h5"
        mutant36.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(36) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant37
    for i in range(1, num):
        print("Mutant " + str(37) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant37_" + str(i) + ".h5"
        mutant37.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(37) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant38
    for i in range(1, num):
        print("Mutant " + str(38) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant38_" + str(i) + ".h5"
        mutant38.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(38) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant39
    for i in range(0, num):
        print("Mutant " + str(39) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant39_" + str(i) + ".h5"
        mutant39.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(39) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant40
    for i in range(1, num):
        print("Mutant " + str(40) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant40_" + str(i) + ".h5"
        mutant40.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(40) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant41
    for i in range(1, num):
        print("Mutant " + str(41) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant41_" + str(i) + ".h5"
        mutant41.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(41) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant42
    for i in range(1, num):
        print("Mutant " + str(42) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant42_" + str(i) + ".h5"
        mutant42.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(42) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant43
    for i in range(1, num):
        print("Mutant " + str(43) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant43_" + str(i) + ".h5"
        mutant43.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(43) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant44
    for i in range(0, num):
        print("Mutant " + str(44) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant44_" + str(i) + ".h5"
        mutant44.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(44) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant45
    for i in range(0, num):
        print("Mutant " + str(45) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant45_" + str(i) + ".h5"
        mutant45.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(45) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant46
    for i in range(0, num):
        print("Mutant " + str(46) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant46_" + str(i) + ".h5"
        mutant46.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(46) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant47
    for i in range(0, num):
        print("Mutant " + str(47) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant47_" + str(i) + ".h5"
        mutant47.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(47) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1


try:
    import mutant48
    for i in range(0, num):
        print("Mutant " + str(48) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant48_" + str(i) + ".h5"
        mutant48.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(48) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1
    
try:
    import mutant49
    for i in range(0, num):
        print("Mutant " + str(49) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant49_" + str(i) + ".h5"
        mutant49.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(49) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1

try:
    import mutant50
    for i in range(0, num):
        print("Mutant " + str(50) + ", Run" + str(i))
        x_train, x_valid, y_train, y_valid = load_data(datasetPath, i)
        model_name = "udacity_mutant50_" + str(i) + ".h5"
        mutant50.train_model(x_train, x_valid, y_train, y_valid, model_name, args)
        K.clear_session()
except Exception as e:
    f.write("for " + str(50) + ", error:" + str(e))
    f.write("\n")
    errNum = errNum + 1

f.write("errNum:" + str(errNum))
f.close()
print("errNum:" + str(errNum))