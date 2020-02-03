#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from keras.datasets import mnist
import h5py
import numpy as np
import csv
from keras import backend as K

num = 20

def get_dataset(i):
    dataset_file = "/home/ubuntu/mnist_datasets/crossval_set_" + str(i) + ".h5"
    hf = h5py.File(dataset_file, 'r')
    xn_train = np.asarray(hf.get('xn_train'))
    xn_test = np.asarray(hf.get('xn_test'))
    yn_train = np.asarray(hf.get('yn_train'))
    yn_test = np.asarray(hf.get('yn_test'))
    
    return xn_train, yn_train, xn_test, yn_test
 
f = open("errors.txt","w+")
errNum = 0

csv_name = "mnist_syntacticmutations_results.csv"    
with open(csv_name, 'w') as f1:
    writer=csv.writer(f1, delimiter=',',lineterminator='\n',)
    writer.writerow(["Mutation Number", "Run Number", "Accuracy", "Loss"])

    try:
        import mutant1
        for i in range(0, num):
                print("Mutant " + str(1) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant1_" + str(i) + ".h5"
                loss, acc = mutant1.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(1), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(1) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant2
        for i in range(0, num):
                print("Mutant " + str(2) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant2_" + str(i) + ".h5"
                loss, acc = mutant2.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(2), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(2) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant3
        for i in range(0, num):
                print("Mutant " + str(3) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant3_" + str(i) + ".h5"
                loss, acc = mutant3.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(3), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(3) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant4
        for i in range(0, num):
                print("Mutant " + str(4) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant4_" + str(i) + ".h5"
                loss, acc = mutant4.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(4), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(4) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant5
        for i in range(0, num):
                print("Mutant " + str(5) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant5_" + str(i) + ".h5"
                loss, acc = mutant5.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(5), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(5) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant6
        for i in range(0, num):
                print("Mutant " + str(6) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant6_" + str(i) + ".h5"
                loss, acc = mutant6.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(6), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(6) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant7
        for i in range(0, num):
                print("Mutant " + str(7) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant7_" + str(i) + ".h5"
                loss, acc = mutant7.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(7), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(7) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant8
        for i in range(0, num):
                print("Mutant " + str(8) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant8_" + str(i) + ".h5"
                loss, acc = mutant8.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(8), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(8) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant9
        for i in range(0, num):
                print("Mutant " + str(9) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant9_" + str(i) + ".h5"
                loss, acc = mutant9.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(9), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(9) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant10
        for i in range(0, num):
                print("Mutant " + str(10) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant10_" + str(i) + ".h5"
                loss, acc = mutant10.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(10), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(10) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant11
        for i in range(0, num):
                print("Mutant " + str(11) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant11_" + str(i) + ".h5"
                loss, acc = mutant11.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(11), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(11) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant12
        for i in range(0, num):
                print("Mutant " + str(12) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant12_" + str(i) + ".h5"
                loss, acc = mutant12.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(12), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(12) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant13
        for i in range(0, num):
                print("Mutant " + str(13) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant13_" + str(i) + ".h5"
                loss, acc = mutant13.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(13), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(13) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant14
        for i in range(0, num):
                print("Mutant " + str(14) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant14_" + str(i) + ".h5"
                loss, acc = mutant14.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(14), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(14) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant15
        for i in range(0, num):
                print("Mutant " + str(15) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant15_" + str(i) + ".h5"
                loss, acc = mutant15.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(15), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(15) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant16
        for i in range(0, num):
                print("Mutant " + str(16) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant16_" + str(i) + ".h5"
                loss, acc = mutant16.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(16), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(16) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant17
        for i in range(0, num):
                print("Mutant " + str(17) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant17_" + str(i) + ".h5"
                loss, acc = mutant17.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(17), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(17) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant18
        for i in range(0, num):
                print("Mutant " + str(18) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant18_" + str(i) + ".h5"
                loss, acc = mutant18.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(18), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(18) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant19
        for i in range(0, num):
                print("Mutant " + str(19) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant19_" + str(i) + ".h5"
                loss, acc = mutant19.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(19), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(19) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant20
        for i in range(0, num):
                print("Mutant " + str(20) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant20_" + str(i) + ".h5"
                loss, acc = mutant20.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(20), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(20) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant21
        for i in range(0, num):
                print("Mutant " + str(21) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant21_" + str(i) + ".h5"
                loss, acc = mutant21.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(21), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(21) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant22
        for i in range(0, num):
                print("Mutant " + str(22) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant22_" + str(i) + ".h5"
                loss, acc = mutant22.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(22), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(22) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant23
        for i in range(0, num):
                print("Mutant " + str(23) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant23_" + str(i) + ".h5"
                loss, acc = mutant23.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(23), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(23) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant24
        for i in range(0, num):
                print("Mutant " + str(24) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant24_" + str(i) + ".h5"
                loss, acc = mutant24.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(24), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(24) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant25
        for i in range(0, num):
                print("Mutant " + str(25) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant25_" + str(i) + ".h5"
                loss, acc = mutant25.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(25), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(25) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant26
        for i in range(0, num):
                print("Mutant " + str(26) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant26_" + str(i) + ".h5"
                loss, acc = mutant26.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(26), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(26) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant27
        for i in range(0, num):
                print("Mutant " + str(27) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant27_" + str(i) + ".h5"
                loss, acc = mutant27.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(27), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(27) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant28
        for i in range(0, num):
                print("Mutant " + str(28) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant28_" + str(i) + ".h5"
                loss, acc = mutant28.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(28), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(28) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant29
        for i in range(0, num):
                print("Mutant " + str(29) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant29_" + str(i) + ".h5"
                loss, acc = mutant29.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(29), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(29) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant30
        for i in range(0, num):
                print("Mutant " + str(30) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant30_" + str(i) + ".h5"
                loss, acc = mutant30.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(30), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(30) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant31
        for i in range(0, num):
                print("Mutant " + str(31) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant31_" + str(i) + ".h5"
                loss, acc = mutant31.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(31), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(31) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant32
        for i in range(0, num):
                print("Mutant " + str(32) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant32_" + str(i) + ".h5"
                loss, acc = mutant32.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(32), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(32) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant33
        for i in range(0, num):
                print("Mutant " + str(33) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant33_" + str(i) + ".h5"
                loss, acc = mutant33.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(33), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(33) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant34
        for i in range(0, num):
                print("Mutant " + str(34) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant34_" + str(i) + ".h5"
                loss, acc = mutant34.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(34), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(34) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant35
        for i in range(0, num):
                print("Mutant " + str(35) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant35_" + str(i) + ".h5"
                loss, acc = mutant35.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(35), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(35) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant36
        for i in range(0, num):
                print("Mutant " + str(36) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant36_" + str(i) + ".h5"
                loss, acc = mutant36.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(36), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(36) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant37
        for i in range(0, num):
                print("Mutant " + str(37) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant37_" + str(i) + ".h5"
                loss, acc = mutant37.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(37), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(37) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant38
        for i in range(0, num):
                print("Mutant " + str(38) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant38_" + str(i) + ".h5"
                loss, acc = mutant38.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(38), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(38) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant39
        for i in range(0, num):
                print("Mutant " + str(39) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant39_" + str(i) + ".h5"
                loss, acc = mutant39.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(39), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(39) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant40
        for i in range(0, num):
                print("Mutant " + str(40) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant40_" + str(i) + ".h5"
                loss, acc = mutant40.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(40), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(40) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant41
        for i in range(0, num):
                print("Mutant " + str(41) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant41_" + str(i) + ".h5"
                loss, acc = mutant41.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(41), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(41) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant42
        for i in range(0, num):
                print("Mutant " + str(42) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant42_" + str(i) + ".h5"
                loss, acc = mutant42.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(42), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(42) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant43
        for i in range(0, num):
                print("Mutant " + str(43) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant43_" + str(i) + ".h5"
                loss, acc = mutant43.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(43), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(43) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant44
        for i in range(0, num):
                print("Mutant " + str(44) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant44_" + str(i) + ".h5"
                loss, acc = mutant44.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(44), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(44) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant45
        for i in range(0, num):
                print("Mutant " + str(45) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant45_" + str(i) + ".h5"
                loss, acc = mutant45.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(45), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(45) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant46
        for i in range(0, num):
                print("Mutant " + str(46) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant46_" + str(i) + ".h5"
                loss, acc = mutant46.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(46), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(46) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant47
        for i in range(0, num):
                print("Mutant " + str(47) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant47_" + str(i) + ".h5"
                loss, acc = mutant47.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(47), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(47) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant48
        for i in range(0, num):
                print("Mutant " + str(48) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant48_" + str(i) + ".h5"
                loss, acc = mutant48.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(48), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(48) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant49
        for i in range(0, num):
                print("Mutant " + str(49) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant49_" + str(i) + ".h5"
                loss, acc = mutant49.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(49), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(49) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant50
        for i in range(0, num):
                print("Mutant " + str(50) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant50_" + str(i) + ".h5"
                loss, acc = mutant50.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(50), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(50) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant51
        for i in range(0, num):
                print("Mutant " + str(51) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant51_" + str(i) + ".h5"
                loss, acc = mutant51.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(51), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(51) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant52
        for i in range(0, num):
                print("Mutant " + str(52) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant52_" + str(i) + ".h5"
                loss, acc = mutant52.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(52), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(52) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant53
        for i in range(0, num):
                print("Mutant " + str(53) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant53_" + str(i) + ".h5"
                loss, acc = mutant53.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(53), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(53) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant54
        for i in range(0, num):
                print("Mutant " + str(54) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant54_" + str(i) + ".h5"
                loss, acc = mutant54.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(54), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(54) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant55
        for i in range(0, num):
                print("Mutant " + str(55) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant55_" + str(i) + ".h5"
                loss, acc = mutant55.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(55), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(55) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant56
        for i in range(0, num):
                print("Mutant " + str(56) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant56_" + str(i) + ".h5"
                loss, acc = mutant56.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(56), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(56) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant57
        for i in range(0, num):
                print("Mutant " + str(57) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name = "mnist1_mutant57_" + str(i) + ".h5"
                loss, acc = mutant57.train(x_train, y_train, x_test, y_test, model_name)
                K.clear_session()
                writer.writerow([str(57), str(i), str(acc), str(loss)])
    except Exception as e:
        f.write("for " + str(57) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1

f.write("errNum:" + str(errNum))
f.close()
print("errNum:" + str(errNum))