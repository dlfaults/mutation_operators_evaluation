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

csv_name = "mnist_tr_syntacticmutations_results.csv"    
with open(csv_name, 'w') as f1:
    writer=csv.writer(f1, delimiter=',',lineterminator='\n',)
    writer.writerow(["Mutation Number", "Run Number", "Accuracy1", "Loss1", "Accuracy2", "Loss2"])

    try:
        import mutant1
        for i in range(0, num):
                print("Mutant " + str(1) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant1_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant1_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant1.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(1), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(1) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant2
        for i in range(0, num):
                print("Mutant " + str(2) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant2_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant2_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant2.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(2), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(2) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant3
        for i in range(0, num):
                print("Mutant " + str(3) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant3_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant3_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant3.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(3), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(3) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant4
        for i in range(0, num):
                print("Mutant " + str(4) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant4_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant4_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant4.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(4), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(4) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant5
        for i in range(0, num):
                print("Mutant " + str(5) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant5_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant5_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant5.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(5), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(5) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant6
        for i in range(0, num):
                print("Mutant " + str(6) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant6_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant6_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant6.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(6), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(6) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant7
        for i in range(0, num):
                print("Mutant " + str(7) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant7_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant7_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant7.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(7), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(7) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant8
        for i in range(0, num):
                print("Mutant " + str(8) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant8_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant8_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant8.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(8), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(8) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant9
        for i in range(0, num):
                print("Mutant " + str(9) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant9_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant9_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant9.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(9), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(9) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant10
        for i in range(0, num):
                print("Mutant " + str(10) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant10_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant10_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant10.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(10), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(10) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant11
        for i in range(0, num):
                print("Mutant " + str(11) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant11_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant11_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant11.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(11), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(11) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant12
        for i in range(0, num):
                print("Mutant " + str(12) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant12_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant12_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant12.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(12), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(12) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant13
        for i in range(0, num):
                print("Mutant " + str(13) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant13_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant13_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant13.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(13), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(13) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant14
        for i in range(0, num):
                print("Mutant " + str(14) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant14_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant14_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant14.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(14), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(14) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant15
        for i in range(0, num):
                print("Mutant " + str(15) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant15_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant15_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant15.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(15), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(15) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant16
        for i in range(0, num):
                print("Mutant " + str(16) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant16_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant16_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant16.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(16), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(16) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant17
        for i in range(0, num):
                print("Mutant " + str(17) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant17_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant17_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant17.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(17), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(17) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant18
        for i in range(0, num):
                print("Mutant " + str(18) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant18_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant18_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant18.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(18), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(18) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant19
        for i in range(0, num):
                print("Mutant " + str(19) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant19_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant19_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant19.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(19), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(19) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant20
        for i in range(0, num):
                print("Mutant " + str(20) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant20_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant20_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant20.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(20), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(20) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant21
        for i in range(0, num):
                print("Mutant " + str(21) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant21_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant21_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant21.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(21), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(21) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant22
        for i in range(0, num):
                print("Mutant " + str(22) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant22_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant22_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant22.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(22), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(22) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant23
        for i in range(0, num):
                print("Mutant " + str(23) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant23_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant23_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant23.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(23), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(23) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant24
        for i in range(0, num):
                print("Mutant " + str(24) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant24_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant24_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant24.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(24), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(24) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant25
        for i in range(0, num):
                print("Mutant " + str(25) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant25_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant25_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant25.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(25), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(25) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant26
        for i in range(0, num):
                print("Mutant " + str(26) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant26_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant26_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant26.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(26), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(26) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant27
        for i in range(0, num):
                print("Mutant " + str(27) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant27_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant27_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant27.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(27), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(27) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant28
        for i in range(0, num):
                print("Mutant " + str(28) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant28_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant28_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant28.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(28), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(28) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant29
        for i in range(0, num):
                print("Mutant " + str(29) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant29_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant29_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant29.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(29), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(29) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant30
        for i in range(0, num):
                print("Mutant " + str(30) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant30_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant30_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant30.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(30), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(30) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant31
        for i in range(0, num):
                print("Mutant " + str(31) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant31_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant31_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant31.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(31), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(31) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant32
        for i in range(0, num):
                print("Mutant " + str(32) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant32_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant32_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant32.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(32), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(32) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant33
        for i in range(0, num):
                print("Mutant " + str(33) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant33_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant33_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant33.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(33), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(33) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant34
        for i in range(0, num):
                print("Mutant " + str(34) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant34_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant34_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant34.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(34), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(34) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant35
        for i in range(0, num):
                print("Mutant " + str(35) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant35_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant35_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant35.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(35), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(35) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant36
        for i in range(0, num):
                print("Mutant " + str(36) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant36_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant36_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant36.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(36), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(36) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant37
        for i in range(0, num):
                print("Mutant " + str(37) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant37_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant37_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant37.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(37), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(37) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant38
        for i in range(0, num):
                print("Mutant " + str(38) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant38_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant38_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant38.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(38), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(38) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant39
        for i in range(0, num):
                print("Mutant " + str(39) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant39_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant39_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant39.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(39), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(39) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant40
        for i in range(0, num):
                print("Mutant " + str(40) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant40_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant40_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant40.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(40), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(40) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant41
        for i in range(0, num):
                print("Mutant " + str(41) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant41_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant41_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant41.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(41), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(41) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant42
        for i in range(0, num):
                print("Mutant " + str(42) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant42_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant42_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant42.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(42), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(42) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant43
        for i in range(0, num):
                print("Mutant " + str(43) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant43_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant43_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant43.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(43), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(43) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant44
        for i in range(0, num):
                print("Mutant " + str(44) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant44_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant44_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant44.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(44), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(44) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant45
        for i in range(0, num):
                print("Mutant " + str(45) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant45_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant45_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant45.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(45), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(45) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant46
        for i in range(0, num):
                print("Mutant " + str(46) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant46_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant46_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant46.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(46), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(46) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant47
        for i in range(0, num):
                print("Mutant " + str(47) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant47_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant47_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant47.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(47), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(47) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant48
        for i in range(0, num):
                print("Mutant " + str(48) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant48_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant48_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant48.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(48), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(48) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant49
        for i in range(0, num):
                print("Mutant " + str(49) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant49_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant49_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant49.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(49), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(49) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant50
        for i in range(0, num):
                print("Mutant " + str(50) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant50_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant50_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant50.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(50), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(50) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant51
        for i in range(0, num):
                print("Mutant " + str(51) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant51_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant51_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant51.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(51), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(51) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant52
        for i in range(0, num):
                print("Mutant " + str(52) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant52_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant52_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant52.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(52), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(52) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant53
        for i in range(0, num):
                print("Mutant " + str(53) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant53_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant53_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant53.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(53), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(53) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant54
        for i in range(0, num):
                print("Mutant " + str(54) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant54_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant54_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant54.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(54), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(54) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant55
        for i in range(0, num):
                print("Mutant " + str(55) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant55_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant55_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant55.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(55), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(55) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant56
        for i in range(0, num):
                print("Mutant " + str(56) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant56_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant56_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant56.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(56), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(56) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant57
        for i in range(0, num):
                print("Mutant " + str(57) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant57_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant57_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant57.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(57), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(57) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant58
        for i in range(0, num):
                print("Mutant " + str(58) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant58_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant58_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant58.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(58), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(58) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant59
        for i in range(0, num):
                print("Mutant " + str(59) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant59_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant59_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant59.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(59), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(59) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant60
        for i in range(0, num):
                print("Mutant " + str(60) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant60_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant60_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant60.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(60), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(60) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant61
        for i in range(0, num):
                print("Mutant " + str(61) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant61_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant61_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant61.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(61), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(61) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant62
        for i in range(0, num):
                print("Mutant " + str(62) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant62_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant62_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant62.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(62), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(62) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant63
        for i in range(0, num):
                print("Mutant " + str(63) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant63_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant63_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant63.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(63), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(63) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant64
        for i in range(0, num):
                print("Mutant " + str(64) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant64_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant64_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant64.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(64), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(64) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant65
        for i in range(0, num):
                print("Mutant " + str(65) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant65_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant65_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant65.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(65), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(65) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant66
        for i in range(0, num):
                print("Mutant " + str(66) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant66_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant66_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant66.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(66), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(66) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant67
        for i in range(0, num):
                print("Mutant " + str(67) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant67_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant67_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant67.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(67), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(67) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant68
        for i in range(0, num):
                print("Mutant " + str(68) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant68_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant68_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant68.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(68), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(68) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant69
        for i in range(0, num):
                print("Mutant " + str(69) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant69_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant69_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant69.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(69), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(69) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant70
        for i in range(0, num):
                print("Mutant " + str(70) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant70_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant70_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant70.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(70), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(70) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant71
        for i in range(0, num):
                print("Mutant " + str(71) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant71_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant71_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant71.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(71), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(71) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant72
        for i in range(0, num):
                print("Mutant " + str(72) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant72_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant72_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant72.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(72), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(72) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant73
        for i in range(0, num):
                print("Mutant " + str(73) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant73_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant73_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant73.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(73), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(73) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant74
        for i in range(0, num):
                print("Mutant " + str(74) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant74_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant74_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant74.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(74), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(74) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant75
        for i in range(0, num):
                print("Mutant " + str(75) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant75_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant75_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant75.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(75), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(75) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant76
        for i in range(0, num):
                print("Mutant " + str(76) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant76_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant76_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant76.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(76), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(76) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant77
        for i in range(0, num):
                print("Mutant " + str(77) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant77_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant77_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant77.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(77), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(77) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant78
        for i in range(0, num):
                print("Mutant " + str(78) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant78_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant78_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant78.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(78), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(78) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant79
        for i in range(0, num):
                print("Mutant " + str(79) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant79_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant79_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant79.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(79), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(79) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant80
        for i in range(0, num):
                print("Mutant " + str(80) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant80_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant80_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant80.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(80), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(80) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant81
        for i in range(0, num):
                print("Mutant " + str(81) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant81_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant81_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant81.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(81), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(81) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant82
        for i in range(0, num):
                print("Mutant " + str(82) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant82_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant82_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant82.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(82), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(82) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant83
        for i in range(0, num):
                print("Mutant " + str(83) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant83_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant83_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant83.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(83), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(83) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant85
        for i in range(0, num):
                print("Mutant " + str(85) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant85_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant85_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant85.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(85), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(85) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant86
        for i in range(0, num):
                print("Mutant " + str(86) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant86_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant86_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant86.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(86), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(86) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant87
        for i in range(0, num):
                print("Mutant " + str(87) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant87_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant87_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant87.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(87), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(87) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant88
        for i in range(0, num):
                print("Mutant " + str(88) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant88_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant88_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant88.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(88), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(88) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant89
        for i in range(0, num):
                print("Mutant " + str(89) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant89_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant89_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant89.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(89), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(89) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant90
        for i in range(0, num):
                print("Mutant " + str(90) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant90_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant90_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant90.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(90), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(90) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant91
        for i in range(0, num):
                print("Mutant " + str(91) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant91_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant91_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant91.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(91), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(91) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant92
        for i in range(0, num):
                print("Mutant " + str(92) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant92_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant92_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant92.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(92), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(92) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant93
        for i in range(0, num):
                print("Mutant " + str(93) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant93_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant93_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant93.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(93), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(93) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant94
        for i in range(0, num):
                print("Mutant " + str(94) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant94_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant94_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant94.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(94), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(94) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant95
        for i in range(0, num):
                print("Mutant " + str(95) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant95_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant95_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant95.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(95), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(95) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant96
        for i in range(0, num):
                print("Mutant " + str(96) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant96_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant96_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant96.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(96), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(96) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant97
        for i in range(0, num):
                print("Mutant " + str(97) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant97_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant97_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant97.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(97), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(97) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant98
        for i in range(0, num):
                print("Mutant " + str(98) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant98_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant98_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant98.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(98), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(98) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant102
        for i in range(0, num):
                print("Mutant " + str(102) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant102_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant102_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant102.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(102), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(102) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant103
        for i in range(0, num):
                print("Mutant " + str(103) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant103_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant103_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant103.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(103), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(103) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant104
        for i in range(0, num):
                print("Mutant " + str(104) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant104_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant104_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant104.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(104), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(104) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant105
        for i in range(0, num):
                print("Mutant " + str(105) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant105_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant105_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant105.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(105), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(105) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant106
        for i in range(0, num):
                print("Mutant " + str(106) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant106_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant106_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant106.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(106), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(106) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant107
        for i in range(0, num):
                print("Mutant " + str(107) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant107_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant107_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant107.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(107), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(107) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant108
        for i in range(0, num):
                print("Mutant " + str(108) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant108_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant108_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant108.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(108), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(108) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant109
        for i in range(0, num):
                print("Mutant " + str(109) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant109_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant109_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant109.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(109), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(109) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant110
        for i in range(0, num):
                print("Mutant " + str(110) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant110_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant110_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant110.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(110), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(110) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant111
        for i in range(0, num):
                print("Mutant " + str(111) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant111_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant111_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant111.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(111), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(111) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1


    try:
        import mutant119
        for i in range(0, num):
                print("Mutant " + str(119) + ", Run" + str(i))
                x_train, y_train, x_test, y_test = get_dataset(i)
                model_name1 = "mnist_tr_mutant119_1_" + str(i) + ".h5"
                model_name2 = "mnist_tr_mutant119_2_" + str(i) + ".h5"
                loss1, acc1, loss2, acc2 = mutant119.train_model(x_train, y_train, x_test, y_test, model_name1, model_name2)
                K.clear_session()
                writer.writerow([str(119), str(i), str(acc1), str(loss1), str(acc2), str(loss2)])
    except Exception as e:
        f.write("for " + str(119) + ", error:" + str(e))
        f.write("\n")
        errNum = errNum + 1

f.write("errNum:" + str(errNum))
f.close()

print("errNum:" + str(errNum))