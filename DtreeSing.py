import os
import os.path
import glob
from keras.models import model_from_json
from keras.layers import Dense
from keras.optimizers import SGD
import cv2
import numpy as np
import pickle as pkl
import sys

from sklearn.preprocessing import MinMaxScaler

from sklearn.tree import DecisionTreeClassifier

def get_data_parts(dataline):
        parts = dataline.split(" ")
        Y = np.array([int(parts[0]),int(parts[1]),int(parts[2]),int(parts[3]),int(parts[4]),int(parts[5]),int(parts[6]),int(parts[7]),int(parts[8])])
        Y = Y.reshape(9,1)
        y_train = np.zeros([1])
        #print(y_train)
        if int(parts[9]) == 1:
                y_train[0] = float(0.1 * 10)
        else :
                y_train[0] = float(0)
        #y_train[0][int(parts[10])] = 0.95/4.5
        #y_train[0][int(parts[11])] = 0.9/4.5
        #y_train[0][int(parts[12])] = 0.8/4.5
        #y_train[0][int(parts[13])] = 0.85/4.5
        #print(y_train[0][int(parts[9])])
        return Y , y_train
        
if __name__ == '__main__':
        num_classes = 21
        f1  = open('datasettest.txt','r')
        n = f1.readline().strip()
        k = 0
        while k != 20000:
                Y , y_train = get_data_parts(n)
                for i in range(19999) :
                        n = f1.readline().strip()
                        z , z_train = get_data_parts(n)
                        #print(np.shape(Y))
                        Y = np.append(Y,z,axis = 1)
                        y_train = np.append(y_train,z_train,axis = 0)
                        k+=1
                n = f1.readline().strip()
                k+=1
        Y = Y.T
        scalar = MinMaxScaler()
        Y = scalar.fit_transform(Y)
        
        dectree = DecisionTreeClassifier()
        dectree.fit(Y,y_train)
        #print(dectree.score(Y,y_train))
        
        #print(Y.shape)
        #print(y_train.shape)
        f2  = open('datasetclass.txt','r')
        n = f2.readline().strip()
        k = 0
        while k != 200000:
                Y_test , y_test = get_data_parts(n)
                for i in range(199999) :
                        n = f2.readline().strip()
                        z , z_train = get_data_parts(n)
                        #print(np.shape(Y))
                        Y_test = np.append(Y_test,z,axis = 1)
                        y_test = np.append(y_test,z_train,axis = 0)
                        k+=1
                n = f2.readline().strip()
                k+=1
        Y_test = Y_test.T
        Y_test = scalar.fit_transform(Y_test)
        
        f1 = open('Results.txt','a')
        f1.write('Accuracy for Decision Tree Multi Class Classification' + '\n')
        f1.write('The training accuracy is : '+ str(dectree.score(Y,y_train))+'\n')
        f1.write('The training accuracy is : '+ str(dectree.score(Y_test,y_test))+'\n\n')
