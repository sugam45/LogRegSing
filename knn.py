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
import scipy
'''
import matplotlib.pyplot as plt
from pylab import reParams
import urllib
import sklearn
from sklearn.neighbors import KNeighborsca
'''
import math
import operator

def euc(ins1, ins2 , leng) :
        dis = 0
        for x in range(leng):
                dis += pow((ins1[x] - ins2[x]),2)
        return math.sqrt(dis)  

def getN

def get_data_parts(dataline):
        parts = dataline.split(" ")
        Y = np.array([int(parts[0]),int(parts[1]),int(parts[2]),int(parts[3]),int(parts[4]),int(parts[5]),int(parts[6]),int(parts[7]),int(parts[8])])
        Y = Y.reshape(9,1)
        y_train = np.zeros([1,21])
        #print(y_train)
        y_train[0][int(parts[9])] = (0.1 * 10)
        #y_train[0][int(parts[10])] = 0.95/4.5
        #y_train[0][int(parts[11])] = 0.9/4.5
        #y_train[0][int(parts[12])] = 0.8/4.5
        #y_train[0][int(parts[13])] = 0.85/4.5
        #print(y_train[0][int(parts[9])])
        return Y , y_train
        
if __name__ == '__main__':
        '''
        num_classes = 21
        f1  = open('datasettest.txt','r')
        n = f1.readline().strip()
        k = 0
        while k != 20000:
                #print("K is",k)
                Y , y_train = get_data_parts(n)
                for i in range(19999) :
                        #print(i)
                        n = f1.readline().strip()
                        z , z_train = get_data_parts(n)
                        #print(np.shape(Y))
                        Y = np.append(Y,z,axis = 1)
                        y_train = np.append(y_train,z_train,axis = 0)
                        k+=1
                n = f1.readline().strip()
                k+=1
        #print(Y)
        #print(y_train)
        '''
        print(euc([1,2,3],[0,2,3],3))
        
