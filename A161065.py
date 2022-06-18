#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 17:04:33 2019
@author: susanvan
"""
import math
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import h5py

def distanceNorm(Norm, D_value):
	# initialization
	# Norm for distance
    if Norm == '1':
        counter = np.absolute(D_value)
        counter = np.sum(counter)
    elif Norm == '2':
        counter = np.power(D_value, 2)
        counter = np.sum(counter)
        counter = np.sqrt(counter)
    elif Norm == 'Infinity':
        counter = np.absolute(D_value)
        counter = np.max(counter)
    else:
        raise Exception('We will program this later......')   
    return counter    
 
def chi(x):
    if x < 0:
        return 1
    else:
        return 0
      
def fit(features, labels, t, distanceMethod = '2'):
	# initialization
    # get perspective value
    # perspective value= 8, 8 pixels in one meter, which one pixel equals 1/8
    if test == 'A':
        perspectivematload = loadmat("PACNN-PerspectiveAware-CrowdCounting-master/perspective-ShanghaiTech/A/test_pmap/IMG_"+strimg+".mat")
        finaldata = perspectivematload['pmap']
    elif test == 'B':
        with h5py.File("PACNN-PerspectiveAware-CrowdCounting-master/perspective-ShanghaiTech/A/train_pmap/IMG_"+strimg+".mat") as f:
            data = [np.array(element) for element in f['pmap']]
            newdata = np.flip(data,axis=1)
            finaldata = np.rot90(newdata,1,(0,1))
    elif test == 'C':
        with h5py.File("PACNN-PerspectiveAware-CrowdCounting-master/perspective-ShanghaiTech/B/test_pmap/IMG_"+strimg+".mat") as f:
            data = [np.array(element) for element in f['pmap']]
            newdata = np.flip(data,axis=1)
            finaldata = np.rot90(newdata,1,(0,1))
    else:
        with h5py.File("PACNN-PerspectiveAware-CrowdCounting-master/perspective-ShanghaiTech/B/train_pmap/IMG_"+strimg+".mat") as f:
            data = [np.array(element) for element in f['pmap']]  
            newdata = np.flip(data,axis=1)
            finaldata = np.rot90(newdata,1,(0,1))
    perspectivedata = pd.DataFrame(finaldata)
    perspectivedata.to_csv('P_B.csv', index=False)
    pdata = pd.read_csv('P_B.csv')
    distance = np.zeros((len(labels), len(labels)))
    distance_sort = list()
    density = np.zeros(len(labels))
    # compute distance with perspective value
    for index_i in range(len(labels)+1):
        for index_j in range(index_i+1, len(labels)):
            pValueOfI = pdata.iat[int(features[index_i, 1]), int(features[index_i, 0])]
            pValueOfJ = pdata.iat[int(features[index_j, 1]), int(features[index_j, 0])]
            avgPValue = (pValueOfI + pValueOfJ)/2
            D_value = features[index_i] - features[index_j]            
            distance[index_i, index_j] = distanceNorm(distanceMethod, D_value)/ avgPValue
            distance_sort.append(distance[index_i, index_j])
    distance += distance.T
	# compute optimal cutoff
    distance_sort = np.array(distance_sort)
    cutoff = int(np.round(distance_sort[len(distance_sort) * t]))
	# computer density
    for index_i in range(len(labels)):
        distance_cutoff_i = distance[index_i] - cutoff
        for index_j in range(1, len(labels)):
            density[index_i] += chi(distance_cutoff_i[index_j])         
	# search for the max density
    Max = np.max(density)
    MaxIndexList = list()
    for index_i in range(len(labels)):
        if density[index_i] == Max:
            MaxIndexList.extend([index_i])
            indexofMax = index_i
    locationofMax = labels[indexofMax]   
    print("Max Density =", Max, "\nIndex of Max Density Point =", indexofMax, "\nLocation for Max Density =", locationofMax)
    plt.plot(locationofMax[0], locationofMax[1], 'ro') 
    # get smallest distance of the max density point
    mindistancethreshold = 10000000000
    aaa = distance[indexofMax].tolist()
    for i in range(len(aaa)):
        if aaa[i] > 0 and aaa[i] < mindistancethreshold:
            mindistancethreshold = aaa[i]
    print("min distance for max density point =", mindistancethreshold)
    densityReal = (mindistancethreshold**2)*math.pi
    print("Density: ", densityReal)
    # categorize crowd
    if (densityReal < 0.5):
        print("Mosh-pit Crowd")
    elif (0.5 <= densityReal < 1.0):
        print("Heavy Dense Crowd")
    elif (1.0 <= densityReal < 1.5):
        print("Medium Dense Crowd")
    else:
        print("Light Dense Crowd")
    return (density)

def GTmatload():
    if test == 'A':
        GTmatload = loadmat('ShanghaiTech/ShanghaiTech/part_A/test_data/ground-truth/GT_IMG_'+strimg+'.mat')
    elif test == 'B':
        GTmatload = loadmat('ShanghaiTech/ShanghaiTech/part_A/train_data/ground-truth/GT_IMG_'+strimg+'.mat')
    elif test == 'C':
        GTmatload = loadmat('ShanghaiTech/ShanghaiTech/part_B/test_data/ground-truth/GT_IMG_'+strimg+'.mat')
    else:
        GTmatload = loadmat('ShanghaiTech/ShanghaiTech/part_B/train_data/ground-truth/GT_IMG_'+strimg+'.mat')
    featureofGT = GTmatload["image_info"]
    # convert ndarray to array, export to new csv in array
    a = featureofGT[0, 0]
    b = a[0, 0]
    GTdata = b[0].tolist()
    GTdatapath = pd.DataFrame(data=GTdata)
    GTdatapath.to_csv('GTValue.csv', index=False)
    # Importing the GT dataset
    data = pd.read_csv('GTValue.csv')
    data.head()
    # Getting the values and plotting it
    f1 = data['0'].values
    f2 = data['1'].values
    fitFeature = np.array(list(zip(f1, f2)))
    # plot first graph
    plt.title('Head position in the image')
    plt.gca().invert_yaxis()
    plt.scatter(f1, f2, c='black')
    plt.show()
    print("Please wait a while for calculation")
    return fitFeature, GTdata

def imgShow():
    if test == 'A':
        img = plt.imread('ShanghaiTech/ShanghaiTech/part_A/test_data/images/IMG_'+strimg+'.jpg')
    elif test == 'B':
        img = plt.imread('ShanghaiTech/ShanghaiTech/part_A/train_data/images/IMG_'+strimg+'.jpg')
    elif test == 'C':
        img = plt.imread('ShanghaiTech/ShanghaiTech/part_B/test_data/images/IMG_'+strimg+'.jpg')
    else:
        img = plt.imread('ShanghaiTech/ShanghaiTech/part_B/train_data/images/IMG_'+strimg+'.jpg')
    plt.title("image with most dense point")
    plt.imshow(img)
    plt.show()

#####-----MAIN_START-----#####
try:
    test = input("Please select which set of images is going to test: \nA. Test A\nB. Train A\nC. Test B\nD. Train B\n")
    while test == 'A' or test == 'B' or test == 'C' or test == 'D':
        if test == 'A':
                strimg = input("Please insert number 1 to 182 to select image: ")
        elif test == 'B':
            strimg = input("Please insert number 1 to 300 to select image: ")
        elif test == 'C':
            strimg = input("Please insert number 1 to 316 to select image: ")
        else:
            strimg = input("Please insert number 1 to 400 to select image: ")
        print("Please wait a while...")
        # get Ground-Truth value
        GT = GTmatload()
        # density peak clustering
        algo = fit(GT[0], GT[1], 0, distanceMethod = '2')
        # plot second graph
        imgShow()
        test = input("Please select which set of images is going to test: \nA. Test A\nB. Train A\nC. Test B\nD. Train B\n")
except:
    print("exit...")