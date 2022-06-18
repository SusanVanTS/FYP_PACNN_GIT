# -*- coding: utf-8 -*-
"""
Created on Sun May 26 01:54:28 2019

@author: Asus
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 17:04:33 2019
@author: susanvan
"""

#coding=utf-8
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
    with h5py.File("PACNN-PerspectiveAware-CrowdCounting-master/perspective-ShanghaiTech/B/test_pmap/IMG_"+strimg+".mat") as f:
        data = [np.array(element) for element in f['pmap']]   
    newdata = np.flip(data,axis=1)
    finaldata = np.rot90(newdata,1,(0,1))
    perspectivedata = pd.DataFrame(finaldata)
    perspectivedatapath1 = 'P_B.csv'
    perspectivedata.to_csv(perspectivedatapath1, index=False)
    pdata = pd.read_csv('P_B.csv')
    distance = np.zeros((len(labels), len(labels)))
    distance_sort = list()
    density = np.zeros(len(labels))
    # compute distance with perspective value
    for index_i in range(len(labels)+1):
        for index_j in range(index_i+1, len(labels)):
            YOfI = fitFeature[index_i, 0]
            XOfI = fitFeature[index_i, 1]
            pValueOfI = pdata.iat[int(XOfI), int(YOfI)]
            YOfJ = fitFeature[index_j, 0]
            XOfJ = fitFeature[index_j, 1]
            pValueOfJ = pdata.iat[int(XOfJ), int(YOfJ)]
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
    locationofMax = GTdata[indexofMax]   
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

#####-----MAIN_START-----#####
try:
    strimg = input("Please insert number 1 to 182 to select image: ")
    print("Please wait a while...")
    while strimg != None:
        # get Ground-Truth value
        GTmatload = loadmat('ShanghaiTech/ShanghaiTech/part_B/test_data/ground-truth/GT_IMG_'+strimg+'.mat')
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
        # density peak clustering
        algo = fit(fitFeature, GTdata, 0, distanceMethod = '2')
        # plot second graph
        img = plt.imread('ShanghaiTech/ShanghaiTech/part_B/test_data/images/IMG_'+strimg+'.jpg')
        plt.title("image with most dense point")
        plt.imshow(img)
        plt.show()
        strimg = input("Please insert number 1 to 182 to select image: ")
except:
    print("exit...")