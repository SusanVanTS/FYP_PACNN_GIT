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
    perspectivematload = loadmat("PACNN-PerspectiveAware-CrowdCounting-master/perspective-ShanghaiTech/A/test_pmap/IMG_"+strimg+".mat")
    featureofPM = perspectivematload['pmap']
    perspectivedata = pd.DataFrame(data=featureofPM.tolist())
    perspectivedata.to_csv('PerspectiveValue.csv', index=False)
    pdata = pd.read_csv('PerspectiveValue.csv')
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
#    plt.plot(locationofMax[0], locationofMax[1], 'ro') 
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
# get Ground-Truth value
countmp = 0
countld = 0
counth = 0 
countm = 0
yessss = [None]*183
yessss[0] =  "Index","Count","Type","Density" 
for i in range( 1, 183):
    strimg = str(i)
    GTmatload = loadmat('ShanghaiTech/ShanghaiTech/part_A/test_data/ground-truth/GT_IMG_'+strimg+'.mat')
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
#    plt.title('Head position in the image')
#    plt.gca().invert_yaxis()
#    plt.scatter(f1, f2, c='black')
#    plt.plot(f1, f2, '-o')
#    plt.show()
    # density peak clustering
    algo = fit(fitFeature, GTdata, 0, distanceMethod = '2')
    # plot second graph
#    img = plt.imread('ShanghaiTech/ShanghaiTech/part_A/test_data/images/IMG_'+strimg+'.jpg')
#    plt.title("image with most dense point")
#    plt.imshow(img)
    densityread = algo[1]
    if (densityread < 0.5):
        yes = "mosh-pit!"
        countmp +=1
        yessss[i] = str(i),str(countmp),yes,densityread
    elif ( 0.5 <= densityread < 1.0):
        yes = "Heavy"
        counth+=1
        yessss[i] = str(i),str(counth),yes,densityread
    elif (1.0 <= densityread < 1.5):
        yes = "Medium"
        countm+=1
        yessss[i] = str(i),str(countm),yes,densityread
    else:
        yes = "Light"
        countld+=1
        yessss[i] = str(i),str(countld),yes,densityread
    print(strimg)

print(yessss)
