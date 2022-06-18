import h5py
import pandas as pd
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import math

def distanceNorm(Norm,D_value):
	# initialization
	# Norm for distance
	if Norm == '1':
		counter = np.absolute(D_value);
		counter = np.sum(counter);
	elif Norm == '2':
		counter = np.power(D_value,2);
		counter = np.sum(counter);
		counter = np.sqrt(counter);
	elif Norm == 'Infinity':
		counter = np.absolute(D_value);
		counter = np.max(counter);
	else:
		raise Exception('We will program this later......');
 
	return counter;     
 
def chi(x):
	if x < 0:
		return 1;
	else:
		return 0;
      
def fit(features,labels,t,distanceMethod = '2'):
    with h5py.File("PACNN-PerspectiveAware-CrowdCounting-master/perspective-ShanghaiTech/B/train_pmap/IMG_"+strimg+".mat") as f:
        data = [np.array(element) for element in f['pmap']]
    
    newdata = np.flip(data,axis=1)
    finaldata = np.rot90(newdata,1,(0,1))
    perspectivedata = pd.DataFrame(finaldata)
    perspectivedatapath1 = 'P_B.csv'
    perspectivedata.to_csv(perspectivedatapath1, index=False)
    pdata = pd.read_csv('P_B.csv')

    distance = np.zeros((len(labels),len(labels)));
    distance_sort = list();
    density = np.zeros(len(labels));
    # compute distance
    for index_i in range(len(labels)+1):
        for index_j in range(index_i+1,len(labels)):
            YOfI = fitFeature[index_i,0]
            XOfI = fitFeature[index_i,1]
            #locationOfI = [XOfI,YOfI]
            pValueOfI = pdata.iat[int(XOfI),int(YOfI)]
            YOfJ = fitFeature[index_j,0]
            XOfJ = fitFeature[index_j,1]
            #locationOfJ = [XOfJ,YOfJ]
            pValueOfJ = pdata.iat[int(XOfJ),int(YOfJ)]
            avgPValue = (pValueOfI + pValueOfJ)/2
            D_value = features[index_i] - features[index_j];            
            distance[index_i,index_j] = distanceNorm(distanceMethod,D_value)/avgPValue;
#            distance[index_i,index_j] = distanceNorm(distanceMethod,D_value);
            distance_sort.append(distance[index_i,index_j]);
    distance += distance.T;
    
	# compute optimal cutoff
    distance_sort = np.array(distance_sort);
    cutoff = int(np.round(distance_sort[len(distance_sort) * t]));
    
	# computer density
    for index_i in range(len(labels)):
        distance_cutoff_i = distance[index_i] - cutoff;
        for index_j in range(1,len(labels)):
            density[index_i] += chi(distance_cutoff_i[index_j]);
        #print(index_i,density[index_i])
            
	# search for the max density
    Max = np.max(density);
    MaxIndexList = list();
    for index_i in range(len(labels)):
        if density[index_i] == Max:
            MaxIndexList.extend([index_i]);
            indexofMax = index_i;
    
    locationofMax = GTdata[indexofMax]   
#    print("Max Density =",Max, "\nIndex of Max Density Point =",indexofMax, "\nLocation for Max Density =", locationofMax)
    plt.plot(locationofMax[0],locationofMax[1],'ro') 
    #get smallest distance of the max density point
    mindistancethreshold = 10000000000
    aaa = distance[indexofMax].tolist()
    for i in range(len(aaa)):
        if aaa[i] >0 and aaa[i] < mindistancethreshold:
            mindistancethreshold = aaa[i]
#    print("min distance for max density point =",mindistancethreshold)
#    print(avgPValue)
    #densityReal = math.sqrt(minthreshold*10.764)/ avgPValue
    densityReal = (mindistancethreshold**2)*math.pi
    
#    print("Density in square feet term: ",densityReal)
#    if (densityReal < 2.5):
#        print("mosh-pit!")
#    elif (densityReal >2.5 and densityReal < 4.5):
#        print("Dense")
#    else:
#        print("Less Dense")
    return (density, densityReal,);

###------get GT MAT-----
countmp = 0
countld = 0
counth = 0 
countm = 0
yessss = [None]*317
yessss[0] =  "Index","Count","Type","Density" 
for i in range(1,317):
    strimg = str(i)
    #strimg = input("Please insert number 1 to 182 to select image: ")
    GTmatload = loadmat('ShanghaiTech/ShanghaiTech/part_B/train_data/ground-truth/GT_IMG_'+strimg+'.mat')#要加载的.mat文件
    featureofGT = GTmatload["image_info"]#"indian_pines_corrected"就是上面加载的文件的数据，它以数组形式展现，就是数组名，python中在console中运行1句，就能看到具体的数组名
    #----------------------convert ndarray to array, export to new csv in array----------------------------------------------------
    a = featureofGT[0,0]
    b = a[0,0]
    GTdata = b[0].tolist()
    GTdatapath = pd.DataFrame(data=GTdata)#构造一个表的数据结构，data为表中的数据
    GTdatapath1 = 'GT_222.csv'#保存为.csv格式的路径;若想保存为.txt格式，只需要将.csv改为.txt即可。
    GTdatapath.to_csv(GTdatapath1, index=False)#保存为.csv
    # Importing the GT dataset
    data = pd.read_csv('GT_222.csv')
    data.head()
    # Getting the values and plotting it
    f1 = data['0'].values
    f2 = data['1'].values
    fitFeature = np.array(list(zip(f1, f2)))
#    plt.title('Head position in the image')
#    plt.gca().invert_yaxis()
#    plt.scatter(f1, f2, c='black')
#    plt.plot(f1, f2, '-o')
#    plt.show()
#    img = plt.imread('ShanghaiTech/ShanghaiTech/part_A/test_data/images/IMG_'+strimg+'.jpg')
#    plt.title("image with most dense point")
#    plt.imshow(img)
    algo = fit(fitFeature,GTdata,0,distanceMethod = '2')
    densityread = algo[1]
    if (densityread < 0.5):
        yes = "mosh-pit!"
        countmp +=1
        yessss[i] = str(i),str(countmp),yes,densityread
    elif (densityread >=0.5 and densityread < 1.0):
        yes = "Heavy"
        counth+=1
        yessss[i] = str(i),str(counth),yes,densityread
    elif (densityread >=1.0 and densityread < 1.5):
        yes = "Medium"
        countm+=1
        yessss[i] = str(i),str(countm),yes,densityread
    else:
        yes = "Light"
        countld+=1
        yessss[i] = str(i),str(countld),yes,densityread
    print(strimg)

print(yessss)
