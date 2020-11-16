import matplotlib.pyplot as plt
import numpy as np
import math
import os
from mymeans import MyKmeans
from mpl_toolkits.mplot3d import Axes3D  
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import time
import tensorflow as tf
from anfis import ANFIS
def loadFiles(name):
    f = open(name, "r")
    total=[]
    for idx,x in enumerate(f):
        arr=x.split(' ')
        line=[]
        for i in arr:
            if i!='':
                line.append(float(i.replace('\n','')))
        total.append(np.array(line))
    f.close()
    return total
def createData(clust):
    trainingPercentage=0.3
    trainingInput=[]
    trainingOutput=[]
    validatingInput=[]
    validatingOutput=[]
    allData=[]
    for clustNumber,singleClust in enumerate(clust):
        for pInd,point in enumerate(singleClust):
            if pInd<trainingPercentage*len(singleClust):
                trainingInput.append(point)
                trainingOutput.append(clustNumber/len(clust))
            else:
                validatingInput.append(point)
                validatingOutput.append(clustNumber/len(clust))
        allData.append(point)
    standard = StandardScaler()
    trainingInput=standard.fit_transform(trainingInput)
    validatingInput=standard.fit_transform(validatingInput)

    fullScaled=standard.fit_transform(allData)
    return (trainingInput,trainingOutput),(validatingInput,validatingOutput),fullScaled
def doS1():
    points=loadFiles("s1.txt")
    startCentroidsNotPrecise=[[ 250615.42836644774 , 848685.4077477583 ],[ 424033.15398022893 , 791208.2192331479 ],[ 664310.7256137813 , 856896.4346784169 ],[ 334190.2358911616 , 572247.5010822512 ],[ 666400.0958018991 , 878792.5064935066 ],[ 599540.249782128 , 572247.5010822512 ],[ 823102.8599107376 , 717308.9768572203 ],[ 833549.7108513268 , 517507.32154452696 ],[ 179576.841970441 , 366971.8278157854 ],[ 403139.4520990505 , 410763.97144596477 ],[ 614165.841098953 , 410763.97144596477 ],[ 789672.9369008521 , 295809.59441674396 ],[ 321654.0147624545 , 156222.13659554734 ],[ 513876.0720692964 , 150748.1186417749 ],[ 848175.3021681518 , 148011.10966488873 ]]
    startCentroids=[[ 242257.94761397637 , 837737.3718402134 ],[ 417765.0434158754 , 777523.1743487169 ],[ 666400.0958018991 , 854159.4257015308 ],[ 131521.3276437305 , 558562.4561978201 ],[ 336279.60607927945 , 553088.4382440477 ],[ 607897.7305345995 , 569510.4921053649 ],[ 823102.8599107376 , 730994.0217416512 ],[ 158683.1400892625 , 342338.7470238096 ],[ 413586.3030396397 , 394341.91758464754 ],[ 620433.9516633066 , 394341.91758464754 ],[ 869069.0040493301 , 539403.3933596166 ],[ 319564.64457433665 , 156222.13659554734 ],[ 501339.8509405893 , 167170.17250309218 ],[ 791762.30708897 , 304020.6213474027 ],[ 846085.931980034 , 156222.13659554734 ]]
    km=MyKmeans(15,points,startCentroids)
    cluster,centroids=km.createClusters()
    return cluster,centroids


def doanfigs(trainingData,validatingData):
    #https://github.com/tiagoCuervo/TensorANFIS
    m = 4  
    alpha = 0.01  
    fis = ANFIS(n_inputs=2, n_rules=m, learning_rate=alpha)
    num_epochs = 1
    with tf.Session() as sess:
        # Initialize model parameters
        sess.run(fis.init_variables)
        trn_costs = []
        val_costs = []
        time_start = time.time()
        val_pred=[]
        for epoch in range(num_epochs):
            #  Run an update step
            print('Current epoch = ',epoch)
            trn_loss, trn_pred = fis.train(sess, trainingData[0], trainingData[1])
            # Evaluate on validation set
            if epoch == num_epochs - 1:
                val_pred = fis.infer(sess, validatingData[0])
        reScaledPreds=np.array(val_pred)*15
        reScaledPreds=[round(item) for item in reScaledPreds]
        reScaledOuts=np.array(validatingData[1])*15
        diff=reScaledOuts-reScaledPreds
        myNonZeros=len(np.nonzero(diff))
        totalGood=1-(myNonZeros/len(diff))
        print('good percentage',totalGood*100)   


    

clust,centr=doS1()
standard = StandardScaler()
scaledCentr=standard.fit_transform(centr)
training,validating,fullScaled=createData(clust)
doanfigs(training,validating)
