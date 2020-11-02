import matplotlib.pyplot as plt
import numpy as np
import math
import os
from mymeans import MyKmeans
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

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

            

def doS1(colors):
    points=loadFiles("s1.txt")
    startCentroidsNotPrecise=[[ 250615.42836644774 , 848685.4077477583 ],[ 424033.15398022893 , 791208.2192331479 ],[ 664310.7256137813 , 856896.4346784169 ],[ 334190.2358911616 , 572247.5010822512 ],[ 666400.0958018991 , 878792.5064935066 ],[ 599540.249782128 , 572247.5010822512 ],[ 823102.8599107376 , 717308.9768572203 ],[ 833549.7108513268 , 517507.32154452696 ],[ 179576.841970441 , 366971.8278157854 ],[ 403139.4520990505 , 410763.97144596477 ],[ 614165.841098953 , 410763.97144596477 ],[ 789672.9369008521 , 295809.59441674396 ],[ 321654.0147624545 , 156222.13659554734 ],[ 513876.0720692964 , 150748.1186417749 ],[ 848175.3021681518 , 148011.10966488873 ]]
    startCentroids=[[ 242257.94761397637 , 837737.3718402134 ],[ 417765.0434158754 , 777523.1743487169 ],[ 666400.0958018991 , 854159.4257015308 ],[ 131521.3276437305 , 558562.4561978201 ],[ 336279.60607927945 , 553088.4382440477 ],[ 607897.7305345995 , 569510.4921053649 ],[ 823102.8599107376 , 730994.0217416512 ],[ 158683.1400892625 , 342338.7470238096 ],[ 413586.3030396397 , 394341.91758464754 ],[ 620433.9516633066 , 394341.91758464754 ],[ 869069.0040493301 , 539403.3933596166 ],[ 319564.64457433665 , 156222.13659554734 ],[ 501339.8509405893 , 167170.17250309218 ],[ 791762.30708897 , 304020.6213474027 ],[ 846085.931980034 , 156222.13659554734 ]]
    km=MyKmeans(15,points,startCentroids)
    cluster,centroids=km.createClusters()
    return cluster,centroids

def createData(clust):
    trainingPercentage=0.8
    trainingInput=[]
    trainingOutput=[]
    validatingInput=[]
    validatingOutput=[]
    for clustNumber,singleClust in enumerate(clust):
        for pInd,point in enumerate(singleClust):
            if pInd<trainingPercentage*len(singleClust):
                trainingInput.append(point)
                trainingOutput.append(clustNumber)
            else:
                validatingInput.append(point)
                validatingOutput.append(clustNumber)
    standard = StandardScaler()
    trainingInput=standard.fit_transform(trainingInput)
    validatingInput=standard.fit_transform(validatingInput)
    return (trainingInput,trainingOutput),(validatingInput,validatingOutput)
def trainAndClasify(trainData,validData):
    clf = MLPClassifier(solver='sgd', random_state=1,activation='logistic',max_iter=200,verbose=True)
    clf.fit(trainData[0], trainData[1])
    predicted=clf.predict(validData[0])

    goodAnswers=0
    for pre,rea in zip(predicted,validData[1]):
        print('predicted value = ',pre,'real value = ',rea)
        if pre==rea:
            goodAnswers+=1
    totalPercentage=goodAnswers*100/len(predicted)
    print('Total good predictions',totalPercentage,'%')
    return predicted
def plotResult(predY,validationXSYS,realY,colors):
    for point,colorIndex in zip(validationXSYS,predY):
        plt.scatter(point[0],point[1],color=colors[colorIndex]) 
    plt.show()


colors=['INDIANRED','SALMON','CRIMSON','PINK','DEEPPINK','YELLOW','DARKKHAKI','LAVENDER','DARKVIOLET','GREENYELLOW','GREEN','AQUA','DEEPSKYBLUE','MIDNIGHTBLUE','GAINSBORO']

clust,centr=doS1(colors)
training,validating=createData(clust)
predicted=trainAndClasify(training,validating)
plotResult(predicted,validating[0],validating[1],colors)










