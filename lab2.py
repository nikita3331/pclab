import matplotlib.pyplot as plt
import numpy as np
import math
import os
from mymeans import MyKmeans
from mpl_toolkits.mplot3d import Axes3D  
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom #https://github.com/JustGlowing/minisom/blob/master/examples/Clustering.ipynb

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
    allData=[]
    for clustNumber,singleClust in enumerate(clust):
        for pInd,point in enumerate(singleClust):
            if pInd<trainingPercentage*len(singleClust):
                trainingInput.append(point)
                trainingOutput.append(clustNumber)
            else:
                validatingInput.append(point)
                validatingOutput.append(clustNumber)
        allData.append(point)
    standard = StandardScaler()
    trainingInput=standard.fit_transform(trainingInput)
    validatingInput=standard.fit_transform(validatingInput)

    fullScaled=standard.fit_transform(allData)
    return (trainingInput,trainingOutput),(validatingInput,validatingOutput),fullScaled
def trainAndClasify(trainData,validData):
    clf = MLPClassifier(solver='sgd', random_state=1,activation='logistic',max_iter=200,verbose=False)
    clf.fit(trainData[0], trainData[1])
    predicted=clf.predict(validData[0])

    goodAnswers=0
    for pre,rea in zip(predicted,validData[1]):
        if pre==rea:
            goodAnswers+=1
    totalPercentage=goodAnswers*100/len(predicted)
    return predicted,totalPercentage
def plotResultNN(predY,validationXSYS,realY,colors,result):
    for point,colorIndex in zip(validationXSYS,predY):
        plt.scatter(point[0],point[1],color=colors[colorIndex])
    plt.title('NN clusters accuracy '+str(result)+'%')
    plt.show()
def testSom(scaledInp,clust_ind,centroids,centrKnn,outPuts,scaledTestInput,scaledTestOutput):
    translatedIndexes=np.zeros(np.shape(centrKnn)[0])
    for  idx,predCentr in enumerate(centroids[0]):#convert cluster indexes from KNN ,to cluster indexes of SOM
        distance=100000
        for knnInd,cen in enumerate(centrKnn):
            currDist=math.sqrt( (cen[0]-predCentr[0])**2+(cen[1]-predCentr[1])**2)
            if currDist<distance:
                distance=currDist
                translatedIndexes[idx]= knnInd# 0-th index of start centroid is 8th index of these centroids

    goodAns=0
    total=0
    for ind in np.unique(clust_ind): #test all points in new clusters, are they correct.
        currPoints=scaledInp[clust_ind == ind]
        for p in currPoints: #better would be list.lindex(), but it returned array ,so I just stick to iterating for conveniency.
            for scalInd,scaledPoint in enumerate(scaledInp):
                if scaledPoint[0]==p[0] and scaledPoint[1]==p[1]:
                    if scaledTestOutput[scalInd]==translatedIndexes[ind]:
                        goodAns+=1
            total+=1
    return goodAns*100/total 

def trainAndCreateSom(scaledInputs,outPuts,scaledTestInput,scaledTestOutput,colors,knnCentr):
    features=2
    som_shape=(1,15) #y shape same as clusters
    som =   MiniSom(som_shape[0],som_shape[1],features, sigma=.5, learning_rate=.5, random_seed=10)
    som.train_batch(scaledInputs, 38000, verbose=True)
    winner_coordinates = np.array([som.winner(x) for x in scaledTestInput]).T
    cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
    for ind,colo in zip(np.unique(cluster_index),colors): #plot all points from clusters
        plt.scatter(scaledTestInput[cluster_index == ind, 0],scaledTestInput[cluster_index == ind, 1], label='cluster='+str(ind),color=colo)
   
    for centroid in som.get_weights():# plotting centroids
        plt.scatter(centroid[:, 0], centroid[:, 1], marker='x',s=80, linewidths=35, color='k', label='centroid')
    result=testSom(scaledTestInput,cluster_index,som.get_weights(),knnCentr,outPuts,scaledTestInput,scaledTestOutput)
    plt.title('SOM clusters accuracy '+str(result)+'%')
    plt.legend()
    plt.show()
def plotNormalKnn(clusters,centroids,colors):
    for row,centroid,color in zip(clusters,centroids,colors):
        xs=[]
        ys=[]
        for element in row:
            xs.append(element[0])
            ys.append(element[1])
        plt.scatter(xs,ys,color=color,s=50)
        plt.scatter(centroid[0],centroid[1],color=color,s=100)
    plt.title('KNN clusters')
    plt.show()
colors=['INDIANRED','SALMON','CRIMSON','PINK','DEEPPINK','YELLOW','DARKKHAKI','LAVENDER','DARKVIOLET','GREENYELLOW','GREEN','AQUA','DEEPSKYBLUE','MIDNIGHTBLUE','GAINSBORO']

clust,centr=doS1(colors)
standard = StandardScaler()

plotNormalKnn(clust,centr,colors)


ne=standard.fit_transform(centr)

training,validating,fullScaled=createData(clust)
np.save('trainingInput.npy', training[0])
np.save('trainingOutput.npy', training[1])
np.save('validatingInput.npy', validating[0])
np.save('validatingOutput.npy', validating[1])


somPredicted=trainAndCreateSom(training[0],training[1],validating[0],validating[1],colors,ne)

predicted,accuracyResult=trainAndClasify(training,validating)
plotResultNN(predicted,validating[0],validating[1],colors,accuracyResult)










