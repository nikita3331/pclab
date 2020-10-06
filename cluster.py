import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
import math
import os

def distEuclid(p1,p2):
    dist=math.sqrt(  (p1[0]-p2[0])**2+ (p1[1]-p2[1])**2 )
    return dist
def makeStartingClusters(clustNum,points,startingCentroids):
    clusters=[[] for i in range(0,clustNum)]
    for point in points:
        distances=[]
        for idx,centroid in enumerate(startingCentroids):
            distance=distEuclid(centroid,point)
            distances.append([distance,idx])
        sortedDist=sorted(distances, key=lambda x: x[0])
        clusters[sortedDist[0][1]].append(point)
    return clusters
def makeCentroids(clusters,points):
    newcentroids=[]
    for cluster in clusters:
        # now we need to find our centroid
        meanDistances=[]
        for idx,semiCluster in enumerate(cluster):
            distanceSum=0
            for point in cluster:
                distanceSum+=distEuclid(semiCluster,point)
            meanDistances.append([distanceSum,idx])
        sortedMean=sorted(meanDistances, key=lambda x: x[0])
        newPointIdx=sortedMean[0][1]
        newcentroids.append( cluster[newPointIdx] )
    return newcentroids


def licz(clustNum,points,startingCentroids):
    startClusters=makeStartingClusters(clustNum,points,startingCentroids)
    newCentroids=makeCentroids(startClusters,points)
    clusters=[]
    breakLoop=False
    for i in range(0,5):
        os.system('cls')
        print('percentage of finish ',(i+1)*100/1000,'%')
        clusters=[[] for i in range(0,clustNum)]
        for point in points:
            distances=[]
            for idx,centroid in enumerate(newCentroids):
                distance=distEuclid(centroid,point)
                distances.append([distance,idx])
            sortedDist=sorted(distances, key=lambda x: x[0])
            clusters[sortedDist[0][1]].append(point)
        newCentroids=makeCentroids(clusters,points)
        # porownac shape
        startClusters=clusters
    return clusters,newCentroids

def plotPoints(startCentroids,colors,cluster):
    for row,centroid,color in zip(cluster,startCentroids,colors):
        x=[]
        y=[]
        for p in row:
            x.append(p[0])
            y.append(p[1])

        plt.scatter(x,y,s=10,color=color)
        plt.scatter(centroid[0],centroid[1],s=50,color=color)
    plt.show()

X, y_true = make_blobs(n_samples=300, centers=4,cluster_std=0.60, random_state=0)
startCentroids=[[-3,0],[3,0],[0,6]]
cluster,centroids=licz(3,X,startCentroids)
colors=['red','green','blue']
plotPoints(centroids,colors,cluster)


# plt.scatter(X[:, 0], X[:, 1], s=25)
# plt.show()