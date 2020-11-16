import matplotlib.pyplot as plt
import numpy as np
import math
import os


class MyKmeans():
    def __init__(self,clustNum,points,startingCentroids):
        self.points=points
        self.clustNum=clustNum
        self.startingCentroids=startingCentroids
    def distEuclid(self,p1,p2): #someday we can put another distance here for better clustering
        differences=p1-p2
        squares=np.square(differences)
        sumed=sum(squares)
        dist=math.sqrt( sumed )
        return dist
    def makeStartingClusters(self):
        clusters=[[] for i in range(0,self.clustNum)]
        for point in self.points:
            distances=[]
            for idx,centroid in enumerate(self.startingCentroids):
                distance=self.distEuclid(centroid,point)
                distances.append([distance,idx])
            sortedDist=sorted(distances, key=lambda x: x[0])
            clusters[sortedDist[0][1]].append(point)
        filtered=[]
        for z in clusters:
            if z!=[]:
                filtered.append(z)
        return filtered
    def makeCentroids(self,clusters):
        newcentroids=[]
        for cluster in clusters:
            # now we need to find our centroid
            meanDistances=[]
            for idx,clu in enumerate(cluster):
                distanceSum=0
                for point in cluster:
                    distanceSum+=self.distEuclid(clu,point)
                    
                meanDistances.append([distanceSum,idx])
            sortedMean=sorted(meanDistances, key=lambda x: x[0])
            if len(sortedMean)!=0:
                newPointIdx=sortedMean[0][1]
                newcentroids.append( cluster[newPointIdx] )
        return newcentroids
    def createClusters(self):
        startClusters=self.makeStartingClusters()
        newCentroids=self.makeCentroids(startClusters)
        clusters=[]
        breakLoop=False
        iterator=0
        while(not breakLoop):
            os.system('cls')
            iterator+=1
            print('Liczba iteracji petli, przeskokow punktow',iterator)
            clusters=[[] for i in range(0,self.clustNum)]
            for point in self.points:
                distances=[]
                for idx,centroid in enumerate(newCentroids):
                    distance=self.distEuclid(centroid,point)
                    distances.append([distance,idx])
                sortedDist=sorted(distances, key=lambda x: x[0])
                clusters[sortedDist[0][1]].append(point)
            areEqual=[]
            for row,startRow in zip(clusters,startClusters):
                #all this routine was made because we can't put equal sign between arrays
                if np.shape(row)==np.shape(startRow):
                    anded=np.logical_and(row,startRow)
                    andedXs=[point[0] for point in anded]
                    andedYs=[point[1] for point in anded]
                    if all(andedXs) and all(andedYs):
                        areEqual.append(True)
                    else:
                        areEqual.append(False)
                else:
                    areEqual.append(False)
            if all(areEqual):#there were no more jumping
                breakLoop=True
            else:
                newCentroids=self.makeCentroids(clusters)
                startClusters=clusters
        return clusters,newCentroids
    def plotPoints(self,startCentroids,colors,cluster):
        for row,centroid,color in zip(cluster,startCentroids,colors):
            x=[]
            y=[]
            for p in row:
                x.append(p[0])
                y.append(p[1])

            plt.scatter(x,y,s=10,color=color)
            plt.scatter(centroid[0],centroid[1],s=100,color='pink')#change color=color for better appereance
        plt.xscale('linear')
        plt.yscale('linear')
        plt.show()
    def plotPoints3D(self,startCentroids,colors,cluster):
        for dim in range(0,3):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for row,centroid,color in zip(cluster,startCentroids,colors):
                x=[]
                y=[]
                z=[]
                for p in row:
                    x.append(p[0+dim*3])
                    y.append(p[1+dim*3])
                    z.append(p[2+dim*3])
                ax.scatter(x, y, z,s=10,color=color)
                ax.scatter(centroid[0],centroid[1],centroid[2],s=100,color=color)
            currTitle='Wymiary '+str(1+dim*3)+str(2+dim*3)+str(3+dim*3)
            ax.set_title(currTitle)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

        
        
        
