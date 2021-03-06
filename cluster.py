import matplotlib.pyplot as plt
import numpy as np
import math
import os
from mymeans import MyKmeans
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

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

    
def plotNormalScatter(points,threeD): #helper function for reading initial cluster points
    if not threeD:
        xs=[]
        ys=[]
        for i in points:
            xs.append(i[0])
            ys.append(i[1])
        fig, ax = plt.subplots()
        onclick=lambda event: print('[',event.xdata, ',',event.ydata,']')
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        ax.scatter(xs,ys)
        plt.show()
    else:
        xs=[]
        ys=[]
        zs=[]
        for i in points:
            xs.append(i[0])
            ys.append(i[1])
            zs.append(i[2])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs, ys, zs)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()
        
def transformPoints(points):
    firstDim=[]
    secondDim=[]
    thirdDim=[]
    for idx,row in enumerate(points):
        firstDim.append([row[0],row[1],row[2]])
        secondDim.append([row[3],row[4],row[5]])
        thirdDim.append([row[6],row[7],row[8]])
    return firstDim,secondDim,thirdDim


def doS1():
    points=loadFiles("s1.txt")
    #plotNormalScatter(points,False) 
    colors=['INDIANRED','SALMON','CRIMSON','PINK','DEEPPINK','YELLOW','DARKKHAKI','LAVENDER','DARKVIOLET','GREENYELLOW','GREEN','AQUA','DEEPSKYBLUE','MIDNIGHTBLUE','GAINSBORO']
    startCentroidsNotPrecise=[[ 250615.42836644774 , 848685.4077477583 ],[ 424033.15398022893 , 791208.2192331479 ],[ 664310.7256137813 , 856896.4346784169 ],[ 334190.2358911616 , 572247.5010822512 ],[ 666400.0958018991 , 878792.5064935066 ],[ 599540.249782128 , 572247.5010822512 ],[ 823102.8599107376 , 717308.9768572203 ],[ 833549.7108513268 , 517507.32154452696 ],[ 179576.841970441 , 366971.8278157854 ],[ 403139.4520990505 , 410763.97144596477 ],[ 614165.841098953 , 410763.97144596477 ],[ 789672.9369008521 , 295809.59441674396 ],[ 321654.0147624545 , 156222.13659554734 ],[ 513876.0720692964 , 150748.1186417749 ],[ 848175.3021681518 , 148011.10966488873 ]]
    startCentroids=[[ 242257.94761397637 , 837737.3718402134 ],[ 417765.0434158754 , 777523.1743487169 ],[ 666400.0958018991 , 854159.4257015308 ],[ 131521.3276437305 , 558562.4561978201 ],[ 336279.60607927945 , 553088.4382440477 ],[ 607897.7305345995 , 569510.4921053649 ],[ 823102.8599107376 , 730994.0217416512 ],[ 158683.1400892625 , 342338.7470238096 ],[ 413586.3030396397 , 394341.91758464754 ],[ 620433.9516633066 , 394341.91758464754 ],[ 869069.0040493301 , 539403.3933596166 ],[ 319564.64457433665 , 156222.13659554734 ],[ 501339.8509405893 , 167170.17250309218 ],[ 791762.30708897 , 304020.6213474027 ],[ 846085.931980034 , 156222.13659554734 ]]














    km=MyKmeans(15,points,startCentroids)
    cluster,centroids=km.createClusters()
    km.plotPoints(centroids,colors,cluster)
def doBreast():
    points=loadFiles("breast.txt")
    first,second,third=transformPoints(points)
    startCentroids=[[5,3,3,3,2,1,6,3,2],[1,1,1,1,1,1,1,1,1]]
    colors=['red','green']
    km=MyKmeans(2,points,startCentroids)
    cluster,centroids=km.createClusters()
    km.plotPoints3D(centroids,colors,cluster)
    

doS1()
doBreast()

# print(secondDim)
# print(thirdDim)

# 










