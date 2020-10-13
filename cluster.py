import matplotlib.pyplot as plt
import numpy as np
import math
import os
from mymeans import MyKmeans
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

def loadFiles():
    f = open("breast.txt", "r")
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
    for idx,row in enumerate(loadedPoints):
        firstDim.append([row[0],row[1],row[2]])
        secondDim.append([row[3],row[4],row[5]])
        thirdDim.append([row[6],row[7],row[8]])
    return firstDim,secondDim,thirdDim
loadedPoints=loadFiles()
first,second,third=transformPoints(loadedPoints)


# print(secondDim)
# print(thirdDim)

# plotNormalScatter(first,True)





startCentroids=[[5,3,3,3,2,1,6,3,2],[1,1,1,1,1,1,1,1,1]]
# colors=['INDIANRED','SALMON','CRIMSON','PINK','DEEPPINK','YELLOW','DARKKHAKI','LAVENDER','DARKVIOLET','GREENYELLOW','GREEN','AQUA','DEEPSKYBLUE','MIDNIGHTBLUE','GAINSBORO']
# startCentroids=[[ 250615.42836644774 , 848685.4077477583 ],[ 424033.15398022893 , 791208.2192331479 ],[ 664310.7256137813 , 856896.4346784169 ],[ 334190.2358911616 , 572247.5010822512 ],[ 666400.0958018991 , 878792.5064935066 ],[ 599540.249782128 , 572247.5010822512 ],[ 823102.8599107376 , 717308.9768572203 ],[ 833549.7108513268 , 517507.32154452696 ],[ 179576.841970441 , 366971.8278157854 ],[ 403139.4520990505 , 410763.97144596477 ],[ 614165.841098953 , 410763.97144596477 ],[ 789672.9369008521 , 295809.59441674396 ],[ 321654.0147624545 , 156222.13659554734 ],[ 513876.0720692964 , 150748.1186417749 ],[ 848175.3021681518 , 148011.10966488873 ]]
colors=['red','green']
km=MyKmeans(2,loadedPoints,startCentroids)
cluster,centroids=km.createClusters()
#km.plotPoints(centroids,colors,cluster)
km.plotPoints3D(centroids,colors,cluster)

