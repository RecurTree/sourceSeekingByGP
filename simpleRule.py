from matplotlib import pyplot as plt
from pylab import *
import math
import numpy as np
import random
def calcuDis(point1,point2):
    return math.sqrt(math.pow(point1[0]-point2[0],2)+math.pow(point1[1]-point2[1],2))

def isInCandiNodes(candiNodes,point):
    for item in candiNodes:
        if item[0]==point[0] and  item[1]==point[1]:
            return True
    return False 

def delCandiNodes(candiNodes,point):
    candiNodes.remove(point)

fig = plt.figure()
size=22
plt.xlim(-size, size)  
plt.xticks(range(-size, size, 2))  
plt.ylim(-size, size)  
plt.yticks(range(-size, size, 2))  
ax = gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))

x=[3]
y=[17]
target=plt.scatter(x, y, marker = 'x',color = 'red', s = 40 ,label = 'target')
x_u=[2]
y_u=[2]
trajectory_x=[]
trajectory_y=[]
sample_nu=0
sample_nq=0
true_nu=0
true_nq=0
mu = 0
sigma = 0.2
threshould=0.5
trajectory=[]
candidateNodes=[]

def calcuCandiMean():
    total_x=0
    total_y=0
    for item in candidateNodes:
        total_x=total_x+item[0]
        total_y=total_y+item[1]
    return [total_x/len(candidateNodes),total_y/len(candidateNodes)]

def sign(x):
    if x <=0:
        return -1
    else:
        return 1

trajectory_dis=[]
step=2

for i in range(1,2):
    trajectory.append([x_u[0],y_u[0]])
    dist=calcuDis([x_u[0],y_u[0]],[x[0],y[0]])+random.gauss(mu,sigma)
    print(i,dist)
    trajectory_dis.append(dist)
    sample_nu=(sample_nu*(i-1)+dist)/i
    true_nq=math.pow(sigma,2)/i
    candidateNodes=[]
    row=-size
    while row<=size:
        row=row+0.02
        col=-size
        # while col<=size:
        #     col=col+0.2
        #     for j in range(0,i):
        #         sample_nq=sample_nq+math.pow(calcuDis([row,col],trajectory[j])-trajectory_dis[j],2)
        #     sample_nq=sample_nq/i
        #     #if sample_nq<=threshould/math.sqrt(i):
        #     if sample_nq<=9*sigma**2:
        #         candidateNodes.append([row,col])
        #     sample_nq=0
        while col<=size:
            col=col+0.2
            count=0
            for j in range(0,i):
                sample_nq=math.pow(calcuDis([row,col],trajectory[j])-trajectory_dis[j],2)
                if(sample_nq>9*sigma**2):
                    break
                count=count+1
            if count==i:
                candidateNodes.append([row,col])
            sample_nq=0
    if i==1:
        y_u[0]=y_u[0]+step*2
    elif i==2:
        x_u[0]=x_u[0]+step*2    
    else:
        candiMeanNode=calcuCandiMean()
        print("candimeanNode",candiMeanNode)
        increX=step*sign(candiMeanNode[0]-x_u[0])
        temp=x_u[0]
        x_u[0]= x_u[0]+increX
        y_u[0]= y_u[0]+sign(candiMeanNode[1]-y_u[0])*abs(increX*(candiMeanNode[1]-y_u[0])/(candiMeanNode[0]-temp))*step
for item in candidateNodes:
    trajectory_x.append(item[0])
    trajectory_y.append(item[1])
candiSet=plt.scatter(trajectory_x, trajectory_y, marker = 'x',color = 'green', s = 4 ,label = 'uav')
uav_tra=plt.scatter([trajectory[i][0] for i in range(0,len(trajectory))],[trajectory[i][1] for i in range(0,len(trajectory)) ],marker = '*',color = 'blue', s = 4 ,label = 'uav')
print(trajectory)  
plt.show()



