from matplotlib import pyplot as plt
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
plt.xlim(2, size)  
plt.xticks(range(0, size, 2))  
plt.ylim(2, size)  
plt.yticks(range(0, size, 2))  
x=[20]
y=[10]
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
sigma = 1

threshould=3

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
for i in range(1,20):
    trajectory.append([x_u[0],y_u[0]])
    dist=calcuDis([x_u[0],y_u[0]],[x[0],y[0]])+random.gauss(mu,sigma)
    trajectory_dis.append(dist)
    sample_nu=(sample_nu*(i-1)+dist)/i
    print(i,"    ")
    true_nq=math.pow(sigma,2)/i
    candidateNodes=[]
    row=0
    while row<=size:
        row=row+0.02
        col=0
        while col<=size:
            col=col+0.2
            for j in range(0,i):
                sample_nq=sample_nq+math.pow(calcuDis([row,col],trajectory[j])-trajectory_dis[j],2)
            sample_nq=sample_nq/i
            if sample_nq<=threshould:
                candidateNodes.append([row,col])
            sample_nq=0
    if i==1:
        #x_u[0]=x_u[0]+step-random.random()*step*2
        y_u[0]=y_u[0]+step*2
    if i==2:
        x_u[0]=x_u[0]+step*2
        #y_u[0]=y_u[0]+step-random.random()*step*2      
    else:
        candiMeanNode=calcuCandiMean()
        increX=step*sign(candiMeanNode[0]-x_u[0])
        temp=x_u[0]
        x_u[0]= x_u[0]+increX
        y_u[0]= y_u[0]+sign(candiMeanNode[1]-y_u[0])*abs(increX*(candiMeanNode[1]-y_u[0])/(candiMeanNode[0]-temp))
for item in candidateNodes:
    trajectory_x.append(item[0])
    trajectory_y.append(item[1])
uav=plt.scatter(trajectory_x, trajectory_y, marker = 'x',color = 'green', s = 4 ,label = 'uav')
print(trajectory)     
plt.show()



