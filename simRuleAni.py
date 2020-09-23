import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from pylab import *
import math
import random
import os

plt.rcParams['animation.ffmpeg_path'] = u"D:\\ffmpeg\\ffmpeg-20200831-4a11a6f-win64-static\\bin\\ffmpeg.exe"
#存储路径
os.chdir("d:\\ffmpeg") 
# No toolbar
matplotlib.rcParams['toolbar'] = 'None'

def calcuDis(point1,point2):
    return math.sqrt(math.pow(point1[0]-point2[0],2)+math.pow(point1[1]-point2[1],2)) 

def delCandiNodes(candiNodes,point):
    candiNodes.remove(point)

size=22
fig = plt.figure(figsize=(2*size+1,2*size+1), facecolor='white')
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

x=[9]
y=[6]
target=plt.scatter(x, y, marker = 'x',color = 'red', s = 40 ,label = 'target')
x_u=[2]
y_u=[2]
trajectory_x=[]
trajectory_y=[]
sample_nu=0
sample_nq=0
true_nu=0
mu = 0
sigma = 0.2
grid_size=0.05
row_number=int(2*size/grid_size)
grid_number=(row_number)**2
trajectory=[]
candidateNodes=[]
candiNodes_w=[[1.0/(grid_number) for _ in range(row_number)] for _ in range(row_number)]
def calcuCandiMean():
    total_x=0
    total_y=0
    for item in candidateNodes:
        total_x=total_x+item[0]
        total_y=total_y+item[1]
    return [total_x/len(candidateNodes),total_y/len(candidateNodes)]

trajectory_dis=[]
candidateNodes=[]
# 散点图绘制
scat = ax.scatter(trajectory_x, trajectory_y, marker = 'x',color = 'green', s = 4 ,label = 'uav')
i=0
frequency_s=70
consSpeed=5

#添加缓冲队列

#更新权重系数,以及归一化
def update_w(y_ober,curPoint):
    sum=0
    for item in candidateNodes:
        candiNodes_w[int(item[0]/grid_size)][int(item[1]/grid_size)]=candiNodes_w[int(item[0]/grid_size)][int(item[1]/grid_size)]/sqrt(2*math.pi)/sigma*math.exp(-(y_ober-calcuDis([item[0],item[1]],curPoint))/(2*sigma**2))
        sum=sum+candiNodes_w[int(item[0]/grid_size)][int(item[1]/grid_size)]
    for item in candidateNodes:
        candiNodes_w[int(item[0]/grid_size)][int(item[1]/grid_size)]=candiNodes_w[int(item[0]/grid_size)][int(item[1]/grid_size)]/sum

#更有效的源点计算方式
def calcuCandiMeanByw():
    total_x=0
    total_y=0
    for item in candidateNodes:
        total_x=total_x+item[0]*candiNodes_w[int(item[0]/grid_size)][int(item[1]/grid_size)]
        total_y=total_y+item[1]*candiNodes_w[int(item[0]/grid_size)][int(item[1]/grid_size)]
    return [total_x,total_y]
        

def update(frame):
    global trajectory_x,trajectory_y,candidateNodes,i,sample_nu,consSpeed
    i=i+1
    trajectory.append([x_u[0],y_u[0]])
    dist=calcuDis([x_u[0],y_u[0]],[x[0],y[0]])+random.gauss(mu,sigma)
    print(i,dist)
    trajectory_dis.append(dist)
    sample_nu=(sample_nu*(i-1)+dist)/i
    #true_nq=math.pow(sigma,2)/i
    candidateNodes=[]
    row=-size
    print("frame:%d" % frame)
    while row<=size:
        row=row+grid_size
        col=-size
        while col<=size:
            col=col+grid_size
            count=0
            for j in range(0,i):
                sample_nq=math.pow(calcuDis([row,col],trajectory[j])-trajectory_dis[j],2)
                if(sample_nq>9*sigma**2):
                    break
                count=count+1
            if count==i:
                candidateNodes.append([row,col])
            sample_nq=0
    update_w(trajectory_dis[-1],trajectory[-1])
    if i==1:
        y_u[0]=y_u[0]+1
    elif i==2:
        x_u[0]=x_u[0]+1
    else:
        candiMeanNode=calcuCandiMeanByw()
        print("candimeanNode",candiMeanNode)
        theta=math.atan((candiMeanNode[1]-y_u[0])/(candiMeanNode[0]-x_u[0]))
        x_u[0]= x_u[0]+consSpeed*cos(theta)*frequency_s/1000.0
        y_u[0]= y_u[0]+consSpeed*sin(theta)*frequency_s/1000.0
    scat.set_offsets(candidateNodes+trajectory)  #设置偏置
    print("candiLength:%d"% len(candidateNodes))
    return scat,
animate = FuncAnimation(fig, update, frames = 20,interval=frequency_s)#interval是每隔多少毫秒更新一次，可以查看help
FFwriter = animation.FFMpegWriter(fps=20)  #frame per second帧每秒
animate.save('autoSeeking.mp4', writer=FFwriter,dpi=180)#设置分辨率
plt.show()