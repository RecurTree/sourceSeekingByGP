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

size=5
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

#源点初始位置
x=[-1]
y=[4]
#源点速度大小
s_vx=[1.5]
s_vy=[-1.5]
target=ax.scatter(x[0], y[0], s=200, marker = 'o',color = 'red', label = 'target')
#追踪无人机初始位置
x_u=[0]
y_u=[-2]
#追踪无人机行走过的路径
trajectory_x=[]
trajectory_y=[]
trajectory=[]
#追踪无人机的所有测距位置
trajectory_sample_x=[]
trajectory_sample_y=[]
trajectory_sample=[]    
#追踪无人机速度
consSpeed=5.0
consSpeed_x=0.0
consSpeed_y=5.0

mu = 0

#地图信息
sigma = 0.2
grid_size=0.05
row_number=int(2*size/grid_size)
grid_number=(row_number)**2

candidateNodes=[]   #候选集合表
candiNodes_w=[[1.0/(grid_number) for _ in range(row_number)] for _ in range(row_number)]

trajectory_dis=[]
candidateNodes=[]

# 散点图绘制
scat = ax.scatter(trajectory_x, trajectory_y, marker = 'o',color = 'green', s = 20 ,label = 'uav')
scat_uav=ax.scatter(x_u[0], y_u[0], marker = 'o',color = 'black', s = 200 ,label = 'uav')
i=0
frequency_s=10    #10ms就更新一下图
sensPeriod=20   #测距周期ms
countPeriod=int(sensPeriod/frequency_s)   #测距计数周期   
     
# 数据融合策略相关
fusionStep=0.5
total_k=int(fusionStep*1000/consSpeed/sensPeriod)   # 
k=int((total_k-1)/2)
trajectory_fusion_dis=[]
trajectory_fusion_pos=[]
count_k=0
deltal=calcuDis([s_vx[0],s_vy[0]],[0,consSpeed])*sensPeriod/1000

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

def linearTrajeMixing(k):
    A=0
    for i in range(0,len(trajectory_dis)):
        A+=math.pow(trajectory_dis[i],2)
    return math.sqrt(A/(2*k+1)-(k+1)*k/3*deltal**2)
#初始化候选集合为全部点
row=-size
while row<=size:
    row=row+grid_size
    col=-size
    while col<=size:
        col=col+grid_size
        candidateNodes.append([row,col])

def calcuphik(dis):
    return math.sqrt(1.0/(2*k+1)+k*(k+1.0)/(3.0*(2*k+1))*deltal**2/dis**2)

def update(frame):
    global trajectory_x,trajectory_y,candidateNodes,i,sample_nu,consSpeed,trajectory_sample_x,trajectory_sample_y,trajectory_sample,count_k,k,consSpeed_x,consSpeed_y,consSpeed,deltal
    trajectory.append([x_u[0],y_u[0]])
    if i%countPeriod==0:    #执行测距
        dist=calcuDis([x_u[0],y_u[0]],[x[0],y[0]])+random.gauss(mu,sigma)
        trajectory_dis.append(dist)
        trajectory_sample.append([x_u[0],y_u[0]])
        if count_k==k:
            trajectory_fusion_pos.append([x_u[0]-s_vx[0]*(i-k)*frequency_s/1000,y_u[0]-s_vy[0]*(i-k)*frequency_s/1000])
        elif count_k==total_k-1:    #开始进行数据融合
            #fusion data
            fusion_dis=linearTrajeMixing(k)
            if fusion_dis<0.1:
                return scat
            trajectory_fusion_dis.append(fusion_dis)
            count_k=0
            #候选集合筛选
            sigma_t=calcuphik(dist)*3*sigma
            print("before",len(candidateNodes),"\n")
            temp=[]
            for item in candidateNodes:
                if abs(calcuDis(trajectory_fusion_pos[-1],item)-trajectory_fusion_dis[-1])>sigma_t:
                    temp.append(item)
                else:
                    continue
            for item in temp:
                candidateNodes.remove(item)
            print("sigma_t",sigma_t,"\n")
            print("after",len(candidateNodes),"\n")
            print("oberserve:",trajectory_dis,"\n")
            print("fusion:",trajectory_fusion_dis[-1],"\n")
            print("truedis:",calcuDis(trajectory_fusion_pos[-1],[x[0],y[0]]),"\n")
            update_w(trajectory_dis[k],trajectory_fusion_pos[-1])
            candiMeanNode=calcuCandiMeanByw()
            candiMeanNode[0]=candiMeanNode[0]+s_vx[0]*(i-k)*frequency_s/1000+s_vx[0]*dist/consSpeed
            candiMeanNode[1]=candiMeanNode[1]+s_vy[0]*(i-k)*frequency_s/1000+s_vy[0]*dist/consSpeed
            theta=math.atan2(candiMeanNode[1]-y_u[0],candiMeanNode[0]-x_u[0])
            consSpeed_x=consSpeed*cos(theta)
            consSpeed_y=consSpeed*sin(theta)
            #重新设置deltal
            deltal=calcuDis([s_vx[0],s_vy[0]],[consSpeed_x,consSpeed_y])*sensPeriod/1000
            trajectory_dis.clear()    #存满2k+1就释放
            trajectory_dis.append(dist)
        count_k=count_k+1
    i=i+1
    #追踪无人机位置改变
    if i<=20*countPeriod and i>=10*countPeriod:
        consSpeed_x=consSpeed_x
        consSpeed_y=0
        deltal=calcuDis([s_vx[0],s_vy[0]],[consSpeed_x,consSpeed_y])*sensPeriod/1000
    
    x_u[0]= x_u[0]+consSpeed_x*frequency_s/1000.0
    y_u[0]= y_u[0]+consSpeed_y*frequency_s/1000.0
    #源点位置改变
    #scat.set_offsets([x_u[0],y_u[0]])  #设置偏置
    x[0]=x[0]+s_vx[0]*frequency_s/1000.0
    y[0]=y[0]+s_vy[0]*frequency_s/1000.0
    scat_uav.set_offsets([x_u[0],y_u[0]])
    target.set_offsets([x[0],y[0]])
    scat.set_offsets(candidateNodes)  #设置偏置
animate = FuncAnimation(fig, update, frames = 200, interval=frequency_s)#interval是每隔多少毫秒更新一次，可以查看help
#FFwriter = animation.FFMpegWriter(fps=40)  #frame per second帧每秒
#animate.save('autoSeeking.mp4', writer=FFwriter,dpi=180)#设置分辨率
plt.show()