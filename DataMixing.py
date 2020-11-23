
import math
import numpy as np
import random
import numpy as np
import matplotlib.pyplot as plt
 
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#matplotlib画图中中文显示会有问题，需要这两行设置默认字体
step=5000
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(xmax=step,xmin=1)
plt.ylim(ymax=1,ymin=0)


def calcuDis(point1,point2):
    return math.sqrt(math.pow(point1[0]-point2[0],2)+math.pow(point1[1]-point2[1],2))


sigma=0.2

k=1 #initial value
deltal=0.1
e=deltal*8
x_u=[i*deltal for i in range(0,2*step+1)]
y_u=[0 for i in range(0,2*step+1)]
x=[k*deltal]
y=[5.0]

estError=[0.0 for i in range(0,step)]
oberdis=[0.0 for i in range(0,2*step+1)]

def getOberDis(k):
    for i in range(0,2*k+1):
        dist=calcuDis([x_u[i],y_u[i]],[x[0],y[0]])+random.gauss(0,sigma)
        oberdis[i]=dist

def linearTrajeMixing(k):
    A=0
    for i in range(0,2*k+1):
        A+=math.pow(oberdis[i],2)
    return math.sqrt(A/(2*k+1)-(k+1)*k/3*deltal**2-1*sigma**2)
if __name__=="__main__":
    for k in range(1,step+1):
        x=[k*deltal]
        getOberDis(k) #update oberdis
        est_middis=linearTrajeMixing(k)
        trudis=calcuDis([x_u[k],y_u[k]],[x[0],y[0]])
        estError[k-1]=abs(trudis-est_middis)
    plt.scatter([k for k in range(1,step+1)], estError, s=np.pi, c='red', alpha=0.4, label='error')
    plt.show()
    # print("truedis=",trudis,"\n")
    # print("estdis=",est_middis,"\n")
    # print("oberdis=",oberdis[k],"\n")
    # print("|truedis-estdis|=",abs(trudis-est_middis),"\n")
    # print("|truedis-oberdis|=",abs(trudis-oberdis[k]),"\n")
    # print("error decrease =",abs(trudis-est_middis)/abs(trudis-oberdis[k]),"\n")
