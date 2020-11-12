
import math
import numpy as np
import random

def calcuDis(point1,point2):
    return math.sqrt(math.pow(point1[0]-point2[0],2)+math.pow(point1[1]-point2[1],2))

x=[10.0]
y=[10.0]
sigma=0.2
k=60
deltal=0.01
e=deltal*8
x_u=[0.0*deltal for i in range(0,2*k+1)]
y_u=[0.0*deltal for i in range(0,2*k+1)]

oberdis=[0.0 for i in range(0,2*k+1)]

for i in range(0,2*k+1):
    dist=calcuDis([x_u[i],y_u[i]],[x[0],y[0]])+random.gauss(0,sigma)
    oberdis[i]=dist
#print(oberdis)

def linearTrajeMixing():
    A=0
    for i in range(0,2*k+1):
        A+=math.pow(oberdis[i],2)
    return math.sqrt(A/(2*k+1)-(k+1)*k/3*deltal**2-1)
if __name__=="__main__":
    est_middis=linearTrajeMixing()
    trudis=calcuDis([x_u[k],y_u[k]],[x[0],y[0]])
    print("truedis=",trudis,"\n")
    print("estdis=",est_middis,"\n")
    print("oberdis=",oberdis[k],"\n")
