import math
import numpy as np
import random

vtracking=[0.3,0]
vtrackingVertical=[0.0,0.3]
vsource=[3.0,4.0]
k=10
deltaT=0.5 #100ms sample once
sigma=0.2
x=[10.0]
y=[10.0]

x_u=[0.0]
y_u=[0.0]

oberdis=[0.0 for i in range(0,(2*k+1)*2)]
def calcuDis(point1,point2):
    return math.sqrt(math.pow(point1[0]-point2[0],2)+math.pow(point1[1]-point2[1],2))

for i in range(0,2*k+1):
    x_u[0]=x_u[0]+vtracking[0]*deltaT
    y_u[0]=y_u[0]+vtracking[1]*deltaT
    x[0]=x[0]+vsource[0]*deltaT
    y[0]=y[0]+vsource[1]*deltaT
    dist=calcuDis([x_u[0],y_u[0]],[x[0],y[0]])+random.gauss(0,sigma)
    oberdis[i]=dist

for i in range(0,2*k+1):
    x_u[0]=x_u[0]+vtrackingVertical[0]*deltaT
    y_u[0]=y_u[0]+vtrackingVertical[1]*deltaT
    x[0]=x[0]+vsource[0]*deltaT
    y[0]=y[0]+vsource[1]*deltaT
    dist=calcuDis([x_u[0],y_u[0]],[x[0],y[0]])+random.gauss(0,sigma)
    oberdis[2*k+1+i]=dist

def preciseVelocityEstimator(start):
    A=0.0
    for i in range(0,2*k+1):
        A+=math.pow(oberdis[start+i],2)
    return math.sqrt((3*A-(6*k+3)*oberdis[start+k]**2)/(k*(k+1)*(2*k+1)*deltaT**2))
def calcuSourceV(point1,point2,r1,r2):     #should be fixed,some bugs
    L=calcuDis(point1,point2)
    K1=(point2[1]-point1[1])/(point2[0]-point1[0])
    K2=-(1/K1)
    V1E=(r2**2-r1**2+L**2)/(2*L)
    x0=point1[0]+V1E/L*(point2[0]-point1[0])
    y0=point1[1]+V1E/L*(point2[1]-point1[1])
    CE=math.sqrt(r1**2-V1E**2)
    xc=x0-CE/math.sqrt(1+K2**2)
    yc=y0+K2*(xc-x0)
    xd=x0+CE/math.sqrt(1+K2**2)
    yd=y0-K2*(xc-x0)
    return [[xc,yc],[xd,yd]]
def calcuSourceV2(point1,point2,r1,r2):   #effective solver
    d=calcuDis(point1,point2)
    a=(r1**2-r2**2+d**2)/(2*d)
    h=math.sqrt(r1**2-a**2)
    x1=point2[0]
    y1=point2[1]
    x0=point1[0]
    y0=point1[1]
    x2=x0+a*(x1-x0)/d   
    y2=y0+a*(y1-y0)/d   
    x3=x2+h*(y1-y0)/d       
    y3=y2-h*(x1-x0)/d       
    x4=x2-h*(y1-y0)/d
    y4=y2+h*(x1-x0)/d
    return [[x3,y3],[x4,y4]]
    
if __name__=="__main__":
    estVValue=preciseVelocityEstimator(0)
    trueVValue=calcuDis(vtracking,vsource)
    print("estVValue1phase:",estVValue,"\n")
    print("trueVValue1phase:",trueVValue,"\n")
    estVValue2=preciseVelocityEstimator(2*k+1)
    trueVValue2=calcuDis(vtrackingVertical,vsource)
    print("estVValue2phase:",estVValue2,"\n")
    print("trueVValue2phase:",trueVValue2,"\n")
    print("estSourceV:",calcuSourceV2(vtracking,vtrackingVertical,estVValue,estVValue2),"\n")