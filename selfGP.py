from scipy.optimize import minimize
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from mpl_toolkits.mplot3d import Axes3D

class GPR:

    def __init__(self, optimize=True):
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": 0.5, "sigma_f": 1, "w_1": 0.0, "w_2": 0.0 ,"v_x": 0.0 ,"v_y": 0.0}
        self.optimize = optimize

    def fit(self, X, y):
        # store train data
        self.train_X = np.asarray(X)
        self.train_y = np.asarray(y)
        #self.params["l"], self.params["sigma_f"],self.params["w_1"], self.params["w_2"]=0.2,1,0.0,0.0
        # hyper parameters optimization
        def negative_log_likelihood_loss(params):
            self.params["l"],self.params["w_1"], self.params["w_2"], self.params["v_x"] ,self.params["v_y"]= params[0], params[1],params[2],params[3],params[4]
            Kyy = self.kernel(self.train_X, self.train_X) + noise_sigma**2 * np.eye(len(self.train_X))
            MY  = self.mean(self.train_X)
            #MY=np.asarray(MY)
            #print("mean shape",MY.shape)
            #print("w1:",self.params["w_1"])
            return (self.train_y.T-MY.T).dot(np.linalg.inv(Kyy)).dot(self.train_y-MY) +  np.linalg.slogdet(Kyy)[1] 

        if self.optimize:
            res = minimize(negative_log_likelihood_loss, [self.params["l"],self.params["w_1"],self.params["w_2"],self.params["v_x"] ,self.params["v_y"]],
                   bounds=((1e-2, 1e2), (-1e2, 1e2), (-1e2, 1e2) ,(-1, 1), (-1, 1)),
                   method='L-BFGS-B')
            self.params["l"], self.params["w_1"], self.params["w_2"],self.params["v_x"] ,self.params["v_y"]= res.x[0], res.x[1],res.x[2], res.x[3],res.x[4]
        self.is_fit = True

    def predict(self, X):
        if not self.is_fit:
            print("GPR Model not fit yet.")
            return
        print(self.params["l"], self.params["sigma_f"],self.params["w_1"], self.params["w_2"])
        X = np.asarray(X)
        Kff = self.kernel(self.train_X, self.train_X)  # (N, N)
        Kyy = self.kernel(X, X)  # (k, k)
        Kfy = self.kernel(self.train_X, X)  # (N, k)
        My  = self.mean(X)
        MY  = self.mean(self.train_X)
        Kff_inv = np.linalg.inv(Kff + 1e-8 * np.eye(len(self.train_X)))  # (N, N)
        
        #set u = 0,modufy latter
        mu = My + Kfy.T.dot(Kff_inv).dot(self.train_y-MY)
        cov = Kyy - Kfy.T.dot(Kff_inv).dot(Kfy)
        return mu, cov
    #important kernel function
    def kernel(self, x1, x2):
        dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return self.params["sigma_f"] ** 2 * np.exp(-0.5 / self.params["l"] ** 2 * dist_matrix)      
    # def mean(self,X):
    #     return np.dot(np.square(X), np.asarray([self.params["w_1"],self.params["w_2"]]).T)

    def mean(self,X):   #here to do slice operation
        X_pos=X[...,0:2]    # slice operation don't include last one
        #print(X_pos)
        X_time=X[...,2]
        #maybe here need transpose
        #X_delta=np.concatenate((self.params["v_x"]*np.asarray([X_time]),self.params["v_y"]*np.asarray([X_time])),axis = 1)
        X_delta=[[d1, d2] for d1, d2 in zip(X_time.ravel()*self.params["v_x"],X_time.ravel()*self.params["v_y"])]
        return np.linalg.norm(X_pos-np.asarray([self.params["w_1"],self.params["w_2"]])-np.asarray(X_delta), axis=1)

        
def y_2d(x, noise_sigma=0.0):
    x = np.asarray(x)
    y = np.linalg.norm(x-sourceP, axis=1)
    y += np.random.normal(0, noise_sigma, size=y.shape)
    return y

def getBeta(time_index):  #time_index!=0
    return 2.0*np.log(gridNumber*time_index**2*np.pi**2/(6.0*delta))

#-----------
candidateNodes=[]
totalSample=10
WIDTH=5
HEIGHT=5
gridStep=0.1
gridNumber=int(4*WIDTH/gridStep*HEIGHT/gridStep)
delta=0.2   #confidence coefficient
noise_sigma=1e-2
sourceP=[-2,-1]
source_vx=0.1
source_vy=0.2
self_v=2.0
#-----------
train_X = np.random.uniform(-4, 4, (1, 2)).tolist()   #firtst step
train_y = y_2d(train_X, noise_sigma)
trajectory_X=np.concatenate((np.asarray(train_X) ,[[0]]),axis = 1).tolist()
trajectory_Y=train_y 

gpr = GPR(optimize=True)
test_d1 = np.arange(-WIDTH, WIDTH, gridStep)
test_d2 = np.arange(-HEIGHT, HEIGHT, gridStep)
test_d1, test_d2 = np.meshgrid(test_d1, test_d2)
test_X = [[d1, d2] for d1, d2 in zip(test_d1.ravel(), test_d2.ravel())]

def findMin(ls):
    min=100000000
    index=-1
    for i in range(0,len(ls)):
        if ls[i]<min:
            min=ls[i]
            index=i
    return index,min

mu=None
def addNoise(ls,t):
    return np.sqrt((ls[0]-sourceP[0]-source_vx*t)**2+(ls[1]-sourceP[1]-source_vy*t)**2)+np.random.normal(0, noise_sigma)

delt_t=0
def caltime(p1,p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)/self_v
for i in range (0,totalSample):    #just cllect totalSmaple's sample
    gpr.fit(trajectory_X, trajectory_Y)
    mu, cov = gpr.predict(np.concatenate((test_X,np.asarray([[delt_t] for i in range(0,gridNumber)])),axis = 1))
    minIndex,minMu=findMin(mu)
    delt_t=delt_t+caltime(trajectory_X[i][0:2],test_X[minIndex])
    trajectory_X.append(np.concatenate(([test_X[minIndex]],[[delt_t]]),axis = 1)[0].tolist())
    new_y=addNoise(test_X[minIndex],delt_t)
    trajectory_Y = trajectory_Y.tolist()
    trajectory_Y.append(new_y)
    trajectory_Y=np.asarray(trajectory_Y).T

z = mu.reshape(test_d1.shape)
print(trajectory_X)

fig = plt.figure(figsize=(7, 5))
ax = Axes3D(fig)
ax.plot_surface(test_d1, test_d2, z, cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)
ax.scatter(np.asarray(trajectory_X)[:,0], np.asarray(trajectory_X)[:,1], trajectory_Y, c= trajectory_Y, cmap=cm.coolwarm)
ax.contourf(test_d1, test_d2, z, zdir='z', offset=0, cmap=cm.coolwarm, alpha=0.6)
ax.set_title("l=%.2f sigma_f=%.2f w=[%.2f,%.2f] v=[%.2f,%.2f]" % (gpr.params["l"], gpr.params["sigma_f"],gpr.params["w_1"], gpr.params["w_2"],gpr.params["v_x"], gpr.params["v_y"]))
plt.show()