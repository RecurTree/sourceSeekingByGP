import numpy as np
import math
import matplotlib.pyplot as plt
x = np.arange(1, 10000, 1)
y = []
di=5
deltaL=0.05
def theoryMinFunc():
    return deltaL*math.sqrt(di**2-deltaL**2/12)/math.sqrt(3)
def theoryMinIndex():
    return round(di*math.sqrt(3)/deltaL)
for t in x:
    y_1 = di**2/(2*t+1)+(t+1)*t*deltaL**2/3/(2*t+1)
    y.append(y_1)
plt.plot(x, y, label="para")
plt.xlabel("x")
plt.ylabel("y")
print("minimal vlaue of func:",min(y))
print("theory minimal vlaue of func:",theoryMinFunc())
print("minimal vlaue of index:",y.index(min(y)))
print("theory minimal vlaue of func:",theoryMinIndex())
plt.ylim(0, 10)
plt.legend()
plt.show()