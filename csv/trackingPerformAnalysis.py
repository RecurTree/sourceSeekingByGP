import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

if __name__=="__main__":
    csv_path1=os.path.dirname(os.path.dirname(__file__))+"\\csv\\NoUsingDataFusion.csv"
    csv_path2=os.path.dirname(os.path.dirname(__file__))+"\\csv\\usingDataFusion.csv"
    plt.figure()
    data1=pd.read_csv(filepath_or_buffer=csv_path1)
    data2=pd.read_csv(filepath_or_buffer=csv_path2)
    sns.lineplot(x="time step",y="distance",data=data1,label="No using datafusion")
    sns.lineplot(x="time step",y="distance",data=data2,label="using datafusion")
    plt.show()