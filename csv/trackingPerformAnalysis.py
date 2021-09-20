import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

if __name__=="__main__":
    csv_path1="NoUsingDataFusion.csv"
    csv_path2="usingDataFusion.csv"
    matplotlib.rcParams.update({"font.size":20})
    fig=plt.figure(figsize=(8,7))
    data1=pd.read_csv(filepath_or_buffer=csv_path1)
    data2=pd.read_csv(filepath_or_buffer=csv_path2)
    sns.lineplot(x=data1["time step"],y="distance",data=data1,label="No using data fusion")
    sns.lineplot(x=data2["time step"],y="distance",data=data2,label="Using data fusion")
    plt.show()
    fig.savefig("simuCompare.pdf",dpi=600,format='pdf')