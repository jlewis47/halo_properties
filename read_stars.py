import numpy as np
import os

def read_stars(path,out_nb):
    cols = ['ID','msol','x','y','z','Age','Z','lvl','tag']
    #datas = np.genfromtxt(os.path.join(path,'averages.txt'),skip_header=1,usecols=[0,2,3,4,5,6,7,8])
    datas = np.genfromtxt(os.path.join(path,'stars_%05i'%out_nb),skip_header=3).T
    datas[6,np.isnan(datas[6])]=0

    return({key:col for col,key in zip(datas,cols)})    
