import numpy as np
import os


def read_halo_file(f):
    with open(f,'r') as src:
        data=np.genfromtxt(f)

    return(data)

def read_phew(path):

    halo_files=[f for f in os.listdir(path)if 'halo_' in f]
    haloes=[]

    for halo_f in halo_files :

        tmp=read_halo_file(os.path.join(path,halo_f))
        tmp=tmp[~np.isnan(tmp)]
        if len(tmp)>0 : haloes.append(np.reshape(tmp,(int(len(tmp)*1./7),7)))

    return(np.vstack(haloes)) 
