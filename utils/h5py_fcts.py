import h5py
import os
import numpy as np


def gather_h5py_files(path, keys):

    files = [os.path.join(path,f) for f in os.listdir(path) if "halo_stats" in f]

    data_len = 0

    types=[]
    with h5py.File(files[0], "r") as src:
        
        if keys==None:
            keys=src.keys()
        
        for k in keys:
            types.append(src[k].dtype.name)
        data_len += src[k].len()
    
    for f in files[1:]:
        with h5py.File(f, "r") as src:
            data_len += src[keys[0]].len()

    dtype = [(k,typ) for typ,k in zip(types, keys)]

    datas = np.empty(data_len, dtype=dtype)

    tot_l=0
    for f in files:
        with h5py.File(f, "r") as src:
            for ik,k in enumerate(keys):
                loc_data = src[k][()]
                loc_l = len(loc_data)
                datas[k][tot_l:tot_l+loc_l] = loc_data
            tot_l+=loc_l
            


    return(datas[k] for k in keys)
