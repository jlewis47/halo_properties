import numpy as np
from os import listdir, path



def read_star_file(path):
    """
    masses are in units of solar mass
    ages are in Myr
    x,y,z are in box length units (>0, <1)
    metallicities are in solar units
    """
    dt=np.dtype([('buf1','i4'),('id','i4'),('mass','f8'),('x','f8'),('y','f8'),('age','f8'),('z','f8'),('Z/0.02','f8'),('ll','i4'),('tag','i1'),('buf2','i4')])

    with open(path, 'rb') as src:
        data=np.fromfile(src,dt)
        
    return(data)

def read_all_star_files(tgt_path):

    targets=[path.join(tgt_path,f) for f in listdir(tgt_path) if 'star' in f]

    datas=read_star_file(targets[0])

    if len(targets)>1:
        for target in targets[1:]:
            datas=np.concatenate([datas,read_star_file(target)])


    return(datas[['id','mass','x','y','z','age','Z/0.02']])
