import numpy as np
import string
import argparse
import os



def read_assoc(out_nb,sim_name):
    '''
    Get right binary files, format correctly and return
    '''

    assoc_out=os.path.join('/gpfswork/rech/xpu/uoj51ok/assoc_outs/',sim_name,'assoc_halos_%s' %out_nb)

    """
    Open association files
    """

    with open(os.path.join(assoc_out,'assoc_halos_%s' %out_nb),'rb') as File:
        fof_nb = np.fromfile(File,dtype=np.int32,count=1)
        fof_tmp = np.fromfile(File,dtype=np.float32)

    star_tmp = np.fromfile(os.path.join(assoc_out,'assoc_stars_%s' %out_nb),dtype=np.float32)
    star_nbs = np.fromfile(os.path.join(assoc_out,'assoc_halosST_%s' %out_nb),dtype=np.int64)
    star_idxs = np.fromfile(os.path.join(assoc_out,'assoc_starsIDs_%s' %out_nb),dtype=np.int64)
    tmp_lone_stars = np.fromfile(os.path.join(assoc_out,'assoc_lone_stars_%s' %out_nb),dtype=np.float32)    
    tot_star_nbs = np.fromfile(os.path.join(assoc_out,'assoc_halosST_tot_%s' %out_nb),dtype=np.int64)
    idxs =  np.fromfile(os.path.join(assoc_out,'assoc_halosID_%s' %out_nb),dtype=np.int64)



    """
    Fit into (objects,propreties) matrixes
    """


    
    fofs = np.reshape(fof_tmp,(int(len(fof_tmp)/5),5))
    stars =np.reshape(star_tmp,(int(len(star_tmp)/6),6))

    #print(len(tmp_lone_stars),float(len(tmp_lone_stars))/6)
    
    lone_stars =np.reshape(tmp_lone_stars,(int(len(tmp_lone_stars)/6),6))
    
    return(idxs,star_idxs,tot_star_nbs,star_nbs,fofs,stars,lone_stars)

