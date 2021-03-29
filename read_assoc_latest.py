import numpy as np
import string
import argparse
import os
from output_paths import *


def read_assoc(out_nb,sim_name,use_fof=False):
    '''
    Get right binary files, format correctly and return
    '''
    

    """
    Open association files
    """

    fof_suffix=''
    if use_fof:fof_suffix='_fof'

    with open(os.path.join(assoc_path,sim_name,('assoc_halos_%s' %out_nb)+fof_suffix,('assoc_halos_%s' %out_nb)+fof_suffix),'rb') as File:
        fof_nb = np.fromfile(File,dtype=np.int32,count=1)
        fof_tmp = np.fromfile(File,dtype=np.float32)

    star_tmp = np.fromfile(os.path.join(assoc_path,sim_name,('assoc_halos_%s' %out_nb)+fof_suffix,('assoc_stars_%s' %out_nb)+fof_suffix),dtype=np.float32)
    star_nbs = np.fromfile(os.path.join(assoc_path,sim_name,('assoc_halos_%s' %out_nb)+fof_suffix,('assoc_halosST_%s' %out_nb)+fof_suffix),dtype=np.int64)
    star_idxs = np.fromfile(os.path.join(assoc_path,sim_name,('assoc_halos_%s' %out_nb)+fof_suffix,('assoc_starsIDs_%s' %out_nb)+fof_suffix),dtype=np.int64)
    tmp_lone_stars = np.fromfile(os.path.join(assoc_path,sim_name,('assoc_halos_%s' %out_nb)+fof_suffix,('assoc_lone_stars_%s' %out_nb)+fof_suffix),dtype=np.float32)    
    tot_star_nbs = np.fromfile(os.path.join(assoc_path,sim_name,('assoc_halos_%s' %out_nb)+fof_suffix,('assoc_halosST_tot_%s' %out_nb)+fof_suffix),dtype=np.int64)
    idxs =  np.fromfile(os.path.join(assoc_path,sim_name,('assoc_halos_%s' %out_nb)+fof_suffix,('assoc_halosID_%s' %out_nb)+fof_suffix),dtype=np.int64)



    """
    Fit into (objects,propreties) matrixes
    """


    
    fofs = np.reshape(fof_tmp,(int(len(fof_tmp)/5),5))
    stars =np.reshape(star_tmp,(int(len(star_tmp)/6),6))
    
    lone_stars =np.reshape(tmp_lone_stars,(int(len(tmp_lone_stars)/6),6))
    
    return(idxs,star_idxs,tot_star_nbs,star_nbs,fofs,stars,lone_stars)

