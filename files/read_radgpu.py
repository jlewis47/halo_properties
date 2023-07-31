#Ouvrir un radgpu

import numpy as np
import os
from scipy.io import FortranFile

def o_rad(data_pth,arg_type=3):#,nb):

    """
    Fetch radgpu at data_pth
    arg_type = 1 -> output energy density
               2 -> output flux
    """
    
    
    #data_pth= np.asarray([data_pth+name for name in os.listdir(data_pth) if 'radgpu' in name])[nb]
    
    with open(data_pth, 'rb' ) as ff: 

        # flux
        #bal   = np.fromfile( ff, dtype=np.int32, count=50 )
        #print(bal)
        
        bal   = np.fromfile( ff, dtype=np.int32, count=1 )[0]
        nx   = np.fromfile( ff, dtype=np.int32, count=1 )[0]
        ny   = np.fromfile( ff, dtype=np.int32, count=1 )[0]
        nz   = np.fromfile( ff, dtype=np.int32, count=1 )[0]
        bal   = np.fromfile( ff, dtype=np.int32, count=1 )[0]

        #print(nx,ny,nz)
        
        #densite de photons 
        bal   = np.fromfile( ff, dtype=np.int32, count=1 )[0]

        nrg = np.fromfile( ff, dtype=np.float64, count= (nx+2)*(ny+2)*(nz+2) )
        
        if arg_type != 1:
             bal   = np.fromfile( ff, dtype=np.int32, count=1 )[0]

             #flux
             bal   = np.fromfile( ff, dtype=np.int32, count=1 )[0]
             fl = np.fromfile( ff, dtype=np.float64, count= (nx+2)*(ny+2)*(nz+2) * 3 )
   
             bal   = np.fromfile( ff, dtype=np.int32, count=1 )[0]
  
    ff.close()
    
    if arg_type == 1:
        return(np.reshape(nrg,(nz+2,ny+2,nx+2),order='A'))
    elif arg_type ==2:
        return(np.reshape(fl,(3,nz+2,ny+2,nx+2),order='A'))
    else:
        return(np.reshape(nrg,(nz+2,ny+2,nx+2),order='A'),np.reshape(fl,(3,nz+2,ny+2,nx+2),order='A'))


                                                                          
#data_pth = '/data/ngillet/titan/snap_Lorenzo/B4_256_CoDa/output_00022/'

def o_rad_cube(data_pth,arg_type,ldx,subdims=[64,64,64]):

       """
       Read radgpu files serially then assemble
       arg_type = 1 -> output energy density
                  2 -> output flux
                  3 -> both
       """
    
    
       files =  os.listdir(data_pth)

       rads= np.asarray([name  for name in files if 'radgpu' in name])
       nbs=np.asarray([int(name.split('.')[-1][:].lstrip('out'))  for name in files if 'radgpu' in name])

       sort_args=np.argsort(nbs)
       files=np.asarray(rads)[sort_args]

       xlen=int(ldx/float(subdims[2]))
       ylen=int(ldx/float(subdims[1]))
       zlen=int(ldx/float(subdims[0]))
       
       if arg_type != 2  : nrg_tot = np.zeros((subdims[2]*xlen,subdims[1]*ylen,subdims[0]*zlen))
       if arg_type != 1 : fl_tot = np.zeros((3,subdims[2]*xlen,subdims[1]*ylen,subdims[0]*zlen))

       subcube_nb = 0


       for subz in range(zlen):
           for suby in range(ylen):
               for subx in range(xlen):

                   #print(subx,suby,subz,subcube_nb)


                   if arg_type==1:
                       nrg=o_rad(os.path.join(data_pth,files[subcube_nb]),1)
                   elif arg_type==2:
                       fl=o_rad(os.path.join(data_pth,files[subcube_nb]),2)
                   else:
                       nrg,fl=o_rad(os.path.join(data_pth,files[subcube_nb]),3)
                   
                   
                   if arg_type!=2:
                       #print(np.shape(nrg))
                       nrg_tot[subz*subdims[0]:(subz+1)*subdims[0],
                               suby*subdims[1]:(suby+1)*subdims[1],
                               subx*subdims[2]:(subx+1)*subdims[2]]=nrg[1:-1,1:-1,1:-1]

                   if arg_type!=1:
                       fl_tot[:,subz*subdims[0]:(subz+1)*subdims[0],
                               suby*subdims[1]:(suby+1)*subdims[1],
                               subx*subdims[2]:(subx+1)*subdims[2]]=fl[:,1:-1,1:-1,1:-1]

                   subcube_nb+=1
                       
       
       if arg_type==1:
            return(nrg_tot)
       elif arg_type==2:
            return(fl_tot)
       else:
            return(nrg_tot,fl_tot)


def get_phote_cube(output, cube_nb, rad_path, rad_size, tgt_ldx, ldx):

    tgt_size = [tgt_ldx]*3
    size = [ldx]*3

    size_ratios = np.int32(np.asarray(tgt_size) / np.asarray(rad_size))
    
    size_ratios_full = np.int32(np.asarray(size) / np.asarray(rad_size))

    tot_rad_nb = np.prod(size_ratios_full)

    phote = np.zeros((tgt_ldx,tgt_ldx,tgt_ldx), dtype=np.float64)
    x0,y0,z0 = np.unravel_index(cube_nb, np.int32(np.asarray(size) / np.asarray(tgt_size)))

    for iz in range(size_ratios[0]):
        for iy in range(size_ratios[1]):
            for ix in range(size_ratios[2]):

                rad_nb=np.ravel_multi_index((iz+z0*size_ratios[0],iy+y0*size_ratios[1],ix+x0*size_ratios[2]), size_ratios_full) + 1
                #print(iz,iy,ix,rad_nb)

                grp_nb = np.ceil(rad_nb / (tot_rad_nb / 16))
                grp_str = "group_%06i"%(grp_nb)


                phote[iz * rad_size[0]:(iz+1) * rad_size[0],
            iy * rad_size[1]:(iy+1) * rad_size[1],
            ix * rad_size[2]:(ix+1) * rad_size[2]] = o_rad(os.path.join(rad_path, grp_str,"radgpu_%06i.out%06i"%(output, rad_nb)), arg_type=1)[1:-1,1:-1,1:-1]

    return(phote)

