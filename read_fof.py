import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from read_stars import o_stars

#data_pth = '/data/ngillet/titan/snap_Lorenzo/B4_256_CoDa/output_00022/'

def o_fof(data_pth):

       """
       First run add total halo number from each masst file
       Then build data array
       Then perform another run to get data
       """


       #Get dir contents
       files =  os.listdir(data_pth)

       masst = list(filter(lambda name : 'masst' in name,files))
    
       nhalos_tot = 0


       #Get halo number
       for target in masst[:]:
           print('Opening %s ... ' %target)
           with open(os.path.join(data_pth,target), 'rb' ) as ff:
                  
               nhalos_tot+=np.fromfile(ff,np.int32,count=3)[1] #Number between 2 buoys
               print('%s halos ... \n' %nhalos_tot)
           ff.close()


       #idx mass x y z file_nb
       halo_data = np.zeros((nhalos_tot,6))


       #Get data
       line = 0
       loc_line = 0
       for target in masst[:]:
           print('Opening %s ... ' %target)
           with open(os.path.join(data_pth,target), 'rb' ) as ff:
               nhalos=np.fromfile(ff,np.int32,count=3)[1] #Number between 2 buoys
               halo_data[line:line+nhalos,-1]=(float(target[-4:])*np.ones((1,nhalos)))
               for loc_line in range(nhalos):

                     buoy = np.fromfile(ff,np.int32,count=1)[0]
                     idx = np.fromfile(ff,np.int64,count=1)[0]
                     mass=np.fromfile(ff,np.int32,count=1)[0]
                     x,y,z = np.fromfile(ff,np.float32,count=3)
                     buoy = np.fromfile(ff,np.int32,count=1)[0]

                     halo_data[line,:-1]= np.asarray([idx,mass,x,y,z])
                     #print(halo_data[line,:])
                     line+=1


           ff.close()

       return(halo_data)

