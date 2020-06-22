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
        bal   = np.fromfile( ff, dtype=np.int32, count=1 )[0]
        nx   = np.fromfile( ff, dtype=np.int32, count=1 )[0]
        ny   = np.fromfile( ff, dtype=np.int32, count=1 )[0]
        nz   = np.fromfile( ff, dtype=np.int32, count=1 )[0]
        bal   = np.fromfile( ff, dtype=np.int32, count=1 )[0]

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

def o_rad_cube(data_pth,arg_type,file_dim=64):

       """
       Read radgpu files serially then assemble
       arg_type = 1 -> output energy density
                  2 -> output flux
       """
    
    
       files =  os.listdir(data_pth)

       rads = np.asarray([name  for name in files if 'radgpu' in name])
       
       dim = len(rads)
       subdim = int(round(dim**(1.0/3)))

       if arg_type != 2 : nrg_tot = np.zeros((subdim*file_dim,subdim*file_dim,subdim*file_dim))
       if arg_type != 1 : fl_tot = np.zeros((3,subdim*file_dim,subdim*file_dim,subdim*file_dim))

       subcube_nb = 0
       subz,suby,subx=0,0,0

       #print(dim,subdim)
       
       for sub in range(dim):
           print('Opening %s' %rads[subcube_nb])


           if arg_type not in [1,2] : tmp_nrg,tmp_fl=o_rad(data_pth+rads[subcube_nb])
           if arg_type != 2 :
               tmp_nrg=o_rad(data_pth+rads[subcube_nb],1)
               nrg_tot[subz*file_dim:(subz+1)*file_dim,suby*file_dim:(suby+1)*file_dim,subx*file_dim:(subx+1)*file_dim]=tmp_nrg[1:-1,1:-1,1:-1]
           if arg_type != 1 :
                tmp_fl=o_rad(data_pth+rads[subcube_nb],2)
                fl_tot[:,subz*file_dim:(subz+1)*file_dim,suby*file_dim:(suby+1)*file_dim,subx*file_dim:(subx+1)*file_dim]=tmp_fl[:,1:-1,1:-1,1:-1]
           subcube_nb+=1
            
           if subcube_nb%subdim**2==0:
                subz+=1
                suby=0
                subx=0
           elif subcube_nb%subdim==0:
                subx=0
                suby+=1
           else:
                subx+=1

       # x,y=np.meshgrid(np.arange(0,subdim*file_dim),np.arange(0,subdim*file_dim))
       # #plt.quiver(x,y,fl_tot[0,256,x,y]/10**6,fl_tot[1,256,x,y]/10**6)
       # plt.imshow(np.log10(nrg_tot[256,:,:]))
       # plt.show()
                
                
       if (arg_type==1):
           return(nrg_tot)
       elif (arg_type==2):
           return(fl_tot)
       else:
           return(nrg_tot,fl_tot)


def o_rad_cube_big(data_pth,arg_type,subdim=4,subcube_nb=0):

       """
       Read radgpu files serially then assemble
       arg_type = 1 -> output energy density
                  2 -> output flux
       subdim       -> If given, number of radgpus to create a subcube
       subcube_nb   -> If given, index of first target 
       """
       
       dim = 128 # size of a single cube

       line_nb=subdim/dim
       square_nb=(subdim/dim)**2
       height_nb=line_nb*2
       
       files =  os.listdir(data_pth)

       rads = np.asarray([name  for name in files if 'radgpu' in name])
       rads_nb = np.asarray([int(name[-5:])  for name in files if 'radgpu' in name])

       sub_rad_nb = np.int32(np.ones((square_nb)) * line_nb*(subcube_nb) + 1 + (line_nb-1)*line_nb*8*(subcube_nb//8) + (height_nb-1)*(square_nb)*64*(subcube_nb//64))
       #print(sub_rad_nb)
       sub_rad_nb=np.asarray([elem+ind%line_nb for ind,elem in enumerate(sub_rad_nb)])   
       #print(sub_rad_nb)
       sub_rad_nb=np.asarray([elem+(ind//line_nb)*8*line_nb for ind,elem in enumerate(sub_rad_nb)])      
       #print(sub_rad_nb)       
       sub_rad_nb = np.tile(sub_rad_nb,height_nb)
       #print(sub_rad_nb)
       sub_rad_nb=([int(elem+(ind//square_nb)*(8*line_nb)**2) for ind,elem in enumerate(sub_rad_nb)])
       #print(sub_rad_nb,np.shape(sub_rad_nb))
      #print(sub_rad_nb,np.shape(sub_rad_nb),[val in rads_nb for val in sub_rad_nb])
       #print(len(np.unique([where for where,val in enumerate(rads_nb) if val in sub_rad_nb])))
       rads = ([rads[np.argwhere(val==rads_nb)[0]][0] for val in sub_rad_nb])
       #print(rads,np.shape(rads))
                             

   
       # dim = 256 # size of a single cube
    
       # files =  os.listdir(data_pth)

       # rads = np.asarray([name  for name in files if 'radgpu' in name])
       # rads_nb = np.asarray([int(name[-5:])  for name in files if 'radgpu' in name])

       # sub_rad_nb = np.int64(np.ones((4)) * 2*(subcube_nb) +1 +16*(subcube_nb//8) + 2*512*(subcube_nb//64) -256*(subcube_nb//64))
       # print(sub_rad_nb)
       # sub_rad_nb[1]+=1;sub_rad_nb[2]=sub_rad_nb[0]+16;sub_rad_nb[3]=sub_rad_nb[2]+1
       # print(sub_rad_nb)
       # sub_rad_nb = np.tile(sub_rad_nb,(4))
       # print(sub_rad_nb)
       # sub_rad_nb=np.asarray([elem+(ind/4)*256 for ind,elem in enumerate(sub_rad_nb)])
       # print(sub_rad_nb)
       # print(sub_rad_nb,rads_nb)       
       # rads = np.sort(rads[[where for where,val in enumerate(rads_nb) if val in sub_rad_nb]])
       # print(rads)
       
       if arg_type != 2 : nrg_tot = np.empty((int(subdim),int(subdim),int(subdim)),dtype=np.float32)
       if arg_type != 1 : fl_tot = np.empty((3,int(subdim),int(subdim),int(subdim)),dtype=np.float32)

       subz,suby,subx=0,0,0

       #print(len(rads))
       #print(len(sub_rad_nb))
       
       for sub in range(square_nb*height_nb):
           #print('Opening %s' %rads[sub])
           #print(sub,subx,suby,subz)

           if arg_type not in [1,2] :
               tmp_nrg,tmp_fl=o_rad(data_pth+rads[sub])
               nrg_tot[subz*int(dim*0.5):(subz+1)*int(dim*0.5),suby*(dim):(suby+1)*(dim),subx*(dim):(subx+1)*(dim)]=tmp_nrg[1:-1,1:-1,1:-1]
               fl_tot[:,subz*int(dim*0.5):(subz+1)*int(dim*0.5),suby*(dim):(suby+1)*(dim),subx*(dim):(subx+1)*(dim)]=tmp_fl[:,1:-1,1:-1,1:-1]
           elif arg_type != 2 :
               tmp_nrg=o_rad(data_pth+rads[sub],1)
               nrg_tot[subz*int(dim*0.5):(subz+1)*int(dim*0.5),suby*(dim):(suby+1)*(dim),subx*(dim):(subx+1)*(dim)]=tmp_nrg[1:-1,1:-1,1:-1]
           elif arg_type != 1 :
               tmp_fl=o_rad(data_pth+rads[sub],2)
               fl_tot[:,subz*int(dim*0.5):(subz+1)*int(dim*0.5),suby*(dim):(suby+1)*(dim),subx*(dim):(subx+1)*(dim)]=tmp_fl[:,1:-1,1:-1,1:-1]
            
           if (sub+1)%square_nb==0:
                subz+=1
                suby=0
                subx=0
           elif (sub+1)%line_nb==0:
               suby+=1
               subx=0
           else:
               subx+=1


           #print(sub,subx,suby,subz)

       vect=np.arange(512,step=2)
       x,y=np.meshgrid(vect,vect)

       #print(vect)
       #print(fl_tot[:,z,y,x])
       ind=0
       
       # fig1=plt.figure()
       # ax1=fig1.add_subplot(111)
       # ax1.quiver(x,y,fl_tot[0,x,ind,y]/10e5*0.7,fl_tot[2,x,ind,y]/10e5*0.7)
       # fig2=plt.figure()
       # ax2=fig2.add_subplot(111)
       # ax2.quiver(x,y,fl_tot[0,ind,x,y]/10e5*0.7,fl_tot[2,ind,x,y]/10e5*0.7)
       # plt.show()
           
       if (arg_type==1):
           return(nrg_tot)
       elif (arg_type==2):
           return(fl_tot)
       else:
           return(nrg_tot,fl_tot)


       
#print(o_rad_cube_big('/data2/CoDaII/reduced/output_00088/raw/',2,512,64))
