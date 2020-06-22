#Ouvrir un fullbox
import numpy as np
import os


def o_data(data_pth):

        """                                                                                                 
    Fetch data from binary at data_pth                                                                  
    """

        buf_size = 2147483639
        with open(data_pth, 'rb', buf_size) as ff:
            
            bal   = np.fromfile( ff, dtype=np.int32, count=1 )[0]
            nx   = np.fromfile( ff, dtype=np.int32, count=1 )[0]
            ny   = np.fromfile( ff, dtype=np.int32, count=1 )[0]
            nz   = np.fromfile( ff, dtype=np.int32, count=1 )[0]
            bal   = np.fromfile( ff, dtype=np.int32, count=1 )[0]
            
            
            bitdata = np.zeros((nx*ny*nz*4),dtype='S1')
            
            counter=0
            while counter < (nx*ny*nz)*4 :
                bal   = np.fromfile( ff, dtype=np.int32, count=1 )[0]
                abal=abs(bal)
                bitdata[counter:counter+abal]=np.frombuffer(ff.read(abal),dtype='S1')
                counter+=abal
                bal2   = np.fromfile( ff, dtype=np.int32, count=1 )[0]

        return(np.reshape(np.frombuffer(bitdata,dtype='f'),(nz,ny,nx),order='A'))

# def o_data(data_pth):#,nb):

#     """
#     Fetch data from binary at data_pth
#     """
    
    
    
#     with open(data_pth, 'rb' ) as ff: 

#         #taille grille
#         bal   = np.fromfile( ff, dtype=np.int32, count=1 )[0]
#         nx   = np.fromfile( ff, dtype=np.int32, count=1 )[0]
#         ny   = np.fromfile( ff, dtype=np.int32, count=1 )[0]
#         nz   = np.fromfile( ff, dtype=np.int32, count=1 )[0]
#         bal   = np.fromfile( ff, dtype=np.int32, count=1 )[0]
      
#         #densite d energie  
#         bal   = np.fromfile( ff, dtype=np.int32, count=1 )[0]
#         data = np.fromfile( ff, dtype=np.float32, count= (nx)*(ny)*(nz) )
#         ff.close()
   
    
#     return(np.reshape(data,(nz,ny,nx),order='A'))


                                                                          
#data_pth = '/data/ngillet/titan/snap_Lorenzo/B4_256_CoDa/output_00022/'


def o_fullbox_big(data_pth,name,subdim=512,dim=512,subcube_nb=0):

       """
       Read files serially then assemble if necessary
       
       if dim==subdim then box is cube and only 1 file is read !
       Otherwise we assemble subdim*dim*dim boxes to make a box of dim**3

       name         -> Data type, eg : 'xion', 'rho', ...
       dim          -> Number of pxs in side of read cube
       subdim       -> Number of pxs in a side of the desired subcube
       subcube_nb   -> If given, index of first target 
       """
       


       line_nb=subdim/dim
       square_nb=(subdim/dim)**2
       height_nb=line_nb

       #print(line_nb,square_nb,height_nb)
       
       files =  os.listdir(data_pth)

       datas = np.asarray([fname  for fname in files if name  in fname])
       datas_nb = np.asarray([int(fname.strip(name))  for fname in files if name  in fname])
       #print(max(datas_nb))
       sub_data_nb = np.int32(np.ones((square_nb)) * line_nb*(subcube_nb) + 1 + (line_nb-1)*line_nb*8*(subcube_nb//1) + (height_nb-1)*(square_nb)*64*(subcube_nb//64))
       #print(sub_data_nb)
       sub_data_nb=np.asarray([elem+ind%line_nb for ind,elem in enumerate(sub_data_nb)]) 
       #print(sub_data_nb)  
       sub_data_nb=np.asarray([elem+(ind//line_nb)*8 for ind,elem in enumerate(sub_data_nb)])
       #print(sub_data_nb)            
       sub_data_nb = np.tile(sub_data_nb,height_nb)
       #print(sub_data_nb)
       sub_data_nb=([int(elem+(ind//square_nb)*(8)**2) for ind,elem in enumerate(sub_data_nb)])
       #print(sub_data_nb)
       #print(sub_data_nb,np.shape(sub_data_nb))
      #print(sub_data_nb,np.shape(sub_data_nb),[val in datas_nb for val in sub_data_nb])
       #print(len(np.unique([where for where,val in enumerate(datas_nb) if val in sub_data_nb])))

       #print(sub_data_nb)
       datas = ([datas[np.argwhere(val==datas_nb)[0]][0] for val in map(lambda x : x-1,sub_data_nb)])
       #print(datas,np.shape(datas))
                             

       data_tot = np.empty((int(subdim),int(subdim),int(subdim)),dtype=np.float32)


       subz,suby,subx=0,0,0

       #print(len(datas))
       #print(len(sub_data_nb))
       
       for sub in range(square_nb*height_nb):
           #print('Opening %s' %datas[sub])
           #print(sub,subx,suby,subz)


           data=o_data(os.path.join(data_pth,datas[sub]))
           
           data_tot[subz*int(dim):(subz+1)*int(dim),suby*(dim):(suby+1)*(dim),subx*(dim):(subx+1)*(dim)]=data[:,:,:]

            
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

       #vect=np.arange(512,step=2)
       #x,y=np.meshgrid(vect,vect)

       #print(vect)
       #print(fl_tot[:,z,y,x])
       ind=0
       

       return(data_tot)


       
#print(o_rad_cube_big('/data2/CoDaII/reduced/output_00088/raw/',2,512,64))


# '''test'''
# xion_test = o_fullbox_big('/data2/CoDaII/reduced/output_00088/fullboxfullres/','rho',512,0)

# plt.imshow(np.log10(xion_test[0,:,:]))
# #plt.figure()
# #plt.imshow(xion_test[:,0,:])
# #plt.figure()
# #plt.imshow(xion_test[:,:,0])
# plt.show()
