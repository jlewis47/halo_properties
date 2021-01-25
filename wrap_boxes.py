import numpy as np
from read_fullbox_big import *
from read_radgpu_big import *
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def get_neighbour_cubes(subcube_nb,n_subcubes):

    nrow=int(round(n_subcubes**(1./3))) #number of subcubes in a row for each cartesian direction
    nsqu=int(round(n_subcubes**(2./3))) #number of subcubes in a square (row*row)
    
    vect=np.arange(0,nrow)
    box_of_nbs=vect+vect[:,np.newaxis]*nrow+vect[:,np.newaxis,np.newaxis]*nsqu

    reflected_box=np.tile(box_of_nbs,(3,3,3))
    
    z_nb=int(subcube_nb//nsqu)
    y_nb=int(subcube_nb%nsqu//nrow)
    x_nb=int(subcube_nb-z_nb*nsqu-y_nb*nrow)

    #print(z_nb,y_nb,x_nb)
    
    vect_rel=np.arange(-1,2)

    z_coords=z_nb+vect_rel
    y_coords=y_nb+vect_rel
    x_coords=x_nb+vect_rel    

    #print(z_coords,y_coords,x_coords)
    
    x_get,y_get,z_get=np.meshgrid(z_coords,y_coords,x_coords)

    #print(x_get,y_get,z_get)


    nbs_to_get=reflected_box[x_get,y_get,z_get]


    
    return(nbs_to_get)


def get_overstep_RT_cubed(subcube_nb,data_path_rad,OOB,n_subcubes=512,size=512,overstep=3,sort=2):

    if sort==2:
        box=np.zeros((3,size*overstep,size*overstep,size*overstep),dtype=np.float32) 
    elif sort==1:
        box=np.zeros((size*overstep,size*overstep,size*overstep),dtype=np.float32)         

    nbs_to_get=get_neighbour_cubes(subcube_nb,n_subcubes)

    delta=int((overstep-1)*0.5*size) #size of overstep in cells
    #so for ex on xvector we have 0:delta then delta:size then size:delta from
    #3 different subcubes where the central one is subcube_nb
    
    under,over=OOB

    #so can use with only one entry in OOB
    if np.shape(np.shape(under))!=(2,):
        under=np.array([under])
        over=np.array([over])


        
    zbnds=under[:,0],np.ones(len(under))==1,over[:,0]
    ybnds=under[:,1],np.ones(len(under))==1,over[:,1]
    xbnds=under[:,2],np.ones(len(under))==1,over[:,2]


    #print(zbnds,ybnds,xbnds)
    
    for ix,x in enumerate(nbs_to_get.swapaxes(0,1)):
        for iy,y in enumerate(x):
            for iz,z in enumerate(y):
    #so can use with only one entry in OOB

                #print(xbnds[ix],ybnds[iy],zbnds[iz])
                #print(np.all([xbnds[ix],ybnds[iy],zbnds[iz]],axis=0))
                
                if np.any(np.all([xbnds[ix],ybnds[iy],zbnds[iz]],axis=0)):


                    if ix==0:
                        xlow,xhigh=0,delta #for big cube of sides size+2*delta
                        load_xlow,load_xhigh=size-delta,size #for cube just opened
                    elif ix==1:
                        xlow,xhigh=delta,size+delta
                        load_xlow,load_xhigh=0,size #for cube just opened                 
                    else:
                        xlow,xhigh=size+delta,size+2*delta
                        load_xlow,load_xhigh=0,delta #for cube just opened                        


                    if iy==0:
                        ylow,yhigh=0,delta #for big cube of sides size+2*delta
                        load_ylow,load_yhigh=size-delta,size #for cube just opened
                    elif iy==1:
                        ylow,yhigh=delta,size+delta
                        load_ylow,load_yhigh=0,size #for cube just opened                 
                    else:
                        ylow,yhigh=size+delta,size+2*delta
                        load_ylow,load_yhigh=0,delta #for cube just opened                        

                    if iz==0:
                        zlow,zhigh=0,delta #for big cube of sides size+2*delta
                        load_zlow,load_zhigh=size-delta,size #for cube just opened
                    elif iz==1:
                        zlow,zhigh=delta,size+delta
                        load_zlow,load_zhigh=0,size #for cube just opened                 
                    else:
                        zlow,zhigh=size+delta,size+2*delta
                        load_zlow,load_zhigh=0,delta #for cube just opened                        
                        
                        
                    if sort==2:

                            print('RT loaded %i'%z)                
                            try:
                                
                                #box[:,ix*size:(ix+1)*size,iy*size:(iy+1)*size,iz*size:(iz+1)*size]=o_rad_cube_big(data_path_rad,2,z)
                                box[:,xlow:xhigh,ylow:yhigh,zlow:zhigh]=o_rad_cube_big(data_path_rad,2,z)[load_zlow:load_zhigh,load_ylow:load_yhigh,load_xlow:load_xhigh]


                                #print([xlow,xhigh,ylow,yhigh,zlow,zhigh],[load_zlow,load_zhigh,load_ylow,load_yhigh,load_xlow,load_xhigh])
                                #print([ix*size,(ix+1)*size,iy*size,(iy+1)*size,iz*size,(iz+1)*size])
                            except IndexError:
                                box[:,xlow:xhigh,ylow:yhigh,zlow:zhigh]=-1
                                continue

                            
                    elif sort==1:

                            print('RT loaded %i'%z)                
                            try:
                               box[xlow:xhigh,ylow:yhigh,zlow:zhigh]=o_rad_cube_big(data_path_rad,2,z)[load_xlow:load_xhigh,load_ylow:load_yhigh,load_zlow:load_zhigh]
                            except IndexError:
                                box[xlow:xhigh,ylow:yhigh,zlow:zhigh]=-1
                                continue
                        
    return(box)



def get_overstep_hydro_cubed(box,subcube_nb,data_path,name,OOB,n_subcubes=512,size=512,overstep=3):


    #we get this from the function call so python handles memory stuff a big
    #box=np.zeros((size*overstep,size*overstep,size*overstep),dtype=np.float32)         

    nbs_to_get=get_neighbour_cubes(subcube_nb,n_subcubes)

    nrows=round(n_subcubes**(1./3))
    ncols=round(n_subcubes**(2./3))
    
    delta=int((overstep-1)*0.5*size) #size of overstep in cells
    #so for ex on xvector we have 0:delta then delta:size then size:delta from
    #3 different subcubes where the central one is subcube_nb
    
    under,over=OOB

    #so can use with only one entry in OOB
    if np.shape(np.shape(under))!=(2,):
        under=np.array([under])
        over=np.array([over])


    #print(nbs_to_get.swapaxes(0,1))

        
    zbnds=under[:,0],np.ones(len(under))==1,over[:,0]
    ybnds=under[:,1],np.ones(len(under))==1,over[:,1]
    xbnds=under[:,2],np.ones(len(under))==1,over[:,2]

    #first we find the subcubes we need by checken all xbnds,ybnds,zbnds
    subs_required=[]

    for ix,x in enumerate(nbs_to_get.swapaxes(0,1)):
        for iy,y in enumerate(x):
            for iz,z in enumerate(y):


                #print(xbnds[ix],ybnds[iy],zbnds[iz])
                #print(np.all([xbnds[ix],ybnds[iy],zbnds[iz]],axis=0))

                if np.any(np.all([xbnds[ix],ybnds[iy],zbnds[iz]],axis=0)):
                    subs_required.append(z)
    

    #for every unique subcube number that we require we load the subcube
    #then we iterate over central and surrounding data cube using the currently loaded subcube when necessary

    for n_subcube in np.unique(subs_required):
    
        data_name=os.path.join(data_path,'%s_%i'%(name,n_subcube))

        print('loaded %s_%i'%(name,n_subcube))                
        try:
            cur_box=o_data(data_name)
            #cur_box=np.ones((size,size,size))*n_subcube
        except IndexError:
            cur_box=np.zeros((size,size,size))
            continue

    
        for ix,x in enumerate(nbs_to_get.swapaxes(0,1)):
            for iy,y in enumerate(x):
                for iz,z in enumerate(y):
        #so can use with only one entry in OOB

                    if z!=n_subcube:continue
                    
                    #print(xbnds[ix],ybnds[iy],zbnds[iz])
                    #print(np.all([xbnds[ix],ybnds[iy],zbnds[iz]],axis=0))

                    if np.any(np.all([xbnds[ix],ybnds[iy],zbnds[iz]],axis=0)):


                        if ix==0:
                            xlow,xhigh=0,delta #for big cube of sides size+2*delta
                            load_xlow,load_xhigh=size-delta,size #for cube just opened
                        elif ix==1:
                            xlow,xhigh=delta,size+delta
                            load_xlow,load_xhigh=0,size #for cube just opened                 
                        else:
                            xlow,xhigh=size+delta,size+2*delta
                            load_xlow,load_xhigh=0,delta #for cube just opened                        


                        if iy==0:
                            ylow,yhigh=0,delta #for big cube of sides size+2*delta
                            load_ylow,load_yhigh=size-delta,size #for cube just opened
                        elif iy==1:
                            ylow,yhigh=delta,size+delta
                            load_ylow,load_yhigh=0,size #for cube just opened                 
                        else:
                            ylow,yhigh=size+delta,size+2*delta
                            load_ylow,load_yhigh=0,delta #for cube just opened                        

                        if iz==0:
                            zlow,zhigh=0,delta #for big cube of sides size+2*delta
                            load_zlow,load_zhigh=size-delta,size #for cube just opened
                        elif iz==1:
                            zlow,zhigh=delta,size+delta
                            load_zlow,load_zhigh=0,size #for cube just opened                 
                        else:
                            zlow,zhigh=size+delta,size+2*delta
                            load_zlow,load_zhigh=0,delta #for cube just opened                        


                        #print(xlow,xhigh,ylow,yhigh,zlow,zhigh)
                        #print(load_xlow,load_xhigh,load_ylow,load_yhigh,load_zlow,load_zhigh)

                        #print(z,ix,iy,iz)
                        
                        box[xlow:xhigh,ylow:yhigh,zlow:zhigh]=cur_box[load_xlow:load_xhigh,load_ylow:load_yhigh,load_zlow:load_zhigh]

    # fig=plt.figure()
    # ax=fig.add_subplot(111)

    # img=ax.imshow(np.log10(box[510,:,:]).T,origin='lower',vmin=-3,vmax=10)
    # ax.set_xlabel('y')
    # ax.set_ylabel('x')
    # plt.colorbar(img)
    # fig.savefig('test_%i_xy'%subcube_nb)

    # img=ax.imshow(np.log10(box[:,:,510]).T,origin='lower',vmin=-3,vmax=10)
    # ax.set_xlabel('z')
    # ax.set_ylabel('y')

    # fig.savefig('test_%i_yz'%subcube_nb)

    # img=ax.imshow(np.log10(box[:,510,:]).T,origin='lower',vmin=-3,vmax=10)
    # ax.set_xlabel('z')
    # ax.set_ylabel('x')    

    # fig.savefig('test_%i_xz'%subcube_nb)

    return(box)



