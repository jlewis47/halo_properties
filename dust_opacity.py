"""
Series of functions for simulating dust absorption from stellar particle host cells
to definable distance in direction of point outside of the box
"""

import numpy as np

def fast_card_ray3D(X0,Y0,Z0,D_lim,lim):
    
    """
    This doesn't do proper path to distant obs but considers observer to so distant that we follow cardinal directions
    """

    rnd_D_lim=np.ceil(D_lim)

    Ds=np.arange(0,rnd_D_lim)

    Xs=X0-Ds
    Ys=np.ones_like(Xs)*Y0
    Zs=np.ones_like(Xs)*Z0

    Xs[Xs<0]=Xs[Xs<0]+lim
    Xs[Xs>=lim]=Xs[Xs>=lim]-lim #wrap around box both ways

    return(Xs,Ys,Zs,Ds,Ds[-1])

def ray3D(X0,Y0,Z0,D_lim,phi,theta,lim):

    """
    phi,theta in rad
    X0,Y0,Z0 in grid units
    D_lim in float of d_units
    lim box size in cells
    """
    tol=0.1
    
    
    xn=X0
    yn=Y0
    zn=Z0

    #need to find direction in x,y,y based on phi
    x_step=np.sign(np.cos(phi))
    y_step=np.sign(np.sin(phi))    
    z_step=np.sign(np.cos(theta))
    
    Xs=[xn]
    Ys=[yn]
    Zs=[zn]
    Ds=[]

    D_tot=0


    while(D_tot<D_lim):


        if xn-lim>=-tol:
            xn=0
            Xs.append(xn)
            
            if np.abs(yn-lim)<tol:
                yn=0
            if np.abs(zn-lim)<tol:
                zn=0
            
            Ys.append(yn)
            Zs.append(zn)
            Ds.append(0)
            
        elif yn-lim>=-tol:
            yn=0
            Ys.append(yn)
            
            if np.abs(xn-lim)<tol:
                xn=0
            if np.abs(zn-lim)<tol:
                zn=0    
            
            Xs.append(xn)
            Zs.append(zn)
            Ds.append(0)               
        
        elif zn-lim>=-tol:
            zn=0
            Zs.append(zn)
            
            if np.abs(xn-lim)<tol:
                xn=0
            if np.abs(yn-lim)<tol:
                yn=0    
            
            Xs.append(xn)
            Ys.append(yn)
            Ds.append(0) 


        #hyp1
    
        xnp1=np.floor(xn)+x_step
        dx=xnp1-xn

        ynp1=np.tan(phi)*dx+yn
        dy=ynp1-yn
        
        rhop1=(dx**2.+dy**2.)**0.5
        rp1=rhop1/np.sin(theta)
        
        znp1=np.cos(theta)*rp1+zn
        dz=znp1-zn
    

        
        #print(dx,dy,dz)
        #print(xnp1,ynp1,znp1)
        #print(xn,yn,zn)
        #print(1)        
    
        if abs(dy)>1 or abs(dz)>1:

            #hyp2
        
            ynp1=np.floor(yn)+y_step
            dy=ynp1-yn
    
            xnp1=dy/np.tan(phi)+xn
            dx=xnp1-xn

            drho=(dx**2.+dy**2.)**0.5
            dr=drho/np.sin(theta)
            
            znp1=np.cos(theta)*dr+zn
            dz=znp1-zn            
        
        
            #print(dx,dy,dz)
            #print(xnp1,ynp1,znp1)
            #print(xn,yn,zn)        
            #print(2)
        
            if abs(dx)>1 or abs(dz)>1:
                
                #hyp 3
                
                znp1=np.floor(zn)+z_step
                dz=znp1-zn
                
                dr=dz/np.cos(theta)
                drho=dr*np.sin(theta)
                
                ynp1=drho*np.sin(phi)+yn
                dy=ynp1-yn
                
                xnp1=drho*np.cos(phi)+xn
                dx=xnp1-xn
                
                #print(dx,dy,dz)
                #print(xnp1,ynp1,znp1)
                #print(xn,yn,zn)            
                #print(3)    
            #print('hyp 2',dy,dx)
        #print(dx,dy,dz)
        #print(xnp1,ynp1,znp1)
        #print(xn,yn,zn)

        #else:
        
            #print('hyp 1',dy,dx)
        
 

        #print(xnp1,ynp1)
        Xs.append(xnp1)
        Ys.append(ynp1)
        Zs.append(znp1)
    
        D=(dx**2.+dy**2.+dz**2.)**0.5
        D_tot+=D
        Ds.append(D)

        xn=xnp1
        yn=ynp1    
        zn=znp1
        
    return(map(np.asarray,[Xs,Ys,Zs,Ds,D_tot]))



def star_path(pos,dist_obs,dust,opacity,rlim,px_to_m,ldx):
    
    """
    pos in cells
    dist_obs in cells
    dust in g/cm^3
    opacity in cm^2/g
    rlim in cells
    px_to_m in cm
    """


    
    xstt,ystt,zstt=pos #in cartesian ref centred at 0,0,0 of box
    
    
    #out of box target in stt ref
 
    xtgt=-0-xstt
    ytgt=-0-ystt
    ztgt=-dist_obs-zstt #our maps are in (x,y) plane so we need extinction in perp direction
    
    ray_vect=[xtgt,ytgt,ztgt]
    
    phi_tgt=np.arctan2(ray_vect[1],ray_vect[0])
    theta_tgt=np.arccos(ray_vect[2]/np.linalg.norm(ray_vect))
    
    # x_lim=-xstt
    # y_lim=np.tan(phi_tgt)*xstt
    # z_lim=(x_lim**2.+y_lim**2.)**0.5/np.tan(theta_tgt)
    # D_lim=(x_lim**2.+y_lim**2.+z_lim**2.)**0.5
    
    #Xs,Ys,Zs,Ds,D_tot=ray3D(xstt,ystt,zstt,D_lim,phi_tgt,theta_tgt,ldx)
    Xs,Ys,Zs,Ds,D_tot=ray3D(xstt,ystt,zstt,rlim,phi_tgt,theta_tgt,ldx)

    #Xs,Ys,Zs,Ds,D_tot=fast_card_ray3D(xstt,ystt,zstt,rlim,ldx)

    Xs,Ys,Zs=np.int16([Xs[:-1],Ys[:-1],Zs[:-1]])

    #Xs,Ys,Zs=np.int16([Xs,Ys,Zs])

    return(np.sum(dust[Zs,Ys,Xs]*Ds)*px_to_m*opacity)





def star_path_cheap(pos,dust,rlim):
    
    """
    doesn't do trig and just takes a straight line
    takes small box around halo not all box

    pos in cells
    """

    xstt,ystt,zstt=pos #in cartesian ref centred at 0,0,0 of box
    
    #out of box target in stt ref
    
    Zs=np.arange(zstt,min(zstt+rlim,len(dust)),1,dtype=np.int16)
    Ys=np.ones_like(Zs,dtype=np.int16)*int(ystt)
    Xs=np.ones_like(Zs,dtype=np.int16)*int(xstt)
    
    return(np.sum(dust[Zs,Ys,Xs]))
