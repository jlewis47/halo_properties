"""
Contains functions for sq integration programmes
"""

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as pat
import matplotlib.ticker as plticker
import os
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost
from scipy import spatial
from read_assoc_latest import *
from matplotlib.ticker import AutoMinorLocator,MultipleLocator,LogLocator,LinearLocator
from constants_latest import *
import sys
from datetime import date
from shutil import copyfile
import filecmp
import matplotlib.patches as pat
from scipy.interpolate import interp2d

def linr(x,a,b):
    return(x*a+b)

def get_Mp(om_m,om_b,H0,Lco,Np_tot):
    return(3.*(H0*1e3/(pc*1e6))**2.*(om_m-om_b)*(Lco*pc*1e6)**3./Np_tot/8./np.pi/G/Msol) #En Msol 4.04*1e5)

def get_infos(info_path,out_nb,whole_side) :


        #Get scale factor and co
        info = np.genfromtxt(os.path.join(info_path,'info_00'+out_nb+'.txt'), skip_header = 8, max_rows = 10, usecols = 2)
        t=info[0]
        a=info[1]
        H0=info[2]
        om_m = info[3]
        om_b = info[6]
        unit_l,unit_d,unit_t = info[7:10]
        l=unit_l/pc/1e6/100.*H0/100./a
        Lco = l/H0*100 #Mpc comobile
        L=Lco*a
        px_to_m = L*pc*1e6/whole_side

        # if out_nb == '092':
        #     tstep=unit_t*2.812E-03/3600./24./365./1e6      
            
        if out_nb == '088':
            tstep=unit_t*2.812E-03/3600./24./365./1e6
        elif out_nb == '071':
            tstep=unit_t*3.250E-03/3600./24./365./1e6
        elif out_nb == '059':
            tstep=unit_t*3.592E-03/3600./24./365./1e6
        elif out_nb == '092':
            tstep=unit_t*2.591E-03/3600./24./365./1e6 #or 2.555E-03 ?
        elif out_nb == '050':
            tstep=unit_t*3.881E-03/3600./24./365./1e6 
        elif out_nb == '042':
            tstep=unit_t*4.834E-03/3600./24./365./1e6 
        elif out_nb == '023':
            tstep=unit_t*1.443E-02/3600./24./365./1e6 

            
        return(t,a,H0,om_m,om_b,unit_l,unit_d,unit_t,Lco,L,px_to_m,tstep)


def get_infos_no_t(info_path,out_nb,whole_side) :

    infos=np.zeros(10,dtype=np.float64)
    j=0
    with open(os.path.join(info_path,'info_%05i.txt'%int(out_nb)),'r') as f:
        for i,line in enumerate(f):
            if (i>7) and (i<18):
                
                infos[j] = (line.split('=')[1]).rstrip('\n')
                j+=1


        t=infos[0]
        a=infos[1]
        H0=infos[2]
        om_m = infos[3]
        om_l = infos[4]
        om_k = infos[5]
        om_b = infos[6]
        unit_l,unit_d,unit_t = infos[7:10]
        
        l=float(np.round(unit_l/pc/1e6/100.*H0/100./a))
        Lco = l/H0*100 #Mpc comobile
        L=Lco*a
        
        px_to_m = L*pc*1e6/whole_side
        
        return(t,a,H0,om_m,om_l,om_k,om_b,unit_l,unit_d,unit_t,l,Lco,L,px_to_m)


    
def get_r200_ticks(r,Mp,Np_tot=4096**3,size=4096 ):
    """
    Return mass of a halo based on r200 (in cell units)
    """

    out = ((r/size)**3.0)*Mp*800*np.pi*Np_tot/3.

    return((out))


def ax_setup(axis,extremax=plt.xlim(),extremay=plt.ylim(),ylog=True,xlog=True):
    """
    Duplicate axes 
    Setup minor ticks
    Set fi limits
    Turn on grid
    """


    axis.set_ylim(extremay)
    axis.set_xlim(extremax)
    
    sizey = extremay[1]-extremay[0]
    sizex = extremax[1]-extremax[0]


    if sizey <1:
        sizey = np.log10(1/np.min([extremay[1],extremay[0]]))

    if sizex <1:
        sizex = np.log10(1/np.min([extremax[1],extremax[0]]))
        
        
    #print(sizex,sizey)
        
    if ylog :
        axis.yaxis.set_minor_locator(LogLocator(base=10,subs=[2,4,6,8],numticks=(sizey)*4))
        #axis.yaxis.set_major_locator(LogLocator(base=10,numticks=sizey))        
    else:
        #print(np.log10(extremay[1]/extremay[0]))
        axis.yaxis.set_minor_locator(AutoMinorLocator(np.floor(np.log10(extremay[1]/extremay[0]))*4))
        #axis.yaxis.set_major_locator(MultipleLocator(np.floor(np.log10(extremay[1]/extremay[0]))))
        
    axis.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,4,6,8],numticks=(sizex)*4))
    #axis.xaxis.set_major_locator(LogLocator(base=10,numticks=sizex))       

    if xlog :
        axis.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,4,6,8],numticks=(sizex)*4))
        #axis.yaxis.set_major_locator(LogLocator(base=10,numticks=sizey))        
    else:
        #print(np.log10(extremay[1]/extremay[0]))
        axis.xaxis.set_minor_locator(AutoMinorLocator(np.floor(np.log10(extremax[1]/extremax[0]))*4))
        #axis.yaxis.set_major_locator(MultipleLocator(np.floor(np.log10(extremay[1]/extremay[0]))))


    axis.grid(linestyle=':',linewidth=1,color='gray')
    
    axis.tick_params(axis='both', which='both',direction='in',right='on',top='on',pad=8)
    
    return()

def get_pos(arr):

    """
    get array N,3 position array
    """

    x,y,z = arr['x'],arr['y'],arr['z']

    return(np.transpose([x,y,z]))

        

        
def read_arr(out_nb,key_word,tstep,path,info_path,overwrite=False):

    def load_dict(name,path,keyword,out_nb):
        with open(os.path.join(path,name),'rb') as src :    
            N=np.load(src)
            Ncol=np.load(src)

            tab=np.load(src)

        return(tab)

    def load_simple(name,path,keyword,out_nb):
        
        with open(os.path.join(path,name),'rb') as src :    
            N=np.load(src)
            tab=np.load(src)

        return(tab)


    
    to_load_dict_names=['tab']
    to_load_else_names=['forgets_SFR','forgets_mult_det','forgets']


    
    to_load_dict_names=['filtrd_{0:s}_{1:s}_{2:s}'.format(name,key_word,out_nb) for name in to_load_dict_names]
    to_load_else_names=['filtrd_{0:s}_{1:s}_{2:s}'.format(name,key_word,out_nb) for name in to_load_else_names] 

    to_load_dict=[d for d in os.listdir(path) if np.any(np.in1d(d,to_load_dict_names))]
    to_load_else=[d for d in os.listdir(path) if np.any(np.in1d(d,to_load_else_names))]

    if (len(to_load_dict)!=1) or overwrite: #only for _latest case
        forgets_SFR,forgets_mult_det,tab,forgets=write_arr(out_nb,key_word,tstep,path,info_path)
        dicts=[tab]
        elses=[forgets_SFR,forgets_mult_det,forgets]


    else:

        dicts=[load_dict(tl,path,key_word,out_nb) for tl in to_load_dict]
        elses=[load_simple(tl,path,key_word,out_nb) for tl in to_load_else]

        
    return(dicts,elses)


def read_outs(out_nb,key_word,path,lim=0):


        dats = np.asarray([tgt for tgt in os.listdir(path) if key_word+'_out_'+out_nb+'_' in tgt if os.path.isfile(os.path.join(path,tgt))])
        
        stripper = out_nb[-1]

        dats_arg=np.argsort([ int(dat_name[-3:].replace(stripper+'_','').strip('_')) for dat_name  in dats])
        #print(dats_arg,(dats_arg.dtype))
        dats = dats[dats_arg]

        assert len(dats)>0, "Didn't find any files matching suffix at path location"
        
        Ntot=0
        if lim==0:lim=len(dats)
        for ind,tgt in enumerate(dats[:lim]):
        #len(dats)//10


             with open(os.path.join(path,tgt),'rb') as out:
                 #print(tgt)
                 N=np.load(out)
                 Ncol=np.load(out)
                 a=np.load(out)
                 #print(N,Ncol)
                 keys=np.load(out)
                 #print(keys)
                 Ntot+=N

             out.close()

        ##print(keys.size)
        ##print([(key,'f8') for key in keys])
        tab=np.zeros((Ntot),dtype=np.dtype({'names':keys,'formats':['f8' for x in range(Ncol)]}))
        #tab=np.zeros((Ntot,Ncol),dtype=[(key,'f8') for key in keys])
        Ntot=0

        
        
        for ind,tgt in enumerate(dats[:lim]):
        #len(dats)//10


             with open(os.path.join(path,tgt),'rb') as out:

                 #print(tgt)
                 N=np.load(out)
                 Ncol=np.load(out)
                 a=np.load(out)
                 #print(N,Ncol)
                 keys=np.load(out)
                 #print(keys)
                 tmp=np.load(out)
                 #print(np.shape(tmp),tmp)

                 for k_nb,key in enumerate(keys[:]) :
                     tab[Ntot:Ntot+N][key]=tmp[:,k_nb]

                 Ntot+=N

    

        return(tab)

    

    
def bound_round(nb):
    """
    Round to closest integer for each number in nb
    Rounding UP from 0.5
    """
    return(np.int32(np.remainder(nb,1)>=0.5*np.ones_like(nb))*np.ceil(nb) +
           np.int32(np.remainder(nb,1)<0.5*np.ones_like(nb))*np.floor(nb))



def get_std(data_set,weights=None):
    """
    Return std using numpy weighted averages
    Assumes 1d input
    """

    if weights==None : weights = np.ones_like(data_set)
    mx2 = np.average(data_set,weights=weights)**2
    x2m = np.average(data_set**2,weights=weights)
    return((x2m-mx2)**0.5)
    
def do_half_round(nb):
    """
    Set coordinates to nearest pixel centre
    """
    return(np.floor(nb)+0.5)
    

# def get_r200(M,Np_tot=4096.**3,size=4096.):
#     """
#     Return r200 (in cell units) based on number of particles in a halo
#     M is mass in number of DM particles
#     """    
    
#     out = (M/Np_tot*3./800./np.pi)**(1/3.)*size
#     if out < 1.5 : out=1.5
#     return((out))


def get_r200(M):
    """
    Return r200 (in cell units) based on number of particles in a halo
    """
   
    return((M*3./800./np.pi)**(1/3.))

def cart_2_sph(x,y,z):

    """
    Takes radians !
    """

    r=(x**2+y**2+z**2)**0.5

    phi=np.arctan2(y,x) #radians
    
    the=np.arccos(z/r) #radians


    return(r,phi,the)

def sph_2_cart(r,phi,the):

    """
    Take radians !
    """

    x=r*np.sin(the)*np.cos(phi)

    y=r*np.sin(the)*np.sin(phi)
                    
    z=r*np.cos(the)


    return(x,y,z)

def get_mag_tab():

    """
    Read mag to stellar mass tab
    """

    path="/data2/jlewis/BPASS_MAB1600/"

    srcs=[src for src in os.listdir(path) if 'Kroupa' in src]
    
    mag_tab=[]
    Zbins=[]


    for src in srcs :

        mag_tab.append(np.genfromtxt(os.path.join(path,src),delimiter=','))
        Zbins.append(float(src.split('=')[-1].split('_')[0]))
    
    #print(mag_tab)
    
    Zsort=np.argsort(Zbins)                                                                          
    mag_tab=np.asarray(mag_tab)[Zsort]                                                               
    Zbins=np.asarray(Zbins)[Zsort]   
    

    Agebins = np.asarray(mag_tab[0])[:,0]
    mag_tab=np.reshape(np.vstack(mag_tab)[:,1],(len(Zbins),len(Agebins)))
    



    return(mag_tab,Zbins,Agebins)


def get_mag_tab_BPASSV221():

        reading=[]
        with open("/home/jlewis/METAL-RAMSES-CUDATON/aton/src_files/Emissivity_MAB1600_BPASSv2.2.1_kroupa_binary_MMax=100.txt",'r') as src:

                for line in src:
                    if '#' not in str(line):
                        tmp=line.split(',')
                        tmp[-1]=tmp[-1].strip('\n')
                        #print(tmp)
                        reading.append(np.float64(tmp))

                

        age_bins=reading[2]
        metal_bins=reading[3]

        xis=np.vstack(reading[3+1:3+13+1])
        mags=np.vstack(reading[3+1+13:3+13+1+13+1])

        return(mags,xis,metal_bins,age_bins)


def get_mag_tab_BPASSV221_betas():

    reading=[] 
    with open("BPASSV221/Emissivity_MAB1600_BPASSv2.2.1_kroupa_binary_MMax=100_betas.txt",'r') as src:

        for line in src:
            if '#' not in str(line):
                tmp=line.split(',')
                tmp[-1]=tmp[-1].strip('\n')
                #print(tmp)
                reading.append(np.float64(tmp))


    age_bins=reading[2]
    metal_bins=reading[3]

    xis=np.vstack(reading[4:17])
    mags=np.vstack(reading[17:30])
    contbetalow=np.vstack(reading[30:43])
    contbetahigh=np.vstack(reading[43:56])
    beta=np.vstack(reading[56:69])

    return(mags,xis,contbetalow,contbetahigh,beta,metal_bins,age_bins)


def get_star_mags(star_age,mag_tab,Agebins,Zbins):

    """
    return stellar magnitude table 
    outside of parameter range (Age,Z), values are table edges
    CoDaII -> Z=0.001
    """

    
    
    interp = interp2d(np.log10(Agebins),np.log10(Zbins),(mag_tab),kind='linear')
    
    return(interp(np.log10(star_age),np.log10(0.001)))


def get_star_mags_metals(star_age,star_metal,mag_tab,Agebins,Zbins):

    """
    return stellar magnitudes
    outside of parameter range (Age,Z), values are table edges
    """

    ages=np.copy(star_age)                                                                           
    Zs=np.copy(star_metal)                                                                               
    
    ages[ages<1]=1                                                                                   
    ages[ages>1e5]=1e5                                                                               
                                                                                                     
    Zs[Zs<1e-5]=1e-5                                                                                 
    Zs[Zs>0.04]=0.04       
    
    interp = interp2d(np.log10(Agebins),np.log10(Zbins),(mag_tab),kind='linear')
    
    return(np.asarray([interp(np.log10(age),np.log10(Z))[0] for age,Z in zip(ages,Zs)]))


def get_star_xis_metals(star_age,star_metal,xis_tab,Agebins,Zbins):

    """
    return stellar emissivity in ph/s/msol
    outside of parameter range (Age,Z), values are table edges
    """

    ages=np.copy(star_age)                                                                           
    Zs=np.copy(star_metal)                                                                               
    
    ages[ages<1]=1                                                                                   
    ages[ages>1e5]=1e5                                                                               
                                                                                                     
    Zs[Zs<1e-5]=1e-5                                                                                 
    Zs[Zs>0.04]=0.04       
    
    Zargs=np.digitize(star_metal,Zbins)
    Zargs[Zargs>len(Zbins)-1]=len(Zbins)-1

    Aargs=np.digitize(star_age,Agebins)
    Aargs[Aargs>len(Agebins)-1]=len(Agebins)-1
    
    return(xis_tab[Zargs,Aargs])


def sum_over_rays(field,ctr,r200,rad_res,X_primes,Y_primes,Z_primes):

    """
    compute sum over rays centred at ctr using given resolution
    field is a box to sum

    """

    size=np.shape(field)[0]

    ctr=np.asarray(ctr)-0.5*size
    delta_R=np.copy(ctr)
    Xs,Ys,Zs=[X_primes,Y_primes,Z_primes]+delta_R[:,np.newaxis,np.newaxis]


    
    Rs=np.linalg.norm([Xs,Ys,Zs],axis=0)


    #used for getting data ... Need to be between 0 and size !!!
    Xs_snap,Ys_snap,Zs_snap=np.int32([Xs+0.5*size,Ys+0.5*size,Zs+0.5*size])

    
    
    IB=Rs<=r200 #if points in r200 ...
    OOB=~IB

    
    #print(np.shape(OOB),np.shape(sampled))
    sampled=np.zeros_like(Xs)
    sampled[IB]=field[Xs_snap[IB],Ys_snap[IB],Zs_snap[IB]]


    
    #sampled[Rs>r200]=0
    


    

    #if paths
    x_matches=np.zeros((np.shape(Xs)[0]),dtype=np.float32)
    y_matches=np.copy(x_matches)
    z_matches=np.copy(y_matches)

    
    argmin=np.argmin(np.abs(Rs[:,:]-r200),axis=1)

    #if paths
    inds=np.arange(len(x_matches))
    x_matches[inds]=Xs[inds,argmin]
    y_matches[inds]=Ys[inds,argmin]
    z_matches[inds]=Zs[inds,argmin]

    
    


    

    #if norms are 0 then we are on the border and the result should be 1
    #scal[np.isnan(scal)]=1

    scal=((x_matches-ctr[0])*x_matches+(y_matches-ctr[1])*y_matches+(z_matches-ctr[2])*z_matches)/((np.linalg.norm([x_matches,y_matches,z_matches],axis=0)*np.linalg.norm([x_matches-ctr[0],y_matches-ctr[1],z_matches-ctr[2]],axis=0)))
    
    scal[np.isnan(scal)]=1

    rays=np.exp(-np.sum(sampled,axis=1)*rad_res*scal)





    return(rays)

def sum_over_rays_bias(field,ctr,r200,rad_res,X_primes,Y_primes,Z_primes):

    """
    compute sum over rays centred at ctr using given resolution
    field is a box to sum

    corrects for angular bias

    """

    size=np.shape(field)[0]

    ctr=np.asarray(ctr)-0.5*size
    delta_R=np.copy(ctr)
    Xs,Ys,Zs=[X_primes,Y_primes,Z_primes]+delta_R[:,np.newaxis,np.newaxis]


    
    Rs=np.linalg.norm([Xs,Ys,Zs],axis=0)


    #used for getting data ... Need to be between 0 and size !!!
    Xs_snap,Ys_snap,Zs_snap=np.int32([Xs+0.5*size,Ys+0.5*size,Zs+0.5*size])

    
    
    IB=Rs<=r200 #if points in r200 ...
    OOB=~IB

    

    sampled=np.zeros_like(Xs)
    sampled[IB]=field[Xs_snap[IB],Ys_snap[IB],Zs_snap[IB]]

    #if paths
    x_matches=np.zeros((np.shape(Xs)[0]),dtype=np.float32)
    y_matches=np.copy(x_matches)
    z_matches=np.copy(y_matches)

    
    argmin=np.argmin(np.abs(Rs[:,:]-r200),axis=1)


    inds=np.arange(len(x_matches))
    x_matches[inds]=Xs[inds,argmin]
    y_matches[inds]=Ys[inds,argmin]
    z_matches[inds]=Zs[inds,argmin]

    #if norms are 0 then we are on the border and the result should be 1
    #scal[np.isnan(scal)]=1

    scal=((x_matches-ctr[0])*x_matches+(y_matches-ctr[1])*y_matches+(z_matches-ctr[2])*z_matches)/((np.linalg.norm([x_matches,y_matches,z_matches],axis=0)*np.linalg.norm([x_matches-ctr[0],y_matches-ctr[1],z_matches-ctr[2]],axis=0)))
    
    scal[np.isnan(scal)]=1

    rays=np.exp(-np.sum(sampled,axis=1)*rad_res*scal)


    
    weights=(Rs[inds,argmin]**-2)/np.sum(Rs[inds,argmin]**-2)
    return(np.sum(rays*scal*weights))


def sum_over_rays_angle(field,ctr,r200,rad_res,X_circ,Y_circ,Z_circ):

    """
    compute sum over rays centred at ctr using given resolution
    field is a box to sum

    This version takes centre and sph points and computes paths (instead of
    taking paths by ref change), this ensures that all measurements are at
    the same positions on the r200 sphere


    Euh .... this seems not to work or to give weird fesc results ... Does it though ???

    """

    #assert False, "this is broken ... makes fescs increase with mass ... ???"
    
    size=np.shape(field)[0]

    box_ctr=np.asarray([0.5*size]*3)
    
    ctr=np.asarray(ctr)

    
    delta_R=np.copy(ctr)-box_ctr


    X_primes,Y_primes,Z_primes=[X_circ,Y_circ,Z_circ]+delta_R[:,np.newaxis]
    #These are pos on sph viewed from studied cell centre

    Rs,Phis,Thes=cart_2_sph(X_primes,Y_primes,Z_primes)

    #get longest path
    longest=int(np.ceil(np.max(Rs)/rad_res))+1

    rays=np.zeros((len(X_primes),longest))

    # #print(np.shape(rays))
    # plt.figure(figsize=(10,10))
    # plt.subplot(111)
    # plt.title("")
    # plt.grid(True)
    # plt.imshow(np.log10(field[int(box_ctr[2]),:,:]*rad_res),origin='lower',extent=[0,size,0,size])
    # plt.colorbar()

    # p=pat.Circle((0.5*size,0.5*size),r200,color="magenta",fill=False,edgecolor='k',linewidth=2)
    # axis=plt.gca()
    # axis.add_patch(p)





    # plt.scatter(X_primes,Y_primes,c="cyan",alpha=0.5,s=0.2)
    # plt.scatter(X_circ,Y_circ,c="k",alpha=0.5,s=0.4)

    for i_path,(R,X_prime,Y_prime,Z_prime) in enumerate(zip(Rs,X_primes,Y_primes,Z_primes)):

            #print(i_path)
            R_vect=np.arange(0,R+rad_res,rad_res)
            Phi=np.arctan2(Y_prime,X_prime)
            The=np.arccos(Z_prime/R)+np.pi
            lr=len(R_vect)
            X_paths,Y_paths,Z_paths=(np.asarray(sph_2_cart(R_vect,Phi,The))+ctr[:,np.newaxis])

            #plt.scatter(X_paths,Y_paths,c="brown",alpha=0.5,s=0.2)

            rays[i_path,:lr]=field[map(np.int32,[X_paths,Y_paths,Z_paths])]
            #print(rays[i_path,:lr])


            
    #plt.scatter(ctr[0],ctr[1],c='r')                              


    scal=((X_primes*X_circ)+(Y_primes*Y_circ)+(Z_primes*Z_circ))/np.linalg.norm([X_circ,Y_circ,Z_circ],axis=0)/Rs
    
    scal[np.isnan(scal)]=1


    rays_int=np.exp(-np.sum(rays,axis=1)*rad_res*scal)


    

    #print(rays_int,scal)
    #print(np.min(rays_int),np.min(rays_int*scal))
    
    # # #    plt.scatter(Xs[OOB],Ys[OOB],c="red",alpha=0.5,marker="x")
    #plt.show()
    #print(rays_int,scal)

    return(rays_int)






def cumsum_over_rays_angle(field,ctr,r200,rad_res,X_circ,Y_circ,Z_circ):

    """
    compute cum over rays centred at ctr using given resolution
    field is a box to sum

    This version takes centre and sph points and computes paths (instead of
    taking paths by ref change), this ensures that all measurements are at
    the same positions on the r200 sphere

    Euh .... this seems not to work or to give weird fesc results ... Does is though ...

    """

    #assert False, "this is broken ... makes fescs increase with mass ... ???"    

    
    size=np.shape(field)[0]

    box_ctr=np.asarray([0.5*size]*3)
    
    ctr=np.asarray(ctr)

    
    delta_R=np.copy(ctr)-box_ctr


    X_primes,Y_primes,Z_primes=[X_circ,Y_circ,Z_circ]+delta_R[:,np.newaxis]
    #These are pos on sph viewed from studied cell centre

    Rs,Phis,Thes=cart_2_sph(X_primes,Y_primes,Z_primes)

    #get longest path
    longest=int(np.ceil(np.max(Rs)/rad_res))+1

    rays=np.zeros((len(X_primes),longest))

    # #print(np.shape(rays))
    # plt.figure(figsize=(10,10))
    # plt.subplot(111)
    # plt.title("")
    # plt.grid(True)
    # plt.imshow(np.log10(field[int(box_ctr[2]),:,:]*rad_res),origin='lower',extent=[0,size,0,size])
    # plt.colorbar()

    # p=pat.Circle((0.5*size,0.5*size),r200,color="magenta",fill=False,edgecolor='k',linewidth=2)
    # axis=plt.gca()
    # axis.add_patch(p)





    # plt.scatter(X_primes,Y_primes,c="cyan",alpha=0.5,s=0.2)
    # plt.scatter(X_circ,Y_circ,c="k",alpha=0.5,s=0.4)

    for i_path,(R,X_prime,Y_prime,Z_prime) in enumerate(zip(Rs,X_primes,Y_primes,Z_primes)):

            #print(i_path)
            R_vect=np.arange(0,R+rad_res,rad_res)
            Phi=np.arctan2(Y_prime,X_prime)
            The=np.arccos(Z_prime/R)+np.pi
            lr=len(R_vect)
            X_paths,Y_paths,Z_paths=(np.asarray(sph_2_cart(R_vect,Phi,The))+ctr[:,np.newaxis])

            #plt.scatter(X_paths,Y_paths,c="brown",alpha=0.5,s=0.2)

            rays[i_path,:lr]=field[map(np.int32,[X_paths,Y_paths,Z_paths])]
            #print(rays[i_path,:lr])


            
    #plt.scatter(ctr[0],ctr[1],c='r')                              





    scal=((X_primes*X_circ)+(Y_primes*Y_circ)+(Z_primes*Z_circ))/np.linalg.norm([X_circ,Y_circ,Z_circ],axis=0)/Rs
    
    scal[np.isnan(scal)]=1


    rays_int=np.exp(np.cumsum(-rays*rad_res,axis=1)*scal[:,np.newaxis]) #cumulative sum of optical depths along the ray's axis
    

    #print(rays_int,scal)

    
    # # #    plt.scatter(Xs[OOB],Ys[OOB],c="red",alpha=0.5,marker="x")
    #plt.show()
    #print(rays_int,scal)

    return(rays_int)


def sum_over_rays_nexp(field,ctr,r200,rad_res,X_primes,Y_primes,Z_primes):

    """
    compute sum over rays centred at ctr using given resolution
    field is a box to sum

    """

    size=np.shape(field)[0]

    ctr=np.asarray(ctr)-0.5*size
    delta_R=np.copy(ctr)
    Xs,Ys,Zs=[X_primes,Y_primes,Z_primes]+delta_R[:,np.newaxis,np.newaxis]


    
    Rs=np.linalg.norm([Xs,Ys,Zs],axis=0)


    #used for getting data ... Need to be between 0 and size !!!
    Xs_snap,Ys_snap,Zs_snap=np.int32([Xs+0.5*size,Ys+0.5*size,Zs+0.5*size])

    
    
    IB=Rs<=r200 #if points in r200 ...
    OOB=~IB

    
    #print(np.shape(OOB),np.shape(sampled))
    sampled=np.zeros_like(Xs)
    sampled[IB]=field[Xs_snap[IB],Ys_snap[IB],Zs_snap[IB]]

    Rs=np.linalg.norm([Xs,Ys,Zs],axis=0)

    #if paths
    x_matches=np.zeros((np.shape(Xs)[0]),dtype=np.float32)
    y_matches=np.copy(x_matches)
    z_matches=np.copy(y_matches)

    
    argmin=np.argmin(np.abs(Rs[:,:]-r200),axis=1)


    inds=np.arange(len(x_matches))
    x_matches[inds]=Xs[inds,argmin]
    y_matches[inds]=Ys[inds,argmin]
    z_matches[inds]=Zs[inds,argmin]

    #if norms are 0 then we are on the border and the result should be 1
    #scal[np.isnan(scal)]=1

    scal=((x_matches-ctr[0])*x_matches+(y_matches-ctr[1])*y_matches+(z_matches-ctr[2])*z_matches)/((np.linalg.norm([x_matches,y_matches,z_matches],axis=0)*np.linalg.norm([x_matches-ctr[0],y_matches-ctr[1],z_matches-ctr[2]],axis=0)))
    
    scal[np.isnan(scal)]=1

    
    rays=np.sum(sampled,axis=1)*rad_res*scal


    return(rays)


    


def sum_at_r200(field,r200,Xvects,Yvects,Zvects):

    """
    sum over surfaces at Xs,Ys,Zs
    """

    
    
    size=max(np.shape(field))

    ctr=[int(0.5*size)+0.5]*3
    
    Xs,Ys,Zs=Xvects*r200,Yvects*r200,Zvects*r200
    
    #used for getting data ... Need to be between 0 and size !!!
    Xs_snap,Ys_snap,Zs_snap=np.int32([Xs+ctr[0],Ys+ctr[1],Zs+ctr[2]])

    # print(size,ctr)
    
    # print(Xs,Xs_snap)
    
    sampled=field[:,Xs_snap,Ys_snap,Zs_snap]

    normal_proj=Xvects*sampled[2] + Yvects*sampled[1] + Zvects*sampled[0]

    # print(r200,size,np.shape(sampled))
    
    #print(normal_proj)

    # plt.figure()

    # plt.imshow((field[0,int(0.5*size),:,:]),origin='lower',extent=[0,size,0,size])
    # plt.colorbar()

    # plt.quiver(Xs+0.5*size,Ys+0.5*size,Xvects,Yvects,color='r')          
    # plt.quiver(Xs+0.5*size,Ys+0.5*size,sampled[2],sampled[1],color='k')
    # plt.quiver(Xs+0.5*size,Ys+0.5*size,normal_proj*Xvects,normal_proj*Yvects,color='magenta')

    # print(np.su(mnormal_proj))
    
    # plt.show()
    
    return(np.sum(normal_proj))


def get_SFR(stars,age_lim):
    """
    Return avg SFR (msol.yr^-1) over last "age_lim" Myrs
    stars is rows of (mass,x,y,z,age)
    """

    underage_stars = stars[:,-1]<age_lim
    formd_mass=0
    
    if np.sum(underage_stars)>0:
        formd_mass = stars[underage_stars,0]

    
    return(np.sum(formd_mass)/(age_lim*1e6))

 
def get_mask(size,ctr,r200):
    """
    From radius and size, give spherical mask centered on ctr
    """

    vect = np.arange(0,size)
    xvect=np.abs(vect)
    yvect=np.abs(vect[:,np.newaxis])
    zvect=np.abs(yvect[:,:,np.newaxis])
    
    return((np.sqrt((xvect-ctr[0])**2+(yvect-ctr[1])**2+(zvect-ctr[2])**2)>r200))


def get_clock(time):

    hh=int(time/3600)
    mm=int(time/60)-60*hh
    ss=int(time)-mm*60-hh*3600

    return([str(hh)+':'+str(mm)+':'+str(ss)])



def get_side_surface(r_px,px_to_m):
    """
    Create surface matrix for one side of the cube
    """

    r_px=r_px
    side=r_px*2
    surfaces = np.ones((side,side))*(px_to_m)**2 #Initialise all pxs of a side as full pxs

    
    return(surfaces)    
    

def get_semissivity(stars,side,tstep,ctr):
    """
    From stars return stars' propreties and codaii style emissivities
    """

    
    

    #Fraction -> adjust for amount of emissive time relative to discrete time steps
    fractions=np.zeros(np.shape(stars[:,0]))


    #print(tstep,stars[:,-2])
    #print(fractions)
    # newly_departed = ((stars[:,-2])<tstep+10)*(stars[:,-2]>10)
    # fractions[newly_departed] = np.minimum((stars[newly_departed,-2]-10)/tstep,np.ones(np.shape(np.sum(newly_departed))))

    new_borns = (stars[:,-2])<tstep 
    fractions[new_borns]=(stars[new_borns,-2])/tstep

    active_or_newly_departed = ((stars[:,-2])<10+tstep)*(stars[:,-2]>=tstep)
    fractions[active_or_newly_departed] = np.minimum((10-stars[active_or_newly_departed,-2]+tstep)/tstep,np.ones(np.shape(np.sum(active_or_newly_departed))))
    
    
    # old_gits = stars[:,-2]>10+tstep
    # fractions[old_gits] = np.zeros(np.shape(np.sum(old_gits)))

    #print(fractions)
    #Get young
    #stars=stars[stars[:,-2]<=10]


    remnants = fractions == 0
    youths = fractions != 0
    stars_ctrs = 4096*stars[:,1:4]-np.transpose(np.tile(ctr[:,np.newaxis],len(stars)))
    radii = np.linalg.norm(stars_ctrs,2,axis=1)


        
    #Youngest + Oldest
    if len(stars)!=0 :
        old = max(stars[:,-2])        
        yng = min(stars[:,-2])
    else:
        yng=-1
        old=-1

    nb_yng_star=np.sum(fractions!=0)
    yng_mass=np.sum(stars[ fractions!=0,0])
    
    return(E0*np.sum(stars[:,0]*fractions)*fesc_star,
           fractions,
           nb_yng_star,
           old,yng,
           yng_mass)


def write_figure(desired_name,figure_to_write):

    '''
    Each prog has its folder with dated folders containing output
    If cur launch of prog is dif than latest archived version, copy over new archived version
    '''
    
    archive_path = "/data2/jlewis/plot_archive"
    prog_name = sys.argv[0]
    #catch non . calls
    prog_name = prog_name.split('/')[-1]

    arch_prog_path = os.path.join(archive_path,prog_name)

    today_date = date.today().isoformat()
    
    arch_prog_path_today = os.path.join(arch_prog_path,today_date)
    
    if not os.path.isdir(arch_prog_path_today) : os.makedirs(arch_prog_path_today)

    run_dir = os.path.dirname(os.path.realpath(__file__))
    exec_version = os.path.join(run_dir,prog_name)

    
    #get latest
    arch_versions = [name for name in os.listdir(arch_prog_path) if prog_name in name]

    if len(arch_versions)>0 : 


        
        arch_version_dates_ints = [map(int,name.split('_')[0].split('-')) for name in arch_versions]
        arch_version_dates = [date(date_int[0],date_int[1],date_int[2]) for date_int in arch_version_dates_ints]


        i=0
        imax=0

        for i in range(len(arch_version_dates)):
            if arch_version_dates[i]>arch_version_dates[imax] : imax = i



        latest_arch_version = os.path.join(arch_prog_path,arch_versions[imax])

        if not filecmp.cmp(latest_arch_version,exec_version) : copyfile(exec_version,os.path.join(arch_prog_path,today_date+'_'+prog_name))

    else: copyfile(exec_version,os.path.join(arch_prog_path,today_date+'_'+prog_name))
    
    figure_to_write.savefig(os.path.join(arch_prog_path_today,desired_name+'.png'),bbox_inches='tight',format='png')
    
def setup_plotting(scale_up=1.5):



    plt.rcParams['ytick.major.size']=15*scale_up
    plt.rcParams['ytick.major.width']=1.5*scale_up
    plt.rcParams['ytick.minor.size']=10*scale_up
    plt.rcParams['ytick.minor.width']=1*scale_up
    plt.rcParams['xtick.major.size']=15*scale_up
    plt.rcParams['xtick.major.width']=1.5*scale_up
    plt.rcParams['xtick.minor.size']=10*scale_up
    plt.rcParams['xtick.minor.width']=1*scale_up
    plt.rcParams['axes.linewidth']=1.75*scale_up
    plt.rcParams['font.size']= 16*scale_up
    #plt.rcParams['figure.dpi']= 200 
    plt.rcParams['figure.figsize']= (10,10)

def print_delimiter(text=""):

    import os
    rows, width = np.int32(os.popen('stty size', 'r').read().split())

    print("*"*width+"\n")
    print(" "*(int((width-len(text))*0.5))+"%s\n"%text)
    print("*"*width+"\n")    
    
def make_figure(*args,**kwargs):


    fig=plt.figure(figsize=(10,10),*args,**kwargs)
    ax=fig.add_subplot(111)

    return(fig,ax)



"""
Functions for dealing with periodic boundaries
"""


def get_27(pos1,pos2,pos_vects):
    """
    Returns all distances including pos_vects reflections
    check that pos_vects and pos are same units !!!!
    """
    
    #return((np.linalg.norm(pos1-(np.asarray(pos_vects)+np.asarray(pos2)[:,np.newaxis]),axis=2,ord=2)))
    return((np.linalg.norm(pos1-(np.asarray(pos_vects)+np.asarray(pos2)),axis=1,ord=2)))

def get_mult_27(pos1,pos2,pos_vects):
    """
    Returns all distances including pos_vects reflections
    check that pos_vects and pos are same units !!!!
    """
    
    return((np.linalg.norm(pos1-(np.asarray(pos_vects)+np.asarray(pos2)[:,np.newaxis]),axis=2,ord=2)))


        
# def get_27_tree(pos,tree,dist,pos_vects):
#     """
#     Return all within dist of pos, accounting for edge repetitions listed in pos_vects
#     check that pos_vects and pos are same units !!!!
#     """
#     dists=get_27(ctr,pos,pos_vects) #allows to pick the reflections that give the closest positions
#     whs=tree.query_ball_point(pos+np.asarray(pos_vects)[np.argmin(dists,axis=1),dist)           
#     return(whs.tolist())

def get_27_tree(pos,tree,dist,pos_vects):
    """
    Return all within dist of pos, accounting for edge repetitions listed in pos_vects
    """
    whs = np.array([])
    for pos_vect in pos_vects :
         whs=np.union1d(whs,tree.query_ball_point(pos+pos_vect,dist))
         
    return(whs.tolist())

                              
def get_27_box(pos,ctr,dist,pos_vects):
    """
    Return all within dist of pos, accounting for edge repetitions listed in pos_vects
    check that pos_vects and pos are same units !!!!
    """
    dists=get_mult_27(ctr,pos,pos_vects)
    whs=catch_box(pos+np.asarray(pos_vects)[np.argmin(dists,axis=1)],ctr,dist)
         
    return(whs)


def get_27_tree_nearest(pos,tree,pos_vects):
    """
    Return all within dist of pos, accounting for edge repetitions listed in pos_vects
    check that pos_vects and pos are same units !!!!
    """
    whs_dists,whs_inds=tree.query(pos+pos_vects,1,p=2)
         
    return(whs_inds[np.argmin(whs_dists)])

def get_ctr(poss,pos_vects):


    prec = 5 #in nb of cells
    shp = np.ones(np.shape(poss))
    if np.any([poss<prec*shp,poss>(512-prec)*shp]):

        ctr=np.median(poss,axis=0)
        
        all_dists = get_27(ctr,poss,pos_vects)
        min_arg = np.argmin(all_dists,axis=1)
        ctr = np.average(poss+np.asarray(pos_vects)[min_arg],axis=0)


    else:
         ctr = np.average(poss,axis=0)
        
    return(ctr)
