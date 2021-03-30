'''
Gather some dust and gas data from haloes in a simulation
metal stuff is in absolute metallicity
'''


import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as pat
#from read_radgpu import o_rad_cube_big
from read_stars import read_stars
from scipy import spatial
import time
import string
import argparse
import os
from read_assoc_latest import *
from read_fullbox_big import *
from functions_latest import *
import healpy as hp
from dust_opacity import *
#from wrap_boxes import *
from output_paths import *



def compute_dust(out_nb,ldx,path,sim_name,use_fof=False):


        fof_suffix=''
        if use_fof:fof_suffix='fof'        
        
        info_path=os.path.join(path,'output_00'+out_nb)
        data_pth_fullres=path
        phew_path=os.path.join(path,'output_00'+out_nb)
        data_pth_assoc=os.path.join(assoc_path,sim_name,'assoc'+out_nb)

        plt.rcParams.update({'font.size': 18})   




        out = os.path.join(analy_path,sim_name)

        #Get scale factor and co
        '''Get scale factor and co'''
        (t,a,H0,om_m,om_l,om_k,om_b,unit_l,unit_d,unit_t,l,Lco,L,px_to_m) = get_infos_no_t(info_path,
                                                                                           out_nb,
                                                                                           ldx)
        tstep=0 #temp need this >0!

        with open(os.path.join(out,'Mp'), 'rb') as mass_file :
            Mp = np.fromfile(mass_file,dtype=np.float64)
        print(Mp)


        print('Getting Phew and stars')
        idxs,star_idxs,phew_tot_star_nb,phew_star_nb,phew_tab,stars,lone_stars = read_assoc(out_nb,sim_name,use_fof)




        prog_bar0=['[',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',']']
        prog_bar=prog_bar0
        nb_halo = np.shape(phew_tab)[0]


        ran = [-ldx,0,ldx]
        pos_vects = np.asarray([[i,j,k] for k in ran for j in ran for i in ran])

        
        #Conversion factors
        tau_fact = px_to_m*sigma_UV*1e6*0.76
        rho_fact=unit_d #g/cm^3
        dust_fact=unit_d #g/cm^3
        vvrho_fact=1e6*1e3*1e-2*unit_l/unit_t/1e3 #m/s #H/m**2/s
        temp_fact=Pmass*(1e-2*unit_l/unit_t)**2/Bmann #K
        fl_fact=(a**-3.) #m**-2*s**-1


        stt = time.time()


        limit_r=phew_tab[:,-1]+1
        sample_r=do_half_round(limit_r)


        M,pos_nrmd = phew_tab[:,0],(phew_tab[:,1:4])
        pos =np.asarray(do_half_round(ldx*pos_nrmd)) #was np.int16

        #(0,0,0) px location of 512**3 cube within whole data set


        ctr_bxd=pos

        lower_bounds=np.int32(ctr_bxd-sample_r[:,np.newaxis])
        upper_bounds=np.int32(ctr_bxd+sample_r[:,np.newaxis])

        under=lower_bounds<=0
        over=upper_bounds>=511

        outside=[under,over]

        large_box=np.any(outside)



        #Get data
        #fl= o_rad_cube_big(data_pth_rad,2,f_side,subcube_nb)
        # if tst :nrg = o_rad_cube_big(data_pth_rad,1,f_side,subcube_nb)
        xion=o_data(os.path.join(data_pth_fullres,'xion_00'+out_nb))
        rho=o_data(os.path.join(data_pth_fullres,'rho_00'+out_nb))
        dust=o_data(os.path.join(data_pth_fullres,'dust_00'+out_nb))
        metals=o_data(os.path.join(data_pth_fullres,'Z_00'+out_nb))
        # temp =  o_fullbox_big(data_pth_fullres,'temp',512,512,subcube_nb)
        # temp = temp/(rho*(1.+xion))
        #rho = rho
        # vx =  o_fullbox_big(data_pth_fullres,'vx',512,512,subcube_nb)
        # vy =  o_fullbox_big(data_pth_fullres,'vy',512,512,subcube_nb)
        # vz =  o_fullbox_big(data_pth_fullres,'vz',512,512,subcube_nb)    
        # vv = np.asarray([vx,vy,vz]) 
        # vvrho=vv*rho



        dilate_fact=1.1
        rescale=(dilate_fact-1)/2.
        delta=int(rescale*ldx)

        #=do_half_round(pos_nrmd*ldx+delta)


        lower_bounds=np.int32(pos-sample_r[:,np.newaxis])+delta
        upper_bounds=np.int32(pos+sample_r[:,np.newaxis])+delta
        ctr_bxd=ctr_bxd+delta
        ctr_big_box=np.int16(pos+delta)

        big_xion=np.zeros((int(ldx*dilate_fact),int(ldx*dilate_fact),int(ldx*dilate_fact)))
        big_rho=np.zeros((int(ldx*dilate_fact),int(ldx*dilate_fact),int(ldx*dilate_fact)))                 
        big_metals=np.zeros((int(ldx*dilate_fact),int(ldx*dilate_fact),int(ldx*dilate_fact)))
        big_dust=np.zeros((int(ldx*dilate_fact),int(ldx*dilate_fact),int(ldx*dilate_fact)))                 

        centre_slice=slice(int(rescale*ldx),int((1+rescale)*ldx))

        big_rho[centre_slice,centre_slice,centre_slice]=rho
        big_xion[centre_slice,centre_slice,centre_slice]=xion
        big_metals[centre_slice,centre_slice,centre_slice]=metals
        big_dust[centre_slice,centre_slice,centre_slice]=dust

        #big_dust=big_dust #g/cm^3


        sides=[-1,0,1]

        def put_and_grab(ind,delta,dim):
            """
            Ind gives pos near side of box
            side of larger box is smaller box side +2*delta
            Returns slices necessary to repeat box sides in larger array
            """


            if ind==0 :
                put=slice(delta,dim+delta)
                grab=slice(0,dim)
            elif ind==1 :
                put=slice(dim+delta,dim+2*delta)
                grab=slice(0,delta)
            elif ind==-1 :
                put=slice(0,delta)
                grab=slice(dim-delta,dim)

            return(grab,put)

        for i in sides:
            for j in sides:
                for k in sides: 

                    xgrab,xput=put_and_grab(i,delta,ldx)
                    ygrab,yput=put_and_grab(j,delta,ldx)
                    zgrab,zput=put_and_grab(k,delta,ldx)            


                    big_rho[xput,yput,zput]=rho[xgrab,ygrab,zgrab]
                    big_xion[xput,yput,zput]=xion[xgrab,ygrab,zgrab]
                    big_metals[xput,yput,zput]=metals[xgrab,ygrab,zgrab]            
                    big_dust[xput,yput,zput]=dust[xgrab,ygrab,zgrab]                        





        dust_mass=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)
        gas_mass=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)
        neutral_gas_mass=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)
        metal_mass=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)

        max_gas_dens=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)
        max_neutral_gas_dens=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)
        max_metal_dens=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)
        max_dust_dens=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)

        gasZmean=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)
        gasZmax=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)
        
        stellarZmax=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)
        stellarZmean=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)
        

        for ind,halo in enumerate(phew_tab):

            sm_rho = big_rho[lower_bounds[ind,2]:upper_bounds[ind,2],lower_bounds[ind,1]:upper_bounds[ind,1],lower_bounds[ind,0]:upper_bounds[ind,0]]
            sm_dust = big_dust[lower_bounds[ind,2]:upper_bounds[ind,2],lower_bounds[ind,1]:upper_bounds[ind,1],lower_bounds[ind,0]:upper_bounds[ind,0]]
            sm_metals = big_metals[lower_bounds[ind,2]:upper_bounds[ind,2],lower_bounds[ind,1]:upper_bounds[ind,1],lower_bounds[ind,0]:upper_bounds[ind,0]]
            sm_xion = big_xion[lower_bounds[ind,2]:upper_bounds[ind,2],lower_bounds[ind,1]:upper_bounds[ind,1],lower_bounds[ind,0]:upper_bounds[ind,0]]
            sm_nHI=sm_rho*(1-sm_xion)


            if phew_star_nb[ind]>0:

                    cur_stars=stars[phew_tot_star_nb[ind]:phew_tot_star_nb[ind]+phew_star_nb[ind],:]


                    fracd_stars=np.copy(cur_stars)
                    fracd_stars[:,1:4] = np.asarray(cur_stars[:,1:4]*(ldx)+delta)
                    #Need to be careful when star is on the other side of te box when compared to halo centre


                    fracd_stars[:,1:4] += pos_vects[np.argmin(get_mult_27(pos[ind],fracd_stars[:,1:4]-delta,pos_vects),axis=1)]
         
                    
                    sm_xgrid=np.arange(lower_bounds[ind,0],upper_bounds[ind,0],1)
                    star_sm_posx=np.digitize(fracd_stars[:,1],sm_xgrid)-1

                    sm_ygrid=np.arange(lower_bounds[ind,1],upper_bounds[ind,1],1)
                    star_sm_posy=np.digitize(fracd_stars[:,2],sm_ygrid)-1
                    
                    sm_zgrid=np.arange(lower_bounds[ind,2],upper_bounds[ind,2],1)
                    star_sm_posz=np.digitize(fracd_stars[:,3],sm_zgrid)-1                      

                    stellar_box_mass = np.zeros_like(sm_rho,dtype=np.float64)
                    
                    for istar,star in enumerate(fracd_stars):

        
                            stellar_box_mass[star_sm_posz[istar],star_sm_posy[istar],star_sm_posx[istar]]+=star[0]


                    # print(np.max(stellar_box_mass),sm_metals[np.max(stellar_box_mass)==stellar_box_mass])
                    # print(np.max(sm_metals),stellar_box_mass[np.max(sm_metals)==sm_metals])

                    #print(stellar_box_mass,sm_metals,sm_rho)
                    
                    stellar_box_mass=stellar_box_mass/np.sum(stellar_box_mass) #weights ...
                    
                    stellarZmax[ind]=np.max(cur_stars[:,-1])*0.02
                    stellarZmean[ind]=np.average(cur_stars[:,-1],weights=cur_stars[:,0])*0.02      

                    dust_mass[ind]=np.sum(sm_dust)*(px_to_m*100)**3*1e-3/Msol*dust_fact
                    metal_mass[ind]=np.sum(sm_metals*sm_rho)*(px_to_m*100)**3*1e-3/Msol*rho_fact


                    max_dust_dens[ind]=np.max(sm_dust)*dust_fact
                    max_metal_dens[ind]=np.max(sm_metals*sm_rho)*rho_fact


                    gasZmean[ind]=np.average(sm_metals,weights=stellar_box_mass)
                    gasZmax[ind]=np.max(sm_metals)            



                    


                    
                    
            max_neutral_gas_dens[ind]=np.max(sm_nHI)*rho_fact
            max_gas_dens[ind]=np.max(sm_rho)*rho_fact
            neutral_gas_mass[ind]=np.sum(sm_nHI)*(px_to_m*100)**3*1e-3/Msol*rho_fact
            gas_mass[ind]=np.sum(sm_rho)*(px_to_m*100)**3*1e-3/Msol*rho_fact



        idx = np.copy(idxs)
        mass,x,y,z = np.transpose(phew_tab[:,:-1])

        print('Writing data file')
        dict_keys=['idx','M','x','y','z','gas_mass','dust_mass','metal_mass','neutral_gas_mass','max_gas_dens','max_dust_dens','max_metal_dens','max_neutral_gas_dens','mean_stellar_Z','max_stellar_Z','max_gas_Z','mean_gas_Z']

        file_bytes = (np.transpose([idx,
                                    mass,
                                    x,
                                    y,
                                    z,
                                    gas_mass,
                                    dust_mass,
                                    metal_mass,
                                    neutral_gas_mass,
                                    max_gas_dens,
                                    max_dust_dens,
                                    max_metal_dens,
                                    max_neutral_gas_dens,
                                    stellarZmax,
                                    stellarZmean,
                                    gasZmax,
                                    gasZmean]))
                                    
        assert len(dict_keys)==len(np.transpose(file_bytes)), "mismatch between number of keys and number of data entries"

        with open(os.path.join(out,'gas_dust_%s_out_'%fof_suffix+out_nb+'_0'),'wb') as newFile:
            np.save(newFile,np.int32(len(idx)))
            np.save(newFile,np.int32(len(dict_keys)))
            np.save(newFile,np.float64(a))
            np.save(newFile,dict_keys)
            np.save(newFile,file_bytes)


"""
Main body
"""


if __name__ =='__main__':

        Arg_parser = argparse.ArgumentParser('Associate stars and halos in full simulation')

        Arg_parser.add_argument('nb',metavar='nsnap',type=str,help='snap number string, give as "XXX" so that 00001 is "001" and 00111 is "111"')
        Arg_parser.add_argument('ldx',metavar='ldx',type=str,help='box size in cells')
        Arg_parser.add_argument('path',metavar='path',type=str,help='path to sim')
        Arg_parser.add_argument('simname',metavar='simname',type=str,help='sim name (will be used to create dirs)')

        args = Arg_parser.parse_args()


        out_nb = args.nb
        ldx=int(args.ldx)
        path=args.path
        sim_name=args.simname

    
        compute_dust(out_nb,ldx,path,sim_name)
