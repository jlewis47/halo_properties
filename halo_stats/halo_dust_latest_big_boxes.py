'''

'''


import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
# import matplotlib.patches as pat
# from tionread_radgpu import o_rad_cube_big
# from tionread_stars import read_all_star_files
# from scipy import spatial
import time
# import string
import argparse
import os
from association.read_assoc_latest import *
from files.read_fullbox_big import *
from utils.functions_latest import *
import healpy as hp
from dust.dust_opacity import *
from params.params import *
from files.wrap_boxes import *
from utils.output_paths import *
from utils.utils import get_r200_suffix, get_fof_suffix, get_suffix



def compute_dust(out_nb,ldx,path,sim_name,overwrite=False,use_fof=False,rtwo_fact=1,rel_fof_path=None):

        fof_suffix=get_fof_suffix(fof_path)
        rtwo_suffix=get_r200_suffix(rtwo_fact)
        suffix=get_suffix(fof_suffix,rtwo_suffix)
                
        info_path=os.path.join(path,'output_00'+out_nb)
        #data_pth_fullres=os.path.join(path,'data_cubes','output_00'+out_nb)
        data_pth_fullres=os.path.join(path,'output_00'+out_nb)
        phew_path=os.path.join(path,'output_00'+out_nb)
        data_pth_assoc=os.path.join(assoc_path,sim_name,'assoc'+out_nb)

        plt.rcParams.update({'font.size': 18})   


        #find number of subcubes
        data_files=os.listdir(data_pth_fullres)
        rho_files=[f for f in data_files if 'rho' in f]

        n_subcubes=len(rho_files)
        subs_per_side=int(n_subcubes**(1./3))

        overstep=1.1
        
        sub_side=int(float(ldx)/subs_per_side)



        
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

        dist_obs=345540.98618977674 #distance to obs point from box (0,0,0); in number of cells for z=6


        #mags,xis,contbetalow,contbetahigh,beta,metal_bins,age_bins=get_mag_tab_BPASSV221_betas()

        upper=27
        grid = np.mgrid[0:upper,0:upper,0:upper]/float(upper-1)

        print('Getting Phew and stars')
        idxs,star_idxs,phew_tot_star_nb,phew_star_nb,phew_tab,stars,lone_stars = read_assoc(out_nb,sim_name,rtwo_fact,rel_fof_path)




        prog_bar0=['[',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',']']
        prog_bar=prog_bar0
        nb_halo = np.shape(phew_tab)[0]

        ran = [-ldx,0,ldx]
        pos_vects = np.asarray([[i,j,k] for k in ran for j in ran for i in ran])

        phew_sumd_fl,phew_nb_cells = np.zeros(3),np.zeros(3)


        #Converstion factors
        rho_fact=unit_d #g/cm**3
        dust_fact=unit_d #g/cm^3
        vvrho_fact=1e6*1e3*1e-2*unit_l/unit_t/1e3 #m/s #H/m**2/s
        temp_fact=Pmass*(1e-2*unit_l/unit_t)**2/Bmann #K
        fl_fact=(a**-3.) #m**-2*s**-1


        stt = time.time()



        nside=8 #2 or 4 else wise too costly
        npix=hp.nside2npix(nside)
        pix=np.arange(npix)
        thes,phis=hp.pix2ang(nside,pix)
        Xvects,Yvects,Zvects=hp.pix2vec(nside,pix) #with radius = 1

        hp_surf_elem=4.*np.pi*px_to_m**2 #in m^2.dx^-2


        #print(len(phis))

        rad_res=0.1 #in units of dx

        xbins=np.arange(8,12.1,0.05)
        ybins=np.arange(-2,0.1,0.05)
        nbs_tot=np.zeros((len(xbins)-1,len(ybins)-1))




        #pre-allocate boxes for data cubes
        big_side=int(sub_side*overstep)
        big_rho=np.zeros((big_side,big_side,big_side),dtype=np.float32)        
        big_dust=np.zeros((big_side,big_side,big_side),dtype=np.float32)
        big_metals=np.zeros((big_side,big_side,big_side),dtype=np.float32)
        big_xtion=np.zeros((big_side,big_side,big_side),dtype=np.float32)        
        
        if overwrite :
                print('Overwriting existing output files')
        else:
                print('Skipping existing files')
        
        
        for x_subnb in range(subs_per_side):
                for y_subnb in range(subs_per_side):
                        for z_subnb in range(subs_per_side):                

                                subcube_nb=x_subnb+y_subnb*subs_per_side+z_subnb*subs_per_side**2.

                                out_file=os.path.join(out,'gas_dust_out_%s_'%suffix+out_nb+'_%i'%subcube_nb)

                                out_exists=os.path.exists(out_file)

                                if out_exists and not overwrite :

                                        print('Skipping subcube #%i since it already exists'%subcube_nb)
                                        continue
                                
                                print('Reading subcube #%s' %(1+subcube_nb))

                                #Retain halos within sub cube
                                x_cond = np.all([phew_tab[:,1]*ldx<=(x_subnb+1)*sub_side,phew_tab[:,1]*ldx>(x_subnb)*sub_side],axis=0)
                                y_cond = np.all([phew_tab[:,2]*ldx<=(y_subnb+1)*sub_side,phew_tab[:,2]*ldx>(y_subnb)*sub_side],axis=0)
                                z_cond = np.all([phew_tab[:,3]*ldx<=(z_subnb+1)*sub_side,phew_tab[:,3]*ldx>(z_subnb)*sub_side],axis=0)



                                ind_subcube = x_cond*y_cond*z_cond

                                #print(np.sum(ind_subcube))

                                sub_phew_tab = phew_tab[ind_subcube,:]
                                sub_phew_tot_star_nb = phew_tot_star_nb[ind_subcube]
                                sub_phew_star_nb = phew_star_nb[ind_subcube]
                                sub_idxs = idxs[ind_subcube]

                                limit_r=phew_tab[ind_subcube,-1]+1
                                sample_r=do_half_round(limit_r)



                                

                                M,pos_nrmd = phew_tab[ind_subcube,0],(phew_tab[ind_subcube,1:4])
                                pos =do_half_round((ldx)*pos_nrmd) #was np.int16

                      
                                #(0,0,0) px locattion of sub_side**3 cube within whole data set
                                edge=np.asarray([x_subnb*sub_side,y_subnb*sub_side,z_subnb*sub_side])


                                
                                ctr_bxd=(pos-edge)

                                lower_bounds=np.int32(ctr_bxd-sample_r[:,np.newaxis])
                                upper_bounds=np.int32(ctr_bxd+sample_r[:,np.newaxis])


                                
                                under=lower_bounds<=0
                                over=upper_bounds>=sub_side-1

                                # under=np.ones_like(pos[:,0])!=1
                                # over=np.copy(under)
                                
                                outside=[under,over]

                                large_box=np.any(outside)

                                

                                
                                dust_mass=np.zeros((np.shape(sub_phew_tab)[0]),dtype=np.float64)
                                gas_mass=np.zeros((np.shape(sub_phew_tab)[0]),dtype=np.float64)
                                neutral_gas_mass=np.zeros((np.shape(sub_phew_tab)[0]),dtype=np.float64)
                                metal_mass=np.zeros((np.shape(sub_phew_tab)[0]),dtype=np.float64)

                                max_gas_dens=np.zeros((np.shape(sub_phew_tab)[0]),dtype=np.float64)
                                max_neutral_gas_dens=np.zeros((np.shape(sub_phew_tab)[0]),dtype=np.float64)
                                max_metal_dens=np.zeros((np.shape(sub_phew_tab)[0]),dtype=np.float64)
                                max_dust_dens=np.zeros((np.shape(sub_phew_tab)[0]),dtype=np.float64)

                                gasZmean=np.zeros((np.shape(sub_phew_tab)[0]),dtype=np.float64)
                                gasZmax=np.zeros((np.shape(sub_phew_tab)[0]),dtype=np.float64)

                                stellarZmax=np.zeros((np.shape(sub_phew_tab)[0]),dtype=np.float64)
                                stellarZmean=np.zeros((np.shape(sub_phew_tab)[0]),dtype=np.float64)


                                get_overstep_hydro_cubed(big_rho,subcube_nb,data_pth_fullres,'rho',outside,n_subcubes=n_subcubes,size=sub_side,overstep=overstep)                        
                                get_overstep_hydro_cubed(big_xtion,subcube_nb,data_pth_fullres,'xtion',outside,n_subcubes=n_subcubes,size=sub_side,overstep=overstep)                        
                                get_overstep_hydro_cubed(big_metals,subcube_nb,data_pth_fullres,'Z',outside,n_subcubes=n_subcubes,size=sub_side,overstep=overstep)                              
                                get_overstep_hydro_cubed(big_dust,subcube_nb,data_pth_fullres,'dust',outside,n_subcubes=n_subcubes,size=sub_side,overstep=overstep)

                                
                                #print(np.max(big_rho)*rho_fact)
                                
                                #big_rho=np.zeros((sub_side*overstep,sub_side*overstep,sub_side*overstep),dtype=np.float32)
                                # big_dust=np.zeros((sub_side*overstep,sub_side*overstep,sub_side*overstep),dtype=np.float32)
                                # big_xtion=np.zeros((sub_side*overstep,sub_side*overstep,sub_side*overstep),dtype=np.float32)
                                # big_metals=np.zeros((sub_side*overstep,sub_side*overstep,sub_side*overstep),dtype=np.float32)

                                ctr_bxd=ctr_bxd+int(sub_side*(overstep-1)*0.5)
                                lower_bounds=lower_bounds+int(sub_side*(overstep-1)*0.5)
                                upper_bounds=upper_bounds+int(sub_side*(overstep-1)*0.5)
                                    
                                #print(lower_bounds,upper_bounds)

                                #print(int(sub_side*(overstep-1)*0.5))
                                
                                for ind,halo in enumerate(sub_phew_tab):



                                    r_px=halo[-1]

                                    r=r_px*px_to_m #Convert to meters
                                    surf_elem=(hp_surf_elem*r_px**2/float(npix))







                                                                     
                                    sm_rho = big_rho[lower_bounds[ind,2]:upper_bounds[ind,2],lower_bounds[ind,1]:upper_bounds[ind,1],lower_bounds[ind,0]:upper_bounds[ind,0]]
                                    sm_dust = big_dust[lower_bounds[ind,2]:upper_bounds[ind,2],lower_bounds[ind,1]:upper_bounds[ind,1],lower_bounds[ind,0]:upper_bounds[ind,0]]
                                    sm_metals = big_metals[lower_bounds[ind,2]:upper_bounds[ind,2],lower_bounds[ind,1]:upper_bounds[ind,1],lower_bounds[ind,0]:upper_bounds[ind,0]]
                                    sm_xtion = big_xtion[lower_bounds[ind,2]:upper_bounds[ind,2],lower_bounds[ind,1]:upper_bounds[ind,1],lower_bounds[ind,0]:upper_bounds[ind,0]]
                                    sm_nHI=sm_rho*(1-sm_xtion)


                                    if sub_phew_star_nb[ind]>0 :


                                            #Get stars for halo from list of associated stars
                                            cur_stars=stars[sub_phew_tot_star_nb[ind]:sub_phew_tot_star_nb[ind]+sub_phew_star_nb[ind],:]


                                            fracd_stars = np.copy(cur_stars)
                                            fracd_stars[:,1:4] = np.asarray(fracd_stars[:,1:4])*(ldx)

                                            fracd_stars[:,1:4] += pos_vects[np.argmin(get_mult_27(pos[ind],fracd_stars[:,1:4],pos_vects),axis=1)]


                                            #basically we get the indices of stars for a posittion histogram
                                            sm_xgrid=np.arange(lower_bounds[ind,0],upper_bounds[ind,0],1)
                                            star_sm_posx=np.digitize(fracd_stars[:,1],sm_xgrid-int(sub_side*(overstep-1)*0.5)+edge[0])-1

                                            sm_ygrid=np.arange(lower_bounds[ind,1],upper_bounds[ind,1],1)
                                            star_sm_posy=np.digitize(fracd_stars[:,2],sm_ygrid-int(sub_side*(overstep-1)*0.5)+edge[1])-1

                                            sm_zgrid=np.arange(lower_bounds[ind,2],upper_bounds[ind,2],1)
                                            star_sm_posz=np.digitize(fracd_stars[:,3],sm_zgrid-int(sub_side*(overstep-1)*0.5)+edge[2])-1                      

                                            stellar_box_mass = np.zeros_like(sm_rho,dtype=np.float64)

                                            
                                            for istar,star in enumerate(fracd_stars):

                                                    stellar_box_mass[star_sm_posz[istar],star_sm_posy[istar],star_sm_posx[istar]]+=star[0]



                                            stellar_box_mass=stellar_box_mass/np.sum(stellar_box_mass) #weights ...

                                            stellarZmax[ind]=np.max(cur_stars[:,-1])*0.02
                                            stellarZmean[ind]=np.average(cur_stars[:,-1],weights=cur_stars[:,0])*0.02      

                                            dust_mass[ind]=np.sum(sm_dust)*(px_to_m*100)**3*1e-3/Msol*dust_fact
                                            metal_mass[ind]=np.sum(sm_metals*sm_rho)*(px_to_m*100)**3*1e-3/Msol*rho_fact


                                            max_dust_dens[ind]=np.max(sm_dust)*dust_fact
                                            max_metal_dens[ind]=np.max(sm_metals*sm_rho)*rho_fact


                                            gasZmean[ind]=np.average(sm_metals,weights=stellar_box_mass)
                                            gasZmax[ind]=np.max(sm_metals)            



                                            
                                            #if halo[0]>5e9:

                                                    #print(map(np.log10,[dust_mass[ind],np.sum(cur_stars[:,0]),metal_mass[ind],np.sum(sm_nHI)*(px_to_m*100)**3*1e-3/Msol*rho_fact]))




                                    max_neutral_gas_dens[ind]=np.max(sm_nHI)*rho_fact
                                    max_gas_dens[ind]=np.max(sm_rho)*rho_fact
                                    neutral_gas_mass[ind]=np.sum(sm_nHI)*(px_to_m*100)**3*1e-3/Msol*rho_fact
                                    gas_mass[ind]=np.sum(sm_rho)*(px_to_m*100)**3*1e-3/Msol*rho_fact




                                idx = np.copy(sub_idxs)
                                mass,x,y,z = np.transpose(sub_phew_tab[:,:-1])

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

                                with open(out_file,'wb') as newFile:
                                    np.save(newFile,np.int32(len(idx)))
                                    np.save(newFile,np.int32(len(dict_keys)))
                                    np.save(newFile,np.float64(a))
                                    np.save(newFile,dict_keys)
                                    np.save(newFile,file_bytes)



"""
Main body
"""


if __name__ =='__main__':

        Arg_parser = argparse.ArgumentParser('Associate stars and halos in full simulattion')

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
