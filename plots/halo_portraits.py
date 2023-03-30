'''
'''
from sre_constants import SUCCESS
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as pat
#from read_radgpu import o_rad_cube_big
# from tionread_stars import read_all_star_files
# from scipy import spatial
# import time
# import string
import argparse
import os
from association.read_assoc_latest import read_assoc, read_specific_stars
from files.read_fullbox_big import *
from utils.functions_latest import *
# from src.bpass_fcts import get_mag_tab_BPASSV221_betas, get_mag_interp_fct, get_xis_interp_fct, get_star_mags_metals, get_star_xis_metals
#from ray_fcts import sph_2_cart, cart_2_sph, sum_over_rays_bias

from time import sleep
#import healpy as hp
#from dust_opacity import *
from files.wrap_boxes import *
from utils.output_paths import *
from mpi4py import MPI
from utils.units import get_unit_facts
from utils.utils import divide_task,ll_to_fof_suffix,get_r200_suffix,get_suffix
from params.params import *
from plot_functions import make_figure

def compute_fesc(out_nb,overwrite=False,rtwo_fact=1,ll=200,assoc_mthd=''):

        overstep=1.2 #important !!! For overstep=1.2 instead of loading a subbox of say 512 cells per side
        #we add edges based on the repetittion/full extent of the simulattion so that we actaully work
        #with a box that has 512*1.2 cells per side.
        #this is done to handle haloes that are on the edges of sub boxes without exploding memory costs
        #unfortunately since we are bounded by memory (on buffy for example) this method leads to multiple
        #loads of the same subcube when processing the simulattion. 

        comm = MPI.COMM_WORLD
        Nproc = comm.Get_size()
        rank = comm.Get_rank()

        if rank == 0:
                if overwrite :
                        print('Overwriting existing output files')
                else:
                        print('Skipping existing files')
        
        
        fof_suffix=ll_to_fof_suffix(ll)
        rtwo_suffix=get_r200_suffix(rtwo_fact)
        suffix=get_suffix(fof_suffix,rtwo_suffix)

        out,assoc_out,analy_out=gen_paths(sim_name,out_nb,suffix,assoc_mthd)

        output_str = 'output_%06i'%out_nb
        
        info_path=os.path.join(sim_path,output_str,'group_000001')
        snap_box_path = os.path.join(box_path,output_str)

        plt.rcParams.update({'font.size': 18})   

        #find number of subcubes
        data_files=os.listdir(os.path.join(box_path,output_str))
        rho_files=[f for f in data_files if 'rho' in f]

        n_subcubes=len(rho_files)
        assert n_subcubes>1, "Couldn't find any 'rho' subcubes... Are you sure about the path?"
        #print(n_subcubes)
        subs_per_side=int(np.round(n_subcubes**(1./3)))
        #print(subs_per_side)
        sub_side=int(float(ldx)/subs_per_side)
        #print(sub_side)

        #Get scale factor and co
        '''Get scale factor and co'''
        (t,a,H0,om_m,om_l,om_k,om_b,unit_l,unit_d,unit_t,l,Lco,L,px_to_m) = get_infos(info_path,
                                                                                           out_nb,
                                                                                           ldx)

        redshift = 1. / a - 1. 

        if rank==0:print("Redshift is %.1f"%redshift)

        with open(os.path.join(out,'Mp'), 'rb') as mass_file :
            Mp = np.fromfile(mass_file,dtype=np.float64)
        if rank==0 : print("DM part mass in msun : %e"%Mp)

        #dist_obs=345540.98618977674 #distance to obs point from box (0,0,0); in number of cells for z=6

        # mags,xis,contbetalow,contbetahigh,beta,metal_bins,age_bins=get_mag_tab_BPASSV221_betas()
        # mags_fct = get_mag_interp_fct(mags, age_bins, metal_bins)
        # low_mags_fct = get_mag_interp_fct(contbetalow, age_bins, metal_bins)
        # high_mags_fct = get_mag_interp_fct(contbetahigh, age_bins, metal_bins)                
        # xis_fct = get_xis_interp_fct(xis, age_bins, metal_bins)



        #upper=27
        #grid = np.mgrid[0:upper,0:upper,0:upper]/float(upper-1)

        if rank==0:print('Getting halos and associated stars')

        halo_tab, halo_star_ids, tot_star_nbs = read_assoc(out_nb,'CoDaIII',rtwo_fact=rtwo_fact,ll=ll,assoc_mthd=assoc_mthd)

        #prog_bar0=['[',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',']']
        #prog_bar=prog_bar0
        #nb_halo = np.shape(halo_tab)[0]

        ran = [-ldx,0,ldx]
        pos_vects = np.asarray([[i,j,k] for k in ran for j in ran for i in ran])

        #halo_sumd_fl,halo_nb_cells = np.zeros(3),np.zeros(3)

        rho_fact = get_unit_facts('rho', px_to_m, unit_d, unit_l, unit_t, a)
        #rhod_fact = get_unit_facts('rhod', px_to_m, unit_d, unit_l, unit_t, a)
        tau_fact = get_unit_facts('tau', px_to_m, unit_d, unit_l, unit_t, a)
        #temp_fact = get_unit_facts('temp', px_to_m, unit_d, unit_l, unit_t, a)

        #stt = time.time()


        # npix=hp.nside2npix(nside)
        # pix=np.arange(npix)
        # thes,phis=hp.pix2ang(nside,pix)
        #Xvects,Yvects,Zvects=hp.pix2vec(nside,pix) #with radius = 1

        #hp_surf_elem=4.*np.pi*px_to_m**2 #in m^2.dx^-2

        #print(len(phis))

        # xbins=np.arange(8,12.1,0.05)
        # ybins=np.arange(-2,0.1,0.05)
        #nbs_tot=np.zeros((len(xbins)-1,len(ybins)-1))

        big_side=int(sub_side*overstep)
        #pre-allocate boxes for data cubes
        big_rho=np.zeros((big_side,big_side,big_side),dtype=np.float32)        
        #big_dust=np.zeros((big_side,big_side,big_side),dtype=np.float32)
        #big_metals=np.zeros((big_side,big_side,big_side),dtype=np.float32)
        big_xtion=np.zeros((big_side,big_side,big_side),dtype=np.float32)        

        fmin, fmax, f_per_proc = divide_task(n_subcubes, Nproc, rank)
                
        for x_subnb in range(subs_per_side):
                for y_subnb in range(subs_per_side):
                        for z_subnb in range(subs_per_side):                

                                #subcube_nb=x_subnb + y_subnb*subs_per_side + z_subnb*subs_per_side**2.
                                subcube_nb = np.ravel_multi_index((x_subnb, y_subnb, z_subnb), (subs_per_side, subs_per_side, subs_per_side))

                                if subcube_nb<fmin or subcube_nb>fmax : continue
                                
                                out_file=os.path.join(analy_out,'halo_stats_%i'%subcube_nb)

                                out_exists=os.path.exists(out_file)


                                if out_exists and not overwrite :

                                        print('RANK %i: Skipping subcube #%i since it already exists'%(rank,subcube_nb))
                                        continue
                                
                                if rank==0 : print('Reading subcube #%s' %(subcube_nb))
                                
                                #Retain halos within sub cube
                                print(x_subnb, y_subnb, z_subnb)
                                x_cond = np.all([halo_tab['x']<=(x_subnb+1)*sub_side,halo_tab['x']>(x_subnb)*sub_side],axis=0)
                                y_cond = np.all([halo_tab['y']<=(y_subnb+1)*sub_side,halo_tab['y']>(y_subnb)*sub_side],axis=0)
                                z_cond = np.all([halo_tab['z']<=(z_subnb+1)*sub_side,halo_tab['z']>(z_subnb)*sub_side],axis=0)

                                ind_subcube = x_cond * y_cond * z_cond
                                if np.sum(ind_subcube)==0 : continue

                                #print(np.sum(ind_subcube))

                                sub_halo_tab = halo_tab[ind_subcube]
                                sub_halo_tot_star_nb = tot_star_nbs[ind_subcube]
                                sub_halo_star_nb = sub_halo_tab['nstar']
                                sub_idxs = sub_halo_tab['ids']

                                limit_r=sub_halo_tab['rpx']+1
                                sample_r=do_half_round(limit_r * 5)

                        
                                M,pos_nrmd = sub_halo_tab['mass'],np.asarray([list(col) for col in sub_halo_tab[['x','y','z']]])

                        
                                pos =do_half_round(pos_nrmd) #was np.int16

                      
                                #(0,0,0) px locattion of sub_side**3 cube within whole data set
                                edge=np.asarray([x_subnb*sub_side,y_subnb*sub_side,z_subnb*sub_side])

                                #print(edge)
                                
                                ctr_bxd=(pos-edge)

                                lower_bounds=np.int32(ctr_bxd-sample_r[:,np.newaxis])
                                upper_bounds=np.int32(ctr_bxd+sample_r[:,np.newaxis])

                                #print(pos, edge, lower_bounds)
                                
                                under=lower_bounds<0
                                over=upper_bounds>sub_side


                                # under=np.ones_like(lower_bounds)==0
                                # over=np.ones_like(lower_bounds)==0                                
                                
                                outside=[under,over]

                                #large_box=np.any(outside)

                                ctr_bxd=ctr_bxd+int(sub_side*(overstep-1)*0.5)
                                lower_bounds=lower_bounds+int(sub_side*(overstep-1)*0.5)
                                upper_bounds=upper_bounds+int(sub_side*(overstep-1)*0.5)
                                                        
                                #import datas and add margin for edge cases !
                                #with mpi we cant read the same files at the same time
                                #smart people would avoid this, here we catch the error and make the task sleep
                                done = False
                                while not done:
                                        try:
                                                get_overstep_hydro_cubed(big_rho,subcube_nb,snap_box_path,'rho',outside,n_subcubes=n_subcubes,size=sub_side,overstep=overstep)  
                                                done = True
                                        except:
                                                sleep(2)                      
                                done = False
                                while not done:
                                        try:
                                                get_overstep_hydro_cubed(big_xtion,subcube_nb,snap_box_path,'xtion',outside,n_subcubes=n_subcubes,size=sub_side,overstep=overstep)
                                                done = True
                                        except:
                                                sleep(2)                        
                                # done = False
                                # while not done:
                                #         try:
                                #                 get_overstep_hydro_cubed(big_metals,subcube_nb,snap_box_path,'Z',outside,n_subcubes=n_subcubes,size=sub_side,overstep=overstep) 
                                #                 done = True
                                #         except:
                                #                 sleep(2)  

                                # print(np.min(big_xtion), np.max(big_xtion), np.mean(big_xtion))
                                # print(np.min(big_rho) * rho_fact, np.max(big_rho) * rho_fact, np.mean(big_rho) * rho_fact)


                                # fig=plt.figure()
                                # ax=fig.add_subplot(111)


                                # w=5

                                # wctr = 256

                                # xy_plane=(ctr_bxd[:,2]<wctr+w)*(ctr_bxd[:,2]>wctr-w)
                                # xz_plane=(ctr_bxd[:,1]<wctr+w)*(ctr_bxd[:,1]>wctr-w)
                                # yz_plane=(ctr_bxd[:,0]<wctr+w)*(ctr_bxd[:,0]>wctr-w)                               

                                
                                # img=ax.imshow(np.log10(big_rho[wctr,:,:]).T,origin='lower')
                                # ax.scatter(ctr_bxd[xy_plane,1],ctr_bxd[xy_plane,0],s=1,alpha=0.5,c='w')
                                
                                # ax.set_xlabel('y')
                                # ax.set_ylabel('x')
                                # plt.colorbar(img)
                                # fig.savefig('test_dust_%i_xy'%subcube_nb)

                                # fig=plt.figure()
                                # ax=fig.add_subplot(111)

                                
                                # img=ax.imshow(np.log10(big_rho[:,:,wctr]).T,origin='lower')
                                # ax.scatter(ctr_bxd[yz_plane,2],ctr_bxd[yz_plane,1],s=1,alpha=0.5,c='w')
                                
                                # ax.set_xlabel('z')
                                # ax.set_ylabel('y')
                                # plt.colorbar(img)
                                # fig.savefig('test_dust_%i_yz'%subcube_nb)

                                # fig=plt.figure()
                                # ax=fig.add_subplot(111)


                                
                                # img=ax.imshow(np.log10(big_rho[:,wctr,:]).T,origin='lower')
                                # ax.scatter(ctr_bxd[xz_plane,2],ctr_bxd[xz_plane,0],s=1,alpha=0.5,c='w')
                                
                                # ax.set_xlabel('z')
                                # ax.set_ylabel('x')    
                                # plt.colorbar(img)
                                # fig.savefig('test_dust_%i_xz'%subcube_nb)

                                
                                # continue



                                if rank==0 : print("Loaded files")                                           
                                #get_overstep_hydro_cubed(big_dust,subcube_nb,snap_box_path,'dust',outside,n_subcubes=n_subcubes,size=sub_side,overstep=overstep)


                                # big_dust_side=int(sub_side*overstep)
                                
                                # big_dust=np.zeros((big_dust_side,big_dust_side,big_dust_side))
                                # big_xtion=np.zeros((big_dust_side,big_dust_side,big_dust_side))
                                # big_metals=np.zeros((big_dust_side,big_dust_side,big_dust_side))
                                # big_rho=np.zeros((big_dust_side,big_dust_side,big_dust_side))                       
                                
                                for ind,halo in enumerate(sub_halo_tab):

                                    #print('    Halo #%i'%ind)
                                    if sub_halo_star_nb[ind]>25 :

                                        r_px=halo["rpx"]

                                        #r=r_px*px_to_m #Convert to meters
                                        #surf_elem=(hp_surf_elem*r_px**2/float(npix))
                                                                        
                                        slices=np.index_exp[lower_bounds[ind,2]:upper_bounds[ind,2],lower_bounds[ind,1]:upper_bounds[ind,1],lower_bounds[ind,0]:upper_bounds[ind,0]]

                                        sm_rho = big_rho[slices]
                                        #sm_dust = big_dust[slices]
                                        #sm_metals = big_metals[slices]
                                        sm_xtion = big_xtion[slices]

                                        #print(np.max(sm_rho)/np.mean(big_rho),np.shape(sm_rho))
                                        # print(np.shape(sm_rho))
                                        # print(lower_bounds[ind],upper_bounds[ind])

                                        #print(sm_rho)
                                        
                                        sm_xHI=1-sm_xtion

                                        sm_tau = (sm_xHI*sm_rho)*(rho_fact*tau_fact) #includes partial dx

                                        #     sm_tau_dust_1600_SMC = (sm_dust)*(dust_1600_opacity_SMC*px_to_m*100.0)*rhod_fact
                                        #     sm_tau_dust_LyC_SMC = (sm_dust)*(dust_LyC_opacity_SMC*px_to_m*100.0)*rhod_fact

                                        #     sm_tau_dust_1600_MW = (sm_dust)*(dust_1600_opacity_MW*px_to_m*100.0)*rhod_fact
                                        #     sm_tau_dust_LyC_MW = (sm_dust)*(dust_LyC_opacity_MW*px_to_m*100.0)*rhod_fact

                                        #     sm_tau_dust_1600_LMC = (sm_dust)*(dust_1600_opacity_LMC*px_to_m*100.0)*rhod_fact
                                        #     sm_tau_dust_LyC_LMC = (sm_dust)*(dust_LyC_opacity_LMC*px_to_m*100.0)*rhod_fact

                                        fig, ax = make_figure()

                                        bounds = [lower_bounds[ind,2], upper_bounds[ind,2], lower_bounds[ind,1], upper_bounds[ind,1]]

                                        img = ax.imshow((sm_rho[:,:,np.int32(sample_r[ind])].T) * rho_fact * Pmass, origin='lower', norm=LogNorm(),
                                        extent=bounds)
                                        ax.set_xlabel('z')
                                        ax.set_ylabel('y')

                                        print(pos[ind], ctr_bxd[ind])
                                        print(lower_bounds[ind], upper_bounds[ind])

                                        cb = plt.colorbar(img)

                                        #Get stars for halo  from list of associated stars
                                        cur_star_ids=halo_star_ids[sub_halo_tot_star_nb[ind] - sub_halo_star_nb[ind]:sub_halo_tot_star_nb[ind]]
                                        cur_stars = read_specific_stars(os.path.join(star_path, output_str), cur_star_ids)

                                        ax.scatter(cur_stars['z']*ldx + int(sub_side*(overstep-1)*0.5),cur_stars['y']*ldx + int(sub_side*(overstep-1)*0.5))

                                        fig.savefig('test_rho')


                                        fig, ax = make_figure()

                                        bounds = [lower_bounds[ind,2], upper_bounds[ind,2], lower_bounds[ind,1], upper_bounds[ind,1]]

                                        img = ax.imshow((sm_xHI[:,:,np.int32(sample_r[ind])].T) * rho_fact * Pmass, origin='lower', norm=LogNorm(),
                                        extent=bounds)
                                        ax.set_xlabel('z')
                                        ax.set_ylabel('y')

                                        print(pos[ind], ctr_bxd[ind])
                                        print(lower_bounds[ind], upper_bounds[ind])

                                        cb = plt.colorbar(img)

                                        #Get stars for halo  from list of associated stars
                                        cur_star_ids=halo_star_ids[sub_halo_tot_star_nb[ind] - sub_halo_star_nb[ind]:sub_halo_tot_star_nb[ind]]
                                        cur_stars = read_specific_stars(os.path.join(star_path, output_str), cur_star_ids)

                                        ax.scatter(cur_stars['z']*ldx + int(sub_side*(overstep-1)*0.5),cur_stars['y']*ldx + int(sub_side*(overstep-1)*0.5))

                                        fig.savefig('test_xHI')                                        

                                        pause = input('any key to continue')
                                        
                                    


"""
Main body
"""


if __name__ =='__main__':

        Arg_parser = argparse.ArgumentParser('Compute gas and stellar properties in halos')

        Arg_parser.add_argument('nb',metavar='nsnap',type=int,help='snap number string, give as "XXX" so that 00001 is "001" and 00111 is "111"')
        Arg_parser.add_argument('--rtwo_fact',metavar='rtwo_fact',type=float,help='1.0 -> associate stellar particles within 1xR200 for all haloes',default=1)
        Arg_parser.add_argument('--ll',metavar='ll',type=float,help='linking length for fof',default=0.2)        
        Arg_parser.add_argument('--assoc_mthd',metavar='assoc_mthd',type=str,help='method for linking stars to fof',default='')
        Arg_parser.add_argument('--overwrite',acttion='store_true',help='When used, code overwrites all found data',default=False) 

        args = Arg_parser.parse_args()


        out_nb = args.nb
        rtwo_fact=args.rtwo_fact
        assoc_mthd=args.assoc_mthd
        ll=args.ll
        overwrite = args.overwrite

        compute_fesc(out_nb,rtwo_fact=rtwo_fact,assoc_mthd=assoc_mthd, ll=ll, overwrite=overwrite)


        
