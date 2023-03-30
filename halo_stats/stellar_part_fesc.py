'''

'''


import numpy as np
import matplotlib as mpl
mpl.use('Agg')
#from read_radgpu import o_rad_cube_big
# from tionread_stars import read_all_star_files
import string
import argparse
import os
from association.read_assoc_latest import *
from files.read_fullbox_big import *
from utils.functions_latest import *
from params.params import *
from utils.output_paths import *
from utils.utils import get_fof_suffix, get_r200_suffix, get_suffix



def compute_stellar_Tr(out_nb,ldx,path,sim_name,use_fof=False,rtwo_fact=1,rel_fof_path=None):


        fof_suffix=get_fof_suffix(fof_path)
        rtwo_suffix=get_r200_suffix(rtwo_fact)
        suffix=get_suffix(fof_suffix,rtwo_suffix)        

        
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

        dist_obs=345540.98618977674 #distance to obs point from box (0,0,0); in number of cells for z=6

        mags,xis,contbetalow,contbetahigh,beta,metal_bins,age_bins=get_mag_tab_BPASSV221_betas()

        upper=27
        grid = np.mgrid[0:upper,0:upper,0:upper]/float(upper-1)

        print('Getting Phew and stars')
        idxs,star_idxs,phew_tot_star_nb,phew_star_nb,phew_tab,stars,lone_stars = read_assoc(out_nb,sim_name,use_fof,rtwo_fact,rel_fof_path)

        #Converstion factors
        tau_fact = px_to_m*sigma_UV*1e6*0.76 #important musnt count the helium...
        rho_fact=1e-3*unit_d/Pmass#unit_d*1e-3*(1e2)**3. #H/cm**3
        dust_fact=unit_d #g/cm^3
        vvrho_fact=1e6*1e3*1e-2*unit_l/unit_t/1e3 #m/s #H/m**2/s
        temp_fact=Pmass*(1e-2*unit_l/unit_t)**2/Bmann #K
        fl_fact=(a**-3.) #m**-2*s**-1



        #print(len(phis))

        rad_res=0.1 #in units of dx

        xbins=np.arange(8,12.1,0.05)
        ybins=np.arange(-2,0.1,0.05)
        nbs_tot=np.zeros((len(xbins)-1,len(ybins)-1))


        limit_r=phew_tab[:,-1]+1
        sample_r=do_half_round(limit_r)

        #Get data
        #fl= o_rad_cube_big(data_pth_rad,2,f_side,subcube_nb)
        # if tst :nrg = o_rad_cube_big(data_pth_rad,1,f_side,subcube_nb)
        xtion=o_data(os.path.join(data_pth_fullres,'xtion_00'+out_nb))
        rho=o_data(os.path.join(data_pth_fullres,'rho_00'+out_nb))
        dust=o_data(os.path.join(data_pth_fullres,'dust_00'+out_nb))
        #metals=o_data(os.path.join(data_pth_fullres,'Z_00'+out_nb)) #absolute
        # temp =  o_fullbox_big(data_pth_fullres,'temp',512,512,subcube_nb)
        # temp = temp/(rho*(1.+xtion))
        #rho = rho
        # vx =  o_fullbox_big(data_pth_fullres,'vx',512,512,subcube_nb)
        # vy =  o_fullbox_big(data_pth_fullres,'vy',512,512,subcube_nb)
        # vz =  o_fullbox_big(data_pth_fullres,'vz',512,512,subcube_nb)    
        # vv = np.asarray([vx,vy,vz]) 
        # vvrho=vv*rho

        lone_TrLintr_H=np.zeros(len(lone_stars))
        assoc_TrLintr_H=np.zeros(len(stars))

        lone_TrLintr_dust=np.zeros(len(lone_stars))
        assoc_TrLintr_dust=np.zeros(len(stars))                


        xHI=1-xtion

        tau = (xHI*rho)*(rho_fact*tau_fact) #includes partial dx

        #tau_dust_1600_SMC = (dust)*(dust_1600_opacity_SMC*px_to_m*100.0)*dust_fact
        tau_dust_LyC_SMC = (dust)*(dust_LyC_opacity_SMC*px_to_m*100.0)*dust_fact

        Tr_H_gas=np.exp(-tau)
        Tr_dust_LyC=np.exp(-tau_dust_LyC_SMC)
        
        assoc_star_inds=np.int16(stars[:,1:4]*ldx)
        lone_star_inds=np.int16(lone_stars[:,1:4]*ldx)


        box_grid=np.arange(0,ldx)

        #assoc_stars        
        star_sm_posx=np.digitize(assoc_star_inds[:,0],box_grid)-1
        star_sm_posy=np.digitize(assoc_star_inds[:,1],box_grid)-1
        star_sm_posz=np.digitize(assoc_star_inds[:,2],box_grid)-1                      
        
        assoc_TrLintr_H=Tr_H_gas[star_sm_posz,star_sm_posy,star_sm_posx]
        assoc_TrLintr_dust=Tr_dust_LyC[star_sm_posz,star_sm_posy,star_sm_posx]        

        with open(os.path.join(out,'associated_stellar_Tr%s_'%suffix+out_nb+'_0'),'wb') as newFile:
                np.save(newFile,np.int32(len(assoc_star_inds)))
                np.save(newFile,np.float64(assoc_TrLintr_H))
                np.save(newFile,np.float64(assoc_TrLintr_dust))


        #lone stars
        star_sm_posx=np.digitize(lone_star_inds[:,0],box_grid)-1
        star_sm_posy=np.digitize(lone_star_inds[:,1],box_grid)-1
        star_sm_posz=np.digitize(lone_star_inds[:,2],box_grid)-1                      
        
        lone_TrLintr_H=Tr_H_gas[star_sm_posz,star_sm_posy,star_sm_posx]
        lone_TrLintr_dust=Tr_dust_LyC[star_sm_posz,star_sm_posy,star_sm_posx]        
        
        with open(os.path.join(out,'lone_stellar_Tr%s_'%suffix+out_nb+'_0'),'wb') as newFile:
                np.save(newFile,np.int32(len(lone_star_inds)))
                np.save(newFile,np.float64(lone_TrLintr_H))
                np.save(newFile,np.float64(lone_TrLintr_dust))

        




                
            
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

    
        compute_stellar_Tr(out_nb,ldx,path,sim_name)
