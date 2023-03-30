from association.read_assoc_latest import read_assoc
import numpy as np
from src.bpass_fcts import *
from utils.output_paths import gen_paths
from utils.utils import ll_to_fof_suffix, get_r200_suffix, get_suffix
import argparse
import os
from utils.functions_latest import get_infos

def get_halo_Lintrs(out_nb,ldx,path,sim_name,rtwo_fact,ll=200):


        output_str='output_%06i'%out_nb
        
        info_path=os.path.join(path,output_str,'group_000001')

        fof_suffix=ll_to_fof_suffix(ll)
        rtwo_suffix=get_r200_suffix(rtwo_fact)
        suffix=get_suffix(fof_suffix,rtwo_suffix)
        
        Np_tot=ldx**3

        '''Get scale factor and co'''
        (t,a,H0,om_m,om_l,om_k,om_b,unit_l,unit_d,unit_t,l,Lco,L,px_to_m) = get_infos(info_path,
                                                                                           out_nb,
                                                                                      ldx)
        eta_sn=0.2
        print('eta_sn is %f'%eta_sn)

        out,assoc_out,analy_out=gen_paths(sim_name,out_nb,suffix)

        idxs,star_idxs,tot_star_nbs,star_nbs,halos,stars,lone_stars=read_assoc(out_nb,'CoDaIII',rtwo_fact=rtwo_fact,ll=ll)

        print(len(stars))
        
        nb_halos=len(halos)

        halo_Lintrs=np.zeros(nb_halos, dtype=np.float32)

        mags_tab,xis_tab,contbetalow,contbetahigh,beta,metal_bins,age_bins=get_mag_tab_BPASSV221_betas()

        xis_fct=get_xis_interp_fct(xis_tab,age_bins,metal_bins)

        # print(np.shape(stars))

        #print(age_bins)
        
        # print(star_nbs, tot_star_nbs)

        # print(len(stars),tot_star_nbs[-1])
        
        for ihalo, halo in enumerate(halos[:]):

            if star_nbs[ihalo]==0:continue

            #print(tot_star_nbs[ihalo]-1,star_nbs[ihalo])
            
            halo_stars=stars[int(tot_star_nbs[ihalo])-1:int(tot_star_nbs[ihalo]+star_nbs[ihalo])-1,:]
            
            halo_Lintrs[ihalo]=np.sum(get_star_xis_metals(halo_stars[:,-2],halo_stars[:,-1]*0.02,xis_fct))



        with open(os.path.join(analy_out,'halo_Lintrs'+suffix),'wb') as dest:
            np.save(dest,halo_Lintrs)
        
"""
Main body
"""


if __name__ =='__main__':

        Arg_parser = argparse.ArgumentParser('From stellar associations, build list of halo intrinsic luminosities as done by ATON')

        Arg_parser.add_argument('nb',metavar='nsnap',type=int,help='snap number string, give as "XXX" so that 00001 is "001" and 00111 is "111"')
        Arg_parser.add_argument('ldx',metavar='ldx',type=int,help='box size in cells')
        Arg_parser.add_argument('path',metavar='path',type=str,help='path to sim')
        Arg_parser.add_argument('simname',metavar='simname',type=str,help='sim name (will be used to create dirs)')
        Arg_parser.add_argument('--rtwo_fact',metavar='rtwo_fact',type=float,help='1.0 -> associate stellar particles within 1xR200 for all haloes',default=1)
        Arg_parser.add_argument('--ll',metavar='ll',type=float,help='linking length e.g. 0.2',default=0.2)            

        
        args = Arg_parser.parse_args()


        out_nb = args.nb
        ldx=args.ldx
        path=args.path
        sim_name=args.simname
        rtwo_fact=args.rtwo_fact
        ll=args.ll
        
        get_halo_Lintrs(out_nb,ldx,path,sim_name,rtwo_fact=rtwo_fact,ll=ll)
    
