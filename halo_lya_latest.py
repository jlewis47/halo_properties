'''

'''


import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from read_radgpu import o_rad_cube_big
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
from lya_func import *
#from wrap_boxes import *
from output_paths import *



def compute_lya(out_nb,ldx,path,sim_name):



        info_path=os.path.join(path,'output_00'+out_nb)
        data_pth_fullres=path

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


        upper=27
        grid = np.mgrid[0:upper,0:upper,0:upper]/float(upper-1)



        prog_bar0=['[',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',']']
        prog_bar=prog_bar0


        ran = [-ldx,0,ldx]
        pos_vects = [[i,j,k] for k in ran for j in ran for i in ran]



        #Conversion factors
        tau_fact = px_to_m*sigma_UV*1e6
        rho_fact=1e-3*unit_d/Pmass#unit_d*1e-3*(1e2)**3. #H/cm**3
        dust_fact=unit_d #g/cm^3
        vvrho_fact=1e6*1e3*1e-2*unit_l/unit_t/1e3 #m/s #H/m**2/s
        temp_fact=Pmass*(1e-2*unit_l/unit_t)**2/Bmann #K
        fl_fact=(a**-3.) #m**-2*s**-1


        stt = time.time()




        #Get data
        #fl= o_rad_cube_big(data_pth_rad,2,f_side,subcube_nb)
        # if tst :nrg = o_rad_cube_big(data_pth_rad,1,f_side,subcube_nb)
        xion=o_data(os.path.join(data_pth_fullres,'xion_%05i'%int(out_nb)))
        rho=o_data(os.path.join(data_pth_fullres,'rho_%05i'%int(out_nb)))
        #dust=o_data(os.path.join(data_pth_fullres,'dust_%05i'%int(out_nb)))
        #metals=o_data(os.path.join(data_pth_fullres,'Z_%05i'%int(out_nb))) #absolute
        temp=o_data(os.path.join(data_pth_fullres,'T_%05i'%int(out_nb)))
        temp = temp/(rho*(1.+xion))
        #rho = rho
        vx=o_data(os.path.join(data_pth_fullres,'vx_%05i'%int(out_nb)))
        vy=o_data(os.path.join(data_pth_fullres,'vy_%05i'%int(out_nb)))

        nHI=(1-xion)*rho



        lamb0=1215.6 #ang                                                    
        nu0=c/((lamb0)*1e-10)

        cell_lim=50.0/(l/(H0/100.)/ldx) #go to 50 cMpc distance like in papers 

        HzSI=H0*(om_m/a**3+om_l)**0.5*1e3/pc/1e6


        lya_n_lines=ldx
        lya_n_path=3
        lya_skip=ldx/4
        lya_angle=40.0

        taus=[]



        v_direction=(np.cos(lya_angle/180.*np.pi)*vx+np.sin(lya_angle/180.*np.pi)*vy) #m.s^-1

        Xstart=0
        Ystart=0

        i_tot=0


        print('z=%.1f, Lya starting ...'%(1.0/a-1.0),flush=True)

        for i_path in range(lya_n_path):                                     



            Xstart=i_path*ldx/float(lya_n_path)                              
            if Xstart>=ldx-1:                                                
                Xstart=0                                                     
                Ystart+=int(ldx*0.5)                                         

            if Ystart>=ldx-1:                                                
                Xstart+=int(ldx*0.5)                                         
                Ystart=0                


            Xs,Ys,Ds,D_tot=ray2D(Xstart,Ystart,cell_lim,lya_angle/180.*np.pi,ldx)

            Ds=Nresample_vect(Ds,8)/8.0





            cum_Ds=np.cumsum(Ds)


            Xray,Yray=np.int16(Xs),np.int16(Ys)

            

            vexp=-cum_Ds[::-1]*HzSI*px_to_m

            i_z=0

            while i_z<int(float(lya_n_lines)/lya_skip):


                Zray=int(i_z*lya_skip)
                    
                loc_T_vects=temp[Xray,Yray,Zray]*temp_fact
                loc_nHI_vects=nHI[Xray,Yray,Zray]*unit_d*1e-3/Pmass*1e6
                loc_v_vects=v_direction[Xray,Yray,Zray]*unit_l/unit_t/100.0 #m/s

                loc_T_vects=Nresample_vect(loc_T_vects,8)
                loc_nHI_vects=Nresample_vect(loc_nHI_vects,8)
                loc_v_vects=Nresample_vect(loc_v_vects,8)

                loc_tau_vects=jon_forest(loc_nHI_vects,loc_T_vects,loc_v_vects,vexp,Ds,cum_Ds,HzSI,px_to_m)

                print('\t %f'%(-np.log(np.nanmean(np.exp(-loc_tau_vects)))))
                taus.append(-np.log(np.nanmean(np.exp(-loc_tau_vects))))


                i_z+=1
                i_tot+=1

                print('... %.1i'%(i_tot),flush=True)



        print('Writing data file')


        file_bytes = np.asarray(taus)

        #Write our output ... 
        with open(os.path.join(out,'lya_out_'+out_nb),'wb') as newFile:
            np.save(newFile,np.int32(np.shape(taus)))
            np.save(newFile,np.float64(a))
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

    
        compute_lya(out_nb,ldx,path,sim_name)
