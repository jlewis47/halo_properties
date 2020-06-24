from star_halo_latest import assoc_stars_to_haloes
from halo_fesc_latest import compute_fesc
import numpy as np


path='/gpfsscratch/rech/xpu/uoj51ok/1024/16Mpc/Metals_He_YDdust_BPASSV221_fesc0.5_Tsf3e4K_eps0.04_dtmmax0.5'
sim_name='Metals_He_YDdust_BPASSV221_fesc0.5_Tsf3e4K_eps0.04_dtmmax0.5'
ldx=1024




for out_nb in np.arange(10,23,1):

    #assoc_stars_to_haloes('%03i'%out_nb,ldx,path,sim_name) 

    
    compute_fesc('%03i'%out_nb,ldx,path,sim_name) 