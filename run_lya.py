from halo_lya_latest import compute_lya
import numpy as np


#path='/gpfsscratch/rech/xpu/uoj51ok/1024/16Mpc/Metals_He_YDdust_BPASSV221_fesc0.5_Tsf3e4K_eps0.04_dtmmax0.5'
#path='/gpfsscratch/rech/xpu/uoj51ok/1024/16Mpc/Metals_He_YDdust_BPASSV221_fesc0.5_Tsf3e4K_eps0.04'
#path='/gpfsscratch/rech/xpu/uoj51ok/1024/16Mpc/Metals_He_YDdust_BPASSV221_fesc0.5_Tsf3e4K_eps0.04_dtmmax0.05_cond0.0005'
path='/gpfsscratch/rech/xpu/uoj51ok/1024/16Mpc/Metals_He_YDdust_BPASSV221_fesc0.52_Tsf3e4K_eps0.04_dtmmax0.5_lowTcooling_properHeCool'




sim_name='Metals_He_YDdust_BPASSV221_fesc0.52_Tsf3e4K_eps0.04_dtmmax0.5_lowTcooling_properHeCool'
ldx=1024




for out_nb in np.arange(10,17):

    compute_lya('%03i'%out_nb,ldx,path,sim_name) 
