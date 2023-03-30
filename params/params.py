"""
global params for all codes
"""

ldx = 8192
sim_name = "CoDaIII"

sim_path = "/gpfs/alpine/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/"
star_path = "/gpfs/alpine/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/reduced/stars"
box_path = "/gpfs/alpine/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/reduced/fullbox"

fof_path = "/gpfs/alpine/proj-shared/ast031/conaboy/fof/"


"""
ramses params from run time
"""
eta_sn = 0.2


"""
healpix resoluttion for fesc computattions
"""
nside=8 #2 or 4 #base elements for  higher -> more angular resoluttion                                       
rad_res=0.1 #in units of dx                                                                                 

"""
outputs
"""
sfr_age_bins = [0,1,5,10,50,100] #Myr
sfr_names = ["SFR%i"%bin for bin in sfr_age_bins[1:]]
Lintr_names = ["Lintr"] + ["Lintr%i"%bin for bin in sfr_age_bins[1:]]
profile_radius_bins = [0.1*i for i in range(1,11)] #in fracttions of r200

"""
Physics
"""

bpass_file_name = "cstSFR_5Myr_BPASSv2.2.1_kroupa_binary_MMax=100_pseudo-f=1_emissivites_mags_betas.txt"

#Physical constants
Pmass=1.6726219e-27 #Kg proton mass
Bmann=1.38064852e-23 #m**2*kg*s**-2*K**-1 #Boltzmann constant
pc = 3.08567758*10**16 #m
Msol = 1.989e30 #kg
G = 6.67408e-11 #m^3 kg^-1 s^-2
sigma_UV = 2.493e-22 # from codaii paper 1.63e-22#m^2  codai paper
c = 299792458 #m.s^-1
fesc_star = 0.1

delta_lamb = 2621./1492.

# #dust opacity stuff SMC 
# dust_1500_opacity_SMC=4.981e4 #cm^2/g
# dust_LyC_opacity_SMC=8.85e4 #611 ang
# dust_1492_opacity_SMC=4.98e4 
# dust_1600_opacity_SMC=3.83e4
# dust_2621_opacity_SMC=8.21e3

# #dust opacity stuff MW
# dust_1500_opacity_MW=2.334e4
# dust_LyC_opacity_MW=4.55e4 #611 ang
# dust_1492_opacity_MW=2.33e4 
# dust_1600_opacity_MW=2.3e4
# dust_2621_opacity_MW=1.58e4 

# #dust opacity stuff LMC
# dust_1500_opacity_LMC=4.468e4 #cm^2/g
# dust_LyC_opacity_LMC=1.36e5 #611 ang
# dust_1492_opacity_LMC=4.89e4 
# dust_1600_opacity_LMC=4.16e4 
# dust_2621_opacity_LMC=2.2e4