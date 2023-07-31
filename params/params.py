"""
global params for all codes
"""

ldx = 8192
sim_name = "CoDaIII"

sim_path = f"/gpfs/alpine/proj-shared/ast031/pocvirk/{sim_name:s}/prod_sr/"
star_path = (
    f"/gpfs/alpine/proj-shared/ast031/pocvirk/{sim_name:s}/prod_sr/reduced/stars"
)
box_path = (
    f"/gpfs/alpine/proj-shared/ast031/pocvirk/{sim_name:s}/prod_sr/reduced/fullbox"
)

# fof_path = "/gpfs/alpine/proj-shared/ast031/conaboy/fof/"
fof_path = "/gpfs/alpine/proj-shared/ast031/jlewis/CoDaIII/prod_sr/mp_cats/"  # MP cats

sixdigits = True

# ldx = 1024
# # sim_name = "8Mpc_with_T_cor"
# sim_name = "8Mpc_no_T_cor"

# sim_path = f"/gpfs/alpine/ast031/scratch/pocvirk/calib/2023/{sim_name:s}/"
# star_path = f"/gpfs/alpine/ast031/scratch/pocvirk/calib/2023/{sim_name:s}/reduced/stars"
# box_path = f"/gpfs/alpine/ast031/scratch/pocvirk/calib/2023/{sim_name:s}/reduced/fullbox"

# fof_path = f"/gpfs/alpine/ast031/scratch/pocvirk/calib/2023/{sim_name:s}"

# sixdigits=False

"""
ramses params from run time
"""
eta_sn = 0.2

"""
healpix resolution for fesc computattions
"""
nside = 8  # 2 or 4 #base elements for  higher -> more angular resoluttion
rad_res = 0.1  # in units of dx

"""
outputs
"""
sfr_age_bins = [0, 1, 5, 10, 50, 100]  # Myr
sfr_names = ["SFR%i" % bin for bin in sfr_age_bins[1:]]
Lintr_names = ["Lintr"] + ["Lintr%i" % bin for bin in sfr_age_bins[1:]]
profile_radius_bins = [0.1 * i for i in range(1, 11)]  # in fracttions of r200

"""
Physics
"""

bpass_file_name = "cstSFR_5Myr_BPASSv2.2.1_kroupa_binary_MMax=100_pseudo-f=1_emissivites_mags_betas.txt"

# Physical constants, from NIST where available
Pmass = 1.6726219e-27  # Kg proton mass
Bmann = 1.38064852e-23  # m**2*kg*s**-2*K**-1 #Boltzmann constant
pc = 3.08567758 * 10**16  # m
Msol = 1.989e30  # kg
G = 6.67408e-11  # m^3 kg^-1 s^-2
sigma_UV = 2.493e-22  # from codaii paper 1.63e-22#m^2  codai paper
c = 299792458  # m.s^-1
# fesc_star = 0.1
