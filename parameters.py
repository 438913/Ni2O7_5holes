import math
import numpy as np
M_PI = math.pi

Mc = 4

# Note that Ni-d and O-p orbitals use hole language
# while Nd orbs use electron language
ed = {'d3z2r2': 1.028,\
        'dx2y2' : 1.028,\
        'dxy'   : 1.937,\
        'dxz'   : 1.937,\
        'dyz'   : 1.937}
ed = {'d3z2r2': 0,\
        'dx2y2' : 0,\
        'dxy'   : 0.909,\
        'dxz'   : 0.909,\
        'dyz'   : 0.909}

# 29.5GPa:
ed = {'d3z2r2': 0.095,\
        'dx2y2' : 0,\
        'dxy'   : 1.06,\
        'dxz'   : 0.94,\
        'dyz'   : 0.94}

eps = np.arange(1.86, 1.861, 1.0)

# 29.5GPa:
eps = np.arange(3.23, 3.231, 1.0)


As = np.arange(6.0, 6.01, 2.0)
B = 0.15
C = 0.58
#As = np.arange(100, 100.1, 1.0)
# As = np.arange(0.0, 0.01, 1.0)
# B = 0
# C = 0

# Note: tpd and tpp are only amplitude signs are considered separately in hamiltonian.py
# Slater Koster integrals and the overlaps between px and d_x^2-y^2 is sqrt(3) bigger than between px and d_3z^2-r^2 
# These two are proportional to tpd,sigma, whereas the hopping between py and d_xy is proportional to tpd,pi

# IMPORTANT: keep all hoppings below positive to avoid confusion
#            hopping signs are considered in dispersion separately
Norb = 4
if Norb==7 or Norb==4:
    #tpds = [0.00001]  # for check_CuO4_eigenvalues.py
    tpds = np.linspace(1.3, 1.3, num=1, endpoint=True) #[0.25]
#     tpds = [0.01]

    # 29.5GPa:
    tpds = np.linspace(1.58, 1.58, num=1, endpoint=True) #[0.25]

    tpps = [0.55]

    # 29.5GPa:
    tpps = [0.35]
    

    tz_a1a1 = 0.028

    # 29.5GPa:
    tz_a1a1 = 0.044

    tz_b1b1 = 0.047
    

    tz_a1a1 = 0.028
    tz_b1b1 = 0.047
    
if_tz_exist = 2
    #if if_tz_exist = 0,tz exist in all orbits.
    #if if_tz_exist = 1,tz exist in d orbits.
    #if if_tz_exist = 2,tz exist in d3z2r2 orbits.        

wmin = -10; wmax = 30
eta = 0.1
Lanczos_maxiter = 600

# restriction on variational space
reduce_VS = 0

if_H0_rotate_byU =1
basis_change_type = 'd_double' # 'all_states' or 'd_double'
if_print_VS_after_basis_change = 0

if_compute_Aw = 0
if if_compute_Aw==1:
    if_find_lowpeak = 0
    if if_find_lowpeak==1:
        peak_mode = 'lowest_peak' # 'lowest_peak' or 'highest_peak' or 'lowest_peak_intensity'
        if_write_lowpeak_ep_tpd = 1
    if_write_Aw = 0
    if_savefig_Aw = 1

if_get_ground_state = 1
if if_get_ground_state==1:
    # see issue https://github.com/scipy/scipy/issues/5612
    Neval = 10
if_compute_Aw_dd_total = 0

if Norb==7 :
    Ni_orbs = ['dx2y2','dxy','dxz','dyz','d3z2r2']
elif Norb==4:
    Ni_orbs = ['dx2y2','d3z2r2']    
    
if Norb==7 or Norb==4:
    O1_orbs  = ['px']
    O2_orbs  = ['py']
   

    
O_orbs = O1_orbs + O2_orbs
# sort the list to facilliate the setup of interaction matrix elements
Ni_orbs.sort()
O1_orbs.sort()
O2_orbs.sort()
O_orbs.sort()

print ("Ni_orbs = ", Ni_orbs)
print ("O1_orbs = ",  O1_orbs)
print ("O2_orbs = ",  O2_orbs)

orbs = Ni_orbs + O_orbs
#assert(len(orbs)==Norb)

Upps = [0]
symmetries = ['1A1','3B1','3B1','1A2','3A2','1E','3E']
print ("compute A(w) for symmetries = ",symmetries)
