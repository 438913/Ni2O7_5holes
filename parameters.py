import math
import numpy as np

M_PI = math.pi

Mc = 2
pressure = 0  # in Gpa

# Note that Ni-d and O-p orbitals use hole language
# while Nd orbs use electron language
if pressure == 0:
    ed = {'d3z2r2': 0.046, \
          'dx2y2': 0, \
          'dxy': 0.823, \
          'dxz': 0.706, \
          'dyz': 0.706}
    eps = np.arange(2.47, 2.471, 1.0)
elif pressure == 4:
    ed = {'d3z2r2': 0.054, \
          'dx2y2': 0, \
          'dxy': 0.879, \
          'dxz': 0.761, \
          'dyz': 0.761}
    eps = np.arange(2.56, 2.561, 1.0)
elif pressure == 8:
    ed = {'d3z2r2': 0.060, \
          'dx2y2': 0, \
          'dxy': 0.920, \
          'dxz': 0.804, \
          'dyz': 0.804}
    eps = np.arange(2.62, 2.621, 1.0)
elif pressure == 16:
    ed = {'d3z2r2': 0.072, \
          'dx2y2': 0, \
          'dxy': 0.997, \
          'dxz': 0.887, \
          'dyz': 0.887}
    eps = np.arange(2.75, 2.751, 1.0)
elif pressure == 29.5:
    ed = {'d3z2r2': 0.095, \
          'dx2y2': 0, \
          'dxy': 1.06, \
          'dxz': 0.94, \
          'dyz': 0.94}
    # ed = {'d3z2r2': 0.0, \
    #       'dx2y2': 0.0, \
    #       'dxy': 0.0, \
    #       'dxz': 0.0, \
    #       'dyz': 0.0}
    eps = np.arange(2.9, 2.91, 1.0)

As = np.arange(6.0, 6.01, 2.0)
B = 0.15
C = 0.58
# As = np.arange(6.0, 6.01, 2.0)
# B = 0
# C = 0

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
if Norb == 7 or Norb == 4:
    #tpds = [0.00001]  # for check_CuO4_eigenvalues.py
    if pressure == 0:
        tpds = np.linspace(1.38, 1.38, num=1, endpoint=True)
        # tpps = [0.537]
        tpps = [0]
    #     tpds = [0.01]
    elif pressure == 4:
        tpds = np.linspace(1.43, 1.43, num=1, endpoint=True)
        # tpps = [0.548]
        tpps = [0]
    elif pressure == 8:
        tpds = np.linspace(1.46, 1.46, num=1, endpoint=True)
        # tpps = [0.554]
        tpps = [0]
    elif pressure == 16:
        tpds = np.linspace(1.52, 1.52, num=1, endpoint=True)
        # tpps = [0.566]
        tpps = [0]
    elif pressure == 29.5:
        tpds = np.linspace(1.58, 1.58, num=1, endpoint=True)
        # tpps = [0.562]
        tpps = [0]
    tz_a1a1 = 0.028

    # 29.5GPa:
    tz_a1a1 = 0.044

    tz_b1b1 = 0.047

    # tz_a1a1 = 0.028
    tz_a1a1 = 0
    tz_b1b1 = 0.047

if_tz_exist = 2
#if if_tz_exist = 0,tz exist in all orbits.
#if if_tz_exist = 1,tz exist in d orbits.
#if if_tz_exist = 2,tz exist in d3z2r2 orbits.

wmin = -10;
wmax = 30
eta = 0.1
Lanczos_maxiter = 600

# restriction on variational space
reduce_VS = 1

if_H0_rotate_byU = 1
basis_change_type = 'd_double'  # 'all_states' or 'd_double'
if_print_VS_after_basis_change = 0

if_compute_Aw = 0
if if_compute_Aw == 1:
    if_find_lowpeak = 0
    if if_find_lowpeak == 1:
        peak_mode = 'lowest_peak'  # 'lowest_peak' or 'highest_peak' or 'lowest_peak_intensity'
        if_write_lowpeak_ep_tpd = 1
    if_write_Aw = 0
    if_savefig_Aw = 1

if_get_ground_state = 1
if if_get_ground_state == 1:
    # see issue https://github.com/scipy/scipy/issues/5612
    Neval = 10
if_compute_Aw_dd_total = 0

if Norb == 7:
    Ni_orbs = ['dx2y2', 'dxy', 'dxz', 'dyz', 'd3z2r2']
elif Norb == 4:
    Ni_orbs = ['dx2y2', 'd3z2r2']

if Norb == 7 or Norb == 4:
    O1_orbs = ['px']
    O2_orbs = ['py']

O_orbs = O1_orbs + O2_orbs
# sort the list to facilliate the setup of interaction matrix elements
Ni_orbs.sort()
O1_orbs.sort()
O2_orbs.sort()
O_orbs.sort()

print("Ni_orbs = ", Ni_orbs)
print("O1_orbs = ", O1_orbs)
print("O2_orbs = ", O2_orbs)

orbs = Ni_orbs + O_orbs
#assert(len(orbs)==Norb)

Upps = [4.0]
print('Upps = ', Upps)
print(f'pressure = {pressure} Gpa')
symmetries = ['1A1', '3B1', '3B1', '1A2', '3A2', '1E', '3E']
print("compute A(w) for symmetries = ", symmetries)
