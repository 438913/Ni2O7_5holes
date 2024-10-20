import math
import numpy as np
from scipy.sparse.linalg import inv
# from numpy.linalg import inv
import scipy.sparse as sps
import scipy.sparse.linalg
from scipy import integrate
import sys
import matplotlib.pyplot as plt

sys.path.append('../../src/')
from pylab import *

import parameters as pam
import lattice as lat
import variational_space as vs

# import hamiltonian as ham
import hamiltonian as ham  # convention of putting U/2 to d8 and d10 separately

import basis_change as basis
import get_state as getstate
import utility as util
import plotfig as fig
import ground_state_eigsh as gse
import ground_state as gs
# import ground_state_lanczos as gs
import lanczos
import time

start_time = time.time()
M_PI = math.pi


#####################################
def compute_Aw_main(A, ep, tpd, tpp, tz_a1a1, tz_b1b1, pds, pdp, pps, ppp, Upp, \
                    d_000_double, d_200_double, p_double, double_000_part, idx_000, hole345_000_part, \
                    double_200_part, idx_200, hole345_200_part, \
                    U_000, S_000_val, Sz_000_val, U_200, S_200_val, Sz_200_val, AorB_000_sym, AorB_200_sym):
    if Norb == 8:
        fname = 'ep' + str(ep) + '_epbilayer' + str(pam.epbilayer) + '_tpd' + str(tpd) + '_tpp' + str(tpp) \
                + '_tpzd' + str(tpzd) + '_tz_a1a1' + str(tz_a1a1) + '_Mc' + str(Mc) + '_Norb' + str(
            Norb) + '_eta' + str(eta)
        flowpeak = 'Norb' + str(Norb) + '_tpp' + str(tpp) + '_Mc' + str(Mc) + '_eta' + str(eta)
    elif Norb == 10 or Norb == 11 or Norb == 12:
        fname = 'ep' + str(ep) + '_epbilayer' + str(pam.epbilayer) + '_pdp' + str(pdp) + '_pps' + str(
            pps) + '_ppp' + str(ppp) \
                + '_tz_a1a1' + str(tz_a1a1) + '_tpzd' + str(tpzd) + '_Mc' + str(Mc) + '_Norb' + str(
            Norb) + '_eta' + str(eta)
        flowpeak = 'Norb' + str(Norb) + '_pps' + str(pps) + '_ppp' + str(ppp) + '_Mc' + str(Mc) + '_eta' + str(eta)

    w_vals = np.arange(pam.wmin, pam.wmax, pam.eta)
    Aw = np.zeros(len(w_vals))
    Aw_dd_total = np.zeros(len(w_vals))
    Aw_d8_total = np.zeros(len(w_vals))

    # set up H0
    if Norb == 7 or Norb == 4:
        tpd_nn_hop_dir, tpd_orbs, tpd_nn_hop_fac, tpp_nn_hop_fac \
            = ham.set_tpd_tpp(Norb, tpd, tpp, 0, 0, 0, 0)
    elif Norb == 10 or Norb == 12:
        tpd_nn_hop_dir, tpd_orbs, tpd_nn_hop_fac, tpp_nn_hop_fac \
            = ham.set_tpd_tpp(Norb, 0, 0, pds, pdp, pps, ppp)

    tz_fac = ham.set_tz(Norb, if_tz_exist, tz_a1a1, tz_b1b1)

    T_pd = ham.create_tpd_nn_matrix(VS, tpd_nn_hop_dir, tpd_orbs, tpd_nn_hop_fac)
    T_pp = ham.create_tpp_nn_matrix(VS, tpp_nn_hop_fac)
    #     T_z    = ham.create_tz_matrix(VS,tz_fac)

    Esite = ham.create_edep_diag_matrix(VS, A, ep)

    H0 = T_pd + T_pp + Esite
    #     H0 = T_pd
    print("H0 %s seconds ---" % (time.time() - start_time))

    '''
    Below probably not necessary to do the rotation by multiplying U and U_d
    the basis_change.py is only for label the state as singlet or triplet
    and assign the interaction matrix
    '''
    if pam.if_H0_rotate_byU == 1:
        H0_000_new = U_000_d.dot(H0.dot(U_000))
        # H0_200_new = U_200_d.dot(H0.dot(U_200))

    clf()

    if Norb == 4 or Norb == 7 or Norb == 10 or Norb == 11 or Norb == 12:
        Hint_000 = ham.create_interaction_matrix_ALL_syms(VS, d_000_double, p_double, double_000_part, idx_000,
                                                          hole345_000_part, \
                                                          S_000_val, Sz_000_val, AorB_000_sym, A, Upp)
        Hint_200 = ham.create_interaction_matrix_ALL_syms(VS, d_200_double, p_double, double_200_part, idx_200,
                                                          hole345_200_part, \
                                                          S_200_val, Sz_200_val, AorB_200_sym, A, Upp)
        if pam.if_H0_rotate_byU == 1:
            H_000 = H0_000_new + Hint_000
            # H_200 = H0_200_new + Hint_200

            # continue rotate the basis for setting Cu layer's interaction (d_Cu_double)
            H0_200_new = U_200_d.dot(H_000.dot(U_200))
            H = H0_200_new + Hint_200
            # H0_000_new = U_000_d.dot(H_200.dot(U_000))
            # H = H0_000_new + Hint_000
        else:
            H = H0 + Hint_200 + Hint_000
        H_bond = U_bond_d.dot(H.dot(U_bond))
        H_bond.tocsr()

        ####################################################################################
        # compute GS only for turning on full interactions
        if pam.if_get_ground_state == 1:
            vals = gs.get_ground_state(H_bond, VS, S_000_val, Sz_000_val, S_200_val, Sz_200_val, bonding_val)
        #             if Norb==8:
        #                 util.write_GS('Egs_'+flowpeak+'.txt',A,ep,tpd,vals[0])
        #                 #util.write_GS_components('GS_weights_'+flowpeak+'.txt',A,ep,tpd,wgt_d8, wgt_d9L, wgt_d10L2)
        #             elif Norb==10 or Norb==11 or Norb==12:
        #                 util.write_GS2('Egs_'+flowpeak+'.txt',A,ep,pds,pdp,vals[0])
        #                 #util.write_GS_components2('GS_weights_'+flowpeak+'.txt',A,ep,pds,pdp,wgt_d8, wgt_d9L, wgt_d10L2)

        #########################################################################
        '''
        Compute A(w) for various states
        '''


#         if pam.if_compute_Aw==1:
#             # compute d8
#             fig.compute_Aw_d8_sym(H, VS, d_double_no_eh, S_val, Sz_val, AorB_sym, A, w_vals, "Aw_d8_sym_", fname)

#             # compute d9L
#             b1L_state_indices, a1L_state_indices, b1L_state_labels, a1L_state_labels \
#                     = getstate.get_d9L_state_indices(VS, S_val, Sz_val)
#             fig.compute_Aw1(H, VS, w_vals, b1L_state_indices, b1L_state_labels, "Aw_b1L_", fname)
#             fig.compute_Aw1(H, VS, w_vals, a1L_state_indices, a1L_state_labels, "Aw_a1L_", fname)

#             # compute d10L2
#             d10L2_state_indices, d10L2_state_labels = getstate.get_d10L2_state_indices(VS, S_val, Sz_val)
#             fig.compute_Aw1(H, VS, w_vals, d10L2_state_indices, d10L2_state_labels, "Aw_d10L2_", fname)

#             # compute d8Ls for some special states
#             a1b1Ls_S0_state_indices, a1b1Ls_S0_state_labels, \
#             a1b1Ls_S1_state_indices, a1b1Ls_S1_state_labels, \
#             a1a1Ls_state_indices, a1a1Ls_state_labels \
#                                             = getstate.get_d8Ls_state_indices(VS, d_double_one_eh, S_val, Sz_val)
#             fig.compute_Aw1(H, VS, w_vals, a1b1Ls_S0_state_indices, a1b1Ls_S0_state_labels, "Aw_a1b1Ls_S0_", fname)
#             fig.compute_Aw1(H, VS, w_vals, a1b1Ls_S1_state_indices, a1b1Ls_S1_state_labels, "Aw_a1b1Ls_S1_", fname)
#             fig.compute_Aw1(H, VS, w_vals, a1a1Ls_state_indices, a1a1Ls_state_labels, "Aw_a1a1Ls_", fname)

#             # compute d9L2s
#             d9L2s_state_indices, d9L2s_state_labels = getstate.get_d9L2s_state_indices(VS)
#             fig.compute_Aw1(H, VS, w_vals, d9L2s_state_indices, d9L2s_state_labels, "Aw_d9L2s_", fname)


##########################################################################
if __name__ == '__main__':
    Mc = pam.Mc
    print('Mc=', Mc)

    Norb = pam.Norb
    eta = pam.eta
    ed = pam.ed

    As = pam.As
    B = pam.B
    C = pam.C

    tz_a1a1 = pam.tz_a1a1
    tz_b1b1 = pam.tz_b1b1

    if_tz_exist = pam.if_tz_exist

    # set up VS
    VS = vs.VariationalSpace(Mc)

    d_000_double, idx_000, hole345_000_part, double_000_part, \
        d_200_double, idx_200, hole345_200_part, double_200_part, \
        p_double = ham.get_double_occu_list(VS)

    # change the basis for d_double states to be singlet/triplet

    if pam.basis_change_type == 'd_double':
        U_000, S_000_val, Sz_000_val, AorB_000_sym, \
            = basis.create_singlet_triplet_basis_change_matrix_d_double \
            (VS, d_000_double, double_000_part, idx_000, hole345_000_part)
        U_200, S_200_val, Sz_200_val, AorB_200_sym, \
            = basis.create_singlet_triplet_basis_change_matrix_d_double \
            (VS, d_200_double, double_200_part, idx_200, hole345_200_part)

    U_bond, bonding_val = basis.create_bonding_anti_bonding_basis_change_matrix(VS)
    U_000_d = (U_000.conjugate()).transpose()
    U_200_d = (U_200.conjugate()).transpose()
    U_bond_d = (U_bond.conjugate()).transpose()
    # check if U if unitary
    # checkU_unitary(U,U_d)

    if Norb == 7 or Norb == 4:
        for tpd in pam.tpds:
            for ep in pam.eps:
                for A in pam.As:
                    util.get_atomic_d8_energy(A, B, C)
                    for tpp in pam.tpps:
                        for Upp in pam.Upps:
                            print('===================================================')
                            print('A=', A, 'ep=', ep, ' tpd=', tpd, ' tpp=', tpp, \
                                  ' Upp=', Upp, 'tz_a1a1=', tz_a1a1)
                            compute_Aw_main(A, ep, tpd, tpp, tz_a1a1, tz_b1b1, 0, 0, 0, 0, Upp, \
                                            d_000_double, d_200_double, p_double, double_000_part, idx_000,
                                            hole345_000_part, \
                                            double_200_part, idx_200, hole345_200_part, \
                                            U_000, S_000_val, Sz_000_val, U_200, S_200_val, Sz_200_val, AorB_000_sym,
                                            AorB_200_sym)

    print("--- %s seconds ---" % (time.time() - start_time))