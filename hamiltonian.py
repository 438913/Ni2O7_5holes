'''
Functions for constructing individual parts of the Hamiltonian. The 
matrices still need to be multiplied with the appropriate coupling 
constants t_pd, t_pp, etc..

If adding into diagonal Nd atoms (with only s-like orbital as approximation)
then only dxy hops with it
'''
import os
import time
import parameters as pam
import lattice as lat
import variational_space as vs
import utility as util
import numpy as np
import scipy.sparse as sps

directions_to_vecs = {'UR': (1, 1, 0), \
                      'UL': (-1, 1, 0), \
                      'DL': (-1, -1, 0), \
                      'DR': (1, -1, 0), \
                      'L': (-1, 0, 0), \
                      'R': (1, 0, 0), \
                      'U': (0, 1, 0), \
                      'D': (0, -1, 0), \
                      'T': (0, 0, 1), \
                      'B': (0, 0, -1), \
                      'L2': (-2, 0, 0), \
                      'R2': (2, 0, 0), \
                      'U2': (0, 2, 0), \
                      'D2': (0, -2, 0), \
                      'T2': (0, 0, 2), \
                      'B2': (0, 0, -2), \
                      'pzL': (-1, 0, 1), \
                      'pzR': (1, 0, 1), \
                      'pzU': (0, 1, 1), \
                      'pzD': (0, -1, 1), \
                      'mzL': (-1, 0, -1), \
                      'mzR': (1, 0, -1), \
                      'mzU': (0, 1, -1), \
                      'mzD': (0, -1, -1)}
tpp_nn_hop_dir = ['UR', 'UL', 'DL', 'DR']

def set_tpd_tpp(Norb, tpd, tpp, pds, pdp, pps, ppp):
    # dxz and dyz has no tpd hopping

    if pam.Norb == 4 or pam.Norb == 7:
        tpd_nn_hop_dir = {'d3z2r2': ['L', 'R', 'U', 'D'], \
                          'dx2y2': ['L', 'R', 'U', 'D'], \
                          'px': ['L', 'R'], \
                          'py': ['U', 'D']}
    elif pam.Norb == 10 or pam.Norb == 11:
        tpd_nn_hop_dir = {'d3z2r2': ['L', 'R', 'U', 'D'], \
                          'dx2y2': ['L', 'R', 'U', 'D'], \
                          'dxy': ['L', 'R', 'U', 'D'], \
                          'px1': ['L', 'R'], \
                          'py1': ['L', 'R'], \
                          'px2': ['U', 'D'], \
                          'py2': ['U', 'D']}
    elif pam.Norb == 12:
        tpd_nn_hop_dir = {'d3z2r2': ['L', 'R', 'U', 'D'], \
                          'dx2y2': ['L', 'R', 'U', 'D'], \
                          'dxy': ['L', 'R', 'U', 'D'], \
                          'dxz': ['L', 'R'], \
                          'dyz': ['U', 'D'], \
                          'px1': ['L', 'R'], \
                          'py1': ['L', 'R'], \
                          'pz1': ['L', 'R'], \
                          'px2': ['U', 'D'], \
                          'py2': ['U', 'D'], \
                          'pz2': ['U', 'D']}

    if pam.Norb == 7 or pam.Norb == 4:
        tpd_orbs = {'d3z2r2', 'dx2y2', 'px', 'py'}
    elif pam.Norb == 10:
        tpd_orbs = {'d3z2r2', 'dx2y2', 'dxy', 'px1', 'py1', 'px2', 'py2'}
    elif pam.Norb == 12:
        tpd_orbs = {'d3z2r2', 'dx2y2', 'dxy', 'dxz', 'dyz', 'px1', 'py1', 'pz1', 'px2', 'py2', 'pz2'}

    # hole language: sign convention followed from Fig 1 in H.Eskes's PRB 1990 paper
    #                or PRB 2016: Characterizing the three-orbital Hubbard model...
    # Or see George's email on Aug.19, 2021:
    # dx2-y2 hoping to the O px in the positive x direction should be positive for electrons and for O in the minus x
    # directions should be negative for electrons, i.e. the hoping integral should be minus sign of overlap integral
    # between two neighboring atoms.
    if pam.Norb == 7 or pam.Norb == 4:
        # d3z2r2 has +,-,+ sign structure so that it is - in x-y plane
        tpd_nn_hop_fac = {('d3z2r2', 'L', 'px'): -tpd / np.sqrt(3), \
                          ('d3z2r2', 'R', 'px'): tpd / np.sqrt(3), \
                          ('d3z2r2', 'U', 'py'): tpd / np.sqrt(3), \
                          ('d3z2r2', 'D', 'py'): -tpd / np.sqrt(3), \
                          ('dx2y2', 'L', 'px'): tpd, \
                          ('dx2y2', 'R', 'px'): -tpd, \
                          ('dx2y2', 'U', 'py'): tpd, \
                          ('dx2y2', 'D', 'py'): -tpd, \
                          # below just inverse dir of the above one by one
                          ('px', 'R', 'd3z2r2'): -tpd / np.sqrt(3), \
                          ('px', 'L', 'd3z2r2'): tpd / np.sqrt(3), \
                          ('py', 'D', 'd3z2r2'): tpd / np.sqrt(3), \
                          ('py', 'U', 'd3z2r2'): -tpd / np.sqrt(3), \
                          ('px', 'R', 'dx2y2'): tpd, \
                          ('px', 'L', 'dx2y2'): -tpd, \
                          ('py', 'D', 'dx2y2'): tpd, \
                          ('py', 'U', 'dx2y2'): -tpd}
    elif pam.Norb == 10:
        c = np.sqrt(3) / 2.0
        tpd_nn_hop_fac = {('d3z2r2', 'L', 'px1'): -pds / 2.0, \
                          ('d3z2r2', 'R', 'px1'): pds / 2.0, \
                          ('d3z2r2', 'U', 'py2'): pds / 2.0, \
                          ('d3z2r2', 'D', 'py2'): -pds / 2.0, \
                          ('dx2y2', 'L', 'px1'): pds * c, \
                          ('dx2y2', 'R', 'px1'): -pds * c, \
                          ('dx2y2', 'U', 'py2'): pds * c, \
                          ('dx2y2', 'D', 'py2'): -pds * c, \
                          ('dxy', 'L', 'py1'): -pdp, \
                          ('dxy', 'R', 'py1'): pdp, \
                          ('dxy', 'U', 'px2'): pdp, \
                          ('dxy', 'D', 'px2'): -pdp, \
                          # below just inverse dir of the above one by one
                          ('px1', 'R', 'd3z2r2'): -pds / 2.0, \
                          ('px1', 'L', 'd3z2r2'): pds / 2.0, \
                          ('py2', 'D', 'd3z2r2'): pds / 2.0, \
                          ('py2', 'U', 'd3z2r2'): -pds / 2.0, \
                          ('px1', 'R', 'dx2y2'): pds * c, \
                          ('px1', 'L', 'dx2y2'): -pds * c, \
                          ('py2', 'D', 'dx2y2'): pds * c, \
                          ('py2', 'U', 'dx2y2'): -pds * c, \
                          ('py1', 'R', 'dxy'): -pdp, \
                          ('py1', 'L', 'dxy'): pdp, \
                          ('px2', 'D', 'dxy'): pdp, \
                          ('px2', 'U', 'dxy'): -pdp}
    elif pam.Norb == 12:
        c = np.sqrt(3) / 2.0
        tpd_nn_hop_fac = {('d3z2r2', 'L', 'px1'): -pds / 2.0, \
                          ('d3z2r2', 'R', 'px1'): pds / 2.0, \
                          ('d3z2r2', 'U', 'py2'): pds / 2.0, \
                          ('d3z2r2', 'D', 'py2'): -pds / 2.0, \
                          ('dx2y2', 'L', 'px1'): pds * c, \
                          ('dx2y2', 'R', 'px1'): -pds * c, \
                          ('dx2y2', 'U', 'py2'): pds * c, \
                          ('dx2y2', 'D', 'py2'): -pds * c, \
                          ('dxy', 'L', 'py1'): -pdp, \
                          ('dxy', 'R', 'py1'): pdp, \
                          ('dxy', 'U', 'px2'): pdp, \
                          ('dxy', 'D', 'px2'): -pdp, \
                          ('dxz', 'L', 'pz1'): -pdp, \
                          ('dxz', 'R', 'pz1'): pdp, \
                          ('dyz', 'U', 'pz2'): pdp, \
                          ('dyz', 'D', 'pz2'): -pdp, \
                          # below just inverse dir of the above one by one
                          ('px1', 'R', 'd3z2r2'): -pds / 2.0, \
                          ('px1', 'L', 'd3z2r2'): pds / 2.0, \
                          ('py2', 'D', 'd3z2r2'): pds / 2.0, \
                          ('py2', 'U', 'd3z2r2'): -pds / 2.0, \
                          ('px1', 'R', 'dx2y2'): pds * c, \
                          ('px1', 'L', 'dx2y2'): -pds * c, \
                          ('py2', 'D', 'dx2y2'): pds * c, \
                          ('py2', 'U', 'dx2y2'): -pds * c, \
                          ('py1', 'R', 'dxy'): -pdp, \
                          ('py1', 'L', 'dxy'): pdp, \
                          ('px2', 'D', 'dxy'): pdp, \
                          ('px2', 'U', 'dxy'): -pdp, \
                          ('pz1', 'R', 'dxz'): -pdp, \
                          ('pz1', 'L', 'dxz'): pdp, \
                          ('pz2', 'D', 'dyz'): pdp, \
                          ('pz2', 'U', 'dyz'): -pdp}
    ########################## tpp below ##############################
    if pam.Norb == 7 or pam.Norb == 4:
        tpp_nn_hop_fac = {('UR', 'px', 'py'): -tpp, \
                          ('UL', 'px', 'py'): tpp, \
                          ('DL', 'px', 'py'): -tpp, \
                          ('DR', 'px', 'py'): tpp}
    elif pam.Norb == 10:
        tpp_nn_hop_fac = {('UR', 'px1', 'px2'): 0.5 * (ppp - pps), \
                          ('UL', 'px1', 'px2'): 0.5 * (ppp - pps), \
                          ('DL', 'px1', 'px2'): 0.5 * (ppp - pps), \
                          ('DR', 'px1', 'px2'): 0.5 * (ppp - pps), \
                          ('UR', 'py1', 'py2'): 0.5 * (ppp - pps), \
                          ('UL', 'py1', 'py2'): 0.5 * (ppp - pps), \
                          ('DL', 'py1', 'py2'): 0.5 * (ppp - pps), \
                          ('DR', 'py1', 'py2'): 0.5 * (ppp - pps), \
                          ('UR', 'px1', 'py2'): -0.5 * (ppp + pps), \
                          ('UL', 'px1', 'py2'): 0.5 * (ppp + pps), \
                          ('DL', 'px1', 'py2'): -0.5 * (ppp + pps), \
                          ('DR', 'px1', 'py2'): 0.5 * (ppp + pps), \
                          ('UR', 'px2', 'py1'): -0.5 * (ppp + pps), \
                          ('UL', 'px2', 'py1'): 0.5 * (ppp + pps), \
                          ('DL', 'px2', 'py1'): -0.5 * (ppp + pps), \
                          ('DR', 'px2', 'py1'): 0.5 * (ppp + pps)}
    elif pam.Norb == 12:
        tpp_nn_hop_fac = {('UR', 'px1', 'px2'): 0.5 * (ppp - pps), \
                          ('UL', 'px1', 'px2'): 0.5 * (ppp - pps), \
                          ('DL', 'px1', 'px2'): 0.5 * (ppp - pps), \
                          ('DR', 'px1', 'px2'): 0.5 * (ppp - pps), \
                          ('UR', 'py1', 'py2'): 0.5 * (ppp - pps), \
                          ('UL', 'py1', 'py2'): 0.5 * (ppp - pps), \
                          ('DL', 'py1', 'py2'): 0.5 * (ppp - pps), \
                          ('DR', 'py1', 'py2'): 0.5 * (ppp - pps), \
                          ('UR', 'px1', 'py2'): -0.5 * (ppp + pps), \
                          ('UL', 'px1', 'py2'): 0.5 * (ppp + pps), \
                          ('DL', 'px1', 'py2'): -0.5 * (ppp + pps), \
                          ('DR', 'px1', 'py2'): 0.5 * (ppp + pps), \
                          ('UR', 'px2', 'py1'): -0.5 * (ppp + pps), \
                          ('UL', 'px2', 'py1'): 0.5 * (ppp + pps), \
                          ('DL', 'px2', 'py1'): -0.5 * (ppp + pps), \
                          ('DR', 'px2', 'py1'): 0.5 * (ppp + pps), \
                          ('UR', 'pz1', 'pz2'): ppp, \
                          ('UL', 'pz1', 'pz2'): ppp, \
                          ('DL', 'pz1', 'pz2'): ppp, \
                          ('DR', 'pz1', 'pz2'): ppp}

    return tpd_nn_hop_dir, tpd_orbs, tpd_nn_hop_fac, tpp_nn_hop_fac


def set_tz(Norb, if_tz_exist, tz_a1a1, tz_b1b1):  #条件Ni向下

    if pam.Norb == 4:
        if if_tz_exist == 0:
            tz_fac = {('px', 'px'): tz_b1b1, \
                      ('py', 'py'): tz_b1b1, \
                      ('d3z2r2', 'd3z2r2'): tz_a1a1, \
                      ('dx2y2', 'dx2y2'): tz_b1b1}
        if if_tz_exist == 1:
            tz_fac = {('d3z2r2', 'd3z2r2'): tz_a1a1, \
                      ('dx2y2', 'dx2y2'): tz_b1b1}
        if if_tz_exist == 2:
            tz_fac = {('d3z2r2', 'd3z2r2'): tz_a1a1}

    if pam.Norb == 7:
        if if_tz_exist == 0:
            tz_fac = {('px', 'px'): tz_b1b1, \
                      ('py', 'py'): tz_b1b1, \
                      ('d3z2r2', 'd3z2r2'): tz_a1a1, \
                      ('dx2y2', 'dx2y2'): tz_b1b1, \
                      ('dxy', 'dxy'): tz_b1b1}
        if if_tz_exist == 1:
            tz_fac = {('d3z2r2', 'd3z2r2'): tz_a1a1, \
                      ('dx2y2', 'dx2y2'): tz_b1b1, \
                      ('dxy', 'dxy'): tz}
        if if_tz_exist == 2:
            tz_fac = {('d3z2r2', 'd3z2r2'): tz_a1a1}

    if pam.Norb == 10:
        tz_fac = {('px1', 'px1'): tz_b1b1, \
                  ('px2', 'px2'): tz_b1b1, \
                  ('py1', 'py1'): tz_b1b1, \
                  ('py2', 'py2'): tz_b1b1, \
                  ('d3z2r2', 'd3z2r2'): tz_a1a1, \
                  ('dx2y2', 'dx2y2'): tz_b1b1, \
                  ('dxy', 'dxy'): tz_b1b1}
    if pam.Norb == 11:
        tz_fac = {('px1', 'px1'): tz_b1b1, \
                  ('px2', 'px2'): tz_b1b1, \
                  ('py1', 'py1'): tz_b1b1, \
                  ('py2', 'py2'): tz_b1b1, \
                  ('pz1', 'pz1'): -tz_a1a1, \
                  ('pz2', 'pz2'): -tz_a1a1, \
                  ('d3z2r2', 'd3z2r2'): tz_a1a1, \
                  ('dx2y2', 'dx2y2'): tz_b1b1, \
                  ('dxy', 'dxy'): tz_b1b1}

    return tz_fac


def get_interaction_mat(A, sym):
    '''
    Get d-d Coulomb and exchange interaction matrix
    total_spin based on lat.spin_int: up:1 and dn:0

    Rotating by 90 degrees, x goes to y and indeed y goes to -x so that this basically interchanges
    spatial wave functions of two holes and can introduce - sign (for example (dxz, dyz)).
    But one has to look at what such a rotation does to the Slater determinant state of two holes.
    Remember the triplet state is (|up,down> +|down,up>)/sqrt2 so in interchanging the holes
    the spin part remains positive so the spatial part must be negative.
    For the singlet state interchanging the electrons the spin part changes sign so the spatial part can stay unchanged.

    Triplets cannot occur for two holes in the same spatial wave function while only singlets only can
    But both singlets and triplets can occur if the two holes have orthogonal spatial wave functions
    and they will differ in energy by the exchange terms

    ee denotes xz,xz or xz,yz depends on which integral <ab|1/r_12|cd> is nonzero, see handwritten notes

    AorB_sym = +-1 is used to label if the state is (e1e1+e2e2)/sqrt(2) or (e1e1-e2e2)/sqrt(2)
    For syms (in fact, all syms except 1A1 and 1B1) without the above two states, AorB_sym is set to be 0

    Here different from all other codes, change A by A/2 to decrease the d8 energy to some extent
    the remaining A/2 is shifted to d10 states, see George's email on Jun.21, 2021
    and also set_edepeOs subroutine
    '''
    B = pam.B
    C = pam.C

    # not useful if treat 1A1 and 1B1 as correct ee states as (exex +- eyey)/sqrt(2)

    if sym == '1AB1':
        fac = np.sqrt(6)
        Stot = 0
        Sz_set = [0]
        state_order = {('d3z2r2', 'd3z2r2'): 0, \
                       ('dx2y2', 'dx2y2'): 1, \
                       ('dxy', 'dxy'): 2, \
                       ('dxz', 'dxz'): 3, \
                       ('dyz', 'dyz'): 4, \
                       ('d3z2r2', 'dx2y2'): 5}
        interaction_mat = [[A + 4. * B + 3. * C, 4. * B + C, 4. * B + C, B + C, B + C, 0], \
                           [4. * B + C, A + 4. * B + 3. * C, C, 3. * B + C, 3. * B + C, 0], \
                           [4. * B + C, C, A + 4. * B + 3. * C, 3. * B + C, 3. * B + C, 0], \
                           [B + C, 3. * B + C, 3. * B + C, A + 4. * B + 3. * C, 3. * B + C, B * fac], \
                           [B + C, 3. * B + C, 3. * B + C, 3. * B + C, A + 4. * B + 3. * C, -B * fac], \
                           [0, 0, 0, B * fac, -B * fac, A + 2. * C]]
    if sym == '1A1':
        Stot = 0
        Sz_set = [0]
        AorB_sym = 1
        fac = np.sqrt(2)

        state_order = {('d3z2r2', 'd3z2r2'): 0, \
                       ('dx2y2', 'dx2y2'): 1, \
                       ('dxy', 'dxy'): 2, \
                       ('dxz', 'dxz'): 3, \
                       ('dyz', 'dyz'): 3}
        interaction_mat = [[A + 4. * B + 3. * C, 4. * B + C, 4. * B + C, fac * (B + C)], \
                           [4. * B + C, A + 4. * B + 3. * C, C, fac * (3. * B + C)], \
                           [4. * B + C, C, A + 4. * B + 3. * C, fac * (3. * B + C)], \
                           [fac * (B + C), fac * (3. * B + C), fac * (3. * B + C), A + 7. * B + 4. * C]]
    if sym == '1B1':
        Stot = 0
        Sz_set = [0]
        AorB_sym = -1
        fac = np.sqrt(3)
        state_order = {('d3z2r2', 'dx2y2'): 0, \
                       ('dxz', 'dxz'): 1, \
                       ('dyz', 'dyz'): 1}
        interaction_mat = [[A + 2. * C, 2. * B * fac], \
                           [2. * B * fac, A + B + 2. * C]]
    if sym == '1A2':
        Stot = 0
        Sz_set = [0]
        AorB_sym = 0
        state_order = {('dx2y2', 'dxy'): 0}
        interaction_mat = [[A + 4. * B + 2. * C]]
    if sym == '3A2':
        Stot = 1
        Sz_set = [-1, 0, 1]
        AorB_sym = 0
        state_order = {('dx2y2', 'dxy'): 0, \
                       ('dxz', 'dyz'): 1}
        interaction_mat = [[A + 4. * B, 6. * B], \
                           [6. * B, A - 5. * B]]
    if sym == '3B1':
        Stot = 1
        Sz_set = [-1, 0, 1]
        AorB_sym = 0
        state_order = {('d3z2r2', 'dx2y2'): 0}
        interaction_mat = [[A - 8. * B]]
    if sym == '1B2':
        Stot = 0
        Sz_set = [0]
        AorB_sym = 0
        fac = np.sqrt(3)
        state_order = {('d3z2r2', 'dxy'): 0, \
                       ('dxz', 'dyz'): 1}
        interaction_mat = [[A + 2. * C, 2. * B * fac], \
                           [2. * B * fac, A + B + 2. * C]]
    if sym == '3B2':
        Stot = 1
        Sz_set = [-1, 0, 1]
        AorB_sym = 0
        state_order = {('d3z2r2', 'dxy'): 0}
        interaction_mat = [[A - 8. * B]]
    if sym == '1E':
        Stot = 0
        Sz_set = [0]
        AorB_sym = 0
        fac = np.sqrt(3)
        state_order = {('d3z2r2', 'dxz'): 0, \
                       ('d3z2r2', 'dyz'): 1, \
                       ('dx2y2', 'dxz'): 2, \
                       ('dx2y2', 'dyz'): 3, \
                       ('dxy', 'dxz'): 4, \
                       ('dxy', 'dyz'): 5}
        interaction_mat = [[A + 3. * B + 2. * C, 0, -B * fac, 0, 0, -B * fac], \
                           [0, A + 3. * B + 2. * C, 0, B * fac, -B * fac, 0], \
                           [-B * fac, 0, A + B + 2. * C, 0, 0, -3. * B], \
                           [0, B * fac, 0, A + B + 2. * C, 3. * B, 0], \
                           [0, -B * fac, 0, 3. * B, A + B + 2. * C, 0], \
                           [-B * fac, 0, -3. * B, 0, 0, A + B + 2. * C]]
    if sym == '3E':
        Stot = 1
        Sz_set = [-1, 0, 1]
        AorB_sym = 0
        fac = np.sqrt(3)
        state_order = {('d3z2r2', 'dxz'): 0, \
                       ('d3z2r2', 'dyz'): 1, \
                       ('dx2y2', 'dxz'): 2, \
                       ('dx2y2', 'dyz'): 3, \
                       ('dxy', 'dxz'): 4, \
                       ('dxy', 'dyz'): 5}
        interaction_mat = [[A + B, 0, -3. * B * fac, 0, 0, -3. * B * fac], \
                           [0, A + B, 0, 3. * B * fac, -3. * B * fac, 0], \
                           [-3. * B * fac, 0, A - 5. * B, 0, 0, 3. * B], \
                           [0, 3. * B * fac, 0, A - 5. * B, -3. * B, 0], \
                           [0, -3. * B * fac, 0, -3. * B, A - 5. * B, 0], \
                           [-3. * B * fac, 0, 3. * B, 0, 0, A - 5. * B]]

    return state_order, interaction_mat, Stot, Sz_set, AorB_sym


def set_matrix_element(row, col, data, new_state, col_index, VS, element):
    '''
    Helper function that is used to set elements of a matrix using the
    sps coo format.

    Parameters
    ----------
    row: python list containing row indices
    col: python list containing column indices
    data: python list containing non-zero matrix elements
    col_index: column index that is to be appended to col
    new_state: new state corresponding to the row index that is to be
        appended.
    VS: VariationalSpace class from the module variationalSpace
    element: (complex) matrix element that is to be appended to data.

    Returns
    -------
    None, but appends values to row, col, data.
    '''
    row_index = VS.get_index(new_state)
    if row_index is not None:
        data.append(element)
        row.append(row_index)
        col.append(col_index)


def create_tpd_nn_matrix(VS, tpd_nn_hop_dir, tpd_orbs, tpd_nn_hop_fac):
    '''
    Create nearest neighbor (NN) pd hopping part of the Hamiltonian
    Only hole can hop with tpd

    Parameters
    ----------
    VS: VariationalSpace class from the module variationalSpace

    Returns
    -------
    matrix: (sps coo format) t_pd hopping part of the Hamiltonian without
        the prefactor t_pd.

    Note from the sps documentation
    -------------------------------
    By default when converting to CSR or CSC format, duplicate (i,j)
    entries will be summed together
    '''
    print("start create_tpd_nn_matrix")
    print("==========================")

    dim = VS.dim
    tpd_keys = tpd_nn_hop_fac.keys()
    data = []
    row = []
    col = []
    for i in range(0, dim):
        start_state = VS.get_state(VS.lookup_tbl[i])

        # double check which cost some time, might not necessary
        assert VS.get_uid(start_state) == VS.lookup_tbl[i]

        s1 = start_state['hole1_spin']
        s2 = start_state['hole2_spin']
        s3 = start_state['hole3_spin']
        s4 = start_state['hole4_spin']
        s5 = start_state['hole5_spin']
        orb1 = start_state['hole1_orb']
        orb2 = start_state['hole2_orb']
        orb3 = start_state['hole3_orb']
        orb4 = start_state['hole4_orb']
        orb5 = start_state['hole5_orb']
        x1, y1, z1 = start_state['hole1_coord']
        x2, y2, z2 = start_state['hole2_coord']
        x3, y3, z3 = start_state['hole3_coord']
        x4, y4, z4 = start_state['hole4_coord']
        x5, y5, z5 = start_state['hole5_coord']

        # hole 1 hops: some d-orbitals might have no tpd
        if orb1 in tpd_orbs:
            for dir_ in tpd_nn_hop_dir[orb1]:
                vx, vy, vz = directions_to_vecs[dir_]
                orbs1 = lat.get_unit_cell_rep(x1 + vx, y1 + vy, z1 + vz)
                if orbs1 == ['NotOnSublattice']:
                    continue

                if not vs.check_in_vs_condition1(x1 + vx, y1 + vy, x2, y2, x3, y3, x4, y4, x5, y5):
                    continue

                # consider t_pd for all cases; when up hole hops, dn hole should not change orb
                for o1 in orbs1:
                    if o1 not in tpd_orbs:
                        continue

                    # consider Pauli principle
                    slabel = [s1, o1, x1 + vx, y1 + vy, z1 + vz, s2, orb2, x2, y2, z2, s3, orb3, x3, y3, z3, s4, orb4,
                              x4, y4, z4, s5, orb5, x5, y5, z5]
                    if not vs.check_Pauli(slabel):
                        continue

                    tmp_state = vs.create_state(slabel)
                    new_state, ph, _ = vs.make_state_canonical(tmp_state)
                    #new_state,ph = vs.make_state_canonical_old(tmp_state)

                    o12 = tuple([orb1, dir_, o1])
                    if o12 in tpd_keys:
                        set_matrix_element(row, col, data, new_state, i, VS, tpd_nn_hop_fac[o12] * ph)

        # hole 2 hops; some d-orbitals might have no tpd
        if orb2 in tpd_orbs:
            for dir_ in tpd_nn_hop_dir[orb2]:
                vx, vy, vz = directions_to_vecs[dir_]
                orbs2 = lat.get_unit_cell_rep(x2 + vx, y2 + vy, z2 + vz)
                if orbs2 == ['NotOnSublattice']:
                    continue

                if not vs.check_in_vs_condition1(x1, y1, x2 + vx, y2 + vy, x3, y3, x4, y4, x5, y5):
                    continue

                for o2 in orbs2:
                    if o2 not in tpd_orbs:
                        continue

                    # consider Pauli principle
                    slabel = [s1, orb1, x1, y1, z1, s2, o2, x2 + vx, y2 + vy, z2 + vz, s3, orb3, x3, y3, z3, s4, orb4,
                              x4, y4, z4, s5, orb5, x5, y5, z5]
                    if not vs.check_Pauli(slabel):
                        continue

                    tmp_state = vs.create_state(slabel)
                    new_state, ph, _ = vs.make_state_canonical(tmp_state)
                    #new_state,ph = vs.make_state_canonical_old(tmp_state)

                    o12 = tuple([orb2, dir_, o2])
                    if o12 in tpd_keys:
                        set_matrix_element(row, col, data, new_state, i, VS, tpd_nn_hop_fac[o12] * ph)

        if orb3 in tpd_orbs:
            for dir_ in tpd_nn_hop_dir[orb3]:
                vx, vy, vz = directions_to_vecs[dir_]
                orbs3 = lat.get_unit_cell_rep(x3 + vx, y3 + vy, z3 + vz)
                if orbs3 == ['NotOnSublattice']:
                    continue

                if not vs.check_in_vs_condition1(x1, y1, x2, y2, x3 + vx, y3 + vy, x4, y4, x5, y5):
                    continue

                for o3 in orbs3:
                    if o3 not in tpd_orbs:
                        continue

                    # consider Pauli principle
                    slabel = [s1, orb1, x1, y1, z1, s2, orb2, x2, y2, z2, s3, o3, x3 + vx, y3 + vy, z3 + vz, s4, orb4,
                              x4, y4, z4, s5, orb5, x5, y5, z5]
                    if not vs.check_Pauli(slabel):
                        continue

                    tmp_state = vs.create_state(slabel)
                    new_state, ph, _ = vs.make_state_canonical(tmp_state)
                    #new_state,ph = vs.make_state_canonical_old(tmp_state)

                    o12 = tuple([orb3, dir_, o3])
                    if o12 in tpd_keys:
                        set_matrix_element(row, col, data, new_state, i, VS, tpd_nn_hop_fac[o12] * ph)

        if orb4 in tpd_orbs:
            for dir_ in tpd_nn_hop_dir[orb4]:
                vx, vy, vz = directions_to_vecs[dir_]
                orbs4 = lat.get_unit_cell_rep(x4 + vx, y4 + vy, z4 + vz)
                if orbs4 == ['NotOnSublattice']:
                    continue

                if not vs.check_in_vs_condition1(x1, y1, x2, y2, x3, y3, x4 + vx, y4 + vy, x5, y5):
                    continue

                for o4 in orbs4:
                    if o4 not in tpd_orbs:
                        continue

                    # consider Pauli principle
                    slabel = [s1, orb1, x1, y1, z1, s2, orb2, x2, y2, z2, s3, orb3, x3, y3, z3, s4, o4, x4 + vx,
                              y4 + vy, z4 + vz, s5, orb5, x5, y5, z5]
                    if not vs.check_Pauli(slabel):
                        continue

                    tmp_state = vs.create_state(slabel)
                    new_state, ph, _ = vs.make_state_canonical(tmp_state)
                    #new_state,ph = vs.make_state_canonical_old(tmp_state)

                    o12 = tuple([orb4, dir_, o4])
                    if o12 in tpd_keys:
                        set_matrix_element(row, col, data, new_state, i, VS, tpd_nn_hop_fac[o12] * ph)

        if orb5 in tpd_orbs:
            for dir_ in tpd_nn_hop_dir[orb5]:
                vx, vy, vz = directions_to_vecs[dir_]
                orbs5 = lat.get_unit_cell_rep(x5 + vx, y5 + vy, z5 + vz)
                if orbs5 == ['NotOnSublattice']:
                    continue

                if not vs.check_in_vs_condition1(x1, y1, x2, y2, x3, y3, x4, y4, x5 + vx, y5 + vy):
                    continue

                for o5 in orbs5:
                    if o5 not in tpd_orbs:
                        continue

                    # consider Pauli principle
                    slabel = [s1, orb1, x1, y1, z1, s2, orb2, x2, y2, z2, s3, orb3, x3, y3, z3, s4, orb4, x4, y4, z4,
                              s5, o5, x5 + vx, y5 + vy, z5 + vz]
                    if not vs.check_Pauli(slabel):
                        continue

                    tmp_state = vs.create_state(slabel)
                    new_state, ph, _ = vs.make_state_canonical(tmp_state)
                    #new_state,ph = vs.make_state_canonical_old(tmp_state)

                    o12 = tuple([orb5, dir_, o5])
                    if o12 in tpd_keys:
                        set_matrix_element(row, col, data, new_state, i, VS, tpd_nn_hop_fac[o12] * ph)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)

    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn)
    #assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data, (row, col)), shape=(dim, dim))

    return out


def create_tpp_nn_matrix(VS, tpp_nn_hop_fac):
    '''
    similar to comments in create_tpp_nn_matrix
    '''
    print("start create_tpp_nn_matrix")
    print("==========================")

    dim = VS.dim
    tpp_orbs = tpp_nn_hop_fac.keys()
    data = []
    row = []
    col = []
    for i in range(0, dim):
        start_state = VS.get_state(VS.lookup_tbl[i])

        s1 = start_state['hole1_spin']
        s2 = start_state['hole2_spin']
        s3 = start_state['hole3_spin']
        s4 = start_state['hole4_spin']
        s5 = start_state['hole5_spin']
        orb1 = start_state['hole1_orb']
        orb2 = start_state['hole2_orb']
        orb3 = start_state['hole3_orb']
        orb4 = start_state['hole4_orb']
        orb5 = start_state['hole5_orb']
        x1, y1, z1 = start_state['hole1_coord']
        x2, y2, z2 = start_state['hole2_coord']
        x3, y3, z3 = start_state['hole3_coord']
        x4, y4, z4 = start_state['hole4_coord']
        x5, y5, z5 = start_state['hole5_coord']

        # hole1 hops: only p-orbitals has t_pp
        if orb1 in pam.O_orbs:
            for dir_ in tpp_nn_hop_dir:
                vx, vy, vz = directions_to_vecs[dir_]
                orbs1 = lat.get_unit_cell_rep(x1 + vx, y1 + vy, z1 + vz)

                if orbs1 != pam.O1_orbs and orbs1 != pam.O2_orbs:
                    continue

                if not vs.check_in_vs_condition1(x1 + vx, y1 + vy, x2, y2, x3, y3, x4, y4, x5, y5):
                    continue

                # consider t_pp for all cases; when one hole hops, the other hole should not change orb
                for o1 in orbs1:
                    # consider Pauli principle
                    slabel = [s1, o1, x1 + vx, y1 + vy, z1 + vz, s2, orb2, x2, y2, z2, s3, orb3, x3, y3, z3, s4, orb4,
                              x4, y4, z4, s5, orb5, x5, y5, z5]
                    if not vs.check_Pauli(slabel):
                        continue

                    tmp_state = vs.create_state(slabel)
                    new_state, ph, _ = vs.make_state_canonical(tmp_state)

                    o12 = sorted([orb1, dir_, o1])
                    o12 = tuple(o12)
                    if o12 in tpp_orbs:
                        set_matrix_element(row, col, data, new_state, i, VS, tpp_nn_hop_fac[o12] * ph)

        # hole 2 hops, only p-orbitals has t_pp
        if orb2 in pam.O_orbs:
            for dir_ in tpp_nn_hop_dir:
                vx, vy, vz = directions_to_vecs[dir_]
                orbs2 = lat.get_unit_cell_rep(x2 + vx, y2 + vy, z2 + vz)

                if orbs2 != pam.O1_orbs and orbs2 != pam.O2_orbs:
                    continue

                if not vs.check_in_vs_condition1(x1, y1, x2 + vx, y2 + vy, x3, y3, x4, y4, x5, y5):
                    continue

                for o2 in orbs2:
                    # consider Pauli principle
                    slabel = [s1, orb1, x1, y1, z1, s2, o2, x2 + vx, y2 + vy, z2 + vz, s3, orb3, x3, y3, z3, s4, orb4,
                              x4, y4, z4, s5, orb5, x5, y5, z5]
                    if not vs.check_Pauli(slabel):
                        continue

                    tmp_state = vs.create_state(slabel)
                    new_state, ph, _ = vs.make_state_canonical(tmp_state)
                    #new_state,ph = vs.make_state_canonical_old(tmp_state)

                    o12 = sorted([orb2, dir_, o2])
                    o12 = tuple(o12)
                    if o12 in tpp_orbs:
                        set_matrix_element(row, col, data, new_state, i, VS, tpp_nn_hop_fac[o12] * ph)

        # hole 3 hops, only p-orbitals has t_pp
        if orb3 in pam.O_orbs:
            for dir_ in tpp_nn_hop_dir:
                vx, vy, vz = directions_to_vecs[dir_]
                orbs3 = lat.get_unit_cell_rep(x3 + vx, y3 + vy, z3 + vz)

                if orbs3 != pam.O1_orbs and orbs3 != pam.O2_orbs:
                    continue

                if not vs.check_in_vs_condition1(x1, y1, x2, y2, x3 + vx, y3 + vy, x4, y4, x5, y5):
                    continue

                for o3 in orbs3:
                    # consider Pauli principle
                    slabel = [s1, orb1, x1, y1, z1, s2, orb2, x2, y2, z2, s3, o3, x3 + vx, y3 + vy, z3 + vz, s4, orb4,x4, y4, z4, s5, orb5, x5, y5, z5]
                    if not vs.check_Pauli(slabel):
                        continue

                    tmp_state = vs.create_state(slabel)
                    new_state, ph, _ = vs.make_state_canonical(tmp_state)
                    #new_state,ph = vs.make_state_canonical_old(tmp_state)

                    o12 = sorted([orb3, dir_, o3])
                    o12 = tuple(o12)
                    if o12 in tpp_orbs:
                        set_matrix_element(row, col, data, new_state, i, VS, tpp_nn_hop_fac[o12] * ph)

        # hole 4 hops, only p-orbitals has t_pp
        if orb4 in pam.O_orbs:
            for dir_ in tpp_nn_hop_dir:
                vx, vy, vz = directions_to_vecs[dir_]
                orbs4 = lat.get_unit_cell_rep(x4 + vx, y4 + vy, z4 + vz)

                if orbs4 != pam.O1_orbs and orbs4 != pam.O2_orbs:
                    continue

                if not vs.check_in_vs_condition1(x1, y1, x2, y2, x3, y3, x4 + vx, y4 + vy, x5, y5):
                    continue

                for o4 in orbs4:
                    # consider Pauli principle
                    slabel = [s1, orb1, x1, y1, z1, s2, orb2, x2, y2, z2, s3, orb3, x3, y3, z3, s4, o4, x4 + vx,y4 + vy, z4 + vz, s5, orb5, x5, y5, z5]
                    if not vs.check_Pauli(slabel):
                        continue

                    tmp_state = vs.create_state(slabel)
                    new_state, ph, _ = vs.make_state_canonical(tmp_state)
                    #new_state,ph = vs.make_state_canonical_old(tmp_state)

                    o12 = sorted([orb4, dir_, o4])
                    o12 = tuple(o12)
                    if o12 in tpp_orbs:
                        set_matrix_element(row, col, data, new_state, i, VS, tpp_nn_hop_fac[o12] * ph)

                        # hole 5 hops, only p-orbitals has t_pp
        if orb5 in pam.O_orbs:
            for dir_ in tpp_nn_hop_dir:
                vx, vy, vz = directions_to_vecs[dir_]
                orbs5 = lat.get_unit_cell_rep(x5 + vx, y5 + vy, z5 + vz)

                if orbs5 != pam.O1_orbs and orbs5 != pam.O2_orbs:
                    continue

                if not vs.check_in_vs_condition1(x1, y1, x2, y2, x3, y3, x4, y4, x5 + vx, y5 + vy):
                    continue

                for o5 in orbs5:
                    # consider Pauli principle
                    slabel = [s1, orb1, x1, y1, z1, s2, orb2, x2, y2, z2, s3, orb3, x3, y3, z3, s4, orb4, x4, y4, z4,s5, o5, x5 + vx, y5 + vy, z5 + vz]
                    if not vs.check_Pauli(slabel):
                        continue

                    tmp_state = vs.create_state(slabel)
                    new_state, ph, _ = vs.make_state_canonical(tmp_state)
                    #new_state,ph = vs.make_state_canonical_old(tmp_state)

                    o12 = sorted([orb5, dir_, o5])
                    o12 = tuple(o12)
                    if o12 in tpp_orbs:
                        set_matrix_element(row, col, data, new_state, i, VS, tpp_nn_hop_fac[o12] * ph)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)

    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn)
    # assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data, (row, col)), shape=(dim, dim))

    return out


def create_tz_matrix(VS, tz_fac):
    '''
    Just think straight up and down
    It is assumed that the interlayer transition does not depend on the orbit
    '''
    print("start create_tz_matrix")
    print("==========================")

    dim = VS.dim
    data = []
    row = []
    col = []
    tz_orbs = tz_fac.keys()
    for i in range(0, dim):
        start_state = VS.get_state(VS.lookup_tbl[i])

        # double check which cost some time, might not necessary
        assert VS.get_uid(start_state) == VS.lookup_tbl[i]

        s1 = start_state['hole1_spin']
        s2 = start_state['hole2_spin']
        s3 = start_state['hole3_spin']
        s4 = start_state['hole4_spin']
        s5 = start_state['hole5_spin']
        orb1 = start_state['hole1_orb']
        orb2 = start_state['hole2_orb']
        orb3 = start_state['hole3_orb']
        orb4 = start_state['hole4_orb']
        orb5 = start_state['hole5_orb']
        x1, y1, z1 = start_state['hole1_coord']
        x2, y2, z2 = start_state['hole2_coord']
        x3, y3, z3 = start_state['hole3_coord']
        x4, y4, z4 = start_state['hole4_coord']
        x5, y5, z5 = start_state['hole5_coord']

        # hole 1 hops: some d-orbitals might have no tpd
        orbs1 = lat.get_unit_cell_rep(x1, y1, 2 - z1)
        if orbs1 == ['NotOnSublattice']:
            continue

        # consider t_z for all cases; when up hole hops, dn hole should not change orb
        for o1 in orbs1:
            if o1 != orb1:
                continue

            o12 = [o1, orb1]
            o12 = tuple(o12)
            if o12 in tz_orbs:
                slabel = [s1, o1, x1, y1, 2 - z1, s2, orb2, x2, y2, z2, s3, orb3, x3, y3, z3, s4, orb4, x4, y4, z4, s5,
                          orb5, x5, y5, z5]
                # consider Pauli principle
                if not vs.check_Pauli(slabel):
                    continue
                tmp_state = vs.create_state(slabel)
                new_state, ph, _ = vs.make_state_canonical(tmp_state)
                #new_state,ph = vs.make_state_canonical_old(tmp_state)

                set_matrix_element(row, col, data, new_state, i, VS, tz_fac[o12] * ph)

        # hole 2 hops; some d-orbitals might have no tpd
        orbs2 = lat.get_unit_cell_rep(x2, y2, 2 - z2)
        if orbs2 == ['NotOnSublattice']:
            continue

        # consider t_pd for all cases; when up hole hops, dn hole should not change orb
        for o2 in orbs2:
            if o2 != orb2:
                continue

            o12 = [o2, orb2]
            o12 = tuple(o12)
            if o12 in tz_orbs:
                slabel = [s1, orb1, x1, y1, z1, s2, o2, x2, y2, 2 - z2, s3, orb3, x3, y3, z3, s4, orb4, x4, y4, z4, s5,
                          orb5, x5, y5, z5]
                # consider Pauli principle
                if not vs.check_Pauli(slabel):
                    continue
                tmp_state = vs.create_state(slabel)
                new_state, ph, _ = vs.make_state_canonical(tmp_state)
                #new_state,ph = vs.make_state_canonical_old(tmp_state)

                set_matrix_element(row, col, data, new_state, i, VS, tz_fac[o12] * ph)

        # hole 3 hops; some d-orbitals might have no tpd
        orbs3 = lat.get_unit_cell_rep(x3, y3, 2 - z3)
        if orbs3 == ['NotOnSublattice']:
            continue

        for o3 in orbs3:
            if o3 != orb3:
                continue

            o12 = [o3, orb3]
            o12 = tuple(o12)
            if o12 in tz_orbs:
                slabel = [s1, orb1, x1, y1, z1, s2, orb2, x2, y2, z2, s3, o3, x3, y3, 2 - z3, s4, orb4, x4, y4, z4, s5,
                          orb5, x5, y5, z5]
                # consider Pauli principle
                if not vs.check_Pauli(slabel):
                    continue
                tmp_state = vs.create_state(slabel)
                new_state, ph, _ = vs.make_state_canonical(tmp_state)
                #new_state,ph = vs.make_state_canonical_old(tmp_state)

                set_matrix_element(row, col, data, new_state, i, VS, tz_fac[o12] * ph)

        # hole 4 hops; some d-orbitals might have no tpd
        orbs4 = lat.get_unit_cell_rep(x4, y4, 2 - z4)
        if orbs4 == ['NotOnSublattice']:
            continue

        for o4 in orbs4:
            if o4 != orb4:
                continue

            o12 = [o4, orb4]
            o12 = tuple(o12)
            if o12 in tz_orbs:
                slabel = [s1, orb1, x1, y1, z1, s2, orb2, x2, y2, z2, s3, orb3, x3, y3, z3, s4, o4, x4, y4, 2 - z4, s5,
                          orb5, x5, y5, z5]
                # consider Pauli principle
                if not vs.check_Pauli(slabel):
                    continue
                tmp_state = vs.create_state(slabel)
                new_state, ph, _ = vs.make_state_canonical(tmp_state)
                #new_state,ph = vs.make_state_canonical_old(tmp_state)

                set_matrix_element(row, col, data, new_state, i, VS, tz_fac[o12] * ph)

                # hole 5 hops; some d-orbitals might have no tpd
        orbs5 = lat.get_unit_cell_rep(x5, y5, 2 - z5)
        if orbs5 == ['NotOnSublattice']:
            continue

        for o5 in orbs5:
            if o5 != orb5:
                continue

            o12 = [o5, orb5]
            o12 = tuple(o12)
            if o12 in tz_orbs:
                slabel = [s1, orb1, x1, y1, z1, s2, orb2, x2, y2, z2, s3, orb3, x3, y3, z3, s4, orb4, x4, y4, z4, s5,
                          o5, x5, y5, 2 - z5]
                # consider Pauli principle
                if not vs.check_Pauli(slabel):
                    continue
                tmp_state = vs.create_state(slabel)
                new_state, ph, _ = vs.make_state_canonical(tmp_state)
                #new_state,ph = vs.make_state_canonical_old(tmp_state)

                set_matrix_element(row, col, data, new_state, i, VS, tz_fac[o12] * ph)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)

    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn)
    #assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data, (row, col)), shape=(dim, dim))

    return out


def create_edep_diag_matrix(VS, A, ep):
    '''
    Create diagonal part of the site energies
    '''
    print("start create_edep_diag_matrix")
    print("=============================")
    dim = VS.dim
    data = []
    row = []
    col = []

    for i in range(dim):
        diag_el = 0
        state = VS.get_state(VS.lookup_tbl[i])

        orb1 = state['hole1_orb']
        orb2 = state['hole2_orb']
        orb3 = state['hole3_orb']
        orb4 = state['hole4_orb']
        orb5 = state['hole5_orb']
        x1, y1, z1 = state['hole1_coord']
        x2, y2, z2 = state['hole2_coord']
        x3, y3, z3 = state['hole3_coord']
        x4, y4, z4 = state['hole4_coord']
        x5, y5, z5 = state['hole5_coord']

        diag_el += util.get_orb_edep(orb1, z1, ep)
        diag_el += util.get_orb_edep(orb2, z2, ep)
        diag_el += util.get_orb_edep(orb3, z3, ep)
        diag_el += util.get_orb_edep(orb4, z4, ep)
        diag_el += util.get_orb_edep(orb5, z5, ep)
        Ni_left, Ni_right = util.get_Number_d(state)
        if Ni_left == 0:
            diag_el += 2 * A
        elif Ni_left == 1:
            diag_el += 3 * A / 2
        elif Ni_left == 3:
            diag_el += 3 * A / 2
        if Ni_right == 0:
            diag_el += 2 * A
        elif Ni_right == 1:
            diag_el += 3 * A / 2
        elif Ni_right == 3:
            diag_el += 3 * A / 2


        data.append(diag_el);row.append(i);col.append(i)
    #         print (i, diag_el)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    #     print (min(data))
    #     print (max(data))
    #print len(row), len(col)

    # for ii in range(0,len(row)):
    #     if data[ii]==0:
    #         print ii

    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn)
    #assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data, (row, col)), shape=(dim, dim))

    return out


def get_double_occu_list(VS):
    '''
    Get the list of states that two holes are both d or p-orbitals
    idx, hole3state, dp_orb, dp_pos record detailed info of states
    '''
    dim = VS.dim
    d_Ni1_list = [];idx_Ni1 = [];hole345_Ni1_part = [];double_Ni1_part = [];
    d_Ni2_list = [];idx_Ni2 = [];hole345_Ni2_part = [];double_Ni2_part = [];
    p_list = [];apz_list = []

    for i in range(0, dim):
        state = VS.get_state(VS.lookup_tbl[i])
        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        s3 = state['hole3_spin']
        s4 = state['hole4_spin']
        s5 = state['hole5_spin']
        o1 = state['hole1_orb']
        o2 = state['hole2_orb']
        o3 = state['hole3_orb']
        o4 = state['hole4_orb']
        o5 = state['hole5_orb']
        x1, y1, z1 = state['hole1_coord']
        x2, y2, z2 = state['hole2_coord']
        x3, y3, z3 = state['hole3_coord']
        x4, y4, z4 = state['hole4_coord']
        x5, y5, z5 = state['hole5_coord']

        # find out which two holes are on Ni1/Ni2(Ni1 = (-1, 0, 0), Ni2 = (1, 0, 0)))
        # idx is to label which hole is not on Ni1/Ni2
        if (x1, y1, z1) == (x2, y2, z2):
            if x1 == -1:
                util.get_double_append(i, 345, s1, o1, x1, y1, z1, s2, o2, x2, y2, z2, s3, o3, x3, y3, z3, s4, o4, x4,y4, z4, s5, o5, x5, y5, z5, \
                                       d_Ni1_list, p_list, idx_Ni1, hole345_Ni1_part, double_Ni1_part)
            elif x1 == 1:
                util.get_double_append(i, 345, s1, o1, x1, y1, z1, s2, o2, x2, y2, z2, s3, o3, x3, y3, z3, s4, o4, x4,y4, z4, s5, o5, x5, y5, z5, \
                                       d_Ni2_list, p_list, idx_Ni2, hole345_Ni2_part, double_Ni2_part)

        if (x1, y1, z1) == (x3, y3, z3):
            if x1 == -1:
                util.get_double_append(i, 245, s1, o1, x1, y1, z1, s3, o3, x3, y3, z3, s2, o2, x2, y2, z2, s4, o4, x4,y4, z4, s5, o5, x5, y5, z5, \
                                       d_Ni1_list, p_list, idx_Ni1, hole345_Ni1_part, double_Ni1_part)
            elif x1 == 1:
                util.get_double_append(i, 245, s1, o1, x1, y1, z1, s3, o3, x3, y3, z3, s2, o2, x2, y2, z2, s4, o4, x4,y4, z4, s5, o5, x5, y5, z5, \
                                       d_Ni2_list, p_list, idx_Ni2, hole345_Ni2_part, double_Ni2_part)

        if (x1, y1, z1) == (x4, y4, z4):
            if x1 == -1:
                util.get_double_append(i, 235, s1, o1, x1, y1, z1, s4, o4, x4, y4, z4, s2, o2, x2, y2, z2, s3, o3, x3,y3, z3, s5, o5, x5, y5, z5, \
                                       d_Ni1_list, p_list, idx_Ni1, hole345_Ni1_part, double_Ni1_part)
            elif x1 == 1:
                util.get_double_append(i, 235, s1, o1, x1, y1, z1, s4, o4, x4, y4, z4, s2, o2, x2, y2, z2, s3, o3, x3,y3, z3, s5, o5, x5, y5, z5, \
                                       d_Ni2_list, p_list, idx_Ni2, hole345_Ni2_part, double_Ni2_part)

        if (x2, y2, z2) == (x3, y3, z3):
            if x2 == -1:
                util.get_double_append(i, 145, s2, o2, x2, y2, z2, s3, o3, x3, y3, z3, s1, o1, x1, y1, z1, s4, o4, x4,y4, z4, s5, o5, x5, y5, z5, \
                                       d_Ni1_list, p_list, idx_Ni1, hole345_Ni1_part, double_Ni1_part)
            elif x2 == 1:
                util.get_double_append(i, 145, s2, o2, x2, y2, z2, s3, o3, x3, y3, z3, s1, o1, x1, y1, z1, s4, o4, x4,y4, z4, s5, o5, x5, y5, z5, \
                                       d_Ni2_list, p_list, idx_Ni2, hole345_Ni2_part, double_Ni2_part)

        if (x2, y2, z2) == (x4, y4, z4):
            if x2 == -1:
                util.get_double_append(i, 135, s2, o2, x2, y2, z2, s4, o4, x4, y4, z4, s1, o1, x1, y1, z1, s3, o3, x3,y3, z3, s5, o5, x5, y5, z5, \
                                       d_Ni1_list, p_list, idx_Ni1, hole345_Ni1_part, double_Ni1_part)
            elif x2 == 1:
                util.get_double_append(i, 135, s2, o2, x2, y2, z2, s4, o4, x4, y4, z4, s1, o1, x1, y1, z1, s3, o3, x3,y3, z3, s5, o5, x5, y5, z5, \
                                       d_Ni2_list, p_list, idx_Ni2, hole345_Ni2_part, double_Ni2_part)

        if (x3, y3, z3) == (x4, y4, z4):
            if x3 == -1:
                util.get_double_append(i, 125, s3, o3, x3, y3, z3, s4, o4, x4, y4, z4, s1, o1, x1, y1, z1, s2, o2, x2,y2, z2, s5, o5, x5, y5, z5, \
                                       d_Ni1_list, p_list, idx_Ni1, hole345_Ni1_part, double_Ni1_part)
            elif x3 == 1:
                util.get_double_append(i, 125, s3, o3, x3, y3, z3, s4, o4, x4, y4, z4, s1, o1, x1, y1, z1, s2, o2, x2,y2, z2, s5, o5, x5, y5, z5, \
                                       d_Ni2_list, p_list, idx_Ni2, hole345_Ni2_part, double_Ni2_part)

        if (x1, y1, z1) == (x5, y5, z5):
            if x1 == -1:
                util.get_double_append(i, 234, s1, o1, x1, y1, z1, s5, o5, x5, y5, z5, s2, o2, x2, y2, z2, s3, o3, x3,y3, z3, s4, o4, x4, y4, z4, \
                                       d_Ni1_list, p_list, idx_Ni1, hole345_Ni1_part, double_Ni1_part)
            elif x1 == 1:
                util.get_double_append(i, 234, s1, o1, x1, y1, z1, s5, o5, x5, y5, z5, s2, o2, x2, y2, z2, s3, o3, x3,y3, z3, s4, o4, x4, y4, z4, \
                                       d_Ni2_list, p_list, idx_Ni2, hole345_Ni2_part, double_Ni2_part)

        if (x2, y2, z2) == (x5, y5, z5):
            if x2 == -1:
                util.get_double_append(i, 134, s2, o2, x2, y2, z2, s5, o5, x5, y5, z5, s1, o1, x1, y1, z1, s3, o3, x3,y3, z3, s4, o4, x4, y4, z4, \
                                       d_Ni1_list, p_list, idx_Ni1, hole345_Ni1_part, double_Ni1_part)
            elif x2 == 1:
                util.get_double_append(i, 134, s2, o2, x2, y2, z2, s5, o5, x5, y5, z5, s1, o1, x1, y1, z1, s3, o3, x3,y3, z3, s4, o4, x4, y4, z4, \
                                       d_Ni2_list, p_list, idx_Ni2, hole345_Ni2_part, double_Ni2_part)

        if (x3, y3, z3) == (x5, y5, z5):
            if x3 == -1:
                util.get_double_append(i, 124, s3, o3, x3, y3, z3, s5, o5, x5, y5, z5, s1, o1, x1, y1, z1, s2, o2, x2,y2, z2, s4, o4, x4, y4, z4, \
                                       d_Ni1_list, p_list, idx_Ni1, hole345_Ni1_part, double_Ni1_part)
            elif x3 == 1:
                util.get_double_append(i, 124, s3, o3, x3, y3, z3, s5, o5, x5, y5, z5, s1, o1, x1, y1, z1, s2, o2, x2,y2, z2, s4, o4, x4, y4, z4, \
                                       d_Ni2_list, p_list, idx_Ni2, hole345_Ni2_part, double_Ni2_part)

        if (x4, y4, z4) == (x5, y5, z5):
            if x4 == -1:
                util.get_double_append(i, 123, s4, o4, x4, y4, z4, s5, o5, x5, y5, z5, s1, o1, x1, y1, z1, s2, o2, x2,y2, z2, s3, o3, x3, y3, z3, \
                                       d_Ni1_list, p_list, idx_Ni1, hole345_Ni1_part, double_Ni1_part)
            elif x4 == 1:
                util.get_double_append(i, 123, s4, o4, x4, y4, z4, s5, o5, x5, y5, z5, s1, o1, x1, y1, z1, s2, o2, x2,y2, z2, s3, o3, x3, y3, z3, \
                                       d_Ni2_list, p_list, idx_Ni2, hole345_Ni2_part, double_Ni2_part)

    print("len(d_Ni1_list)", len(d_Ni1_list))
    print("len(d_Ni2_list)", len(d_Ni2_list))
    print("len(p_list)", len(p_list))
    # print("len(apz_list)", len(apz_list))
    print("len(idx_Ni1)", len(idx_Ni1))
    print("len(idx_Ni2)", len(idx_Ni2))

    return d_Ni1_list, idx_Ni1, hole345_Ni1_part, double_Ni1_part, \
        d_Ni2_list, idx_Ni2, hole345_Ni2_part, double_Ni2_part, \
        p_list


def create_interaction_matrix_ALL_syms(VS, d_double, p_double, double_part, idx, hole345_part, S_val, Sz_val, AorB_sym,
                                       A, Upp):
    '''
    Create Coulomb-exchange interaction matrix of d-multiplets including all symmetries

    Loop over all d_double states, find the corresponding sym channel;
    the other loop over all d_double states, if it has same sym channel and S, Sz
    enter into the matrix element

    There are some complications or constraints due to three holes and one Nd electron:
    From H_matrix_reducing_VS file, to set up interaction between states i and j:
    1. i and j belong to the same type, same order of orbitals to label the state (idxi==idxj below)
    2. i and j's spins are same; or L and s should also have same spin
    3. Positions of L and Nd-electron should also be the same

    Generate the states interacting with i for a particular sym instead of finding it
    '''
    t1 = time.time()
    print("start create_interaction_matrix_ALL_syms (five holes)")
    print("======================================================")

    Norb = pam.Norb
    dim = VS.dim
    data = []
    row = []
    col = []
    dd_state_indices = []

    channels = ['1A1', '1A2', '3A2', '1B1', '3B1', '1E', '3E', '1B2', '3B2']

    for sym in channels:
        state_order, interaction_mat, Stot, Sz_set, AorB = get_interaction_mat(A, sym)
        sym_orbs = state_order.keys()
        #print ("orbitals in sym ", sym, "= ", sym_orbs)

        for i, double_id in enumerate(d_double):
            count = []  # temporarily store states interacting with i to avoid double count

            s1 = double_part[i][0]
            o1 = double_part[i][1]
            s2 = double_part[i][5]
            o2 = double_part[i][6]
            dpos = double_part[i][2:5]

            o12 = sorted([o1, o2])
            o12 = tuple(o12)

            # S_val, Sz_val obtained from basis.create_singlet_triplet_basis_change_matrix
            S12 = S_val[double_id]
            Sz12 = Sz_val[double_id]

            # continue only if (o1,o2) is within desired sym
            if o12 not in sym_orbs or S12 != Stot or Sz12 not in Sz_set:
                continue

            if (o1 == o2 == 'dxz' or o1 == o2 == 'dyz') and AorB_sym[double_id] != AorB:
                continue

            # get the corresponding index in sym for setting up matrix element
            idx1 = state_order[o12]

            #             if double_id==1705:
            #                 print(S12,Sz12,AorB_sym[double_id],idx1,AorB)

            '''
            Below: generate len(sym_orbs) states that interact with i for a particular sym
            '''
            for idx2, o34 in enumerate(sym_orbs):
                # ('dyz','dyz') is degenerate with ('dxz','dxz') for D4h
                if pam.Norb == 4:
                    if o34 == ('dyz', 'dyz') or o34 == ('dxz', 'dxz') or o34 == ('dxy', 'dxy') or o34 == ('d3z2r2', 'dxy') or o34 == ('dx2y2', 'dxy') or \
                            o34 == ('d3z2r2', 'dxz') or o34 == ('dx2y2', 'dxz') or o34 == ('dxy', 'dxz') or o34 == ('d3z2r2', 'dyz') or o34 == ('dx2y2', 'dyz') \
                            or o34 == ('dxy', 'dyz') or o34 == ('dxz', 'dyz'):
                        continue

                if o34 == ('dyz', 'dyz'):
                    idx2 -= 1

                # Because VS's make_state_canonical follows the rule of up, dn order
                # then the state like ['up', 'dxy', 0, 0, 0, 'dn', 'dx2y2', 0, 0, 0]'s
                # order is opposite to (dx2y2,dxy) order in interteration_mat
                # Here be careful with o34's order that can be opposite to o12 !!

                for s1 in ('up', 'dn'):
                    for s2 in ('up', 'dn'):
                        if idx[i] == 345:
                            slabel = [s1, o34[0]] + dpos + [s2, o34[1]] + dpos + hole345_part[i][0:5] + hole345_part[i][5:15]
                        if idx[i] == 245:
                            slabel = [s1, o34[0]] + dpos + hole345_part[i][0:5] + [s2, o34[1]] + dpos + hole345_part[i][5:15]
                        if idx[i] == 145:
                            slabel = hole345_part[i][0:5] + [s1, o34[0]] + dpos + [s2, o34[1]] + dpos + hole345_part[i][5:15]
                        if idx[i] == 235:
                            slabel = [s1, o34[0]] + dpos + hole345_part[i][0:10] + [s2, o34[1]] + dpos + hole345_part[ i][10:15]
                        if idx[i] == 135:
                            slabel = hole345_part[i][0:5] + [s1, o34[0]] + dpos + hole345_part[i][5:10] + [s2, o34[1]] + dpos + hole345_part[i][10:15]
                        if idx[i] == 125:
                            slabel = hole345_part[i][0:10] + [s1, o34[0]] + dpos + [s2, o34[1]] + dpos + hole345_part[i][10:15]
                        if idx[i] == 123:
                            slabel = hole345_part[i][0:15] + [s1, o34[0]] + dpos + [s2, o34[1]] + dpos
                        if idx[i] == 124:
                            slabel = hole345_part[i][0:10] + [s1, o34[0]] + dpos + hole345_part[i][10:15] + [s2, o34[1]] + dpos
                        if idx[i] == 134:
                            slabel = hole345_part[i][0:5] + [s1, o34[0]] + dpos + hole345_part[i][5:15] + [s2, o34[1]] + dpos
                        if idx[i] == 234:
                            slabel = [s1, o34[0]] + dpos + hole345_part[i][0:15] + [s2, o34[1]] + dpos

                        if not vs.check_Pauli(slabel):
                            continue

                        tmp_state = vs.create_state(slabel)
                        new_state, _, _ = vs.make_state_canonical(tmp_state)
                        j = VS.get_index(new_state)

                        #                             # debug:
                        #                             if double_id==1705:
                        #                                 tse = new_state['e_spin']
                        #                                 ts1 = new_state['hole1_spin']
                        #                                 ts2 = new_state['hole2_spin']
                        #                                 ts3 = new_state['hole3_spin']
                        #                                 torbe = new_state['e_orb']
                        #                                 torb1 = new_state['hole1_orb']
                        #                                 torb2 = new_state['hole2_orb']
                        #                                 torb3 = new_state['hole3_orb']
                        #                                 txe, tye, tze = new_state['e_coord']
                        #                                 tx1, ty1, tz1 = new_state['hole1_coord']
                        #                                 tx2, ty2, tz2 = new_state['hole2_coord']
                        #                                 tx3, ty3, tz3 = new_state['hole3_coord']
                        #                                 print (j,tse,torbe,txe,tye,tze,ts1,torb1,tx1,ty1,tz1,ts2,torb2,tx2,ty2,tz2,ts3,torb3,tx3,ty3,tz3)

                        if j != None and j not in count:
                            S34 = S_val[j]
                            Sz34 = Sz_val[j]

                            if not (S34 == S12 and Sz34 == Sz12):
                                continue

                            if o34 == ('dxz', 'dxz') or o34 == ('dyz', 'dyz'):
                                if AorB_sym[j] != AorB:
                                    continue

                            # debug
                            #                             if j>=double_id:
                            #                                 if double_id==1705:
                            #                                     print (double_id,o12[0],o12[1],S12,Sz12," ",j,o34[0],o34[1],S34,Sz34," ", \
                            #                                            interaction_mat[idx1][idx2])

                            #                                 f.write('{:.6e}\t{:.6e}\t{:.6e}\n'.format(double_id, j, interaction_mat[idx1][idx2]))

                            val = interaction_mat[idx1][idx2]
                            data.append(val); row.append(double_id); col.append(j)
                            count.append(j)

    # Create Upp matrix for p-orbital multiplets
    if Upp != 0:
        for i in p_double:
            data.append(Upp);
            row.append(i);
            col.append(i)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)

    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn)
    #assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data, (row, col)), shape=(dim, dim))

    print("---create_interaction_matrix_ALL_syms %s seconds ---" % (time.time() - t1))

    return out
