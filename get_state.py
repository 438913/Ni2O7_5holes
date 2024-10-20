'''
Get state index for A(w)
'''
import subprocess
import os
import sys
import time
import shutil
import numpy

import parameters as pam
import hamiltonian_d10U_2 as ham
import lattice as lat
import variational_space as vs 
import utility as util

#####################################################################
def get_d9d9L_state_indices_sym(VS, sym, d_double, S_val, Sz_val, AorB_sym, A):
    '''
    Get d8 state index including one dx2y2 hole
    Note: for transformed basis of singlet/triplet
    index can differ from that in original basis
    '''    
    Norb = pam.Norb
    dim = VS.dim
    d8_state_indices = []
    
    # get info specific for particular sym
    state_order, interaction_mat, Stot, Sz_set, AorB = ham.get_interaction_mat(A, sym)
    sym_orbs = state_order.keys()

    for i in d_double:
        # state is original state but its orbital info remains after basis change
        state = VS.get_state(VS.lookup_tbl[i])
        itype = state['type']
        
        if itype == 'two_hole_one_eh':
            continue
            
        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        o1 = state['hole1_orb']
        o2 = state['hole2_orb']
       
        o12 = sorted([o1,o2])
        o12 = tuple(o12)

        # S_val, Sz_val obtained from basis.create_singlet_triplet_basis_change_matrix
        S12  = S_val[i]
        Sz12 = Sz_val[i]
        
        # continue only if (o1,o2) is within desired sym
        if o12 not in sym_orbs or S12!=Stot or Sz12 not in Sz_set:
            continue
            
        # distinguish (e1e1+e2e2)/sqrt(2) or (e1e1-e2e2)/sqrt(2)
        if (o1==o2=='dxz' or o1==o2=='dyz') and AorB_sym[i]!=AorB:
            continue
            
        if 'dx2y2' in o12:
            # for triplet, only need one Sz state; other Sz states have the same A(w)
            if Sz12==0:
                d8_state_indices.append(i)
                print ("d8_state_indices", i, ", state: S= ", S12, " Sz= ", Sz12, "orb= ", o1,o2)

        # Special syms without b1 orbital:
        #if (sym=='1B2' or sym=='3B2') and 'd3z2r2' in o12 and 'dxy' in o12:
        #    dd_state_indices.append(i)
        #    print "d8_state_indices", i, ", state: S= ", S12, " Sz= ", Sz12, "orb= ", o1,o2
              
    return d8_state_indices

def get_d9L_state_indices(VS, S_val, Sz_val):
    '''
    Get d9L state index including one dx2y2 or dz2 hole
    '''    
    Norb = pam.Norb
    dim = VS.dim
    b1L_state_indices = []; b1L_state_labels = []
    a1L_state_indices = []; a1L_state_labels = []

    for i in range(0,dim):
        # state is original state but its orbital info remains after basis change
        state = VS.get_state(VS.lookup_tbl[i])
        itype = state['type']
        
        if itype == 'two_hole_one_eh':
            continue
            
        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        o1 = state['hole1_orb']
        o2 = state['hole2_orb']
        x1, y1, z1 = state['hole1_coord']
        x2, y2, z2 = state['hole2_coord']

        nNi, nO, dorbs, porbs = util.get_statistic_2orb(o1,o2)
        
        if not (nNi==1 and nO==1):
            continue
        
        # L not far away from Ni impurity
        orbs = [o1,o2]
        xs = [x1,x2]
        ys = [y1,y2]
        idx = orbs.index(porbs[0])
            
        if not (abs(xs[idx]-1.0)<1.e-3 and abs(ys[idx])<1.e-3):
            continue
        
        # S_val, Sz_val obtained from basis.create_singlet_triplet_basis_change_matrix
        S12  = S_val[i]
        Sz12 = Sz_val[i]
        
        # continue only singlet
        if S12!=0:
            continue
            
        if dorbs[0]=='dx2y2':
            # for triplet, only need one Sz state; other Sz states have the same A(w)
            if Sz12==0:
                b1L_state_indices.append(i); b1L_state_labels.append('$S=0, S_z=0, d^9_{x^2-y^2}L$')
                print ("b1L_state_indices", i, ", state: S= ", S12, " Sz= ", Sz12, "orb= ", o1,o2)

        if dorbs[0]=='d3z2r2':
            # for triplet, only need one Sz state; other Sz states have the same A(w)
            if Sz12==0:
                a1L_state_indices.append(i); a1L_state_labels.append('$S=0, S_z=0, d^9_{z^2}L$')
                print ("a1L_state_indices", i, ", state: S= ", S12, " Sz= ", Sz12, "orb= ", o1,o2)
                 
    return b1L_state_indices, a1L_state_indices, b1L_state_labels, a1L_state_labels

def get_d10L2_state_indices(VS, S_val, Sz_val):
    '''
    Get d10L2 state index two L holes
    '''    
    Norb = pam.Norb
    dim = VS.dim
    d10L2_state_indices = []; d10L2_state_labels = []

    for i in range(0,dim):
        # state is original state but its orbital info remains after basis change
        state = VS.get_state(VS.lookup_tbl[i])
        itype = state['type']
        
        if itype == 'two_hole_one_eh':
            continue
            
        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        o1 = state['hole1_orb']
        o2 = state['hole2_orb']
        x1, y1, z1 = state['hole1_coord']
        x2, y2, z2 = state['hole2_coord']

        if not (abs(x1-1.0)<1.e-3 and abs(x2-1.0)<1.e-3 and abs(y1)<1.e-3 and abs(y2)<1.e-3):
            continue
        
        if o1 in pam.O_orbs and o2 in pam.O_orbs:
            o12 = sorted([o1,o2])
            o12 = tuple(o12)

            # S_val, Sz_val obtained from basis.create_singlet_triplet_basis_change_matrix
            S12  = S_val[i]
            Sz12 = Sz_val[i]

            # continue only singlet
            if S12!=0:
                continue

            # for triplet, only need one Sz state; other Sz states have the same A(w)
            if Sz12==0:
                d10L2_state_indices.append(i); d10L2_state_labels.append('$d^{10}L^2$')
                print ("d10L2_state_indices", i, ", state: S= ", S12, " Sz= ", Sz12, "orb= ", o1,o2)
                 
    return d10L2_state_indices, d10L2_state_labels

############################################################################
def get_d8Ls_state_indices(VS,d_double,S_val, Sz_val):
    '''
    Get d8Ls state index
    one hole must be dz2 corresponding to s electron
    
    Choose interesing states to plot spectra
    see George's email on Aug.21, 2021
    '''    
    Norb = pam.Norb
    dim = VS.dim
    a1b1Ls_S0_state_indices = []; a1b1Ls_S0_state_labels = []
    a1b1Ls_S1_state_indices = []; a1b1Ls_S1_state_labels = []
    a1a1Ls_state_indices = []; a1a1Ls_state_labels = []
    
    for i in d_double:
        # state is original state but its orbital info remains after basis change
        state = VS.get_state(VS.lookup_tbl[i])
        itype = state['type']
        
        if itype == 'two_hole_no_eh':
            continue
            
        se = state['e_spin']
        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        s3 = state['hole3_spin']
        oe = state['e_orb']
        o1 = state['hole1_orb']
        o2 = state['hole2_orb']
        o3 = state['hole3_orb']
        xe, ye, ze = state['e_coord']
        x1, y1, z1 = state['hole1_coord']
        x2, y2, z2 = state['hole2_coord']
        x3, y3, z3 = state['hole3_coord']
            
        nNi, nO, dorbs, porbs = util.get_statistic_3orb(o1,o2,o3)
            
        if not (nNi==2 and nO==1):
            continue
            
        orbs = [o1,o2,o3]
        xs = [x1,x2,x3]
        ys = [y1,y2,y3]
        zs = [z1,z2,z3]
        idx = orbs.index(porbs[0])
        
        if not (abs(xe)<1.e-3 and abs(ye)<1.e-3 and abs(ze-1)<1.e-3):
            continue
            
        if not (abs(xs[idx]-1.0)<1.e-3 and abs(ys[idx])<1.e-3):
            continue
            
        # S_val, Sz_val obtained from basis.create_singlet_triplet_basis_change_matrix
        S12  = S_val[i]
        Sz12 = Sz_val[i]

        # for triplet, only need one Sz state; other Sz states have the same A(w)
       # if Sz12==0 and 'd3z2r2' in dorbs and se=='up':
        dorbs = sorted(dorbs)
        
        # d8_{a1b1} singlet:
        if S12==0 and Sz12==0 and se=='up' and dorbs[0]=='d3z2r2' and dorbs[1]=='dx2y2':
            a1b1Ls_S0_state_indices.append(i); a1b1Ls_S0_state_labels.append('$S=0,S_z=0,d^8_{z^2,x^2-y^2}Ls$')
            print ("a1b1Ls_state_indices", i, ", state: orb= ", se,oe,xe, ye, ze,\
                s1,o1,x1, y1, z1,s2,o2,x2, y2, z2,s3,o3,x3, y3, z3)
                
        # d8_{a1b1} triplet:
        if S12==1 and Sz12==0 and se=='up' and dorbs[0]=='d3z2r2' and dorbs[1]=='dx2y2':
            a1b1Ls_S1_state_indices.append(i); a1b1Ls_S1_state_labels.append('$S=1,S_z=0,d^8_{z^2,x^2-y^2}Ls$')
            print ("a1b1Ls_state_indices", i, ", state: orb= ", se,oe,xe, ye, ze,\
                s1,o1,x1, y1, z1,s2,o2,x2, y2, z2,s3,o3,x3, y3, z3)
        
        # d8_{a1a1} singlet:
        if S12==0 and Sz12==0 and se=='up' and dorbs[0]=='d3z2r2' and dorbs[1]=='d3z2r2':
            a1a1Ls_state_indices.append(i); a1a1Ls_state_labels.append('$S=0,S_z=0,d^8_{z^2,z^2}Ls$')
            print ("a1a1Ls_state_indices", i, ", state: orb= ", se,oe,xe, ye, ze,\
                s1,o1,x1, y1, z1,s2,o2,x2, y2, z2,s3,o3,x3, y3, z3)

    return a1b1Ls_S0_state_indices, a1b1Ls_S0_state_labels, \
           a1b1Ls_S1_state_indices, a1b1Ls_S1_state_labels, \
           a1a1Ls_state_indices, a1a1Ls_state_labels

def get_d9L2s_state_indices(VS):
    '''
    Get d9L2s state index
    one hole must be dz2 corresponding to s electron
    '''    
    Norb = pam.Norb
    dim = VS.dim
    d9L2s_state_indices = []; d9L2s_state_labels = []
    
    for i in range(0,dim):
        # state is original state but its orbital info remains after basis change
        state = VS.get_state(VS.lookup_tbl[i])
        itype = state['type']
        
        if itype == 'two_hole_no_eh':
            continue
            
        se = state['e_spin']
        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        s3 = state['hole3_spin']
        oe = state['e_orb']
        o1 = state['hole1_orb']
        o2 = state['hole2_orb']
        o3 = state['hole3_orb']
        xe, ye, ze = state['e_coord']
        x1, y1, z1 = state['hole1_coord']
        x2, y2, z2 = state['hole2_coord']
        x3, y3, z3 = state['hole3_coord']

        nNi, nO, dorbs, porbs = util.get_statistic_3orb(o1,o2,o3)
            
        if not (nNi==1 and nO==2):
            continue
            
        orbs = [o1,o2,o3]
        xs = [x1,x2,x3]
        ys = [y1,y2,y3]
        zs = [z1,z2,z3]
        
        if not (abs(xe)<1.e-3 and abs(ye)<1.e-3 and abs(ze-1)<1.e-3):
            continue
            
        idx = orbs.index(porbs[0])
        if not (abs(xs[idx]-1.0)<1.e-3 and abs(ys[idx])<1.e-3):
            continue
        idx = orbs.index(porbs[1])
        if not (abs(xs[idx]-1.0)<1.e-3 and abs(ys[idx])<1.e-3):
            continue
            
        if dorbs[0]=='dx2y2' and se=='up':
            d9L2s_state_indices.append(i); d9L2s_state_labels.append('$d^9_{x^2-y^2}L^2s$')
            print ("d9L2s_state_indices", i, ", state: orb= ", o1,o2,o3,oe)

    return d9L2s_state_indices, d9L2s_state_labels

