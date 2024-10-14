import time
import variational_space as vs
import lattice as lat
import bisect
import numpy as np
import scipy.sparse as sps
import parameters as pam
import utility as util


def find_singlet_triplet_partner_d_double(VS, d_part, index, h345_part):
    '''
    For a given state find its partner state to form a singlet/triplet.
    Right now only applied for d_double states
    
    Note: idx is to label which hole is not on Ni

    Returns
    -------
    index: index of the singlet/triplet partner state in the VS
    phase: phase factor with which the partner state needs to be multiplied.
    '''
    get_lookup_tle_index = []
    phase = 0
    if index == 145:
        slabel = h345_part[0:5] + [d_part[0]] + d_part[6:10] + [d_part[5]] + d_part[1:5] + h345_part[5:15]
    elif index == 245:
        slabel = [d_part[0]] + d_part[6:10] + h345_part[0:5] + [d_part[5]] + d_part[1:5] + h345_part[5:15]
    elif index == 345:
        slabel = [d_part[0]] + d_part[6:10] + [d_part[5]] + d_part[1:5] + h345_part[0:5] + h345_part[5:15]
    elif index == 135:
        slabel = h345_part[0:5] + [d_part[0]] + d_part[6:10] + h345_part[5:10] + [d_part[5]] + d_part[1:5] + h345_part[
                                                                                                             10:15]
    elif index == 235:
        slabel = [d_part[0]] + d_part[6:10] + h345_part[0:5] + h345_part[5:10] + [d_part[5]] + d_part[1:5] + h345_part[
                                                                                                             10:15]
    elif index == 125:
        slabel = h345_part[0:5] + h345_part[5:10] + [d_part[0]] + d_part[6:10] + [d_part[5]] + d_part[1:5] + h345_part[
                                                                                                             10:15]
    elif index == 123:
        slabel = h345_part[0:15] + [d_part[0]] + d_part[6:10] + [d_part[5]] + d_part[1:5]
    elif index == 124:
        slabel = h345_part[0:10] + [d_part[0]] + d_part[6:10] + h345_part[10:15] + [d_part[5]] + d_part[1:5]
    elif index == 134:
        slabel = h345_part[0:5] + [d_part[0]] + d_part[6:10] + h345_part[5:15] + [d_part[5]] + d_part[1:5]
    elif index == 234:
        slabel = [d_part[0]] + d_part[6:10] + h345_part[0:15] + [d_part[5]] + d_part[1:5]

    # if vs.check_Pauli(slabel):
    if True:
        exists_partner = True
        tmp_state = vs.create_state(slabel)
        partner_state, _, _ = vs.make_state_canonical(tmp_state)
        get_lookup_tle_index = VS.get_index(partner_state)
        phase = -1.0
    else:
        exists_partner = False
    return get_lookup_tle_index, phase, exists_partner


def create_singlet_triplet_basis_change_matrix_d_double(VS, d_double, double_part, idx, hole345_part):
    '''
    Similar to above create_singlet_triplet_basis_change_matrix but only applies
    basis change for d_double states
    
    Note that for three hole state, its partner state must have exactly the same
    spin and positions of L and Nd-electron
    
    This function is required for create_interaction_matrix_ALL_syms !!!
    '''
    data = []
    row = []
    col = []
    start_time = time.time()
    count_singlet = 0
    count_triplet = 0

    # store index of partner state in d_double to avoid double counting
    # otherwise, when arriving at i's partner j, its partner would be i
    count_list = []

    # denote if the new state is singlet (0) or triplet (1)
    S_d8_val = np.zeros(VS.dim, dtype=int)
    Sz_d8_val = np.zeros(VS.dim, dtype=int)
    AorB_d8_sym = np.zeros(VS.dim, dtype=int)

    # first set the matrix to be identity matrix (for states not d_double)

    for i in range(VS.dim):
        if i not in d_double:
            data.append(np.sqrt(2.0));
            row.append(i);
            col.append(i)

    for i, double_id in enumerate(d_double):
        s1 = double_part[i][0]
        o1 = double_part[i][1]
        s2 = double_part[i][5]
        o2 = double_part[i][6]
        dpos = double_part[i][2:5]

        if s1 == s2:
            # must be triplet
            # see case 2 of make_state_canonical in vs.py, namely
            # for same spin states, always order the orbitals
            S_d8_val[double_id] = 1
            data.append(np.sqrt(2.0));
            row.append(double_id);
            col.append(double_id)
            if s1 == 'up':
                Sz_d8_val[double_id] = 1
            elif s1 == 'dn':
                Sz_d8_val[double_id] = -1
            count_triplet += 1

        elif s1 == 'dn' and s2 == 'up':
            print('Error: d_double cannot have states with s1=dn, s2=up !')
            tstate = VS.get_state(VS.lookup_tbl[double_id])
            ts1 = tstate['hole1_spin']
            ts2 = tstate['hole2_spin']
            ts3 = tstate['hole3_spin']
            ts4 = tstate['hole4_spin']
            ts5 = tstate['hole5_spin']
            torb1 = tstate['hole1_orb']
            torb2 = tstate['hole2_orb']
            torb3 = tstate['hole3_orb']
            torb4 = tstate['hole4_orb']
            torb5 = tstate['hole5_orb']
            tx1, ty1, tz1 = tstate['hole1_coord']
            tx2, ty2, tz2 = tstate['hole2_coord']
            tx3, ty3, tz3 = tstate['hole3_coord']
            tx4, ty4, tz4 = tstate['hole4_coord']
            tx5, ty5, tz5 = tstate['hole5_coord']
            print('Error state', double_id, ts1, torb1, tx1, ty1, tz1, ts2, torb2, tx2, ty2, tz2, ts3, torb3, tx3, ty3,
                  tz3, \
                  ts4, torb4, tx4, ty4, tz4, ts5, torb5, tx5, ty5, tz5)
            break

        elif s1 == 'up' and s2 == 'dn':
            if o1 == o2:
                if o1 != 'dxz' and o1 != 'dyz':
                    data.append(np.sqrt(2.0));
                    row.append(double_id);
                    col.append(double_id)
                    S_d8_val[double_id] = 0
                    Sz_d8_val[double_id] = 0
                    count_singlet += 1

                # get state as (e1e1 +- e2e2)/sqrt(2) for A and B sym separately 
                # instead of e1e1 and e2e2
                elif o1 == 'dxz':  # no need to consider e2='dyz' case
                    # generate paired e2e2 state:
                    if idx[i] == 345:
                        slabel = [s1, 'dyz'] + dpos + [s2, 'dyz'] + dpos + hole345_part[i][0:5] + hole345_part[i][5:15]
                    elif idx[i] == 245:
                        slabel = [s1, 'dyz'] + dpos + hole345_part[i][0:5] + [s2, 'dyz'] + dpos + hole345_part[i][5:15]
                    elif idx[i] == 145:
                        slabel = hole345_part[i][0:5] + [s1, 'dyz'] + dpos + [s2, 'dyz'] + dpos + hole345_part[i][5:15]
                    elif idx[i] == 135:
                        slabel = hole345_part[i][0:5] + [s1, 'dyz'] + dpos + hole345_part[i][5:10] + [s2,
                                                                                                      'dyz'] + dpos + \
                                 hole345_part[i][10:15]
                    elif idx[i] == 235:
                        slabel = [s1, 'dyz'] + dpos + hole345_part[i][0:5] + hole345_part[i][5:10] + [s2,
                                                                                                      'dyz'] + dpos + \
                                 hole345_part[i][10:15]
                    elif idx[i] == 125:
                        slabel = hole345_part[i][0:5] + hole345_part[i][5:10] + [s1, 'dyz'] + dpos + [s2,
                                                                                                      'dyz'] + dpos + \
                                 hole345_part[i][10:15]
                    elif idx[i] == 123:
                        slabel = hole345_part[i][0:15] + [s1, 'dyz'] + dpos + [s2, 'dyz'] + dpos
                    elif idx[i] == 124:
                        slabel = hole345_part[i][0:10] + [s1, 'dyz'] + dpos + hole345_part[i][10:15] + [s2,
                                                                                                        'dyz'] + dpos
                    elif idx[i] == 134:
                        slabel = hole345_part[i][0:5] + [s1, 'dyz'] + dpos + hole345_part[i][5:15] + [s2, 'dyz'] + dpos
                    elif idx[i] == 234:
                        slabel = [s1, 'dyz'] + dpos + hole345_part[i][0:15] + [s2, 'dyz'] + dpos

                    tmp_state = vs.create_state(slabel)
                    new_state, _, _ = vs.make_state_canonical(tmp_state)
                    e2 = VS.get_index(new_state)

                    data.append(1.0);
                    row.append(double_id);
                    col.append(double_id)
                    data.append(1.0);
                    row.append(e2);
                    col.append(double_id)
                    AorB_d8_sym[double_id] = 1
                    S_d8_val[double_id] = 0
                    Sz_d8_val[double_id] = 0
                    count_singlet += 1
                    data.append(1.0);
                    row.append(double_id);
                    col.append(e2)
                    data.append(-1.0);
                    row.append(e2);
                    col.append(e2)
                    AorB_d8_sym[e2] = -1
                    S_d8_val[e2] = 0
                    Sz_d8_val[e2] = 0
                    count_singlet += 1


            else:
                if double_id not in count_list:
                    j, ph, exists_partner = find_singlet_triplet_partner_d_double(VS, double_part[i], idx[i], hole345_part[i])
                    if not exists_partner:
                        continue

                    # append matrix elements for singlet states
                    # convention: original state col i stores singlet and 
                    #             partner state col j stores triplet
                    data.append(1.0);
                    row.append(double_id);
                    col.append(double_id)
                    data.append(-ph);
                    row.append(j);
                    col.append(double_id)
                    S_d8_val[double_id] = 0
                    Sz_d8_val[double_id] = 0

                    #print "partner states:", i,j
                    #print "state i = ", s1, orb1, s2, orb2
                    #print "state j = ",'up',orb2,'dn',orb1

                    # append matrix elements for triplet states
                    data.append(1.0);
                    row.append(double_id);
                    col.append(j)
                    data.append(ph);
                    row.append(j);
                    col.append(j)
                    S_d8_val[j] = 1
                    Sz_d8_val[j] = 0

                    count_list.append(j)

                    count_singlet += 1
                    count_triplet += 1

    print("basis %s seconds ---" % (time.time() - start_time))
    return sps.coo_matrix((data, (row, col)), shape=(VS.dim, VS.dim)) / np.sqrt(2.0), S_d8_val, Sz_d8_val, AorB_d8_sym


def print_VS_after_basis_change(VS, S_val, Sz_val):
    print('print_VS_after_basis_change:')
    for i in range(0, VS.dim):
        state = VS.get_state(VS.lookup_tbl[i])
        ts1 = state['hole1_spin']
        ts2 = state['hole2_spin']
        torb1 = state['hole1_orb']
        torb2 = state['hole2_orb']
        tx1, ty1, tz1 = state['hole1_coord']
        tx2, ty2, tz2 = state['hole2_coord']
        #if ts1=='up' and ts2=='up':
        if torb1 == 'dx2y2' and torb2 == 'px':
            print(i, ts1, torb1, tx1, ty1, tz1, ts2, torb2, tx2, ty2, tz2, 'S=', S_val[i], 'Sz=', Sz_val[i])
