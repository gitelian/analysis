# This analysis will focus on the sensory evoked responses for the 8pos ephys
# paper. Here I will try to answer the follwing question:
#
# - Is there a correlation between the sensory drive of a unit and the
# amount of change during light conditions? e.g. are only strongly driven M1
# units effected by S1 silencing?
# Hypothesis: The more driven a unit is the more S1 silencing attenutates that
# activity.
#
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import h5py
#import 3rd party code found on github
import dunn


##### Collect mean evoked spike rates for non-driven units and driven units

driven = list()
manip = list()

nodrive = list()
nomanip = list()

control_pos = neuro.control_pos - 1
driven = list()
# compare only no light positions with control/no contact position
to_compare = [ (k, control_pos) for k in range(control_pos)]

for unit in range(neuro.num_units):
    groups = list()
    if neuro.shank_ids[unit] == 0:
        stim_ind = 9
#        else:
#            stim_ind = 18

        # dedent this if doing s1 as well
        for k in range(control_pos + 1):
            # append all rates
            groups.append(neuro.abs_count[k][:, unit])

        # test for sensory drive
        H, p_omnibus, Z_pairs, p_corrected, reject = dunn.kw_dunn(groups, to_compare=to_compare, alpha=0.05, method='simes-hochberg') # or 'bonf' for bonferoni

        inds = np.where(reject == True)[0]
        if inds.sum() != 0:
            for i in inds:
                driven.append(neuro.evk_rate[i][:, unit].mean())
                manip.append(neuro.evk_rate[i + stim_ind][:, unit].mean())

        inds = np.where(reject == False)[0]
        if inds.sum() != 0:
            for i in inds:
                nodrive.append(neuro.evk_rate[i][:, unit].mean())
                nomanip.append(neuro.evk_rate[i + stim_ind][:, unit].mean())





