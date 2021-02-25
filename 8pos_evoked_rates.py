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

## consider taking the absolute value from baseline?
fids = ['1336', '1338', '1339', '1340', '1343', '1345']
experiments = list()
for fid_name in fids:
    get_ipython().magic(u"run hdfanalyzer.py {}".format(fid_name))
    experiments.append(neuro)

# each entry is the mean evoked rate for one unit at one position for light off
# and S1 light on [evoked rate, evoked rate + S1 silencing]
m1 = np.zeros((1, 2))
s1 = np.zeros((1, 2))
to_compare = [ (k, neuro.control_pos-1) for k in range(neuro.control_pos-1)]

for neuro in experiments:
    # dedent for single experiment analysis
    for unit in range(neuro.num_units):

        groups = [neuro.abs_count[k][:, unit] for k in range(neuro.control_pos)]
        H, p_omnibus, Z_pairs, p_corrected, reject = dunn.kw_dunn(groups, to_compare=to_compare, alpha=0.05, method='simes-hochberg') # or 'bonf' for bonferoni
        inds = np.where(reject == True)[0]
        if inds.sum() != 0:
    #    for k in range(neuro.control_pos):
            for k in inds:
                # dedent and comment all above for loop to look at all conditions
                # not just conditions with a significant sensory evoked activity
                if neuro.shank_ids[unit] == 0:
                    stim_ind = 9
                    nolight = neuro.evk_rate[k][:, unit].mean()
                    s1light = neuro.evk_rate[k + stim_ind][:, unit].mean()
                    mean_rates = np.asarray([nolight, s1light]).reshape(1, 2)
                    m1 = np.concatenate((m1, mean_rates), axis=0)
                else:
                    stim_ind = 18
                    nolight = neuro.evk_rate[k][:, unit].mean()
                    m1light = neuro.evk_rate[k + stim_ind][:, unit].mean()
                    mean_rates = np.asarray([nolight, m1light]).reshape(1, 2)
                    s1 = np.concatenate((s1, mean_rates), axis=0)

m1 = m1[1::, :]
s1 = s1[1::, :]

plt.figure()
plt.scatter(m1[:, 0], m1[:, 1], s=25, c='tab:blue')
plt.scatter(s1[:, 0], s1[:, 1], s=25, facecolors='none', edgecolor='tab:red')
plt.plot([-15, 35], [-15, 35])


##### scratch space #####
##### scratch space ##### 




