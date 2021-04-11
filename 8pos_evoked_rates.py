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

# NOTE The two methods here provide the same results. The top one includes all
# drive units, combining RS and FS units.
# The bottom method has a more strict criterion for what is considered sensory
# driven (testing each position with its own baseline instead of the baseline
# at the control position) and separates RS and FS
# NOTE vS1 RS units are just barely modulated by M1 but it is not significant

##### Collect mean evoked spike rates for non-driven units and driven units

## consider taking the absolute value from baseline?
fids = ['1336', '1338', '1339', '1340', '1343', '1345']
experiments = list()
for fid_name in fids:
    get_ipython().magic(u"run hdfanalyzer.py {}".format(fid_name))
    experiments.append(neuro)

## NOTE METHOD 1
# each entry is the mean evoked rate for one unit at one position for light off
# and S1 light on [evoked rate, evoked rate + S1 silencing]

m1 = np.zeros((1, 2))
s1 = np.zeros((1, 2))
#to_compare = [ (k, neuro.control_pos-1) for k in range(neuro.control_pos-1)]
to_compare = [ (k, n.control_pos-1) for k in range(n.control_pos-1)]

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

# M1
plt.figure()
plt.scatter(m1[:, 0], m1[:, 1], s=15, c='tab:blue')
plt.plot([-12, 32], [-12, 32], 'k')
xlim([-12, 30]); ylim([-12, 30])
xlabel('Evoked firing rate (Hz)')
ylabel('Evoked firing rate (Hz)\n+ vS1 silencing')
title('vM1 changes in evoked firing rate')
plt.hlines(0, -12, 32, color='tab:grey', linewidth=1.0)
plt.vlines(0, -12, 32, color='tab:grey', linewidth=1.0)


# S1
#plt.scatter(s1[:, 0], s1[:, 1], s=25, facecolors='none', edgecolor='tab:red')
plt.figure()
plt.scatter(s1[:, 0], s1[:, 1], s=15, c='tab:red')
plt.plot([-15, 80], [-15, 80], 'k')
xlim([-15, 80]); ylim([-15, 80])
xlabel('Evoked firing rate (Hz)')
ylabel('Evoked firing rate (Hz)\n+ vM1 silencing')
title('vS1 changes in evoked firing rate')
plt.hlines(0, -15, 80, color='tab:grey', linewidth=1.0)
plt.vlines(0, -15, 80, color='tab:grey', linewidth=1.0)


##### scratch space #####
##### scratch space #####


## NOTE METHOD 2

for k, ctype in enumerate(['RS', 'FS']):

    low_val = list()
    m1_driven_unit_inds = np.where(npand(region==0, driven==True))[0]
    #m1_driven_unit_inds = np.where(npand(npand(region==0, driven==True), cell_type == ctype))[0]
    m1_total_driven_pos = int(sum(num_driven[m1_driven_unit_inds]))
    m1_evk_rate = np.zeros((m1_total_driven_pos, ))
    m1_evk_rate_Light = np.zeros((m1_total_driven_pos, ))
    count = 0
    for m1_unit_ind in m1_driven_unit_inds:
        for pos_ind in driven_inds[m1_unit_ind]:
            m1_evk_rate[count] = evk_rate[m1_unit_ind, pos_ind, 0]
            m1_evk_rate_Light[count] = evk_rate[m1_unit_ind, pos_ind+9, 0]
            if np.abs(m1_evk_rate[count]) < 1:
                low_val.append((m1_unit_ind, pos_ind))
            count+=1

    scatter(m1_evk_rate, m1_evk_rate_Light, s=15, c='tab:blue')
    plt.plot([-12, 32], [-12, 32], 'k')
    xlim([-12, 30]); ylim([-12, 30])
    xlabel('Evoked firing rate (Hz)')
    ylabel('Evoked firing rate (Hz)\n+ vS1 silencing')
    title('vM1 changes in evoked firing rate')
    plt.hlines(0, -12, 32, color='tab:grey', linewidth=1.0)
    plt.vlines(0, -12, 32, color='tab:grey', linewidth=1.0)



    low_val = list()
    s1_driven_unit_inds = np.where(npand(npand(region==1, driven==True), cell_type == ctype))[0]
    s1_total_driven_pos = int(sum(num_driven[s1_driven_unit_inds]))
    s1_evk_rate = np.zeros((s1_total_driven_pos, ))
    s1_evk_rate_Light = np.zeros((s1_total_driven_pos, ))
    count = 0
    for s1_unit_ind in s1_driven_unit_inds:
        for pos_ind in driven_inds[s1_unit_ind]:
            s1_evk_rate[count] = evk_rate[s1_unit_ind, pos_ind, 0]
            s1_evk_rate_Light[count] = evk_rate[s1_unit_ind, pos_ind+9+9, 0]
            if np.abs(s1_evk_rate[count]) < 1:
                low_val.append((s1_unit_ind, pos_ind))
            count+=1

    scatter(s1_evk_rate, s1_evk_rate_Light, s=15, c='tab:red')
    plt.plot([-12, 60], [-12, 60], 'k')
    xlim([-12, 60]); ylim([-12, 60])
    xlabel('Evoked firing rate (Hz)')
    ylabel('Evoked firing rate (Hz)\n+ vM1 silencing')
    title('vM1 changes in evoked firing rate')
    plt.hlines(0, -12, 60, color='tab:grey', linewidth=1.0)
    plt.vlines(0, -12, 60, color='tab:grey', linewidth=1.0)











