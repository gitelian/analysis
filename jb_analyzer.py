#!/bin/bash
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import scipy.io as sio

# change default figure type to PDF
mpl.rcParams['savefig.format'] = 'pdf'
plt.rc('font',family='Arial')
sns.set_style("whitegrid", u'axes.grid' : False})


##
neuro.performance_vs_time(split=True)
## set engaged/disengaged trials
# fid2139
neuro.set_engaged_disengaged_trials(e_ind=[[1, 290]], d_ind=[[440, 670],[800, 1002]])
# fid2155
#neuro.set_engaged_disengaged_trials(e_ind=[[1, 290]], d_ind=[[360, 550]])
#neuro.set_engaged_disengaged_trials(e_ind=[[0, 20]], d_ind=[[21, 600]])
#neuro.set_engaged_disengaged_trials(e_ind=[[350,650]], d_ind=[[0, 270]])

neuro.rates(kind='run_boolean', engaged=None)

##### make summary figure for entire experiment ###
##### make summary figure for entire experiment ###
fig, ax = plt.subplots(5, 2, figsize=(10, 14))
fig.subplots_adjust(top=0.945,
        bottom=0.05,
        left=0.125,
        right=0.9,
        hspace=0.655,
        wspace=0.3)

# PC % vs t
neuro.performance_vs_time(split=True, axis=ax[0][0])

# psychometric curve
neuro.get_psychometric_curve(axis=ax[0][1])

## Easy GO/NOGO mean set-point
neuro.plot_mean_whisker(t_window=[-1.5, 2.0], cond2plot=[0], axis=ax[1][0], correct=None)
neuro.plot_mean_whisker(t_window=[-1.5, 2.0], cond2plot=[0+4], axis=ax[1][1], correct=None)

ax[1][0].set_title('GO (EASY)')
ax[1][1].set_title('NOGO (EASY)')
ax[1][0].set_ylim((70, 140))
ax[1][1].set_ylim((70, 140))
ax[1][0].hlines(139, 0, 1, colors='tab:grey', linewidth=2)
ax[1][0].hlines(137, -1, 1, colors='tab:red', linewidth=2)
ax[1][1].hlines(139, 0, 1, colors='tab:grey', linewidth=2)
ax[1][1].hlines(137, -1, 1, colors='tab:red', linewidth=2)

## Hard GO/NOGO mean set-point
neuro.plot_mean_whisker(t_window=[-1.5, 2.0], cond2plot=[3], axis=ax[2][0], correct=None)
neuro.plot_mean_whisker(t_window=[-1.5, 2.0], cond2plot=[3+4], axis=ax[2][1], correct=None)

ax[2][0].set_title('GO (HARD)')
ax[2][1].set_title('NOGO (HARD)')
ax[2][0].set_ylim((70, 140))
ax[2][1].set_ylim((70, 140))
ax[2][0].hlines(139, 0, 1, colors='tab:grey', linewidth=2)
ax[2][0].hlines(137, -1, 1, colors='tab:red', linewidth=2)
ax[2][1].hlines(139, 0, 1, colors='tab:grey', linewidth=2)
ax[2][1].hlines(137, -1, 1, colors='tab:red', linewidth=2)

### set-point ONSET/OFFSET times
#deltas, merrs = neuro.get_trial_delta(plot=False)
#
## ONSET
#a = ax[3][0]
#neuro.plot_mean_err(merrs[0][0], merrs[0][1], axis=ax[3][0])
#a.set_title('setpoint retraction onset')
#a.set_ylabel('retract start (s)')
#a.set_xlabel('<--GO -- NOGO-->\npositions')
#a.hlines(0, 0, 11, 'tab:grey', linestyles='dashed')
#a.set_xlim(0, 9.5)
#a.set_ylim(-1, 1)
#
## OFFSET
#a = ax[3][1]
#neuro.plot_mean_err(merrs[1][0], merrs[1][1], axis=ax[3][1])
#a.set_title('setpoint retraction stop')
#a.set_ylabel('retract stop (s)')
#a.set_xlabel('<--GO -- NOGO-->\npositions')
#a.hlines(0, 0, 11, 'tab:grey', linestyles='dashed')
#a.set_xlim(0, 9.5)
#a.set_ylim(-1, 1)

## runspeed plots HARD GO/NOGO
neuro.plot_mean_runspeed(t_window=[-1.5, 2.0], cond2plot=[3], axis=ax[3][0], correct=None)
neuro.plot_mean_runspeed(t_window=[-1.5, 2.0], cond2plot=[3+4], axis=ax[3][1], correct=None)
ax[3][0].set_title('GO (HARD)')
ax[3][1].set_title('NOGO (HARD)')
ax[3][0].set_ylim((0, 56))
ax[3][1].set_ylim((0, 56))
ax[3][0].hlines(54, 0, 1, colors='tab:grey', linewidth=2)
ax[3][0].hlines(52, -1, 1, colors='tab:red', linewidth=2)
ax[3][1].hlines(54, 0, 1, colors='tab:grey', linewidth=2)
ax[3][1].hlines(52, -1, 1, colors='tab:red', linewidth=2)

## time to first lick
neuro.plot_time2lick(t_start=-1.0, axis=ax[4][0])
ax[4][0].hlines(0, 0, 10, 'tab:grey', linestyles='dashed')
ax[4][0].set_xlim(0.5, 8.5)

## lick rate
neuro.plot_lick_rate(t_start=-0.5, t_stop=1.5, axis=ax[4][1])




##############################################################################
## plot ver nice single trial example (whisker movement and runspeed) with
## light on & object idicator bars

cond = 3
trial = 0
fig, ax = plt.subplots(1, 1)
ax.plot(neuro.wtt, neuro.wt[cond][:, 0, trial], 'k')
ax.plot(neuro.wtt, neuro.wt[cond + 9][:, 0, trial], 'tab:blue')
#ax.vlines([-1, 1], 70, 180, colors='cyan', alpha=0.7, linewidth=3)
ax.hlines(162, -1, 1, colors='tab:blue', alpha=0.7, linewidth=3, label='vM1 silencing')
ax.hlines(161, 0, 1, colors='dimgray', alpha=0.7, linewidth=3, label='object stationary')
ax.set_ylim(85, 165)
ax.legend(loc='lower left')
ax.set_ylabel('whisker angle (deg)')
ax.set_xlabel('time (s)')
ax.set_title('Example GO trial ({})'.format(neuro.fid))



##############################################################################
                    ##### Review and revise this #####

##### compute setpoint and runspeed correlation #####
##### compute setpoint and runspeed correlation #####

run_inds2keep = list()

for t in neuro.wtt:
    run_inds2keep.append(np.argmin(np.abs(neuro.run_t - t)))



# analyze time before light on (up to 1 sec) and each subsequent second after
# that.
#t_bins = np.asarray([-2, -1, 0, 1, 2])
step = 0.500
step = 1
t_bins = np.arange(-1.5, 1.5 + step, step)
corrcoef = np.zeros((len(neuro.stim_ids), len(t_bins) - 1))

t = neuro.wtt
print('\nwtt ranges from {} to {}'.format(t[0], t[-1]))

for cond in range(len(neuro.stim_ids)):

    setpoints = list()
    runspeeds = list()
    for trial in range(len(neuro.lick_bool[cond])):
        setpoints.append(neuro.wt[cond][:, 1, trial])
        runspeeds.append(neuro.run[cond][run_inds2keep, trial])

    setpoints = np.asarray(setpoints)
    runspeeds = np.asarray(runspeeds)

    for t_ind in range(len(t_bins) - 1):

        t_start = t_bins[t_ind]
        t_stop  = t_bins[t_ind + 1]

        t_start_ind = np.argmin(np.abs(t - t_start))
        t_stop_ind  = np.argmin(np.abs(t - t_stop))

#        setpoints_in_bin = np.ravel(np.mean(setpoints[:, t_start_ind:t_stop_ind], axis=1))
#        runspeeds_in_bin = np.ravel(np.mean(runspeeds[:, t_start_ind:t_stop_ind], axis=1))
        setpoints_in_bin = np.ravel(setpoints[:, t_start_ind:t_stop_ind])
        runspeeds_in_bin = np.ravel(runspeeds[:, t_start_ind:t_stop_ind])

        corrcoef[cond, t_ind] = sp.stats.pearsonr(runspeeds_in_bin, setpoints_in_bin)[0]

fig, ax = plt.subplots()
ax.plot(t_bins[0:-1]+step/2, corrcoef[3, :], 'k', t_bins[0:-1]+step/2, corrcoef[3+9, :], 'r')
ax.vlines([-1, 1], -1, 1, colors='c')
ax.axvspan(-1, 0, alpha=0.2, color='green')
ax.set_xlim(t[0], t[-1])
ax.set_ylim(-1, 1)




###############################################################################
###############################################################################
############### original mice that performed well on task #####################
############## notes about good experiment FIDs and original plots ############
###############################################################################
###############################################################################


# 8-bar-position experiments
# good neural data and good whisker tracking
# fids = ['1336', '1338', '1339', '1340', '1343', '1345']


### JB_behavior Notes

####### GT015_LT #####
## # GT015_LT vM1 silencing with GOOD behavior but NO WHISKER TRACKING
##
## # THERE was tracking recorded...but I couldn't get it to work...maybe there is
## # something I can do now to recover it??
## # fids = ['1855', '1874', '1882', '1891', '1892']
## #experiments that may be good: 1861
##
## ##### Gt015_LT vS1 silencing + tracking #####
## # fids = ['1895', '1898', '1904']
##
## ##### GT017_NT vM1 silencing + tracking #####
## # fids = ['1911', '1912', '1913', '1923', '1924', '1929']
## # experiments that may be good: [1896, 1916, [1919 and 1920 are of same day...dropped frames]


### original plots ####

#### whisker angle and setpoint plots ####
## vM1 silencing + whisking
## FID1911 vM1 silencing, FID1895 vS1 silencing
# control + NoLight
# vM1 light
## FID1911 vM1 silencing
# trial index in control position (no contact) used to make figures
# 3, 9 huge decrease with STEADY quick running, (5,7,8, set-point decreases while runspeed increases!!!)

# vS1 light
## FID1895 and 1898 with vS1 silencing
# trial index in control position (no contact) used to make figures
# (5, 2), (9, 9) (setpoint light on, vs light off) very good examples
# fid1895(0, 0), fid1898(0, 0), fid1898(3 , 2) (vS1 silencing plots)

#### vS1 silencing psychometric curve ####
#### FID1895 ####


##############################################################################
#### working code ####
##### make psychometric curves with errors for each mouse

### GT015_LT vM1 silencing
### GT015_LT vM1 silencing
### GT015_LT vM1 silencing
fids = ['1855', '1874', '1882', '1891', '1892'] # none of these have whisker tracking
gt015_lt_vm1 = list()
gt015_lt_vm1_psy  = np.zeros((len(fids), 18))
fig, ax = plt.subplots()
for k, fid in enumerate(fids):
    #get_ipython().magic(u"run neoanalyzer.py {'1290'}")
    get_ipython().magic(u"run hdfanalyzer.py {}".format(fid))
    #neuro.rates(kind='wsk_boolean')
    gt015_lt_vm1.append(neuro)
    gt015_lt_vm1_psy[k, :] = neuro.get_psychometric_curve(axis=ax)

# compute mean and sem of psychometric curve and plot
plt.figure()
gt015_lt_vm1_mean = np.reshape(np.mean(gt015_lt_vm1_psy, axis=0), [2, 9])
gt015_lt_vm1_sem = np.reshape(sp.stats.sem(gt015_lt_vm1_psy, axis=0), [2, 9])
plt.errorbar(np.arange(1, 9), gt015_lt_vm1_mean[0, 0:-1], yerr=gt015_lt_vm1_sem[0, 0:-1], c='k', marker='o', markersize=6.0, linewidth=2)
plt.errorbar(np.arange(1, 9), gt015_lt_vm1_mean[1, 0:-1], yerr=gt015_lt_vm1_sem[1, 0:-1], c='tab:blue', marker='o', markersize=6.0, linewidth=2)

plt.errorbar(9, gt015_lt_vm1_mean[0, -1], yerr=gt015_lt_vm1_sem[0, -1], c='k', marker='o', markersize=6.0, linewidth=2)
plt.errorbar(9, gt015_lt_vm1_mean[1, -1], yerr=gt015_lt_vm1_sem[1, -1], c='tab:blue', marker='o', markersize=6.0, linewidth=2)
plt.title('GT015_LT Psychometric curve (n=5)')


### GT015_LT vS1 silencing
### GT015_LT vS1 silencing
### GT015_LT vS1 silencing
fids = ['1895', '1898', '1904'] # whisker tracking is good
gt015_lt_vs1 = list()
gt015_lt_vs1_psy  = np.zeros((len(fids), 18))
fig, ax = plt.subplots()
for k, fid in enumerate(fids):
    #get_ipython().magic(u"run neoanalyzer.py {'1290'}")
    get_ipython().magic(u"run hdfanalyzer.py {}".format(fid))
    #neuro.rates(kind='wsk_boolean')
    gt015_lt_vs1.append(neuro)
    gt015_lt_vs1_psy[k, :] = neuro.get_psychometric_curve(axis=ax)

# compute mean and sem of psychometric curve and plot
plt.figure()
gt015_lt_vs1_mean = np.reshape(np.mean(gt015_lt_vs1_psy, axis=0), [2, 9])
gt015_lt_vs1_sem = np.reshape(sp.stats.sem(gt015_lt_vs1_psy, axis=0), [2, 9])
plt.errorbar(np.arange(1, 9), gt015_lt_vs1_mean[0, 0:-1], yerr=gt015_lt_vs1_sem[0, 0:-1], c='k', marker='o', markersize=6.0, linewidth=2)
plt.errorbar(np.arange(1, 9), gt015_lt_vs1_mean[1, 0:-1], yerr=gt015_lt_vs1_sem[1, 0:-1], c='tab:blue', marker='o', markersize=6.0, linewidth=2)

plt.errorbar(9, gt015_lt_vs1_mean[0, -1], yerr=gt015_lt_vs1_sem[0, -1], c='k', marker='o', markersize=6.0, linewidth=2)
plt.errorbar(9, gt015_lt_vs1_mean[1, -1], yerr=gt015_lt_vs1_sem[1, -1], c='tab:blue', marker='o', markersize=6.0, linewidth=2)


### GT017_NT vM1 silencing
### GT017_NT vM1 silencing
### GT017_NT vM1 silencing
fids = ['1911', '1912', '1913', '1923', '1924', '1929'] # whisker tracking is good 1929 is the only with ephys
gt017_nt_vm1 = list()
gt017_nt_vm1_psy = np.zeros((len(fids), 18))
fig, ax = plt.subplots()
for k, fid in enumerate(fids):
    #get_ipython().magic(u"run neoanalyzer.py {'1290'}")
    get_ipython().magic(u"run hdfanalyzer.py {}".format(fid))
    #neuro.rates(kind='wsk_boolean')
    gt017_nt_vm1.append(neuro)
    gt017_nt_vm1_psy[k, :] = neuro.get_psychometric_curve(axis=ax)[0]

# compute mean and sem of psychometric curve and plot
plt.figure()
gt017_nt_vm1_mean = np.reshape(np.mean(gt017_nt_vm1_psy, axis=0), [2, 9])
gt017_nt_vm1_sem = np.reshape(sp.stats.sem(gt017_nt_vm1_psy, axis=0), [2, 9])
plt.errorbar(np.arange(1, 9), gt017_nt_vm1_mean[0, 0:-1], yerr=gt017_nt_vm1_sem[0, 0:-1], c='k', marker='o', markersize=6.0, linewidth=2)
plt.errorbar(np.arange(1, 9), gt017_nt_vm1_mean[1, 0:-1], yerr=gt017_nt_vm1_sem[1, 0:-1], c='tab:blue', marker='o', markersize=6.0, linewidth=2)

plt.errorbar(9, gt017_nt_vm1_mean[0, -1], yerr=gt017_nt_vm1_sem[0, -1], c='k', marker='o', markersize=6.0, linewidth=2)
plt.errorbar(9, gt017_nt_vm1_mean[1, -1], yerr=gt017_nt_vm1_sem[1, -1], c='tab:blue', marker='o', markersize=6.0, linewidth=2)
plt.title('GT017_NT Psychometric curve (n=6)')

















#class BehaviorAnalysis(object):
#    def __init__(self, neuro):
#        self.neuro = neuro
#
#    def blah(self):
#        print('blah')
#
#if __name__ == "__main__":
#    print('loaded jb_analyzer')
