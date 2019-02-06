#!/bin/bash
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
import h5py

#import 3rd party code found on github
#import icsd
import ranksurprise
import dunn
from IPython.core.debugger import Tracer





# plot mean set-point, indicate light on
# TODO: overlay object x-position

cond2plot = np.arange(9)
#cond2plot=[3, 4]

## plot mean set-point
fig, ax = neuro.plot_mean_whisker(t_window=[-1.5, 1.5], kind='setpoint', cond2plot=cond2plot, all_trials=True)
fig.suptitle(fid + ' mean set-point ALL TRIALS')
ylim = ax[0].get_ylim()
for a in ax:
    a.vlines([-1, 1], ylim[0], ylim[1], colors='c')
    a.set_ylim(ylim)
    a.axvspan(-1, 0, alpha=0.2, color='green')

## plot change in set-point
fig, ax = neuro.plot_mean_whisker(t_window=[-1.5, 1.5], kind='setpoint', cond2plot=cond2plot, all_trials=True, delta=True)
fig.suptitle(fid + ' mean change in set-point ALL TRIALS')
ylim = ax[0].get_ylim()
for a in ax:
    a.vlines([-1, 1], ylim[0], ylim[1], colors='c')
    a.set_ylim(ylim)
    a.axvspan(-1, 0, alpha=0.2, color='green')

## plot mean amplitude
fig, ax = neuro.plot_mean_whisker(t_window=[-1.5, 1.5], kind='amplitude', cond2plot=cond2plot, all_trials=True)
fig.suptitle(fid + ' mean amplitude ALL TRIALS')
ylim = ax[0].get_ylim()
for a in ax:
    a.vlines([-1, 1], ylim[0], ylim[1], colors='c')
    a.set_ylim(ylim)
    a.axvspan(-1, 0, alpha=0.2, color='green')

## plot PSD of whisker angle
neuro.plot_wt_freq(t_window=[-1, 0], cond2plot=cond2plot, all_trials=True)

## plot mean runspeed
fig, ax = neuro.plot_mean_runspeed(t_window=[-1.5, 1.5], cond2plot=cond2plot, all_trials=True)
fig.suptitle(fid + ' mean runspeed ALL TRIALS')
ylim = ax[0].get_ylim()
for a in ax:
    a.vlines([-1, 1], ylim[0], ylim[1], colors='c')
    a.set_ylim(ylim)
    a.axvspan(-1, 0, alpha=0.2, color='green')

## plot change in runspeed
fig, ax = neuro.plot_mean_runspeed(t_window=[-1.5, 1.5], cond2plot=cond2plot, all_trials=True, delta=True)
fig.suptitle(fid + ' mean change in runspeed ALL TRIALS\nLIGHT - noLIGHT')
ylim = ax[0].get_ylim()
for a in ax:
    a.vlines([-1, 1], ylim[0], ylim[1], colors='c')
    a.set_ylim(ylim)
    a.axvspan(-1, 0, alpha=0.2, color='green')

## plot mean set-point with NOLIGHT for CORRECT vs INCORRECT
sp_mean_right, sp_sem_right, num_trials_right = neuro.get_setpoint(correct=True)
sp_mean_wrong, sp_sem_wrong, num_trials_wrong = neuro.get_setpoint(correct=False)
num_manipulations = neuro.stim_ids.shape[0]/neuro.control_pos

print(np.min(np.concatenate((num_trials_right, num_trials_wrong), axis=1), axis=1))
cond2plot = np.where(np.min(np.concatenate((num_trials_right, num_trials_wrong), axis=1), axis=1) > 4)[0]

fig, ax = plt.subplots(len(cond2plot), 1, sharey=True)
for k, cond in enumerate(cond2plot):
    if cond < neuro.control_pos:
        # no light
        ax[k].plot(neuro.wtt, sp_mean_right[cond], 'k', label='correct')
        ax[k].fill_between(neuro.wtt, sp_mean_right[cond] + sp_sem_right[cond], sp_mean_right[cond] - sp_sem_right[cond], color='k', alpha=0.3)
        ax[k].plot(neuro.wtt, sp_mean_wrong[cond], '--k', label='incorrect')
        ax[k].fill_between(neuro.wtt, sp_mean_wrong[cond] + sp_sem_wrong[cond], sp_mean_wrong[cond] - sp_sem_wrong[cond], color='k', alpha=0.3)
        ax[k].set_title('No light\nCondition {}'.format(cond + 1))
        ax[k].set_xlabel('time (s)')
        ax[k].set_ylabel('set-point (deg)')
    elif cond >= neuro.control_pos and cond < neuro.control_pos*2:
        # vM1 light
        ax[k].plot(neuro.wtt, sp_mean_right[cond], 'b', label='correct')
        ax[k].fill_between(neuro.wtt, sp_mean_right[cond] + sp_sem_right[cond], sp_mean_right[cond] - sp_sem_right[cond], color='b', alpha=0.3)
        ax[k].plot(neuro.wtt, sp_mean_wrong[cond], '--b', label='incorrect')
        ax[k].fill_between(neuro.wtt, sp_mean_wrong[cond] + sp_sem_wrong[cond], sp_mean_wrong[cond] - sp_sem_wrong[cond], color='b', alpha=0.3)
        ax[k].set_title('Light\nCondition {}'.format(cond + 1))
        ax[k].set_xlabel('time (s)')
        ax[k].set_ylabel('set-point (deg)')

#        ax[k][manip].fill_between(self.wtt[start_ind:stop_ind], mean_sp[cond + (self.control_pos*manip)][start_ind:stop_ind] - sem_sp[cond + (self.control_pos*manip)][start_ind:stop_ind],\
#                mean_sp[cond + (self.control_pos*manip)][start_ind:stop_ind] + sem_sp[cond + (self.control_pos*manip)][start_ind:stop_ind], facecolor=line_color[manip], alpha=0.3)





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




##### make psychometric curves with errors for each mouse

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
plt.errorbar(np.arange(9), gt015_lt_vm1_mean[0, :], yerr=gt015_lt_vm1_sem[0, :], c='k')
plt.errorbar(np.arange(9), gt015_lt_vm1_mean[1, :], yerr=gt015_lt_vm1_sem[1, :], c='r')


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
plt.errorbar(np.arange(9), gt015_lt_vs1_mean[0, :], yerr=gt015_lt_vs1_sem[0, :], c='k')
plt.errorbar(np.arange(9), gt015_lt_vs1_mean[1, :], yerr=gt015_lt_vs1_sem[1, :], c='r')


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
    gt017_nt_vm1_psy[k, :] = neuro.get_psychometric_curve(axis=ax)

# compute mean and sem of psychometric curve and plot
plt.figure()
gt017_nt_vm1_mean = np.reshape(np.mean(gt017_nt_vm1_psy, axis=0), [2, 9])
gt017_nt_vm1_sem = np.reshape(sp.stats.sem(gt017_nt_vm1_psy, axis=0), [2, 9])
plt.errorbar(np.arange(9), gt017_nt_vm1_mean[0, :], yerr=gt017_nt_vm1_sem[0, :], c='k')
plt.errorbar(np.arange(9), gt017_nt_vm1_mean[1, :], yerr=gt017_nt_vm1_sem[1, :], c='r')


















