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
cond2plot=[3, 4]

## plot mean set-point
fig, ax = neuro.plot_mean_whisker(t_window=[-1.5, 1.5], kind='setpoint', cond2plot=cond2plot, all_trials=True)
fig.suptitle(fid + ' mean set-point ALL TRIALS')
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



for cond in range(len(neuro.stim_ids)):

    cond = 3
    all_setpoints_nolight = list()
    all_runspeeds_nolight  = list()

    all_setpoints_light = list()
    all_runspeeds_light  = list()

    for trial in range(len(neuro.lick_bool[cond])):
        all_setpoints_nolight.append(neuro.wt[cond][:, 1, trial])
        all_runspeeds_nolight.append(neuro.run[cond][run_inds2keep, trial])

    for trial in range(len(neuro.lick_bool[cond + neuro.control_pos])):
        all_setpoints_light.append(neuro.wt[cond + neuro.control_pos][:, 1, trial])
        all_runspeeds_light.append(neuro.run[cond + neuro.control_pos][run_inds2keep, trial])

    all_setpoints_nolight = np.asarray(all_setpoints_nolight)
    all_runspeeds_nolight = np.asarray(all_runspeeds_nolight)

    all_setpoints_light = np.asarray(all_setpoints_light)
    all_runspeeds_light = np.asarray(all_runspeeds_light)

    corrcoef_nolight = list()
    corrcoef_light = list()
    for sample in range(all_setpoints_nolight.shape[1]):
        corrcoef_nolight.append(sp.stats.pearsonr(all_runspeeds_nolight[:, sample], all_setpoints_nolight[:, sample])[0])
        corrcoef_light.append(sp.stats.pearsonr(all_runspeeds_light[:, sample], all_setpoints_light[:, sample])[0])


#### THINK ABOUT THIS. SOMETHING ISNT RIGHT. why am I getting a positive correlation
#### when there is clearly a negative one?????
    plot(neuro.wtt, corrcoef_nolight, 'k', neuro.wtt, corrcoef_light, 'b')






























#        if all_trials:
#            whisk_kinematic[cond].append(neuro.wt[cond][:, whisk_ind, trial])
#
#        # if mouse made correct choice
#        elif neuro.trial_choice[cond][trial]:
#            # get set-point
#            whisk_kinematic[cond].append(neuro.wt[cond][:, whisk_ind, trial])





























