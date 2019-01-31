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


## plot mean set-point
fig, ax = neuro.plot_mean_whisker(t_window=[-1.5, 1.5], kind='setpoint', cond2plot=[0, 3, 4, 7, 8], all_trials=True)
fig.suptitle(fid + ' mean set-point ALL TRIALS')
ylim = ax[0].get_ylim()
for a in ax:
    a.vlines([-1, 1], ylim[0], ylim[1], colors='c')
    a.set_ylim(ylim)
    a.axvspan(-1, 0, alpha=0.2, color='green')

## plot mean amplitude
fig, ax = neuro.plot_mean_whisker(t_window=[-1.5, 1.5], kind='amplitude', cond2plot=[0, 3, 4, 7, 8], all_trials=True)
fig.suptitle(fid + ' mean amplitude ALL TRIALS')
ylim = ax[0].get_ylim()
for a in ax:
    a.vlines([-1, 1], ylim[0], ylim[1], colors='c')
    a.set_ylim(ylim)
    a.axvspan(-1, 0, alpha=0.2, color='green')

##TODO plot PSD of whisker angle

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

