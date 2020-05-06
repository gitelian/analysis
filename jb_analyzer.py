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

### make summary figure for entire experiment ###
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



## for loading in other experiments if necessary
#get_ipython().magic(u"run hdfanalyzer.py {}".format(fid_name))
















#class BehaviorAnalysis(object):
#    def __init__(self, neuro):
#        self.neuro = neuro
#
#    def blah(self):
#        print('blah')
#
#if __name__ == "__main__":
#    print('loaded jb_analyzer')
