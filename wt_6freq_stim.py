import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal
import statsmodels.stats.multitest as smm
from scipy.optimize import curve_fit
import sys
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


sns.set_style("whitegrid", {'axes.grid' : False})
npand   = np.logical_and
#dt      = 0.005 # seconds
#window  = [-0.06, 0.06]
#stim_times = np.arange(0.5, 1.5, 0.050)
#
#with PdfPages(fid + '_unit_stimulation_locked_summaries.pdf') as pdf:
#    for uid in range(neuro.num_units):
#        fig, ax = subplots(4, 2, figsize=(10,8))
#        fig.suptitle('Region: {}, depth: {}, unit type: {}, mouse: {}, driven: {}'.format(\
#                neuro.region_dict[neuro.shank_ids[uid]], \
#                neuro.depths[uid], \
#                neuro.cell_type[uid], \
#                fid, \
#                neuro.driven_units[uid]))
#
#        best_pos = int(neuro.best_contact[uid])
##        best_pos = 3
#
#        # best position
#        spks_per_bin, sem, bins = neuro.eta(stim_times, cond=best_pos, unit_ind=uid, window=window, dt=dt)
#        ax[0][0].bar(bins[:-1], spks_per_bin, width=dt, edgecolor='none')
#        ax[0][0].set_ylabel('Firing Rate (Hz)')
#
# plot mean and individual trial angles for all stim types
# two columns: S1 (left), M1 (right)
# rows: different stimulation types

s1_stim_inds = np.arange(1, 18, 3)
m1_stim_inds = np.arange(2, 18, 3)
num_stims    = len(whisk.stim_ids)

fig, ax = subplot(num_stims, 2)

for stim in range(num_stims):
    # plot S1 traces for stim
    ax[stim][0].plot(whisk.wtt, whisk.wt[stim][:, 0, :], linewidth=1.0) # this grey line
    ax[stim][0].plot(whisk.wtt, whisk.wt[stim][:, 0, :].mean(axis=0)) # thick black line

# PSD of all stimulation types


























