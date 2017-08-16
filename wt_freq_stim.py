import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal
import statsmodels.stats.multitest as smm
from scipy.optimize import curve_fit
import sys
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl

whisk.reclassify_run_trials(mean_thresh=200, low_thresh=0)
whisk.wt_organize(running=False)
# general useful things
sns.set_style("whitegrid", {'axes.grid' : False})
npand   = np.logical_and

# plot mean and individual trial angles for all stim types
# two columns: S1 (left), M1 (right)
# rows: different stimulation types

#whisk.wt_organize(all_trials=True)

# stimulation parameters
stim_freqs = [2, 4, 8, 16, 32, 64]
#stim_freqs = [1, 4, 8, 12, 16, 20, 24, 28]
stim_times = [np.arange(0, 1, 1.0/freq) for freq in stim_freqs]

#s1_stim_inds = np.arange(1, 18, 3)
#m1_stim_inds = np.arange(2, 18, 3)
s1_stim_inds = np.arange(1, len(stim_freqs)*3, 3)
m1_stim_inds = np.arange(2, len(stim_freqs)*3, 3)
num_stims    = len(s1_stim_inds)
stim_time_inds = npand(whisk.wtt >= 0.0, whisk.wtt <= 1.0)



##### angle traces #####

# plotting parameters
ylow, yhigh = 60, 160 # degrees (yaxis lims)
#ylow, yhigh = -2*np.pi, 2*np.pi # degrees (yaxis lims)
#ylow, yhigh = 60, 80 # degrees (yaxis lims) # for phase tuning
xlow, xhigh = -1.0, 1.0 # time (xaxis lims)
wt_type = 0

fig, ax = subplots(num_stims, 2, figsize=(10,16), sharex=True, sharey=True)
plt.subplots_adjust(left=0.10, bottom=0.10, right=0.90, top=0.90, wspace=0.20, hspace=0.45)
# add annotations
ax[0][0].set_title('S1 stimulation\n{} Hz'.format(stim_freqs[0]))
ax[0][1].set_title('M1 stimulation\n{} Hz'.format(stim_freqs[0]))

for stim in range(num_stims):
    # plot S1 traces for stim
    ax[stim][0].plot(whisk.wtt, whisk.wt[s1_stim_inds[stim]][:, wt_type, :], linewidth=0.25, color='grey') # this grey line
    ax[stim][0].plot(whisk.wtt, whisk.wt[s1_stim_inds[stim]][:, wt_type, :].mean(axis=1), linewidth=2.0, color='black') # thick black line
    ax[stim][0].vlines(stim_times[stim], ylow, yhigh, color='red', linewidth=0.5)
    ax[stim][0].set_ylim([ylow, yhigh])
    ax[stim][0].set_xlim([xlow, xhigh])

    # plot M1 traces for stim
    ax[stim][1].plot(whisk.wtt, whisk.wt[m1_stim_inds[stim]][:, wt_type, :], linewidth=0.25, color='grey') # this grey line
    ax[stim][1].plot(whisk.wtt, whisk.wt[m1_stim_inds[stim]][:, wt_type, :].mean(axis=1), linewidth=2.0, color='black') # thick black line
    ax[stim][1].vlines(stim_times[stim], ylow, yhigh, color='red', linewidth=0.5)
    ax[stim][1].set_ylim([ylow, yhigh])
    ax[stim][1].set_xlim([xlow, xhigh])


    # annotations
    ax[stim][0].set_ylabel('angle (deg)')

    if stim > 0:
        ax[stim][0].set_title('{} Hz'.format(stim_freqs[stim]))
        ax[stim][1].set_title('{} Hz'.format(stim_freqs[stim]))

    if stim == num_stims - 1:
        ax[stim][0].set_xlabel('time (s)')
        ax[stim][1].set_xlabel('time (s)')



##### PSD of all stimulation types #####

# plotting parameters
ylow, yhigh = 0, 200 # arbitrary power (yaxis lims)
ylow, yhigh = 0, 20 # arbitrary power (yaxis lims)
xlow, xhigh = 0, 30  # frequency (xaxis lims)

cmap = mpl.cm.viridis
cmap = mpl.cm.hot
fig, ax = subplots(1, 2, figsize=(8,4), sharex=True, sharey=True)

# add annotations
ax[0].set_title('S1 stimulation')
ax[1].set_title('M1 stimulation')

for k, stim in enumerate(stim_freqs):
    # compute and plot PSD for S1
#    if stim == 1 or stim == 12 or stim == 20:
    f, frq_mat_temp = whisk.get_psd(whisk.wt[s1_stim_inds[k]][stim_time_inds, 0, :], 500.0)
    whisk.plot_freq(f, frq_mat_temp, axis=ax[0], color=cmap(k / float(num_stims)))
    ax[0].set_ylim([ylow, yhigh])
    ax[0].set_xlim([xlow, xhigh])

    # compute and plot PSD for M1
    f, frq_mat_temp = whisk.get_psd(whisk.wt[m1_stim_inds[k]][stim_time_inds, 0, :], 500.0)
    whisk.plot_freq(f, frq_mat_temp, axis=ax[1], color=cmap(k/ float(num_stims)))
    ax[1].set_ylim([ylow, yhigh])
    ax[1].set_xlim([xlow, xhigh])

ax[0].legend(stim_freqs)
ax[1].legend(stim_freqs)




















