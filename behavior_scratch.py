#!/bin/bash

########## MAIN CODE ##########
########## MAIN CODE ##########

if __name__ == "__main__":
    # JB Behavior stimulus ID 9 is CATCH TRIAL
    #
    # when defining a position (usually variable 'pos') use the experiment ID
    # for example position 1 is the first "GO" position, the code will subtract
    # one to deal with zero based indexing

    sns.set_style("whitegrid", {'axes.grid' : False})

    if os.path.isdir('/media/greg/data/behavior/hdfbehavior/'):
        data_dir = '/media/greg/data/behavior/hdfbehavior/'
    elif os.path.isdir('/jenny/add/your/path/here/'):
        data_dir = '/jenny/add/your/path/here/'

    # each experiment needs an entry in the CSV file!
    mouse      = 'GT009'
    experiment = 'FID1664'

    hdf_name   = mouse + '_' + experiment
    fid        = experiment
    hdf_fname  = data_dir + hdf_name + '.hdf5'

    f = h5py.File(hdf_fname,'r+')

    whisk = BehaviorAnalyzer(f, fid, data_dir)

    # reclassify trials as run vs non-run
    whisk.reclassify_run_trials(mean_thresh=100, low_thresh=50)
    whisk.wt_organize()

    # remove entries for trial type "0"
    stim_ids = np.unique(whisk.stim_ids_all)
    if 0 in stim_ids:
        whisk.stim_ids = whisk.stim_ids[1:]
        whisk.wt       = whisk.wt[1:]
        whisk.bids     = whisk.bids[1:]
        whisk.run      = whisk.run[1:]
        whisk.licks    = whisk.licks[1:]


##### SCRATCH SPACE #####
##### SCRATCH SPACE #####



##### plot whisking variable vs time for all trials #####
##### plot whisking variable vs time for all trials #####

fig, ax = plt.subplots(2, 1)
dtype = 1 # 0, angle; 1, set-point; 2, amplitude; 3, phase; 4, velocity; 5, "whisk".
pos = 4

# GO trial
ax[0].plot(whisk.wtt, whisk.wt[pos-1][:, dtype, :], linewidth=0.5)
ax[0].plot(whisk.wtt, np.mean(whisk.wt[pos-1][:, dtype, :], axis=1), 'k')
ax[0].set_ylim(90, 160)

# NOGO trial
ax[1].plot(whisk.wtt, whisk.wt[9-pos-1][:, dtype, :], linewidth=0.5)
ax[1].plot(whisk.wtt, np.mean(whisk.wt[9-pos-1][:, dtype, :], axis=1), 'k')
ax[1].set_ylim(90, 160)
fig.show()


#### for S2 session ####
########################
### DELETE THIS ###
########################
fig, ax = plt.subplots(2, 1)
dtype = 2 # 0, angle; 1, set-point; 2, amplitude; 3, phase; 4, velocity; 5, "whisk".
pos = 4

offset = 0
# GO trial
ax[0].plot(whisk.wtt, whisk.wt[0+offset][:, dtype, :], linewidth=0.5)
ax[0].plot(whisk.wtt, np.mean(whisk.wt[0+offset][:, dtype, :], axis=1), 'k')
ax[0].set_ylim(90, 160)

# NOGO trial
ax[1].plot(whisk.wtt, whisk.wt[1+offset][:, dtype, :], linewidth=0.5)
ax[1].plot(whisk.wtt, np.mean(whisk.wt[1+offset][:, dtype, :], axis=1), 'k')
ax[1].set_ylim(90, 160)
fig.show()

ax[0].hlines(155, -1, 1, 'b')
ax[1].hlines(155, -1, 1, 'b')
##### plot whisking variable vs time for running and hit/miss trials #####
##### plot whisking variable vs time for running and hit/miss trials #####

dtype = 1 # 0, angle; 1, set-point; 2, amplitude; 3, phase; 4, velocity; 5, "whisk".
pos = 1

fig, ax = plt.subplots(4, 1, figsize=(7, 12))
fig.subplots_adjust(hspace=0.4)
fig.suptitle('Position {}'.format(pos))
# GO trial
# given an angle/position get indices for hit trials
good_inds = np.where(whisk.bids[pos-1] == 1)[0]
if len(good_inds) > 0:
    ax[0].plot(whisk.wtt, whisk.wt[pos-1][:, dtype, good_inds], linewidth=0.5)
    ax[0].plot(whisk.wtt, np.mean(whisk.wt[pos-1][:, dtype, good_inds], axis=1), 'k')
    ax[0].set_ylim(90, 160)
ax[0].set_title('GO + "lick"')

# GO + miss
good_inds = np.where(whisk.bids[pos-1] == 3)[0]
if len(good_inds) > 0:
    ax[1].plot(whisk.wtt, whisk.wt[pos-1][:, dtype, good_inds], linewidth=0.5)
    ax[1].plot(whisk.wtt, np.mean(whisk.wt[pos-1][:, dtype, good_inds], axis=1), 'k')
    ax[1].set_ylim(90, 160)
ax[1].set_title('GO + "miss"')

# NOGO + false alarm
good_inds = np.where(whisk.bids[9-pos-1] == 2)[0]
if len(good_inds) > 0:
    ax[2].plot(whisk.wtt, whisk.wt[9-pos-1][:, dtype, good_inds], linewidth=0.5)
    ax[2].plot(whisk.wtt, np.mean(whisk.wt[9-pos-1][:, dtype, good_inds], axis=1), 'k')
    ax[2].set_ylim(90, 160)
ax[2].set_title('NOGO + "false alarm"')

# NOGO + correct rejection
good_inds = np.where(whisk.bids[9-pos-1] == 4)[0]
if len(good_inds) > 0:
    ax[3].plot(whisk.wtt, whisk.wt[9-pos-1][:, dtype, good_inds], linewidth=0.5)
    ax[3].plot(whisk.wtt, np.mean(whisk.wt[9-pos-1][:, dtype, good_inds], axis=1), 'k')
    ax[3].set_ylim(90, 160)
ax[3].set_title('NOGO + "correct rejection"')


##### plot whisking variable vs time for running and hit/miss trials aligned to first lick #####
##### plot whisking variable vs time for running and hit/miss trials aligned to first lick #####

dtype = 0 # 0, angle; 1, set-point; 2, amplitude; 3, phase; 4, velocity; 5, "whisk".
pos = 4

#### scratch ####
def get_first_lick_times(self, pos=pos, window=[0, 1]):
    first_licks = list()
    lick_lists = self.licks[pos]
    for licks in lick_lists:
        lick_inds = np.logical_and(licks >= window[0], licks <= window[1])

        if sum(lick_inds) > 0:
            first_licks.append(licks[lick_inds][0])
        else:
            first_licks.append(np.nan)

    return np.asarray(first_licks)

event_times = get_first_lick_times(whisk,pos=pos, window=[0, 1])
good_inds = np.where(whisk.bids[pos-1] == 1)[0]
event_times = event_times[good_inds]
mean_trace, sem_trace, mat_trace, t_trace = whisk.eta_wt(event_times, whisk.wt[pos-1][:, dtype, good_inds], cond=pos, kind='set-point', window=[-2, 1], etimes_per_trial=True)
plt.plot(t_trace, mean_trace)#, color='k')
plt.fill_between(t_trace, mean_trace - sem_trace, mean_trace + sem_trace, alpha=0.3)#, facecolor='k', alpha=0.3)
#### scratch ####


fig, ax = plt.subplots(4, 1, figsize=(7, 12))
fig.subplots_adjust(hspace=0.4)
fig.suptitle('Position {}'.format(pos))
# GO trial
# given an angle/position get indices for hit trials
good_inds = np.where(whisk.bids[pos-1] == 1)[0]
if len(good_inds) > 0:
    ax[0].plot(whisk.wtt, whisk.wt[pos-1][:, dtype, good_inds], linewidth=0.5)
    ax[0].plot(whisk.wtt, np.mean(whisk.wt[pos-1][:, dtype, good_inds], axis=1), 'k')
    ax[0].set_ylim(90, 160)
ax[0].set_title('GO + "lick"')

# GO + miss
good_inds = np.where(whisk.bids[pos-1] == 3)[0]
if len(good_inds) > 0:
    ax[1].plot(whisk.wtt, whisk.wt[pos-1][:, dtype, good_inds], linewidth=0.5)
    ax[1].plot(whisk.wtt, np.mean(whisk.wt[pos-1][:, dtype, good_inds], axis=1), 'k')
    ax[1].set_ylim(90, 160)
ax[1].set_title('GO + "miss"')

# NOGO + false alarm
good_inds = np.where(whisk.bids[9-pos-1] == 2)[0]
if len(good_inds) > 0:
    ax[2].plot(whisk.wtt, whisk.wt[9-pos-1][:, dtype, good_inds], linewidth=0.5)
    ax[2].plot(whisk.wtt, np.mean(whisk.wt[9-pos-1][:, dtype, good_inds], axis=1), 'k')
    ax[2].set_ylim(90, 160)
ax[2].set_title('NOGO + "false alarm"')

# NOGO + correct rejection
good_inds = np.where(whisk.bids[9-pos-1] == 4)[0]
if len(good_inds) > 0:
    ax[3].plot(whisk.wtt, whisk.wt[9-pos-1][:, dtype, good_inds], linewidth=0.5)
    ax[3].plot(whisk.wtt, np.mean(whisk.wt[9-pos-1][:, dtype, good_inds], axis=1), 'k')
    ax[3].set_ylim(90, 160)
ax[3].set_title('NOGO + "correct rejection"')


##### plot whisking variable + running (mean +/- sem ) vs time for running and hit/miss trials #####
##### plot whisking variable + running (mean +/- sem ) vs time for running and hit/miss trials #####

dtype = 1 # 0, angle; 1, set-point; 2, amplitude; 3, phase; 4, velocity; 5, "whisk".
pos = 1

fig, ax = plt.subplots(4, 1, figsize=(7, 12))
fig.subplots_adjust(hspace=0.4)
fig.suptitle('Position {}'.format(pos))
# GO trial
# given an angle/position get indices for hit trials
good_inds = np.where(whisk.bids[pos-1] == 1)[0]
if len(good_inds) > 0:
    mean_wt = np.mean(whisk.wt[pos-1][:, dtype, good_inds], axis=1)
    sem_wt  = sp.stats.sem(whisk.wt[pos-1][:, dtype, good_inds], axis=1)
    ax[0].plot(whisk.wtt, mean_wt, color='k')
    ax[0].fill_between(whisk.wtt, mean_wt - sem_wt, mean_wt + sem_wt, facecolor='k', alpha=0.3)

    ax2 = ax[0].twinx()
    mean_run = np.mean(whisk.run[pos-1][:, good_inds], axis=1)
    sem_run  = sp.stats.sem(whisk.run[pos-1][:, good_inds], axis=1)
    ax2.plot(whisk.run_t, mean_run, color='r')
    ax2.fill_between(whisk.run_t, mean_run - sem_run, mean_run + sem_run, facecolor='r', alpha=0.3)

    ax[0].set_ylim(90, 160)
ax[0].set_title('GO + "lick"')

# GO + miss
good_inds = np.where(whisk.bids[pos-1] == 3)[0]
if len(good_inds) > 0:
    mean_wt = np.mean(whisk.wt[pos-1][:, dtype, good_inds], axis=1)
    sem_wt  = sp.stats.sem(whisk.wt[pos-1][:, dtype, good_inds], axis=1)
    ax[1].plot(whisk.wtt, mean_wt, color='k')
    ax[1].fill_between(whisk.wtt, mean_wt - sem_wt, mean_wt + sem_wt, facecolor='k', alpha=0.3)

    ax2 = ax[1].twinx()
    mean_run = np.mean(whisk.run[pos-1][:, good_inds], axis=1)
    sem_run  = sp.stats.sem(whisk.run[pos-1][:, good_inds], axis=1)
    ax2.plot(whisk.run_t, mean_run, color='r')
    ax2.fill_between(whisk.run_t, mean_run - sem_run, mean_run + sem_run, facecolor='r', alpha=0.3)

    ax[1].set_ylim(90, 160)

ax[1].set_title('GO + "miss"')

# NOGO + false alarm
good_inds = np.where(whisk.bids[9-pos-1] == 2)[0]
if len(good_inds) > 0:
    mean_wt = np.mean(whisk.wt[9-pos-1][:, dtype, good_inds], axis=1)
    sem_wt  = sp.stats.sem(whisk.wt[9-pos-1][:, dtype, good_inds], axis=1)
    ax[2].plot(whisk.wtt, mean_wt, color='k')
    ax[2].fill_between(whisk.wtt, mean_wt - sem_wt, mean_wt + sem_wt, facecolor='k', alpha=0.3)

    ax2 = ax[2].twinx()
    mean_run = np.mean(whisk.run[9-pos-1][:, good_inds], axis=1)
    sem_run  = sp.stats.sem(whisk.run[9-pos-1][:, good_inds], axis=1)
    ax2.plot(whisk.run_t, mean_run, color='r')
    ax2.fill_between(whisk.run_t, mean_run - sem_run, mean_run + sem_run, facecolor='r', alpha=0.3)

    ax[2].set_ylim(90, 160)

ax[2].set_title('NOGO + "false alarm"')

# NOGO + correct rejection
good_inds = np.where(whisk.bids[9-pos-1] == 4)[0]
if len(good_inds) > 0:
    mean_wt = np.mean(whisk.wt[9-pos-1][:, dtype, good_inds], axis=1)
    sem_wt  = sp.stats.sem(whisk.wt[9-pos-1][:, dtype, good_inds], axis=1)
    ax[3].plot(whisk.wtt, mean_wt, color='k')
    ax[3].fill_between(whisk.wtt, mean_wt - sem_wt, mean_wt + sem_wt, facecolor='k', alpha=0.3)

    ax2 = ax[3].twinx()
    mean_run = np.mean(whisk.run[9-pos-1][:, good_inds], axis=1)
    sem_run  = sp.stats.sem(whisk.run[9-pos-1][:, good_inds], axis=1)
    ax2.plot(whisk.run_t, mean_run, color='r')
    ax2.fill_between(whisk.run_t, mean_run - sem_run, mean_run + sem_run, facecolor='r', alpha=0.3)

    ax[3].set_ylim(90, 160)

ax[3].set_title('NOGO + "correct rejection"')


##### make power spectral density plots of whisking #####
##### make power spectral density plots of whisking #####
pos=1
fig, ax = plt.subplots(4,1)

for pos in range(1,5):
    t_inds = np.logical_and(whisk.wtt > -0.5, whisk.wtt < 0.5)
    f, frq_mat_temp = whisk.get_psd(whisk.wt[pos-1][t_inds, 0, :], 500)
    whisk.plot_freq(f, frq_mat_temp, axis=ax[pos-1], color='black')

#    f, frq_mat_temp = whisk.get_psd(whisk.wt[9-pos-1][t_inds, 0, :], 500)
#    whisk.plot_freq(f, frq_mat_temp, axis=ax[pos-1], color='red')
    ax[pos-1].set_xlim(0, 30)
    ax[pos-1].set_ylim(1e-1, 1e3)
    ax[pos-1].set_yscale('log')
    ax[pos-1].set_title('position {}'.format(pos))
fig.show()

########################
### DELETE THIS
########################
pos=1
fig, ax = plt.subplots(2,1)

# GO
t_inds = np.logical_and(whisk.wtt > -0.5, whisk.wtt < 0.5)
f, frq_mat_temp = whisk.get_psd(whisk.wt[0][t_inds, 0, :], 500)
whisk.plot_freq(f, frq_mat_temp, axis=ax[0], color='black')

# NOGO
t_inds = np.logical_and(whisk.wtt > -0.5, whisk.wtt < 0.5)
f, frq_mat_temp = whisk.get_psd(whisk.wt[1][t_inds, 0, :], 500)
whisk.plot_freq(f, frq_mat_temp, axis=ax[1], color='black')

# GO vM1 light
t_inds = np.logical_and(whisk.wtt > -0.5, whisk.wtt < 0.5)
f, frq_mat_temp = whisk.get_psd(whisk.wt[3][t_inds, 0, :], 500)
whisk.plot_freq(f, frq_mat_temp, axis=ax[0], color='blue')

# NOGO vM1 light
t_inds = np.logical_and(whisk.wtt > -0.5, whisk.wtt < 0.5)
f, frq_mat_temp = whisk.get_psd(whisk.wt[4][t_inds, 0, :], 500)
whisk.plot_freq(f, frq_mat_temp, axis=ax[1], color='blue')



#    f, frq_mat_temp = whisk.get_psd(whisk.wt[9-pos-1][t_inds, 0, :], 500)
#    whisk.plot_freq(f, frq_mat_temp, axis=ax[pos-1], color='red')
for a in range(2):
    ax[a].set_xlim(0, 30)
    ax[a].set_ylim(1e-1, 1e3)
    ax[a].set_yscale('log')
    ax[a].set_title('position {}'.format(pos))
fig.show()

##### make spectrogram plots of whisking #####
##### make spectrogram plots of whisking #####
pos = 4
fig, ax = plt.subplots(2,1)
fig.suptitle('position {}'.format(pos))

f, t, Sxx_mat_temp = whisk.get_spectrogram(whisk.wt[pos-1][:, 0, :], 500)
im = whisk.plot_spectrogram(f, t, Sxx_mat_temp, axis=ax[0], vmin=0.1, vmax=100, log=True)
fig.colorbar(im, ax=ax[0])
ax[0].set_ylim(0, 30)

f, t, Sxx_mat_temp = whisk.get_spectrogram(whisk.wt[9-pos-1][:, 0, :], 500)
im = whisk.plot_spectrogram(f, t, Sxx_mat_temp, axis=ax[1], vmin=0.1, vmax=100, log=True)
fig.colorbar(im, ax=ax[1])
ax[1].set_ylim(0, 30)
fig.show()


##### make difference mean spectrogram plots of whisking #####
##### make difference mean spectrogram plots of whisking #####

vmin, vmax = -2.5, 2.5
fig, ax = plt.subplots(4,1)

# position 1
# get indices for samples between -0.5s and 0.5s
f, t, Sxx_mat_go   = whisk.get_spectrogram(whisk.wt[1-1][:, 0, :], 500)
f, t, Sxx_mat_nogo = whisk.get_spectrogram(whisk.wt[9-1-1][:, 0, :], 500)
mean_Sxx_go = np.mean(Sxx_mat_go, axis=2)
mean_Sxx_nogo = np.mean(Sxx_mat_nogo, axis=2)
diff_go_nogo = mean_Sxx_go - mean_Sxx_nogo

im = ax[0].pcolormesh(t-1, f, diff_go_nogo, cmap='coolwarm', vmin=-2.5, vmax=2.5)#, norm=colors.LogNorm(vmin=0.1, vmax=vmax))
ax[0].set_ylim(0, 30)
fig.colorbar(im, ax=ax[0])
ax[0].set_title('position 1')
ax[0].set_ylabel('frequency (Hz)')

# position 2
f, t, Sxx_mat_go   = whisk.get_spectrogram(whisk.wt[2-1][:, 0, :], 500)
f, t, Sxx_mat_nogo = whisk.get_spectrogram(whisk.wt[9-2-1][:, 0, :], 500)
mean_Sxx_go = np.mean(Sxx_mat_go, axis=2)
mean_Sxx_nogo = np.mean(Sxx_mat_nogo, axis=2)
diff_go_nogo = mean_Sxx_go - mean_Sxx_nogo

im = ax[1].pcolormesh(t-1, f, diff_go_nogo, cmap='coolwarm', vmin=-2.5, vmax=2.5)#, norm=colors.LogNorm(vmin=0.1, vmax=vmax))
ax[1].set_ylim(0, 30)
fig.colorbar(im, ax=ax[1])
ax[1].set_title('position 2')

# position 3
f, t, Sxx_mat_go   = whisk.get_spectrogram(whisk.wt[3-1][:, 0, :], 500)
f, t, Sxx_mat_nogo = whisk.get_spectrogram(whisk.wt[9-3-1][:, 0, :], 500)
mean_Sxx_go = np.mean(Sxx_mat_go, axis=2)
mean_Sxx_nogo = np.mean(Sxx_mat_nogo, axis=2)
diff_go_nogo = mean_Sxx_go - mean_Sxx_nogo

im = ax[2].pcolormesh(t-1, f, diff_go_nogo, cmap='coolwarm', vmin=-2.5, vmax=2.5)#, norm=colors.LogNorm(vmin=0.1, vmax=vmax))
ax[2].set_ylim(0, 30)
fig.colorbar(im, ax=ax[2])
ax[2].set_title('position 3')

# position 4
f, t, Sxx_mat_go   = whisk.get_spectrogram(whisk.wt[4-1][:, 0, :], 500)
f, t, Sxx_mat_nogo = whisk.get_spectrogram(whisk.wt[9-4-1][:, 0, :], 500)
mean_Sxx_go = np.mean(Sxx_mat_go, axis=2)
mean_Sxx_nogo = np.mean(Sxx_mat_nogo, axis=2)
diff_go_nogo = mean_Sxx_go - mean_Sxx_nogo

im = ax[3].pcolormesh(t-1, f, diff_go_nogo, cmap='coolwarm', vmin=-2.5, vmax=2.5)#, norm=colors.LogNorm(vmin=0.1, vmax=vmax))
ax[3].set_ylim(0, 30)
fig.colorbar(im, ax=ax[3])
ax[3].set_title('position 4')
ax[3].set_xlabel('time (s)')


##### make difference mean spectrogram plots of whisking #####
##### make difference mean spectrogram plots of whisking #####

vmin, vmax = -2.5, 2.5
fig, ax = plt.subplots(4,1)

# position 1
good_go_inds = np.where(whisk.bids[1-1] == 1)[0]
good_nogo_inds = np.where(whisk.bids[9-1-1] == 4)[0]
f, t, Sxx_mat_go   = whisk.get_spectrogram(whisk.wt[1-1][:, 0, good_go_inds], 500)
f, t, Sxx_mat_nogo = whisk.get_spectrogram(whisk.wt[9-1-1][:, 0, good_nogo_inds], 500)
mean_Sxx_go = np.mean(Sxx_mat_go, axis=2)
mean_Sxx_nogo = np.mean(Sxx_mat_nogo, axis=2)
diff_go_nogo = mean_Sxx_go - mean_Sxx_nogo

im = ax[0].pcolormesh(t-1, f, diff_go_nogo, cmap='coolwarm', vmin=-2.5, vmax=2.5)#, norm=colors.LogNorm(vmin=0.1, vmax=vmax))
ax[0].set_ylim(0, 30)
fig.colorbar(im, ax=ax[0])
ax[0].set_title('position 1')
ax[0].set_ylabel('frequency (Hz)')

# position 2
good_go_inds = np.where(whisk.bids[2-1] == 1)[0]
good_nogo_inds = np.where(whisk.bids[9-2-1] == 4)[0]
f, t, Sxx_mat_go   = whisk.get_spectrogram(whisk.wt[2-1][:, 0, good_go_inds], 500)
f, t, Sxx_mat_nogo = whisk.get_spectrogram(whisk.wt[9-2-1][:, 0, good_nogo_inds], 500)
mean_Sxx_go = np.mean(Sxx_mat_go, axis=2)
mean_Sxx_nogo = np.mean(Sxx_mat_nogo, axis=2)
diff_go_nogo = mean_Sxx_go - mean_Sxx_nogo

im = ax[1].pcolormesh(t-1, f, diff_go_nogo, cmap='coolwarm', vmin=-2.5, vmax=2.5)#, norm=colors.LogNorm(vmin=0.1, vmax=vmax))
ax[1].set_ylim(0, 30)
fig.colorbar(im, ax=ax[1])
ax[1].set_title('position 2')
ax[1].set_ylabel('frequency (Hz)')

# position 3
good_go_inds = np.where(whisk.bids[3-1] == 1)[0]
good_nogo_inds = np.where(whisk.bids[9-3-1] == 4)[0]
f, t, Sxx_mat_go   = whisk.get_spectrogram(whisk.wt[3-1][:, 0, good_go_inds], 500)
f, t, Sxx_mat_nogo = whisk.get_spectrogram(whisk.wt[9-3-1][:, 0, good_nogo_inds], 500)
mean_Sxx_go = np.mean(Sxx_mat_go, axis=2)
mean_Sxx_nogo = np.mean(Sxx_mat_nogo, axis=2)
diff_go_nogo = mean_Sxx_go - mean_Sxx_nogo

im = ax[2].pcolormesh(t-1, f, diff_go_nogo, cmap='coolwarm', vmin=-2.5, vmax=2.5)#, norm=colors.LogNorm(vmin=0.1, vmax=vmax))
ax[2].set_ylim(0, 30)
fig.colorbar(im, ax=ax[2])
ax[2].set_title('position 3')
ax[2].set_ylabel('frequency (Hz)')

# position 4
good_go_inds = np.where(whisk.bids[4-1] == 1)[0]
good_nogo_inds = np.where(whisk.bids[9-4-1] == 4)[0]
f, t, Sxx_mat_go   = whisk.get_spectrogram(whisk.wt[4-1][:, 0, good_go_inds], 500)
f, t, Sxx_mat_nogo = whisk.get_spectrogram(whisk.wt[9-4-1][:, 0, good_nogo_inds], 500)
mean_Sxx_go = np.mean(Sxx_mat_go, axis=2)
mean_Sxx_nogo = np.mean(Sxx_mat_nogo, axis=2)
diff_go_nogo = mean_Sxx_go - mean_Sxx_nogo

im = ax[3].pcolormesh(t-1, f, diff_go_nogo, cmap='coolwarm', vmin=-2.5, vmax=2.5)#, norm=colors.LogNorm(vmin=0.1, vmax=vmax))
ax[3].set_ylim(0, 30)
fig.colorbar(im, ax=ax[3])
ax[3].set_title('position 4')
ax[3].set_ylabel('frequency (Hz)')



##### make difference mean spectrogram plots of whisking for GO positions #####
##### make difference mean spectrogram plots of whisking for GO positions #####

vmin, vmax = -2.5, 2.5
fig, ax = plt.subplots(3,3)

# position 1 vs 2
good_go01_inds = np.where(whisk.bids[1-1] == 1)[0]
good_go04_inds = np.where(whisk.bids[2-1] == 1)[0]
f, t, Sxx_mat_go   = whisk.get_spectrogram(whisk.wt[1-1][:, 0, good_go01_inds], 500)
f, t, Sxx_mat_nogo = whisk.get_spectrogram(whisk.wt[2-1][:, 0, good_go04_inds], 500)
mean_Sxx_go = np.mean(Sxx_mat_go, axis=2)
mean_Sxx_nogo = np.mean(Sxx_mat_nogo, axis=2)
diff_go_nogo = mean_Sxx_go - mean_Sxx_nogo

im = ax[0][0].pcolormesh(t-1, f, diff_go_nogo, cmap='coolwarm', vmin=-2.5, vmax=2.5)#, norm=colors.LogNorm(vmin=0.1, vmax=vmax))
ax[0][0].set_ylim(0, 30)
fig.colorbar(im, ax=ax[0][0])
ax[0][0].set_title('position 1 vs 2')
ax[0][0].set_ylabel('frequency (Hz)')

# position 1 vs 3
good_go01_inds = np.where(whisk.bids[1-1] == 1)[0]
good_go04_inds = np.where(whisk.bids[3-1] == 1)[0]
f, t, Sxx_mat_go   = whisk.get_spectrogram(whisk.wt[1-1][:, 0, good_go01_inds], 500)
f, t, Sxx_mat_nogo = whisk.get_spectrogram(whisk.wt[3-1][:, 0, good_go04_inds], 500)
mean_Sxx_go = np.mean(Sxx_mat_go, axis=2)
mean_Sxx_nogo = np.mean(Sxx_mat_nogo, axis=2)
diff_go_nogo = mean_Sxx_go - mean_Sxx_nogo

im = ax[0][1].pcolormesh(t-1, f, diff_go_nogo, cmap='coolwarm', vmin=-2.5, vmax=2.5)#, norm=colors.LogNorm(vmin=0.1, vmax=vmax))
ax[0][1].set_ylim(0, 30)
fig.colorbar(im, ax=ax[0][1])
ax[0][1].set_title('position 1 vs 3')
ax[0][1].set_ylabel('frequency (Hz)')

# position 1 vs 4
good_go01_inds = np.where(whisk.bids[1-1] == 1)[0]
good_go04_inds = np.where(whisk.bids[4-1] == 1)[0]
f, t, Sxx_mat_go   = whisk.get_spectrogram(whisk.wt[1-1][:, 0, good_go01_inds], 500)
f, t, Sxx_mat_nogo = whisk.get_spectrogram(whisk.wt[4-1][:, 0, good_go04_inds], 500)
mean_Sxx_go = np.mean(Sxx_mat_go, axis=2)
mean_Sxx_nogo = np.mean(Sxx_mat_nogo, axis=2)
diff_go_nogo = mean_Sxx_go - mean_Sxx_nogo

im = ax[0][2].pcolormesh(t-1, f, diff_go_nogo, cmap='coolwarm', vmin=-2.5, vmax=2.5)#, norm=colors.LogNorm(vmin=0.1, vmax=vmax))
ax[0][2].set_ylim(0, 30)
fig.colorbar(im, ax=ax[0][2])
ax[0][2].set_title('position 1 vs 4')
ax[0][2].set_ylabel('frequency (Hz)')

# position 2 vs 3
good_go01_inds = np.where(whisk.bids[2-1] == 1)[0]
good_go04_inds = np.where(whisk.bids[3-1] == 1)[0]
f, t, Sxx_mat_go   = whisk.get_spectrogram(whisk.wt[2-1][:, 0, good_go01_inds], 500)
f, t, Sxx_mat_nogo = whisk.get_spectrogram(whisk.wt[3-1][:, 0, good_go04_inds], 500)
mean_Sxx_go = np.mean(Sxx_mat_go, axis=2)
mean_Sxx_nogo = np.mean(Sxx_mat_nogo, axis=2)
diff_go_nogo = mean_Sxx_go - mean_Sxx_nogo

im = ax[1][1].pcolormesh(t-1, f, diff_go_nogo, cmap='coolwarm', vmin=-2.5, vmax=2.5)#, norm=colors.LogNorm(vmin=0.1, vmax=vmax))
ax[1][1].set_ylim(0, 30)
fig.colorbar(im, ax=ax[1][1])
ax[1][1].set_title('position 2 vs 3')
ax[1][1].set_ylabel('frequency (Hz)')

# position 2 vs 4
good_go01_inds = np.where(whisk.bids[2-1] == 1)[0]
good_go04_inds = np.where(whisk.bids[4-1] == 1)[0]
f, t, Sxx_mat_go   = whisk.get_spectrogram(whisk.wt[2-1][:, 0, good_go01_inds], 500)
f, t, Sxx_mat_nogo = whisk.get_spectrogram(whisk.wt[4-1][:, 0, good_go04_inds], 500)
mean_Sxx_go = np.mean(Sxx_mat_go, axis=2)
mean_Sxx_nogo = np.mean(Sxx_mat_nogo, axis=2)
diff_go_nogo = mean_Sxx_go - mean_Sxx_nogo

im = ax[1][2].pcolormesh(t-1, f, diff_go_nogo, cmap='coolwarm', vmin=-2.5, vmax=2.5)#, norm=colors.LogNorm(vmin=0.1, vmax=vmax))
ax[1][2].set_ylim(0, 30)
fig.colorbar(im, ax=ax[1][2])
ax[1][2].set_title('position 2 vs 4')
ax[1][2].set_ylabel('frequency (Hz)')

# position 3 vs 4
good_go01_inds = np.where(whisk.bids[3-1] == 1)[0]
good_go04_inds = np.where(whisk.bids[4-1] == 1)[0]
f, t, Sxx_mat_go   = whisk.get_spectrogram(whisk.wt[3-1][:, 0, good_go01_inds], 500)
f, t, Sxx_mat_nogo = whisk.get_spectrogram(whisk.wt[4-1][:, 0, good_go04_inds], 500)
mean_Sxx_go = np.mean(Sxx_mat_go, axis=2)
mean_Sxx_nogo = np.mean(Sxx_mat_nogo, axis=2)
diff_go_nogo = mean_Sxx_go - mean_Sxx_nogo

im = ax[2][2].pcolormesh(t-1, f, diff_go_nogo, cmap='coolwarm', vmin=-2.5, vmax=2.5)#, norm=colors.LogNorm(vmin=0.1, vmax=vmax))
ax[2][2].set_ylim(0, 30)
fig.colorbar(im, ax=ax[2][2])
ax[2][2].set_title('position 3 vs 4')
ax[2][2].set_ylabel('frequency (Hz)')


##### make difference mean spectrogram plots of whisking for NOGO positions #####
##### make difference mean spectrogram plots of whisking for NOGO positions #####

vmin, vmax = -2.5, 2.5
fig, ax = plt.subplots(3,3)

# position 1 vs 2
good_nogo01_inds = np.where(whisk.bids[9-1-1] == 4)[0]
good_nogo04_inds = np.where(whisk.bids[9-2-1] == 4)[0]
f, t, Sxx_mat_go   = whisk.get_spectrogram(whisk.wt[9-1-1][:, 0, good_nogo01_inds], 500)
f, t, Sxx_mat_nogo = whisk.get_spectrogram(whisk.wt[9-2-1][:, 0, good_nogo04_inds], 500)
mean_Sxx_go = np.mean(Sxx_mat_go, axis=2)
mean_Sxx_nogo = np.mean(Sxx_mat_nogo, axis=2)
diff_go_nogo = mean_Sxx_go - mean_Sxx_nogo

im = ax[0][0].pcolormesh(t-1, f, diff_go_nogo, cmap='coolwarm', vmin=-2.5, vmax=2.5)#, norm=colors.LogNorm(vmin=0.1, vmax=vmax))
ax[0][0].set_ylim(0, 30)
fig.colorbar(im, ax=ax[0][0])
ax[0][0].set_title('position 1 vs 2')
ax[0][0].set_ylabel('frequency (Hz)')

# position 1 vs 3
good_nogo01_inds = np.where(whisk.bids[9-1-1] == 4)[0]
good_nogo04_inds = np.where(whisk.bids[9-3-1] == 4)[0]
f, t, Sxx_mat_go   = whisk.get_spectrogram(whisk.wt[9-1-1][:, 0, good_nogo01_inds], 500)
f, t, Sxx_mat_nogo = whisk.get_spectrogram(whisk.wt[9-3-1][:, 0, good_nogo04_inds], 500)
mean_Sxx_go = np.mean(Sxx_mat_go, axis=2)
mean_Sxx_nogo = np.mean(Sxx_mat_nogo, axis=2)
diff_nogo_nogo = mean_Sxx_go - mean_Sxx_nogo

im = ax[0][1].pcolormesh(t-1, f, diff_nogo_nogo, cmap='coolwarm', vmin=-2.5, vmax=2.5)#, norm=colors.LogNorm(vmin=0.1, vmax=vmax))
ax[0][1].set_ylim(0, 30)
fig.colorbar(im, ax=ax[0][1])
ax[0][1].set_title('position 1 vs 3')
ax[0][1].set_ylabel('frequency (Hz)')

# position 1 vs 4
good_nogo01_inds = np.where(whisk.bids[9-1-1] == 4)[0]
good_nogo04_inds = np.where(whisk.bids[9-4-1] == 4)[0]
f, t, Sxx_mat_go   = whisk.get_spectrogram(whisk.wt[9-1-1][:, 0, good_nogo01_inds], 500)
f, t, Sxx_mat_nogo = whisk.get_spectrogram(whisk.wt[9-4-1][:, 0, good_nogo04_inds], 500)
mean_Sxx_go = np.mean(Sxx_mat_go, axis=2)
mean_Sxx_nogo = np.mean(Sxx_mat_nogo, axis=2)
diff_nogo_nogo = mean_Sxx_go - mean_Sxx_nogo

im = ax[0][2].pcolormesh(t-1, f, diff_nogo_nogo, cmap='coolwarm', vmin=-2.5, vmax=2.5)#, norm=colors.LogNorm(vmin=0.1, vmax=vmax))
ax[0][2].set_ylim(0, 30)
fig.colorbar(im, ax=ax[0][2])
ax[0][2].set_title('position 1 vs 4')
ax[0][2].set_ylabel('frequency (Hz)')

# position 2 vs 3
good_nogo01_inds = np.where(whisk.bids[9-2-1] == 4)[0]
good_nogo04_inds = np.where(whisk.bids[9-3-1] == 4)[0]
f, t, Sxx_mat_go   = whisk.get_spectrogram(whisk.wt[9-2-1][:, 0, good_nogo01_inds], 500)
f, t, Sxx_mat_nogo = whisk.get_spectrogram(whisk.wt[9-3-1][:, 0, good_nogo04_inds], 500)
mean_Sxx_go = np.mean(Sxx_mat_go, axis=2)
mean_Sxx_nogo = np.mean(Sxx_mat_nogo, axis=2)
diff_nogo_nogo = mean_Sxx_go - mean_Sxx_nogo

im = ax[1][1].pcolormesh(t-1, f, diff_nogo_nogo, cmap='coolwarm', vmin=-2.5, vmax=2.5)#, norm=colors.LogNorm(vmin=0.1, vmax=vmax))
ax[1][1].set_ylim(0, 30)
fig.colorbar(im, ax=ax[1][1])
ax[1][1].set_title('position 2 vs 3')
ax[1][1].set_ylabel('frequency (Hz)')

# position 2 vs 4
good_nogo01_inds = np.where(whisk.bids[9-2-1] == 4)[0]
good_nogo04_inds = np.where(whisk.bids[9-4-1] == 4)[0]
f, t, Sxx_mat_go   = whisk.get_spectrogram(whisk.wt[9-2-1][:, 0, good_nogo01_inds], 500)
f, t, Sxx_mat_nogo = whisk.get_spectrogram(whisk.wt[9-4-1][:, 0, good_nogo04_inds], 500)
mean_Sxx_go = np.mean(Sxx_mat_go, axis=2)
mean_Sxx_nogo = np.mean(Sxx_mat_nogo, axis=2)
diff_nogo_nogo = mean_Sxx_go - mean_Sxx_nogo

im = ax[1][2].pcolormesh(t-1, f, diff_nogo_nogo, cmap='coolwarm', vmin=-2.5, vmax=2.5)#, norm=colors.LogNorm(vmin=0.1, vmax=vmax))
ax[1][2].set_ylim(0, 30)
fig.colorbar(im, ax=ax[1][2])
ax[1][2].set_title('position 2 vs 4')
ax[1][2].set_ylabel('frequency (Hz)')

# position 3 vs 4
good_nogo01_inds = np.where(whisk.bids[9-3-1] == 4)[0]
good_nogo04_inds = np.where(whisk.bids[9-4-1] == 4)[0]
f, t, Sxx_mat_go   = whisk.get_spectrogram(whisk.wt[9-3-1][:, 0, good_nogo01_inds], 500)
f, t, Sxx_mat_nogo = whisk.get_spectrogram(whisk.wt[9-4-1][:, 0, good_nogo04_inds], 500)
mean_Sxx_go = np.mean(Sxx_mat_go, axis=2)
mean_Sxx_nogo = np.mean(Sxx_mat_nogo, axis=2)
diff_nogo_nogo = mean_Sxx_go - mean_Sxx_nogo

im = ax[2][2].pcolormesh(t-1, f, diff_nogo_nogo, cmap='coolwarm', vmin=-2.5, vmax=2.5)#, norm=colors.LogNorm(vmin=0.1, vmax=vmax))
ax[2][2].set_ylim(0, 30)
fig.colorbar(im, ax=ax[2][2])
ax[2][2].set_title('position 3 vs 4')
ax[2][2].set_ylabel('frequency (Hz)')



















