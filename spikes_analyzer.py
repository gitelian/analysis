import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.io as sio

# remove gride lines
sns.set_style("whitegrid", {'axes.grid' : False})
get_ipython().magic(u"run neoanalyzer.py {}".format(sys.argv[1]))

fid = sys.argv[1]
# create multipage PDF of unit summaries

with PdfPages(fid + '_unit_summaries.pdf') as pdf:
    for unit_index in range(neuro.num_units):

        # get best contact position from evoked rates
        meanr     = np.array([np.mean(k[:, unit_index]) for k in neuro.evk_rate])
        meanr_abs = np.array([np.mean(k[:, unit_index]) for k in neuro.abs_rate])
        best_contact = np.argmax(meanr[0:8])

        fig, ax = plt.subplots(4, 3, figsize=(10,8))
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.36, hspace=0.60)
        fig.suptitle('Region: {}, depth: {}, unit type: {}, mouse: {}, driven: {}'.format(\
                neuro.region_dict[neuro.shank_ids[unit_index]], \
                neuro.depths[unit_index], \
                neuro.cell_type[unit_index], \
                fid, \
                neuro.driven_units[unit_index]))

        # top left: best contact PSTH
        neuro.plot_psth(axis=ax[0][0], unit_ind=unit_index, trial_type=best_contact, error='sem', color='k')
        neuro.plot_psth(axis=ax[0][0], unit_ind=unit_index, trial_type=best_contact+9, error='sem', color='r')
        neuro.plot_psth(axis=ax[0][0], unit_ind=unit_index, trial_type=best_contact+9+9, error='sem', color='b')
        ax[0][0].set_xlim(-0.5, 2)
        #ax[0][0].set_ylim(0.5, ax[0][0].get_ylim()[1])
        ax[0][0].hlines(0, 0, 2, colors='k', linestyles='dashed')
        ax[0][0].vlines(0.5, ax[0][0].get_ylim()[0], ax[0][0].get_ylim()[1], colors='m', linestyles='dashed')
        ax[0][0].vlines(1.5, ax[0][0].get_ylim()[0], ax[0][0].get_ylim()[1], colors='m', linestyles='dashed')
        ax[0][0].set_xlabel('time (s)')
        ax[0][0].set_ylabel('firing rate (Hz)')
        #ax[0][0].set_yscale("log")
        ax[0][0].set_title('best contact')

        # top middle: control PSTH
        neuro.plot_psth(axis=ax[0][1], unit_ind=unit_index, trial_type=neuro.control_pos-1, error='sem', color='k')
        neuro.plot_psth(axis=ax[0][1], unit_ind=unit_index, trial_type=neuro.control_pos-1+9, error='sem', color='r')
        neuro.plot_psth(axis=ax[0][1], unit_ind=unit_index, trial_type=neuro.control_pos-1+9+9, error='sem', color='b')
        ax[0][1].set_xlim(-0.5, 2)
        #ax[0][1].set_ylim(0.5, ax[0][0].get_ylim()[1])
        ax[0][1].vlines(0.5, ax[0][1].get_ylim()[0], ax[0][1].get_ylim()[1], colors='m', linestyles='dashed')
        ax[0][1].vlines(1.5, ax[0][1].get_ylim()[0], ax[0][1].get_ylim()[1], colors='m', linestyles='dashed')
        ax[0][1].hlines(0, 0, 2, colors='k', linestyles='dashed')
        ax[0][1].set_xlabel('time (s)')
        #ax[0][1].set_yscale("log")
        ax[0][1].set_title('no contact')

        # top right: evoked tuning curves
        neuro.plot_tuning_curve(unit_ind=unit_index, kind='evk_count', axis=ax[0][2])
        ax[0][2].set_xlim(0, 10)
        ax[0][2].hlines(0, 0, 10, colors='k', linestyles='dashed')
        ax[0][2].set_xlabel('bar position')
        ax[0][2].set_title('evoked tc')

        # middle left: raster during no light and best contact
        neuro.plot_raster(unit_ind=unit_index, trial_type=best_contact, axis=ax[1][0])
        ax[1][0].set_xlim(-0.5, 2)
        ax[1][0].vlines(0.5, ax[0][0].get_ylim()[0], ax[0][0].get_ylim()[1], colors='m', linestyles='dashed')
        ax[1][0].vlines(1.5, ax[0][0].get_ylim()[0], ax[0][0].get_ylim()[1], colors='m', linestyles='dashed')
        ax[1][0].set_xlabel('time (s)')
        ax[1][0].set_ylabel('no light trials')

        # middle middle: raster during no light and control position
        neuro.plot_raster(unit_ind=unit_index, trial_type=neuro.control_pos-1, axis=ax[1][1])
        ax[1][1].set_xlim(-0.5, 2)
        ax[1][1].vlines(0.5, ax[0][0].get_ylim()[0], ax[0][0].get_ylim()[1], colors='m', linestyles='dashed')
        ax[1][1].vlines(1.5, ax[0][0].get_ylim()[0], ax[0][0].get_ylim()[1], colors='m', linestyles='dashed')
        ax[1][1].set_xlabel('time (s)')
        ax[1][1].set_ylabel('no light trials')

        # middle right: OMI tuning curves
        omi_s1light = (meanr_abs[neuro.control_pos:neuro.control_pos+9] - meanr_abs[:neuro.control_pos]) / \
                (meanr_abs[neuro.control_pos:neuro.control_pos+9] + meanr_abs[:neuro.control_pos])
        omi_m1light = (meanr_abs[neuro.control_pos+9:neuro.control_pos+9+9] - meanr_abs[:neuro.control_pos]) / \
        (meanr_abs[neuro.control_pos+9:neuro.control_pos+9+9] + meanr_abs[:neuro.control_pos])
        ax[1][2].plot(np.arange(1,10), omi_s1light, '-ro', np.arange(1,10), omi_m1light, '-bo')
        ax[1][2].hlines(0, 0, 10, colors='k', linestyles='dashed')
        ax[1][2].set_xlim(0, 10)
        ax[1][2].set_ylim(-1, 1)
        ax[1][2].set_xlabel('bar position')
        ax[1][2].set_ylabel('OMI')
        ax[1][2].set_title('OMI tc')

        # bottom left: raster for best contact and S1 light
        neuro.plot_raster(unit_ind=unit_index, trial_type=best_contact+9, axis=ax[2][0])
        ax[2][0].set_xlim(-0.5, 2)
        ax[2][0].vlines(0.5, ax[0][0].get_ylim()[0], ax[0][0].get_ylim()[1], colors='m', linestyles='dashed')
        ax[2][0].vlines(1.5, ax[0][0].get_ylim()[0], ax[0][0].get_ylim()[1], colors='m', linestyles='dashed')
        ax[2][0].set_xlabel('time (s)')
        ax[2][0].set_ylabel('S1 light trials')

        # bottom middle: bursty ISI plot control position
        neuro.plot_raster(unit_ind=unit_index, trial_type=neuro.control_pos-1+9, axis=ax[2][1])
        ax[2][1].set_xlim(-0.5, 2)
        ax[2][1].vlines(0.5, ax[0][0].get_ylim()[0], ax[0][0].get_ylim()[1], colors='m', linestyles='dashed')
        ax[2][1].vlines(1.5, ax[0][0].get_ylim()[0], ax[0][0].get_ylim()[1], colors='m', linestyles='dashed')
        ax[2][1].set_xlabel('time (s)')
        ax[2][1].set_ylabel('S1 light trials')

        #ax[1][0].hist2d(pre, post, bins=arange(0,0.3,0.001))

        # bottom right: mean waveform
        ax[2][2].plot(np.arange(neuro.waves[unit_index, :].shape[0]), neuro.waves[unit_index, :], 'k')
        ax[2][2].set_xlim(0, neuro.waves[unit_index, :].shape[0])
        ax[2][2].set_title('Mean waveform')
        ax[2][2].set_xlabel('dur: {}, ratio: {}'.format(\
                neuro.duration[unit_index],\
                neuro.ratio[unit_index]))

        # bottom bottom left: raster for best contact and S1 light
        neuro.plot_raster(unit_ind=unit_index, trial_type=best_contact+9+9, axis=ax[3][0])
        ax[3][0].set_xlim(-0.5, 2)
        ax[3][0].vlines(0.5, ax[0][0].get_ylim()[0], ax[0][0].get_ylim()[1], colors='m', linestyles='dashed')
        ax[3][0].vlines(1.5, ax[0][0].get_ylim()[0], ax[0][0].get_ylim()[1], colors='m', linestyles='dashed')
        ax[3][0].set_xlabel('time (s)')
        ax[3][0].set_ylabel('M1 light trials')

        # bottom bottom middle: bursty ISI plot control position
        neuro.plot_raster(unit_ind=unit_index, trial_type=neuro.control_pos-1+9+9, axis=ax[3][1])
        ax[3][1].set_xlim(-0.5, 2)
        ax[3][1].vlines(0.5, ax[0][0].get_ylim()[0], ax[0][0].get_ylim()[1], colors='m', linestyles='dashed')
        ax[3][1].vlines(1.5, ax[0][0].get_ylim()[0], ax[0][0].get_ylim()[1], colors='m', linestyles='dashed')
        ax[3][1].set_xlabel('time (s)')
        ax[3][1].set_ylabel('M1 light trials')

        pdf.savefig()
        fig.clear()
        plt.close()

##### population analysis #####
##### population analysis #####

# selectivity, center of mass, burstiness, OMI, decoder!!!
# do this for best position and no contact position. Plot things overall and
# then look at things as a function of depth.

fids = ['1295', '1302', '1318', '1328', '1329']
experiments = list()
for fid in fids:
    get_ipython().magic(u"run neoanalyzer.py {}".format(fid))
#    exp1 = block[0]
#    neuro = NeuroAnalyzer(exp1)
    # del neo objects to save memory
    del neuro.neo_obj
    del block
    del exp1
    del manager
    experiments.append(neuro)

# create arrays and lists for concatenating specified data from all experiments
region      = np.empty((1, ))
depths      = np.empty((1, ))
cell_type   = list()
driven      = np.empty((1, ))
omi         = np.empty((1, 2))
selectivity = np.empty((1, 3))
preference  = np.empty((1, 3))
best_pos    = np.empty((1, ))
abs_rate    = np.empty((1, 27, 2))
burst_rate  = np.empty((1, 27, 2))

for neuro in experiments:
    # calculate measures that weren't calculated at init
    neuro.get_best_contact()

    # concatenate measures
    cell_type.extend(neuro.cell_type)
    region      = np.append(region, neuro.shank_ids)
    depths      = np.append(depths, np.asarray(neuro.depths))
    selectivity = np.append(selectivity, neuro.selectivity, axis=0)
    driven      = np.append(driven, neuro.driven_units, axis=0)
    omi         = np.append(omi, neuro.get_omi(), axis=0)
    preference  = np.append(preference, neuro.preference, axis=0)
    best_pos    = np.append(best_pos, neuro.best_contact)

    for unit_index in range(neuro.num_units):

        # compute absolute rate (mean and sem)
        temp = np.zeros((1, 27, 2))
        temp[0, :, 0] = np.array([np.mean(k[:, unit_index]) for k in neuro.abs_rate])[:]
        temp[0, :, 1] = np.array([sp.stats.sem(k[:, unit_index]) for k in neuro.abs_rate])
        abs_rate = np.append(abs_rate, temp, axis=0)

        # compute burst rate for RS cells only (mean and sem)
        temp = np.zeros((1, 27, 2))
        if neuro.cell_type[unit_index] == 'RS':
            for trial in range(27):
                burst_rates = neuro.get_burst_rate(unit_ind=unit_index, trial_type=trial)
                temp[0, trial, 0] = np.mean(burst_rates)
                temp[0, trial, 1] = sp.stats.sem(burst_rates)
        burst_rate = np.append(burst_rate, temp, axis=0)


cell_type = np.asarray(cell_type)
region = region[1:,]
depths = depths[1:,]
driven = driven[1:,]
omi    = omi[1:,]
selectivity = selectivity[1:, :]
preference  = preference[1:, :]
best_pos    = best_pos[1:,]
abs_rate    = abs_rate[1:, :]
burst_rate  = burst_rate[1:, :]

##### select units #####
npand   = np.logical_and
#m1_inds = npand(npand(region==0, driven==True), cell_type=='MU')
#s1_inds = npand(npand(region==1, driven==True), cell_type=='MU')

##### save burst matrix #####
a = dict()
burst_path = '/Users/Greg/Documents/AdesnikLab/Data/burst.mat'
a['burst_rate'] = burst_rate
sio.savemat(burst_path, a)

###### Plot selectivity #####

## RS
m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')
bins = np.arange(0, 1, 0.05)
fig, ax = plt.subplots(3, 3, figsize=(16,9))
fig.suptitle('selectivity', fontsize=20)
ax[0][0].hist(selectivity[m1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[0][0].hist(selectivity[s1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[0][0].set_title('RS units M1: {} units, S1: {} units, no light'.format(sum(m1_inds), sum(s1_inds)))
ax[0][0].legend(['M1', 'S1'])
ax[1][0].hist(selectivity[m1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[1][0].hist(selectivity[m1_inds, 1], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[1][0].set_title('M1 units, S1 light')
ax[1][0].legend(['M1', 'S1 silencing'])
ax[2][0].hist(selectivity[s1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[2][0].hist(selectivity[s1_inds, 2], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[2][0].set_title('S1 units, M1 light')
ax[2][0].legend(['M1 silencing', 'S1'])

## FS
m1_inds = npand(npand(region==0, driven==True), cell_type=='FS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='FS')
ax[0][1].hist(selectivity[m1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[0][1].hist(selectivity[s1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[0][1].set_title('FS units M1: {} units, S1: {} units, no light'.format(sum(m1_inds), sum(s1_inds)))
ax[0][1].legend(['M1', 'S1'])
ax[1][1].hist(selectivity[m1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[1][1].hist(selectivity[m1_inds, 1], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[1][1].set_title('M1 units, S1 light')
ax[1][1].legend(['M1', 'S1 silencing'])
ax[2][1].hist(selectivity[s1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[2][1].hist(selectivity[s1_inds, 2], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[2][1].set_title('S1 units, M1 light')
ax[2][1].legend(['M1 silencing', 'S1'])

## MU
m1_inds = npand(npand(region==0, driven==True), cell_type=='MU')
s1_inds = npand(npand(region==1, driven==True), cell_type=='MU')
ax[0][2].hist(selectivity[m1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[0][2].hist(selectivity[s1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[0][2].set_title('MUA M1: {} units, S1: {} units, no light'.format(sum(m1_inds), sum(s1_inds)))
ax[0][2].legend(['M1', 'S1'])
ax[1][2].hist(selectivity[m1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[1][2].hist(selectivity[m1_inds, 1], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[1][2].set_title('M1, S1 light')
ax[1][2].legend(['M1', 'S1 silencing'])
ax[2][2].hist(selectivity[s1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[2][2].hist(selectivity[s1_inds, 2], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[2][2].set_title('S1, M1 light')
ax[2][2].legend(['M1 silencing', 'S1'])

## set ylim to the max ylim of all subplots
ylim_max = 0
for row in ax:
    for col in row:
        ylim_temp = col.get_ylim()[1]
        if ylim_temp > ylim_max:
            ylim_max = ylim_temp
for row in ax:
    for col in row:
        col.set_ylim(0, ylim_max)

###### Plot selectivity by depth
#fig, ax = plt.subplots(1, 1, figsize=(8,8))
#ax.plot(selectivity[m1_inds, 0], depths[m1_inds], 'ko')
#ax.plot(selectivity[s1_inds, 0], depths[s1_inds], 'ro')
#ax.set_ylim(0, 1100)


###### Plot preferred position #####

## RS
m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')
bins = np.arange(-1.0, 1.0, 0.10)
fig, ax = plt.subplots(3, 3, figsize=(16,9))
fig.suptitle('preferred position', fontsize=20)
ax[0][0].hist(preference[m1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[0][0].hist(preference[s1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[0][0].set_title('RS units M1: {} units, S1: {} units, no light'.format(sum(m1_inds), sum(s1_inds)))
ax[0][0].legend(['M1', 'S1'])
ax[1][0].hist(preference[m1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[1][0].hist(preference[m1_inds, 1], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[1][0].set_title('M1 units, S1 light')
ax[1][0].legend(['M1', 'S1 silencing'])
ax[2][0].hist(preference[s1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[2][0].hist(preference[s1_inds, 2], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[2][0].set_title('S1 units, M1 light')
ax[2][0].legend(['M1 silencing', 'S1'])

## FS
m1_inds = npand(npand(region==0, driven==True), cell_type=='FS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='FS')
ax[0][1].hist(preference[m1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[0][1].hist(preference[s1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[0][1].set_title('FS units M1: {} units, S1: {} units, no light'.format(sum(m1_inds), sum(s1_inds)))
ax[0][1].legend(['M1', 'S1'])
ax[1][1].hist(preference[m1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[1][1].hist(preference[m1_inds, 1], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[1][1].set_title('M1 units, S1 light')
ax[1][1].legend(['M1', 'S1 silencing'])
ax[2][1].hist(preference[s1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[2][1].hist(preference[s1_inds, 2], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[2][1].set_title('S1 units, M1 light')
ax[2][1].legend(['M1 silencing', 'S1'])

## MU
m1_inds = npand(npand(region==0, driven==True), cell_type=='MU')
s1_inds = npand(npand(region==1, driven==True), cell_type=='MU')
ax[0][2].hist(preference[m1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[0][2].hist(preference[s1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[0][2].set_title('MUA M1: {} units, S1: {} units, no light'.format(sum(m1_inds), sum(s1_inds)))
ax[0][2].legend(['M1', 'S1'])
ax[1][2].hist(preference[m1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[1][2].hist(preference[m1_inds, 1], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[1][2].set_title('M1, S1 light')
ax[1][2].legend(['M1', 'S1 silencing'])
ax[2][2].hist(preference[s1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[2][2].hist(preference[s1_inds, 2], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[2][2].set_title('S1, M1 light')
ax[2][2].legend(['M1 silencing', 'S1'])

## set ylim to the max ylim of all subplots
ylim_max = 0
for row in ax:
    for col in row:
        ylim_temp = col.get_ylim()[1]
        if ylim_temp > ylim_max:
            ylim_max = ylim_temp
for row in ax:
    for col in row:
        col.set_ylim(0, ylim_max)

## plot OMI

## RS
m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')
bins = np.arange(-1.0, 1.0, 0.10)
fig, ax = plt.subplots(2, 3, figsize=(16,9))
fig.suptitle('Mean OMI', fontsize=20)
ax[0][0].hist(omi[m1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[0][0].hist(omi[s1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[0][0].set_title('RS units M1: {} units, S1: {} units, \nS1 light'.format(sum(m1_inds), sum(s1_inds)))
ax[0][0].legend(['M1', 'S1'])
ax[1][0].hist(omi[m1_inds, 1], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[1][0].hist(omi[s1_inds, 1], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[1][0].set_title('M1 light')
ax[1][0].legend(['M1', 'S1'])

## FS
m1_inds = npand(npand(region==0, driven==True), cell_type=='FS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='FS')
ax[0][1].hist(omi[m1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[0][1].hist(omi[s1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[0][1].set_title('FS units M1: {} units, S1: {} units, \nS1 light'.format(sum(m1_inds), sum(s1_inds)))
ax[0][1].legend(['M1', 'S1'])
ax[1][1].hist(omi[m1_inds, 1], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[1][1].hist(omi[s1_inds, 1], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[1][1].set_title('M1 light')
ax[1][1].legend(['M1', 'S1'])

## MU
m1_inds = npand(npand(region==0, driven==True), cell_type=='MU')
s1_inds = npand(npand(region==1, driven==True), cell_type=='MU')
ax[0][2].hist(omi[m1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[0][2].hist(omi[s1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[0][2].set_title('MU units M1: {} units, S1: {} units, \nS1 light'.format(sum(m1_inds), sum(s1_inds)))
ax[0][2].legend(['M1', 'S1'])
ax[1][2].hist(omi[m1_inds, 1], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[1][2].hist(omi[s1_inds, 1], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[1][2].set_title('M1 light')
ax[1][2].legend(['M1', 'S1'])


## plot spontaneous rates

## RS
m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')
fig, ax = plt.subplots(2, 3, figsize=(16,9))
fig.suptitle('Spontaneous Rates', fontsize=20)
ax[0][0].errorbar(abs_rate[m1_inds, 8, 0], abs_rate[m1_inds, 8+9, 0], \
        xerr=abs_rate[m1_inds, 8, 1], yerr=abs_rate[m1_inds, 8+9, 1], c='k', fmt='o', ecolor='k')
ax[0][0].errorbar(abs_rate[s1_inds, 8, 0], abs_rate[s1_inds, 8+9, 0], \
        xerr=abs_rate[s1_inds, 8, 1], yerr=abs_rate[s1_inds, 8+9, 1], c='r', fmt='o', ecolor='r')
max_val = np.max([ax[0][0].get_xlim(), ax[0][0].get_ylim()])
ax[0][0].set_xlim(0, max_val)
ax[0][0].set_ylim(0, max_val)
ax[0][0].plot([0, max_val], [0, max_val], 'b')
ax[0][0].set_title('RS units M1: {} units, S1: {} units, \nS1 light'.format(sum(m1_inds), sum(s1_inds)))

ax[1][0].errorbar(abs_rate[m1_inds, 8, 0], abs_rate[m1_inds, 8+9+9, 0], \
        xerr=abs_rate[m1_inds, 8, 1], yerr=abs_rate[m1_inds, 8+9+9, 1], c='k', fmt='o', ecolor='k')
ax[1][0].errorbar(abs_rate[s1_inds, 8, 0], abs_rate[s1_inds, 8+9+9, 0], \
        xerr=abs_rate[s1_inds, 8, 1], yerr=abs_rate[s1_inds, 8+9+9, 1], c='r', fmt='o', ecolor='r')
max_val = np.max([ax[1][0].get_xlim(), ax[1][0].get_ylim()])
ax[1][0].set_xlim(0, max_val)
ax[1][0].set_ylim(0, max_val)
ax[1][0].plot([0, max_val], [0, max_val], 'b')
ax[1][0].set_title('M1 light'.format(sum(m1_inds), sum(s1_inds)))

## FS
m1_inds = npand(npand(region==0, driven==True), cell_type=='FS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='FS')
ax[0][1].errorbar(abs_rate[m1_inds, 8, 0], abs_rate[m1_inds, 8+9, 0], \
        xerr=abs_rate[m1_inds, 8, 1], yerr=abs_rate[m1_inds, 8+9, 1], c='k', fmt='o', ecolor='k')
ax[0][1].errorbar(abs_rate[s1_inds, 8, 0], abs_rate[s1_inds, 8+9, 0], \
        xerr=abs_rate[s1_inds, 8, 1], yerr=abs_rate[s1_inds, 8+9, 1], c='r', fmt='o', ecolor='r')
max_val = np.max([ax[0][1].get_xlim(), ax[0][1].get_ylim()])
ax[0][1].set_xlim(0, max_val)
ax[0][1].set_ylim(0, max_val)
ax[0][1].plot([0, max_val], [0, max_val], 'b')
ax[0][1].set_title('FS units M1: {} units, S1: {} units, \nS1 light'.format(sum(m1_inds), sum(s1_inds)))

ax[1][1].errorbar(abs_rate[m1_inds, 8, 0], abs_rate[m1_inds, 8+9+9, 0], \
        xerr=abs_rate[m1_inds, 8, 1], yerr=abs_rate[m1_inds, 8+9+9, 1], c='k', fmt='o', ecolor='k')
ax[1][1].errorbar(abs_rate[s1_inds, 8, 0], abs_rate[s1_inds, 8+9+9, 0], \
        xerr=abs_rate[s1_inds, 8, 1], yerr=abs_rate[s1_inds, 8+9+9, 1], c='r', fmt='o', ecolor='r')
max_val = np.max([ax[1][1].get_xlim(), ax[1][1].get_ylim()])
ax[1][1].set_xlim(0, max_val)
ax[1][1].set_ylim(0, max_val)
ax[1][1].plot([0, max_val], [0, max_val], 'b')
ax[1][1].set_title('M1 light'.format(sum(m1_inds), sum(s1_inds)))

## MU
m1_inds = npand(npand(region==0, driven==True), cell_type=='MU')
s1_inds = npand(npand(region==1, driven==True), cell_type=='MU')
ax[0][2].errorbar(abs_rate[m1_inds, 8, 0], abs_rate[m1_inds, 8+9, 0], \
        xerr=abs_rate[m1_inds, 8, 1], yerr=abs_rate[m1_inds, 8+9, 1], c='k', fmt='o', ecolor='k')
ax[0][2].errorbar(abs_rate[s1_inds, 8, 0], abs_rate[s1_inds, 8+9, 0], \
        xerr=abs_rate[s1_inds, 8, 1], yerr=abs_rate[s1_inds, 8+9, 1], c='r', fmt='o', ecolor='r')
max_val = np.max([ax[0][1].get_xlim(), ax[0][1].get_ylim()])
ax[0][2].set_xlim(0, max_val)
ax[0][2].set_ylim(0, max_val)
ax[0][2].plot([0, max_val], [0, max_val], 'b')
ax[0][2].set_title('MU units M1: {} units, S1: {} units, \nS1 light'.format(sum(m1_inds), sum(s1_inds)))

ax[1][2].errorbar(abs_rate[m1_inds, 8, 0], abs_rate[m1_inds, 8+9+9, 0], \
        xerr=abs_rate[m1_inds, 8, 1], yerr=abs_rate[m1_inds, 8+9+9, 1], c='k', fmt='o', ecolor='k')
ax[1][2].errorbar(abs_rate[s1_inds, 8, 0], abs_rate[s1_inds, 8+9+9, 0], \
        xerr=abs_rate[s1_inds, 8, 1], yerr=abs_rate[s1_inds, 8+9+9, 1], c='r', fmt='o', ecolor='r')
max_val = np.max([ax[1][1].get_xlim(), ax[1][1].get_ylim()])
ax[1][2].set_xlim(0, max_val)
ax[1][2].set_ylim(0, max_val)
ax[1][2].plot([0, max_val], [0, max_val], 'b')
ax[1][2].set_title('M1 light'.format(sum(m1_inds), sum(s1_inds)))








## plot driven rates best position

## RS
m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')
m1_best_pos = best_pos[m1_inds].astype(int)
s1_best_pos = best_pos[s1_inds].astype(int)
fig, ax = plt.subplots(2, 3, figsize=(16,9))
fig.suptitle('Driven Rates', fontsize=20)
ax[0][0].errorbar(abs_rate[m1_inds, m1_best_pos, 0], abs_rate[m1_inds, m1_best_pos+9, 0], \
        xerr=abs_rate[m1_inds, m1_best_pos, 1], yerr=abs_rate[m1_inds, m1_best_pos+9, 1], c='k', fmt='o', ecolor='k')
ax[0][0].errorbar(abs_rate[s1_inds, s1_best_pos, 0], abs_rate[s1_inds, s1_best_pos+9, 0], \
        xerr=abs_rate[s1_inds, s1_best_pos, 1], yerr=abs_rate[s1_inds, s1_best_pos+9, 1], c='r', fmt='o', ecolor='r')
max_val = np.max([ax[0][0].get_xlim(), ax[0][0].get_ylim()])
ax[0][0].set_xlim(0, max_val)
ax[0][0].set_ylim(0, max_val)
ax[0][0].plot([0, max_val], [0, max_val], 'b')
ax[0][0].set_title('RS units M1: {} units, S1: {} units, \nS1 light'.format(sum(m1_inds), sum(s1_inds)))

ax[1][0].errorbar(abs_rate[m1_inds, m1_best_pos, 0], abs_rate[m1_inds, m1_best_pos+9+9, 0], \
        xerr=abs_rate[m1_inds, m1_best_pos, 1], yerr=abs_rate[m1_inds, m1_best_pos+9+9, 1], c='k', fmt='o', ecolor='k')
ax[1][0].errorbar(abs_rate[s1_inds, s1_best_pos, 0], abs_rate[s1_inds, s1_best_pos+9+9, 0], \
        xerr=abs_rate[s1_inds, s1_best_pos, 1], yerr=abs_rate[s1_inds, s1_best_pos+9+9, 1], c='r', fmt='o', ecolor='r')
max_val = np.max([ax[1][0].get_xlim(), ax[1][0].get_ylim()])
ax[1][0].set_xlim(0, max_val)
ax[1][0].set_ylim(0, max_val)
ax[1][0].plot([0, max_val], [0, max_val], 'b')
ax[1][0].set_title('M1 light'.format(sum(m1_inds), sum(s1_inds)))

## FS
m1_inds = npand(npand(region==0, driven==True), cell_type=='FS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='FS')
m1_best_pos = best_pos[m1_inds].astype(int)
s1_best_pos = best_pos[s1_inds].astype(int)
ax[0][1].errorbar(abs_rate[m1_inds, m1_best_pos, 0], abs_rate[m1_inds, m1_best_pos+9, 0], \
        xerr=abs_rate[m1_inds, m1_best_pos, 1], yerr=abs_rate[m1_inds, m1_best_pos+9, 1], c='k', fmt='o', ecolor='k')
ax[0][1].errorbar(abs_rate[s1_inds, s1_best_pos, 0], abs_rate[s1_inds, s1_best_pos+9, 0], \
        xerr=abs_rate[s1_inds, s1_best_pos, 1], yerr=abs_rate[s1_inds, s1_best_pos+9, 1], c='r', fmt='o', ecolor='r')
max_val = np.max([ax[0][1].get_xlim(), ax[0][1].get_ylim()])
ax[0][1].set_xlim(0, max_val)
ax[0][1].set_ylim(0, max_val)
ax[0][1].plot([0, max_val], [0, max_val], 'b')
ax[0][1].set_title('FS units M1: {} units, S1: {} units, \nS1 light'.format(sum(m1_inds), sum(s1_inds)))

ax[1][1].errorbar(abs_rate[m1_inds, m1_best_pos, 0], abs_rate[m1_inds, m1_best_pos+9+9, 0], \
        xerr=abs_rate[m1_inds, m1_best_pos, 1], yerr=abs_rate[m1_inds, m1_best_pos+9+9, 1], c='k', fmt='o', ecolor='k')
ax[1][1].errorbar(abs_rate[s1_inds, s1_best_pos, 0], abs_rate[s1_inds, s1_best_pos+9+9, 0], \
        xerr=abs_rate[s1_inds, s1_best_pos, 1], yerr=abs_rate[s1_inds, s1_best_pos+9+9, 1], c='r', fmt='o', ecolor='r')
max_val = np.max([ax[1][1].get_xlim(), ax[1][1].get_ylim()])
ax[1][1].set_xlim(0, max_val)
ax[1][1].set_ylim(0, max_val)
ax[1][1].plot([0, max_val], [0, max_val], 'b')
ax[1][1].set_title('M1 light'.format(sum(m1_inds), sum(s1_inds)))

## MU
m1_inds = npand(npand(region==0, driven==True), cell_type=='MU')
s1_inds = npand(npand(region==1, driven==True), cell_type=='MU')
m1_best_pos = best_pos[m1_inds].astype(int)
s1_best_pos = best_pos[s1_inds].astype(int)
ax[0][2].errorbar(abs_rate[m1_inds, m1_best_pos, 0], abs_rate[m1_inds, m1_best_pos+9, 0], \
        xerr=abs_rate[m1_inds, m1_best_pos, 1], yerr=abs_rate[m1_inds, m1_best_pos+9, 1], c='k', fmt='o', ecolor='k')
ax[0][2].errorbar(abs_rate[s1_inds, s1_best_pos, 0], abs_rate[s1_inds, s1_best_pos+9, 0], \
        xerr=abs_rate[s1_inds, s1_best_pos, 1], yerr=abs_rate[s1_inds, s1_best_pos+9, 1], c='r', fmt='o', ecolor='r')
max_val = np.max([ax[0][2].get_xlim(), ax[0][2].get_ylim()])
ax[0][2].set_xlim(0, max_val)
ax[0][2].set_ylim(0, max_val)
ax[0][2].plot([0, max_val], [0, max_val], 'b')
ax[0][2].set_title('MU units M1: {} units, S1: {} units, \nS1 light'.format(sum(m1_inds), sum(s1_inds)))

ax[1][2].errorbar(abs_rate[m1_inds, m1_best_pos, 0], abs_rate[m1_inds, m1_best_pos+9+9, 0], \
        xerr=abs_rate[m1_inds, m1_best_pos, 1], yerr=abs_rate[m1_inds, m1_best_pos+9+9, 1], c='k', fmt='o', ecolor='k')
ax[1][2].errorbar(abs_rate[s1_inds, s1_best_pos, 0], abs_rate[s1_inds, s1_best_pos+9+9, 0], \
        xerr=abs_rate[s1_inds, s1_best_pos, 1], yerr=abs_rate[s1_inds, s1_best_pos+9+9, 1], c='r', fmt='o', ecolor='r')
max_val = np.max([ax[1][2].get_xlim(), ax[1][2].get_ylim()])
ax[1][2].set_xlim(0, max_val)
ax[1][2].set_ylim(0, max_val)
ax[1][2].plot([0, max_val], [0, max_val], 'b')
ax[1][2].set_title('M1 light'.format(sum(m1_inds), sum(s1_inds)))

## plot burst rates

m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')
m1_best_pos = best_pos[m1_inds].astype(int)
s1_best_pos = best_pos[s1_inds].astype(int)
bins = np.arange(0, 100, 2)
fig, ax = plt.subplots(3, 2, figsize=(16,9))

## RS no light best position
ax[0][0].hist(burst_rate[m1_inds, m1_best_pos, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[0][0].hist(burst_rate[s1_inds, s1_best_pos, 0], bins=bins, edgecolor='None', alpha=0.5, color='r')

## RS S1 light best position
ax[1][0].hist(burst_rate[m1_inds, m1_best_pos + 9, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[1][0].hist(burst_rate[s1_inds, s1_best_pos + 9, 0], bins=bins, edgecolor='None', alpha=0.5, color='r')

## RS M1 light best position
ax[2][0].hist(burst_rate[m1_inds, m1_best_pos +9+9, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[2][0].hist(burst_rate[s1_inds, s1_best_pos +9+9, 0], bins=bins, edgecolor='None', alpha=0.5, color='r')



fig, ax = plt.subplots(2,2)
# m1 bursts S1 light
# paired plot
ax[0][0].scatter(np.zeros(sum(m1_inds)), burst_rate[m1_inds, m1_best_pos, 0], c='k')
ax[0][0].scatter(np.ones(sum(m1_inds)), burst_rate[m1_inds, m1_best_pos+9, 0], c='k')
# plotting the lines
for i in range(sum(m1_inds)):
        ax[0][0].plot( [0,1], [burst_rate[m1_inds, m1_best_pos, 0][i], burst_rate[m1_inds, m1_best_pos+9, 0][i]], 'k')
        ax[0][0].set_xticks([0,1], ['before', 'after'])

# plots histograms
#ax[0][0].hist(burst_rate[m1_inds, m1_best_pos, 0], bins=np.arange(0, 20, 1), alpha=0.5)
#ax[0][0].hist(burst_rate[m1_inds, m1_best_pos+9, 0], bins=np.arange(0, 20, 1), alpha=0.5)
stat, pval = sp.stats.wilcoxon(burst_rate[m1_inds, m1_best_pos, 0], burst_rate[m1_inds, m1_best_pos+9, 0])
ax[0][0].set_title('wilcoxon signed rank test p-val: {0:.5f}'.format(pval))
ax[0][0].set_xlim(-1.5, 2.5)

# s1 bursts M1 light
# paired plot
ax[1][0].scatter(np.zeros(sum(m1_inds)), burst_rate[m1_inds, m1_best_pos, 0], c='r')
ax[1][0].scatter(np.ones(sum(m1_inds)), burst_rate[m1_inds, m1_best_pos+9, 0], c='r')
# plotting the lines
for i in range(sum(m1_inds)):
        ax[1][0].plot( [0,1], [burst_rate[m1_inds, m1_best_pos, 0][i], burst_rate[m1_inds, m1_best_pos+9, 0][i]], 'r')
        ax[1][0].set_xticks([0,1], ['before', 'after'])

#ax[1][0].hist(burst_rate[s1_inds, s1_best_pos, 0], bins=np.arange(0, 20, 1), alpha=0.5)
#ax[1][0].hist(burst_rate[s1_inds, s1_best_pos+9+9, 0], bins=np.arange(0, 20, 1), alpha=0.5)
stat, pval = sp.stats.wilcoxon(burst_rate[s1_inds, s1_best_pos, 0], burst_rate[s1_inds, s1_best_pos+9+9, 0])
ax[1][0].set_title('wilcoxon signed rank test: {0:.5f}'.format(pval))
ax[1][0].set_xlim(-1.5, 2.5)

# m1 burst difference distribution S1 light
m1_diff = burst_rate[m1_inds, m1_best_pos+9, 0] - burst_rate[m1_inds, m1_best_pos, 0]
ax[0][1].hist(m1_diff, bins=np.arange(-10, 10, 2), alpha=0.5)
ax[0][1].set_xlim(-10, 10)

# s1 burst difference distribution S1 light
s1_diff = burst_rate[s1_inds, s1_best_pos+9+9, 0] - burst_rate[s1_inds, s1_best_pos, 0]
ax[1][1].hist(s1_diff, bins=np.arange(-10, 10, 2), alpha=0.5)
ax[1][1].set_xlim(-10, 10)








































