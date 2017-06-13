import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.io as sio

# remove gride lines
sns.set_style("whitegrid", {'axes.grid' : False})
#get_ipython().magic(u"run neoanalyzer.py {}".format(sys.argv[1]))

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
        ## if opto:
        neuro.plot_psth(axis=ax[0][0], unit_ind=unit_index, trial_type=best_contact+9, error='sem', color='r')
        neuro.plot_psth(axis=ax[0][0], unit_ind=unit_index, trial_type=best_contact+9+9, error='sem', color='b')
        ax[0][0].set_xlim(-0.5, 2)
        #ax[0][0].set_ylim(0.5, ax[0][0].get_ylim()[1])
        ax[0][0].hlines(0,-0.5, 2, colors='k', linestyles='dashed')
        # stimulus onset lines
        ax[0][0].vlines(0.0, ax[0][0].get_ylim()[0], ax[0][0].get_ylim()[1], colors='r', linewidth=1)
        ax[0][0].vlines(1.51, ax[0][0].get_ylim()[0], ax[0][0].get_ylim()[1], colors='r', linewidth=1)
        # light onset lines
        ax[0][0].vlines(0.5, ax[0][0].get_ylim()[0], ax[0][0].get_ylim()[1], colors='m', linestyles='dashed', linewidth=1)
        ax[0][0].vlines(1.5, ax[0][0].get_ylim()[0], ax[0][0].get_ylim()[1], colors='m', linestyles='dashed', linewidth=1)
        ax[0][0].set_xlabel('time (s)')
        ax[0][0].set_ylabel('firing rate (Hz)')
        ax[0][0].set_title('best contact')

        # top middle: control PSTH
        neuro.plot_psth(axis=ax[0][1], unit_ind=unit_index, trial_type=neuro.control_pos-1, error='sem', color='k')
#        neuro.plot_psth(axis=ax[0][1], unit_ind=unit_index, trial_type=neuro.control_pos-1+9, error='sem', color='r')
#        neuro.plot_psth(axis=ax[0][1], unit_ind=unit_index, trial_type=neuro.control_pos-1+9+9, error='sem', color='b')
        ax[0][1].set_xlim(-0.5, 2)
        #ax[0][1].set_ylim(0.5, ax[0][0].get_ylim()[1])
        # stimulus onset lines
        ax[0][1].vlines(0.00, ax[0][1].get_ylim()[0], ax[0][1].get_ylim()[1], colors='r', linewidth=1)
        ax[0][1].vlines(1.51, ax[0][1].get_ylim()[0], ax[0][1].get_ylim()[1], colors='r', linewidth=1)
        # light onset lines
        ax[0][1].vlines(0.5, ax[0][1].get_ylim()[0], ax[0][1].get_ylim()[1], colors='m', linestyles='dashed', linewidth=1)
        ax[0][1].vlines(1.5, ax[0][1].get_ylim()[0], ax[0][1].get_ylim()[1], colors='m', linestyles='dashed', linewidth=1)
        ax[0][1].hlines(0,-0.5, 2, colors='k', linestyles='dashed')
        ax[0][1].set_xlabel('time (s)')
        #ax[0][1].set_yscale("log")
        ax[0][1].set_title('no contact')

        # top right: evoked tuning curves
        #neuro.plot_tuning_curve(unit_ind=unit_index, kind='evk_count', axis=ax[0][2])
        neuro.plot_tuning_curve(unit_ind=unit_index, kind='abs_count', axis=ax[0][2])
        ax[0][2].set_xlim(0, 10)
        ax[0][2].hlines(0, 0, 10, colors='k', linestyles='dashed')
        ax[0][2].set_xlabel('bar position')
        #ax[0][2].set_title('evoked tc')
        ax[0][2].set_title('absolute tc')

        # middle left: raster during no light and best contact
        neuro.plot_raster(unit_ind=unit_index, trial_type=best_contact, axis=ax[1][0], burst=False)
        ax[1][0].set_xlim(-0.5, 2)
        # stimulus onset lines
        ax[1][0].vlines(0.00, ax[1][0].get_ylim()[0], ax[1][0].get_ylim()[1], colors='r', linewidth=1)
        ax[1][0].vlines(1.51, ax[1][0].get_ylim()[0], ax[1][0].get_ylim()[1], colors='r', linewidth=1)
        # light onset lines
        ax[1][0].vlines(0.5, ax[1][0].get_ylim()[0], ax[1][0].get_ylim()[1], colors='m', linestyles='dashed', linewidth=1)
        ax[1][0].vlines(1.5, ax[1][0].get_ylim()[0], ax[1][0].get_ylim()[1], colors='m', linestyles='dashed', linewidth=1)
        ax[1][0].set_xlabel('time (s)')
        ax[1][0].set_ylabel('no light trials')

        # middle middle: raster during no light and control position
        neuro.plot_raster(unit_ind=unit_index, trial_type=neuro.control_pos-1, axis=ax[1][1], burst=False)
        ax[1][1].set_xlim(-0.5, 2)
        # stimulus onset lines
        ax[1][1].vlines(0.00, ax[1][1].get_ylim()[0], ax[1][1].get_ylim()[1], colors='r', linewidth=1)
        ax[1][1].vlines(1.51, ax[1][1].get_ylim()[0], ax[1][1].get_ylim()[1], colors='r', linewidth=1)
        # light onset lines
        ax[1][1].vlines(0.5, ax[1][1].get_ylim()[0], ax[1][1].get_ylim()[1], colors='m', linestyles='dashed', linewidth=1)
        ax[1][1].vlines(1.5, ax[1][1].get_ylim()[0], ax[1][1].get_ylim()[1], colors='m', linestyles='dashed', linewidth=1)
        ax[1][1].set_xlabel('time (s)')
        ax[1][1].set_ylabel('no light trials')

        # middle right: OMI tuning curves
       ## if opto:
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
        neuro.plot_raster(unit_ind=unit_index, trial_type=best_contact+9, axis=ax[2][0], burst=False)
        ax[2][0].set_xlim(-0.5, 2)
        # stimulus onset lines
        ax[2][0].vlines(0.00, ax[2][0].get_ylim()[0], ax[2][0].get_ylim()[1], colors='r', linewidth=1)
        ax[2][0].vlines(1.51, ax[2][0].get_ylim()[0], ax[2][0].get_ylim()[1], colors='r', linewidth=1)
        # light onset lines
        ax[2][0].vlines(0.5, ax[2][0].get_ylim()[0], ax[2][0].get_ylim()[1], colors='m', linestyles='dashed', linewidth=1)
        ax[2][0].vlines(1.5, ax[2][0].get_ylim()[0], ax[2][0].get_ylim()[1], colors='m', linestyles='dashed', linewidth=1)
        ax[2][0].set_xlabel('time (s)')
        ax[2][0].set_ylabel('S1 light trials')

        # bottom middle: bursty ISI plot control position and S1 light
        neuro.plot_raster(unit_ind=unit_index, trial_type=neuro.control_pos-1+9, axis=ax[2][1], burst=False)
        ax[2][1].set_xlim(-0.5, 2)
        # stimulus onset lines
        ax[2][1].vlines(0.00, ax[2][1].get_ylim()[0], ax[2][1].get_ylim()[1], colors='r', linewidth=1)
        ax[2][1].vlines(1.51, ax[2][1].get_ylim()[0], ax[2][1].get_ylim()[1], colors='r', linewidth=1)
        # light onset lines
        ax[2][1].vlines(0.5, ax[2][1].get_ylim()[0], ax[2][1].get_ylim()[1], colors='m', linestyles='dashed', linewidth=1)
        ax[2][1].vlines(1.5, ax[2][1].get_ylim()[0], ax[2][1].get_ylim()[1], colors='m', linestyles='dashed', linewidth=1)
        ax[2][1].set_xlabel('time (s)')
        ax[2][1].set_ylabel('S1 light trials')

        #ax[1][0].hist2d(pre, post, bins=arange(0,0.3,0.001))
       ## END if opto:

        # bottom right: mean waveform
        ax[2][2].plot(np.arange(neuro.waves[unit_index, :].shape[0]), neuro.waves[unit_index, :], 'k')
        ax[2][2].set_xlim(0, neuro.waves[unit_index, :].shape[0])
        ax[2][2].set_title('Mean waveform')
        ax[2][2].set_xlabel('dur: {}, ratio: {}'.format(\
                neuro.duration[unit_index],\
                neuro.ratio[unit_index]))

        # bottom bottom left: raster for best contact and S1 light
       ## if opto:
        neuro.plot_raster(unit_ind=unit_index, trial_type=best_contact+9+9, axis=ax[3][0], burst=False)
        ax[3][0].set_xlim(-0.5, 2)
        # stimulus onset lines
        ax[3][0].vlines(0.00, ax[3][0].get_ylim()[0], ax[3][0].get_ylim()[1], colors='r', linewidth=1)
        ax[3][0].vlines(1.51, ax[3][0].get_ylim()[0], ax[3][0].get_ylim()[1], colors='r', linewidth=1)
        # light onset lines
        ax[3][0].vlines(0.5, ax[3][0].get_ylim()[0], ax[3][0].get_ylim()[1], colors='m', linestyles='dashed', linewidth=1)
        ax[3][0].vlines(1.5, ax[3][0].get_ylim()[0], ax[3][0].get_ylim()[1], colors='m', linestyles='dashed', linewidth=1)
        ax[3][0].set_xlabel('time (s)')
        ax[3][0].set_ylabel('M1 light trials')

        # bottom bottom middle: bursty ISI plot control position
        neuro.plot_raster(unit_ind=unit_index, trial_type=neuro.control_pos-1+9+9, axis=ax[3][1], burst=False)
        ax[3][1].set_xlim(-0.5, 2)
        # stimulus onset lines
        ax[3][1].vlines(0.00, ax[3][1].get_ylim()[0], ax[3][1].get_ylim()[1], colors='r', linewidth=1)
        ax[3][1].vlines(1.51, ax[3][1].get_ylim()[0], ax[3][1].get_ylim()[1], colors='r', linewidth=1)
        # light onset lines
        ax[3][1].vlines(0.5, ax[3][1].get_ylim()[0], ax[3][1].get_ylim()[1], colors='m', linestyles='dashed', linewidth=1)
        ax[3][1].vlines(1.5, ax[3][1].get_ylim()[0], ax[3][1].get_ylim()[1], colors='m', linestyles='dashed', linewidth=1)
        ax[3][1].set_xlabel('time (s)')
        ax[3][1].set_ylabel('M1 light trials')
        ## END if opto:

        pdf.savefig()
        fig.clear()
        plt.close()

##### population analysis #####
##### population analysis #####

# selectivity, center of mass, burstiness, OMI, decoder!!!
# do this for best position and no contact position. Plot things overall and
# then look at things as a function of depth.

fids = ['1295', '1302', '1318', '1328', '1329', '1330']
#fids = ['1302', '1318', '1330']
#fids = ['1330']
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
evk_rate    = np.empty((1, 27, 2))
max_fr      = np.empty((1, ))
burst_rate  = np.empty((1, 27, 2))
adapt_ratio = np.empty((1, 27, 2))

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
        max_fr   = np.append(max_fr, np.max(temp))

        # compute absolute rate (mean and sem)
        temp = np.zeros((1, 27, 2))
        temp[0, :, 0] = np.array([np.mean(k[:, unit_index]) for k in neuro.evk_rate])[:]
        temp[0, :, 1] = np.array([sp.stats.sem(k[:, unit_index]) for k in neuro.evk_rate])
        evk_rate = np.append(evk_rate, temp, axis=0)

#        # compute burst rate for RS cells only (mean and sem)
#        temp = np.zeros((1, 27, 2))
#        if neuro.cell_type[unit_index] == 'RS':
#            for trial in range(27):
#                burst_rates = neuro.get_burst_rate(unit_ind=unit_index, trial_type=trial)
#                temp[0, trial, 0] = np.mean(burst_rates)
#                temp[0, trial, 1] = sp.stats.sem(burst_rates)
#        burst_rate = np.append(burst_rate, temp, axis=0)

        # compute adaptation ratio
        adapt_ratio_temp = neuro.get_adaptation_ratio(unit_ind=unit_index)
        adapt_ratio = np.append(adapt_ratio, adapt_ratio_temp, axis=0)


cell_type = np.asarray(cell_type)
region = region[1:,]
region = region.astype(int)

depths = depths[1:,]
driven = driven[1:,]
driven = driven.astype(int)

omi    = omi[1:,]
omi    = np.nan_to_num(omi)
selectivity = selectivity[1:, :]
preference  = preference[1:, :]
best_pos    = best_pos[1:,]
best_pos    = best_pos.astype(int)
abs_rate    = abs_rate[1:, :]
evk_rate    = evk_rate[1:, :]
max_fr      = max_fr[1:,]
burst_rate  = burst_rate[1:, :]
adapt_ratio = adapt_ratio[1:, :]

##### select units #####
npand   = np.logical_and
#m1_inds = npand(npand(region==0, driven==True), cell_type=='MU')
#s1_inds = npand(npand(region==1, driven==True), cell_type=='MU')

##### loadt burst matrix #####
burst_path = '/Users/Greg/Documents/AdesnikLab/Data/burst.mat'
burst_rate = sio.loadmat(burst_path)['burst_rate']

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
ax[0][0].set_title('RS units M1: {} units, S1: {} units\nno light'.format(sum(m1_inds), sum(s1_inds)))
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
ax[0][1].set_title('FS units M1: {} units, S1: {} units\nno light'.format(sum(m1_inds), sum(s1_inds)))
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
ax[0][2].set_title('MUA M1: {} units, S1: {} units\nno light'.format(sum(m1_inds), sum(s1_inds)))
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
ax[0][0].set_title('RS units M1: {} units, S1: {} units\nno light'.format(sum(m1_inds), sum(s1_inds)))
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
ax[0][1].set_title('FS units M1: {} units, S1: {} units\nno light'.format(sum(m1_inds), sum(s1_inds)))
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
ax[0][2].set_title('MUA M1: {} units, S1: {} units\nno light'.format(sum(m1_inds), sum(s1_inds)))
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



##### plot OMI #####
##### plot OMI #####

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


##### plot spontaneous rates #####
##### plot spontaneous rates #####

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
ax[0][0].set_ylabel('Light On\nfiring rate (Hz)')

ax[1][0].errorbar(abs_rate[m1_inds, 8, 0], abs_rate[m1_inds, 8+9+9, 0], \
        xerr=abs_rate[m1_inds, 8, 1], yerr=abs_rate[m1_inds, 8+9+9, 1], c='k', fmt='o', ecolor='k')
ax[1][0].errorbar(abs_rate[s1_inds, 8, 0], abs_rate[s1_inds, 8+9+9, 0], \
        xerr=abs_rate[s1_inds, 8, 1], yerr=abs_rate[s1_inds, 8+9+9, 1], c='r', fmt='o', ecolor='r')
max_val = np.max([ax[1][0].get_xlim(), ax[1][0].get_ylim()])
ax[1][0].set_xlim(0, max_val)
ax[1][0].set_ylim(0, max_val)
ax[1][0].plot([0, max_val], [0, max_val], 'b')
ax[1][0].set_title('M1 light'.format(sum(m1_inds), sum(s1_inds)))
ax[1][0].set_xlabel('Light On\nfiring rate (Hz)')
ax[1][0].set_ylabel('Light Off\nfiring rate (Hz)')

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
ax[1][1].set_xlabel('Light Off\nfiring rate (Hz)')

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
ax[1][2].set_xlabel('Light Off\nfiring rate (Hz)')



##### plot driven rates best position #####
##### plot driven rates best position #####

## RS top left
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
ax[0][0].set_ylabel('Light On\nfiring rate (Hz)')

## RS bottom left
ax[1][0].errorbar(abs_rate[m1_inds, m1_best_pos, 0], abs_rate[m1_inds, m1_best_pos+9+9, 0], \
        xerr=abs_rate[m1_inds, m1_best_pos, 1], yerr=abs_rate[m1_inds, m1_best_pos+9+9, 1], c='k', fmt='o', ecolor='k')
ax[1][0].errorbar(abs_rate[s1_inds, s1_best_pos, 0], abs_rate[s1_inds, s1_best_pos+9+9, 0], \
        xerr=abs_rate[s1_inds, s1_best_pos, 1], yerr=abs_rate[s1_inds, s1_best_pos+9+9, 1], c='r', fmt='o', ecolor='r')
max_val = np.max([ax[1][0].get_xlim(), ax[1][0].get_ylim()])
ax[1][0].set_xlim(0, max_val)
ax[1][0].set_ylim(0, max_val)
ax[1][0].plot([0, max_val], [0, max_val], 'b')
ax[1][0].set_title('M1 light'.format(sum(m1_inds), sum(s1_inds)))
ax[1][0].set_ylabel('Light On\nfiring rate (Hz)')
ax[1][0].set_xlabel('Light Off\nfiring rate (Hz)')

## FS top middle
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

## FS bottom middle
ax[1][1].errorbar(abs_rate[m1_inds, m1_best_pos, 0], abs_rate[m1_inds, m1_best_pos+9+9, 0], \
        xerr=abs_rate[m1_inds, m1_best_pos, 1], yerr=abs_rate[m1_inds, m1_best_pos+9+9, 1], c='k', fmt='o', ecolor='k')
ax[1][1].errorbar(abs_rate[s1_inds, s1_best_pos, 0], abs_rate[s1_inds, s1_best_pos+9+9, 0], \
        xerr=abs_rate[s1_inds, s1_best_pos, 1], yerr=abs_rate[s1_inds, s1_best_pos+9+9, 1], c='r', fmt='o', ecolor='r')
max_val = np.max([ax[1][1].get_xlim(), ax[1][1].get_ylim()])
ax[1][1].set_xlim(0, max_val)
ax[1][1].set_ylim(0, max_val)
ax[1][1].plot([0, max_val], [0, max_val], 'b')
ax[1][1].set_title('M1 light'.format(sum(m1_inds), sum(s1_inds)))
ax[1][1].set_xlabel('Light Off\nfiring rate (Hz)')

## MU top right
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
ax[0][2].set_xlabel('Light Off\nfiring rate (Hz)')

## MU bottom right
ax[1][2].errorbar(abs_rate[m1_inds, m1_best_pos, 0], abs_rate[m1_inds, m1_best_pos+9+9, 0], \
        xerr=abs_rate[m1_inds, m1_best_pos, 1], yerr=abs_rate[m1_inds, m1_best_pos+9+9, 1], c='k', fmt='o', ecolor='k')
ax[1][2].errorbar(abs_rate[s1_inds, s1_best_pos, 0], abs_rate[s1_inds, s1_best_pos+9+9, 0], \
        xerr=abs_rate[s1_inds, s1_best_pos, 1], yerr=abs_rate[s1_inds, s1_best_pos+9+9, 1], c='r', fmt='o', ecolor='r')
max_val = np.max([ax[1][2].get_xlim(), ax[1][2].get_ylim()])
ax[1][2].set_xlim(0, max_val)
ax[1][2].set_ylim(0, max_val)
ax[1][2].plot([0, max_val], [0, max_val], 'b')
ax[1][2].set_title('M1 light'.format(sum(m1_inds), sum(s1_inds)))
ax[1][2].set_xlabel('Light Off\nfiring rate (Hz)')



##### plot burst rates for RS units #####
##### plot burst rates for RS units #####

m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')
m1_best_pos = best_pos[m1_inds].astype(int)
s1_best_pos = best_pos[s1_inds].astype(int)
y_max = 17
bins = np.arange(0, 100, 5)
fig, ax = plt.subplots(3, 2, figsize=(16,9))
fig.suptitle('Burst rates for RS units', fontsize=20)

## RS no light best position
ax[0][0].hist(burst_rate[m1_inds, m1_best_pos, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[0][0].hist(burst_rate[s1_inds, s1_best_pos, 0], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[0][0].legend(['M1', 'S1'])
ax[0][0].set_title('Best position\nno light')
ax[0][0].set_xlim(0, 40)
ax[0][0].set_ylim(0, y_max)
ax[0][0].set_ylabel('counts')

## RS S1 light best position
ax[1][0].hist(burst_rate[m1_inds, m1_best_pos + 9, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[1][0].hist(burst_rate[s1_inds, s1_best_pos + 9, 0], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[1][0].set_title('S1 light')
ax[1][0].set_xlim(0, 40)
ax[1][0].set_ylim(0, y_max)
ax[1][0].set_ylabel('counts')

## RS M1 light best position
ax[2][0].hist(burst_rate[m1_inds, 8+9+9, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[2][0].hist(burst_rate[s1_inds, 8+9+9, 0], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[2][0].set_title('M1 light')
ax[2][0].set_xlim(0, 40)
ax[2][0].set_ylim(0, y_max)
ax[2][0].set_xlabel('bursts/sec')
ax[2][0].set_ylabel('counts')

## RS no light no contact position
ax[0][1].hist(burst_rate[m1_inds, 8, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[0][1].hist(burst_rate[s1_inds, 8, 0], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[0][1].legend(['M1', 'S1'])
ax[0][1].set_title('No contact position\nno light')
ax[0][1].set_xlim(0, 40)
ax[0][1].set_ylim(0, y_max)

## RS S1 light no contact position
ax[1][1].hist(burst_rate[m1_inds, 8+ 9, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[1][1].hist(burst_rate[s1_inds, 8+ 9, 0], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[1][1].set_title('S1 light')
ax[1][1].set_xlim(0, 40)
ax[1][1].set_ylim(0, y_max)


## RS M1 light no contact position
ax[2][1].hist(burst_rate[m1_inds, 8+9+9, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[2][1].hist(burst_rate[s1_inds, 8+9+9, 0], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[2][1].set_title('M1 light')
ax[2][1].set_xlim(0, 40)
ax[2][1].set_ylim(0, y_max)
ax[2][1].set_xlabel('bursts/sec')



#### changes in m1 or s1 with s1 silencing and m1 silencing respectively at best position
#### changes in m1 or s1 with s1 silencing and m1 silencing respectively at best position

m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')
m1_best_pos = best_pos[m1_inds].astype(int)
s1_best_pos = best_pos[s1_inds].astype(int)
fig, ax = plt.subplots(2,2)
fig.suptitle('Changes in burst rates\nBest contact position', fontsize=15)
# m1 bursts S1 light
# paired plot
ax[0][0].scatter(np.zeros(sum(m1_inds)), burst_rate[m1_inds, m1_best_pos, 0], c='k')
ax[0][0].scatter(np.ones(sum(m1_inds)), burst_rate[m1_inds, m1_best_pos+9, 0], c='k')
# plotting the lines
for i in range(sum(m1_inds)):
        ax[0][0].plot( [0,1], [burst_rate[m1_inds, m1_best_pos, 0][i], burst_rate[m1_inds, m1_best_pos+9, 0][i]], 'k')
        ax[0][0].set_xticks([0,1])
        ax[0][0].set_xticklabels(('No light', 'S1 silencing'))

# plots histograms
#ax[0][0].hist(burst_rate[m1_inds, m1_best_pos, 0], bins=np.arange(0, 20, 1), alpha=0.5)
#ax[0][0].hist(burst_rate[m1_inds, m1_best_pos+9, 0], bins=np.arange(0, 20, 1), alpha=0.5)
stat, pval = sp.stats.wilcoxon(burst_rate[m1_inds, m1_best_pos, 0], burst_rate[m1_inds, m1_best_pos+9, 0])
ax[0][0].set_title('M1 burst rates with S1 silencing\nwilcoxon signed rank test p-val: {0:.5f}'.format(pval))
ax[0][0].set_xlim(-1.5, 2.5)

# s1 bursts M1 light
# paired plot
ax[1][0].scatter(np.zeros(sum(s1_inds)), burst_rate[s1_inds, s1_best_pos, 0], c='r')
ax[1][0].scatter(np.ones(sum(s1_inds)), burst_rate[s1_inds, s1_best_pos+9+9, 0], c='r')
# plotting the lines
for i in range(sum(s1_inds)):
        ax[1][0].plot( [0,1], [burst_rate[s1_inds, s1_best_pos, 0][i], burst_rate[s1_inds, s1_best_pos+9+9, 0][i]], 'r')
        ax[1][0].set_xticks([0,1])
        ax[1][0].set_xticklabels(('No light', 'M1 silencing'))

#ax[1][0].hist(burst_rate[s1_inds, s1_best_pos, 0], bins=np.arange(0, 20, 1), alpha=0.5)
#ax[1][0].hist(burst_rate[s1_inds, s1_best_pos+9+9, 0], bins=np.arange(0, 20, 1), alpha=0.5)
stat, pval = sp.stats.wilcoxon(burst_rate[s1_inds, s1_best_pos, 0], burst_rate[s1_inds, s1_best_pos+9+9, 0])
ax[1][0].set_title('S1 burst rates with M1 silencing\nwilcoxon signed rank test: {0:.5f}'.format(pval))
ax[1][0].set_xlim(-1.5, 2.5)

# m1 burst difference distribution S1 light
m1_diff = burst_rate[m1_inds, m1_best_pos+9, 0] - burst_rate[m1_inds, m1_best_pos, 0]
ax[0][1].hist(m1_diff, bins=np.arange(-10, 10, 2), alpha=0.5)
ax[0][1].set_xlim(-10, 10)
ax[0][1].set_title('Change in M1 burst rates\nS1 silencing')

# s1 burst difference distribution S1 light
s1_diff = burst_rate[s1_inds, s1_best_pos+9+9, 0] - burst_rate[s1_inds, s1_best_pos, 0]
ax[1][1].hist(s1_diff, bins=np.arange(-10, 10, 2), alpha=0.5)
ax[1][1].set_xlim(-10, 10)
ax[1][1].set_title('Change in S1 burst rates\nM1 silencing')



#### changes in m1 or s1 with s1 silencing and m1 silencing respectively at no contact position
#### changes in m1 or s1 with s1 silencing and m1 silencing respectively at no contact position

m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')
fig, ax = plt.subplots(2,2)
fig.suptitle('Changes in burst rates\nNo contact position', fontsize=15)
# m1 bursts S1 light
# paired plot
ax[0][0].scatter(np.zeros(sum(m1_inds)), burst_rate[m1_inds, 8, 0], c='k')
ax[0][0].scatter(np.ones(sum(m1_inds)), burst_rate[m1_inds, 8+9, 0], c='k')
# plotting the lines
for i in range(sum(m1_inds)):
        ax[0][0].plot( [0,1], [burst_rate[m1_inds, 8, 0][i], burst_rate[m1_inds, 8+9, 0][i]], 'k')
        ax[0][0].set_xticks([0,1])
        ax[0][0].set_xticklabels(('No light', 'M1 silencing'))

# plots histograms
#ax[0][0].hist(burst_rate[m1_inds, 8, 0], bins=np.arange(0, 20, 1), alpha=0.5)
#ax[0][0].hist(burst_rate[m1_inds, 8+9, 0], bins=np.arange(0, 20, 1), alpha=0.5)
stat, pval = sp.stats.wilcoxon(burst_rate[m1_inds, 8, 0], burst_rate[m1_inds, 8+9, 0])
ax[0][0].set_title('M1 burst rates with S1 silencing\nwilcoxon signed rank test p-val: {0:.5f}'.format(pval))
ax[0][0].set_xlim(-1.5, 2.5)

# s1 bursts M1 light
# paired plot
ax[1][0].scatter(np.zeros(sum(s1_inds)), burst_rate[s1_inds, 8, 0], c='r')
ax[1][0].scatter(np.ones(sum(s1_inds)), burst_rate[s1_inds, 8+9+9, 0], c='r')
# plotting the lines
for i in range(sum(s1_inds)):
        ax[1][0].plot( [0,1], [burst_rate[s1_inds, 8, 0][i], burst_rate[s1_inds, 8+9+9, 0][i]], 'r')
        ax[1][0].set_xticks([0,1])
        ax[1][0].set_xticklabels(('No light', 'M1 silencing'))

#ax[1][0].hist(burst_rate[s1_inds, 8, 0], bins=np.arange(0, 20, 1), alpha=0.5)
#ax[1][0].hist(burst_rate[s1_inds, 8+9+9, 0], bins=np.arange(0, 20, 1), alpha=0.5)
stat, pval = sp.stats.wilcoxon(burst_rate[s1_inds, 8, 0], burst_rate[s1_inds, 8+9+9, 0])
ax[1][0].set_title('S1 burst rates with M1 silencing\nwilcoxon signed rank test p-val: {0:.5f}'.format(pval))
ax[1][0].set_xlim(-1.5, 2.5)

# m1 burst difference distribution S1 light
m1_diff = burst_rate[m1_inds, 8+9, 0] - burst_rate[m1_inds, 8, 0]
ax[0][1].hist(m1_diff, bins=np.arange(-10, 10, 2), alpha=0.5)
ax[0][1].set_xlim(-10, 10)
ax[0][1].set_title('Change in M1 burst rates\nS1 silencing')

# s1 burst difference distribution S1 light
s1_diff = burst_rate[s1_inds, 8+9+9, 0] - burst_rate[s1_inds, 8, 0]
ax[1][1].hist(s1_diff, bins=np.arange(-10, 10, 2), alpha=0.5)
ax[1][1].set_xlim(-10, 10)
ax[1][1].set_title('Change in S1 burst rates\nM1 silencing')



##### changes in adaptation ratio in M1 with S1 silencing at best position
##### changes in adaptation ratio in M1 with S1 silencing at best position

m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')
m1_best_pos = best_pos[m1_inds].astype(int)
s1_best_pos = best_pos[s1_inds].astype(int)
fig, ax = plt.subplots(2,2)

fig.suptitle('Adaptation ratios\nbest position', fontsize=20)
# m1 adaptation rates s1 light
# paired plot
ax[0][0].scatter(np.zeros(sum(m1_inds)), adapt_ratio[m1_inds, m1_best_pos, 1], c='k')
ax[0][0].scatter(np.ones(sum(m1_inds)), adapt_ratio[m1_inds, m1_best_pos+9, 1], c='k')
# plotting the lines
for i in range(sum(m1_inds)):
        ax[0][0].plot( [0,1], [adapt_ratio[m1_inds, m1_best_pos, 1][i], adapt_ratio[m1_inds, m1_best_pos+9, 1][i]], 'k')
        ax[0][0].set_xticks([0,1])
        ax[0][0].set_xticklabels(('No light', 'S1 silencing'))

# plots histograms
#ax[0][0].hist(burst_rate[m1_inds, 8, 0], bins=np.arange(0, 20, 1), alpha=0.5)
#ax[0][0].hist(burst_rate[m1_inds, 8+9, 0], bins=np.arange(0, 20, 1), alpha=0.5)
stat, pval = sp.stats.wilcoxon(adapt_ratio[m1_inds, m1_best_pos, 0], adapt_ratio[m1_inds, m1_best_pos+9, 1])
ax[0][0].set_title('M1 adaptation rates with S1 silencing\nwilcoxon signed rank test p-val: {0:.5f}'.format(pval))
ax[0][0].set_xlim(-1.5, 2.5)

# s1 adaptation rates M1 light
# paired plot
ax[1][0].scatter(np.zeros(sum(s1_inds)), adapt_ratio[s1_inds, s1_best_pos, 1], c='r')
ax[1][0].scatter(np.ones(sum(s1_inds)), adapt_ratio[s1_inds, s1_best_pos+9+9, 1], c='r')
# plotting the lines
for i in range(sum(s1_inds)):
        ax[1][0].plot( [0,1], [adapt_ratio[s1_inds, s1_best_pos, 1][i], adapt_ratio[s1_inds, s1_best_pos+9+9, 1][i]], 'r')
        ax[1][0].set_xticks([0,1])
        ax[1][0].set_xticklabels(('No light', 'M1 silencing'))

#ax[1][0].hist(adapt_ratio[s1_inds, 8, 0], bins=np.arange(0, 20, 1), alpha=0.5)
#ax[1][0].hist(adapt_ratio[s1_inds, 8+9+9, 0], bins=np.arange(0, 20, 1), alpha=0.5)
stat, pval = sp.stats.wilcoxon(adapt_ratio[s1_inds, s1_best_pos, 1], adapt_ratio[s1_inds, s1_best_pos+9+9, 1])
ax[1][0].set_title('S1 adaptation rates with M1 silencing\nwilcoxon signed rank test p-val: {0:.5f}'.format(pval))
ax[1][0].set_xlim(-1.5, 2.5)

# m1 adaptation rates difference distribution S1 light
m1_diff = adapt_ratio[m1_inds, m1_best_pos+9, 1] - adapt_ratio[m1_inds, m1_best_pos, 1]
ax[0][1].hist(m1_diff, bins=np.arange(-2, 2, 0.05), alpha=0.5)
ax[0][1].set_xlim(-1, 1)
ax[0][1].set_title('Change in M1 adaptation rates\nS1 silencing')

# s1 adaptation rates difference distribution S1 light
s1_diff = adapt_ratio[s1_inds, s1_best_pos+9+9, 1] - adapt_ratio[s1_inds, s1_best_pos, 1]
ax[1][1].hist(s1_diff, bins=np.arange(-2, 2, 0.05), alpha=0.5)
ax[1][1].set_xlim(-1, 1)
ax[1][1].set_title('Change in S1 adaptation rates\nM1 silencing')



##### changes in adaptation ratio in M1 with S1 silencing at no contact position
##### changes in adaptation ratio in M1 with S1 silencing at no contact position

m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')
bins = np.arange(0, 100, 2)
fig, ax = plt.subplots(2,2)
fig.suptitle('Adaptation ratios\nno contact position', fontsize=20)

# m1 adaptation S1 light
# paired plot
ax[0][0].scatter(np.zeros(sum(m1_inds)), adapt_ratio[m1_inds, 8, 1], c='k')
ax[0][0].scatter(np.ones(sum(m1_inds)), adapt_ratio[m1_inds, 8+9, 1], c='k')
# plotting the lines
for i in range(sum(m1_inds)):
        ax[0][0].plot( [0,1], [adapt_ratio[m1_inds, 8, 1][i], adapt_ratio[m1_inds, 8+9, 1][i]], 'k')
        ax[0][0].set_xticks([0,1])
        ax[0][0].set_xticklabels(('No light', 'S1 silencing'))

# plots histograms
#ax[0][0].hist(burst_rate[m1_inds, 8, 0], bins=np.arange(0, 20, 1), alpha=0.5)
#ax[0][0].hist(burst_rate[m1_inds, 8+9, 0], bins=np.arange(0, 20, 1), alpha=0.5)
stat, pval = sp.stats.wilcoxon(adapt_ratio[m1_inds, 8, 0], adapt_ratio[m1_inds, 8+9, 1])
ax[0][0].set_title('M1 adaptation rates with S1 silencing\nwilcoxon signed rank test p-val: {0:.5f}'.format(pval))
ax[0][0].set_xlim(-1.5, 2.5)

# s1 adaptation rates M1 light
# paired plot
ax[1][0].scatter(np.zeros(sum(s1_inds)), adapt_ratio[s1_inds, 8, 1], c='r')
ax[1][0].scatter(np.ones(sum(s1_inds)), adapt_ratio[s1_inds, 8+9+9, 1], c='r')
# plotting the lines
for i in range(sum(s1_inds)):
        ax[1][0].plot( [0,1], [adapt_ratio[s1_inds, 8, 1][i], adapt_ratio[s1_inds, 8+9+9, 1][i]], 'r')
        ax[1][0].set_xticks([0,1], ['before', 'after'])

#ax[1][0].hist(adapt_ratio[s1_inds, 8, 0], bins=np.arange(0, 20, 1), alpha=0.5)
#ax[1][0].hist(adapt_ratio[s1_inds, 8+9+9, 0], bins=np.arange(0, 20, 1), alpha=0.5)
stat, pval = sp.stats.wilcoxon(adapt_ratio[s1_inds, 8, 1], adapt_ratio[s1_inds, 8+9+9, 1])
ax[1][0].set_title('S1 adaptation rates with M1 silencing\nwilcoxon signed rank test p-val: {0:.5f}'.format(pval))
ax[1][0].set_xlim(-1.5, 2.5)

# m1 adaptation rates difference distribution S1 light
m1_diff = adapt_ratio[m1_inds, 8+9, 1] - adapt_ratio[m1_inds, 8, 1]
ax[0][1].hist(m1_diff, bins=np.arange(-2, 2, 0.05), alpha=0.5)
ax[0][1].set_xlim(-1, 1)
ax[0][1].set_title('Change in M1 adaptation rates\nS1 silencing')

# s1 adaptation rates difference distribution S1 light
s1_diff = adapt_ratio[s1_inds, 8+9+9, 1] - adapt_ratio[s1_inds, 8, 1]
ax[1][1].hist(s1_diff, bins=np.arange(-2, 2, 0.05), alpha=0.5)
ax[1][1].set_xlim(-1, 1)
ax[1][1].set_title('Change in S1 adaptation rates\nM1 silencing')


##### m1 and s1 adaptation during best contact position
m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')
m1_best_pos = best_pos[m1_inds].astype(int)
s1_best_pos = best_pos[s1_inds].astype(int)
bins = np.arange(0, 100, 2)
fig, ax = plt.subplots(1,1)

fig.suptitle('Adaptation ratios', fontsize=20)
# plots histograms
ax.hist(adapt_ratio[m1_inds, m1_best_pos, 1], bins=np.arange(0, 2, 0.1), alpha=0.5, color='k')
ax.hist(adapt_ratio[s1_inds, s1_best_pos+9, 1], bins=np.arange(0, 2, 0.1), alpha=0.5, color='r')


                    ##### NEW PDF FLIP BOOK CODE #####
                    ##### NEW PDF FLIP BOOK CODE #####
                    ##### NEW PDF FLIP BOOK CODE #####

##### REMOVE #####
unit_count = 2
unit_index = 2
neuro = experiments[0]
##### REMOVE #####

unit_count = 0
unit_type  = 'FS'
fig_dir = os.path.expanduser('~/Documents/AdesnikLab/temp/')

with PdfPages(fig_dir + 'all_RS_unit_summaries.pdf') as pdf:
    for neuro in experiments:
        for unit_index in range(neuro.num_units):


            # if unit matches criteria add it to PDF book
            if cell_type[unit_count] == unit_type \
                    and max_fr[unit_count] > 2 \
                    and depths[unit_count] > 40 \
                    and driven[unit_count] == 1:

                # M1
                if region[unit_count] == 0:
                    offset = 9
                    c = 'r'
                # S1
                elif region[unit_count] == 1:
                    offset = 9+9
                    c = 'b'
                max_evk_rate = np.max([np.max(evk_rate[unit_count, 0:neuro.control_pos, 0]),\
                        np.max(evk_rate[unit_count, offset:neuro.control_pos+offset, 0])])
                min_evk_rate = np.min([np.min(evk_rate[unit_count, 0:neuro.control_pos, 0]),\
                        np.min(evk_rate[unit_count, offset:neuro.control_pos+offset, 0])])
                burst_nolight = np.zeros((neuro.control_pos-1, 2))
                burst_light = np.zeros((neuro.control_pos-1, 2))
                for pos in range(neuro.control_pos-1):
                    temp_nolight = neuro.get_burst_rate(unit_ind=unit_index, trial_type=pos)
                    burst_nolight[pos, 0] = np.mean(temp_nolight)
                    burst_nolight[pos, 1] = sp.stats.sem(temp_nolight)

                    temp_light = neuro.get_burst_rate(unit_ind=unit_index, trial_type=pos+offset)
                    burst_light[pos, 0] = np.mean(temp_light)
                    burst_light[pos, 1] = sp.stats.sem(temp_light)

                # create figure and add to PDF book
                fig = plt.figure(figsize=(12, 14))
                fig.suptitle('Region: {}, depth: {}, unit type: {}, mouse: {}'.format(\
                        neuro.region_dict[neuro.shank_ids[unit_index]], \
                        neuro.depths[unit_index], \
                        neuro.cell_type[unit_index], \
                        neuro.fid))
                ax1 = plt.subplot2grid((5,3), (0,0), colspan=1, rowspan=1)
                ax2 = plt.subplot2grid((5,3), (0,1), colspan=1, rowspan=1)
                ax3 = plt.subplot2grid((5,3), (0,2), colspan=1, rowspan=1)
                ax4 = plt.subplot2grid((5,3), (1,0), colspan=2, rowspan=2)
                ax5 = plt.subplot2grid((5,3), (1,2), colspan=1, rowspan=1)
                ax6 = plt.subplot2grid((5,3), (2,2), colspan=1, rowspan=1)
                ax7 = plt.subplot2grid((5,3), (3,0), colspan=2, rowspan=2)
                ax8 = plt.subplot2grid((5,3), (3,2), colspan=1, rowspan=1)
                ax9 = plt.subplot2grid((5,3), (4,2), colspan=1, rowspan=1)
                plt.subplots_adjust(left=0.10, bottom=0.05, right=0.90, top=0.94, wspace=0.45, hspace=0.45)

                # axis 1 (0,0) PSTH best position
                neuro.plot_psth(axis=ax1, unit_ind=unit_index, trial_type=best_pos[unit_count], error='sem', color='k')
                neuro.plot_psth(axis=ax1, unit_ind=unit_index, trial_type=best_pos[unit_count]+offset, error='sem', color=c)
                ax1.set_xlim(-0.5, 2)
                ax1.set_ylim(0, ax1.get_ylim()[1])
                ax1.hlines(0,-0.5, 2, colors='k', linestyles='dashed')
                ax1.vlines(0.5, ax1.get_ylim()[0], ax1.get_ylim()[1], colors='c', linestyles='dashed')
                ax1.vlines(1.5, ax1.get_ylim()[0], ax1.get_ylim()[1], colors='c', linestyles='dashed')
                ax1.set_xlabel('time (s)')
                ax1.set_ylabel('firing rate (Hz)')
                ax1.set_title('best contact')

                # axis 2 (0,1) PSTH no contact position
                neuro.plot_psth(axis=ax2, unit_ind=unit_index, trial_type=neuro.control_pos-1, error='sem', color='k')
                neuro.plot_psth(axis=ax2, unit_ind=unit_index, trial_type=neuro.control_pos-1+offset, error='sem', color=c)
                ax2.set_xlim(-0.5, 2)
                ax2.set_ylim(0, ax2.get_ylim()[1])
                ax2.hlines(0,-0.5, 2, colors='k', linestyles='dashed')
                ax2.vlines(0.5, ax2.get_ylim()[0], ax2.get_ylim()[1], colors='c', linestyles='dashed')
                ax2.vlines(1.5, ax2.get_ylim()[0], ax2.get_ylim()[1], colors='c', linestyles='dashed')
                ax2.set_xlabel('time (s)')
                ax2.set_ylabel('firing rate (Hz)')
                ax2.set_title('no contact')

                # axis 3 (0,2) absolute tuning curve
                neuro.plot_tuning_curve(unit_ind=unit_index, kind='abs_count', axis=ax3)
                ax3.set_xlim(0, 10)
                ax3.hlines(0, 0, 10, colors='k', linestyles='dashed')
                ax3.set_xlabel('bar position')
                ax3.set_title('absolute tc')

                # axis 4 (1,0) evoked tuning curve
                neuro.plot_tuning_curve(unit_ind=unit_index, kind='evk_count', axis=ax4)
                ax4.set_xlim(0, 10)
                ax4.hlines(0, 0, 10, colors='k', linestyles='dashed')
                ax4.set_xlabel('bar position')
                ax4.set_title('evoked tc')
                ax4.set_ylim(min_evk_rate*1.5-2, max_evk_rate*1.5)

                # axis 5 (1, 2), OMI tuning curves
                omi_s1light = (abs_rate[unit_count, neuro.control_pos:neuro.control_pos+9, 0] - abs_rate[unit_count, :neuro.control_pos, 0]) / \
                        (abs_rate[unit_count, neuro.control_pos:neuro.control_pos+9, 0] + abs_rate[unit_count, :neuro.control_pos, 0])
                omi_m1light = (abs_rate[unit_count, neuro.control_pos+9:neuro.control_pos+9+9, 0] - abs_rate[unit_count, :neuro.control_pos, 0]) / \
                        (abs_rate[unit_count, neuro.control_pos+9:neuro.control_pos+9+9, 0] + abs_rate[unit_count, :neuro.control_pos, 0])
                ax5.plot(np.arange(1,10), omi_s1light, '-ro', np.arange(1,10), omi_m1light, '-bo')
                ax5.hlines(0, 0, 10, colors='k', linestyles='dashed')
                ax5.set_xlim(0, 10)
                ax5.set_ylim(-1, 1)
                ax5.set_xlabel('bar position')
                ax5.set_ylabel('OMI')
                ax5.set_title('OMI tc')

                # axis 6 (2, 2) mean waveform
                ax6.plot(np.arange(neuro.waves[unit_index, :].shape[0]), neuro.waves[unit_index, :], 'k')
                ax6.set_xlim(0, neuro.waves[unit_index, :].shape[0])
                ax6.set_title('Mean waveform')
#                ax6.set_xlabel('dur: {}, ratio: {}'.format(\
#                        neuro.duration[unit_index],\
#                        neuro.ratio[unit_index]))

                # axis 7 (3, 0) burst rate tuning curve
                ax7.errorbar(np.arange(1,neuro.control_pos),\
                        burst_nolight[:, 0],\
                        yerr=burst_nolight[:, 1],\
                        fmt='k', marker='o', markersize=8.0, linewidth=2)
                ax7.errorbar(np.arange(1,neuro.control_pos),\
                        burst_light[:, 0],\
                        yerr=burst_light[:, 1],\
                        fmt=c, marker='o', markersize=8.0, linewidth=2)
                ax7.set_xlim(0, 9)
                ax7.set_xlabel('bar position')
                ax7.set_ylabel('Burst rate (bursts/sec)')
                ax7.set_title('Burst rate tc')

                # axis 8 (3, 2) selectivity histogram
#                ax8.hist(selectivity[cell_type==unit_type, 0], bins=np.arange(0, 1, 0.05), edgecolor='None', alpha=0.5, color='k')
#                ax8.hist(selectivity[cell_type==unit_type, offset/neuro.control_pos], bins=np.arange(0, 1, 0.05), edgecolor='None', alpha=0.5, color=c)
                ax8.hist(selectivity[npand(cell_type==unit_type, region == region[unit_count]), 0], bins=np.arange(0, 1, 0.05), edgecolor='None', alpha=0.5, color='k')
                ax8.hist(selectivity[npand(cell_type==unit_type, region == region[unit_count]), offset/neuro.control_pos], bins=np.arange(0, 1, 0.05), edgecolor='None', alpha=0.5, color=c)
#                ax8.arrow(selectivity[unit_index, 0], ax8.get_ylim()[1]+1,0, -1, head_width=0.05, head_length=0.2, color='k')
#                ax8.arrow(selectivity[unit_index, offset/neuro.control_pos], ax8.get_ylim()[1]+1,0, -1, head_width=0.05, head_length=0.2, color=c)
                ax8.arrow(neuro.selectivity[unit_index, 0], ax8.get_ylim()[1]+1,0, -1, head_width=0.05, head_length=0.2, color='k')
                ax8.arrow(neuro.selectivity[unit_index, offset/neuro.control_pos], ax8.get_ylim()[1]+1,0, -1, head_width=0.05, head_length=0.2, color=c)
                ax8.set_ylim(0, ax8.get_ylim()[1]+3)
                ax8.set_title('selectivity')

                # axis 9 (4, 2) OMI histogram
                ax9.hist(omi[npand(cell_type==unit_type, region == region[unit_count]), offset/neuro.control_pos - 1], bins=np.arange(-1, 1, 0.1), edgecolor='None', alpha=0.5, color=c)
                ax9.arrow(neuro.get_omi()[unit_index, offset/neuro.control_pos - 1], ax9.get_ylim()[1]+1,0, -1, head_width=0.1, head_length=0.4, color=c)
                ax9.set_ylim(0, ax9.get_ylim()[1]+3)
                ax9.set_xlim(-1, 1)
                ax9.set_title('OMIs')

                pdf.savefig()
#                fig.clear()
                plt.close(fig)

            unit_count += 1







