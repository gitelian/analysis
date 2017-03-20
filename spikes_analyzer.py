import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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

for neuro in experiments:
    # calculate measures that weren't calculated at init

    # concatenate measures
    cell_type.extend(neuro.cell_type)
    region      = np.append(region, neuro.shank_ids)
    depths      = np.append(depths, np.asarray(neuro.depths))
    selectivity = np.append(selectivity, neuro.selectivity, axis=0)
    driven      = np.append(driven, neuro.driven_units, axis=0)
    omi         = np.append(omi, neuro.get_omi(), axis=0)
    preference  = np.append(preference, neuro.preference, axis=0)

cell_type = np.asarray(cell_type)
region = region[1:,]
depths = depths[1:,]
driven = driven[1:,]
omi    = omi[1:,]
selectivity = selectivity[1:, :]
preference  = preference[1:, :]

##### select units #####
npand   = np.logical_and
#m1_inds = npand(npand(region==0, driven==True), cell_type=='MU')
#s1_inds = npand(npand(region==1, driven==True), cell_type=='MU')

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

bins = np.arange(-1.5, 1.5, 0.10)
fig, ax = plt.subplots(3, 1, figsize=(8,8))
ax[0].hist(preference[m1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[0].hist(preference[s1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[0].set_title('M1, S1, no light')
ax[1].hist(preference[m1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[1].hist(preference[m1_inds, 1], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[1].set_title('M1, S1 light')
ax[2].hist(preference[s1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[2].hist(preference[s1_inds, 2], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[2].set_title('S1, M1 light')

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

# make scatter plot of M1 firing rate for best contact positions with and
# without S1 silencing.

#plt.figure()
## m1
#plt.scatter(m1_rates[8], m1_rates[8+9], color='b')
## s1
#plt.scatter(s1_rates[8], s1_rates[8+9+9], color='r')
## unity line
#plt.plot([0, 40], [0, 40], 'k')
#plt.xlim(0, 40); plt.ylim(0, 40)
#
#plt.figure()
## m1
#plt.scatter(m1_rates[8], m1_rates[8+9+9], color='b')
## s1
#plt.scatter(s1_rates[8], s1_rates[8+9], color='r')
## unity line
#plt.plot([0, 100], [0, 100], 'k')
#plt.xlim(0, 100); plt.ylim(0, 100)
#
## violin plot of spontaneous rates
#pos = [1, 2]
#violinplot([m1_rates[8], s1_rates[8]], pos, vert=True, widths=0.7,

###### Plot selectivity stuff #####
## m1 selectivity with and without s1 silencing
#bins = np.arange(0, 1, 0.05)
#plt.figure()
#plt.hist(m1_sel_nolight, bins=bins, edgecolor='None', alpha=0.5, color='k')
#plt.hist(m1_sel_s1light, bins=bins, edgecolor='None', alpha=0.5, color='r')
#
#bins = np.arange(-1, 1, 0.05)
#plt.figure()
#plt.hist(m1_sel_s1light-m1_sel_nolight, bins=bins, edgecolor='None', alpha=0.5, color='k')
#
#bins = np.arange(0, 1, 0.05)
#plt.figure()
#plt.hist(s1_sel_nolight, bins=bins, edgecolor='None', alpha=0.5, color='k')
#plt.hist(s1_sel_s1light, bins=bins, edgecolor='None', alpha=0.5, color='r')
#
#



