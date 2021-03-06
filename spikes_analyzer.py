import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import scipy.io as sio


### !!! gode code for making spikes figures !!! ###
### !!! gode code for making spikes figures !!! ###
### should be looked over and maybe cleaned up ###

#get_ipython().magic(u"run neoanalyzer.py {}".format(sys.argv[1]))

# create multipage PDF of unit summaries
t_before, t_after = -0.5, 2

with PdfPages(fid + '_unit_summaries.pdf') as pdf:
    for unit_index in range(neuro.num_units):

        # get best contact position from evoked rates
        meanr     = np.array([np.mean(k[:, unit_index]) for k in neuro.evk_rate])
        meanr_abs = np.array([np.mean(k[:, unit_index]) for k in neuro.abs_rate])
        best_contact = np.argmax(meanr[0:8])

        fig, ax = plt.subplots(4, 3, figsize=(10,8))
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.36, hspace=0.60)
        fig.suptitle('Region: {0}, depth: {1}, unit type: {2}, mouse: {3}, driven: {4}, selectivity {5:.3f}'.format(\
                neuro.region_dict[neuro.shank_ids[unit_index]], \
                neuro.depths[unit_index], \
                neuro.cell_type[unit_index], \
                fid, \
                neuro.driven_units[unit_index],\
                neuro.selectivity[unit_index, 0]))

        # top left: best contact PSTH
        neuro.plot_psth(axis=ax[0][0], unit_ind=unit_index, trial_type=best_contact, error='sem', color='k')
        ## if opto:
        neuro.plot_psth(axis=ax[0][0], unit_ind=unit_index, trial_type=best_contact+9, error='sem', color='r')
        neuro.plot_psth(axis=ax[0][0], unit_ind=unit_index, trial_type=best_contact+9+9, error='sem', color='b')
        ax[0][0].set_xlim(t_before, t_after)
        ax[0][0].set_ylim(0.5, ax[0][0].get_ylim()[1])
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
        neuro.plot_psth(axis=ax[0][1], unit_ind=unit_index, trial_type=neuro.control_pos-1+9, error='sem', color='r')
        neuro.plot_psth(axis=ax[0][1], unit_ind=unit_index, trial_type=neuro.control_pos-1+9+9, error='sem', color='b')
        ax[0][1].set_xlim(t_before, t_after)
        ax[0][1].set_ylim(0.5, ax[0][0].get_ylim()[1])
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
        neuro.plot_tuning_curve(unit_ind=unit_index, kind='abs_count', axis=ax[0][2])
        ax[0][2].set_xlim(0, 10)
        ax[0][2].hlines(0, 0, 10, colors='k', linestyles='dashed')
        ax[0][2].set_xlabel('bar position')
        ax[0][2].set_title('absolute tc')

        # middle left: raster during no light and best contact
        neuro.plot_raster(unit_ind=unit_index, trial_type=best_contact, axis=ax[1][0], burst=False)
        ax[1][0].set_xlim(t_before, t_after)
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
        ax[1][1].set_xlim(t_before, t_after)
        # stimulus onset lines
        ax[1][1].vlines(0.00, ax[1][1].get_ylim()[0], ax[1][1].get_ylim()[1], colors='r', linewidth=1)
        ax[1][1].vlines(1.51, ax[1][1].get_ylim()[0], ax[1][1].get_ylim()[1], colors='r', linewidth=1)
        # light onset lines
        ax[1][1].vlines(0.5, ax[1][1].get_ylim()[0], ax[1][1].get_ylim()[1], colors='m', linestyles='dashed', linewidth=1)
        ax[1][1].vlines(1.5, ax[1][1].get_ylim()[0], ax[1][1].get_ylim()[1], colors='m', linestyles='dashed', linewidth=1)
        ax[1][1].set_xlabel('time (s)')
        ax[1][1].set_ylabel('no light trials')

        # middle right: evoked tuning curves
       ## if opto:
        neuro.plot_tuning_curve(unit_ind=unit_index, kind='evk_count', axis=ax[1][2])
        ax[1][2].set_xlim(0, 10)
        ax[1][2].hlines(0, 0, 10, colors='k', linestyles='dashed')
        ax[1][2].set_xlabel('bar position')
        ax[1][2].set_title('evoked tc')

#        # bottom left: raster for best contact and S1 light
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

        # bottom right: OMI tuning curves
        omi_s1light = (meanr_abs[neuro.control_pos:neuro.control_pos+9] - meanr_abs[:neuro.control_pos]) / \
                (meanr_abs[neuro.control_pos:neuro.control_pos+9] + meanr_abs[:neuro.control_pos])
        omi_m1light = (meanr_abs[neuro.control_pos+9:neuro.control_pos+9+9] - meanr_abs[:neuro.control_pos]) / \
        (meanr_abs[neuro.control_pos+9:neuro.control_pos+9+9] + meanr_abs[:neuro.control_pos])
        ax[2][2].plot(np.arange(1,10), omi_s1light, '-ro', np.arange(1,10), omi_m1light, '-bo')
        ax[2][2].hlines(0, 0, 10, colors='k', linestyles='dashed')
        ax[2][2].set_xlim(0, 10)
        ax[2][2].set_ylim(-1, 1)
        ax[2][2].set_xlabel('bar position')
        ax[2][2].set_ylabel('OMI')
        ax[2][2].set_title('OMI tc')

        # bottom bottom left: raster for best contact and S1 light ## if opto:
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

        # bottom bottom right: mean waveform
        ax[3][2].plot(np.arange(neuro.waves[unit_index, :].shape[0]), neuro.waves[unit_index, :], 'k')
        ax[3][2].set_xlim(0, neuro.waves[unit_index, :].shape[0])
        ax[3][2].set_title('Mean waveform')
        ax[3][2].set_xlabel('dur: {}, ratio: {}'.format(neuro.duration[unit_index], neuro.ratio[unit_index]))
        ## END if opto:

        pdf.savefig()
        fig.clear()
        plt.close()

##### spike time correlation analysis #####
##### spike time correlation analysis #####

rebinned_spikes, t = neuro.rebin_spikes(bin_size=0.020, analysis_window=[0.5, 1.5])

pos = 8
R_nolight, sorted_inds = neuro.spike_time_corr(rebinned_spikes, cond=pos)
R_light, sorted_inds = neuro.spike_time_corr(rebinned_spikes, cond=pos+9)

vmin, vmax  = -0.2, 0.2
fig, ax = plt.subplots(1, 2)
ax[0].imshow(R_nolight, vmin=vmin, vmax=vmax, cmap='coolwarm')
im = ax[1].imshow(R_light, vmin=vmin, vmax=vmax, cmap='coolwarm')
#im = ax[2].imshow(R_light - R_nolight, vmin=vmin, vmax=vmax, cmap='coolwarm')
fig.colorbar(im, ax=ax[1])

# m1/s1 border
border = np.where(np.diff(neuro.shank_ids)==1)[0]
ax[0].hlines(border, 0, neuro.num_units-1, linewidth=0.5, color='k')
ax[0].vlines(border, 0, neuro.num_units-1, linewidth=0.5, color='k')
ax[1].hlines(border, 0, neuro.num_units-1, linewidth=0.5, color='k')
ax[1].vlines(border, 0, neuro.num_units-1, linewidth=0.5, color='k')

ax[0].set_title('no light')
ax[1].set_title('s1 light')
fig.suptitle(fid + ' position {}'.format(str(pos)))

## histogram of all correlations values for either M1-M1 or S1-S1
# grab M1-M1 correlation values
tri_inds  = np.triu_indices(border, k=1)

m1m1_corr  = R_nolight[0:int(border), 0:int(border)]
m1m1_nolight = np.nan_to_num(m1m1_corr[tri_inds[0], tri_inds[1]], 0)

m1m1_corr = R_light[0:int(border), 0:int(border)]
m1m1_light= np.nan_to_num(m1m1_corr[tri_inds[0], tri_inds[1]], 0)

bins = np.arange(-1, 1, 0.01)
fig, ax = plt.subplots(1, 1)
ax.hist(m1m1_nolight, bins=bins, edgecolor='None', alpha=0.5, color='k')
ax.hist(m1m1_light, bins=bins, edgecolor='None', alpha=0.5, color='r')
sp.stats.wilcoxon(m1m1_nolight, m1m1_light)

# grab S1-S1 correlation values
tri_inds  = np.triu_indices(neuro.num_units - border - 1, k=1)

s1s1_corr  = R_nolight[int(border):neuro.num_units-1, int(border):neuro.num_units-1]
s1s1_nolight = np.nan_to_num(s1s1_corr[tri_inds[0], tri_inds[1]], 0)

s1s1_corr  = R_light[int(border):neuro.num_units-1, int(border):neuro.num_units-1]
s1s1_light = np.nan_to_num(s1s1_corr[tri_inds[0], tri_inds[1]], 0)

bins = np.arange(-1, 1, 0.01)
fig, ax = plt.subplots(1, 1)
ax.hist(s1s1_nolight, bins=bins, edgecolor='None', alpha=0.5, color='k')
ax.hist(s1s1_light, bins=bins, edgecolor='None', alpha=0.5, color='r')

sp.stats.wilcoxon(s1s1_nolight, s1s1_light)

##
# grab s1-m1 correlation values
R_nocontact, sorted_inds = neuro.spike_time_corr(rebinned_spikes, cond=8)
R_contact, sorted_inds = neuro.spike_time_corr(rebinned_spikes, cond=pos)
a = R_nocontact[0:border[0], border[0]:neuro.num_units]
b = R_contact[0:border[0], border[0]:neuro.num_units]


##### noise correlation analysis #####
##### noise correlation analysis #####

pos = 4
R_nolight, sorted_inds = neuro.noise_corr(cond=pos)
R_light, sorted_inds   = neuro.noise_corr(cond=pos+9+9)

vmin, vmax  = -1, 1
fig, ax = plt.subplots(1, 2)
ax[0].imshow(R_nolight, vmin=vmin, vmax=vmax, cmap='coolwarm')
im = ax[1].imshow(R_light, vmin=vmin, vmax=vmax, cmap='coolwarm')
#im = ax[2].imshow(R_light - R_nolight, vmin=vmin, vmax=vmax, cmap='coolwarm')
fig.colorbar(im, ax=ax[1])

# m1/s1 border
border = np.where(np.diff(neuro.shank_ids)==1)[0]
ax[0].hlines(border, 0, neuro.num_units-1, linewidth=0.5, color='k')
ax[0].vlines(border, 0, neuro.num_units-1, linewidth=0.5, color='k')
ax[1].hlines(border, 0, neuro.num_units-1, linewidth=0.5, color='k')
ax[1].vlines(border, 0, neuro.num_units-1, linewidth=0.5, color='k')

ax[0].set_title('no light')
ax[1].set_title('light')
fig.suptitle(fid + ' position {}'.format(str(pos)))

## histogram of all correlations values for either M1-M1 or S1-S1

# grab M1-M1 correlation values
tri_inds  = np.triu_indices(border, k=1)

m1m1_corr  = R_nolight[0:int(border), 0:int(border)]
m1m1_nolight = np.nan_to_num(m1m1_corr[tri_inds[0], tri_inds[1]], 0)

m1m1_corr = R_light[0:int(border), 0:int(border)]
m1m1_light= np.nan_to_num(m1m1_corr[tri_inds[0], tri_inds[1]], 0)

bins = np.arange(-1, 1, 0.05)
fig, ax = plt.subplots(1, 1)
ax.hist(m1m1_nolight, bins=bins, edgecolor='None', alpha=0.5, color='k')
ax.hist(m1m1_light, bins=bins, edgecolor='None', alpha=0.5, color='r')
ax.set_title('m1-m1 correlation values')
sp.stats.wilcoxon(m1m1_nolight, m1m1_light)

# grab S1-S1 correlation values
tri_inds  = np.triu_indices(neuro.num_units - border - 1, k=1)

s1s1_corr  = R_nolight[int(border):neuro.num_units-1, int(border):neuro.num_units-1]
s1s1_nolight = np.nan_to_num(s1s1_corr[tri_inds[0], tri_inds[1]], 0)

s1s1_corr  = R_light[int(border):neuro.num_units-1, int(border):neuro.num_units-1]
s1s1_light = np.nan_to_num(s1s1_corr[tri_inds[0], tri_inds[1]], 0)

bins = np.arange(-1, 1, 0.05)
fig, ax = plt.subplots(1, 1)
ax.hist(s1s1_nolight, bins=bins, edgecolor='None', alpha=0.5, color='k')
ax.hist(s1s1_light, bins=bins, edgecolor='None', alpha=0.5, color='r')
ax.set_title('s1-s1 correlation values')
sp.stats.wilcoxon(s1s1_nolight, s1s1_light)

##### firing rate vs run speed no contact position #####
##### firing rate vs run speed no contact position #####
### NOT IMPLEMENTED ###


# there was either no or a very weak correlation


npand   = np.logical_and
m1_inds = npand(npand(neuro.shank_ids == 0, neuro.driven_units ==True), neuro.cell_type=='RS')
s1_inds = npand(npand(neuro.shank_ids == 1, neuro.driven_units ==True), neuro.cell_type=='RS')

S_mean, S_all = neuro.get_sparseness(kind='lifetime')


##### Sparsity analysis #####
##### Sparsity analysis #####

## M1
m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')
plt.figure()
S_m1_nolight = S_all[m1_inds, 0]
S_m1_s1light = S_all[m1_inds, 1]
num_points = S_m1_nolight.shape[0]
for k in range(num_points):
    plt.scatter(0, S_m1_nolight[k], color='k')
    plt.scatter(1, S_m1_s1light[k], color='b')
    plt.plot([0, 1], [S_m1_nolight[k], S_m1_s1light[k]], 'k')

## S1
plt.figure()
S_s1_nolight = S_all[s1_inds, 0]
S_s1_m1light = S_all[s1_inds, 2]
# plot paired scatter plot for S1
num_points = S_s1_nolight.shape[0]
for k in range(num_points):
    plt.scatter(0, S_s1_nolight[k], color='k')
    plt.scatter(1, S_s1_m1light[k], color='b')
    plt.plot([0, 1], [S_s1_nolight[k], S_s1_m1light[k]], 'k')

## S1 vs M1 no light
plt.figure()
hist(S_m1_nolight, bins=np.arange(0, 1, 0.1), alpha=0.5)
hist(S_s1_nolight, bins=np.arange(0, 1, 0.1), alpha=0.5)


##### population analysis #####
##### population analysis #####

# selectivity, center of mass, burstiness, OMI, decoder!!!
# do this for best position and no contact position. Plot things overall and
# then look at things as a function of depth.

# 1302 and 1318 gave errors while loading...investigate this
fids = ['1295', '1302', '1318', '1328', '1329', '1330', '1336', '1338', '1340', '1343', '1345']
# FID1330 and beyond has episodic HSV
fids = ['1336', '1338', '1339', '1340', '1343', '1345']
#fids = ['1336','1338', '1340', '1343', '1345']

#fids = ['1302', '1318', '1330']
#fids = ['1336', '1338', '1339']
# fid with good whisker tracking
#fids = ['1330', '1336', '1338', '1339', '1340']
experiments = list()
for fid_name in fids:
    get_ipython().magic(u"run hdfanalyzer.py {}".format(fid_name))
    experiments.append(neuro)

##### multiple experiment optogenetic analysis #####
##### multiple experiment optogenetic analysis #####

change = list()
cell_type= list()
fid_list = list()

# m1 analysis
for k, neuro in enumerate(experiments):
    # calculate measures that weren't calculated at init
    neuro.reclassify_units()

    for uind in range(neuro.num_units):
        if neuro.shank_ids[uind] == 1:
            best_contact = neuro.best_contact[uind]
            abs_rate_light   = neuro.abs_rate[best_contact+9+9][:, uind].mean()
            abs_rate_nolight = neuro.abs_rate[best_contact][:, uind].mean()

            temp = (abs_rate_light - abs_rate_nolight) / (abs_rate_light + abs_rate_nolight)
            change.append(temp)
            #change.append(abs_rate_light/abs_rate_nolight*100)
            cell_type.append(neuro.cell_type[uind])
            fid_list.append(fids[k])



df = pd.DataFrame({'fid': fid_list, 'cell_type': cell_type, 'change': change})
#df = df.set_index('fid')
sns.set_color_codes()
plt.figure()
#sns.boxplot(x="fid", y="change", hue="cell_type", data=df, palette=['b', 'r'])
sns.boxplot(x="fid", y="change", hue="cell_type", data=df, palette={"RS":'b', "FS":'r'})
plt.hlines(0, plt.xlim()[0], plt.xlim()[1], linestyles='dashed')
plt.ylim(-1, 1)


##### create arrays and lists for concatenating specified data from all experiments #####
##### create arrays and lists for concatenating specified data from all experiments #####
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
abs_tc      = np.empty((1, 3, 2))
evk_tc      = np.empty((1, 3, 2))
max_fr      = np.empty((1, ))
burst_rate  = np.empty((1, 27, 2))
adapt_ratio = np.empty((1, 27, 2))
meanr       = np.empty((1, 9, 3))
S_all       = np.empty((1, 3))

num_driven  = np.empty((1, ))

for neuro in experiments:
    # calculate measures that weren't calculated at init
    neuro.get_best_contact()
    neuro.get_sensory_drive(num_sig_pos=True)

    # concatenate measures
    cell_type.extend(neuro.cell_type)
    region      = np.append(region, neuro.shank_ids)
    depths      = np.append(depths, np.asarray(neuro.depths))
    selectivity = np.append(selectivity, neuro.selectivity, axis=0)
    driven      = np.append(driven, neuro.driven_units, axis=0)
    omi         = np.append(omi, neuro.get_omi(), axis=0)
    preference  = np.append(preference, neuro.preference, axis=0)
    best_pos    = np.append(best_pos, neuro.best_contact)
    meanr       = np.append(meanr, neuro.get_rates_vs_strength(normed=True)[0], axis=0)
    S_all       = np.append(S_all, neuro.get_sparseness(kind='lifetime')[1], axis=0)

    num_driven  = np.append(num_driven, neuro.num_driven_pos, axis=0)

    # compute mean tuning curve
    abs_tc = np.append(abs_tc, neuro.get_mean_tc(kind='abs_rate'), axis=0)
    evk_tc = np.append(evk_tc, neuro.get_mean_tc(kind='evk_rate'), axis=0)


    for unit_index in range(neuro.num_units):

        # compute absolute rate (mean and sem)
        temp = np.zeros((1, 27, 2))
        temp[0, :, 0] = np.array([np.mean(k[:, unit_index]) for k in neuro.abs_rate])[:]
        temp[0, :, 1] = np.array([sp.stats.sem(k[:, unit_index]) for k in neuro.abs_rate])
        abs_rate = np.append(abs_rate, temp, axis=0)
        max_fr   = np.append(max_fr, np.max(temp))

        # compute evoked rate (mean and sem)
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

num_driven = np.asarray(num_driven[1:,])

omi    = omi[1:,]
omi    = np.nan_to_num(omi)
selectivity = selectivity[1:, :]
preference  = preference[1:, :]
best_pos    = best_pos[1:,]
best_pos    = best_pos.astype(int)
meanr       = meanr[1:, :]
abs_rate    = abs_rate[1:, :]
evk_rate    = evk_rate[1:, :]
abs_tc      = abs_tc[1:, :]
evk_tc      = evk_tc[1:, :]
max_fr      = max_fr[1:,]
burst_rate  = burst_rate[1:, :]
adapt_ratio = adapt_ratio[1:, :]
S_all       = S_all[1:, :]

##### select units #####
npand   = np.logical_and
#m1_inds = npand(npand(region==0, driven==True), cell_type=='MU')
#s1_inds = npand(npand(region==1, driven==True), cell_type=='MU')

##### loadt burst matrix #####
#burst_path = '/Users/Greg/Documents/AdesnikLab/Data/burst.mat'
#burst_rate = sio.loadmat(burst_path)['burst_rate']
#
###### save burst matrix #####
#a = dict()
#burst_path = '/Users/Greg/Documents/AdesnikLab/Data/burst.mat'
#a['burst_rate'] = burst_rate
#sio.savemat(burst_path, a)



##### Selectivity analysis #####
##### Selectivity analysis #####


###### Plot selectivity histogram #####
###### Plot selectivity histogram #####

## RS
m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')
#m1_inds = npand(region==0, cell_type=='RS')
#s1_inds = npand(region==1, cell_type=='RS')
bins = np.arange(0, 1, 0.05)
fig, ax = plt.subplots(3, 2, figsize=(12,9))
fig.suptitle('selectivity', fontsize=20)

#hist(selectivity[npand(cell_type==unit_type, region == region[unit_count]), 0], bins=np.arange(0, 1, 0.05)

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
#m1_inds = npand(region==0, cell_type=='FS')
#s1_inds = npand(region==1, cell_type=='FS')
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


##### plot selectivity histogram M1 vs S1 #####
##### plot selectivity histogram M1 vs S1 #####

bins = np.arange(0, 1, 0.05)
fig, ax = plt.subplots(1, 2)
fig.suptitle('selectivity', fontsize=20)

## RS
m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')
#hist(selectivity[npand(cell_type==unit_type, region == region[unit_count]), 0], bins=np.arange(0, 1, 0.05)

ax[0].hist(selectivity[m1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[0].hist(selectivity[s1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[0].set_title('RS units M1: {} units, S1: {} units\nno light'.format(sum(m1_inds), sum(s1_inds)))
ax[0].legend(['M1', 'S1'])

## FS
m1_inds = npand(npand(region==0, driven==True), cell_type=='FS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='FS')
#hist(selectivity[npand(cell_type==unit_type, region == region[unit_count]), 0], bins=np.arange(0, 1, 0.05)

ax[1].hist(selectivity[m1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[1].hist(selectivity[s1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[1].set_title('FS units M1: {} units, S1: {} units\nno light'.format(sum(m1_inds), sum(s1_inds)))
ax[1].legend(['M1', 'S1'])


###### Plot selectivity Scatter #####
###### Plot selectivity Scatter #####

m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')
m1_FS_inds = npand(npand(region==0, driven==True), cell_type=='FS')
s1_FS_inds = npand(npand(region==1, driven==True), cell_type=='FS')

#m1_inds = npand(region==0, cell_type=='RS')
#s1_inds = npand(region==1, cell_type=='RS')
#m1_FS_inds = npand(region==0, cell_type=='FS')
#s1_FS_inds = npand(region==1, cell_type=='FS')

fig, ax = plt.subplots(2, 2, figsize=(10,9))
fig.suptitle('selectivity paired\nRS units M1: {}, S1: {}\nFS units: M1: {}, S1: {}'\
        .format(sum(m1_inds), sum(s1_inds), sum(m1_FS_inds), sum(s1_FS_inds)), fontsize=16)

ax[0][0].plot(selectivity[m1_inds, 0], selectivity[m1_inds, 1], 'ok')
ax[0][0].set_title('M1 RS units')
ax[0][0].set_xlabel('No Light')
ax[0][0].set_ylabel('S1 Silencing')
ax[0][0].plot([0, 1], [0, 1], 'k')

ax[1][0].plot(selectivity[s1_inds, 0], selectivity[s1_inds, 2], 'or')
ax[1][0].set_title('S1 RS units')
ax[1][0].set_xlabel('No Light')
ax[1][0].set_ylabel('M1 Silencing')
ax[1][0].plot([0, 1], [0, 1], 'k')

m1_inds = npand(npand(region==0, driven==True), cell_type=='FS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='FS')
ax[0][1].plot(selectivity[m1_inds, 0], selectivity[m1_inds, 1], 'ok')
ax[0][1].set_title('M1 FS units')
ax[0][1].set_xlabel('No Light')
ax[0][1].set_ylabel('S1 Silencing')
ax[0][1].plot([0, 1], [0, 1], 'k')

ax[1][1].plot(selectivity[s1_inds, 0], selectivity[s1_inds, 2], 'or')
ax[1][1].set_title('S1 FS units')
ax[1][1].set_xlabel('No Light')
ax[1][1].set_ylabel('M1 Silencing')
ax[1][1].plot([0, 1], [0, 1], 'k')


###### Plot selectivity histogram #####
###### Plot selectivity histogram #####

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
bins = np.arange(-1, 1, 0.04)

# RS units
m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')

m1_diff = selectivity[m1_inds, 1] - selectivity[m1_inds, 0]
s1_diff = selectivity[s1_inds, 2] - selectivity[s1_inds, 0]

ax[0][0].hist(m1_diff, bins=bins)
ax[0][0].set_title('M1 RS units')
ax[0][0].set_xlabel('Change in selectivity')

ax[0][1].hist(s1_diff, bins=bins)
ax[0][1].set_title('S1 RS units')
ax[0][1].set_xlabel('Change in selectivity')

# FS units
m1_inds = npand(npand(region==0, driven==True), cell_type=='FS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='FS')

m1_diff = selectivity[m1_inds, 1] - selectivity[m1_inds, 0]
s1_diff = selectivity[s1_inds, 2] - selectivity[s1_inds, 0]

ax[1][0].hist(m1_diff, bins=bins)
ax[1][0].set_title('M1 FS units')
ax[1][0].set_xlabel('Change in selectivity')

ax[1][1].hist(s1_diff, bins=bins)
ax[1][1].set_title('S1 FS units')
ax[1][1].set_xlabel('Change in selectivity')

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
        col.vlines(0, 0, ylim_max, 'k', linestyle='dashed', linewidth=1)



##### depth analysis #####
##### depth analysis #####


###### Plot selectivity by depth

m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')

fig, ax = plt.subplots(1, 1, figsize=(8,8))
ax.plot(selectivity[m1_inds, 0], depths[m1_inds], 'ko')
ax.plot(selectivity[s1_inds, 0], depths[s1_inds], 'ro')
ax.set_ylim(0, 1100)
ax.invert_yaxis()

# plot change in selectivity by depth for M1
delta_sel = selectivity[m1_inds, 1] - selectivity[m1_inds, 0]
fig, ax = plt.subplots(1, 1, figsize=(8,8))
ax.plot(delta_sel, depths[m1_inds], 'ko')
ax.set_ylim(0, 1100)
ax.vlines(0, 0, 1100, linestyles='dashed', color='k')
ax.set_title('Change in M1 selectivity with S1 silencing')
ax.set_xlabel('change in selectivity')
ax.set_ylabel('Depth (um)')
ax.invert_yaxis()

# plot change in selectivity by depth for S1
delta_sel = selectivity[s1_inds, 2] - selectivity[s1_inds, 0]
fig, ax = plt.subplots(1, 1, figsize=(8,8))
ax.plot(delta_sel, depths[s1_inds], 'ko')
ax.set_ylim(0, 1100)
ax.vlines(0, 0, 1100, linestyles='dashed', color='k')
ax.set_title('Change in S1 selectivity with M1 silencing')
ax.set_ylabel('Depth (um)')
ax.invert_yaxis()

###### Plot preferred position by depth

m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')
fig, ax = plt.subplots(1, 1, figsize=(8,8))
ax.plot(preference[m1_inds, 0], depths[m1_inds], 'ko')
ax.plot(preference[s1_inds, 0], depths[s1_inds], 'ro')
ax.set_ylim(0, 1100)
ax.set_xlim(-2,2)

###### Plot OMI by depth
m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')
fig, ax = plt.subplots(1, 1, figsize=(8,8))
ax.plot(omi[m1_inds, 0], depths[m1_inds], 'ko')
ax.plot(omi[s1_inds, 1], depths[s1_inds], 'ro')
ax.set_ylim(0, 1100)

###### Plot spontaneous rates by depth
m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')
fig, ax = plt.subplots(1, 1, figsize=(8,8))
#ax.plot(abs_rate[m1_inds, 8, 0], depths[m1_inds], 'ko')
ax.plot(abs_rate[s1_inds, 8, 0], depths[s1_inds], 'ro')
ax.set_ylim(0, 1100)

###### Plot evoked rates by depth
m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')
fig, ax = plt.subplots(1, 1, figsize=(8,8))
ax.plot(evk_rate[m1_inds, best_pos[m1_inds], 0], depths[m1_inds], 'ko')
ax.plot(evk_rate[s1_inds, best_pos[s1_inds], 0], depths[s1_inds], 'ro')
ax.set_ylim(0, 1100)


##### plot change in firing rate by depth
m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')

# M1
m1_diff = abs_tc[m1_inds, 1, 0] - abs_tc[m1_inds, 0, 0]
fig, ax = plt.subplots(1, 1, figsize=(8,8))
ax.plot(m1_diff, depths[m1_inds], 'ko')
ax.set_ylim(0, 1100)
ax.vlines(0, 0, 1100, linestyles='dashed', color='k')
ax.set_title('Change in M1 firing rates with S1 silencing')
ax.set_ylabel('Depth (um)')
ax.set_xlabel('Change in firing rate (Hz)')
ax.invert_yaxis()

# S1
s1_diff = abs_tc[s1_inds, 2, 0] - abs_tc[s1_inds, 0, 0]
fig, ax = plt.subplots(1, 1, figsize=(8,8))
ax.plot(s1_diff, depths[s1_inds], 'ko')
ax.set_ylim(0, 1100)
ax.vlines(0, 0, 1100, linestyles='dashed', color='k')
ax.set_title('Change in S1 firing rates with M1 silencing')
ax.set_ylabel('Depth (um)')
ax.set_xlabel('Change in firing rate (Hz)')
ax.invert_yaxis()


##### preferred position analysis #####
##### preferred position analysis #####


###### Plot preferred position scatter #####
###### Plot preferred position scatter #####

m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')
m1_FS_inds = npand(npand(region==0, driven==True), cell_type=='FS')
s1_FS_inds = npand(npand(region==1, driven==True), cell_type=='FS')

fig, ax = plt.subplots(2, 2, figsize=(10,9), sharex=True, sharey=True)
fig.suptitle('preference paired\nRS units M1: {}, S1: {}\nFS units: M1: {}, S1: {}'\
        .format(sum(m1_inds), sum(s1_inds), sum(m1_FS_inds), sum(s1_FS_inds)), fontsize=16)

ax[0][0].plot(preference[m1_inds, 0], preference[m1_inds, 1], 'ok')
ax[0][0].set_title('M1 RS units')
ax[0][0].set_xlabel('No Light')
ax[0][0].set_ylabel('S1 Silencing')
#ax[0][0].plot([0, 1], [0, 1], 'k')

ax[1][0].plot(preference[s1_inds, 0], preference[s1_inds, 2], 'or')
ax[1][0].set_title('S1 RS units')
ax[1][0].set_xlabel('No Light')
ax[1][0].set_ylabel('M1 Silencing')

m1_inds = npand(npand(region==0, driven==True), cell_type=='FS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='FS')
ax[0][1].plot(preference[m1_inds, 0], preference[m1_inds, 1], 'ok')
ax[0][1].set_title('M1 FS units')
ax[0][1].set_xlabel('No Light')
ax[0][1].set_ylabel('S1 Silencing')

ax[1][1].plot(preference[s1_inds, 0], preference[s1_inds, 2], 'or')
ax[1][1].set_title('S1 FS units')
ax[1][1].set_xlabel('No Light')
ax[1][1].set_ylabel('M1 Silencing')

## set ylim to the max ylim of all subplots
ylim_max = 0
xlim_max = 0
for row in ax:
    for col in row:
        ylim_temp = np.max(np.abs(col.get_ylim()))
        xlim_temp = np.max(np.abs(col.get_xlim()))
        if ylim_temp > ylim_max:
            ylim_max = ylim_temp
        if xlim_temp > xlim_max:
            xlim_max = xlim_temp
for row in ax:
    for col in row:
#        col.set_ylim(-ylim_max, ylim_max)
#        col.set_xlim(-xlim_max, xlim_max)
        col.plot([-xlim_max, xlim_max], [-ylim_max, ylim_max], 'k')



##### OMI analysis #####
##### OMI analysis #####


##### plot OMI #####
##### plot OMI #####

## RS
m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')
bins = np.arange(-1.0, 1.0, 0.05)

fig, ax = plt.subplots(2, 2, figsize=(10,9), sharex=True, sharey=True)
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



##### Driven/evoked rate analysis #####
##### Driven/evoked rate analysis #####


##### plot change in mean absolute rates histogram #####
##### plot change in mean absolute driven rates histogram #####

fig, ax = plt.subplots(2, 2)
bins = np.arange(-20, 20, 1)

# RS units
m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')

m1_diff = abs_tc[m1_inds, 1, 0] - abs_tc[m1_inds, 0, 0]
s1_diff = abs_tc[s1_inds, 2, 0] - abs_tc[s1_inds, 0, 0]

ax[0][0].hist(m1_diff, bins=bins)
ax[0][0].set_title('M1 RS units')
ax[0][0].set_xlabel('Change in mean rate')

ax[0][1].hist(s1_diff, bins=bins)
ax[0][1].set_title('S1 RS units')
ax[0][1].set_xlabel('Change in mean rate')

# FS units
m1_inds = npand(npand(region==0, driven==True), cell_type=='FS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='FS')

m1_diff = abs_tc[m1_inds, 1, 0] - abs_tc[m1_inds, 0, 0]
s1_diff = abs_tc[s1_inds, 2, 0] - abs_tc[s1_inds, 0, 0]

ax[1][0].hist(m1_diff, bins=bins)
ax[1][0].set_title('M1 FS units')
ax[1][0].set_xlabel('Change in mean rate')

ax[1][1].hist(s1_diff, bins=bins)
ax[1][1].set_title('S1 FS units')
ax[1][1].set_xlabel('Change in mean rate')

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
        col.vlines(0, 0, ylim_max, 'k', linestyle='dashed', linewidth=1)
##### plot driven rates best position #####
##### plot driven rates best position #####

## RS top left
m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')
m1_best_pos = best_pos[m1_inds].astype(int)
s1_best_pos = best_pos[s1_inds].astype(int)

fig, ax = plt.subplots(2, 2, figsize=(10,9), sharex=True, sharey=True)
fig.suptitle('Driven Rates', fontsize=20)
ax[0][0].errorbar(evk_rate[m1_inds, m1_best_pos, 0], evk_rate[m1_inds, m1_best_pos+9, 0], \
        xerr=evk_rate[m1_inds, m1_best_pos, 1], yerr=evk_rate[m1_inds, m1_best_pos+9, 1], c='k', fmt='o', ecolor='k')
ax[0][0].errorbar(evk_rate[s1_inds, s1_best_pos, 0], evk_rate[s1_inds, s1_best_pos+9, 0], \
        xerr=evk_rate[s1_inds, s1_best_pos, 1], yerr=evk_rate[s1_inds, s1_best_pos+9, 1], c='r', fmt='o', ecolor='r')
ax[0][0].set_title('RS units M1: {} units, S1: {} units, \nS1 light'.format(sum(m1_inds), sum(s1_inds)))
ax[0][0].set_ylabel('Light On\nfiring rate (Hz)')

## RS bottom left
ax[1][0].errorbar(evk_rate[m1_inds, m1_best_pos, 0], evk_rate[m1_inds, m1_best_pos+9+9, 0], \
        xerr=evk_rate[m1_inds, m1_best_pos, 1], yerr=evk_rate[m1_inds, m1_best_pos+9+9, 1], c='k', fmt='o', ecolor='k')
ax[1][0].errorbar(evk_rate[s1_inds, s1_best_pos, 0], evk_rate[s1_inds, s1_best_pos+9+9, 0], \
        xerr=evk_rate[s1_inds, s1_best_pos, 1], yerr=evk_rate[s1_inds, s1_best_pos+9+9, 1], c='r', fmt='o', ecolor='r')
ax[1][0].set_title('M1 light'.format(sum(m1_inds), sum(s1_inds)))
ax[1][0].set_ylabel('Light On\nfiring rate (Hz)')
ax[1][0].set_xlabel('Light Off\nfiring rate (Hz)')

## FS top middle
m1_inds = npand(npand(region==0, driven==True), cell_type=='FS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='FS')
m1_best_pos = best_pos[m1_inds].astype(int)
s1_best_pos = best_pos[s1_inds].astype(int)
ax[0][1].errorbar(evk_rate[m1_inds, m1_best_pos, 0], evk_rate[m1_inds, m1_best_pos+9, 0], \
        xerr=evk_rate[m1_inds, m1_best_pos, 1], yerr=evk_rate[m1_inds, m1_best_pos+9, 1], c='k', fmt='o', ecolor='k')
ax[0][1].errorbar(evk_rate[s1_inds, s1_best_pos, 0], evk_rate[s1_inds, s1_best_pos+9, 0], \
        xerr=evk_rate[s1_inds, s1_best_pos, 1], yerr=evk_rate[s1_inds, s1_best_pos+9, 1], c='r', fmt='o', ecolor='r')
ax[0][1].set_title('FS units M1: {} units, S1: {} units, \nS1 light'.format(sum(m1_inds), sum(s1_inds)))

## FS bottom middle
ax[1][1].errorbar(evk_rate[m1_inds, m1_best_pos, 0], evk_rate[m1_inds, m1_best_pos+9+9, 0], \
        xerr=evk_rate[m1_inds, m1_best_pos, 1], yerr=evk_rate[m1_inds, m1_best_pos+9+9, 1], c='k', fmt='o', ecolor='k')
ax[1][1].errorbar(evk_rate[s1_inds, s1_best_pos, 0], evk_rate[s1_inds, s1_best_pos+9+9, 0], \
        xerr=evk_rate[s1_inds, s1_best_pos, 1], yerr=evk_rate[s1_inds, s1_best_pos+9+9, 1], c='r', fmt='o', ecolor='r')
ax[1][1].set_title('M1 light'.format(sum(m1_inds), sum(s1_inds)))
ax[1][1].set_xlabel('Light Off\nfiring rate (Hz)')

##### THIS ONLY WORKS WITH SHARED X AND Y AXES! #####
xylim_max = 0
for row in ax:
    for col in row:
        ylim_temp = np.max(np.abs(col.get_ylim()))
        xlim_temp = np.max(np.abs(col.get_xlim()))
        xylim_temp = np.max([xlim_temp, ylim_temp])
        if xylim_temp > xylim_max:
            xylim_max = xylim_temp
for row in ax:
    for col in row:
#        col.set_ylim(-ylim_max, ylim_max)
#        col.set_xlim(-xlim_max, xlim_max)
        col.plot([-xylim_max, xylim_max], [-xylim_max, xylim_max], 'k')



##### Absolute firing rate analysis #####
##### Absolute firing rate analysis #####


#### NOTE THIS IS THE MAIN RESULT FIGURE NOTE ####
##### plot absolute firing rates best position #####
##### plot absolute firing rates best position #####
#### NOTE THIS IS THE MAIN RESULT FIGURE NOTE ####

## RS top left
m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')
m1_best_pos = best_pos[m1_inds].astype(int)
s1_best_pos = best_pos[s1_inds].astype(int)

fig, ax = plt.subplots(2, 2, figsize=(10,9), sharex=True, sharey=True)
fig.suptitle('Mean Absolute Firing Rates (best pos)', fontsize=20)
ax[0][0].errorbar(abs_rate[m1_inds, m1_best_pos, 0], abs_rate[m1_inds, m1_best_pos+9, 0], \
        xerr=abs_rate[m1_inds, m1_best_pos, 1], yerr=abs_rate[m1_inds, m1_best_pos+9, 1], c='k', fmt='o', ecolor='k')
ax[0][0].errorbar(abs_rate[s1_inds, s1_best_pos, 0], abs_rate[s1_inds, s1_best_pos+9, 0], \
        xerr=abs_rate[s1_inds, s1_best_pos, 1], yerr=abs_rate[s1_inds, s1_best_pos+9, 1], c='r', fmt='o', ecolor='r')
ax[0][0].set_title('RS units M1: {} units, S1: {} units, \nS1 light'.format(sum(m1_inds), sum(s1_inds)))
ax[0][0].set_ylabel('Light On\nfiring rate (Hz)')
# set log scale
#ax[0][0].set_yscale('log')
#ax[0][0].set_xscale('log')

## RS bottom left
ax[1][0].errorbar(abs_rate[m1_inds, m1_best_pos, 0], abs_rate[m1_inds, m1_best_pos+9+9, 0], \
        xerr=abs_rate[m1_inds, m1_best_pos, 1], yerr=abs_rate[m1_inds, m1_best_pos+9+9, 1], c='k', fmt='o', ecolor='k')
ax[1][0].errorbar(abs_rate[s1_inds, s1_best_pos, 0], abs_rate[s1_inds, s1_best_pos+9+9, 0], \
        xerr=abs_rate[s1_inds, s1_best_pos, 1], yerr=abs_rate[s1_inds, s1_best_pos+9+9, 1], c='r', fmt='o', ecolor='r')
ax[1][0].set_title('M1 light'.format(sum(m1_inds), sum(s1_inds)))
ax[1][0].set_ylabel('Light On\nfiring rate (Hz)')
ax[1][0].set_xlabel('Light Off\nfiring rate (Hz)')
# set log scale
#ax[1][0].set_yscale('log')
#ax[1][0].set_xscale('log')

## FS top middle
m1_inds = npand(npand(region==0, driven==True), cell_type=='FS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='FS')
m1_best_pos = best_pos[m1_inds].astype(int)
s1_best_pos = best_pos[s1_inds].astype(int)
ax[0][1].errorbar(abs_rate[m1_inds, m1_best_pos, 0], abs_rate[m1_inds, m1_best_pos+9, 0], \
        xerr=abs_rate[m1_inds, m1_best_pos, 1], yerr=abs_rate[m1_inds, m1_best_pos+9, 1], c='k', fmt='o', ecolor='k')
ax[0][1].errorbar(abs_rate[s1_inds, s1_best_pos, 0], abs_rate[s1_inds, s1_best_pos+9, 0], \
        xerr=abs_rate[s1_inds, s1_best_pos, 1], yerr=abs_rate[s1_inds, s1_best_pos+9, 1], c='r', fmt='o', ecolor='r')
ax[0][1].set_title('FS units M1: {} units, S1: {} units, \nS1 light'.format(sum(m1_inds), sum(s1_inds)))
# set log scale
#ax[0][1].set_yscale('log')
#ax[0][1].set_xscale('log')

## FS bottom middle
ax[1][1].errorbar(abs_rate[m1_inds, m1_best_pos, 0], abs_rate[m1_inds, m1_best_pos+9+9, 0], \
        xerr=abs_rate[m1_inds, m1_best_pos, 1], yerr=abs_rate[m1_inds, m1_best_pos+9+9, 1], c='k', fmt='o', ecolor='k')
ax[1][1].errorbar(abs_rate[s1_inds, s1_best_pos, 0], abs_rate[s1_inds, s1_best_pos+9+9, 0], \
        xerr=abs_rate[s1_inds, s1_best_pos, 1], yerr=abs_rate[s1_inds, s1_best_pos+9+9, 1], c='r', fmt='o', ecolor='r')
ax[1][1].set_title('M1 light'.format(sum(m1_inds), sum(s1_inds)))
ax[1][1].set_xlabel('Light Off\nfiring rate (Hz)')
# set log scale
#ax[1][1].set_yscale('log')
#ax[1][1].set_xscale('log')

##### THIS ONLY WORKS WITH SHARED X AND Y AXES! #####
xylim_max = 0
for row in ax:
    for col in row:
        ylim_temp = np.max(np.abs(col.get_ylim()))
        xlim_temp = np.max(np.abs(col.get_xlim()))
        xylim_temp = np.max([xlim_temp, ylim_temp])
        if xylim_temp > xylim_max:
            xylim_max = xylim_temp
for row in ax:
    for col in row:
#        col.set_ylim(-ylim_max, ylim_max)
#        col.set_xlim(-xlim_max, xlim_max)
        col.plot([0, xylim_max], [0, xylim_max], 'k')


##### plot absolute firing rates mean tuning curve #####
##### plot absolute firing rates mean tuning curve #####

## RS top left
m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')

fig, ax = plt.subplots(2, 2, figsize=(10,9), sharex=True, sharey=True)
fig.suptitle('Mean Absolute Firing Rates', fontsize=20)
ax[0][0].errorbar(abs_tc[m1_inds, 0, 0], abs_tc[m1_inds, 1, 0], \
        xerr=abs_tc[m1_inds, 0, 1], yerr=abs_tc[m1_inds, 1, 1], c='k', fmt='o', ecolor='k')
ax[0][0].errorbar(abs_tc[s1_inds, 0, 0], abs_tc[s1_inds, 1, 0], \
        xerr=abs_tc[s1_inds, 0, 1], yerr=abs_tc[s1_inds, 1, 1], c='r', fmt='o', ecolor='r')
ax[0][0].set_title('RS units M1: {} units, S1: {} units, \nS1 light'.format(sum(m1_inds), sum(s1_inds)))
ax[0][0].set_ylabel('Light On\nfiring rate (Hz)')
# set log scale
#ax[0][0].set_yscale('log')
#ax[0][0].set_xscale('log')

## RS bottom left
ax[1][0].errorbar(abs_tc[m1_inds, 0, 0], abs_tc[m1_inds, 2, 0], \
        xerr=abs_tc[m1_inds, 0, 1], yerr=abs_tc[m1_inds, 2, 1], c='k', fmt='o', ecolor='k')
ax[1][0].errorbar(abs_tc[s1_inds, 0, 0], abs_tc[s1_inds, 2, 0], \
        xerr=abs_tc[s1_inds, 0, 1], yerr=abs_tc[s1_inds, 2, 1], c='r', fmt='o', ecolor='r')
ax[1][0].set_title('M1 light'.format(sum(m1_inds), sum(s1_inds)))
ax[1][0].set_ylabel('Light On\nfiring rate (Hz)')
ax[1][0].set_xlabel('Light Off\nfiring rate (Hz)')
# set log scale
#ax[1][0].set_yscale('log')
#ax[1][0].set_xscale('log')

## FS top middle
m1_inds = npand(npand(region==0, driven==True), cell_type=='FS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='FS')
ax[0][1].errorbar(abs_tc[m1_inds, 0, 0], abs_tc[m1_inds, 1, 0], \
        xerr=abs_tc[m1_inds, 0, 1], yerr=abs_tc[m1_inds, 1, 1], c='k', fmt='o', ecolor='k')
ax[0][1].errorbar(abs_tc[s1_inds, 0, 0], abs_tc[s1_inds, 1, 0], \
        xerr=abs_tc[s1_inds, 0, 1], yerr=abs_tc[s1_inds, 1, 1], c='r', fmt='o', ecolor='r')
ax[0][1].set_title('FS units M1: {} units, S1: {} units, \nS1 light'.format(sum(m1_inds), sum(s1_inds)))
# set log scale
#ax[0][1].set_yscale('log')
#ax[0][1].set_xscale('log')

## FS bottom middle
ax[1][1].errorbar(abs_tc[m1_inds, 0, 0], abs_tc[m1_inds, 2, 0], \
        xerr=abs_tc[m1_inds, 0, 1], yerr=abs_tc[m1_inds, 2, 1], c='k', fmt='o', ecolor='k')
ax[1][1].errorbar(abs_tc[s1_inds, 0, 0], abs_tc[s1_inds, 2, 0], \
        xerr=abs_tc[s1_inds, 0, 1], yerr=abs_tc[s1_inds, 2, 1], c='r', fmt='o', ecolor='r')
ax[1][1].set_title('M1 light'.format(sum(m1_inds), sum(s1_inds)))
ax[1][1].set_xlabel('Light Off\nfiring rate (Hz)')
# set log scale
#ax[1][1].set_yscale('log')
#ax[1][1].set_xscale('log')

##### THIS ONLY WORKS WITH SHARED X AND Y AXES! #####
xylim_max = 0
for row in ax:
    for col in row:
        ylim_temp = np.max(np.abs(col.get_ylim()))
        xlim_temp = np.max(np.abs(col.get_xlim()))
        xylim_temp = np.max([xlim_temp, ylim_temp])
        if xylim_temp > xylim_max:
            xylim_max = xylim_temp
for row in ax:
    for col in row:
#        col.set_ylim(-ylim_max, ylim_max)
#        col.set_xlim(-xlim_max, xlim_max)
        col.plot([0, xylim_max], [0, xylim_max], 'k')


##### plot change in mean absolute rates histogram best position #####
##### plot change in mean absolute rates histogram best position #####
#### NOTE THIS MAY BE AN IMPORTANT IDEA TO REVISIT NOTE ####

fig, ax = plt.subplots(2, 2)
fig.suptitle('Change in firing rate (best pos)', fontsize=16)

bins = np.arange(-20, 20, 1)

# RS units
m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')

m1_diff = abs_rate[m1_inds, best_pos[m1_inds]+9, 0] - abs_rate[m1_inds, best_pos[m1_inds], 0]
s1_diff = abs_rate[s1_inds, best_pos[s1_inds]+9+9, 0] - abs_rate[s1_inds, best_pos[s1_inds], 0]

ax[0][0].hist(m1_diff, bins=bins)
ax[0][0].set_title('M1 RS units')
ax[0][0].set_xlabel('Change in mean rate')

ax[0][1].hist(s1_diff, bins=bins)
ax[0][1].set_title('S1 RS units')
ax[0][1].set_xlabel('Change in mean rate')

# FS units
m1_inds = npand(npand(region==0, driven==True), cell_type=='FS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='FS')

m1_diff = abs_rate[m1_inds, best_pos[m1_inds]+9, 0] - abs_rate[m1_inds, best_pos[m1_inds], 0]
s1_diff = abs_rate[s1_inds, best_pos[s1_inds]+9+9, 0] - abs_rate[s1_inds, best_pos[s1_inds], 0]

ax[1][0].hist(m1_diff, bins=bins)
ax[1][0].set_title('M1 FS units')
ax[1][0].set_xlabel('Change in mean rate')

ax[1][1].hist(s1_diff, bins=bins)
ax[1][1].set_title('S1 FS units')
ax[1][1].set_xlabel('Change in mean rate')

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
        col.vlines(0, 0, ylim_max, 'k', linestyle='dashed', linewidth=1)


##### plot change in mean absolute rates histogram #####
##### plot change in mean absolute rates histogram #####

fig, ax = plt.subplots(2, 2, sharex=True)
fig.suptitle('Change in firing rate (mean tc)', fontsize=16)

bins = np.arange(-20, 20, 1)

# RS units
m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')

m1_diff = abs_tc[m1_inds, 1, 0] - abs_tc[m1_inds, 0, 0]
s1_diff = abs_tc[s1_inds, 2, 0] - abs_tc[s1_inds, 0, 0]

ax[0][0].hist(m1_diff, bins=bins)
ax[0][0].set_title('M1 RS units')
ax[0][0].set_xlabel('Change in mean rate')

ax[0][1].hist(s1_diff, bins=bins)
ax[0][1].set_title('S1 RS units')
ax[0][1].set_xlabel('Change in mean rate')

# FS units
m1_inds = npand(npand(region==0, driven==True), cell_type=='FS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='FS')

m1_diff = abs_tc[m1_inds, 1, 0] - abs_tc[m1_inds, 0, 0]
s1_diff = abs_tc[s1_inds, 2, 0] - abs_tc[s1_inds, 0, 0]

ax[1][0].hist(m1_diff, bins=bins)
ax[1][0].set_title('M1 FS units')
ax[1][0].set_xlabel('Change in mean rate')

ax[1][1].hist(s1_diff, bins=bins)
ax[1][1].set_title('S1 FS units')
ax[1][1].set_xlabel('Change in mean rate')

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
        col.vlines(0, 0, ylim_max, 'k', linestyle='dashed', linewidth=1)


##### gain/offset analysis of absolute rates #####
##### gain/offset analysis of absolute rates #####

## RS top left
m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')

m1_nolight = abs_tc[m1_inds, 0, 0]
m1_s1light = abs_tc[m1_inds, 1, 0]
m1slope, m1intercept, m1r_value, m1p_value, m1std_err = sp.stats.linregress(m1_nolight, m1_s1light)

s1_nolight = abs_tc[s1_inds, 0, 0]
s1_m1light = abs_tc[s1_inds, 2, 0]
s1slope, s1intercept, s1r_value, s1p_value, s1std_err = sp.stats.linregress(s1_nolight, s1_m1light)


numerator = s1slope - m1slope
denominator = pow( ( pow(s1std_err, 2) + pow(m1std_err, 2)), 1/2.)
z = numerator / denominator
p_values = sp.stats.norm.sf(abs(z))*2 #twosided



##### Spontaneous/baseline rate analysis #####
##### Spontaneous/baseline rate analysis #####


##### plot spontaneous/baseline rates #####
##### plot spontaneous/baseline rates #####

fig, ax = plt.subplots(2, 2, figsize=(10,9), sharex=True, sharey=True)
fig.suptitle('Spontaneous Rates', fontsize=20)

## RS
m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')

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


##### THIS ONLY WORKS WITH SHARED X AND Y AXES! #####
ylim_max = 0
xlim_max = 0
for row in ax:
    for col in row:
        ylim_temp = np.max(np.abs(col.get_ylim()))
        xlim_temp = np.max(np.abs(col.get_xlim()))
        if ylim_temp > ylim_max:
            ylim_max = ylim_temp
        if xlim_temp > xlim_max:
            xlim_max = xlim_temp
for row in ax:
    for col in row:
#        col.set_ylim(-ylim_max, ylim_max)
#        col.set_xlim(-xlim_max, xlim_max)
        col.plot([-xlim_max, xlim_max], [-ylim_max, ylim_max], 'k')


##### plot change in mean spontaneous rates histogram #####
##### plot change in mean spontaneous rates histogram #####

fig, ax = plt.subplots(2, 2)
fig.suptitle('Change in baseline firing rate', fontsize=16)

bins = np.arange(-20, 20, 1)

# RS units
m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')

m1_diff = abs_rate[m1_inds, neuro.control_pos-1+9, 0] - abs_rate[m1_inds, neuro.control_pos-1, 0]
s1_diff = abs_rate[s1_inds, neuro.control_pos-1+9+9, 0] - abs_rate[s1_inds, neuro.control_pos-1, 0]

ax[0][0].hist(m1_diff, bins=bins)
ax[0][0].set_title('M1 RS units')
ax[0][0].set_xlabel('Change in baseline rate')

ax[0][1].hist(s1_diff, bins=bins)
ax[0][1].set_title('S1 RS units')
ax[0][1].set_xlabel('Change in baseline rate')

# FS units
m1_inds = npand(npand(region==0, driven==True), cell_type=='FS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='FS')

m1_diff = abs_rate[m1_inds, neuro.control_pos -1+9, 0] - abs_rate[m1_inds, neuro.control_pos -1, 0]
s1_diff = abs_rate[s1_inds, neuro.control_pos -1+9+9, 0] - abs_rate[s1_inds, neuro.control_pos -1, 0]

ax[1][0].hist(m1_diff, bins=bins)
ax[1][0].set_title('M1 FS units')
ax[1][0].set_xlabel('Change in baseline rate')

ax[1][1].hist(s1_diff, bins=bins)
ax[1][1].set_title('S1 FS units')
ax[1][1].set_xlabel('Change in baseline rate')

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
        col.vlines(0, 0, ylim_max, 'k', linestyle='dashed', linewidth=1)



##### Firing rate vs stimulus strength analysis #####
##### Firing rate vs stimulus strength analysis #####


##### plot FR vs strength for all units #####
##### plot FR vs strength for all units #####

m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')

# M1 mean + sem sorted tuning curves No light
m1_mean_nolight = mean(meanr[m1_inds, :, 0], axis=0)
m1_sem_nolight  = sp.stats.sem(meanr[m1_inds, :, 0], axis=0)

# M1 mean + sem sorted tuning curves S1 light
m1_mean_s1light = mean(meanr[m1_inds, :, 1], axis=0)
m1_sem_s1light  = sp.stats.sem(meanr[m1_inds, :, 1], axis=0)

# S1 mean + sem sorted tuning curves No light
s1_mean_nolight = mean(meanr[s1_inds, :, 0], axis=0)
s1_sem_nolight  = sp.stats.sem(meanr[s1_inds, :, 0], axis=0)

# S1 mean + sem sorted tuning curves m1 light
s1_mean_m1light = mean(meanr[s1_inds, :, 2], axis=0)
s1_sem_m1light  = sp.stats.sem(meanr[s1_inds, :, 1], axis=0)

fig, ax = plt.subplots(1, 2)
fig.suptitle('Firing rate vs stimulus strength')

pos = np.arange(9)

# M1 nolight
ax[0].plot(pos, m1_mean_nolight, color='k')
ax[0].fill_between(pos, m1_mean_nolight - m1_sem_nolight, m1_mean_nolight + m1_sem_nolight, facecolor='k', alpha=0.3)
# M1 s1light
ax[0].plot(pos, m1_mean_s1light, color='r')
ax[0].fill_between(pos, m1_mean_s1light - m1_sem_s1light, m1_mean_s1light + m1_sem_s1light, facecolor='r', alpha=0.3)
ax[0].set_xlabel('stimulus strength')
ax[0].set_ylim(0.35, 1.05)
ax[0].set_title('M1 RS units')

# S1 nolight
ax[1].plot(pos, s1_mean_nolight, color='k')
ax[1].fill_between(pos, s1_mean_nolight - s1_sem_nolight, s1_mean_nolight + s1_sem_nolight, facecolor='k', alpha=0.3)
# S1 s1light
ax[1].plot(pos, s1_mean_m1light, color='b')
ax[1].fill_between(pos, s1_mean_m1light - s1_sem_m1light, s1_mean_m1light + s1_sem_m1light, facecolor='b', alpha=0.3)
ax[1].set_xlabel('stimulus strength')
ax[1].set_ylim(0.35, 1.05)
ax[1].set_title('S1 RS units')







##### plot change from no manipulation of FR vs strength tuning curves #####
##### plot change from no manipulation of FR vs strength tuning curves #####

m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')

# M1 diff
m1_diff = meanr[m1_inds, :, 1] - meanr[m1_inds, :, 0]
m1mean  = np.mean(m1_diff, axis=0)
m1sem   = sp.stats.sem(m1_diff, axis=0)

# M1 diff
s1_diff = meanr[s1_inds, :, 2] - meanr[s1_inds, :, 0]
s1mean  = np.mean(s1_diff, axis=0)
s1sem   = sp.stats.sem(s1_diff, axis=0)

fig, ax = plt.subplots(1, 1)
pos = np.arange(9)

# M1
ax.plot(pos, m1mean, color='r')
ax.fill_between(pos, m1mean- m1sem, m1mean+ m1sem, facecolor='r', alpha=0.3)

# S1
ax.plot(pos, s1mean, color='b')
ax.fill_between(pos, s1mean- s1sem, s1mean+ s1sem, facecolor='b', alpha=0.3)

ax.hlines(0, pos[0], pos[-1], linestyle='dashed', color='k')
ax.set_ylabel('Change in normalized firing rate')
ax.set_xlabel('stimulus strength')
ax.set_title('Change in spike rate with light and stimulus strength')


##### Burst rate analysis #####
##### Burst rate analysis #####


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
unit_type  = 'RS'
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





# M1 analysis
m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')

# absolute firing rates
# how fast do M1 units fire and how much do they change with touch?
bins = np.arange(0, 40, 2)
g = sns.jointplot(abs_rate[s1_inds, 8, 0], abs_rate[s1_inds, best_pos[s1_inds], 0],\
        size=5, ratio=3,\
        marginal_kws=dict(bins=bins, rug=True))
g.ax_joint.set_xlim(0, 50)
g.ax_joint.set_ylim(0, 50)
g.ax_joint.plot([0,50], [0, 50], '--k')

z, p = sp.stats.wilcoxon(abs_rate[s1_inds, 8, 0], abs_rate[s1_inds, best_pos[s1_inds], 0])

# evoked rate at best position histogram
# how much do units change their firing rates?

bins = np.arange(0, 40, 2)
fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios":(0.15, 0.85)})
sns.boxplot(evk_rate[s1_inds, best_pos[s1_inds], 0], ax=ax[0])
ax[1].hist(evk_rate[s1_inds, best_pos[s1_inds], 0], normed_hist=True, bins=bins)

sns.despine(ax=ax[0], left=True)
sns.despine(ax=ax[1])

ax[1].set_xlabel('Evoked firing rate (Hz)')
ax[1].set_ylabel('Probability density')
fig.suptitle('Evoked firing rate at best positions')



# absolute and evoked rates with silencing
# how does silencing S1 change the firing rates in M1?
pos_inds = best_pos[s1_inds]

fig, ax = plt.subplots(1, 2, figsize=(10,4))
fig.suptitle("Best contact position\nM1 RS unit's firing rates")
ax[0].scatter(abs_rate[s1_inds, pos_inds, 0], abs_rate[s1_inds, pos_inds+9+9, 0])
ax[0].set_xlim(0, 50)
ax[0].set_ylim(0, 50)
ax[0].plot([0, 50], [0, 50], '--k')
ax[0].set_title('M1 absolute rates')
ax[0].set_xlabel('No light')
ax[0].set_ylabel('S1 silencing')

ax[1].scatter(evk_rate[s1_inds, pos_inds, 0], evk_rate[s1_inds, pos_inds+9+9, 0])
ax[1].set_xlim(0, 30)
ax[1].set_ylim(0, 30)
ax[1].plot([0, 30], [0, 30], '--k')
ax[1].set_title('M1 evoked rates')
ax[1].set_xlabel('No light')
ax[1].set_ylabel('S1 silencing')


z, p = sp.stats.wilcoxon(evk_rate[s1_inds, pos_inds, 0], evk_rate[s1_inds, pos_inds+9+9, 0])


# histogram of change in firing rates
delta = evk_rate[s1_inds, pos_inds+9+9, 0] - evk_rate[s1_inds, pos_inds, 0]
bins = np.arange(-20, 0, 2)
fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios":(0.15, 0.85)})

sns.boxplot(delta, ax=ax[0])
ax[1].hist(delta, bins=bins, color='b', normed=True)
sns.despine(ax=ax[0], left=True)
sns.despine(ax=ax[1])



m1_nolight = np.asarray(evk_rate[m1_inds, best_pos[m1_inds], 0])
m1_s1light = np.asarray(evk_rate[m1_inds, best_pos[m1_inds]+9, 0])
s1_nolight = np.asarray(evk_rate[s1_inds, best_pos[s1_inds], 0])
s1_m1light = np.asarray(evk_rate[s1_inds, best_pos[s1_inds]+9+9, 0])


mean_m1 = [np.mean(m1_nolight), np.mean(s1_nolight)]
mean_s1 = [np.mean(m1_s1light), np.mean(s1_m1light)]
sem_m1  = [sp.stats.sem(m1_nolight), sp.stats.sem(s1_nolight)]
sem_s1  = [sp.stats.sem(m1_s1light), sp.stats.sem(s1_m1light)]

fig, ax = plt.subplots()

n_groups = 2
index = np.arange(n_groups)
bar_width = 0.35

opacity = 0.7
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index, mean_m1, bar_width,
    alpha=opacity,
    color='b',
    yerr=sem_m1,
    error_kw=error_config,
    label='M1')

rects2 = plt.bar(index + bar_width, mean_s1, bar_width,
    alpha=opacity,
    color='b',
    yerr=sem_s1,
    error_kw=error_config,
    label='Women')















