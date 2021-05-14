import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import scipy.io as sio
import statsmodels.stats.multitest as smm

# This is the main place where I will explore and prepare figures for the
# 8-pos-ephys experiment. Much of the population code will be copied over from
# previous scripts and cleaned up so they are easy to read and reliably
# recreate any figure at a moments notice. The first half of the script will be
# dedicated to single unit examples and possibly single trial whisker tracking
# figures as well. The second half will be the population analysis, unless this
# script gets out of hand, then the population analysis code will be moved to
# its own file.
#
# As of now the standard figure size will be 6in wide x 4in height
# Font will be Arial
# Font size for figure titles, axis labels, and tick markers will be 16pt,
# 14pt, and 12pt respectively.
#
# G.Telian 01/18/2021

# how to do multiple comparisons
#rej_s1, pval_corr = smm.multipletests(raw_p_vals, alpha=0.05, method='sh')[:2]

# change default figure type to PDF
mpl.rcParams['savefig.format'] = 'pdf'
#plt.rc('font',family='Arial')
#sns.set_style("whitegrid", {'axes.grid' : False})

with PdfPages('01_all_driven_units_raster_tuningcurves.pdf') as pdf:
    for exp_num, n in enumerate(experiments):
        n.get_sensory_drive(num_sig_pos=True)
        for unit_num in range(n.num_units):
            if n.driven_units[unit_num]:
                if n.shank_ids[unit_num]:
                    region = 'S1'
                else:
                    region = 'M1'
                sel = n.selectivity[unit_num, :]

                fig, ax = plt.subplots(1, 3, figsize=(11,3))
                fig.subplots_adjust(left=0.125, bottom=0.155, right=0.9, top=0.88, wspace=0.2, hspace=0.2)
                fig.suptitle('Exp#: {}, unit#: {}, region: {}, selectivity {:.3f}, Num evk pos: {}'.format(\
                        exp_num, unit_num, region, sel[0], n.num_driven_pos[unit_num]))
                n.plot_raster_all_conditions(unit_ind=unit_num, axis=ax[0])
                n.plot_tuning_curve(unit_ind=unit_num, kind='abs_rate', axis=ax[1])
                n.plot_tuning_curve(unit_ind=unit_num, kind='evk_rate', axis=ax[2])
                ax[2].set_title('selectivty vals {:.3f}, {:.3f}, {:.3f}'.format(sel[0], sel[1], sel[2]))


                pdf.savefig()
                fig.clear()
                plt.close()



##### ##### ##### NOTE single unit analysis NOTE ##### ##### #####
##### ##### ##### NOTE single unit analysis NOTE ##### ##### #####

#### FIGURE 2 classic single unit spike analysis ####
#### FIGURE 2 classic single unit spike analysis ####

# FID1345 will be used as a source of example figures
# currently:
#   M1 units /21, /8, /0 (classic shape). /21 and /1 (oddity / S1 light increase FR)
#   S1 units /44, /33 (classic), /32 (offset with center near back object
#   positions), 35 (peaky in center and it increases FR with M1 silencig at all
#   positions??? near back)

# NOTE: The figure size is limited by my monitor size. If I need larger figures
# I can switch to a non-interactive matplotlib backend such as Agg. This will
# allow me to make any kind of plot I want and save it using fig.savefig(...)
# This may be good for itteratively remaking figures without having to manually
# save them. It would also be benificial if I need to make a plot with a height
# greater than 7ish-inches (limite of my screen right now).

save_dir = '/home/greg/Desktop/desktop2dropbox/8pos_figures/fig2_basic_spike_data/'
uind = 10
cell_type = neuro.cell_type[uind].lower()
region    = neuro.region_dict[neuro.shank_ids[uind]].lower()
trial_ind = 2 # change this after looking at the Raster and Tuning Curve plots

# RASTER all conditions
fig1, ax1 = plt.subplots(1, figsize=(6,4))
neuro.plot_raster_all_conditions(unit_ind=uind, num_trials=20, offset=7)
ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Subset of all trials')
ax1.set_title('{} {} Raster: subset of all trials'.format(region, cell_type.upper()))

# Tuning curve
fig2, ax2 = plt.subplots(1, figsize=(6,4))
neuro.plot_tuning_curve(unit_ind=uind, axis=ax2)
ax2.set_xlabel('Bar position'); ax2.set_ylabel('Firing rate (Hz)')
ax2.set_title('{} {} Tuning curve'.format(region, cell_type.upper()))

# PSTH + RASTER ?
fig3, ax3 = plt.subplots(1, figsize=(6,4))
neuro.plot_psth(unit_ind=uind, trial_type=trial_ind, color='k',error='sem')
neuro.plot_psth(unit_ind=uind, trial_type=trial_ind+9, color='r',error='sem')
neuro.plot_psth(unit_ind=uind, trial_type=trial_ind+9+9, color='b',error='sem')
ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Firing rate (Hz)')
ax3.set_title('{} {} PSTH for position {}'.format(region, cell_type.upper(), trial_ind))

fig4, ax4 = plt.subplots(3, figsize=(6, 6.85))
fig4.subplots_adjust(top=0.88, bottom=0.08, left=0.125, right=0.9, hspace=0.4, wspace=0.2)
neuro.plot_raster(unit_ind=uind, trial_type=trial_ind,     axis=ax4 [2], stim_choice=False)
neuro.plot_raster(unit_ind=uind, trial_type=trial_ind+9,   axis=ax4 [1], stim_choice=True)
neuro.plot_raster(unit_ind=uind, trial_type=trial_ind+9+9, axis=ax4 [0], stim_choice=True)

ax4 [2].set_xlabel('Time (s)'); ax4 [2].set_title('No light')
ax4 [2].set_ylabel('Trials'); ax4 [1].set_ylabel('Trials'); ax4 [0].set_ylabel('Trials')
ax4 [1].set_title('vS1 silencing')
ax4 [0].set_title('vM1 silencing')
fig4.suptitle('{} {} Rasters for position {}'.format(region, cell_type.upper(), trial_ind), size=18)


## save all figures as PDFs
# file names should use this standard naming scheme
# I messed up, want M1 and S1 units to be enar each other
# {fid###} _ u{uind} _ {region} _ {cell type} _ {figure_type} _ {maybe stim number} .pdf
# New naming scheme
# {fid###} _ {region} _ {cell type} _ u{uind} _ {figure_type} _ {maybe stim number} .pdf

fname1 = '{}_{}_{}_u{:02d}_rasterALL.pdf'.format(fid.lower(), region, cell_type, uind)
fraster = os.path.join(save_dir, fname1)
fname2 = '{}_{}_{}_u{:02d}_tuning_curve.pdf'.format(fid.lower(), region, cell_type, uind)
ftuning = os.path.join(save_dir, fname2)
fname3 = '{}_{}_{}_u{:02d}_psth_stim{:02d}.pdf'.format(fid.lower(), region, cell_type, uind, trial_ind)
fpsth = os.path.join(save_dir, fname3)
fname4 = '{}_{}_{}_u{:02d}_psthRasters_stim{:02d}.pdf'.format(fid.lower(), region, cell_type, uind, trial_ind)
fpsthRaster = os.path.join(save_dir, fname4)

fig1.savefig(fraster, format='pdf')
fig2.savefig(ftuning, format='pdf')
fig3.savefig(fpsth, format='pdf')
fig4.savefig(fpsthRaster, format='pdf')

##### ##### ##### NOTE population analysis NOTE ##### ##### #####
##### ##### ##### NOTE population analysis NOTE ##### ##### #####

fids = ['1295', '1302', '1318', '1328', '1329', '1330', '1336', '1338', '1340', '1343', '1345']
# FID1330 and beyond has episodic HSV
fids = ['1336', '1338', '1339', '1340', '1343', '1345']
#fids = ['1336', '1338'] # for faster testing and debugging
#fids = ['1336', '1339','1340'] # for faster testing and debugging
# fid with good whisker tracking
#fids = ['1330', '1336', '1338', '1339', '1340']

### collect individual experiments and store in a list ###
experiments = list()
for fid_name in fids:
    get_ipython().magic(u"run hdfanalyzer.py {}".format(fid_name))
    experiments.append(neuro)

##### create arrays and lists for concatenating specified data from all experiments #####
##### create arrays and lists for concatenating specified data from all experiments #####
region      = np.empty((1, ))
exp_ind     = np.empty((1, 2))
depths      = np.empty((1, ))
cell_type   = list()
driven      = np.empty((1, ))
omi         = np.empty((1, 2))
selectivity = np.empty((1, 3))
selectivity_shuff = np.empty((1, 3, 2))
preference  = np.empty((1, 3))
preference_zero_mean  = np.empty((1, 3))
best_pos    = np.empty((1, ))
base_rate    = np.empty((1, 27, 2))
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
driven_inds = np.empty((1, ))

light_driven_units = np.zeros((1, 2))
light_num_driven_pos = np.zeros((1, 2))
light_driven_inds = list()


for exp_index, neuro in enumerate(experiments):
    # calculate measures that weren't calculated at init
    neuro.get_best_contact()
    neuro.get_sensory_drive(num_sig_pos=True)

    # concatenate measures
    cell_type.extend(neuro.cell_type)
    region      = np.append(region, neuro.shank_ids)
    depths      = np.append(depths, np.asarray(neuro.depths))
    selectivity = np.append(selectivity, neuro.selectivity, axis=0)
    selectivity_shuff = np.append(selectivity_shuff, neuro.selectivity_shuffled, axis=0)
    driven      = np.append(driven, neuro.driven_units, axis=0)
    omi         = np.append(omi, neuro.get_omi(), axis=0)
    #preference  = np.append(preference, neuro.get_preference, axis=0)
    #preference= np.append(preference, neuro.get_preferred_position(), axis=0)
    best_pos    = np.append(best_pos, neuro.best_contact)
    meanr       = np.append(meanr, neuro.get_rates_vs_strength(normed=True)[0], axis=0)
    S_all       = np.append(S_all, neuro.get_sparseness(kind='lifetime')[1], axis=0)

    pref_temp0, pref_temp1 = neuro.get_preferred_position()
    preference_zero_mean = np.append(preference_zero_mean, pref_temp0, axis=0)
    preference = np.append(preference, pref_temp1, axis=0)

    num_driven  = np.append(num_driven, neuro.num_driven_pos, axis=0)
    driven_inds = np.append(driven_inds, neuro.driven_indices, axis=0)

    # NEW 5/13/2021 find light modulated units and positions
    neuro.get_light_modulated_units()
    light_driven_units = np.append(light_driven_units, neuro.light_driven_units, axis=0)
    light_num_driven_pos = np.append(light_num_driven_pos, neuro.light_num_driven_pos, axis=0)
    light_driven_inds.extend(neuro.light_driven_indices)


    # compute mean tuning curve
    abs_tc = np.append(abs_tc, neuro.get_mean_tc(kind='abs_rate'), axis=0)
    evk_tc = np.append(evk_tc, neuro.get_mean_tc(kind='evk_rate'), axis=0)


    for unit_index in range(neuro.num_units):

        exp_temp = np.asarray([[exp_index, unit_index]])
        exp_ind = np.append(exp_ind, exp_temp, axis=0)
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

        # compute baseline rate (mean and sem)
        temp = np.zeros((1, 27, 2))
        temp[0, :, 0] = np.array([np.mean(k[:, unit_index]) for k in neuro.baseline_rate])[:]
        temp[0, :, 1] = np.array([sp.stats.sem(k[:, unit_index]) for k in neuro.baseline_rate])
        base_rate = np.append(base_rate, temp, axis=0)

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

del neuro

cell_type = np.asarray(cell_type)
region = region[1:,]
region = region.astype(int)

exp_ind = exp_ind[1:, :]
exp_ind = exp_ind.astype(int)

depths = depths[1:,]
driven = driven[1:,]
driven = driven.astype(int)

num_driven = np.asarray(num_driven[1:,])
driven_inds = np.asarray(driven_inds[1:,])

omi    = omi[1:,]
omi    = np.nan_to_num(omi)
selectivity = selectivity[1:, :]
selectivity_shuff = selectivity_shuff[1:, :, :]
preference  = preference[1:, :]
preference_zero_mean  = preference_zero_mean[1:, :]
best_pos    = best_pos[1:,]
best_pos    = best_pos.astype(int)
meanr       = meanr[1:, :]
base_rate    = base_rate[1:, :]
abs_rate    = abs_rate[1:, :]
evk_rate    = evk_rate[1:, :]
abs_tc      = abs_tc[1:, :]
evk_tc      = evk_tc[1:, :]
max_fr      = max_fr[1:,]
burst_rate  = burst_rate[1:, :]
adapt_ratio = adapt_ratio[1:, :]
S_all       = S_all[1:, :]

light_driven_units = light_driven_units[1:, :]
light_driven_pos = light_num_driven_pos[1:, :]
#light_driven_inds
npand   = np.logical_and

##### Num driven units per region / num driven positions per driven unit #####
##### Num driven units per region / num driven positions per driven unit #####
#NOTE 01

### can use chi test to see if ratio of driven units / total units is different

### !!! statistically significant difference in the mean number of driven pos / unit
#   sp.stats.ranksums

#TODO Compute driven units for silencing conditions!
cumulative=False # True makes it easier to see the difference, M1 more on left, S1 more on right)
bins = np.arange(1, 10)
fig, ax = plt.subplots(1,2)
for k, ctype in enumerate(['RS', 'FS']):
    m1_driven_unit_inds = np.where(npand(npand(region==0, driven==True), cell_type == ctype))[0]

    m1_total_units = sum(npand(region == 0, cell_type == ctype))
    m1_total_driven_units = len(m1_driven_unit_inds)                 # replace with Light modulated version
    m1_driven_pos_per_driven_units = num_driven[m1_driven_unit_inds] # replace with Light modulated version

    s1_driven_unit_inds = np.where(npand(npand(region==1, driven==True), cell_type == ctype))[0]

    s1_total_units = sum(npand(region == 1, cell_type == ctype))
    s1_total_driven_units = len(s1_driven_unit_inds)
    s1_driven_pos_per_driven_units = num_driven[s1_driven_unit_inds]

    ## what percentage of units where sensory driven??
    m1_percent = np.round(float(m1_total_driven_units) / m1_total_units * 100, decimals=1)
    s1_percent = np.round(float(s1_total_driven_units) / s1_total_units * 100, decimals=1)

    ## of those driven units how many bar positions elicited a sensory response?
    # "density" produces a probability density, allows one to compare m1 and s1
    # directly even though they have different numbers of units
    # bar histogram
    ax[k].hist(m1_driven_pos_per_driven_units, bins=bins, density=True, alpha=0.35, color='tab:blue', align='left', cumulative=cumulative)
    ax[k].hist(s1_driven_pos_per_driven_units, bins=bins, density=True, alpha=0.35, color='tab:red', align='left', cumulative=cumulative)

    # line histogram
    #ax[k].plot(bins[:-1], np.histogram(m1_driven_pos_per_driven_units, bins=bins, density=True)[0], color='tab:blue')
    #ax[k].plot(bins[:-1], np.histogram(s1_driven_pos_per_driven_units, bins=bins, density=True)[0],color='tab:red')
    ax[k].legend(['M1 {}'.format(ctype), 'S1 {}'.format(ctype)])
    ax[k].set_xlabel('Number of driven positions')
    ax[k].set_ylabel('Prob density')
    ax[k].set_title('M1 {}%, S1 {} %'.format(m1_percent, s1_percent))

##NOTE scratch space for light modulated analysis
cumulative=False # True makes it easier to see the difference, M1 more on left, S1 more on right)
bins = np.arange(1, 10)
fig, ax = plt.subplots(1,2)
for k, ctype in enumerate(['RS', 'FS']):
    # get m1 indices for all light modulated units (both sensory driven and
    # not-driven)
    m1_unit_inds = np.where(npand(region==0, cell_type == ctype))[0]
    m1_driven_unit_inds = np.where(npand(npand(region==0, cell_type == ctype), light_driven_units[:, 0]==1))[0]


    m1_total_units = sum(npand(region == 0, cell_type == ctype))
    m1_total_driven_units = len(m1_driven_unit_inds)                 # replace with Light modulated version
    m1_driven_pos_per_driven_units = light_num_driven_pos[m1_driven_unit_inds, 0] # replace with Light modulated version

    s1_unit_inds = np.where(npand(region==1, cell_type == ctype))[0]
    s1_driven_unit_inds = np.where(npand(npand(region==1, cell_type == ctype), light_driven_units[:, 1]==1))[0]

    s1_total_units = sum(npand(region == 1, cell_type == ctype))
    s1_total_driven_units = len(s1_driven_unit_inds)
    s1_driven_pos_per_driven_units = light_num_driven_pos[s1_driven_unit_inds, 1]

    ## what percentage of units where sensory driven??
    m1_percent = np.round(float(m1_total_driven_units) / m1_total_units * 100, decimals=1)
    s1_percent = np.round(float(s1_total_driven_units) / s1_total_units * 100, decimals=1)

    ## of those driven units how many bar positions elicited a sensory response?
    # "density" produces a probability density, allows one to compare m1 and s1
    # directly even though they have different numbers of units
    # bar histogram
    ax[k].hist(m1_driven_pos_per_driven_units, bins=bins, density=True, alpha=0.35, color='tab:blue', align='left', cumulative=cumulative)
    ax[k].hist(s1_driven_pos_per_driven_units, bins=bins, density=True, alpha=0.35, color='tab:red', align='left', cumulative=cumulative)
# line histogram
    #ax[k].plot(bins[:-1], np.histogram(m1_driven_pos_per_driven_units, bins=bins, density=True)[0], color='tab:blue')
    #ax[k].plot(bins[:-1], np.histogram(s1_driven_pos_per_driven_units, bins=bins, density=True)[0],color='tab:red')
    ax[k].legend(['M1 {}'.format(ctype), 'S1 {}'.format(ctype)])
    ax[k].set_xlabel('Number of driven positions')
    ax[k].set_ylabel('Prob density')
    ax[k].set_title('M1 {}%, S1 {} %'.format(m1_percent, s1_percent))




### Driven evoked rate for all positions ###
### Driven evoked rate for all positions ###
### Plots histogram (counts of binned evoked rates, e.g. 5 counts of evoked
### rates between 5-10 spikes/sec)

# m1_driven_unit_inds : which units are driven; driven_inds: indices of positions
# that are driven per unit
#NOTE: M1 RS driven units have ZERO positions with abs(evoked rates) < 1
#NOTE: S1 RS driven units have 11 positions with abs(evoked rates) < 1
# RS units evoked rate the same between M1 and S1
# FS units evoked rate higher in S1
# sp.stats.ks_2samp(sample1, sample2) tests whether these samples come from the same distribution

#NOTE 02

bins = np.arange(-15, 55, 2)
p = [0, 0]
cumulative=False
align='mid' # 'left' or 'mid' or 'right'
fig, ax = plt.subplots(1,2)
for k, ctype in enumerate(['RS', 'FS']):

    low_val = list()
    m1_driven_unit_inds = np.where(npand(npand(region==0, driven==True), cell_type == ctype))[0]
    m1_total_driven_pos = int(sum(num_driven[m1_driven_unit_inds]))
    m1_evk_rate = np.zeros((m1_total_driven_pos, ))
    count = 0
    for m1_unit_ind in m1_driven_unit_inds:
        for pos_ind in driven_inds[m1_unit_ind]:
            m1_evk_rate[count] = evk_rate[m1_unit_ind, pos_ind, 0]
            if np.abs(m1_evk_rate[count]) < 1:
                low_val.append((m1_unit_ind, pos_ind))
            count += 1

    s1_driven_unit_inds = np.where(npand(npand(region==1, driven==True), cell_type == ctype))[0]
    s1_total_driven_pos = int(sum(num_driven[s1_driven_unit_inds]))
    s1_evk_rate = np.zeros((s1_total_driven_pos, ))
    count = 0
    low_val = list()
    for s1_unit_ind in s1_driven_unit_inds:
        for pos_ind in driven_inds[s1_unit_ind]:
            s1_evk_rate[count] = evk_rate[s1_unit_ind, pos_ind, 0]
            if np.abs(s1_evk_rate[count]) < 1:
                low_val.append((s1_unit_ind, pos_ind))
            count += 1

    ax[k].hist(m1_evk_rate, bins=bins, density=True, alpha=0.45, color='tab:blue', align=align, cumulative=cumulative, histtype='stepfilled')
    ax[k].hist(s1_evk_rate, bins=bins, density=True, alpha=0.45, color='tab:red', align=align, cumulative=cumulative, histtype='stepfilled')
    ax[k].set_xlabel('Evoked rate')

    _, pval = sp.stats.ks_2samp(m1_evk_rate, s1_evk_rate)
    p[k] = pval
    ax[k].set_title('{} units significant: {}'.format(ctype, pval<0.05))

fig.suptitle('Proportion of driven positions')

## average absolute value of all evoked rates RS UNITS
#mean(np.abs(m1_evk_rate)) = 6.143286315401589
#std(np.abs(m1_evk_rate))  = 4.766264569191643
#mean(np.abs(s1_evk_rate)) = 7.651201939555056
#std(np.abs(s1_evk_rate))  = 7.791566943481743

## average absolute value of all evoked rates FS UNITS
#mean(np.abs(m1_evk_rate)) = 6.022781555314741
#std(np.abs(m1_evk_rate))  = 4.105780055687484
#mean(np.abs(s1_evk_rate)) = 13.794976632150387
#std(np.abs(s1_evk_rate))  = 16.682823029236335


## TODO make same figure as above but with light modualted units and positions
#  combine sensory driven and light "driven"/modulated into one array for easy
#  comparisons

 all_driven = np.concatenate((driven[:, None], light_driven_units), axis=1)

# 23 / 61 vM1 RS sensory driven units were modulated by vS1 silencing!
# 3 / 41 vS1 RS driven units were modulated by vM1 silencing!

##### Basic properties of spiking units #####
##### Basic properties of spiking units #####

###   Compare the baseline firing rates (baseline or control position) between
###   S1 and M1
###   Compare absolute firing rates at the best_position
###   Compare absolute firing rates at a forward position (pos 2 or 3)
###     positions 2 and 3 should have nice whisker contacts compared to more rear positions

###  BASELINE OR CONTROL POS        | BEST POSITION FR | POSTION 2/3 FR   |
###  RS driven M1 vs RS no_drive M1 |           ______ |           ______ |
###  -------------------------------
###  Split non-driven and driven units into two figures
###  RS no_drive M1xS1       ______ |           ______ |           ______ |
###  RS driven M1xS1         ______ |           ______ |           ______ |

#NOTE 03

bins=np.arange(0, 80, 2.5)
cumulative=False
normed=True
col2_pos = 3
align='mid' # 'left' or 'mid' or 'right'

#k=0; ctype='RS'
for k, ctype in enumerate(['RS', 'FS']):
    fig, ax = plt.subplots(2,3, sharex=True)
    for row in range(2):

        if row == 0:
            # non_driven M1 and S1
            ind0_region = 0 # M1
            ind1_region = 1 # S1
            ind0_driven = 0
            ind1_driven = 0
            ax[0, 0].set_ylabel('Non-driven units')
        elif row == 1:
            # driven M1 and S1
            ind0_region = 0 # M1
            ind1_region = 1 # S1
            ind0_driven = 1
            ind1_driven = 1
            ax[1, 0].set_ylabel('Driven units')

        # label x-axis
        ax[1][0].set_xlabel('Firing rate')
        ax[1][1].set_xlabel('Firing rate')
        ax[1][2].set_xlabel('Firing rate')

        row_temp = list()
        region_temp = list()
        cond_temp = list()

        #### Column 0 ####
        # get region ind0 indices
#        ind0_inds    = np.where(npand(npand(region==ind0_region, driven==ind0_driven), cell_type == ctype))[0]
        ind0_inds    = np.where(npand(region==ind0_region, driven==ind0_driven))[0]
        ind0_FR      = abs_rate[ind0_inds, 8, 0] # baseline rate / control position
        ax[row, 0].hist(ind0_FR, bins=bins, align=align, cumulative=cumulative, \
                normed=normed, color='tab:blue', alpha=0.5, histtype='stepfilled')

        # get region ind1 indices
#        ind1_inds = np.where(npand(npand(region==ind1_region, driven==ind1_driven), cell_type == ctype))[0]
        ind1_inds = np.where(npand(region==ind1_region, driven==ind1_driven))[0]
        ind1_FR   = abs_rate[ind1_inds, 8, 0]
        ax[row, 0].hist(ind1_FR, bins=bins, align=align, cumulative=cumulative, \
                normed=normed, color='tab:red', alpha=0.5, histtype='stepfilled')

        # col 0 stats
        [val, p] = sp.stats.levene(ind0_FR, ind1_FR)
        if p < 0.05:
            win_ind = np.argmax([np.var(ind0_FR), np.var(ind1_FR)])
            ax[row, 0].set_title('Baseline FR\n{} variance significantly greater'.format(win_ind), size=10)
            print('P SIG')
        else:
            ax[row, 0].set_title('Baseline FR\nvariance not significantly greater', size=10)
        print('row {} col 0 val={:.2f} p={:.4f}'.format(row, val, p))

        row_temp.append(ind0_FR); row_temp.append(ind1_FR)
        region_temp.append('vM1'); region_temp.append('vS1');
        cond_temp.append('Position 2 FR'); cond_temp.append('Position 2 FR')

        #### Column 1 ####
        ind0_best_pos_inds = best_pos[ind0_inds]
        ind0_FR            = abs_rate[ind0_inds, ind0_best_pos_inds, 0]
        ax[row, 1].hist(ind0_FR, bins=bins, align=align, cumulative=cumulative, \
                normed=normed, color='tab:blue', alpha=0.5, histtype='stepfilled')

        ind1_best_pos_inds = best_pos[ind1_inds]
        ind1_FR            = abs_rate[ind1_inds, ind1_best_pos_inds, 0]
        ax[row, 1].hist(ind1_FR, bins=bins, align=align, cumulative=cumulative, \
                normed=normed, color='tab:red', alpha=0.5, histtype='stepfilled')

        [val, p] = sp.stats.levene(ind0_FR, ind1_FR)
        if p < 0.05:
            win_ind = np.argmax([np.var(ind0_FR), np.var(ind1_FR)])
            ax[row, 1].set_title('Best driven FR\n{} variance significantly greater'.format(win_ind), size=10)
        else:
            ax[row, 1].set_title('Best driven FR\nvariance not significantly greater', size=10)
        print('row {} col 1 Wstat={:.2f} p={:.4f}'.format(row, val, p))

        row_temp.append(ind0_FR); row_temp.append(ind1_FR)
        region_temp.append('vM1'); region_temp.append('vS1');
        cond_temp.append('Position {} FR'.format(col2_pos)); cond_temp.append('Position {} FR'.format(col2_pos))

        #### Column 2 ####
        temp=list()
        for x in ind0_inds:
            uinds = driven_inds[x]
            temp.extend(abs_rate[x, uinds, 0].ravel())

        ind0_FR = temp
        #ind0_FR = abs_rate[ind0_inds, col2_pos, 0]
        ax[row, 2].hist(ind0_FR, bins=bins, align=align, cumulative=cumulative, \
                normed=normed, color='tab:blue', alpha=0.5, histtype='stepfilled')

        temp=list()
        for x in ind0_inds:
            uinds = driven_inds[x]
            temp.extend(abs_rate[x, uinds, 0].ravel())

        ind0_FR = temp
        #ind1_FR = abs_rate[ind1_inds, col2_pos, 0]
        ax[row, 2].hist(ind1_FR, bins=bins, align=align, cumulative=cumulative, \
                normed=normed, color='tab:red', alpha=0.5, histtype='stepfilled')
        ax[row, 2].set_title('Position {}'.format(col2_pos))

        [val, p] = sp.stats.levene(ind0_FR, ind1_FR)
        if p < 0.05:
            win_ind = np.argmax([np.var(ind0_FR), np.var(ind1_FR)])
            ax[row, 2].set_title('Position {} FR\n{} variance significantly greater'.format(col2_pos, win_ind), size=10)
        else:
            ax[row, 2].set_title('Position {} FR\nvariance not significantly greater'.format(col2_pos), size=10)
        print('row {} col 2 Wstat={:.2f} p={:.4f}'.format(row, val, p))

        row_temp.append(ind0_FR); row_temp.append(ind1_FR)
        region_temp.append('vM1'); region_temp.append('vS1');
        cond_temp.append('Position {} FR'.format(col2_pos)); cond_temp.append('Position {} FR'.format(col2_pos))


#        df = pd.DataFrame({'FR':row_temp, 'region':region_temp, 'condition':cond_temp})
##        df = df.set_index('fid')
#        sns.set_color_codes()
##        sns.boxplot(x="fid", y="change", hue="cell_type", data=df, palette={"RS":'b', "FS":'r'})
##        sns.boxplot(x="condition", y="FR", hue="cell_type", data=df, palette={"RS":'b', "FS":'r'})
#        sns.boxplot(x="condition", y="FR", data=df, palette={"RS":'b', "FS":'r'})






##### Tuning curve correlation analysis: abs_rate tuning curves #####
##### Tuning curve correlation analysis: abs_rate tuning curves #####

##TODO for generic no silencing analysis. Compare the distributions of all M1
# and all S2 pair-wise correlations. Hypothesis: S1 is more correlated and M1
# is more uncorrelated, distributed.

corr_m1 = list()
corr_m1_vals = list()
corr_s1 = list()
corr_s1_vals = list()
ctype='RS'
#m1_driven_unit_inds = npand(npand(region==0, driven==True), cell_type == ctype)
m1_driven_unit_inds = np.where(npand(region==0, driven==True))[0]
m1_vals = np.zeros((m1_driven_unit_inds.shape[0], 2)) # no light, s1 light , correlation values (upper triangle)

s1_driven_unit_inds = npand(npand(region==1, driven==True), cell_type == ctype)
s1_driven_unit_inds = np.where(npand(region==1, driven==True))[0]
s1_vals = np.zeros((np.sum(s1_driven_unit_inds), 2)) # no light, s1 light , correlation values (upper triangle)



##### start here or switch for above #####
m1_vals = np.zeros((1, 2)) # col 1: no light, col 2: light
s1_vals = np.zeros((1, 2)) # col 1: no light, col 2: light

m1_vals_shuff = np.zeros((1, 2)) # col 1: no light, col 2: light
s1_vals_shuff = np.zeros((1, 2)) # col 1: no light, col 2: light


m1s1 = list()
m1s1_light = list()

m1s1_shuff = list()
m1s1_light_shuff = list()

for exp_ID, n in enumerate(experiments):

    ## Get all pair-wise correlations for M1 and S1 with and without Light

    ## Simple analysis: Image of M1 and S1 correlations
    m1inds = np.where(npand(n.shank_ids == 0, n.driven_units==True))[0]
    m1temp, m1temp_shuff = n.tc_corr(m1inds, light_condition=0)
#    m1temp_light, m1temp_light_shuff = n.tc_corr(m1inds, light_condition=1)

    s1inds = np.where(npand(n.shank_ids == 1, n.driven_units==True))[0]
##    s1temp, s1temp_shuff = n.tc_corr(s1inds, light_condition=0)
##    s1temp_light, s1temp_light_shuff = n.tc_corr(s1inds, light_condition=2)

    m1s1.append((m1temp,s1temp))
    m1s1_light.append((m1temp_light,s1temp_light))

##    m1s1_shuff.append((m1temp_shuff,s1temp_shuff))
##    m1s1_light_shuff.append((m1temp_light_shuff, s1temp_light_shuff))


    ## take all the pair-wise values and append to a nx2 array

    #  M1 and M1 Light
    m1_size= m1inds.shape[0]
    m1_triu_inds = np.triu_indices(m1_size, k=1)
    # The True adds the second axis so they can be concatenated along axis=1
    m1_combo = np.concatenate( (m1temp[m1_triu_inds][:, True], \
            m1temp_light[m1_triu_inds][:, True]), axis=1)
    m1_vals = np.concatenate( (m1_vals, m1_combo), axis=0)

##    m1_combo_shuff = np.concatenate( (m1temp_shuff[m1_triu_inds][:, True], \
            m1temp_light_shuff[m1_triu_inds][:, True]), axis=1)
##    m1_vals = np.concatenate( (m1_vals, m1_combo), axis=0)
 ##   m1_vals_shuff = np.concatenate( (m1_vals_shuff, m1_combo_shuff), axis=0)


    s1_size= s1inds.shape[0]
    s1_triu_inds = np.triu_indices(s1_size, k=1)
    # The True adds the second axis so they can be concatenated along axis=1
    s1_combo = np.concatenate( (s1temp[s1_triu_inds][:, True], \
            s1temp_light[s1_triu_inds][:, True]), axis=1)
    s1_vals = np.concatenate( (s1_vals, s1_combo), axis=0)

##    s1_combo_shuff = np.concatenate( (s1temp_shuff[s1_triu_inds][:, True], \
##            s1temp_light_shuff[s1_triu_inds][:, True]), axis=1)
##    s1_vals_shuff = np.concatenate( (s1_vals_shuff, s1_combo_shuff), axis=0)


m1_vals = m1_vals[1:, :]
s1_vals = s1_vals[1:, :]

m1_vals_shuff = m1_vals_shuff[1:, :]
s1_vals_shuff = s1_vals_shuff[1:, :]


#    # TODO clean this up a bit
#    # M1
#    exp_ID_inds = npand(exp_ind[:, 0] == exp_ID, m1_driven_unit_inds)
#    tcm1_lc0 = abs_rate[exp_ID_inds, 0:8, 0] # unit x observations (mean pos 01, mean pos 02...mean pos 08)
#    tcm1_lc1 = abs_rate[exp_ID_inds, 9:17, 0]
#    array_size = tcm1_lc0.shape[0]
#
#    corr_m1_lc0 = np.corrcoef(tcm1_lc0)
#    corr_m1_lc1 = np.corrcoef(tcm1_lc1)
#    corr_m1.append([corr_m1_lc0, corr_m1_lc1])
#    lc0_vals = corr_m1_lc0[np.triu_indices(array_size, k=1)]
#    lc1_vals = corr_m1_lc1[np.triu_indices(array_size, k=1)]
#    corr_m1_vals.append([lc0_vals, lc1_vals])
#    m1_vals= np.concatenate((m1_vals, np.asarray([lc0_vals, lc1_vals]).T), axis=0)
#
#    # S1
#    exp_ID_inds = npand(exp_ind[:, 0] == exp_ID, s1_driven_unit_inds)
#    tcs1_lc0 = abs_rate[exp_ID_inds, 0:8, 0] # unit x observations (mean pos 01, mean pos 02...mean pos 08)
#    tcs1_lc1 = abs_rate[exp_ID_inds, 18:26, 0]
#    array_size = tcs1_lc0.shape[0]
#
#    corr_s1_lc0 = np.corrcoef(tcs1_lc0)
#    corr_s1_lc1 = np.corrcoef(tcs1_lc1)
#    corr_s1.append([corr_s1_lc0, corr_s1_lc1])
#    lc0_vals = corr_s1_lc0[np.triu_indices(array_size, k=1)]
#    lc1_vals = corr_s1_lc1[np.triu_indices(array_size, k=1)]
#    corr_s1_vals.append([lc0_vals, lc1_vals])
#    s1_vals = np.concatenate((s1_vals, np.asarray([lc0_vals, lc1_vals]).T), axis=0)



#####  THIS IS THE MAIN HIST PLOT FOR SIGNAL CORRELATIONS #####
## _, pval = sp.stats.ks_2samp(m1_evk_rate, s1_evk_rate)
## m1_vals = np.zeros((1, 2)) # col 1: no light, col 2: light
## s1_vals = np.zeros((1, 2)) # col 1: no light, col 2: light

fig, ax = plt.subplots(1,2)
bins=np.arange(-1,1,0.05)
ax[0].hist(m1_vals[:,0], bins=bins, alpha=0.5, align='mid', normed=True, color='tab:blue', histtype='stepfilled', label='vM1 + NoLight')
ax[0].hist(s1_vals[:,0], bins=bins, alpha=0.5, align='mid', normed=True, color='tab:red', histtype='stepfilled', label='vS1 + NoLight')
ax[0].set_xlabel('Signal correlation')
ax[0].set_ylabel('density')
ax[0].set_title('vS1 has more positively\ncorrelated units than vM1', size=12)
ax[0].legend(loc='upper left')
ax[0].vlines(0, 0, ax[0].get_ylim()[1]+2, 'k', linestyles='dashed')

ax[1].hist(m1_vals[:,1], bins=bins, alpha=0.5, align='mid', normed=True, color='tab:blue', histtype='stepfilled', label='vM1 + vS1 silencing')
ax[1].hist(s1_vals[:,1], bins=bins, alpha=0.5, align='mid', normed=True, color='tab:red', histtype='stepfilled', label='vS1 + vM1 silencing')
ax[1].set_xlabel('Signal correlation')
ax[1].set_ylabel('density')
ax[1].set_title('Both regions correlations change\n but not in a single direction', size=12)
ax[1].vlines(0, 0, ax[1].get_ylim()[1]+2, 'k', linestyle='dashed')
ax[1].legend(loc='upper left')


violinplot([m1_vals[:, 1], m1_vals[:, 0], s1_vals[:,0], s1_vals[:, 1]])

#### differences between light and no light conditions
fig, ax = plt.subplots(1,2)
bins=np.arange(-1,1,0.05)
ax[0].hist(m1_vals[:,1] - m1_vals[:,0], bins=bins, alpha=0.5, align='mid', normed=True, color='tab:blue', histtype='stepfilled', label='vM1 vS1Light - NoLight')
ax[0].hist(s1_vals[:,1] - s1_vals[:,0], bins=bins, alpha=0.5, align='mid', normed=True, color='tab:red', histtype='stepfilled', label='vS1 vM1Light - Nolight')

## TODO figure out a way to actually learn something from this analysis
fig, ax = plt.subplots(1,2)
ax[0].scatter(m1_vals[:,0], m1_vals[:,1], s=12, color='tab:blue')
ax[0].plot([-1,1],[-1,1], 'k')
ax[0].set_title('vM1 signal correlations')
ax[0].set_xlabel('vM1 corr')
ax[0].set_ylabel('vM1 corr\n+vS1 silencing')
ax[1].scatter(s1_vals[:,0], s1_vals[:,1], s=12, color='tab:red')
ax[1].plot([-1,1],[-1,1], 'k')
ax[1].set_title('vS1 signal correlations')
ax[1].set_xlabel('vS1 corr')
ax[1].set_ylabel('vS1 corr\n+vM1 silencing')

## heatmap
fig, ax = plt.subplots(6,2)
for x in range(6):
    # No Light M1 vs S1
    im0 = ax[x][0].imshow(m1s1[x][1], vmin=-1, vmax=1, cmap='coolwarm', aspect='auto')
#    im1 = ax[x][1].imshow(m1s1_light[x][1], vmin=-1, vmax=1, cmap='coolwarm', aspect='auto')
    im1 = ax[x][1].imshow(m1s1_light[x][1], vmin=-1, vmax=1, cmap='coolwarm', aspect='auto')

#    # Differences between light and no light conditions (M1+S1 silencing) - M1NoLight, and vice versa)
#    im0 = ax[x][0].imshow(m1s1_light[x][0] - m1s1[x][0], vmin=-1, vmax=1, cmap='coolwarm', aspect='auto')
#    im1 = ax[x][1].imshow(m1s1_light[x][1] - m1s1[x][1], vmin=-1, vmax=1, cmap='coolwarm', aspect='auto')


#fig.colorbar(im0, ax=ax[0])
#fig.colorbar(im1, ax=ax[1])

##### Test Space Signal correlation plots
#m1 = where(npand(n.shank_ids==0, n.driven_units))[0]
#s1 = where(npand(n.shank_ids==1, n.driven_units))[0]
#
#a =  n.tc_corr(m1)
#b =  n.tc_corr(s1,light_condition=0)
#
#fig, ax = plt.subplots(1,2)
#blah = ax[0].imshow(a, vmin=-1,vmax=1, cmap='coolwarm')
#blah = ax[1].imshow(b, vmin=-1,vmax=1, cmap='coolwarm')


##### Noise correlation analysis #####
##### Noise correlation analysis #####

#unit_indices = np.where(np.logical_and(neuro.shank_ids == 0, neuro.cell_type=='RS'))[0]
#unit_indices = np.where(region == 0)[0]
unit_indices = np.where(neuro.shank_ids == 0)[0]
R0 = neuro.noise_corr(unit_indices, cond=3)
R1 = neuro.noise_corr(unit_indices, cond=3+9)

sns.heatmap(R0, vmin=-0.4, vmax=.4, cmap='coolwarm', annot=True, xticklabels=unit_indices, yticklabels=unit_indices)

bins = np.arange(-1,1,0.05)
hist(R0, bins=bins, color='tab:blue', alpha=0.4)
hist(R1, bins=bins, color='tab:red', alpha=0.4)
dt = np.asarray(R1) - np.asarray(R0)
hist(dt, bins=bins, color='tab:red', alpha=0.4)

R0 = list()
R1 = list()
for k in range(4):
    k=8
    #R0.extend(neuro.noise_corr(unit_indices, cond=k, return_vals=True))
    R0 = neuro.noise_corr(unit_indices, cond=k, return_vals=True)
    hist(R0, bins=bins, alpha=0.4, cumulative=False)
#    plot(np.cumsum(R0)/float(len(R0)))
#    R1.extend(neuro.noise_corr(unit_indices, cond=k+9, return_vals=True))
    R1 = neuro.noise_corr(unit_indices, cond=k+9, return_vals=True)
    hist(R1, bins=bins, alpha=0.4, cumulative=False)


m1_vals = np.zeros((1, 2)) # col 1: no light, col 2: light
s1_vals = np.zeros((1, 2)) # col 1: no light, col 2: light

condition = 3

for exp_ID, n in enumerate(experiments):

    ## Get all pair-wise correlations for M1 and S1 with and without Light

    ## Simple analysis: Image of M1 and S1 correlations
    m1inds = np.where(npand(n.shank_ids == 0, n.driven_units==True))[0]
    m1temp = n.noise_corr(m1inds, cond=condition, return_vals=True)
    m1temp_light = n.noise_corr(m1inds, cond=condition+9, return_vals=True)
    m1_combo = np.concatenate( (m1temp[:, True], m1temp_light[:, True]), axis=1)
    m1_vals = np.concatenate( (m1_vals, m1_combo), axis=0)

    ## Simple analysis: Image of M1 and S1 correlations
    s1inds = np.where(npand(n.shank_ids == 1, n.driven_units==True))[0]
    s1temp = n.noise_corr(s1inds, cond=condition, return_vals=True)
    s1temp_light = n.noise_corr(s1inds, cond=condition+9+9, return_vals=True)
    s1_combo = np.concatenate( (s1temp[:, True], s1temp_light[:, True]), axis=1)
    s1_vals = np.concatenate( (s1_vals, s1_combo), axis=0)

m1_vals = m1_vals[1:, :]
s1_vals = s1_vals[1:, :]


##### Selectivity analysis #####
##### Selectivity analysis #####

### Selectivity of a unit vs its evoked rate at best position ###
### Selectivity of a unit vs its evoked rate at best position ###
fig, ax = plt.subplots(1,2)
for k, ctype in enumerate(['RS', 'FS']):
    m1_driven_unit_inds = np.where(npand(npand(region==0, driven==True), cell_type == ctype))[0]
    #m1_best_fr = evk_rate[m1_driven_unit_inds, best_pos[m1_driven_unit_inds], 0]
    m1_num_driven_pos = num_driven[m1_driven_unit_inds]
    m1_sel = selectivity[m1_driven_unit_inds, 0]

    s1_driven_unit_inds = np.where(npand(npand(region==1, driven==True), cell_type == ctype))[0]
    #s1_best_fr = evk_rate[s1_driven_unit_inds, best_pos[s1_driven_unit_inds], 0]
    s1_num_driven_pos = num_driven[s1_driven_unit_inds]
    s1_sel = selectivity[s1_driven_unit_inds, 0]

    ax[k].scatter(m1_num_driven_pos, m1_sel, c='tab:blue')
    ax[k].scatter(s1_num_driven_pos, s1_sel, c='tab:red')

fig, ax = plt.subplots(2,2)
for k, ctype in enumerate(['RS', 'FS']):
    m1_driven_unit_inds = np.where(npand(npand(region==0, driven==True), cell_type == ctype))[0]
    m1_sel = selectivity[m1_driven_unit_inds, 0]
    m1_sel_l = selectivity[m1_driven_unit_inds, 1]
    ax[0][k].scatter(m1_sel, m1_sel_l, color='tab:blue', s=12, marker='o')
    ax[0][k].plot([0,1],[0,1], color='k', linewidth=1.5)
    ax[0][k].set_xlim(0,1)
    ax[0][k].set_ylim(0,1)
    ax[0][k].set_title('vM1 {} units'.format(ctype))
    ax[0][k].set_ylabel('Selectivity\n+ vS1 silencing')

    s1_driven_unit_inds = np.where(npand(npand(region==1, driven==True), cell_type == ctype))[0]
    s1_sel = selectivity[s1_driven_unit_inds, 0]
    s1_sel_l = selectivity[s1_driven_unit_inds, 2]
    ax[1][k].scatter(s1_sel, s1_sel_l, color='tab:red', s=12, marker='o')
    ax[1][k].plot([0,1],[0,1], color='k', linewidth=1.5)
    ax[1][k].set_xlim(0,1)
    ax[1][k].set_ylim(0,1)
    ax[1][k].set_title('vS1 {} units'.format(ctype))
    ax[1][k].set_xlabel('Selectivity NoLight')
    ax[1][k].set_ylabel('Selectivity\n+ vM1 silencing')

























##### Change in preferred position with silencing #####
##### Change in preferred position with silencing #####

#TODO revert back to original best_position method
#TODO think about how I can weigh positions better! z-score??
### Scatter Plot (M1 vs M1+S1 silencing, vice versa) ###

ctype = 'RS'
m1_inds = np.where(npand(npand(region==0, driven==True), cell_type == ctype))[0]
s1_inds = np.where(npand(npand(region==1, driven==True), cell_type == ctype))[0]

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
fig.suptitle("Preferred positions don't change significantly from Silencing")
ax[0].scatter(preference[m1_inds,0], preference[m1_inds,1])
ax[0].plot([-10,10],[-10,10])
ax[0].set_xlabel('vM1 preferred position\nNo silencing')
ax[0].set_ylabel('vM1 preferred position\nvS1 silencing')

ax[1].scatter(preference[s1_inds,0], preference[s1_inds,2])
ax[1].plot([-10,10],[-10,10])
ax[1].set_xlabel('vS1 preferred position\nNo silencing')
ax[1].set_ylabel('vS1 preferred position\nvM1 silencing')

### Boxplot of same data as scatter plot (M1 vs M1+S1 silencing, vice versa) ###
fig, ax = plt.subplots()
ax. boxplot([preference[m1_inds,0], preference[m1_inds,1], preference[s1_inds,0], preference[s1_inds,2]])



############ best position and selectivity scratch space ############
############ best position and selectivity scratch space ############

#[position IDs, selectivity] <-- n x 2 array (vals, weights)

ctype = 'RS'
m1ind = np.where(npand(npand(region==0, driven==True), cell_type == ctype))[0]
s1ind = np.where(npand(npand(region==1, driven==True), cell_type == ctype))[0]

# No light
m1sel = selectivity[m1ind, 0]
m1pos = preference_zero_mean[m1ind, 1]

s1sel = selectivity[s1ind, 0]
s1pos = preference_zero_mean[s1ind, 2]

# light
m1sel_L = selectivity[m1ind, 1]
m1pos_L = preference_zero_mean[m1ind, 1]

s1sel_L = selectivity[s1ind, 2]
s1pos_L = preference_zero_mean[s1ind, 2]



### WEIGHTED preferred position by selectivity
## it appears that S1 favors the space anterior to region preferred pos
## while M1 prefers posterior!!
m1_counts = np.zeros((11,))
s1_counts = np.zeros((11,))
x = np.arange(-5,6)
for k, val in enumerate(m1pos):
    m1_counts[int(val)+5] = m1sel[k] # 5 in center of array
    #m1_counts_light[int(val)+5] = selectivity[m1ind, 1][k] # 5 in center of array
for k, val in enumerate(s1pos):
    s1_counts[int(val)+5] = s1sel[k] # 5 in center of array

# this normalizes it right? so it sums to 1?
m1_counts = m1_counts / sum(m1_counts)
s1_counts = s1_counts / sum(s1_counts)

figure()
bar(x, m1_counts, alpha=0.5)
bar(x, s1_counts, alpha=0.5)

figure()
scatter(m1pos, m1sel)
scatter(s1pos, s1sel)
xlim([-7,7])
ylim([0,1])

scatter(m1_pos

### unweighted preferred position
hist(preference_zero_mean[m1ind,0],bins=bins, normed=True,alpha=0.5)
hist(preference_zero_mean[s1ind,0],bins=bins, normed=True,alpha=0.5)

### weighted??? preferred position
## NO this is garbage
#figure()
#m1_w = m1pos * m1sel
#s1_w = s1pos * s1sel
#hist(m1_w,bins=bins, normed=True,alpha=0.5)
#hist(s1_w,bins=bins, normed=True,alpha=0.5)


        ##### NOTES #####
### How to weigh preferred positions ###
### iterate through the preferred positions.
### use the preferred position as an INDEX
### INDEX into the selectivity values and add that value to COUNTS
### in a bar graph this should show which positions were preferred based on
### the strength of the evoked response!!!


### maybe a good way of looking at the effects of light is by computing the
### difference between preferred positions (light - nolight) and computing the
### difference between selectivity for units of interest.
### TODO try using a scatter plot of the differences to look for trends
### hypothesis: s1 silencing moves M1's preferred position forward (negative vals)
### and makes the units broader (i.e. less selective).
### hypothesis: m1 silencing move S1's pref position back and increases
### selectivity making those units more sensitive to anterior whiskers.


###### Plot selectivity histogram with opto #####
###### Plot selectivity histogram with opto #####

## M1 RS selectivity significantly different with S1 silencing
# sp.stats.wilcoxon(selectivity[m1_inds,0], selectivity[m1_inds,1])
# WilcoxonResult(statistic=548.0, pvalue=0.0043016056098121504)

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
ax[1][0].set_title('M1 units, M1 units + S1 silencing')
ax[1][0].legend(['M1', 'M1 + S1 silencing'])
ax[2][0].hist(selectivity[s1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='k')
ax[2][0].hist(selectivity[s1_inds, 2], bins=bins, edgecolor='None', alpha=0.5, color='r')
ax[2][0].set_title('S1 units, S1 units + M1 silencing')
ax[2][0].legend(['S1 units', 'S1 units + M1 silencing'])

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


##### plot selectivity histogram M1 vs S1 overlaid no opto #####
##### plot selectivity histogram M1 vs S1 overlaid no opto #####

bins = np.arange(0, 1, 0.05)
fig, ax = plt.subplots(1, 2)
fig.suptitle('selectivity', fontsize=20)

## RS
m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')
#hist(selectivity[npand(cell_type==unit_type, region == region[unit_count]), 0], bins=np.arange(0, 1, 0.05)

ax[0].hist(selectivity[m1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='tab:blue', normed=True)
ax[0].hist(selectivity[s1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='tab:red', normed=True)
ax[0].set_title('RS units M1: {} units, S1: {} units\nno light'.format(sum(m1_inds), sum(s1_inds)))
ax[0].legend(['M1', 'S1'])

## FS
m1_inds = npand(npand(region==0, driven==True), cell_type=='FS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='FS')
#hist(selectivity[npand(cell_type==unit_type, region == region[unit_count]), 0], bins=np.arange(0, 1, 0.05)

ax[1].hist(selectivity[m1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='tab:blue', normed=True)
ax[1].hist(selectivity[s1_inds, 0], bins=bins, edgecolor='None', alpha=0.5, color='tab:red', normed=True)
ax[1].set_title('FS units M1: {} units, S1: {} units\nno light'.format(sum(m1_inds), sum(s1_inds)))
ax[1].legend(['M1', 'S1'])


###### Plot selectivity Scatter for all driven units #####
###### Plot selectivity Scatter for all driven units  #####

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

ax[0][0].plot(selectivity[m1_inds, 0], selectivity[m1_inds, 1], 'o', color='tab:blue')
ax[0][0].set_title('M1 RS units')
ax[0][0].set_xlabel('No Light')
ax[0][0].set_ylabel('S1 Silencing')
ax[0][0].plot([0, 1], [0, 1], 'k')

ax[1][0].plot(selectivity[s1_inds, 0], selectivity[s1_inds, 2], 'o', color='tab:red')
ax[1][0].set_title('S1 RS units')
ax[1][0].set_xlabel('No Light')
ax[1][0].set_ylabel('M1 Silencing')
ax[1][0].plot([0, 1], [0, 1], 'k')

m1_inds = npand(npand(region==0, driven==True), cell_type=='FS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='FS')
ax[0][1].plot(selectivity[m1_inds, 0], selectivity[m1_inds, 1], 'o', color='tab:blue')
ax[0][1].set_title('M1 FS units')
ax[0][1].set_xlabel('No Light')
ax[0][1].set_ylabel('S1 Silencing')
ax[0][1].plot([0, 1], [0, 1], 'k')

ax[1][1].plot(selectivity[s1_inds, 0], selectivity[s1_inds, 2], 'o', color='tab:red')
ax[1][1].set_title('S1 FS units')
ax[1][1].set_xlabel('No Light')
ax[1][1].set_ylabel('M1 Silencing')
ax[1][1].plot([0, 1], [0, 1], 'k')


###### Plot change in selectivity histogram #####
###### Plot change in selectivity histogram #####

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
bins = np.arange(-1, 1, 0.04)

# RS units
m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')

m1_diff = selectivity[m1_inds, 1] - selectivity[m1_inds, 0]
s1_diff = selectivity[s1_inds, 2] - selectivity[s1_inds, 0]

ax[0][0].hist(m1_diff, bins=bins, color='tab:blue', alpha=0.5, histtype='stepfilled')
ax[0][0].set_title('M1 RS units')
ax[0][0].set_xlabel('Change in selectivity')
ax[0][0].set_ylabel('Counts')

ax[1][0].hist(s1_diff, bins=bins, color='tab:red', alpha=0.5, histtype='stepfilled')
ax[1][0].set_title('S1 RS units')
ax[1][0].set_xlabel('Change in selectivity')
ax[1][0].set_ylabel('Counts')

# FS units
m1_inds = npand(npand(region==0, driven==True), cell_type=='FS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='FS')

m1_diff = selectivity[m1_inds, 1] - selectivity[m1_inds, 0]
s1_diff = selectivity[s1_inds, 2] - selectivity[s1_inds, 0]

ax[0][1].hist(m1_diff, bins=bins, color='tab:blue', alpha=0.5, histtype='stepfilled')
ax[0][1].set_title('M1 FS units')
ax[0][1].set_xlabel('Change in selectivity')

ax[1][1].hist(s1_diff, bins=bins, color='tab:red', alpha=0.5, histtype='stepfilled')
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
        col.set_xlim(-0.55, 0.55)
        col.vlines(0, 0, ylim_max, 'k', linestyle='dashed', linewidth=1)



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



##### Spontaneous/baseline rate analysis #####
##### Spontaneous/baseline rate analysis #####


##### plot spontaneous/baseline rates #####
##### plot spontaneous/baseline rates #####

fig, ax = plt.subplots(2, 2, figsize=(10,9), sharex=True, sharey=True)
#fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
fig.suptitle('Baseline firing rates', fontsize=20)

## RS
m1_inds = npand(npand(region==0, driven==True), cell_type=='RS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='RS')

# M1 RS and S1 RS + S1 silencing
ax[0][0].errorbar(abs_rate[m1_inds, 8, 0], abs_rate[m1_inds, 8+9, 0], \
        xerr=abs_rate[m1_inds, 8, 1], yerr=abs_rate[m1_inds, 8+9, 1], c='tab:blue', fmt='o', ecolor='tab:blue', markersize=4)
#ax[0][0].errorbar(abs_rate[s1_inds, 8, 0], abs_rate[s1_inds, 8+9, 0], \
        #        xerr=abs_rate[s1_inds, 8, 1], yerr=abs_rate[s1_inds, 8+9, 1], c='tab:red', fmt='o', ecolor='tab:red')

max_val = np.max([ax[0][0].get_xlim(), ax[0][0].get_ylim()])
ax[0][0].set_xlim(-1, max_val)
ax[0][0].set_ylim(-1, max_val)
ax[0][0].plot([0, max_val], [0, max_val], 'k')
ax[0][0].hlines(0, -2, max_val, linestyles='dashed')
ax[0][0].set_title('vM1 RS units + vS1 silencing')
ax[0][0].set_ylabel('firing rate (Hz)\n+ vS1 silencing')
ax[0][0].set_xlabel('firing rate (Hz)\n+ NoLight')

# S1 RS and M1 RS + M1 silencing
#ax[1][0].errorbar(abs_rate[m1_inds, 8, 0], abs_rate[m1_inds, 8+9+9, 0], \
#        xerr=abs_rate[m1_inds, 8, 1], yerr=abs_rate[m1_inds, 8+9+9, 1], c='tab:blue', fmt='o', ecolor='tab:blue', markersize=4)
ax[1][0].errorbar(abs_rate[s1_inds, 8, 0], abs_rate[s1_inds, 8+9+9, 0], \
        xerr=abs_rate[s1_inds, 8, 1], yerr=abs_rate[s1_inds, 8+9+9, 1], c='tab:red', fmt='o', ecolor='tab:red', markersize=4)

max_val = np.max([ax[1][0].get_xlim(), ax[1][0].get_ylim()])
ax[1][0].set_xlim(-1, max_val)
ax[1][0].set_ylim(-1, max_val)
ax[1][0].plot([0, max_val], [0, max_val], 'k')
ax[1][0].set_title('vS1 RS units + vM1 silencing')
ax[1][0].hlines(0, -2, max_val, linestyles='dashed')
ax[1][0].set_title('vS1 RS units + vM1 silencing')
ax[1][0].set_ylabel('firing rate (Hz)\n+ vM1 silencing')
ax[1][0].set_xlabel('firing rate (Hz)\n+ NoLight')

## FS
m1_inds = npand(npand(region==0, driven==True), cell_type=='FS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='FS')

# S1 sielncing
ax[0][1].errorbar(abs_rate[m1_inds, 8, 0], abs_rate[m1_inds, 8+9, 0], \
        xerr=abs_rate[m1_inds, 8, 1], yerr=abs_rate[m1_inds, 8+9, 1], c='tab:blue', fmt='o', ecolor='tab:blue', markersize=4)
#ax[0][1].errorbar(abs_rate[s1_inds, 8, 0], abs_rate[s1_inds, 8+9, 0], \
#        xerr=abs_rate[s1_inds, 8, 1], yerr=abs_rate[s1_inds, 8+9, 1], c='tab:red', fmt='o', ecolor='tab:red', markersize=4)

max_val = np.max([ax[0][1].get_xlim(), ax[0][1].get_ylim()])
ax[0][1].set_xlim(-1, max_val)
ax[0][1].set_ylim(-1, max_val)
ax[0][1].plot([0, max_val], [0, max_val], 'k')
ax[0][1].set_title('vM1 FS units + vS1 silencing')
ax[0][1].hlines(0, -2, max_val, linestyles='dashed')
ax[0][1].set_xlabel('Light Off\nfiring rate (Hz)')
ax[0][1].set_ylabel('firing rate (Hz)\n+ vS1 silencing')
ax[0][1].set_xlabel('firing rate (Hz)\n+ NoLight')

#ax[1][1].errorbar(abs_rate[m1_inds, 8, 0], abs_rate[m1_inds, 8+9+9, 0], \
#        xerr=abs_rate[m1_inds, 8, 1], yerr=abs_rate[m1_inds, 8+9+9, 1], c='tab:blue', fmt='o', ecolor='tab:blue', markersize=4)
ax[1][1].errorbar(abs_rate[s1_inds, 8, 0], abs_rate[s1_inds, 8+9+9, 0], \
        xerr=abs_rate[s1_inds, 8, 1], yerr=abs_rate[s1_inds, 8+9+9, 1], c='tab:red', fmt='o', ecolor='tab:red', markersize=4)

max_val = np.max([ax[1][1].get_xlim(), ax[1][1].get_ylim()])
ax[1][1].set_xlim(-1, max_val)
ax[1][1].set_ylim(-1, max_val)
ax[1][1].plot([0, max_val], [0, max_val], 'k')
ax[1][1].set_title('vS1 FS units + vM1 silencing')
ax[1][1].hlines(0, -2, max_val, linestyles='dashed')
ax[1][1].set_xlabel('Light Off\nfiring rate (Hz)')
ax[1][1].set_ylabel('firing rate (Hz)\n+ vM1 silencing')
ax[1][1].set_xlabel('firing rate (Hz)\n+ NoLight')



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

m1_diff = abs_rate[m1_inds, 9-1+9, 0] - abs_rate[m1_inds, 9-1, 0]
s1_diff = abs_rate[s1_inds, 9-1+9+9, 0] - abs_rate[s1_inds, 9-1, 0]

ax[0][0].hist(m1_diff, bins=bins, color='tab:blue', alpha=0.5, normed=True)
ax[0][0].set_title('M1 RS units')
ax[0][0].set_xlabel('Change in baseline rate')
ax[0][0].vlines(0, 0, ax[0][0].get_ylim()[1], 'k', linestyle='dashed', linewidth=1)

ax[0][1].hist(s1_diff, bins=bins, color='tab:red', alpha=0.5, normed=True)
ax[0][1].set_title('S1 RS units')
ax[0][1].set_xlabel('Change in baseline rate')
ax[0][1].vlines(0, 0, ax[0][1].get_ylim()[1], 'k', linestyle='dashed', linewidth=1)

# FS units
m1_inds = npand(npand(region==0, driven==True), cell_type=='FS')
s1_inds = npand(npand(region==1, driven==True), cell_type=='FS')

m1_diff = abs_rate[m1_inds, 9 -1+9, 0] - abs_rate[m1_inds, 9 -1, 0]
s1_diff = abs_rate[s1_inds, 9 -1+9+9, 0] - abs_rate[s1_inds, 9 -1, 0]

ax[1][0].hist(m1_diff, bins=bins, color='tab:blue', alpha=0.5, normed=True)
ax[1][0].set_title('M1 FS units')
ax[1][0].set_xlabel('Change in baseline rate')
ax[1][0].vlines(0, 0, ax[1][0].get_ylim()[1], 'k', linestyle='dashed', linewidth=1)

ax[1][1].hist(s1_diff, bins=bins, color='tab:red', alpha=0.5, normed=True)
ax[1][1].set_title('S1 FS units')
ax[1][1].set_xlabel('Change in baseline rate')
ax[1][1].vlines(0, 0, ax[1][1].get_ylim()[1], 'k', linestyle='dashed', linewidth=1)

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


##### SCRATCH #####
##### SCRATCH #####

#for unit_index in range(n.num_units):
#
#    # compute absolute rate (mean and sem)
#    temp = np.zeros((1, 27, 2))
#    temp[0, :, 0] = np.array([np.mean(k[:, unit_index]) for k in n.abs_rate])[:]
#    temp[0, :, 1] = np.array([sp.stats.sem(k[:, unit_index]) for k in n.abs_rate])
#    abs_rate = np.append(abs_rate, temp, axis=0)













