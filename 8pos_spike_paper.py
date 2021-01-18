import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import scipy.io as sio


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

for neuro in experiments:
    # calculate measures that weren't calculated at init
    neuro.get_best_contact()
    neuro.get_sensory_drive()

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



