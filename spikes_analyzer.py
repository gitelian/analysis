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
        fig.suptitle('Region: {}, depth: {}, unit type: {}, mouse: {}'.format(\
                neuro.region_dict[neuro.shank_ids[unit_index]], \
                neuro.neo_obj.segments[0].spiketrains[unit_index].annotations['depth'], \
                neuro.cell_type[unit_index], \
                fid))

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





#TODO: population analysis
#TODO: population analysis
# selectivity, center of mass, burstiness, OMI, decoder!!!
# do this for best position and no contact position. Plot things overall and
# then look at things as a function of depth.

for experiment in experiments:
    # neuro.sensory_drives() # write this to use KW and Dunn's to test for
    # sensory driven activity and return a vector of length units with a 0 or a
    # 1 to indicate not driven or driven.


## baseline firing rate analysis
#m1_rates = list()
#s1_rates = list()
#m1_sel   = list()
#s1_sel   = list()
#
#for k in range(27):
##for k in range(18):
#    m1_temp = np.empty(1)
#    s1_temp = np.empty(1)
#    #for neuro in exps: #exps[2::]:
#    for neuro in exps: #exps[2::]:
#        rates_temp = neuro.abs_rate[k].mean(axis=0)
#        #m1_inds = np.logical_and(neuro.shank_ids == 0, neuro.cell_type == 'RS')
#        m1_inds = neuro.shank_ids == 0
#        s1_inds = neuro.shank_ids == 1
#        m1_temp = np.append(m1_temp, rates_temp[m1_inds])
#        s1_temp = np.append(s1_temp, rates_temp[s1_inds])
#
#        if k == 0:
#            neuro.get_selectivity()
#            m1_sel.append(neuro.selectivity[m1_inds, :])
#            s1_sel.append(neuro.selectivity[s1_inds, :])
#    m1_rates.append(m1_temp)
#    s1_rates.append(s1_temp)
#
##    m1_rates.append(rates_temp[m1_inds])
##    s1_rates.append(rates_temp[s1_inds])
#
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
#
## OMI for control position (diff over the sum)
#m1_omi = (m1_rates[8+9] - m1_rates[8])/ (m1_rates[8+9] + m1_rates[8])
#s1_omi = (s1_rates[8+9] - s1_rates[8])/ (s1_rates[8+9] + s1_rates[8])
#violinplot([m1_omi, s1_omi], pos, vert=True, widths=0.7,
#                              showextrema=True, showmedians=True)
#
## selectivity
#
#m1_temp = list()
#s1_temp = list()
#for k in m1_sel:
#    m1_temp.extend(k[:, 0].ravel())
#for k in s1_sel:
#    s1_temp.extend(k[:, 0].ravel())
#
#plt.subplots(1,2)
#plt.subplot(1,2,1)
#plt.hist(m1_temp)
#plt.subplot(1,2,2)
#plt.hist(s1_temp)

###### Plot selectivity stuff #####
#
#neuro.get_selectivity()
#m1_inds = np.where(neuro.shank_ids == 0)[0]
#s1_inds = np.where(neuro.shank_ids == 1)[0]
#m1_sel_nolight  = neuro.selectivity[m1_inds, 0]
#m1_sel_s1light  = neuro.selectivity[m1_inds, 1]
#s1_sel_nolight  = neuro.selectivity[s1_inds, 0]
#s1_sel_s1light  = neuro.selectivity[s1_inds, 1]
#
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



