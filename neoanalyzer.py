#!/bin/bash
import os.path
import sys
import warnings
import scipy as sp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib as mpl
import scipy.signal
from neo.io import NeoHdf5IO
import quantities as pq
import icsd
import ranksurprise
# for LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# for t-distributed stochastic neighbor embedding
from sklearn.manifold import TSNE
# for MDS
from sklearn import manifold


# change default figure type to PDF
mpl.rcParams['savefig.format'] = 'pdf'
sns.set_style("whitegrid", {'axes.grid' : False})

class NeuroAnalyzer(object):
    """
    Analyzes data contained in a neo object

    REMEMBER: any changes to the passed in neo object also occur to the original
    object. They point to the same place in memory. No copy is made.
    """

    def __init__(self, neo_obj):

        print('\n-----__init__-----')
        # segments can get out of the order they were created
        sorted_index_list = np.argsort([k.index for k in neo_obj.segments])
        neo_obj.segments  = [neo_obj.segments[k] for k in sorted_index_list]

        # sort by shank/region and then by depth
        sorted_index_list = self.__sort_units(neo_obj)

        # add neo object to class instance
        self.neo_obj         = neo_obj

        # find stimulus IDs
        self.stim_ids        = np.sort(np.unique([k.annotations['trial_type'] for k in neo_obj.segments]))

        # find number of units
        self.num_units       = len(self.neo_obj.segments[0].spiketrains)

        # find shank names (i.e. names of electrodes/recording sites, e.g.
        # 'e1', 'e2')
        self.shank_names     = np.sort(np.unique([k.annotations['shank'] for k in neo_obj.segments[0].spiketrains]))

        # Find depths of each shank and add it to self.shank_depths
        self.shank_depths = self.__get_shank_depths()

        # find shank IDs for each unit (e.g. [0, 0, 1, 1, 1])
        self.shank_ids       = self.__get_shank_ids()

        # get the control position
        self.control_pos     = int(self.neo_obj.annotations['control_pos'])

        # creat lists or array with units duration, ratio, and mean waveform
        self.__get_waveinfo()

        # make dictionary of cell types and get cell type IDs
        self.cell_type_dict = {0:'MU', 1:'RS', 2:'FS', 3:'UC'}
        self.cell_type      = self.__get_celltypeID()
        self.cell_type_og   = self.cell_type

        # set time after stimulus to start analyzing
        # time_after_stim + stim_start MUST BE LESS THAN stim_stop time
        self.t_after_stim    = 0.500
        print('time after stim is set to: ' + str(self.t_after_stim))

        # find shortest baseline and trial length
        self.__find_min_times()

        # trim whisker tracking data and align it to shortest trial
        self.__trim_wt()

        # trim LFP data and align it to shortest trial
        self.__trim_lfp()

        # return a list with the number of good trials for each stimulus condition
        self.get_num_good_trials()

        # classify a trial as good if whisking occurs during a specified period.
        # a new annotation ('wsk_boolean') is added to each segment
        #self.classify_whisking_trials(threshold='user')

        # calculate rates, psths, whisking array, etc.
        self.rates()

        # create region dictionary
        self.region_dict = {0:'M1', 1:'S1'}

        # reclassify units using their mean OMI (assuming ChR2 is in PV
        # cells). This is dependent on everything above!
        self.reclassify_units()

    def __sort_units(self, neo_obj):
        '''
        Units are saved out of order in the neo object.
        This reorders units by experiment FID, shank, and depth across all segments.
        '''
        for i, seg in enumerate(neo_obj.segments):
            fid, shank, depth = list(), list(), list()

            for spike in seg.spiketrains:
                fid.append(spike.annotations['fid'])
                shank.append(spike.annotations['shank'])
                # would freak out if you tried to index into an array with an
                # nan because it was not an array
                if np.isnan(spike.annotations['depth']):
                    depth.append(np.asarray(-1))
                else:
                    depth.append(spike.annotations['depth'][0])

            spike_msr_sort_inds = np.lexsort((tuple(depth), tuple(shank), tuple(fid)))
            sorted_spiketrains = [seg.spiketrains[j] for j in spike_msr_sort_inds]
            for k, ordered_spiketrain in enumerate(sorted_spiketrains):
                neo_obj.segments[i].spiketrains[k] = ordered_spiketrain

        self.depths = depth

    def __get_shank_ids(self):
        '''
        Return shank IDs for each unit
        '''
        shank_ids = np.zeros((self.num_units, ))
        for k, shank_name in enumerate(self.shank_names):
            for j, unit in enumerate(self.neo_obj.segments[0].spiketrains):
                if unit.annotations['shank'] == shank_name:
                    shank_ids[j] = k
        return shank_ids

    def __get_shank_depths(self):
        '''Find depths of each shank and add it to self.shank_depths'''
        depth = list()
        for shank in self.shank_names:
            depth_temp0 = 0
            for spikes in self.neo_obj.segments[0].spiketrains:
                if spikes.annotations['shank'] == shank:
                    depth_temp1 = spikes.annotations['depth']
                    if depth_temp0 < depth_temp1:
                        depth_temp0 = depth_temp1
            depth.append(depth_temp0)
        return depth

    def __get_waveinfo(self):
        '''gets waveform duration, ratio, and mean waveform for each unit'''
        duration = list()
        ratio    = list()
        waves    = list()
        for spikes in self.neo_obj.segments[0].spiketrains:
            duration.append(spikes.annotations['duration'])
            ratio.append(spikes.annotations['ratio'])
            waves.append(spikes.annotations['waveform'])
        self.duration = duration
        self.ratio    = ratio
        self.waves    = np.asarray(waves).squeeze()

    def __get_celltypeID(self):
        '''
        Put celltype IDs in an array that corresponds to the unit order
        '''
        cell_type = list()
        for spike in self.neo_obj.segments[0].spiketrains:
            try:
                cell_type.append(self.cell_type_dict[spike.annotations['cell_type'][0]])
            except:
                cell_type.append(self.cell_type_dict[spike.annotations['cell_type']])

        return np.asarray(cell_type)

    def __find_min_times(self):
        print('\n-----finding minimum trial lengths----')
        # iterate through all trials and find smallest baseline and smallest
        # post-trial time
        for k, seg in enumerate(self.neo_obj.segments):
            # get baseline and stimulus period times for this trial
            stim_start = seg.annotations['stim_times'][0] + self.t_after_stim
            stim_stop  = seg.annotations['stim_times'][1]
            base_start = seg.annotations['stim_times'][0] - (stim_stop - stim_start)
            base_stop  = seg.annotations['stim_times'][0]
            baseline_length = base_stop - base_start # time before stimulus

            # iterate through all units and find longest trial length
            temp_max = 0
            for unit, spike_train in enumerate(seg.spiketrains):
                if spike_train.t_stop > temp_max:
                    temp_max = np.asarray(spike_train.t_stop)
            # temp_max = temp_max - trial_start_time
            temp_max -= np.asarray(seg.annotations['stim_times'][0]) # time after stimulus

            if k == 0:
                min_tbefore_stim = baseline_length
                min_tafter_stim = temp_max

            if baseline_length < min_tbefore_stim:
                min_tbefore_stim = baseline_length

            if temp_max < min_tafter_stim:
                min_tafter_stim = temp_max

        self.min_tbefore_stim = np.asarray(min_tbefore_stim)
        self.min_tafter_stim  = np.asarray(min_tafter_stim)
        print('smallest baseline period (time before stimulus): {0}\nsmallest trial length (time after stimulus): {1}'.format(str(min_tbefore_stim), str(min_tafter_stim)))

    def __trim_wt(self):
        '''
        Trim whisker tracking arrays to the length of the shortest trial.
        Time zero of the wt time corresponds to stimulus onset.
        '''
        print('\n-----__trim_wt-----')

        fps        = 500.0
        wt_boolean = False
        for anlg in self.neo_obj.segments[0].analogsignals:
            if anlg.name == 'angle':
                wt_boolean = True

        if wt_boolean:

            print('whisker tracking data found! trimming data to be all the same length in time')
            # make time vector for whisker tracking data
            num_samples = int( (self.min_tafter_stim + self.min_tbefore_stim)*fps ) # total time (s) * frames/sec
            wt_indices  = np.arange(num_samples) - int( self.min_tbefore_stim * fps )
            wtt         = wt_indices / fps

            for i, seg in enumerate(self.neo_obj.segments):
                for k, anlg in enumerate(seg.analogsignals):

                    # find number of samples in the trial
                    if anlg.name == 'angle':
                        num_samp = len(anlg)

                        # get stimulus onset
                        stim_start = seg.annotations['stim_times'][0]

                        # slide indices window over
                        # get the frame that corresponds to the stimulus time
                        # and add it to wt_indices.
                        good_inds = wt_indices + int( stim_start*fps )

                        if i == 0:
                            min_trial_length = len(good_inds)
                        elif min_trial_length > len(good_inds):
                            warnings.warn('**** MINIMUM TRIAL LENGTH IS NOT THE SAME ****\n\
                                    LINE 208 __trim_wt')

                    if anlg.name == 'angle' or \
                            anlg.name == 'set-point' or\
                            anlg.name == 'amplitude' or\
                            anlg.name == 'phase' or\
                            anlg.name == 'velocity'or\
                            anlg.name == 'whisking':
                                if num_samp > len(good_inds):
                                    self.neo_obj.segments[i].analogsignals[k] = anlg[good_inds]
                                else:
                                    warnings.warn('\n**** length of whisker tracking signals is smaller than the length of the good indices ****\n'\
                                            + '**** this data must have already been trimmed ****')

            self.wtt          = wtt
            self.wt_boolean   = wt_boolean
            self._wt_min_samp = num_samples
        else:
            print('NO WHISKER TRACKING DATA FOUND!\nSetting wt_boolean to False'\
                    '\nuse runspeed to classify trials')
            self.wt_boolean = wt_boolean

    def __trim_lfp(self):
        '''
        Trim LFP arrays to the length of the shortest trial.
        Time zero of the LFP time corresponds to stimulus onset.
        '''
        print('\n-----__trim_lfp-----')

        chan_per_shank = list()
        for anlg in self.neo_obj.segments[0].analogsignalarrays:
            if 'LFPs' in anlg.name:
                # get the sampling rate
                lfp_boolean = True
                chan_per_shank.append(anlg.shape[1])
                if anlg.sampling_rate.units == pq.kHz:
                    sr = np.asarray(anlg.sampling_rate)*1000.0
                elif anlg.sampling_rate.units == pq.Hz:
                    sr = np.asarray(anlg.sampling_rate)*1.0

        if lfp_boolean:

            print('LFP data found! trimming data to be all the same length in time')
            # make time vector for LFP data
            num_samples = int( (self.min_tafter_stim + self.min_tbefore_stim)*sr ) # total time (s) * samples/sec
            lfp_indices = np.arange(num_samples) - int( self.min_tbefore_stim * sr )
            lfp_t       = lfp_indices / sr

            for i, seg in enumerate(self.neo_obj.segments):
                for k, anlg in enumerate(seg.analogsignalarrays):

                    # find number of samples in the trial
                    num_samp = len(anlg)

                    # get stimulus onset
                    stim_start = seg.annotations['stim_times'][0]

                    # slide indices window over
                    # get the frame that corresponds to the stimulus time
                    # and add it to lfp_indices.
                    good_inds = lfp_indices + int( stim_start*sr )

                    if i == 0:
                        min_trial_length = len(good_inds)
                    elif min_trial_length > len(good_inds):
                        warnings.warn('**** MINIMUM TRIAL LENGTH IS NOT THE SAME ****\n\
                                LINE 208 __trim_lfp')

                    if num_samp > len(good_inds):
                        self.neo_obj.segments[i].analogsignalarrays[k] = anlg[good_inds, :]
                    else:
                        warnings.warn('\n**** length of LFPs is smaller than the length of the good indices ****\n'\
                                + '**** this data must have already been trimmed ****')

            self.lfp_t          = lfp_t
            self.lfp_boolean    = lfp_boolean
            self._lfp_min_samp  = num_samples
            self.chan_per_shank = chan_per_shank
        else:
            print('NO LFP DATA FOUND!\nSetting lfp_boolean to False')
            self.lfp_boolean = lfp_boolean

    def get_num_good_trials(self, kind='run_boolean'):
        '''
        Return a list with the number of good trials for each stimulus condition
        And the specified annotations to use.
        kind can be set to either 'wsk_boolean' or 'run_boolean' (default)
        '''
        num_good_trials = list()
        for stim_id in self.stim_ids:
            run_count = 0
            for trial in self.neo_obj.segments:
                if trial.annotations['trial_type'] == stim_id and trial.annotations[kind]:
                    run_count += 1
            num_good_trials.append(run_count)
        self.num_good_trials = num_good_trials

    def classify_whisking_trials(self, threshold='user'):
        '''
        Classify a trial as good if whisking occurs during a specified period.
        A trial is considered whisking if the mouse was whisking during the
        baseline period and the stimulus period.

        threshold can take the values 'median' or 'user' (default)
        A new annotation ('wsk_boolean') is added to each segment
        '''

        if self.wt_boolean:
            print('whisker tracking data found! trimming data to be all the same length in time')
            # make "whisking" distribution and compute threshold
            print('\n-----classify_whisking_trials-----')
            #wsk_dist = np.empty((self._wt_min_samp, 1))
            wsk_dist = list()
            for i, seg in enumerate(self.neo_obj.segments):
                for k, anlg in enumerate(seg.analogsignals):
                    if anlg.name == 'whisking':
                        #wsk_dist = np.append(wsk_dist, anlg.reshape(-1, 1), axis=1) # reshape(-1, 1) changes array to a 2d array (e.g. (1978,) --> (1978, 1)
                        wsk_dist.extend(anlg.tolist()) # reshape(-1, 1) changes array to a 2d array (e.g. (1978,) --> (1978, 1)
            #wsk_dist = np.ravel(wsk_dist[:, 1:])
            wsk_dist = np.ravel(np.asarray(wsk_dist))
            wsk_dist = wsk_dist[~np.isnan(wsk_dist)]

            # plot distribution
            print('making "whisking" histogram')
            sns.set(style="ticks")
            f, (ax_box, ax_hist) = sns.plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (0.15, 0.85)})
            sns.boxplot(wsk_dist, vert=False, ax=ax_box)
            sns.distplot(wsk_dist, ax=ax_hist)
            ax_box.set(yticks=[])
            sns.despine(ax=ax_hist)
            sns.despine(ax=ax_box, left=True)
            sns.plt.xlim(70, 180)
            sns.plt.show()

            # select threshold
            if threshold is 'median':
                thresh = np.median(wsk_dist)
            elif threshold is 'user':
                thresh = int(raw_input("Enter a threshold value: "))

            plt.close(f)
            del wsk_dist

            wtt = self.wtt
            # trimming whisking data sets negative values to baseline period
            wsk_base_ind = wtt < 0
            # min_tbefore is the min baseline length (equal to the stimulus
            # period to be analyzed) offset by t_after_stim
            wsk_stim_ind = np.logical_and( wtt > self.t_after_stim, wtt < (self.min_tbefore_stim + self.t_after_stim) )
            for i, seg in enumerate(self.neo_obj.segments):

                for k, anlg in enumerate(seg.analogsignals):
                    if anlg.name == 'whisking':
                        wsk = anlg.reshape(-1, 1)
                        base_high  = np.sum(wsk[wsk_base_ind] > thresh)
                        stim_high  = np.sum(wsk[wsk_stim_ind] > thresh)
                        total_high = base_high + stim_high
                        total_samp = np.sum(wsk_base_ind) + np.sum(wsk_stim_ind)
                        fraction_high = float(total_high)/float(total_samp)

                        if fraction_high > 0.8:
                            self.neo_obj.segments[i].annotations['wsk_boolean'] = True
                        else:
                            self.neo_obj.segments[i].annotations['wsk_boolean'] = False
        else:
            print('NO WHISKER TRACKING DATA FOUND!\nuse runspeed to classify trials')

    def __make_alpha_kernel(self, resolution=0.025):
        '''Build alpha kernel with specified 25msec (default) resolution'''
        alpha = 1.0/resolution
        tau   = np.arange(0,1/alpha*10, 0.001)
        alpha_kernel = alpha**2*tau*np.exp(-alpha*tau)
        return alpha_kernel

    def get_protraction_times(self):
        # TODO TODO
        # CLEAN THIS UP
        # WORK IN PROGRESS
        # TODO TODO
        #                    condition 1                    condition 2                     condition n
        # [  [ [phase trial 1], [phase trial 2],...,[phase trial i]], [ [phase trial 1], [phs trial i]], ..., [condition ] ] ]
        # phs[condition i][trial j][timestamp k]
        phs_cond = list()
        for cond in self.wt:
            phs_trial = list()
            for k in range(cond.shape[2]):
                phs_timestamps = list()
                phase = cond[:, 3, k]
                # find zero crossings
                zero_crossings = numpy.where(numpy.diff(numpy.sign(phase)))[0]
                phs_trial.append(timestamps[zero_crossings])
            phs_cond.append(phs_trial)
        self.protraction_times = phs_cond

    def update_t_after_stim(self, t_after_stim):
        self.t_after_stim = t_after_stim
        self.classify_whisking_trials(threshold='user')
        self.rates()

    def get_annotations_index(self, key, value):
        '''Returns trial index for the given key value pair'''
        stim_index = [ index for index, segment in enumerate(self.neo_obj.segments) \
                if segment.annotations[key] == value]
        return stim_index

    def rates(self, psth_t_start= -0.500, psth_t_stop=2.000, kind='run_boolean'):
        '''
        rates computes the absolute and evoked firing rate and counts for the
        specified stimulus period. The time to start analyzing after the stimulus
        start time is specified by neuro.t_after_stim. The baseline period to
        be analyzed is taken to be the same size as the length of the stimulus
        period.

        MAKE SURE THAT THE EXPERIMENT WAS DESIGNED TO HAVE A LONG ENOUGH BASELINE
        THAT IT IS AT LEAST AS LONG AS THE STIMULUS PERIOD TO BE ANALYZED

        kind can be set to either 'wsk_boolean' or 'run_boolean' (default)
        '''

        print('\n-----computing rates----')
        absolute_rate   = list()
        evoked_rate     = list()
        absolute_counts = list()
        evoked_counts   = list()
        binned_spikes   = list()
        psth            = list()

        # make whisker tracking list wt
        if self.wt_boolean:
            wt          = list()
        if kind == 'wsk_boolean' and not self.wt_boolean:
            warnings.warn('**** NO WHISKER TRACKING DATA AVAILABLE ****\n\
                    using run speed to select good trials')
            kind = 'run_boolean'
        elif kind == 'wsk_boolean' and self.wt_boolean:
            print('using whisking to find good trials')
            self.get_num_good_trials(kind='wsk_boolean')
        elif kind == 'run_boolean':
            print('using running to find good trials')
            self.get_num_good_trials(kind='run_boolean')

        # make bins for rasters and PSTHs
        bins = np.arange(-self.min_tbefore_stim, self.min_tafter_stim, 0.001)
#        bins = np.arange(psth_t_start, psth_t_stop, 0.001)
        alpha_kernel = self.__make_alpha_kernel()
        self._bins = bins
        self.bins_t = bins[0:-1]

        # preallocation loop
        for k, trials_ran in enumerate(self.num_good_trials):
                absolute_rate.append(np.zeros((trials_ran, self.num_units)))
                evoked_rate.append(np.zeros((trials_ran, self.num_units)))
                absolute_counts.append(np.zeros((trials_ran, self.num_units)))
                evoked_counts.append(np.zeros((trials_ran, self.num_units)))
                binned_spikes.append(np.zeros((bins.shape[0]-1, trials_ran, self.num_units)))
                psth.append(np.zeros((bins.shape[0]-1,trials_ran, self.num_units))) # samples x trials x units

                if self.wt_boolean:
                    wt.append(np.zeros((self.wtt.shape[0], 6, trials_ran)))

        for stim_ind, stim_id in enumerate(self.stim_ids):
            good_trial_ind = 0

            for trial in self.neo_obj.segments:
                if trial.annotations['trial_type'] == stim_id and trial.annotations[kind]:

                    # organize whisker tracking data by trial type
                    if self.wt_boolean:
                        k = 0
                        for anlg in trial.analogsignals:
                            if anlg.name == 'angle' or \
                                    anlg.name == 'set-point' or\
                                    anlg.name == 'amplitude' or\
                                    anlg.name == 'phase' or\
                                    anlg.name == 'velocity'or\
                                    anlg.name == 'whisking':
                                        wt[stim_ind][:, k, good_trial_ind] = anlg[:]
                                        k += 1

                    # get baseline and stimulus period times for this trial
                    stim_start = trial.annotations['stim_times'][0] + self.t_after_stim
                    stim_stop  = trial.annotations['stim_times'][1]
                    base_start = trial.annotations['stim_times'][0] - (stim_stop - stim_start)
                    base_stop  = trial.annotations['stim_times'][0]

                    # iterate through all units and count calculate various
                    # spike rates (e.g. absolute firing and evoked firing rates
                    # and counts)
                    for unit, spike_train in enumerate(trial.spiketrains):
                        spk_times = np.asarray(spike_train.tolist())

                        # bin spikes for rasters (time 0 is stimulus start)
                        spk_times_relative = spk_times - trial.annotations['stim_times'][0]
                        counts = np.histogram(spk_times_relative, bins=bins)[0]
                        binned_spikes[stim_ind][:, good_trial_ind, unit] = counts

                        # convolve binned spikes to make PSTH
                        psth[stim_ind][:, good_trial_ind, unit] =\
                                np.convolve(counts, alpha_kernel)[:-alpha_kernel.shape[0]+1]

                        # calculate absolute and evoked counts
                        abs_count = np.logical_and(spk_times > stim_start, spk_times < stim_stop).sum()
                        evk_count   = (np.logical_and(spk_times > stim_start, spk_times < stim_stop).sum()) - \
                                (np.logical_and(spk_times > base_start, spk_times < base_stop).sum())
                        absolute_counts[stim_ind][good_trial_ind, unit] = abs_count
                        evoked_counts[stim_ind][good_trial_ind, unit]   = evk_count

                        # calculate absolute and evoked rate
                        abs_rate = float(abs_count)/float((stim_stop - stim_start))
                        evk_rate = float(evk_count)/float((stim_stop - stim_start))
                        absolute_rate[stim_ind][good_trial_ind, unit] = abs_rate
                        evoked_rate[stim_ind][good_trial_ind, unit]   = evk_rate


                    good_trial_ind += 1

        self.abs_rate      = absolute_rate
        self.abs_count     = absolute_counts
        self.evk_rate      = evoked_rate
        self.evk_count     = evoked_counts
        self.binned_spikes = binned_spikes
        self.psth          = psth

        if self.wt_boolean:
            self.wt        = wt

    def reclassify_units(self):
        '''use OMI and wave duration to reclassify units'''

        new_labels = list()
        for index in range(len(self.cell_type_og)):
            region   = int(self.shank_ids[index])
            label    = self.cell_type[index]
            ratio    = self.ratio[index]
            duration = self.duration[index]

            meanr     = np.array([np.mean(k[:, index]) for k in self.evk_rate])
            meanr_abs = np.array([np.mean(k[:, index]) for k in self.abs_rate])
            omi_s1light = (meanr_abs[self.control_pos:self.control_pos+9] - meanr_abs[:self.control_pos]) / \
                    (meanr_abs[self.control_pos:self.control_pos+9] + meanr_abs[:self.control_pos])
            omi_m1light = (meanr_abs[self.control_pos+9:self.control_pos+9+9] - meanr_abs[:self.control_pos]) / \
                    (meanr_abs[self.control_pos+9:self.control_pos+9+9] + meanr_abs[:self.control_pos])

            # reclassify M1 units

            if label == 'MU':
                new_labels.append('MU')

            elif self.region_dict[region] == 'M1':

                if omi_m1light.mean() < 0:# and duration > 0.36
                    new_labels.append('RS')

                elif omi_m1light.mean() > 0:# and duration < 0.34
                    new_labels.append('FS')
                else:
                    new_labels.append(label)

            elif self.region_dict[region] == 'S1':

                if omi_s1light.mean() < 0:# and duration > 0.36:
                    new_labels.append('RS')

                elif omi_s1light.mean() > 0:# and duration < 0.34:
                    new_labels.append('FS')
                else:
                    new_labels.append(label)

        self.cell_type = new_labels

    def get_lfps(self, kind='run_boolean'):
        lfps = [list() for x in range(len(self.shank_names))]
        # preallocation loop
        for shank in range(len(self.shank_names)):
            for k, trials_ran in enumerate(self.num_good_trials):
                lfps[shank].append(np.zeros(( self._lfp_min_samp, self.chan_per_shank[shank], trials_ran )))

        for shank in range(len(self.shank_names)):
            for stim_ind, stim_id in enumerate(self.stim_ids):
                good_trial_ind = 0

                for trial in self.neo_obj.segments:
                    if trial.annotations['trial_type'] == stim_id and trial.annotations[kind]:
                        lfps[shank][stim_ind][:, :, good_trial_ind] = trial.analogsignalarrays[shank]
                        good_trial_ind += 1
        self.lfps = lfps

    def get_psd(self, input_array, sr):
        '''
        compute PSD of input array sampled at sampling rate sr
        input_array: a samples x trial array
        sr: sampling rate of input signal

        Returns x, and mean y and sem y
        '''
        f_temp = list()
        num_trials = input_array.shape[1]
        for trial in range(num_trials):
            f, Pxx_den = sp.signal.periodogram(input_array[:, trial], sr)
            if trial == 0:
                frq_mat_temp = np.zeros((Pxx_den.shape[0], num_trials))
            frq_mat_temp[:, trial] = Pxx_den

        return f, frq_mat_temp

    def plot_freq(self, f, frq_mat_temp, color='k', error='sem'):
        ax = plt.gca()
        mean_frq = np.mean(frq_mat_temp, axis=1)
        se       = sp.stats.sem(frq_mat_temp, axis=1)

        # inverse of the CDF is the percentile function. ppf is the percent point funciton of t.
        if error == 'ci':
            err = se*sp.stats.t.ppf((1+0.95)/2.0, frq_mat_temp.shape[1]-1) # (1+1.95)/2 = 0.975
        elif error == 'sem':
            err = se

        plt.plot(f, mean_frq, color)
        plt.fill_between(f, mean_frq - err, mean_frq + err, facecolor=color, alpha=0.3)
        ax.set_yscale('log')
        return ax

    def make_design_matrix(self, rate_type='evk_count', trode=None, trim_trials=True):
        '''make design matrix for classification and regressions'''

        print('\n-----make design matrix----')
        min_trials     = np.min(self.num_good_trials)
        num_cond       = len(self.stim_ids)
        kind_dict      = {'abs_rate': 0, 'abs_count': 1, 'evk_rate': 2, 'evk_count': 3}
        kind_of_tuning = [self.abs_rate, self.abs_count, self.evk_rate, self.evk_count]
        rates          = kind_of_tuning[kind_dict[rate_type]]

        if trode:
            trode_inds = np.where(self.shank_ids == trode-1)[0]
            num_units = len(trode_inds)
        else:
            print('Collecting data from electrode {}'.format(str(trode)))
            trode_inds = np.where(self.shank_ids >= 0)[0]
            num_units  = self.num_units

        if trim_trials:
            X = np.zeros((num_cond*min_trials, num_units))
            y = np.ones((num_cond*min_trials, ))
        else:
            X = np.zeros((np.sum(self.num_good_trials), num_units))
            y = np.ones((np.sum(self.num_good_trials), ))

        last_t_ind = 0
        for k, cond in enumerate(rates):
            if trim_trials:
                X[min_trials*k:min_trials*(k+1)] = cond[0:min_trials, trode_inds]
                y[min_trials*k:min_trials*(k+1)] = k*y[min_trials*k:min_trials*(k+1)]
            else:
                min_trials = cond.shape[0]
                X[last_t_ind:(min_trials + last_t_ind)] = cond[:, trode_inds]
                y[last_t_ind:(min_trials + last_t_ind)] = k*y[last_t_ind:(min_trials + last_t_ind)]
                last_t_ind = min_trials + last_t_ind

        return X, y

    def get_burst_isi(self, kind='run_boolean'):
        '''
        Compute the interspike interval for spikes during the stimulus period.
        get_burst_isi creates a list that has n_stimulus_types entries. Each
        stimulus type has a list which contains a numpy array for each unit.

        These values can be used to identify bursting activity.
        '''

        if self.wt_boolean:
            wt          = list()
        if kind == 'wsk_boolean' and not self.wt_boolean:
            warnings.warn('**** NO WHISKER TRACKING DATA AVAILABLE ****\n\
                    using run speed to select good trials')
            kind = 'run_boolean'
        elif kind == 'wsk_boolean' and self.wt_boolean:
            print('using whisking to find good trials')
            self.get_num_good_trials(kind='wsk_boolean')
        elif kind == 'run_boolean':
            print('using running to find good trials')
            self.get_num_good_trials(kind='run_boolean')

        t_after_stim = self.t_after_stim

        # create list with entries for each condition
        bisi_list = [list() for stim in self.stim_ids]
        isi_list  = [list() for stim in self.stim_ids]

        # iterate through all stimulus IDs
        for stim_ind, stim_id in enumerate(self.stim_ids):
            good_trial_ind = 0
            # this is a list of numpy arrays. There is an array for each unit
            bisi_np_list = [np.empty((1, 4)) for k in range(self.num_units)]
            isi_np_list  = [np.empty((1, 3)) for k in range(self.num_units)] # ISI, stim_ID, good_trial_index

            # iterate through all neo trial segments
            for trial in self.neo_obj.segments:

                # if a trial segment is from the current stim_id and it is a
                # running trial get the spike ISIs.
                if trial.annotations['trial_type'] == stim_id and trial.annotations[kind]:

                    # get the stimulus start and stop times for this trial
                    stim_start = trial.annotations['stim_times'][0]
                    stim_stop  = trial.annotations['stim_times'][1]

                    # iterate through all unit's spike trains. find times units fired
                    # during the stimulus period and calculate the inter-spike-
                    # interval between them. This will be used to determine
                    # bursting
                    for unit, spike_train in enumerate(trial.spiketrains):

                        # get spike tiimes during stimulus period
                        spk_times_all = np.asarray(spike_train.tolist()) # spike times for current spike train
                        spk_times     = spk_times_all[np.logical_and(spk_times_all > stim_start, spk_times_all < stim_stop)] # spike times during stimulus period
                        num_spk_times = len(spk_times)
                        if num_spk_times > 3:
                            # create the burst isi array (bisi) if there are  more
                            # than 3 spikes all spikes except the first and last
                            # will be analyze. this way we don't measure ISIs
                            # outside of the stimulus period

                            bisi_temp = np.zeros((num_spk_times-2, 4))
                            for k in range(1, num_spk_times - 1):
                                t_before = spk_times[k] - spk_times[k-1]
                                t_after  = spk_times[k+1] - spk_times[k]
                                bisi_temp[k-1, 0] = t_before
                                bisi_temp[k-1, 1] = t_after
                                bisi_temp[k-1, 2] = stim_id
                                bisi_temp[k-1, 3] = good_trial_ind # identifies spikes from the same trial

                            bisi_np_list[unit] = np.concatenate((bisi_np_list[unit], bisi_temp), axis=0)

                        # compute ISIs and add it to isi_np_list
                        if num_spk_times > 2:
                            isi_temp  = np.zeros((num_spk_times-1, 3))
                            isi_temp[:, 0] = np.diff(spk_times)
                            isi_temp[:, 1] = np.ones((num_spk_times - 1, ))*stim_id
                            isi_temp[:, 2] = np.ones((num_spk_times - 1, ))*good_trial_ind

                            isi_np_list[unit] = np.concatenate((isi_np_list[unit], isi_temp), axis=0)

                    good_trial_ind += 1

            # remove first row which is junk
            for k, nparray in enumerate(bisi_np_list):
                bisi_np_list[k] = nparray[1::, :]
                isi_np_list[k]  = nparray[1::, :]
            bisi_list[stim_ind] = bisi_np_list
            isi_list[stim_ind]  = isi_np_list

        self.bisi_list = bisi_list
        self.isi_list  = isi_list

    def get_selectivity(self):
        if hasattr(self, 'abs_count') is False:
            self.rates()
        control_pos = self.control_pos
        num_manipulations = self.stim_ids.shape[0]/control_pos
        sel_mat = np.zeros((self.num_units, num_manipulations))

        for manip in range(num_manipulations):
            for unit in range(self.num_units):
                meanr = [np.mean(k[:, unit]) for k in self.abs_count]
                x = np.asarray(meanr[(manip*control_pos):((manip+1)*control_pos-1)])
                # calculate selectivity for unit during manipulation
                sel_mat[unit, manip] = \
                        1 - ((np.linalg.norm(x/np.max(x))- 1)/\
                        (np.sqrt(x.shape[0]) - 1))

        self.selectivity = sel_mat

    def plot_tuning_curve(self, unit_ind=[], kind='abs_count', axis=None):
        '''
        make_simple_tuning_curve allows one to specify what type of tuning
        curve to plot as well as which unit for a single tuning curve or all
        units for a subplot of tuning curves

        Kinds of tuning curves:
        Absolute rate:   'abs_rate'
        Absolute counts: 'abs_count'
        Evoked rate:     'evk_rate'
        Evoked counts:   'evk_count'
        '''

        # if the rates have not been calculated do that now
        if hasattr(self, 'abs_rate') is False:
            self.rates()
        # dictionary of types of plots. allows user to specify what type of
        # tuning curve should be plotted.
        kind_dict      = {'abs_rate': 0, 'abs_count': 1, 'evk_rate': 2, 'evk_count': 3}
        kind_of_tuning = [self.abs_rate, self.abs_count, self.evk_rate, self.evk_count]

        control_pos = self.control_pos
        # setup x-axis with correct labels and range
        pos = range(1,control_pos)
        x_vals = range(1, control_pos+1)
        labels = [str(i) for i in pos]; labels.append('NC')
        line_color = ['k','r','b']

        # if tuning curves from one unit will be plotted on a given axis
        if unit_ind:

            if axis == None:
                ax = plt.gca()
            else:
                ax = axis

            # compute the means and standard errors
            meanr = [np.mean(k[:, unit_ind]) for k in kind_of_tuning[kind_dict[kind]]]
            stder = [np.std(k[:, unit_ind]) / np.sqrt(k[:, unit_ind].shape[0]) for k in kind_of_tuning[kind_dict[kind]]]
            for control_pos_count, first_pos in enumerate(range(0, len(self.stim_ids), control_pos )):
                ax.errorbar(pos[0:control_pos-1],\
                        meanr[(control_pos_count*control_pos):((control_pos_count+1)*control_pos-1)],\
                        yerr=stder[(control_pos_count*control_pos):((control_pos_count+1)*control_pos-1)],\
                        fmt=line_color[control_pos_count], marker='o', markersize=8.0, linewidth=2)
                # plot control position separately from stimulus positions
                ax.errorbar(control_pos, meanr[(control_pos_count+1)*control_pos-1], yerr=stder[(control_pos_count+1)*control_pos-1],\
                        fmt=line_color[control_pos_count], marker='o', markersize=8.0, linewidth=2)

        # if all tuning curves from all units are to be plotted
        else:
            # determine number of rows and columns in the subplot
            unit_ind = range(self.num_units)
            num_rows, num_cols = 3, 3

            num_manipulations = len(self.stim_ids)/control_pos # no light, light 1 region, light 2 regions

            unit_count, plot_count = 0, 0
            fig = plt.subplots(num_rows, num_cols, figsize=(14, 10))
            for unit in unit_ind:
                meanr = [np.mean(k[:, unit]) for k in kind_of_tuning[kind_dict[kind]]]
                stder = [np.std(k[:, unit]) / np.sqrt(k[:, unit].shape[0]) for k in kind_of_tuning[kind_dict[kind]]]
                for control_pos_count, first_pos in enumerate(range(0, len(self.stim_ids), control_pos )):
                    # compute the means and standard errors

                    ax = plt.subplot(num_rows, num_cols, plot_count+1)
                    # plot stimulus positions separately from control so the
                    # control position is not connected with the others
                    plt.errorbar(pos[0:control_pos-1],\
                            meanr[(control_pos_count*control_pos):((control_pos_count+1)*control_pos-1)],\
                            yerr=stder[(control_pos_count*control_pos):((control_pos_count+1)*control_pos-1)],\
                            fmt=line_color[control_pos_count], marker='o', markersize=8.0, linewidth=2)
                    # plot control position separately from stimulus positions
                    plt.errorbar(control_pos, meanr[(control_pos_count+1)*control_pos-1], yerr=stder[(control_pos_count+1)*control_pos-1],\
                            fmt=line_color[control_pos_count], marker='o', markersize=8.0, linewidth=2)

                plt.title('shank: ' + self.shank_names[self.shank_ids[unit]] + \
                        ' depth: ' + str(self.neo_obj.segments[0].spiketrains[unit].annotations['depth']) + \
                        '\ncell type: ' + str(self.cell_type[unit]))
                plt.plot([0, control_pos+1],[0,0],'--k')
                plt.xlim(0, control_pos+1)
                plt.ylim(plt.ylim()[0]-1, plt.ylim()[1]+1)
                ax.set_xticks([])
                plt.xticks(x_vals, labels)
                plt.xlabel('Bar Position', fontsize=8)
                plt.show()
                # count after each plot is made
                plot_count += 1
                unit_count += 1

                if unit_count == len(unit_ind) and plot_count == (num_rows*num_cols):
                    print('finished')
                    plt.subplots_adjust(left=0.04, bottom=0.07, right=0.99, top=0.92, wspace=0.17, hspace=0.35)
                    plt.suptitle(kind)
                    plt.show()
                    return ax
                elif unit_count == len(unit_ind):
                    print('finished')
                    plt.subplots_adjust(left=0.04, bottom=0.07, right=0.99, top=0.92, wspace=0.17, hspace=0.35)
                    plt.suptitle(kind)
                    plt.show()
                elif plot_count == (num_rows*num_cols):
                    print('made a new figure')
                    plt.subplots_adjust(left=0.04, bottom=0.07, right=0.99, top=0.92, wspace=0.17, hspace=0.35)
                    plt.suptitle(kind)
                    plt.show()
                    # create a new plot
                    if plot_count != len(unit_ind):
                        fig = plt.subplots(num_rows, num_cols, figsize=(14, 10))
                    plot_count = 0

    def plot_raster(self, unit_ind=0, trial_type=0, axis=None, burst=True):
        '''
        Makes a raster plot for the given unit index and trial type.
        If called alone it will plot a raster to the current axis. This function
        is called by plot_all_rasters and returns an axis handle to the current
        subplot. This allows plot_all_rasters to plot rasters in the appropriate
        subplots.
        '''
        # if the rates have not been calculated do that now
        if hasattr(self, 'binned_spikes') is False:
            print('Spikes have not been binned! Binning data now.')
            self.rates()

        print('Making raster for unit {} and trial_type {}'.format(unit_ind, trial_type))
        if axis == None:
            ax = plt.gca()
        else:
            ax = axis

        count_mat = self.binned_spikes[trial_type][:, :, unit_ind] # returns bins x num_trials array

        for trial in range(count_mat.shape[1]):
            trial_inds = np.where(count_mat[:, trial] > 0)[0]
            spike_times = self._bins[trial_inds]
            ax.vlines(spike_times, trial, trial+1, color='k')

            if burst:
                burst_times = list()
                data = spike_times
                if len(data) > 3:
                    start, length, RS = ranksurprise.burst(data, limit=50e-3, RSalpha=0.1)
                    for k in range(len(start)):
                        burst_times.extend(data[start[k]:(start[k]+length[k])])

                    if len(burst_times) > 0:
                        ax.vlines(burst_times, trial, trial+1, 'r', linestyles='dashed', linewidth=0.5)

        ax.hlines(trial+1, 0, 1.5, color='k')
        ax.set_xlim(self._bins[0], self._bins[-1])
        ax.set_ylim(0, trial+1)

        return ax

    def plot_psth(self, axis=None, unit_ind=0, trial_type=0, error='ci', color='k'):
        '''
        Makes a PSTH plot for the given unit index and trial type.
        If called alone it will plot a PSTH to the current axis. This function
        is called by plot_all_PSTHs and returns an axis handle to the current
        subplot. This allows plot_all_PSTHs to plot PSTHs in the appropriate
        subplots.
        '''
        # if the rates have not been calculated do that now
        if hasattr(self, 'psth') is False:
            print('Spikes have not been binned! Binning data now.')
            self.rates()

        print('Making PSTH for unit {} and trial_type {}'.format(unit_ind, trial_type))

        if axis == None:
            ax = plt.gca()
        else:
            ax = axis

        psth_temp = self.psth[trial_type][:, :, unit_ind]
        mean_psth = np.mean(psth_temp, axis=1) # mean across all trials
        se = sp.stats.sem(psth_temp, axis=1)
        # inverse of the CDF is the percentile function. ppf is the percent point funciton of t.
        if error == 'ci':
            err = se*sp.stats.t.ppf((1+0.95)/2.0, psth_temp.shape[1]-1) # (1+1.95)/2 = 0.975
        elif error == 'sem':
            err = se

        ax.plot(self._bins[0:-1], mean_psth, color)
        ax.fill_between(self._bins[0:-1], mean_psth - err, mean_psth + err, facecolor=color, alpha=0.3)
        plt.xlim(self._bins[0], self._bins[-1])

        return ax

    def plot_all_rasters(self, unit_ind=0, burst=True):
        '''
        Plots all rasters for a given unit with subplots.
        Each positions is a row and each manipulation is a column.
        '''
        num_manipulations = int(self.stim_ids.shape[0]/self.control_pos)
        subplt_indices    = np.arange(self.control_pos*num_manipulations).reshape(self.control_pos, num_manipulations)
        fig = plt.subplots(self.control_pos, num_manipulations, figsize=(6*num_manipulations, 12))

        for manip in range(num_manipulations):
            for trial in range(self.control_pos):
                plt.subplot(self.control_pos, num_manipulations, subplt_indices[trial, manip]+1)
                self.plot_raster(unit_ind=unit_ind, trial_type=(trial + self.control_pos*manip), burst=burst)

    def plot_all_psths(self, unit_ind=0, error='sem'):
        '''
        Plots all PSTHs for a given unit with subplots.
        Each positions is a row and each manipulation is a column.
        '''
        ymax = 0
        color = ['k','r','b']
        num_manipulations = int(self.stim_ids.shape[0]/self.control_pos)
        fig, ax = plt.subplots(self.control_pos, 1, figsize=(6, 12), sharex=True, sharey=True)
        plt.subplots_adjust(hspace=0.001)

        for manip in range(num_manipulations):
            for trial in range(self.control_pos):
                #plt.subplot(self.control_pos,1,trial+1)
                self.plot_psth(axis=ax[trial], unit_ind=unit_ind, trial_type=(trial + self.control_pos*manip),\
                        error=error, color=color[manip])

                if plt.ylim()[1] > ymax:
                    ymax = plt.ylim()[1]
        plt.show()

###############################################################################
######## Doesn't work in Ubuntu...figure out why ##############################
###############################################################################
#    def make_raster_movie(self, trial_type=1, run_trials=True):
#        # stim_times is the time of the stimulus onset and offset
#
#        stim_inds = self.get_annotations_index(key='trial_type', value=trial_type)
#        run_inds  = self.get_annotations_index(key='run_boolean', value=True)
#        stim_index = stim_inds and run_inds
#        num_frames = len(stim_index)
#        shank_borders = np.where(np.diff(self.shank_ids) > 0)[0]
#
#        def __get_spike_xy_coord(segment):
#            x = list()
#            y = list()
#            for unit in range(self.num_units):
#                x.extend(segment.spiketrains[unit].tolist())
#                y.extend(np.ones(len(segment.spiketrains[unit]), )*unit)
#            return np.array(x), np.array(y)
#
#        FFMpegWriter = animation.writers['ffmpeg']
#        writer = FFMpegWriter(fps=3)
#        fig = plt.figure()
#
#        with writer.saving(fig, self.neo_obj.name + '_allunitraster_trial_'  + str(trial_type) + ".mp4", 70):
#            for sind in stim_index:
#                x, y = __get_spike_xy_coord(self.neo_obj.segments[sind])
#                stim_times = self.neo_obj.segments[sind].annotations['stim_times']
#                t_start = self.neo_obj.segments[sind].spiketrains[0].t_start
#                t_stop  = self.neo_obj.segments[sind].spiketrains[0].t_stop
#
#                plt.vlines(x, y-1, y, 'k')
#                plt.vlines(np.array(stim_times), 0, self.num_units, 'r')
#                plt.hlines(shank_borders, t_start, t_stop, 'g')
#                plt.ylim(0, self.num_units)
#                plt.xlim(t_start, t_stop)
#                plt.gca().invert_yaxis()
#                plt.xlabel('time (s)')
#                plt.ylabel('unit')
#                plt.title('all unit raster')
#                writer.grab_frame()
#                plt.clf()
#        plt.close()
#
#def single_trial_all_unit_raster(segments, trial_type, stim_times, stim_ids, run_trials=True):
#    '''
#    Plot single trial rasters with all recorded units
#
#    segments: neo segments object.
#    trial_typ: trial type to analyze. int
#    stim_times: list containing onset and offset of the stimulus period (s). float
#    run_trials: boolean. whether to analyze only running trials.
#
#    OUTPUT
#    multiple raster plots
#
#
#
#    '''
#
#    num_units = len(segments[0].spiketrains)
#    stim_index = get_stimulus_index(segments, trial_type, run_trials=True)
#
#    for k, sind in enumerate(stim_index):
#
#        print('on iteration ' + str(k))
#        fig = plt.figure()
#        for unit in range(num_units):
#            plt.vlines(segments[sind].spiketrains[unit], unit, unit+1, 'k')
#        plt.vlines(np.array(stim_times)*1000, 0, plt.ylim()[1], 'r')
#        plt.show()
#
#




########## MAIN CODE ##########
########## MAIN CODE ##########
sns.set_style("whitegrid", {'axes.grid' : False})

if os.path.isdir('/Users/Greg/Documents/AdesnikLab/Data/'):
    data_dir = '/Users/Greg/Documents/AdesnikLab/Data/'
elif os.path.isdir('/media/greg/data/neuro/neo/'):
    data_dir = '/media/greg/data/neuro/neo/'

#manager = NeoHdf5IO(os.path.join(data_dir + 'FID1295_neo_object.h5'))
print(sys.argv)
fid = 'FID' + sys.argv[1]
manager = NeoHdf5IO(os.path.join(data_dir + 'FID' + sys.argv[1] + '_neo_object.h5'))
#manager = NeoHdf5IO(os.path.join(data_dir + 'FID1302_neo_object.h5'))
print('Loading...')
block = manager.read()
print('...Loading Complete!')
manager.close()

exp1 = block[0]
neuro = NeuroAnalyzer(exp1)

##### SCRATCH SPACE #####
##### SCRATCH SPACE #####

#
#plt.figure()
#lda = LinearDiscriminantAnalysis(n_components=2)
#X, y = neuro.make_design_matrix('evk_count', trode=1)
#X_r0 = X[y<8, :]
#y_r0 = y[y<8]
#X_r0 = lda.fit(X_r0, y_r0).transform(X_r0)
#plt.subplot(1,2,1)
#color=iter(cm.rainbow(np.linspace(0,1,len(np.unique(y_r0)))))
#for k in range(len(np.unique(y_r0))):
#    c = next(color)
#    plt.plot(X_r0[y_r0==k, 0], X_r0[y_r0==k, 1], 'o', c=c, label=str(k))
#plt.legend(loc='best')
#
#X, y = neuro.make_design_matrix('evk_count', trode=1)
#trial_inds = np.logical_and(y>=9, y<17) # no control position
#X_r0 = X[trial_inds, :]
#y_r0 = y[trial_inds]
#X_r0 = lda.fit(X_r0, y_r0).transform(X_r0)
#color=iter(cm.rainbow(np.linspace(0,1,len(np.unique(y_r0)))))
#plt.subplot(1,2,2)
#for k in range(len(np.unique(y_r0))):
#    c = next(color)
#    plt.plot(X_r0[y_r0==k+9, 0], X_r0[y_r0==k+9, 1], 'o', c=c, label=str(k))
#plt.legend(loc='best')
#plt.show()

#plt.figure()
#lda = LinearDiscriminantAnalysis(n_components=2)
#X, y = neuro.make_design_matrix('evk_count', trode=2)
#X_r0 = X[y<9, :]
#y_r0 = y[y<9]
#X_r0 = lda.fit(X_r0, y_r0).transform(X_r0)
#plt.subplot(1,2,1)
#color=iter(cm.rainbow(np.linspace(0,1,len(np.unique(y_r0)))))
#for k in range(9):
#    c = next(color)
#    plt.plot(X_r0[y_r0==k, 0], X_r0[y_r0==k, 1], 'o', c=c, label=str(k))
#plt.legend(loc='best')
#
#X, y = neuro.make_design_matrix('evk_count', trode=2)
#trial_inds = np.where(y>= 18)[0]
#X_r0 = X[trial_inds, :]
#y_r0 = y[trial_inds]
#X_r0 = lda.fit(X_r0, y_r0).transform(X_r0)
#color=iter(cm.rainbow(np.linspace(0,1,len(np.unique(y_r0)))))
#plt.subplot(1,2,2)
#for k in range(9):
#    c = next(color)
#    plt.plot(X_r0[y_r0==k+9*2, 0], X_r0[y_r0==k+9*2, 1], 'o', c=c, label=str(k))
#plt.legend(loc='best')
################
##### TSNE or MDS #####
################

#from sklearn import manifold
#
#plt.figure()
#X, y = neuro.make_design_matrix('evk_count', trode=1)
#X_r0 = X[y<9, :]
#y_r0 = y[y<9]
#clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
#X_r0 = clf.fit_transform(X_r0)
#plt.subplot(1,2,1)
#color=iter(cm.rainbow(np.linspace(0,1,len(np.unique(y_r0)))))
#for k in range(9):
#    c = next(color)
#    plt.plot(X_r0[y_r0==k, 0], X_r0[y_r0==k, 1], 'o', c=c, label=str(k))
#plt.legend(loc='best')
#
#X, y = neuro.make_design_matrix('evk_count', trode=1)
#trial_inds = np.logical_and(y>=9, y<18)
#X_r0 = X[trial_inds, :]
#y_r0 = y[trial_inds]
#clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
#X_r0 = clf.fit_transform(X_r0)
##model = TSNE(n_components=2, random_state=0)
##model.fit_transform(X_r0)
#color=iter(cm.rainbow(np.linspace(0,1,len(np.unique(y_r0)))))
#plt.subplot(1,2,2)
#for k in range(9):
#    c = next(color)
#    plt.plot(X_r0[y_r0==k+9, 0], X_r0[y_r0==k+9, 1], 'o', c=c, label=str(k))
#plt.legend(loc='best')


##how to get spike times: block.segments[0].spiketrains[0].tolist()
#
##############!!!!!!!!!!!!!!!!!!!!!!!#########################
##### HOW TO TRANSFORM A MATRIX SO THAT TRIALS ARE APPENDED TOGETHER
#### i.e. take 3d stack and move backwards in the 3d dimention every time we
# iterate to the next page we place the page beneath the first and then the
# third page goes beneath the second etc.
# when we reshape in numpy the 3rd dimension changes the fastest and we want to
# make it the slowest (this will put it at the bottom of the page after going
# through all the ones and twos from the first page). these are the order of
# the matrix coordinates when we use reshape:
# (1,1,1)-(1,1,2)-(1,1,3)-(1,2,1)-(1,2,2)-(1,2,3) <-look how we choose the
# third dimension the most.
# we want this pattern
# (1,1,1)-(1,2,1)-(2,1,1)-(2,2,1)-(3,1,1)-(3,2,1)-(4,1,1)-(4,2,1)-(1,1,2)-(1,2,2)-(2,1,2)-(2,1,2)-(2,2,2)-(3,1,2)
# here 2 is most frequent then 1 and lastly 3. to acheive this we have to switch
# dim 3 (the most frequent) with 1 (least frequent) and switch 1 with 2 (middle frequent)
# make this pattern 3-2-1 into this 2-1-3

# use this: a.swapaxes(0,2).swapaxes(1,2).reshape(6,2)


