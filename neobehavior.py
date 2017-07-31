#!/bin/bash
import os.path
import sys
import pandas as pd
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
#import 3rd party code found on github
import icsd
import ranksurprise
import dunn
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

    def __init__(self, neo_obj, fid):

        print('\n-----__init__-----')
        # specify where the data is
        if os.path.isdir('/Users/Greg/Documents/AdesnikLab/Data/'):
            data_dir = '/Users/Greg/Documents/AdesnikLab/Data/'
            self.exp_csv_dir = data_dir
        elif os.path.isdir('/media/greg/data/behavior/neobehavior/'):
            data_dir = '/media/greg/data/behavior/neobehavior/'
            self.exp_csv_dir = '/media/greg/data/behavior/'
        self.data_dir = data_dir

        self.fid = fid
        # segments can get out of the order they were created
        sorted_index_list = np.argsort([k.index for k in neo_obj.segments])
        neo_obj.segments  = [neo_obj.segments[k] for k in sorted_index_list]

        # add neo object to class instance
        self.neo_obj         = neo_obj

        # find stimulus IDs
        self.stim_ids        = np.sort(np.unique([k.annotations['trial_type'] for k in neo_obj.segments]))

        # get the control position
        self.control_pos     = int(self.neo_obj.annotations['control_pos'])

        # made dictionary of whisker tracking information
        self.wt_type_dict = {0:'angle', 1:'set-point', 2:'amplitude', 3:'phase', 4:'velocity', 5:'whisking'}

        # set time after stimulus to start analyzing
        # time_after_stim + stim_start MUST BE LESS THAN stim_stop time
        self.t_after_stim    = 0.000
        print('time after stim is set to: ' + str(self.t_after_stim))

        # find shortest baseline and trial length
        self.__find_min_times()

        # trim whisker tracking data and align it to shortest trial
        self.__trim_wt()

        # return a list with the number of good trials for each stimulus condition
        self.get_num_good_trials()

        # classify a trial as good if whisking occurs during a specified period.
        # a new annotation ('wsk_boolean') is added to each segment
        self.classify_whisking_trials(threshold='median')

        # calculate rates, psths, whisking array, etc.
        self.wt_organize()

    def get_exp_details_info(self, key):
        ##### LOAD IN EXPERIMENT DETAILS CSV FILE #####
        print('\n----- get_exp_details_info -----')
        fid = self.fid
        experiment_details_path = self.exp_csv_dir + 'experiment_details_behavior.csv'
        print('Loading experiment details file for FID: ' + str(self.fid) + '\nkey: ' + key)
        df_exp_det_regular_index = pd.read_csv(experiment_details_path,sep=',')
        df_exp_det_fid_index = df_exp_det_regular_index.set_index('fid')

        # check if experiment is in the experiment details csv file
        # need this info to extract electrode depth

        if fid not in df_exp_det_fid_index.index:
            warn('fid' + str(fid) + ' not found in experiment details csv file.\n\
                            Please update file and try again.')
            return None

        exp_info = df_exp_det_fid_index.loc[fid]

        if key not in exp_info:
            warn('key "' + key + '" not found in experiment details csv file.\n\
                            Setting to None.')
            return None
        else:
            return exp_info[key]

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

    def __get_waveinfo(self):
        '''gets waveform duration, ratio, and mean waveform for each unit'''

        duration = list()
        ratio    = list()
        waves    = list()
        for spikes in self.neo_obj.segments[0].spiketrains:
            if spikes.annotations.has_key('duration') is True:
                duration.append(spikes.annotations['duration'])
                ratio.append(spikes.annotations['ratio'])
                waves.append(spikes.annotations['waveform'])

        #if hasattr(spikes, 'annotations.duration') is True:
        if spikes.annotations.has_key('duration') is True:
            self.duration = duration
            self.ratio    = ratio
            self.waves    = np.asarray(waves).squeeze()
        else:
            self.duration = None
            self.ratio    = None
            self.waves    = None

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
#        # iterate through all trials and find smallest baseline and smallest
#        # post-trial time
#        for k, seg in enumerate(self.neo_obj.segments):
#            # get baseline and stimulus period times for this trial
#            stim_start = seg.annotations['stim_times'][0] + self.t_after_stim
#            stim_stop  = seg.annotations['stim_times'][1]
#            base_start = seg.annotations['stim_times'][0] - (stim_stop - stim_start)
#            base_stop  = seg.annotations['stim_times'][0]
#            baseline_length = base_stop - base_start # time before stimulus
#
#            # iterate through all units and find longest trial length
#            temp_max = 0
#            for unit, spike_train in enumerate(seg.spiketrains):
#                if spike_train.t_stop > temp_max:
#                    temp_max = np.asarray(spike_train.t_stop)
#            # temp_max = temp_max - trial_start_time
#            temp_max -= np.asarray(seg.annotations['stim_times'][0]) # time after stimulus
#
#            if k == 0:
#                min_tbefore_stim = baseline_length
#                min_tafter_stim = temp_max
#
#            if baseline_length < min_tbefore_stim:
#                min_tbefore_stim = baseline_length
#
#            if temp_max < min_tafter_stim:
#                min_tafter_stim = temp_max

        # over writes dynamic baselines and trials. Now all trials will be the
        # same length. This is what the trimming functions did anyway.
        min_tbefore_stim = self.get_exp_details_info('latency')
        min_tafter_stim  = self.get_exp_details_info('duration') - self.get_exp_details_info('latency')

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

    def reclassify_run_trials(self, time_before_stimulus= -1,\
            mean_thresh=250, sigma_thresh=150, low_thresh=200, set_all_to_true=False):
        '''
        If the neo object is still associated with the NeuroAnalyzer class this
        function uses the velocity data stored in the analogsignals array to
        reclassify running trials as running or not running.

        Two specified time windows are used to classify trials as running. One
        region during the pre-stimulus/baseline period and one during the stimulus
        period.

        set_all_to_true will set all trials to true regardless of running data.

        TODO: classify trials as not_running and use that classification to extract
        data for those trials. This will allow for the analysis of non-running
        trial. Right now trials classified as not running aren't analyzed.
        '''

        for count, trial in enumerate(self.neo_obj.segments):
            for anlg_signals in trial.analogsignals:
                if anlg_signals.name == 'run speed':
                    run_speed = np.asarray(anlg_signals)
                elif anlg_signals.name == 'run speed time':
                    run_time = np.asarray(anlg_signals)

            base_ind = np.logical_and( run_time > time_before_stimulus, run_time < 0)
            wsk_stim_ind = np.logical_and( run_time > self.t_after_stim, run_time < (self.min_tbefore_stim + self.t_after_stim) )

#            vel = run_speed[stim_period_inds]
            vel = np.concatenate( (run_speed[base_ind], run_speed[wsk_stim_ind]))
            if set_all_to_true == 0:
                if np.mean(vel) >= mean_thresh and np.std(vel) <= sigma_thresh and (sum(vel <= low_thresh)/len(vel)) <= 0.1:
                    self.neo_obj.segments[count].annotations['run_boolean'] = True
            elif set_all_to_true == 1:
                self.neo_obj.segments[count].annotations['run_boolean'] = True


    def get_num_good_trials(self, kind='run_boolean'):
        '''
        Return a list with the number of good trials for each stimulus condition
        And the specified annotations to use.
        kind can be set to either 'wsk_boolean' or 'run_boolean' (default)
        '''
        num_good_trials = list()
        num_slow_trials = list()
        num_all_trials  = list()
        for stim_id in self.stim_ids:
            run_count  = 0
            slow_count = 0
            all_count  = 0
            for trial in self.neo_obj.segments:
                if trial.annotations['trial_type'] == stim_id and trial.annotations[kind] == True:
                    run_count += 1
                elif trial.annotations['trial_type'] == stim_id and trial.annotations[kind] == False:
                    slow_count += 1

                if trial.annotations['trial_type'] == stim_id:
                    all_count += 1
            num_good_trials.append(run_count)
            num_slow_trials.append(slow_count)
            num_all_trials.append(all_count)
        self.num_good_trials = num_good_trials
        self.num_slow_trials = num_slow_trials
        self.num_all_trials  = num_all_trials

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

    def __make_kernel(self, resolution=0.025, kind='square'):
        '''Build alpha kernel with specified 25msec (default) resolution'''
        if kind == 'alpha':
            alpha  = 1.0/resolution
            tau    = np.arange(0,1/alpha*10, 0.001)
            kernel = alpha**2*tau*np.exp(-alpha*tau)

        elif kind == 'square':
            dt = 1.0/resolution
            num_samples_total = int(resolution/0.001*10)
            num_samples_high  = int(resolution/0.001)
            kernel = np.zeros((num_samples_total,))
            kernel[0:num_samples_high] = dt

        return kernel

    def get_protraction_times(self, pos, trial, analysis_window=[0.5, 1.5]):
        '''
        Calculates the protration times for a specified trial
        time_window: start and end time of analysis window of interest (in seconds).
        '''
        phase             = self.wt[pos][:, 3, trial]
        phs_crossing      = phase > 0
        phs_crossing      = phs_crossing.astype(int)
        protraction_inds  = np.where(np.diff(phs_crossing) > 0)[0] + 1
        protraction_times = self.wtt[protraction_inds]

        good_inds = np.logical_and(protraction_times > analysis_window[0],\
                protraction_times < analysis_window[1])
        timestamps = protraction_times[good_inds]

        return timestamps

    def update_t_after_stim(self, t_after_stim):
        self.t_after_stim = t_after_stim
        self.classify_whisking_trials(threshold='user')
        self.rates()

    def get_annotations_index(self, key, value):
        '''Returns trial index for the given key value pair'''
        stim_index = [ index for index, segment in enumerate(self.neo_obj.segments) \
                if segment.annotations[key] == value]
        return stim_index

    def wt_organize(self, kind='run_boolean', running=True, all_trials=False):
        '''
        rates computes the absolute and evoked firing rate and counts for the
        specified stimulus period. The time to start analyzing after the stimulus
        start time is specified by neuro.t_after_stim. The baseline period to
        be analyzed is taken to be the same size as the length of the stimulus
        period.

        MAKE SURE THAT THE EXPERIMENT WAS DESIGNED TO HAVE A LONG ENOUGH BASELINE
        THAT IT IS AT LEAST AS LONG AS THE STIMULUS PERIOD TO BE ANALYZED

        kind can be set to either 'wsk_boolean' or 'run_boolean' (default)

        Recomputes whisker tracking data and adds it to self.wt
        '''

        print('\n-----computing rates----')

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

        if running == True:
            num_trials = self.num_good_trials
        elif running == False:
            print('!!!!! NOT ALL FUNCTIONS WILL USE NON-RUNNING TRIALS !!!!!')
            num_trials = self.num_slow_trials

        if all_trials == True:
            print('!!!!! NOT ALL FUNCTIONS WILL USE NON-RUNNING TRIALS !!!!!')
            print('USING ALL RUNNING TRIALS')
            num_trials = self.num_all_trials

        # preallocation loop
        for k, trials_ran in enumerate(num_trials):
                wt.append(np.zeros((self.wtt.shape[0], 6, trials_ran)))

        for stim_ind, stim_id in enumerate(self.stim_ids):
            good_trial_ind = 0

            for trial in self.neo_obj.segments:
                if trial.annotations['trial_type'] == stim_id and (trial.annotations[kind] == running or \
                        all_trials == True):

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

                    good_trial_ind += 1

        self.wt = wt

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

########## MAIN CODE ##########
########## MAIN CODE ##########

if __name__ == "__main__":
    sns.set_style("whitegrid", {'axes.grid' : False})

    if os.path.isdir('/Users/Greg/Documents/AdesnikLab/Data/'):
        data_dir = '/Users/Greg/Documents/AdesnikLab/Data/'
    elif os.path.isdir('/media/greg/data/behavior/neobehavior/'):
        data_dir = '/media/greg/data/behavior/neobehavior/'

    #manager = NeoHdf5IO(os.path.join(data_dir + 'FID1295_neo_object.h5'))
    print(sys.argv)
    try:
        fid = sys.argv[1]
    except:
        print('no argument provided!')

    manager = NeoHdf5IO(os.path.join(data_dir + sys.argv[1] + '_neo_object.h5'))
    #manager = NeoHdf5IO(os.path.join(data_dir + 'FID1302_neo_object.h5'))
    print('Loading...')
    block = manager.read()
    print('...Loading Complete!')
    manager.close()

    exp1 = block[0]
    neuro = NeuroAnalyzer(exp1, fid)

##### SCRATCH SPACE #####
##### SCRATCH SPACE #####


