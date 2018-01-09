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
import h5py


# change default figure type to PDF
mpl.rcParams['savefig.format'] = 'pdf'
sns.set_style("whitegrid", {'axes.grid' : False})

class BehaviorAnalyzer(object):
    """
    Analyzes data contained in a neo object

    REMEMBER: any changes to the passed in neo object also occur to the original
    object. They point to the same place in memory. No copy is made.
    """

    def __init__(self, f, fid, data_dir):

        print('\n-----__init__-----')
        # specify where the data is
        self.exp_csv_dir = data_dir
        self.data_dir = data_dir

        # initialize trial_class dict (this will have running and whisking
        # boolean arrays)
        self.trial_class = dict()

        self.fid = fid

        # add hdf5 object to class instance
        self.f = f

        # add jb_behavior boolean to object
        self.jb_behavior = self.f.attrs['jb_behavior']

        # find stimulus IDs
        self.stim_ids = np.sort(np.unique([self.f[k].attrs['trial_type'] for k in f])).astype(int)

        # get the control position
        self.control_pos = int(f.attrs['control_pos'])

        # made dictionary of whisker tracking information
        self.wt_type_dict = {0:'angle', 1:'set-point', 2:'amplitude', 3:'phase', 4:'velocity', 5:'whisking'}

        # set time after stimulus to start analyzing
        # time_after_stim + stim_start MUST BE LESS THAN stim_stop time
        self.t_after_stim = 0.500
        print('time after stim is set to: ' + str(self.t_after_stim))

        # find shortest baseline and trial length
        self.__find_min_times()

        # create run_boolean array
        self.__get_run_bool()

        # create array with all the stimulus IDs
        self.__get_all_stim_ids()

        # trim whisker tracking data and align it to shortest trial
        self.__trim_wt()

        # classify a trial as good if whisking occurs during a specified period.
        # a new annotation ('wsk_boolean') is added to self.trial_class
        self.classify_whisking_trials(threshold='median')

        # return a list with the number of good trials for each stimulus condition
        self.get_num_good_trials()

        # calculate rates, psths, whisking array, etc.
        self.wt_organize()

###############################################################################
##### Class initialization functions #####
###############################################################################

    def __get_exp_details_info(self, key):
        ##### LOAD IN EXPERIMENT DETAILS CSV FILE #####
        print('\n----- get_exp_details_info -----')
        fid = self.fid
        print(fid)
        experiment_details_path = self.exp_csv_dir + 'experiment_details_behavior.csv'
        print('Loading experiment details file for FID: ' + str(fid) + '\nkey: ' + key)
        df_exp_det_regular_index = pd.read_csv(experiment_details_path,sep=',')
        df_exp_det_fid_index = df_exp_det_regular_index.set_index('fid')

        # check if experiment is in the experiment details csv file
        # need this info to extract electrode depth

        if fid not in df_exp_det_fid_index.index:
            warnings.warn('fid' + str(fid) + ' not found in experiment details csv file.\n\
                            Please update file and try again.')
            return None

        exp_info = df_exp_det_fid_index.loc[fid]

        if key not in exp_info:
            warnings.warn('key "' + key + '" not found in experiment details csv file.\n\
                            Setting to None.')
            return None
        else:
            return exp_info[key]

    def __find_min_times(self):
        print('\n-----finding minimum trial lengths----')
#        # iterate through all trials and find smallest baseline and smallest
#        # post-trial time
#        for k, seg in enumerate(self.f):
#            # get baseline and stimulus period times for this trial
#            stim_start = self.f[seg].attrs['stim_times'][0] + self.t_after_stim
#            stim_stop  = self.f[seg].attrs['stim_times'][1]
#            base_start = self.f[seg].attrs['stim_times'][0] - (stim_stop - stim_start)
#            base_stop  = self.f[seg].attrs['stim_times'][0]
#            baseline_length = base_stop - base_start # time before stimulus
#
#            # iterate through all units and find longest trial length
#            temp_max = 0
#            for unit, spike_train in enumerate(self.f[seg + '/spiketrains'):
#                t_stop = self.f[seg + '/spiketrains/' + spike_trains].attrs['t_stop']
#                if t_stop > temp_max:
#                    temp_max = np.asarray(t_stop)
#            # temp_max = temp_max - trial_start_time
#            temp_max -= np.asarray(self.f[seg].attrs['stim_times'][0]) # time after stimulus
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
        min_tbefore_stim = self.__get_exp_details_info('latency')
        min_tafter_stim  = self.__get_exp_details_info('duration') - self.__get_exp_details_info('latency')

        self.min_tbefore_stim = np.asarray(min_tbefore_stim)
        self.min_tafter_stim  = np.asarray(min_tafter_stim)
        print('smallest baseline period (time before stimulus): {0}\nsmallest trial length (time after stimulus): {1}'.format(str(min_tbefore_stim), str(min_tafter_stim)))

    def __get_run_bool(self):
        '''make a boolean array with an entry for each segment/trial'''
        run_boolean = list()
        for k, seg in enumerate(self.f):
            run_boolean.append(self.f[seg].attrs['run_boolean'])

        self.trial_class['run_boolean'] = run_boolean


    def __get_all_stim_ids(self):
        '''make a list with every segment/trials stimulus ID'''
        stim_ids_all = list()
        for k, seg in enumerate(self.f):
            stim_ids_all.append(int(self.f[seg].attrs['trial_type']))

        self.stim_ids_all = stim_ids_all

    def __trim_wt(self):
        """
        Trim whisker tracking arrays to the length of the shortest trial.
        Time zero of the wt time corresponds to stimulus onset.
        """
        print('\n-----__trim_wt-----')

        fps        = 500.0
        wt_boolean = False
        for anlg in self.f['/segment-0000/analog-signals/']:
            anlg_path = '/segment-0000/analog-signals/' + anlg
            if self.f[anlg_path].attrs['name'] == 'angle':
                wt_boolean = True

        if wt_boolean:

            print('whisker tracking data found! trimming data to be all the same length in time')

            wt_start_time = float(self.__get_exp_details_info('hsv_start'))
            wt_stop_time  = float(self.__get_exp_details_info('hsv_stop'))
            wt_num_frames = int(self.__get_exp_details_info('hsv_num_frames'))
            num_samples   = wt_num_frames
            wtt = np.linspace(wt_start_time, wt_stop_time, wt_num_frames) - self.min_tbefore_stim
            wt_indices = np.arange(wtt.shape[0]) - int(self.min_tbefore_stim *fps)

            for i, seg in enumerate(self.f):
                for k, anlg in enumerate(self.f[seg + '/analog-signals']):

                    anlg_path = seg + '/analog-signals/' + anlg

                    # find number of samples in the trial
                    if self.f[anlg_path].attrs['name'] == 'angle' or \
                            self.f[anlg_path].attrs['name'] == 'set-point' or\
                            self.f[anlg_path].attrs['name'] == 'amplitude' or\
                            self.f[anlg_path].attrs['name'] == 'phase' or\
                            self.f[anlg_path].attrs['name'] == 'velocity'or\
                            self.f[anlg_path].attrs['name'] == 'whisking':
                        num_samp = len(self.f[anlg_path])

                        # get stimulus onset
                        stim_start = self.f[seg].attrs['stim_times'][0]

                        # slide indices window over
                        # get the frame that corresponds to the stimulus time
                        # and add it to wt_indices.
                        good_inds = wt_indices + int( stim_start*fps )

                        if i == 0 and k == 0:
                            min_trial_length = len(good_inds)
                            # pre-allocate array for all whisker tracking data
                            # this way the original file/data is left untouched
                            wt_data = np.zeros((min_trial_length, 6, len(f)))
                        elif min_trial_length > len(good_inds):
                            warnings.warn('**** MINIMUM TRIAL LENGTH IS NOT THE SAME ****\n\
                                    LINE 208 __trim_wt')


                        anlg_name = self.f[anlg_path].attrs['name']
                        if num_samp > len(good_inds):
                            if  anlg_name == 'angle':
                                wt_data[:, 0, i] = self.f[anlg_path][good_inds]
                            elif anlg_name == 'set-point':
                                wt_data[:, 1, i] = self.f[anlg_path][good_inds]
                            elif anlg_name == 'amplitude':
                                wt_data[:, 2, i] = self.f[anlg_path][good_inds]
                            elif anlg_name == 'phase':
                                wt_data[:, 3, i] = self.f[anlg_path][good_inds]
                            elif anlg_name == 'velocity':
                                wt_data[:, 4, i] = self.f[anlg_path][good_inds]
                            elif anlg_name == 'whisking':
                                wt_data[:, 5, i] = self.f[anlg_path][good_inds]

                        else:
                            warnings.warn('\n**** length of whisker tracking signals is smaller than the length of the good indices ****\n'\
                                    + '**** this data must have already been trimmed ****')
                            if  anlg_name == 'angle':
                                wt_data[:, 0, i] = self.f[anlg_path][:]
                            elif anlg_name == 'set-point':
                                wt_data[:, 1, i] = self.f[anlg_path][:]
                            elif anlg_name == 'amplitude':
                                wt_data[:, 2, i] = self.f[anlg_path][:]
                            elif anlg_name == 'phase':
                                wt_data[:, 3, i] = self.f[anlg_path][:]
                            elif anlg_name == 'velocity':
                                wt_data[:, 4, i] = self.f[anlg_path][:]
                            elif anlg_name == 'whisking':
                                wt_data[:, 5, i] = self.f[anlg_path][:]

            self.wtt          = wtt
            self.wt_boolean   = wt_boolean # indicates whether whisker tracking data is present
            self._wt_min_samp = num_samples
            self.wt_data      = wt_data
        else:
            print('NO WHISKER TRACKING DATA FOUND!\nSetting wt_boolean to False'\
                    '\nuse runspeed to classify trials')
            self.wt_boolean = wt_boolean

    def reclassify_run_trials(self, time_before_stimulus= -1,\
            mean_thresh=250, sigma_thresh=150, low_thresh=200, set_all_to_true=False):
        """
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
        """

        for count, seg in enumerate(self.f):

            for anlg in self.f['/segment-0000/analog-signals/']:
                anlg_path = seg + '/analog-signals/' + anlg

                if self.f[anlg_path].attrs['name'] == 'run_speed':
                    run_speed = self.f[anlg_path][:] # as array?
                elif self.f[anlg_path].attrs['name'] == 'run_speed_time':
                    run_time = self.f[anlg_path][:]

            base_ind = np.logical_and( run_time > time_before_stimulus, run_time < 0)
            wsk_stim_ind = np.logical_and( run_time > self.t_after_stim, run_time < (self.min_tbefore_stim + self.t_after_stim) )

            vel = np.concatenate( (run_speed[base_ind], run_speed[wsk_stim_ind]))
            if set_all_to_true == 0:
                if np.mean(vel) >= mean_thresh and np.std(vel) <= sigma_thresh and (sum(vel <= low_thresh)/len(vel)) <= 0.1:
                    self.trial_class['run_boolean'][count] = True
                    #self.neo_obj.segments[count].annotations['run_boolean'] = True
                else:
                    self.trial_class['run_boolean'][count] = False
                    #self.neo_obj.segments[count].annotations['run_boolean'] = False
            elif set_all_to_true == 1:
                self.trial_class['run_boolean'][count] = True


    def get_num_good_trials(self, kind='run_boolean'):
        """
        Return a list with the number of good trials for each stimulus condition
        And the specified annotations to use.
        kind can be set to either 'wsk_boolean' or 'run_boolean' (default)
        """
        num_good_trials = list()
        num_slow_trials = list()
        num_all_trials  = list()

        for stim_id in self.stim_ids:

            run_count  = 0
            slow_count = 0
            all_count  = 0

            for k, seg in enumerate(self.f):

                if self.stim_ids_all[k] == stim_id and self.trial_class[kind][k] == True:
                    run_count += 1
                elif self.stim_ids_all[k] == stim_id and self.trial_class[kind][k] == False:
                    slow_count += 1

                if self.stim_ids_all[k] == stim_id:
                    all_count += 1

            num_good_trials.append(run_count)
            num_slow_trials.append(slow_count)
            num_all_trials.append(all_count)

        self.num_good_trials = num_good_trials
        self.num_slow_trials = num_slow_trials
        self.num_all_trials  = num_all_trials

    def classify_whisking_trials(self, threshold='user'):
        """
        Classify a trial as good if whisking occurs during a specified period.
        A trial is considered whisking if the mouse was whisking during the
        baseline period and the stimulus period.

        threshold can take the values 'median' or 'user' (default)
        A new annotation ('wsk_boolean') is added to each segment
        """

        if self.wt_boolean:
            print('whisker tracking data found! trimming data to be all the same length in time')
            # make "whisking" distribution and compute threshold
            print('\n-----classify_whisking_trials-----')
            wsk_dist = list()

            for k in range(self.wt_data.shape[2]):
                wsk_dist.extend(self.wt_data[:, 5, k].ravel()) # reshape(-1, 1) changes array to a 2d array (e.g. (1978,) --> (1978, 1)

            #wsk_dist = np.ravel(wsk_dist[:, 1:])
            wsk_dist = np.ravel(np.asarray(wsk_dist))
            wsk_dist = wsk_dist[~np.isnan(wsk_dist)]

            # plot distribution
            print('making "whisking" histogram')
            sns.set(style="ticks")
            f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (0.15, 0.85)})
            sns.boxplot(wsk_dist, ax=ax_box)
            sns.distplot(wsk_dist, ax=ax_hist)
            ax_box.set(yticks=[])
            sns.despine(ax=ax_hist)
            sns.despine(ax=ax_box, left=True)
            plt.xlim(70, 180)
            plt.show()

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
            wsk_boolean = list()
            for k in range(self.wt_data.shape[2]):
#                wsk = anlg.reshape(-1, 1)
                wsk = self.wt_data[:, 5, k]
                base_high  = np.sum(wsk[wsk_base_ind] > thresh)
                stim_high  = np.sum(wsk[wsk_stim_ind] > thresh)
                total_high = base_high + stim_high
                total_samp = np.sum(wsk_base_ind) + np.sum(wsk_stim_ind)
                fraction_high = float(total_high)/float(total_samp)

                if fraction_high > 0.8:
                    wsk_boolean.append(True)
                else:
                    wsk_boolean.append(False)

            self.trial_class['wsk_boolean']   = wsk_boolean
        else:
            print('NO WHISKER TRACKING DATA FOUND!\nuse runspeed to classify trials')

    def __make_kernel(self, resolution=0.025, kind='square'):
        """Build alpha kernel with specified 25msec (default) resolution"""
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

    def update_t_after_stim(self, t_after_stim):
        self.t_after_stim = t_after_stim
        self.classify_whisking_trials(threshold='user')
        self.rates()

    def wt_organize(self, psth_t_start= -0.500, psth_t_stop=2.000, kind='run_boolean', running=True, all_trials=False):
        """
        rates computes the absolute and evoked firing rate and counts for the
        specified stimulus period. The time to start analyzing after the stimulus
        start time is specified by neuro.t_after_stim. The baseline period to
        be analyzed is taken to be the same size as the length of the stimulus
        period.

        MAKE SURE THAT THE EXPERIMENT WAS DESIGNED TO HAVE A LONG ENOUGH BASELINE
        THAT IT IS AT LEAST AS LONG AS THE STIMULUS PERIOD TO BE ANALYZED

        kind can be set to either 'wsk_boolean' or 'run_boolean' (default)

        Recomputes whisker tracking data and adds it to self.wt
        """

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

        licks = list()

        # preallocation loop
        for k, trials_ran in enumerate(num_trials):
                if self.wt_boolean:
                    wt.append(np.zeros((self.wtt.shape[0], 6, trials_ran)))

                if self.jb_behavior:
                    licks.append([list() for x in range(trials_ran)])

        for stim_ind, stim_id in enumerate(self.stim_ids):
            good_trial_ind = 0

            # iterate through all segments in HDF5 file
            for k, seg in enumerate(self.f):

                # if running or whisking trial add data to arrays
                if  self.stim_ids_all[k] == stim_id and (self.trial_class[kind][k] == running or \
                        all_trials == True):

                    # organize whisker tracking data by trial type
                    if self.wt_boolean:
                        for wt_ind in range(self.wt_data.shape[1]):

                            if self.jb_behavior:
                                lick_times = self.f[seg + '/analog-signals/lick-timestamps'][:]
                                licks[stim_ind][good_trial_ind] = lick_times

                            # k should be the segment/trial index
                            wt[stim_ind][:, wt_ind, good_trial_ind] = self.wt_data[:, wt_ind, k]
                        good_trial_ind += 1

        if self.wt_boolean:
            self.wt        = wt

        if self.jb_behavior:
            self.licks = licks

###############################################################################
##### Whisker tracking functions #####
###############################################################################

    def get_protraction_times(self, pos, trial, analysis_window=[0.5, 1.5]):
        """
        Calculates the protration times for a specified trial

        Given a position/trial type index and the trial index this will return
        the timestamp for each whisker at maximum protraction. The protraction
        time will be withing the specified analysis_window

        Parameters
        ----------
        pos: int
            index specifying which position/trial type to analyze
        trial: int
            index specifying which trial to analyze
        analysis_window: array-like
            array or list containing two numbers specifying the beginning and
            end of the analysis period to anlyze

        Returns
        _______
        timestampt: array
            an array containing the timestamps for maximum protractions during
            the analysis window.
        """
        phase             = self.wt[pos][:, 3, trial]
        phs_crossing      = phase > 0
        phs_crossing      = phs_crossing.astype(int)
        protraction_inds  = np.where(np.diff(phs_crossing) > 0)[0] + 1
        protraction_times = self.wtt[protraction_inds]

        good_inds = np.logical_and(protraction_times > analysis_window[0],\
                protraction_times < analysis_window[1])
        timestamps = protraction_times[good_inds]

        return timestamps

    def get_psd(self, input_array, sr):
        """
        compute PSD of input array sampled at sampling rate sr
        input_array: a samples x trial array
        sr: sampling rate of input signal

        Returns x, a 2-d matrix with Pxx_density values
        """

        num_trials = input_array.shape[1]
        for trial in range(num_trials):
            f, Pxx_den = sp.signal.periodogram(input_array[:, trial], sr)
            if trial == 0:
                frq_mat_temp = np.zeros((Pxx_den.shape[0], num_trials))
            frq_mat_temp[:, trial] = Pxx_den

        return f, frq_mat_temp

    def get_spectrogram(self, input_array, sr):
        num_trials = input_array.shape[1]
        for trial in range(num_trials):
            f, t, Sxx = sp.signal.spectrogram(input_array[:, trial], sr, nperseg=256, noverlap=230, nfft=256)
            if trial == 0:
                Sxx_mat_temp = np.zeros((Sxx.shape[0], Sxx.shape[1], num_trials))
            Sxx_mat_temp[:, :, trial] = Sxx

        return f, t, Sxx_mat_temp

    def plot_freq(self, f, frq_mat_temp, axis=None, color='k', error='sem'):
        if axis == None:
            axis = plt.gca()

        mean_frq = np.mean(frq_mat_temp, axis=1)
        se       = sp.stats.sem(frq_mat_temp, axis=1)

        # inverse of the CDF is the percentile function. ppf is the percent point funciton of t.
        if error == 'ci':
            err = se*sp.stats.t.ppf((1+0.95)/2.0, frq_mat_temp.shape[1]-1) # (1+1.95)/2 = 0.975
        elif error == 'sem':
            err = se

        axis.plot(f, mean_frq, color=color)
        axis.fill_between(f, mean_frq - err, mean_frq + err, facecolor=color, alpha=0.3)
        #axis.set_yscale('log')

    def plot_spectrogram(self, f, t, Sxx_mat_temp, axis=None, color='k', error='sem', vmin=None, vmax=None):

        if axis == None:
            axis = plt.gca()

        mean_Sxx = np.mean(Sxx_mat_temp, axis=2)

        if vmin == None:
            axis.pcolormesh(t, f, mean_Sxx)
        else:
            axis.pcolormesh(t, f, mean_Sxx, vmin=vmin, vmax=vmax)
        #axis.set_yscale('log')
        axis.set_ylabel('Frequency (Hz)')
        axis.set_xlabel('Time (s)')

########## MAIN CODE ##########
########## MAIN CODE ##########

if __name__ == "__main__":
    sns.set_style("whitegrid", {'axes.grid' : False})

    if os.path.isdir('/media/greg/data/behavior/hdfbehavior/'):
        data_dir = '/media/greg/data/behavior/hdfbehavior/'
    elif os.path.isdir('/jenny/add/your/path/here/'):
        data_dir = '/jenny/add/your/path/here/'

    # each experiment needs an entry in the CSV file!
    mouse      = 'GT0007'
    experiment = 'FID1014'

    hdf_name   = mouse + '_' + experiment
    fid        = hdf_name
    hdf_fname  = data_dir + hdf_name + '.hdf5'

    f = h5py.File(hdf_fname,'r+')

    whisk = BehaviorAnalyzer(f, fid, data_dir)

    # reclassify trials as run vs non-run
    whisk.reclassify_run_trials(mean_thresh=100, low_thresh=50)
    whisk.wt_organize()

    # remove entries for trial type "0"
    stim_ids = np.unique(whisk.stim_ids_all)
    if 0 in stim_ids:
        whisk.stim_ids = whisk.stim_ids[:-1]
        whisk.wt = whisk.wt[1::]


##### SCRATCH SPACE #####
##### SCRATCH SPACE #####



##### plot whisking variable vs time for all trials #####
##### plot whisking variable vs time for all trials #####

fig, ax = plt.subplots(2, 1)
dtype = 2 # 0, angle; 1, set-point; 2, amplitude; 3, phase; 4, velocity; 5, "whisk".
pos = 1
ax[0].plot(whisk.wtt, whisk.wt[pos-1][:, dtype, :], linewidth=0.5)
ax[0].plot(whisk.wtt, np.mean(whisk.wt[pos-1][:, dtype, :], axis=1), 'k')
ax[0].set_ylim(90, 160)

ax[1].plot(whisk.wtt, whisk.wt[9-pos-1][:, dtype, :], linewidth=0.5)
ax[1].plot(whisk.wtt, np.mean(whisk.wt[9-pos-1][:, dtype, :], axis=1), 'k')
ax[1].set_ylim(90, 160)
fig.show()

##### make power spectral density plots of whisking #####
##### make power spectral density plots of whisking #####
pos=4
fig, ax = plt.subplots(1,1)
f, frq_mat_temp = whisk.get_psd(whisk.wt[pos-1][:, 0, :], 500)
whisk.plot_freq(f, frq_mat_temp, axis=ax, color='black')

f, frq_mat_temp = whisk.get_psd(whisk.wt[9-pos-1][:, 0, :], 500)
whisk.plot_freq(f, frq_mat_temp, axis=ax, color='red')
ax.set_xlim(0,30)
ax.set_ylim(1e-1, 1e3)
ax.set_yscale('log')
fig.show()

##### make spectrogram plots of whisking #####
##### make spectrogram plots of whisking #####
pos = 2
fig, ax = plt.subplots(2,1)
f, t, Sxx_mat_temp = whisk.get_spectrogram(whisk.wt[pos-1][:, 0, :], 500)
whisk.plot_spectrogram(f, t, Sxx_mat_temp, axis=ax[0], vmin=0, vmax=3)
ax[0].set_ylim(0, 30)

f, t, Sxx_mat_temp = whisk.get_spectrogram(whisk.wt[9-pos-1][:, 0, :], 500)
whisk.plot_spectrogram(f, t, Sxx_mat_temp, axis=ax[1], vmin=0, vmax=3)
ax[1].set_ylim(0, 30)
fig.show()



