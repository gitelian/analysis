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
from scipy.signal import butter, lfilter
#import 3rd party code found on github
#import icsd
import ranksurprise
import dunn
# for LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# for t-distributed stochastic neighbor embedding
from sklearn.manifold import TSNE
# for MDS
from sklearn import manifold
# for python circular statistics
import pycircstat as pycirc

from IPython.core.debugger import Tracer

# change default figure type to PDF
mpl.rcParams['savefig.format'] = 'pdf'
plt.rc('font',family='Arial')
sns.set_style("whitegrid", {'axes.grid' : False})

class NeuroAnalyzer(object):
    """
    Analyzes data contained in a custom hdf5 object

    """

    def __init__(self, f, fid):

        print('\n-----__init__-----')
        # specify where the data is
        if os.path.isdir('/Users/Greg/Documents/AdesnikLab/Data/'):
            data_dir = '/Users/Greg/Documents/AdesnikLab/Data/'
            self.exp_csv_dir = data_dir
        elif os.path.isdir('/media/greg/data/neuro/hdf5/'):
            data_dir = '/media/greg/data/neuro/hdf5/'
            self.exp_csv_dir = '/media/greg/data/neuro/'
        self.data_dir = data_dir

        # initialize trial_class dict (this will haveing running and whisking
        # and jb_behavior boolean arrays
        self.trial_class = dict()

        self.fid = fid

        # add hdf5 object to class instance
        self.f = f

        # get the control position
        self.control_pos = int(f.attrs['control_pos'])

        # is this a jb_behavior experiment
        self.jb_behavior = self.f.attrs['jb_behavior']

        # add time before and time after stimulus start/stop respectively
        # add stimulus period duration (avoid dependence on CSV file
        self.time_before = self.f.attrs['time_before']
        self.time_after  = self.f.attrs['time_after'] # not really used
        self.stim_duration = self.f.attrs['stim_duration']

        # add stimulus indices (i.e. which dio channels correspond to
        # what...will be different for experiments prior to FID1729)
        self.stim_ind = f.attrs['stim_ind']

        # set time after stimulus to start analyzing
        # time_after_stim + stim_start MUST BE LESS THAN stim_stop time
        self.t_after_stim = f.attrs['t_after_stim']

        # is dynamic time set
        self.dynamic_time = f.attrs['dynamic_time']

        # are there spikes
        self.spikes_bool = f.attrs['spikes_bool']

        # are there LFPs
        self.lfp_bool = f.attrs['lfp_bool']

        # is there whisker tracking
        #self.wt_bool = f.attrs['wt_bool']
        self.wt_bool = False

        # find stimulus IDs
        self.stim_ids = np.sort(np.unique([self.f[k].attrs['trial_type'] for k in f])).astype(int)

        # made dictionary of whisker tracking information
        self.wt_type_dict = {0:'angle', 1:'set-point', 2:'amplitude', 3:'phase', 4:'velocity', 5:'whisking'}

        # find shortest baseline and trial length
        self.__find_min_times()

        # create run_boolean array
        self.__get_run_bool()

        # create array with all the stimulus IDs
        self.__get_all_stim_ids()

        # trim run data and align it to shortest trial
        self.__trim_run()

        # classify behavior using licks and go/no-go angle position
        if self.jb_behavior and not self.spikes_bool:
            self.__classify_behavior()
            self.rates(psth_t_start= -1.500, psth_t_stop=2.500, kind='jb_engaged', engaged=True, all_trials=False)

        # trim whisker tracking data and align it to shortest trial
        # defaults to False unless it finds whisker tracking data
        self.__trim_wt()

        # trim LFP data and align it to shortest trial
        if self.lfp_bool:
            self.__trim_lfp()

        # classify a trial as good if whisking occurs during a specified period.
        # a new annotation ('wsk_boolean') is added to self.trial_class
#        self.classify_whisking_trials(threshold='median')

        # return a list with the number of good trials for each stimulus condition
        self.get_num_good_trials()

        if self.spikes_bool:
            # get depths for all units
            self.__get_depths()

            # find number of units
            self.num_units = len(self.f['/segment-0000/spiketrains'])

            # find shank names (i.e. names of electrodes/recording sites, e.g.
            # 'e1', 'e2')
            self.shank_names = np.sort(np.unique([self.f['/segment-0000/spiketrains/' + k].attrs['shank'] \
                    for k in self.f['segment-0000/spiketrains']]))

            # Find depths of each shank and add it to self.shank_depths
            self.shank_depths = self.__get_shank_depths()

            # find shank IDs for each unit (e.g. [0, 0, 1, 1, 1])
            self.shank_ids = self.__get_shank_ids()

            # creat lists or array with units duration, ratio, and mean waveform
            self.__get_waveinfo()

            # make dictionary of cell types and get cell type IDs
            self.cell_type_dict = {0:'MU', 1:'RS', 2:'FS', 3:'UC'}
            self.cell_type      = self.__get_celltypeID()
            self.cell_type_og   = self.cell_type

            if not self.jb_behavior:
                self.rates(psth_t_start= -0.500, psth_t_stop=2.000, kind='run_boolean', engaged=True, all_trials=False)
            if self.jb_behavior:
                self.__classify_behavior()
                self.rates(psth_t_start= -1.500, psth_t_stop=2.500, kind='jb_engaged', engaged=True, all_trials=False)

            # reclassify units using their mean OMI (assuming ChR2 is in PV
            # cells). This is dependent on everything above!
    #        self.reclassify_units()

            # create region dictionary
            self.region_dict = {0:'M1', 1:'S1'}

            # get selectivity for all units
            self.get_selectivity()

            # get preferred position for all units
            self.get_preferred_position()

            # get best contact for each unit
            self.get_best_contact()

            # kruskal wallis and dunn's test to ID sensory driven units
            self.get_sensory_drive()

        # reclassify running trials
#        #self.reclassify_run_trials(self, time_before_stimulus= -1,\
#        #            mean_thresh=250, sigma_thresh=150, low_thresh=200, set_all_to_true=False)

###############################################################################
##### Class initialization functions #####
###############################################################################

    def __get_exp_details_info(self, key):
        ##### LOAD IN EXPERIMENT DETAILS CSV FILE #####
        print('\n----- get_exp_details_info -----')
        fid = int(self.fid[3::])
        experiment_details_path = self.exp_csv_dir + 'experiment_details.csv'
        print('Loading experiment details file for FID: ' + str(self.fid) + '\nkey: ' + key)
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

    def __get_depths(self):
        """Get depths for all units"""
        seg_iter = f.iterkeys()
        for i, seg in enumerate(f):
            fid, shank, depth = list(), list(), list()

            for spike in self.f[seg]['spiketrains']:
                ff = self.f[seg]['spiketrains'][spike]

                if np.isnan(ff.attrs['depth']):
                    depth.append(np.asarray(-1))
                else:
                    #depth.append(ff.attrs['depth'][0])
                    depth.append(ff.attrs['depth'])

        self.depths = np.ravel(np.asarray(depth))

    def __get_shank_ids(self):
        """
        Return shank IDs for each unit
        """
        shank_ids = np.zeros((self.num_units, ))
        for k, shank_name in enumerate(self.shank_names):
            for j, unit in enumerate(self.f['/segment-0000/spiketrains/']):
                spike_path = '/segment-0000/spiketrains/' + unit
                if self.f[spike_path].attrs['shank'] == shank_name:
                    shank_ids[j] = k
        shank_ids = shank_ids.astype(int)
        return shank_ids

    def __get_shank_depths(self):
        """Find depths of each shank and add it to self.shank_depths"""
        depth = list()
        for shank in self.shank_names:
            depth_temp0 = 0
            for spikes in self.f['/segment-0000/spiketrains/']:
                spike_path = '/segment-0000/spiketrains/' + spikes
                if self.f[spike_path].attrs['shank'] == shank:
                    depth_temp1 = self.f[spike_path].attrs['depth']
                    if depth_temp0 < depth_temp1:
                        depth_temp0 = depth_temp1
            depth.append(depth_temp0)
        return np.asarray(depth)

    def __get_waveinfo(self):
        """gets waveform duration, ratio, and mean waveform for each unit"""

        duration = list()
        ratio    = list()
        waves    = list()

        for spikes in self.f['/segment-0000/spiketrains']:

            spike_path = '/segment-0000/spiketrains/' + spikes

            if 'duration' in self.f[spike_path].attrs.keys():
                duration.append(self.f[spike_path].attrs['duration'])
                ratio.append(self.f[spike_path].attrs['ratio'])
                waves.append(self.f[spike_path].attrs['waveform'])

        #if hasattr(spikes, 'annotations.duration') is True:
        if 'duration' in self.f[spike_path].attrs.keys():
            self.duration = duration
            self.ratio    = ratio
            self.waves    = np.asarray(waves).squeeze()
        else:
            self.duration = None
            self.ratio    = None
            self.waves    = None

    def __get_celltypeID(self):
        """
        Put celltype IDs in an array that corresponds to the unit order
        """
        cell_type = list()
        for spikes in self.f['/segment-0000/spiketrains']:
            spike_path = '/segment-0000/spiketrains/' + spikes
            try:
                cell_type.append(self.cell_type_dict[self.f[spike_path].attrs['cell_type'][0][0]])
            except:
                #cell_type.append(self.cell_type_dict[self.f[spike_path].attrs['cell_type'][0]])
                cell_type.append(self.cell_type_dict[self.f[spike_path].attrs['cell_type']])

        return np.asarray(cell_type)

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

        ## this worked last before switching to time_before/time_after
#        min_tbefore_stim = self.__get_exp_details_info('latency')
#        min_tafter_stim  = self.__get_exp_details_info('duration') - self.__get_exp_details_info('latency')

        self.min_tbefore_stim = self.time_before
        #self.min_tafter_stim  = self.stim_duration + self.time_after
        self.min_tafter_stim  = self.stim_duration
        print('smallest baseline period (time before stimulus): {0}\nsmallest \
                trial length (time after stimulus): {1}'.format(str(self.min_tbefore_stim), \
                str(self.min_tafter_stim)))

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

    def __classify_behavior(self):
        print('\n-----classify_behavior----')
        print('\n-----THIS WILL BE WRONG WITH OPTOGENETICS:----')
        behavior_ids = list()
        licks_all    = list()
        for k, seg in enumerate(self.f):
            trial_type = int(self.f[seg].attrs['trial_type'])
            stim_start = self.f[seg].attrs['stim_times'][0]
            stim_stop  = self.f[seg].attrs['stim_times'][1]

            try:
                lick_times = self.f[seg + '/analog-signals/lick-timestamps'][:]
                lick_times_relative = lick_times - stim_start

                licks_after  = lick_times_relative > 0
                licks_before = lick_times_relative < 1
            except:
                lick_times = None

            if lick_times is None:
                lick = False
                licks_all.append(0)
            elif np.sum(np.logical_and(licks_after, licks_before)) > 0:
                # licks in the response window
                lick = True
                licks_all.append(1)
            else:
                lick = False
                licks_all.append(0)

            #TODO double check this! it currently deals with optogenetics...figure that out.
            if trial_type < 5:
                go = True
#            # this isn't used in the experiment
#            elif trial_type == 5:
#                go = None
            elif trial_type >= 5 and trial_type < 9:
                go = False
            elif trial_type == 9:
                # control position
                go = None

            # label the type of behavior using logic
            if go == None:
                behavior_ids.append(5)
            elif go and lick:
                # hit
                behavior_ids.append(1)
            elif go and not lick:
                # miss
                behavior_ids.append(3)
            elif not go and lick:
                # false alarm
                behavior_ids.append(2)
            elif not go and not lick:
                # correct reject
                behavior_ids.append(4)

        # find when transition from licking to not licking happens (offset by one since
        # diff shortens array by 1)
        #low = np.where(np.diff(licks_all) == -1)[0][0:-1] + 1
        low = np.where(np.diff(licks_all) == -1)[0] + 1

        # find when transition from not licking to licking happens, skip the first lick
        high = np.where(np.diff(licks_all) == 1)[0]

        if low[0] < high[0]:
            low = low[1::]

        # get duration between licking trials
        down_time = high - low

        # find when mouse stopped licking
        stop_ind = np.where(down_time > 30)[0]

        # get trial number when mouse stopped licking
        if stop_ind.shape[0] != 0:
            stop_lick_trial = low[stop_ind[0]]
        else:
            stop_lick_trial = None

        jb_engaged = list()
        for k, seg in enumerate(self.f):
            if stop_lick_trial is not None:
                if k < stop_lick_trial:
                    jb_engaged.append(True)
                else:
                    jb_engaged.append(False)
            else:
                jb_engaged.append(True)

        self.trial_class['jb_engaged'] = jb_engaged
        self.last_engaged_trial = stop_lick_trial


        self.behavior_ids = behavior_ids
        self.licks_all    = licks_all
        #self.behavior_labels = {'hit':1, 'miss':3, 'false_alarm':2, 'correct_reject':4}
        self.behavior_labels = {1:'hit', 3:'miss', 2:'false_alarm', 4:'correct_reject'}

    def __trim_wt(self):
        """
        Trim whisker tracking arrays to the length of the shortest trial.
        Time zero of the wt time corresponds to stimulus onset.
        """
        print('\n-----__trim_wt-----')

        fps        = 500.0
        if self.wt_bool:

            print('whisker tracking data found! trimming data to be all the same length in time')

            #TODO: load in the time when HSV camera starts and stops!!!
            if int(fid[3::]) < 1729:
                wt_start_time = float(self.__get_exp_details_info('hsv_start'))
                wt_stop_time  = float(self.__get_exp_details_info('hsv_stop'))
                wt_num_frames = int(self.__get_exp_details_info('hsv_num_frames'))

                num_samples = wt_num_frames
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
                            wt_data = np.zeros((min_trial_length, 7, len(f)))
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
                            elif anlg_name == 'curvature':
                                wt_data[:, 6, i] = self.f[anlg_path][good_inds]

                        else:
                            if i == 0 and k == 0:
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
                            elif anlg_name == 'curvature':
                                wt_data[:, 6, i] = self.f[anlg_path][:]

# TODO collect some example data and finish this!!!

#            elif int(fid[3::]) >= 1729:
#                for i, seg in enumerate(self.f):
#                    for k, anlg in enumerate(self.f[seg + '/analog-signals']):
#                        anlg_path = seg + '/analog-signals/' + anlg
#
#                        if self.f[anlg_path].attrs['name'] == 'hsv_times':
#                            hsv_times_temp = self.f[anlg_path]
#                            num_samp = len(hsv_times)
#
#                            # find number of samples before stim start
#                            hsv_times_temp - self.f[seg].attrs['stim_times'][0]
#
#                            # find number of samples after stim stop

            self.wtt          = wtt
            self._wt_min_samp = num_samples
            self.wt_data      = wt_data
        else:
            print('NO WHISKER TRACKING DATA FOUND!\nSetting wt_bool to False'\
                    '\nuse runspeed to classify trials')

    def __trim_run(self):
        """
        Trim LFP arrays to the length of the shortest trial.
        Time zero of the LFP time corresponds to stimulus onset.
        """
        print('\n-----__trim_run-----')

        run_path = '/segment-0000/analog-signals/run_speed/'
        # get the sampling rate
        sr = float(self.f[run_path].attrs['sampling_rate'])


        print('trimming run data to be all the same length in time')
        # make time vector for run data
        num_samples = int( (self.min_tafter_stim + self.min_tbefore_stim)*sr ) # total time (s) * samples/sec
        run_indices = np.arange(num_samples) - int( self.min_tbefore_stim * sr )
        run_t       = run_indices / sr

        for i, seg in enumerate(self.f):
            run_path = seg + '/analog-signals/run_speed/'

            # find number of samples in the trial
            num_samp = len(self.f[run_path])

            # get stimulus onset
            stim_start = self.f[seg].attrs['stim_times'][0]

            # slide indices window over
            # get the frame that corresponds to the stimulus time
            # and add it to run_indices.
            good_inds = run_indices + int( stim_start*sr )

            if i == 0:
                min_trial_length = len(good_inds)
                run_data = np.zeros((min_trial_length, len(f)))
            elif min_trial_length > len(good_inds):
                warnings.warn('**** MINIMUM TRIAL LENGTH IS NOT THE SAME ****\n\
                        LINE 356 __trim_run')

            if num_samp > len(good_inds):
                # hdf5 didn't like the good_inds slicing. so I had to use a
                # temp numpy array that supports this simple slicing
                data_temp = self.f[run_path][:]
                run_data[:, i] = data_temp[good_inds]
            else:
                warnings.warn('\n**** length of run data is smaller than the length of the good indices ****\n'\
                        + '**** this data must have already been trimmed ****')
                run_data[:, i] = self.f[run_path][:]


        self.run_t          = run_t
        self._run_min_samp  = num_samples
        self.run_data       = run_data * (12*np.pi/360.0)

    def __trim_lfp(self):
        """
        Trim LFP arrays to the length of the shortest trial.
        Time zero of the LFP time corresponds to stimulus onset.
        """
        print('\n-----__trim_lfp-----')

        num_shanks = 0
        chan_per_shank = list()
        lfp_boolean = False
        for item in self.f['segment-0000'].items():
            if 'lfps' in item[0]:
                lfp_path = '/segment-0000/' + item[0]
                # get the sampling rate
                lfp_boolean = True
                num_shanks += 1
                chan_per_shank.append(self.f[lfp_path].shape[1])
                sr = float(self.f[lfp_path].attrs['sampling_rate'])

        if lfp_boolean:

            print('LFP data found! trimming data to be all the same length in time')
            # make time vector for LFP data
            num_samples = int( (self.min_tafter_stim + self.min_tbefore_stim)*sr ) # total time (s) * samples/sec
            lfp_indices = np.arange(num_samples) - int( self.min_tbefore_stim * sr )
            lfp_t       = lfp_indices / sr

            for i, seg in enumerate(self.f):
                shank_ind = 0
                for k, item in enumerate(self.f[seg].items()):
                    if 'lfps' in item[0]:
                        lfp_path = seg + '/' + item[0]

                        # find number of samples in the trial
                        num_samp = len(self.f[lfp_path])

                        # get stimulus onset
                        stim_start = self.f[seg].attrs['stim_times'][0]

                        # slide indices window over
                        # get the frame that corresponds to the stimulus time
                        # and add it to lfp_indices.
                        good_inds = lfp_indices + int( stim_start*sr )

                        if i == 0 and shank_ind == 0:
                            min_trial_length = len(good_inds)
                            lfp_data = [np.zeros((min_trial_length, x, len(f)), 'int16') for x in chan_per_shank]
                        elif min_trial_length > len(good_inds):
                            warnings.warn('**** MINIMUM TRIAL LENGTH IS NOT THE SAME ****\n\
                                    LINE 208 __trim_lfp')

                        if num_samp > len(good_inds):
                            lfp_data[shank_ind][:, :, i] = self.f[lfp_path][good_inds, :]
                        else:
                            warnings.warn('\n**** length of LFPs is smaller than the length of the good indices ****\n'\
                                    + '**** this data must have already been trimmed ****')
                            lfp_data[shank_ind][:, :, i] = self.f[lfp_path][:]

                        shank_ind += 1

            self.lfp_t          = lfp_t
            self.lfp_boolean    = lfp_boolean
            self._lfp_min_samp  = num_samples
            self.chan_per_shank = chan_per_shank
            self.lfp_data       = lfp_data
        else:
            print('NO LFP DATA FOUND!\nSetting lfp_boolean to False')
            self.lfp_boolean = lfp_boolean

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
            #wsk_stim_ind = np.logical_and( run_time > self.t_after_stim, run_time < (self.min_tbefore_stim + self.t_after_stim) )
            wsk_stim_ind = np.logical_and( run_time > self.t_after_stim, run_time < self.stim_duration)

            vel = np.concatenate( (run_speed[base_ind], run_speed[wsk_stim_ind]))
            if set_all_to_true == 0:
                if np.mean(vel) >= mean_thresh and np.std(vel) <= sigma_thresh and (sum(vel <= low_thresh)/len(vel)) <= 0.1:
                    self.trial_class['run_boolean'][count] = True
                else:
                    self.trial_class['run_boolean'][count] = False
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

        if self.wt_bool:
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
            #wsk_stim_ind = np.logical_and( wtt > self.t_after_stim, wtt < (self.min_tbefore_stim + self.t_after_stim) )
            wsk_stim_ind = np.logical_and( wtt > self.t_after_stim, wtt < self.stim_duration)
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

            self.trial_class['wsk_boolean'] = wsk_boolean
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

    def rates(self, psth_t_start= -0.500, psth_t_stop=2.000, kind='run_boolean', engaged=True, all_trials=False):
        """
        rates computes the absolute and evoked firing rate and counts for the
        specified stimulus period. The time to start analyzing after the stimulus
        start time is specified by neuro.t_after_stim. The baseline period to
        be analyzed is taken to be the same size as the length of the stimulus
        period.

        MAKE SURE THAT THE EXPERIMENT WAS DESIGNED TO HAVE A LONG ENOUGH BASELINE
        THAT IT IS AT LEAST AS LONG AS THE STIMULUS PERIOD TO BE ANALYZED

        kind can be set to either 'wsk_boolean' or 'run_boolean' (default) or 'jb_engaged'

        Recomputes whisker tracking data and adds it to self.wt
        """

        print('\n-----computing rates----')
        absolute_rate   = list()
        evoked_rate     = list()
        absolute_counts = list()
        evoked_counts   = list()
        binned_spikes   = list()
        psth            = list()
        run             = list()
        licks           = list()
        bids            = list() # behavior IDs
        lick_bool       = list()

        # make whisker tracking list wt
        if self.wt_bool:
            wt          = list()

        if kind == 'wsk_boolean' and not self.wt_bool:
            warnings.warn('**** NO WHISKER TRACKING DATA AVAILABLE ****\n\
                    using run speed to select good trials')
            kind = 'run_boolean'

        elif kind == 'wsk_boolean' and self.wt_bool:
            print('using whisking to find good trials')
            self.get_num_good_trials(kind='wsk_boolean')

        elif self.jb_behavior and kind == 'jb_engaged':
            self.get_num_good_trials(kind='jb_engaged')

        elif kind == 'run_boolean':
            print('using running to find good trials')
            self.get_num_good_trials(kind='run_boolean')

        if engaged == True:
            num_trials = self.num_good_trials

        elif engaged == False:
            print('!!!!! NOT ALL FUNCTIONS WILL USE NON-RUNNING TRIALS !!!!!')
            num_trials = self.num_slow_trials

        if all_trials == True:
            reclassify_run_trials(self, set_all_to_true=True)
            num_trials = self.num_all_trials

        # make bins for rasters and PSTHs
#        bins = np.arange(-self.min_tbefore_stim, self.min_tafter_stim, 0.001)
        bins = np.arange(psth_t_start, psth_t_stop, 0.001)
        kernel = self.__make_kernel(kind='square', resolution=0.100)
#        kernel = self.__make_kernel(kind='square', resolution=0.025)
#        kernel = self.__make_kernel(kind='alpha', resolution=0.025)
        self._bins = bins
        self.bins_t = bins[0:-1]


        # preallocation loop
        for k, trials_ran in enumerate(num_trials):
                run.append(np.zeros((self.run_t.shape[0], trials_ran)))

                if self.spikes_bool:
                    absolute_rate.append(np.zeros((trials_ran, self.num_units)))
                    evoked_rate.append(np.zeros((trials_ran, self.num_units)))
                    absolute_counts.append(np.zeros((trials_ran, self.num_units)))
                    evoked_counts.append(np.zeros((trials_ran, self.num_units)))
                    binned_spikes.append(np.zeros((bins.shape[0]-1, trials_ran, self.num_units)))
                    psth.append(np.zeros((bins.shape[0]-1,trials_ran, self.num_units))) # samples x trials x units

                if self.wt_bool:
                    wt.append(np.zeros((self.wtt.shape[0], 7, trials_ran)))

                if self.jb_behavior:
                    licks.append([list() for x in range(trials_ran)])
                    lick_bool.append(np.zeros((trials_ran,)))
                    bids.append([list() for x in range(trials_ran)])


        for stim_ind, stim_id in enumerate(self.stim_ids):
            good_trial_ind = 0

            # iterate through all segments in HDF5 file
            for k, seg in enumerate(self.f):

                # if running or whisking or jb_engaged trial add data to arrays
                # TODO stop adding trial data if the mouse has stopped
                # licking/performing in the task
                if  self.stim_ids_all[k] == stim_id and (self.trial_class[kind][k] == engaged or \
                        all_trials == True):

                    # add run data to list
                    run[stim_ind][:, good_trial_ind] = self.run_data[:, k]

                    # organize whisker tracking data by trial type
                    if self.wt_bool:
                        for wt_ind in range(self.wt_data.shape[1]):
                            # k should be the segment/trial index
                            wt[stim_ind][:, wt_ind, good_trial_ind] = self.wt_data[:, wt_ind, k]

                    if self.jb_behavior:
                        # get lick times
                        stim_start = self.f[seg].attrs['stim_times'][0] + self.t_after_stim
                        try:
                            lick_times = self.f[seg + '/analog-signals/lick-timestamps'][:]
                            lick_times= lick_times - stim_start
                        except:
                            lick_times = np.nan

                        licks[stim_ind][good_trial_ind] = lick_times

                        lick_bool[stim_ind][good_trial_ind] = self.licks_all[k]
                        bids[stim_ind][good_trial_ind] = self.behavior_ids[k]

                    if self.spikes_bool:
                        # get baseline and stimulus period times for this trial
                        # this uses ABSOLUTE TIME not relative time
                        obj_stop = self.f[seg].attrs['stim_times'][0]
                        stim_start = obj_stop + self.t_after_stim
                        stim_stop= obj_stop + self.stim_duration
                        base_start = obj_stop - np.abs((stim_stop - stim_start))
                        base_stop  = obj_stop

                        # TODO add ability to change the analysis window
                        #stim_start = obj_stop - 1.5
                        #stim_stop  = obj_stop
                        #base_start = obj_stop - self.time_before
                        #base_stop  = base_start + np.abs(stim_stop - stim_start)

                        # iterate through all units and count calculate various
                        # spike rates (e.g. absolute firing and evoked firing rates
                        # and counts)
                        for unit, spike_train in enumerate(self.f[seg + '/spiketrains']):
                            spk_times = self.f[seg + '/spiketrains/' + spike_train][:]

                            # bin spikes for rasters (time 0 is stimulus start)
                            spk_times_relative = spk_times - self.f[seg].attrs['stim_times'][0]
                            counts = np.histogram(spk_times_relative, bins=bins)[0]
                            binned_spikes[stim_ind][:, good_trial_ind, unit] = counts

                            # convolve binned spikes to make PSTH
                            psth[stim_ind][:, good_trial_ind, unit] =\
                                    np.convolve(counts, kernel)[:-kernel.shape[0]+1]

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

        self.run           = run

        if self.spikes_bool:
            self.abs_rate      = absolute_rate
            self.abs_count     = absolute_counts
            self.evk_rate      = evoked_rate
            self.evk_count     = evoked_counts
            self.binned_spikes = binned_spikes
            self.psth          = psth
            self.psth_t        = bins

        if self.wt_bool:
            self.wt        = wt

        if self.jb_behavior:
            self.licks = licks
            self.lick_bool = lick_bool
            self.binds = [np.asarray(x) for x in bids]

    def rebin_spikes(self, bin_size=0.005, analysis_window=[0.5, 1.5]):
        '''bin spike times with specified bin size and analysis window'''

        rebinned_spikes = list()
        num_trials = self.num_good_trials
        bins = np.arange(analysis_window[0], analysis_window[1], bin_size)
        t = bins[0:-1]

        # pre-allocate arrays
        for k, trials_ran in enumerate(num_trials):
            rebinned_spikes.append(np.zeros((bins.shape[0]-1, trials_ran, self.num_units)))

        # iterate through all trials and count spikes
        for cond in range(self.stim_ids.shape[0]):
            for trial_ind in range(num_trials[cond]):
                for unit_ind in range(self.num_units):
                    all_spike_times = self.bins_t[self.binned_spikes[cond][:, trial_ind, unit_ind].astype(bool)]
                    windowed_spike_indices = np.logical_and(all_spike_times > analysis_window[0],\
                            all_spike_times < analysis_window[1])
                    windowed_spike_times = all_spike_times[windowed_spike_indices] # spike times to align LFPs to
                    counts = np.histogram(windowed_spike_times, bins=bins)[0]
                    rebinned_spikes[cond][:, trial_ind, unit_ind] = counts

        return rebinned_spikes, t

    def spike_time_corr(self, rebinned_spikes, cond=0):
        """
        compute spike time correlation for all units in a specified condition

        rebinned_spikes: the output of rebin_spikes, should be a list of length
            number of conditions. And each entry should be an array of size
            samples x trials x units
        """

        # get indices to sort array by region and then by depth
        # lexsort sorts by last entry to first
        sort_inds= np.lexsort((np.squeeze(np.asarray(self.depths)), self.shank_ids))

        # reshape array so it is units x time-samples
        temp_array = rebinned_spikes[cond][:, :, sort_inds].T
        mp = temp_array.reshape(temp_array.shape[0], temp_array.shape[1]*temp_array.shape[2])

        # compute correlation matrix
        R = np.corrcoef(mp)

        # fill diagonal with zeros
        np.fill_diagonal(R, 0)

        return R, sort_inds

    def noise_corr(self, cond=0):
        """
        compute noise correlation for all units for a specified condition
        """
        # get indices to sort array by region and then by depth
        # lexsort sorts by last entry to first
        sort_inds= np.lexsort((np.squeeze(np.asarray(self.depths)), self.shank_ids))

        # get absolute spike counts for the specified condition
        sc = self.abs_count[cond].T

        # reorder array so it is by region and then by depth
        sc = sc[sort_inds, :]

        # compute correlation coefficient (each row is a variable, each column
        # is an observation)
        R = np.corrcoef(sc)

        # fill diagonal with zeros
        np.fill_diagonal(R, 0)

        return R, sort_inds

    def spike_vs_runspeed(self, cond=0, binsize=0.010, analysis_window=[0, 1.5]):
        # use all trials
        self.rates(all_trials=True)
        bins = np.arange(analysis_window[0], analysis_window[1], binsize)
        t = bins[0:-1]
        num_trials = self.num_all_trials[cond]

        # preallocate arrays
        rebinned_spikes = np.zeros((bins.shape[0]-1, self.num_units, num_trials))
        binned_runspeed = np.zeros((bins.shape[0]-1, num_trials))

        # iterate through all trials and all units
        for trial_ind in range(num_trials):

            # count spikes and add counts to rebinned_spikes array
            for unit_ind in range(self.num_units):
                all_spike_times = self.bins_t[self.binned_spikes[cond][:, trial_ind, unit_ind].astype(bool)]
                windowed_spike_indices = np.logical_and(all_spike_times > analysis_window[0],\
                        all_spike_times < analysis_window[1])
                windowed_spike_times = all_spike_times[windowed_spike_indices] # spike times to align run speed to
                counts = np.histogram(windowed_spike_times, bins=bins)[0]
                rebinned_spikes[:, unit_ind, trial_ind] = counts

            # bin runspeed
            run_speed = self.run[cond][:, trial_ind]
            for k in range(len(bins) -1):
                start_ind  = np.argmin(np.abs(self.run_t - bins[k]))
                stop_ind   = np.argmin(np.abs(self.run_t - bins[k+1]))
                mean_speed = np.mean(run_speed[start_ind:stop_ind])
                binned_runspeed[k, trial_ind] = mean_speed

        return rebinned_spikes, binned_runspeed, t

    def get_mean_tc(self, kind='abs_rate'):
        """
        compute mean tuning curve with standard error in 3rd dimension

        returns a matrix of size (num_units x num_maipulations (e.g. 3) x 2
        (i.e. mean and sem)

        """

        kind_dict      = {'abs_rate': 0, 'abs_count': 1, 'evk_rate': 2, 'evk_count': 3}
        kind_of_tuning = [self.abs_rate, self.abs_count, self.evk_rate, self.evk_count]
        rates          = kind_of_tuning[kind_dict[kind]]

        num_manipulations = self.stim_ids.shape[0]/self.control_pos
        mean_tc = np.zeros((self.num_units, num_manipulations, 2))

        for manip in range(num_manipulations):
            temp = np.zeros((1, self.num_units)) # was originally empty()
            for k in range(self.control_pos - 1):
                # rates is size: num_trials x num_units
                temp = np.append(temp, rates[(manip*self.control_pos + k)], axis=0)
            temp = temp[1:, :] # remove first row of junk
            mean_tc[:, manip, 0] = np.mean(temp, axis=0)
            mean_tc[:, manip, 1] = sp.stats.sem(temp, axis=0)

        return mean_tc

    def reclassify_units(self):
        """use OMI and wave duration to reclassify units"""
        new_labels = list()
        if hasattr(self, 'cell_type_og') is False:
            print('no units were reclassified')
        else:
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

    def get_lfps(self, kind='run_boolean', engaged=True):
        lfps = [list() for x in range(len(self.shank_names))]
        # preallocation loop
        for shank in range(len(self.shank_names)):

            if engaged:
                for k, trials_ran in enumerate(self.num_good_trials):
                    lfps[shank].append(np.zeros(( self._lfp_min_samp, self.chan_per_shank[shank], trials_ran )))
            elif not engaged:
                for k, trials_not_ran in enumerate(self.num_slow_trials):
                    lfps[shank].append(np.zeros(( self._lfp_min_samp, self.chan_per_shank[shank], trials_not_ran )))

        for shank in range(len(self.shank_names)):
            for stim_ind, stim_id in enumerate(self.stim_ids):
                good_trial_ind = 0

                for k, seg in enumerate(self.f):
                    #if self.stim_ids_all[k] == stim_id and self.trial_class[kind][k] == engaged:
                    if  self.stim_ids_all[k] == stim_id and (self.trial_class[kind][k] == engaged):
                        lfps[shank][stim_ind][:, :, good_trial_ind] = self.lfp_data[shank][:, :, k]
                        good_trial_ind += 1
        self.lfps = lfps

    def get_psd(self, input_array, sr):
        """
        compute PSD of input array sampled at sampling rate sr
        input_array: a samples x trial array
        sr: sampling rate of input signal

        Returns x, and mean y and sem y
        """
        f_temp = list()
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

    def plot_spectrogram(self, f, t, Sxx_mat_temp, axis=None, color='k', error='sem', vmin=None, vmax=None, log=False):

        if axis == None:
            axis = plt.gca()

        mean_Sxx = np.mean(Sxx_mat_temp, axis=2)

        if vmin == None:
            im = axis.pcolormesh(t, f, mean_Sxx)
        elif vmin != None and log == True:
            im = axis.pcolormesh(t, f, mean_Sxx, norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        elif vmin!= None and log == False:
            im = axis.pcolormesh(t, f, mean_Sxx, vmin=vmin, vmax=vmax)
        #axis.set_yscale('log')
        axis.set_ylabel('Frequency (Hz)')
        axis.set_xlabel('Time (s)')

        return im

    def get_design_matrix(self, rate_type='abs_count', cond_inds=None, trode=None, cell_type=None, trim_trials=True):
        """
        creates design matrix for classification and regressions

        produces a design matrix where each row is data from a single trial and
        each column is a single unit.

        Parameters
        _________
        cond_inds: optional, default (None)
            Specify which trial types to extract data from
        trode: optional, default (None)
            Specify which electrode to extract data from
            TODO: allow user to specify more than one electrode
        cell_type: optional, default (None)
            Specify which cell types ('RS' or 'FS') to extract data from
        trim_trials: boolean, default (True)
            Specify whether to extract an equal number of trials from each
            condition using the condition with the least amount of trials.

        Returns
        -------
        X: 2-d array
            The design matrix containing all of the spike rates for all trials
            and units
        y: 1-d array
            The stimulus array where each element corresponds to a row in the
            desing matrix. This is necessary in order to specify which rates
            come from where.
        unit_inds: 1-d array
            The indices of the units selected.
        """

        print('\n-----make design matrix----')
        min_trials     = np.min(self.num_good_trials)
        num_cond       = len(self.stim_ids)
        kind_dict      = {'abs_rate': 0, 'abs_count': 1, 'evk_rate': 2, 'evk_count': 3}
        kind_of_tuning = [self.abs_rate, self.abs_count, self.evk_rate, self.evk_count]
        rates          = kind_of_tuning[kind_dict[rate_type]]

        # Find indices for units to be included. user can selected a specific
        # electrode/region, cell type (e.g. 'RS', 'FS') or a combinations of
        # both ('RS' cells from 'S1')

        if trode is not None and cell_type is not None:
            unit_inds = np.where(
                    np.logical_and(\
                    neuro.shank_ids == trode, neuro.cell_type == cell_type))[0]
            print('Collecting data from {} units and electrode {}'.format(cell_type, trode))
        elif trode is not None:
            unit_inds = np.where(self.shank_ids == trode)[0]
            print('Collecting data from all units and electrode {}'.format(trode))

        elif cell_type is not None:
            print('Collecting data from all electrodes and {} units'.format(cell_type))
            unit_inds = np.where(self.cell_type == cell_type)[0]

        else:
            print('Collecting data from all units and all electrodes')
            unit_inds = np.where(self.shank_ids >= 0)[0]

        num_units = len(unit_inds)

        # Preallocate the design matrix and stimulus ID array
        if trim_trials:
            X = np.zeros((num_cond*min_trials, num_units))
            y = np.ones((num_cond*min_trials, ))
        else:
            X = np.zeros((np.sum(self.num_good_trials), num_units))
            y = np.ones((np.sum(self.num_good_trials), ))

        # Create design matrix: go through all trials and add specified data to
        # the design and stimulus arrays
        last_t_ind = 0
        for k, cond in enumerate(rates):
            if trim_trials:
                X[min_trials*k:min_trials*(k+1)] = cond[0:min_trials, unit_inds]
                y[min_trials*k:min_trials*(k+1)] = k*y[min_trials*k:min_trials*(k+1)]
            else:
                min_trials = cond.shape[0]
                X[last_t_ind:(min_trials + last_t_ind)] = cond[:, unit_inds]
                y[last_t_ind:(min_trials + last_t_ind)] = k*y[last_t_ind:(min_trials + last_t_ind)]
                last_t_ind = min_trials + last_t_ind

        # Limit trial types:  only include data if it is apart of the specified
        # trial types/conditions
        if cond_inds is not None:
            good_inds = np.empty(())

            for cond_ind in cond_inds:
                # get indices from specified trial types/conditions
                good_inds = np.append(good_inds, np.where(y == cond_ind)[0])

            good_inds = good_inds[1::]
            good_inds = good_inds.astype(int)
            X = X[good_inds, :]
            y = y[good_inds,]

        # else do nothing and return all the data

        return X, y, unit_inds

    def spike_wt_design_matrix(self, bin_size=0.002, kernel_size=0.025, cond=8, wt_type=0, \
            trode=None, cell_type=None, analysis_window=[-1.0, 2.0]):
        """
        creates design matrix for classification and regressions

        TODO: edit the description
        produces a design matrix where each row is data from a single trial and
        each column is a single unit.

        Parameters
        _________
        cond_inds: optional, default (None)
            Specify which trial types to extract data from
        trode: optional, default (None)
            Specify which electrode to extract data from
            TODO: allow user to specify more than one electrode
        cell_type: optional, default (None)
            Specify which cell types ('RS' or 'FS') to extract data from
        trim_trials: boolean, default (True)
            Specify whether to extract an equal number of trials from each
            condition using the condition with the least amount of trials.

        Returns
        -------
        X: 2-d array
            The design matrix containing all of the spike rates (per bin) for all trials
            and units
        y: 1-d array
            The stimulus array where each element corresponds to a row in the
            desing matrix. This is necessary in order to specify which rates
            come from where.
        unit_inds: 1-d array
            The indices of the units selected.
        """

        print('\n-----make design matrix----')
        num_trials     = self.num_good_trials[cond]
        bins           = np.arange(analysis_window[0], analysis_window[1], bin_size)
        num_bins       = len(bins) - 1 # the last bin doesn't correspond to anything

        # Find indices for units to be included. user can selected a specific
        # electrode/region, cell type (e.g. 'RS', 'FS') or a combinations of
        # both ('RS' cells from 'S1')

        if trode is not None and cell_type is not None:
            unit_inds = np.where(
                    np.logical_and(\
                    neuro.shank_ids == trode, neuro.cell_type == cell_type))[0]
            print('Collecting data from {} units and electrode {}'.format(cell_type, trode))
        elif trode is not None:
            unit_inds = np.where(self.shank_ids == trode)[0]
            print('Collecting data from all units and electrode {}'.format(trode))

        elif cell_type is not None:
            print('Collecting data from all electrodes and {} units'.format(cell_type))
            unit_inds = np.where(self.cell_type == cell_type)[0]

        else:
            print('Collecting data from all units and all electrodes')
            unit_inds = np.where(self.shank_ids >= 0)[0]

        num_units = len(unit_inds)

        # Preallocate the design matrix and stimulus ID array
        # num_bins*num_trials x num_units
        X = np.zeros((num_bins*num_trials, num_units))
        y = np.ones((num_bins*num_trials, ))

        # Create design matrix: go through all trials and add specified data to
        # the design and stimulus arrays
        kernel = self.__make_kernel(kind='alpha', resolution=kernel_size)
#        kernel = make_kernel(kind='alpha', resolution=0.025)
        rebinned_spikes, t = self.rebin_spikes(bin_size=bin_size, analysis_window=analysis_window)
        wtt_start = np.argmin(np.abs(self.wtt - analysis_window[0]))
        wtt_stop  = np.argmin(np.abs(self.wtt - analysis_window[1]))

        if wtt_stop == num_bins:
            print('number of bins equal to number of whisker tracking indices')
        else:
            print('number of bins DOES NOT EQUAL number of whisker tracking indices')

        for k in range(num_trials):
            # for each trial get spike data and match it to whisker data

            # spikes
            spike_counts = rebinned_spikes[cond][:, k, unit_inds]

            psth = np.zeros((num_bins, num_units))
            for unit in range(num_units):
                psth[:, unit] = np.convolve(spike_counts[:, unit], kernel)[:-kernel.shape[0]+1]

            X[num_bins*k:num_bins*(k+1), :] = psth

            # whisker tracking
            y[num_bins*k:num_bins*(k+1), ]  = self.wt[cond][wtt_start:wtt_stop, wt_type, k]

        return X, y, unit_inds

    def get_burst_isi(self, kind='run_boolean'):
        """
        Compute the interspike interval for spikes during the stimulus period.
        get_burst_isi creates a list that has n_stimulus_types entries. Each
        stimulus type has a list which contains a numpy array for each unit.

        These values can be used to identify bursting activity.
        """

        if self.wt_bool:
            wt          = list()
        if kind == 'wsk_boolean' and not self.wt_bool:
            warnings.warn('**** NO WHISKER TRACKING DATA AVAILABLE ****\n\
                    using run speed to select good trials')
            kind = 'run_boolean'
        elif kind == 'wsk_boolean' and self.wt_bool:
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

            # iterate through all HDF5 trial segments
            for k, seg in enumerate(self.f):

                # if a trial segment is from the current stim_id and it is a
                # running trial get the spike ISIs.
                if self.stim_ids_all[k] == stim_id and self.trial_class[kind][k] == True:

                    # get the stimulus start and stop times for this trial
                    stim_start = self.f[seg].attrs['stim_times'][0]
                    stim_stop  = self.f[seg].attrs['stim_times'][1]

                    # iterate through all unit's spike trains. find times units fired
                    # during the stimulus period and calculate the inter-spike-
                    # interval between them. This will be used to determine
                    # bursting
                    for unit, spike_train in enumerate(self.f[seg + '/spiketrains']):

                        # get spike times during stimulus period
                        spk_times_all = self.f[seg + '/spiketrains/' + spike_train][:]
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

################################################################################
###### Eight-Position experiment specific functions #####
################################################################################

    def get_selectivity(self):
        """compute selectivity for all units and manipulations"""

        if hasattr(self, 'abs_rate') is False:
            self.rates()
        control_pos = self.control_pos
        num_manipulations = self.stim_ids.shape[0]/control_pos
        sel_mat = np.zeros((self.num_units, num_manipulations))

        for manip in range(num_manipulations):
            for unit_index in range(self.num_units):
                meanr = [np.mean(k[:, unit_index]) for k in self.abs_rate]
                # minus 1 to exclude no contact/control position
                x = np.asarray(meanr[(manip*control_pos):((manip+1)*control_pos-1)])
                # calculate selectivity for unit_index during manipulation
                sel_mat[unit_index, manip] = \
                        1 - ((np.linalg.norm(x/np.max(x))- 1)/\
                        (np.sqrt(x.shape[0]) - 1))

        self.selectivity = sel_mat

    def get_bootstrap_selectivity(self):
        if hasattr(self, 'abs_rate') is False:
            self.rates()

        if hasattr(self, 'selectivity') is False:
            self.get_selectivity()

        control_pos = self.control_pos
        num_manip   = self.stim_ids.shape[0]/control_pos
        num_pos     = control_pos - 1
        num_units   = self.num_units

        # compute difference in selectivity values for each unit and manip
        sel_diff = np.diff(self.selectivity, axis=1)

        # combine spike rate data
        for manip in range(1, num_manip):

            # for the no light conditions
            all_rates = np.zeros((1, num_units))
            for pos in np.arange(0, num_pos):
                all_rates = np.append(all_rates, self.abs_rate[pos], axis=0)

            all_rates = all_rates[1:, :]

            # for manipulation n conditions:
            manip_rates = np.zeros((1, num_units))
            for pos in np.arange(manip*control_pos, ((manip+1)*control_pos-1)):
                manip_rates = np.append(manip_rates, self.abs_rate[pos], axis=0)

            manip_rates = manip_rates[1:, :]

        ##### This is the part that needs development #####
        ##### This is the part that needs development #####

        # iterate through each position for a given manipulation
        boot_samps = np.zeros((10000,))
        boot_samps1 = np.zeros((10000,))
        for n in range(10000):
            temp_tc = np.zeros((num_pos, ))
            temp_tc1 = np.zeros((num_pos,))
            for k in range(num_pos):
                # grab random samples from each position (no light)
                nsamp = neuro.num_good_trials[k]
                meanr_temp = np.mean(np.random.choice(self.abs_rate[k][:, 2], size=nsamp, replace=True))
                temp_tc[k] = meanr_temp

                # grab random samples from each position (s1 light)
                nsamp = neuro.num_good_trials[k+9]
                meanr_temp = np.mean(np.random.choice(self.abs_rate[k+9][:, 2], size=nsamp, replace=True))
                temp_tc1[k] = np.random.choice(self.abs_rate[k+9][:, 0])

            # compute selectivity for fake tuning curve 1
            sel_temp = \
                    1 - ((np.linalg.norm(temp_tc/np.max(temp_tc))- 1)/\
                        (np.sqrt(temp_tc.shape[0]) - 1))
            boot_samps[n] = sel_temp

            # compute selectivity for fake tuning curve 2
            sel_temp1 = \
                    1 - ((np.linalg.norm(temp_tc1/np.max(temp_tc1))- 1)/\
                        (np.sqrt(temp_tc1.shape[0]) - 1))
            boot_samps1[n] = sel_temp1

    def get_omi(self, pos=-1):
        """
        calculates mean OMI or OMI at a specifie position. Returns a
        unit X number of optogenetic manipulations

        if pos=-1, mean OMI is calculated
        """
        if hasattr(self, 'abs_rate') is False:
            self.rates()
        control_pos = self.control_pos
        num_manipulations = self.stim_ids.shape[0]/control_pos
        omi_mat = np.zeros((self.num_units, num_manipulations-1))

        for manip in range(num_manipulations - 1):
            for unit_index in range(self.num_units):
                meanr_abs = np.array([np.mean(k[:, unit_index]) for k in self.abs_rate])
                # NO minus 1, no contact/control position is INCLUDED
                # (manip+1)*control_pos = (0+1)*9, (1+1)(9) = 18
                omi_tc  = (meanr_abs[((manip+1)*control_pos):((manip+1+1)*control_pos)] - meanr_abs[:self.control_pos]) / \
                                (meanr_abs[((manip+1)*control_pos):((manip+1+1)*control_pos)] + meanr_abs[:self.control_pos])
                if pos == -1:
                    omi_mat[unit_index, manip] = omi_tc.mean()
                else:
                    omi_mat[unit_index, manip] = omi_tc[pos]

        return omi_mat

    def get_preferred_position(self):
        """
        calculated the preferred position. Returns a
        unit X number of manipulations
        """
        if hasattr(self, 'abs_rate') is False:
            self.rates()
        control_pos = self.control_pos
        num_manipulations = self.stim_ids.shape[0]/control_pos
        pref_mat = np.zeros((self.num_units, num_manipulations))
        positions = range(control_pos-1)

        for manip in range(num_manipulations):
            for unit_index in range(self.num_units):
                meanr_abs = np.array([np.mean(k[:, unit_index]) for k in self.abs_rate])
                weights = meanr_abs[(manip*control_pos):((manip+1)*control_pos-1)]
                pref_mat[unit_index, manip] = np.sum(weights*positions)/np.sum(weights)

        for region in self.shank_ids:
            pref_mat[self.shank_ids == region, :] = pref_mat[self.shank_ids == region, :] - pref_mat[self.shank_ids == region, 0].mean()

        self.preference = pref_mat

    def get_best_contact(self):
        """calculates best contact for all units from evoked rate tuning curve"""
        best_contact = np.zeros((self.num_units,))
        for unit_index in range(self.num_units):
            meanr = np.array([np.mean(k[:, unit_index]) for k in self.evk_rate])
            best_contact[unit_index,] = np.argmax(meanr[:self.control_pos])
        self.best_contact = best_contact.astype(int)

    def get_sensory_drive(self):
        """determine which units are sensory driven"""
        if hasattr(self, 'abs_count') is False:
            self.rates()
        control_pos = self.control_pos - 1
        driven = list()
        # compare only no light positions with control/no contact position
        to_compare = [ (k, control_pos) for k in range(control_pos)]

        for unit in range(self.num_units):
            groups = list()
            for k in range(control_pos + 1):
                # append all rates
                groups.append(self.abs_count[k][:, unit])
            # test for sensory drive
            H, p_omnibus, Z_pairs, p_corrected, reject = dunn.kw_dunn(groups, to_compare=to_compare, alpha=0.05, method='simes-hochberg') # or 'bonf' for bonferoni

            if reject.any():
                driven.append(True)
            else:
                driven.append(False)

        self.driven_units = np.asarray(driven)

    def get_burst_rate(self, unit_ind=0, trial_type=0, start_time=0.5, stop_time=1.5):
        """
        """
        # if the rates have not been calculated do that now
        if hasattr(self, 'binned_spikes') is False:
            print('Spikes have not been binned! Binning data now.')
            self.rates()

        print('Computing burst rate for unit {} and trial_type {}'.format(unit_ind, trial_type))

        count_mat = self.binned_spikes[trial_type][:, :, unit_ind] # returns bins x num_trials array
        burst_rate_mat = np.zeros((count_mat.shape[1],))

        for trial in range(count_mat.shape[1]):
            trial_inds = np.where(count_mat[:, trial] > 0)[0]
            spike_times = self._bins[trial_inds]

            burst_times = list()
            data = spike_times
            if len(data) > 3:
                start, length, RS = ranksurprise.burst(data, limit=50e-3, RSalpha=0.1)
                for k in range(len(start)):
                    burst_times.extend(data[start[k]:(start[k]+length[k])])

                if len(burst_times) > 0:
                    num_bursts = np.sum(np.logical_and(np.asarray(burst_times) > start_time, \
                            np.asarray(burst_times) < stop_time))
                    analysis_time = stop_time - start_time
                    burst_rate = num_bursts/analysis_time
                else:
                    burst_rate = 0
                burst_rate_mat[trial] = burst_rate

        return burst_rate_mat

    def get_spike_rates_per_bin(self, bins=[0, 0.5, 1.0, 1.5], unit_ind=0, trial_type=0):
        """
        Compute firing rate for arbitrary bin sizes

        This bins spike count data for specified bin locations and computes the
        firing rate for that bin.

        Parameters
        ----------
        bins: array_like
            bin locations. Bin width will be determined from these values.
            Must be equally spaced bins!
        unit_ind: int
            unit index used to specify which unit's spikes should be binned
        trial_type: int
            trial type index used to specify which trial to be analyzed

        Returns
        -------
        rates_per_bins: an array of size "number of bins" X "number of trials"


        """
        # if the rates have not been calculated do that now
        if hasattr(self, 'binned_spikes') is False:
            print('Spikes have not been binned! Binning data now.')
            self.rates()

        print('Computing spike rates per bin for unit {} and trial_type {}'.format(unit_ind, trial_type))

        bins = np.asarray(bins)
        dt = np.diff(bins)[0]

        count_mat = self.binned_spikes[trial_type][:, :, unit_ind] # returns bins x num_trials array
        rates_per_bin = np.zeros((count_mat.shape[1], bins.shape[0]-1))
        for trial in range(count_mat.shape[1]):
            trial_inds = np.where(count_mat[:, trial] > 0)[0]
            spike_times = self._bins[trial_inds]
            counts = np.histogram(spike_times, bins=bins)[0]
            rates  = counts/dt
            rates_per_bin[trial, :] = rates

        return rates_per_bin

    def get_adaptation_ratio(self, unit_ind=0, bins=[0.5, 1.0, 1.5]):
        """compute adaptation ratio for a given unit and bins"""
        bins = np.asarray(bins)
        ratios = np.zeros((1, self.stim_ids.shape[0], bins.shape[0]-1))
        for cond in range(self.stim_ids.shape[0]):
            ratios[0, cond, :] = np.mean(self.get_spike_rates_per_bin(bins=bins, unit_ind=unit_ind, trial_type=cond), axis=0)
        for cond in range(self.stim_ids.shape[0]):
            ratios[0, cond, :] = ratios[0, cond, :]/float(ratios[0, cond, 0])

        return ratios

    def get_rates_vs_strength(self, normed=False):
        """sort firing rates from min to max. normalize if specified"""

        if hasattr(self, 'abs_rate') is False:
            self.rates()

        control_pos = self.control_pos
        num_cond = self.stim_ids.shape[0]
        num_manipulations = num_cond/control_pos
        meanr_sorted = np.zeros((self.num_units, control_pos, num_manipulations))
        semr_sorted = np.zeros((self.num_units, control_pos, num_manipulations))

        for unit_index in range(self.num_units):

            # compute mean and standard error firing rates
            meanr_abs = np.array([np.mean(k[:, unit_index]) for k in self.abs_rate])
            semr_abs  = np.array([sp.stats.sem(k[:, unit_index]) for k in self.abs_rate])

            # sort from smallest response to larget
            sort_inds = np.argsort(meanr_abs[0:control_pos-1])

            # if norm then normalize by max firing rate of no light conditions
            if normed:
                maxr = meanr_abs[sort_inds[-1]] # last index is for max value
                for pos in range(num_cond):
                    meanr_abs = np.array([np.mean(k[:, unit_index]/maxr) for k in self.abs_rate])
                    semr_abs  = np.array([sp.stats.sem(k[:, unit_index]/maxr) for k in self.abs_rate])

            # for each manipulation sort and save values to arrays
            for manip in range(num_manipulations):

                # add control/no contact position
                meanr_sorted[unit_index, 0, manip] = meanr_abs[(control_pos*(manip+1)-1)]
                semr_sorted[unit_index, 0, manip]  = semr_abs[(control_pos*(manip+1)-1)]

                # add contact positions
                temp_mean = meanr_abs[(manip*control_pos):(manip+1)*control_pos]
                temp_sem  = semr_abs[(manip*control_pos):(manip+1)*control_pos]

                meanr_sorted[unit_index, 1:(control_pos), manip] = temp_mean[sort_inds]
                semr_sorted[unit_index, 1:(control_pos), manip]  = temp_sem[sort_inds]

        return meanr_sorted, semr_sorted

    def get_sparseness(self, kind='lifetime', cond=None):
        """
        compute one of two sparsity measures

        if kind is "lifetime" each a matrix with k rows and n columns will
        be returned. Where k is the number of units and n is the number of
        manipulations used (e.g. no light + light conditions).

        if kind is "population" nothing will be returned yet

        """
        control_pos = self.control_pos
        num_cond    = self.stim_ids.shape[0]
        num_manip   = num_cond/control_pos
        num_units   = self.num_units
        n           = control_pos - 1

        S_life_mean = np.zeros((num_units, num_manip))
        S_life_all= np.zeros((num_units, num_manip))

        # compute lifetime sparseness
        if kind == 'lifetime' and cond == None:
            for unit in range(num_units):
                meanr_abs = np.array([np.mean(k[:, unit]) for k in self.abs_rate])
                for manip in range(num_manip):

                    # method one: use mean response values
                    start_ind = manip*control_pos
                    stop_ind  = (manip + 1)*control_pos - 1

                    L1 = (np.sum(meanr_abs[start_ind:stop_ind])/n)**2
                    L2 = np.sum(meanr_abs[start_ind:stop_ind]**2)/n
                    S  = (1 - (L1/L2)) / (1 - (1/n))

                    S_life_mean[unit, manip] = S

                    # method two: use single trial responses
                    r_all = list()
                    for k in np.arange(start_ind, stop_ind):
                        unit_response = self.abs_rate[k][:, unit]
                        r_all.extend(unit_response)
                    r_all = np.asarray(r_all)

                    n  = r_all.shape[0]
                    L1 = (np.sum(r_all)/n)**2
                    L2 = np.sum(r_all**2)/n
                    S  = (1 - (L1/L2)) / (1 - (1/n))

                    S_life_all[unit, manip] = S

#        # compute population sparseness
#        if kind == 'population' and cond != None:
#            for unit in range(num_units):
#                #TODO specify whether to use M1 or S1 or all units!!!

        return S_life_mean, S_life_all


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

    def pta(self, cond=0, unit_ind=0, window=[-0.100, 0.100], dt=0.005, analysis_window=[0.5, 1.5]):
        """
        Create a protraction triggered histogram/average
        Returns: mean counts per bin, sem counts per bin, and bins
        """
        if hasattr(self, 'wt') is False:
            warnings.warn('no whisking data!')

        bins      = np.arange(window[0], window[1], dt)
        count_mat = np.zeros((1, bins.shape[0] - 1)) # trials x bins

        # iterate through all trials and count spikes
        for trial_ind in range(self.num_good_trials[cond]):
            all_spike_times = self.bins_t[self.binned_spikes[cond][:, trial_ind, unit_ind].astype(bool)]
            windowed_spike_times = np.logical_and(all_spike_times > analysis_window[0],\
                    all_spike_times < analysis_window[1])
            protraction_times = self.get_protraction_times(cond, trial_ind, analysis_window)

            # bin spikes for each protraction and add to count_mat
            for p_time in protraction_times:
                temp_counts = np.histogram(all_spike_times[windowed_spike_times] - p_time, bins)[0].reshape(1, bins.shape[0]-1)
                count_mat   = np.concatenate((count_mat, temp_counts), 0) # extend the number of rows

        count_mat = count_mat[1::, :]
        out1 = np.mean(count_mat, 0)/dt
        out2 = sp.stats.sem(count_mat, 0)
        out3 = bins

        return out1, out2, out3

    def get_pta_depth(self, window=[-0.100, 0.100], dt=0.005, analysis_window=[0.5, 1.5]):
        """
        Calculates modulation depth from protraction triggered averages
        Here modulation depth is the coefficient of variation (std/mean)
        """
        if hasattr(self, 'wt') is False:
            warnings.warn('no whisking data!')
        control_pos = self.control_pos
        num_conditions = self.stim_ids.shape[0]
        mod_mat = np.zeros((self.num_units, num_conditions))

        print ('working on unit:')
        for unit_index in range(self.num_units):
            print(unit_index)
            for cond in range(num_conditions):
                    spks_bin, _, _ = self.pta(cond=cond, unit_ind=unit_index, window=window, dt=dt, analysis_window=analysis_window)
                    #mod_depth = (np.max(spks_bin) - np.min(spks_bin)) / np.mean(spks_bin)
                    mod_depth = np.var(spks_bin)/np.mean(spks_bin)
                    mod_mat[unit_index, cond] = mod_depth

        self.mod_index = mod_mat

    def eta(self, event_times, cond=0, unit_ind=0, window=[-0.050, 0.050], dt=0.001, analysis_window=[0.5, 1.5]):
        """
        Create an event triggered histogram/average. Given a list of times this
        will calculate the mean spike rate plus standard error within a given
        windowed centered at each event time.

        Input: event times as a vector
        TODO: event times as a matrix (i.e. a vector per specific trial)
        Returns: mean counts per bin, sem counts per bin, and bins
        """

        bins      = np.arange(window[0], window[1], dt)
        count_mat = np.zeros((1, bins.shape[0] - 1)) # trials x bins

        # iterate through all trials and count spikes
        for trial_ind in range(self.num_good_trials[cond]):
            all_spike_times = self.bins_t[self.binned_spikes[cond][:, trial_ind, unit_ind].astype(bool)]
            windowed_spike_times = np.logical_and(all_spike_times > analysis_window[0],\
                    all_spike_times < analysis_window[1])
            protraction_times = self.get_protraction_times(cond, trial_ind, analysis_window)

            # bin spikes for each protraction and add to count_mat
            for e_time in event_times:
                temp_counts = np.histogram(all_spike_times[windowed_spike_times] - e_time, bins)[0].reshape(1, bins.shape[0]-1)
                count_mat   = np.concatenate((count_mat, temp_counts), 0) # extend the number of rows

        count_mat = count_mat[1::, :]
        out1 = np.mean(count_mat, 0)/dt
        out2 = sp.stats.sem(count_mat, 0)
        out3 = bins

        return out1, out2, out3

    def eta_wt(self, event_times, cond=0, kind='angle', window=[-0.050, 0.050]):
#            cond=0, unit_ind=0, window=[-0.050, 0.050], dt=0.001, analysis_window=[0.5, 1.5]):
        """
        Create an event triggered trace of a specified whisking parameter
        Input: event times as a vector
        TODO: event times as a matrix (i.e. a vector per specific trial)
        Returns: mean trace, sem of traces, and the trace matrix (trials x window length)
        """
        self.wt_type_dict = {'angle':0, 'set-point':1, 'amplitude':2, 'phase':3, 'velocity':4, 'whisking':5}

        num_inds_pre  = len(np.where(np.logical_and(self.wtt >= 0, self.wtt < np.abs(window[0])))[0])
        num_inds_post = len(np.where(np.logical_and(self.wtt >= 0, self.wtt < np.abs(window[1])))[0])
        dt = self.wtt[1] - self.wtt[0]
        trace_time = np.arange(-num_inds_pre, num_inds_post+1)*dt

        # matrix size: trials x number of indices taken from interested trace
        trace_mat = np.zeros((1, num_inds_pre + num_inds_post + 1))
        kind_ind = self.wt_type_dict[kind]

        # iterate through all trials and count spikes
        for trial_ind in range(self.num_good_trials[cond]):
            for e_time in event_times:
                trace_ind  = np.argmin(np.abs(self.wtt - e_time)) # find wt index closest to event time
                trace_inds = np.arange(trace_ind - num_inds_pre, trace_ind + num_inds_post + 1)
                trace_temp = self.wt[cond][trace_inds, kind_ind, trial_ind].reshape(1, trace_mat.shape[1])
                trace_mat = np.concatenate((trace_mat, trace_temp))

        trace_mat = trace_mat[1::, :]
        out1 = np.mean(trace_mat, axis=0)
        out2 = sp.stats.sem(trace_mat, axis=0)
        out3 = trace_mat
        out4 = trace_time

        return out1, out2, out3, out4

    def sta_wt(self, cond=0, unit_ind=0, analysis_window=[0.5, 1.5]):
        """
        Create a spike triggered array for whisker tracking
        Returns: array (total spikes for specified unit x values) where each
        entry is the whisker tracking value (e.g. angle, phase, set-point) when
        the specified unit spiked.

        The second array is all the whisker tracking values that occurred
        during the analysis window for all analyzed trials
        """
        st_vals = np.zeros((1, 5))
        all_vals    = np.zeros((1, 5))
        stim_inds   = np.logical_and(self.wtt >= analysis_window[0], self.wtt <= analysis_window[1])

        # iterate through all trials and count spikes
        for trial_ind in range(self.num_good_trials[cond]):
            all_spike_times = self.bins_t[self.binned_spikes[cond][:, trial_ind, unit_ind].astype(bool)]
            windowed_spike_indices = np.logical_and(all_spike_times > analysis_window[0],\
                    all_spike_times < analysis_window[1])
            windowed_spike_times = all_spike_times[windowed_spike_indices]

            # iterate through all spike times and measure specified whisker
            # parameter
            for stime in windowed_spike_times:
                wt_index = np.argmin(np.abs(stime - self.wtt))
                temp_st_vals = self.wt[cond][wt_index, 0:5, trial_ind].reshape(1, 5)
                st_vals = np.concatenate((st_vals, temp_st_vals), axis=0)

            # add all whisker tracking values in analysis window to matrix
            all_vals = np.concatenate((all_vals, self.wt[cond][stim_inds, 0:5, trial_ind]), axis=0)

        st_vals  = st_vals[1::, :]
        all_vals = all_vals[1::, :]

        return st_vals, all_vals

    def st_norm(self, st_vals, all_vals, wt_type, bins, dt):
        """
        normalizes phase/spike counts

        Divides each phase bin by the occupancy of that bin. That is, it
        divides by the number of times the whiskers happened to occupy each bin
        """

        if wt_type == 0:
            st_count = np.histogram(st_vals, bins=bins)[0].astype(float)
            all_count = np.histogram(all_vals, bins=bins)[0].astype(float)
        else:
            st_count = np.histogram(st_vals[:, wt_type], bins=bins)[0].astype(float)
            all_count = np.histogram(all_vals[:, wt_type], bins=bins)[0].astype(float)

        count_norm = np.nan_to_num(st_count/all_count) / (dt * 0.002)

        return count_norm

    def sg_smooth(self, data, win_len=5, poly=3, neg_vals=False):
        """
        Smooth a 1-d array with a Savitzky-Golay filter

        Arguments
        Data: the 1-d array to be smoothed
        win_len: the size of the smoothing window to be used. It MUST be an odd.
        poly: order of the smoothing polynomial. Must be less than win_len
        neg_vals: whether to convert negative values to zero.

        Returns smoothed array
        """
        smooth_data = sp.signal.savgol_filter(data, win_len, poly, mode='wrap')
        if neg_vals is False:
            smooth_data[smooth_data < 0] = 0
        return smooth_data

    def get_phase_modulation_depth(self, bins=np.linspace(-np.pi, np.pi, 40)):
        """
        computes spike-phase modulation depth for each unit

        Parameters
        _________
        bins: array-like
            specify the bin locations used to bin spike-phase data

        Returns
        -------
        mod_index: 3-d array
            The 3-d array has dimensions:
                row: number of units
                col: number of trial types
                3-d: vector strength, vector direction, coefficient of variation
            The array is added to the neuro class and can be called with dot
            notations <name of class object>.mod_index
        """

        # Main function code
        print('\n-----get_phase_modulation_depth-----')
        if hasattr(self, 'wt') is False:
            warnings.warn('no whisking data!')

        dt       = bins[1] - bins[0]
        bins_pos = bins + np.pi # must use positive bins with pycircstat, remeber to offset results by -pi
        wt_type  = 3 # {0:'angle', 1:'set-point', 2:'amplitude', 3:'phase', 4:'velocity'}
        mod_mat  = np.zeros((self.num_units, len(self.num_all_trials), 3))
        mod_pval = np.zeros((self.num_units, len(self.num_all_trials)))

        for uid in range(self.num_units):
            print('working on unit: {}'.format(uid))
            for k in range(len(self.num_all_trials)):

                # compute spike-rate per phase bin
                st_vals, all_vals = self.sta_wt(cond=k, unit_ind=uid) # analysis_window=[0.5, 1.5]
                count_norm        = self.st_norm(st_vals, all_vals, wt_type, bins, dt)
                smooth_data       = self.sg_smooth(count_norm)

                # pycircstat returns positive values (yes, verified). So zero
                # corresponds to negative pi and six is close to positive pi

                # compute vector strength
                mod_mat[uid, k, 0] = pycirc.descriptive.resultant_vector_length(bins_pos[:-1], smooth_data) # angles in radian, weightings (counts)

                # compute vector angle (mean direction)
                mod_mat[uid, k, 1] = pycirc.descriptive.mean(bins_pos[:-1], smooth_data) - np.pi # angles in radian, weightings (counts)
                # compute coefficient of variation
                mod_mat[uid, k, 2] = np.std(smooth_data)/np.mean(smooth_data)

                # stats
                # compute Rayleigh test for non-uniformity around unit circle
                mod_pval[uid, k] = pycirc.tests.rayleigh(bins_pos[:-1], smooth_data)[0]

        self.mod_index = mod_mat
        self.mod_pval  = mod_pval

    def sta_lfp(self, unit_ind=0, shank=0, contact=0, cond=0, analysis_window=[0.6, 1.4], bin_window=[-0.25, 0.25]):
        """
        Create a spike triggered array of LFPs
        Returns: array (total spikes for specified unit x values) where each
        entry is the whisker tracking value (e.g. angle, phase, set-point) when
        the specified unit spiked.

        The second array is all the whisker tracking values that occurred
        during the analysis window for all analyzed trials
        """
        sr = 1500
        window = np.arange(bin_window[0]*sr, bin_window[1]*sr) # 1/1500 is the sampling rate of the LFP traces
        window_size = window.shape[0]
        num_trials = self.num_good_trials[cond]
        sta_array = np.zeros((1, window_size))

        # iterate through all trials and count spikes
        for trial_ind in range(num_trials):
            all_spike_times = self.bins_t[self.binned_spikes[cond][:, trial_ind, unit_ind].astype(bool)]
            windowed_spike_indices = np.logical_and(all_spike_times > analysis_window[0],\
                    all_spike_times < analysis_window[1])
            windowed_spike_times = all_spike_times[windowed_spike_indices] # spike times to align LFPs to
#            windowed_spike_times = np.random.rand(windowed_spike_times.shape[0]) + 0.6

            # iterate through all spike times and grabe LFP signal of
            # window_size
            for k, stime in enumerate(windowed_spike_times):
                lfp_index = np.argmin(np.abs(stime - self.lfp_t))
                lfp_inds  = window - lfp_index
                lfp_inds = lfp_inds.astype(int)
                lfp_temp = self.lfps[shank][cond][lfp_inds, contact, cond] # get individual LFP trace
                sta_lfp  = lfp_temp.reshape(1, lfp_temp.shape[0])
                # I may need to filter it at the specified frequency here
                #TODO filter LFP trace with a Butterworth filter (see online example)
                y = self.butterworth_filter(sta_lfp, 10, 55, 1500.0)

                # add all whisker tracking values in analysis window to matrix
                sta_array = np.concatenate((sta_array, y), axis=0)

        sta_array = sta_array[1::, :]

        return sta_array, window

    def butterworth_filter(self, data, lowcut, highcut, fs, order=4):
        #ndpass(lowcut, highcut, fs, order=5):
        nyq  = 0.5 * fs
        low  = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y    = lfilter(b, a, data)

        return y



################################################################################
###### JB_Behavior angle experiment specific functions #####
################################################################################

    def get_psychometric_curve(self):
        """
        compute psychometric curve for entire behavioral experiment
        """
        num_cond = len(self.stim_ids)
        prob_lick = np.zeros((num_cond, ))
        for manip in range(num_cond/self.control_pos):
            for cond in range(num_cond - 1):
                prob_lick[cond] = float(np.sum(self.lick_bool[cond + manip]))\
                        / self.lick_bool[cond + manip].shape[0]


        pos = range(1, self.control_pos)
        line_color = ['k','r','b']
        fig, ax = plt.subplots()
        for control_pos_count, first_pos in enumerate(range(0, len(self.stim_ids), self.control_pos)):
            ax.plot(pos[0:self.control_pos-1],\
                    prob_lick[(control_pos_count*self.control_pos):((control_pos_count+1)*self.control_pos-1)],\
                    color=line_color[control_pos_count], marker='o', markersize=6.0, linewidth=2)
            # plot control position separately from stimulus positions
            ax.plot(self.control_pos, prob_lick[(control_pos_count+1)*self.control_pos-1],\
                    color=line_color[control_pos_count], marker='o', markersize=6.0, linewidth=2)

        return prob_lick

    def get_lick_rate(self):
        """
        compute lick rate for each condition
        """
        num_cond = len(self.stim_ids)
        lick_rate = [list() for x in range(num_cond)]
        for cond in range(num_cond):
            for licks in self.licks[cond]:

                lick_rate_temp = np.sum(np.logical_and(licks > 0, licks < 1))
                if lick_rate_temp == 0:
                    lick_rate[cond].append(0)
                elif  lick_rate_temp > 0:
                    lick_rate[cond].append(lick_rate_temp)

        return lick_rate

    def plot_lick_raster(self):

        fig, ax = plt.subplots(1, self.control_pos, figsize=(12, 6), sharex=True, sharey=True)
        for cond in range(len(self.stim_ids)):
            trial = 0
            for licks in self.licks[cond]:
                if not np.isnan(np.sum(licks)):
                    ax[cond].vlines(licks, trial, trial+1, color='k', linewidth=1.0)
                    trial += 1
        max_ylim = ax[cond].get_ylim()[1]
        for cond in range(len(self.stim_ids)):
            ax[cond].axvspan(0, 1, alpha=0.3, color='green')
            ax[cond].set_ylim(0, max_ylim)

    def plot_time2lick(self, t_start=0):
        """
        compute time (mean +/- sem) to first lick for all angles
        """
        num_cond = len(self.stim_ids)
        time2lick_mean = np.zeros((num_cond, ))
        time2lick_sem = np.zeros((num_cond, ))

        for cond in range(num_cond):
            lick_temp = list()
            for licks in self.licks[cond]:
                if not np.isnan(np.sum(licks)):
                    lick_inds = np.where(licks > t_start)[0]
                    if lick_inds.shape[0] > 0:
                        lick_temp.append(licks[lick_inds[0]])

            time2lick_mean[cond] = np.mean(lick_temp)
            time2lick_sem[cond]  = sp.stats.sem(lick_temp)

        pos = range(1, self.control_pos)
        line_color = ['k','r','b']
        fig, ax = plt.subplots()
        for control_pos_count, first_pos in enumerate(range(0, len(self.stim_ids), self.control_pos)):
            ax.errorbar(pos[0:self.control_pos-1],\
                    time2lick_mean[(control_pos_count*self.control_pos):((control_pos_count+1)*self.control_pos-1)],\
                    yerr=time2lick_sem[(control_pos_count*self.control_pos):((control_pos_count+1)*self.control_pos-1)],\
                    color=line_color[control_pos_count], marker='o', markersize=6.0, linewidth=2)
            # plot control position separately from stimulus positions
            ax.errorbar(self.control_pos, time2lick_mean[(control_pos_count+1)*self.control_pos-1],\
                    yerr=time2lick_sem[(control_pos_count+1)*self.control_pos-1],\
                    color=line_color[control_pos_count], marker='o', markersize=6.0, linewidth=2)



###############################################################################
##### Plotting Functions #####
###############################################################################

    def plot_tuning_curve(self, unit_ind=None, kind='abs_count', axis=None):
        """
        make_simple_tuning_curve allows one to specify what type of tuning
        curve to plot as well as which unit for a single tuning curve or all
        units for a subplot of tuning curves

        Kinds of tuning curves:
        Absolute rate:   'abs_rate'
        Absolute counts: 'abs_count'
        Evoked rate:     'evk_rate'
        Evoked counts:   'evk_count'
        """

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
        if unit_ind != None:

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
                        fmt=line_color[control_pos_count], marker='o', markersize=6.0, linewidth=2)
                # plot control position separately from stimulus positions
                ax.errorbar(control_pos, meanr[(control_pos_count+1)*control_pos-1], yerr=stder[(control_pos_count+1)*control_pos-1],\
                        fmt=line_color[control_pos_count], marker='o', markersize=6.0, linewidth=2)

        # if all tuning curves from all units are to be plotted
        else:
            # determine number of rows and columns in the subplot
            unit_ind = range(self.num_units)
            num_rows, num_cols = 3, 3

            num_manipulations = len(self.stim_ids)/control_pos # no light, light 1 region, light 2 regions

            unit_count, plot_count = 0, 0
            fig, ax = plt.subplots(num_rows, num_cols, figsize=(14, 10))
            for unit in unit_ind:
                meanr = [np.mean(k[:, unit]) for k in kind_of_tuning[kind_dict[kind]]]
                stder = [np.std(k[:, unit]) / np.sqrt(k[:, unit].shape[0]) for k in kind_of_tuning[kind_dict[kind]]]
                for control_pos_count, first_pos in enumerate(range(0, len(self.stim_ids), control_pos )):
                    row, col = np.unravel_index(plot_count, (num_rows, num_cols))
                    # compute the means and standard errors

#                    ax = plt.subplot(num_rows, num_cols, plot_count+1)
                    # plot stimulus positions separately from control so the
                    # control position is not connected with the others
                    ax[row][col].errorbar(pos[0:control_pos-1],\
                            meanr[(control_pos_count*control_pos):((control_pos_count+1)*control_pos-1)],\
                            yerr=stder[(control_pos_count*control_pos):((control_pos_count+1)*control_pos-1)],\
                            fmt=line_color[control_pos_count], marker='o', markersize=6.0, linewidth=2)
                    # plot control position separately from stimulus positions
                    ax[row][col].errorbar(control_pos, meanr[(control_pos_count+1)*control_pos-1], yerr=stder[(control_pos_count+1)*control_pos-1],\
                            fmt=line_color[control_pos_count], marker='o', markersize=6.0, linewidth=2)

                ax[row][col].set_title('shank: ' + self.shank_names[self.shank_ids[unit]] + \
                        ' depth: ' + str(self.depths[unit]) + \
                        '\ncell type: ' + str(self.cell_type[unit]))
                        #' depth: ' + str(self.neo_obj.segments[0].spiketrains[unit].annotations['depth']) + \
                ax[row][col].plot([0, control_pos+1],[0,0],'--k')
                ax[row][col].set_xlim(0, control_pos+1)
                ax[row][col].set_ylim(ax[row][col].get_ylim()[0]-1, ax[row][col].get_ylim()[1]+1)
                ax[row][col].set_xticks([])
                ax[row][col].set_xticks(x_vals, labels)
                ax[row][col].set_xlabel('Bar Position', fontsize=8)
                fig.show()
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
                        #fig = plt.subplots(num_rows, num_cols, figsize=(14, 10))
                        fig, ax = plt.subplots(num_rows, num_cols, figsize=(14, 10))
                    plot_count = 0

    def plot_raster(self, unit_ind=0, trial_type=0, axis=None, burst=False):
        """
        Makes a raster plot for the given unit index and trial type.
        If called alone it will plot a raster to the current axis. This function
        is called by plot_all_rasters and returns an axis handle to the current
        subplot. This allows plot_all_rasters to plot rasters in the appropriate
        subplots.
        """
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
            ax.vlines(spike_times, trial, trial+1, color='k', linewidth=1.0)

            if burst:
                burst_times = list()
                data = spike_times
                if len(data) > 3:
                    start, length, RS = ranksurprise.burst(data, limit=50e-3, RSalpha=0.1)
                    for k in range(len(start)):
                        burst_times.extend(data[start[k]:(start[k]+length[k])])

                    if len(burst_times) > 0:
                        ax.vlines(burst_times, trial, trial+1, 'r', linestyles='dashed', linewidth=0.5)

        #ax.hlines(trial+1, 0, 1.5, color='k')
        ax.axvspan(self.t_after_stim, self.stim_duration, alpha=0.2, color='green')
        ax.set_xlim(self._bins[0], self._bins[-1])
        ax.set_ylim(0, trial+1)

        return ax

    def plot_raster_all_conditions(self, unit_ind=0, num_trials=None, offset=5, axis=None, burst=False):
        """
        Makes a raster plot for the given unit index and trial type.
        If called alone it will plot a raster to the current axis. This function
        is called by plot_all_rasters and returns an axis handle to the current
        subplot. This allows plot_all_rasters to plot rasters in the appropriate
        subplots.
        """
        # if the rates have not been calculated do that now
        if hasattr(self, 'binned_spikes') is False:
            print('Spikes have not been binned! Binning data now.')
            self.rates()

        print('Making raster for unit {}'.format(unit_ind))
        if axis == None:
            ax = plt.gca()
        else:
            ax = axis

        min_trials = np.min(self.num_good_trials)
        if num_trials != None and num_trials < min_trials:
            min_trials = num_trials

        for trial_type in range(self.stim_ids.shape[0]):
            count_mat = self.binned_spikes[trial_type][:, :, unit_ind] # returns bins x num_trials array
            shift = offset*trial_type+min_trials*trial_type

            for trial in range(min_trials):
                trial_inds = np.where(count_mat[:, trial] > 0)[0]
                spike_times = self._bins[trial_inds]
                ax.vlines(spike_times, trial+shift, trial+1+shift, color='k', linewidth=1.0)

                if burst:
                    burst_times = list()
                    data = spike_times
                    if len(data) > 3:
                        start, length, RS = ranksurprise.burst(data, limit=50e-3, RSalpha=0.1)
                        for k in range(len(start)):
                            burst_times.extend(data[start[k]:(start[k]+length[k])])

                        if len(burst_times) > 0:
                            ax.vlines(burst_times, trial+shift, trial+1+shift, 'r', linestyles='dashed', linewidth=0.5)

#                ax.hlines(trial+1, 0, 1.5, color='k')
        ax.axvspan(self.t_after_stim, self.stim_duration, alpha=0.2, color='green')
        ax.set_xlim(self._bins[0], self._bins[-1])
        ax.set_ylim(0, trial+shift+1)

        return ax

    def plot_psth(self, axis=None, unit_ind=0, trial_type=0, error='ci', color='k'):
        """
        Makes a PSTH plot for the given unit index and trial type.
        If called alone it will plot a PSTH to the current axis. This function
        is called by plot_all_PSTHs and returns an axis handle to the current
        subplot. This allows plot_all_PSTHs to plot PSTHs in the appropriate
        subplots.
        """
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

    def plot_all_rasters(self, unit_ind=0, burst=False):
        """
        Plots all rasters for a given unit with subplots.
        Each positions is a row and each manipulation is a column.
        """
        num_manipulations = int(self.stim_ids.shape[0]/self.control_pos)
        subplt_indices    = np.arange(self.control_pos*num_manipulations).reshape(self.control_pos, num_manipulations)
        fig = plt.subplots(self.control_pos, num_manipulations, figsize=(6*num_manipulations, 12))

        for manip in range(num_manipulations):
            for trial in range(self.control_pos):
                plt.subplot(self.control_pos, num_manipulations, subplt_indices[trial, manip]+1)
                self.plot_raster(unit_ind=unit_ind, trial_type=(trial + self.control_pos*manip), burst=burst)

    def plot_all_psths(self, unit_ind=0, error='sem'):
        """
        Plots all PSTHs for a given unit with subplots.
        Each positions is a row and each manipulation is a column.
        """
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
        for trial in range(self.control_pos):
            ax[trial].set_ylim(0, ymax)
        plt.show()

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



########## MAIN CODE ##########
########## MAIN CODE ##########

if __name__ == "__main__":
    sns.set_style("whitegrid", {'axes.grid' : False})

    if os.path.isdir('/Users/Greg/Documents/AdesnikLab/Data/'):
        data_dir = '/Users/Greg/Documents/AdesnikLab/Data/'
    elif os.path.isdir('/media/greg/data/neuro/hdf5/'):
        data_dir = '/media/greg/data/neuro/hdf5/'

    #manager = NeoHdf5IO(os.path.join(data_dir + 'FID1295_neo_object.h5'))
    print(sys.argv)
    try:
        fid = 'FID' + sys.argv[1]
    except:
        print('no argument provided!')

    f = h5py.File(os.path.join(data_dir + 'FID' + sys.argv[1] + '.hdf5'),'r+')

    neuro = NeuroAnalyzer(f, fid)

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





##### SCRATCH SPACE #####
##### SCRATCH SPACE #####




















