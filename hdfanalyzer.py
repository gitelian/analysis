#!/bin/bash
import os.path
import sys
import pandas as pd
import warnings
import scipy as sp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
import h5py
from scipy.signal import butter, lfilter
import statsmodels.stats.proportion

#import 3rd party code found on github
#import icsd
import ranksurprise
import dunn
# for python circular statistics
#import pycircstat as pycirc

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
        if os.path.isdir('/Users/Greg/Dropbox/A1Thesis/data/hdf5/'):
            data_dir = '/Users/Greg/Dropbox/A1Thesis/data/hdf5/'
            self.exp_csv_dir = '/Users/Greg/Dropbox/A1Thesis/data/'
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
        self.jb_behavior = int(self.f.attrs['jb_behavior'])

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
#        self.spikes_bool = 0

        # are there LFPs
        self.lfp_bool = f.attrs['lfp_bool']
        #self.lfp_bool = False

        # is there whisker tracking
        self.wt_bool = f.attrs['wt_bool']
#        self.wt_bool = False

        # find stimulus IDs
        stim_ids_temp = np.sort(np.unique([self.f[k].attrs['trial_type'] for k in f])).astype(int)
        self.stim_ids = stim_ids_temp[np.nonzero(stim_ids_temp)[0]]

        # made dictionary of whisker tracking information
        self.wt_type_dict = {0:'angle', 1:'set-point', 2:'amplitude', 3:'phase', 4:'velocity', 5:'whisking'}

        # find shortest baseline and trial length
        self.__find_min_times()

        # create run_boolean array
        self.__get_run_bool()

        # create list of trials where True represents trials with no problems
        # and False represents trials that should be skipped due to error
        self.good_trials = [True]*len(self.f)

        # create array with all the stimulus IDs
        self.__get_all_stim_ids()

        # trim run data and align it to shortest trial
        self.__trim_run()

        # trim whisker tracking data and align it to shortest trial
        # defaults to False unless it finds whisker tracking data
        self.__trim_wt()

        # classify behavior using licks and go/no-go angle position
        if self.jb_behavior and not self.spikes_bool:
            #self.reclassify_run_trials(mean_thresh=150, sigma_thresh=200, low_thresh=100)
            self.__classify_behavior()
            self.rates(psth_t_start= -1.500, psth_t_stop=2.500, kind='run_boolean')#, engaged=True, all_trials=False)

        # trim LFP data and align it to shortest trial
        if self.lfp_bool:
            self.__trim_lfp()

        # classify a trial as good if whisking occurs during a specified period.
        # a new annotation ('wsk_boolean') is added to self.trial_class
#        self.classify_whisking_trials(threshold='median')

        # return a boolean array indicating whether a trial should be analyzed
        #   also an array indicating number of good/bad/total trials

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
                self.rates(psth_t_start= -1.000, psth_t_stop=2.500, kind='run_boolean', engaged=True, all_trials=False)
            if self.jb_behavior:
                self.__classify_behavior()
                self.rates(psth_t_start= -1.500, psth_t_stop=2.500, kind='jb_engaged', engaged=True, all_trials=False)

            # reclassify units using their mean OMI (assuming ChR2 is in PV
            # cells). This is dependent on everything above!
    #        self.reclassify_units()

            # create region dictionary
            self.region_dict = {0:'M1', 1:'S1'}


            ## TODO remove this ##
#            self.rates(psth_t_start= -1, psth_t_stop=2, all_trials=True)
#            print('REMOVE HARD CODED RATES: ALL_TRIALS SET TO TRUE')

            ## TODO uncomment below this ##

            # get selectivity for all units
#            self.get_selectivity()
#
#            # get preferred position for all units
#            self.get_preferred_position()
#
#            # get best contact for each unit
#            self.get_best_contact()

            # kruskal wallis and dunn's test to ID sensory driven units
#            self.get_sensory_drive()

        if not self.jb_behavior and not self.spikes_bool:
            self.rates(psth_t_start= -1.500, psth_t_stop=2.500, all_trials=False)
            print('no JB_behavior or spikes')

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
        seg_iter = self.f.iterkeys()
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
        self.min_tafter_stim  = self.stim_duration + self.time_after
        print('using stim_duration + time_after as min_tafter_stim')
        #self.min_tafter_stim  = self.stim_duration
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

        self.stim_ids_all = np.asarray(stim_ids_all)

    def __classify_behavior(self):
        print('\n-----classify_behavior----')
        behavior_ids = list()
        licks_all    = list()
        correct_list = list() # if mouse correctly IDs GO or NOGO mark as True
        for k, seg in enumerate(self.f):
            # trial_type is 1 based! Does not start from 0, these are IDed from
            # the stimulus ID pulses sent from Jenny's equipment to Spike
            # Gadgets...but put into a sequential range (no gaps)
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

            # the trial_type values are based on a 9 pivot position for each
            # experiment (i.e. trial_type 10 is position 1 manipulation 2)
            if trial_type > 9:
                # trial_type - (9*((trial_type - 1)/9)) i.e. which manipulation
                # is it?
                trial_type = trial_type - (self.control_pos*(int(trial_type - 1)/int(self.control_pos)))

            if trial_type < 5:
                go = True

                if lick:
                    correct_list.append(True)
                else:
                    correct_list.append(False)

            elif trial_type >= 5 and trial_type < 9:
                go = False

                if not lick:
                    correct_list.append(True)
                else:
                    correct_list.append(False)

            elif trial_type == 9:
                # control position
                go = None
                correct_list.append(True)

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

        if sum(licks_all) > 10:
            # Sometimes mice stop licking towards the end of an experiment. This
            # finds that transition point defined as the point after which 30+
            # trials have no licking

            # consider replacing with
            #np.diff(np.where(asarray(neuro.licks_all) == 1)[0])
            low = np.where(np.diff(licks_all) == -1)[0] + 1

            # find when transition from not licking to licking happens, skip the first lick
            high = np.where(np.diff(licks_all) == 1)[0]

            if high[0] < low[0]:
                high = high[1::]

            if low[-1] > high[-1]:
                low = low[:-1]

            # get duration between licking trials
            down_time = high - low

            # find when mouse stopped licking
            stop_ind = np.where(down_time > 30)[0]

            # get trial number when mouse stopped licking
            if stop_ind.shape[0] != 0:
                stop_lick_trial = low[stop_ind[0]]
                print('\nMouse stopped licking on trial #{}\nexcluding all trials past it'.format(stop_lick_trial))
            else:
                stop_lick_trial = None
                print('\nMouse did not lick at all!\nIncluding all trials for analysis')
        else:
            print('\n\n#!#!#! hack used to ignore no to very low lick experiments #!#!#!\n\n')
            stop_lick_trial = None
            print('\nMouse did not lick at all!\nIncluding all trials for analysis')

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


        self.behavior_ids = np.asarray(behavior_ids)
        self.licks_all    = np.asarray(licks_all)
        self.correct_list = np.asarray(correct_list)
        #self.behavior_labels = {'hit':1, 'miss':3, 'false_alarm':2, 'correct_reject':4}
        self.behavior_labels = {1:'hit', 3:'miss', 2:'false_alarm', 4:'correct_reject'}

    def __trim_wt(self):
        """
        Trim whisker tracking arrays to the length of the shortest trial.
        Time zero of the wt time corresponds to stimulus onset.
        """
        print('\n-----__trim_wt-----')

        fps = 500.0
        if self.wt_bool:

            print('whisker tracking data found! trimming data to be all the same length in time')

            #TODO: load in the time when HSV camera starts and stops!!!
            if int(fid[3::]) < 1729: # or self.jb_behavior == 0:
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
                                wt_data = np.zeros((min_trial_length, 7, len(self.f)))
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

            else:

                cam_start_stop_times = np.zeros((len(self.f), 2))
                num_camera_triggers = list()

                # find the latest start time and earliest stop time and align
                # videos to these times
                for i, seg in enumerate(self.f):
                    anlg_path = seg + '/analog-signals/' + 'cam_times/'
                    # find number of samples in the trial
                    cam_time = self.f[anlg_path][:] - self.f[seg].attrs['stim_times'][0]
                    cam_start_stop_times[i, :] = cam_time[0], cam_time[-1]
                    num_camera_triggers.append(len(cam_time))

                # determine which is the most common number of camera triggers
                list_mode = sp.stats.mode(num_camera_triggers)[0][0]

                start_time = np.max(cam_start_stop_times[:, 0])
                stop_time  = np.min(cam_start_stop_times[:, 1])

                if start_time > 0:
                    bad_index = np.argmax(cam_start_stop_times[:, 0])
                    self.good_trials[bad_index] = False
                    start_time = max(i for i in cam_start_stop_times[:, 0] if i < 0)

                if stop_time < 0:
                    bad_index = np.argmin(cam_start_stop_times[:, 1])
                    self.good_trials[bad_index] = False
                    stop_time = min(i for i in cam_start_stop_times[:, 1] if i > 0)

                # number of samples to take from all trials
                num_samples = int(fps*stop_time) + int(fps*np.abs(start_time)) - 20


                wtt = np.linspace(start_time, stop_time, num_samples)
                wt_data = np.zeros((num_samples, 7, len(self.f)))

                for i, seg in enumerate(self.f):
                    if self.good_trials[i]:
                        anlg_path = seg + '/analog-signals/' + 'cam_times/'
                        cam_time = self.f[anlg_path][:] - self.f[seg].attrs['stim_times'][0]

                        if cam_time.shape[0] != list_mode:
                            # uh-oh! the current number of camera pulses does NOT
                            # equal the most common number of pulses (2000 most likely)
                            # TODO IGNORE BAD TRIALS (maybe with a bad_trial = True
                            # flag, or fill with NaNs and change things to nanmean)
                            warnings.warn("\n\n#!#!#!\n#!#!#!\nuh-oh! The current number of pulses {} does NOT equal the most common number of pulses {}\nTaking first samples (no way to know when time of capture)".format(cam_time.shape[0], list_mode))
                            start_index = 0
                            stop_index = num_samples
                            number_of_samples_in_trial = self.f[anlg_path].shape[0]
                        else:
                            # find index in cam_time that is closest to camera start
                            # and stop times
                            start_index = np.argmin(np.abs(cam_time - start_time))
                            stop_index = start_index + num_samples
                            number_of_samples_in_trial = self.f[anlg_path].shape[0]


                        if stop_index > number_of_samples_in_trial:
                            print('WARNING TRYING TO INDEX OUT OF ARRAY USING CRAPPY HACK TO MAKE IT WORK')
                            start_index = self.f[anlg_path].shape[0] - num_samples
                            stop_index = self.f[anlg_path].shape[0]

                        #stop_index = np.argmin(np.abs(cam_time - stop_time))

                        for k, anlg in enumerate(self.f[seg + '/analog-signals']):
                            anlg_path = seg + '/analog-signals/' + anlg
                            anlg_name = self.f[anlg_path].attrs['name']

                            if  anlg_name == 'angle':
                                wt_data[:, 0, i] = self.f[anlg_path][start_index:stop_index]
                            elif anlg_name == 'set-point':
                                wt_data[:, 1, i] = self.f[anlg_path][start_index:stop_index]
                            elif anlg_name == 'amplitude':
                                wt_data[:, 2, i] = self.f[anlg_path][start_index:stop_index]
                            elif anlg_name == 'phase':
                                wt_data[:, 3, i] = self.f[anlg_path][start_index:stop_index]
                            elif anlg_name == 'velocity':
                                wt_data[:, 4, i] = self.f[anlg_path][start_index:stop_index]
                            elif anlg_name == 'whisking':
                                wt_data[:, 5, i] = self.f[anlg_path][start_index:stop_index]
                            elif anlg_name == 'curvature':
                                wt_data[:, 6, i] = self.f[anlg_path][start_index:stop_index]


            #TODO: deal with ONLY tracking data (e.g. when I do a stimulation experiment
                 # without behavior and neuro data

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
                run_data = np.zeros((min_trial_length, len(self.f)))
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
                            lfp_data = [np.zeros((min_trial_length, x, len(self.f)), 'int16') for x in chan_per_shank]
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

    def reclassify_run_trials(self, time_before_stimulus= -1.5,\
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

            base_ind = np.logical_and( run_time > time_before_stimulus, run_time < -0.5)
            #wsk_stim_ind = np.logical_and( run_time > self.t_after_stim, run_time < (self.min_tbefore_stim + self.t_after_stim) )

            if self.jb_behavior:
                # this gets running velocity as the object moves in and during
                # the last second of the trial data (usually 1 second after the
                # stimulus leaves (assumming the trial time after stimulus stop
                # is 3sec and the time before stim stop is 1sec)

                #wsk_stim_ind = np.logical_and( run_time > (self.time_after - np.abs(time_before_stimulus)), run_time < self.time_after)

                #vel = np.concatenate( (run_speed[base_ind], run_speed[wsk_stim_ind]))
                vel = run_speed[base_ind]

                if set_all_to_true == 0:

                    if np.mean(vel) >= mean_thresh and np.std(vel) <= sigma_thresh and (sum(vel <= low_thresh)/len(vel)) <= 0.1:
                        self.trial_class['run_boolean'][count] = True
                    else:
                        self.trial_class['run_boolean'][count] = False

                elif set_all_to_true == 1:
                    self.trial_class['run_boolean'][count] = True
            else:
                wsk_stim_ind = np.logical_and( run_time > self.t_after_stim, run_time < self.stim_duration)
                vel = np.concatenate( (run_speed[base_ind], run_speed[wsk_stim_ind]))

                if set_all_to_true == 0:

                    if np.mean(vel) >= mean_thresh and np.std(vel) <= sigma_thresh and (sum(vel <= low_thresh)/len(vel)) <= 0.1:
                        self.trial_class['run_boolean'][count] = True
                    else:
                        self.trial_class['run_boolean'][count] = False

                elif set_all_to_true == 1:
                    self.trial_class['run_boolean'][count] = True

    def set_engaged_disengaged_trials(self, e_ind=None, d_ind=None, reset=False):
        """
        manually specify engaged vs disengaged trials

        e_ind: an nx2 array where each row represents indices that should be
            marked as "engaged" (i.e. True)

        d_ind: an nx2 array where each row represents indices that should be
            marked as "dis-engaged" (i.e. False)

        reset will revert to the original Engaged/Disengaged indication...where
            trials prior to self.last_engaged_trial are engaged (True) and trials
            after are disengaged (False)
        """

        num_trials = len(self.f)

        if reset:
            e_list = [True if x < self.last_engaged_trial else False for x in range(num_trials)]
            self.trial_class['jb_engaged'] = e_list
            del self.engaged_trials
            del self.disengaged_trials

        else:

            if e_ind is not None and d_ind is not None:
                e_list = [None]*num_trials
                e_ind = np.asarray(e_ind)
                d_ind = np.asarray(d_ind)

                self.engaged_trials = e_ind
                self.disengaged_trials = d_ind

                for x in range(num_trials):

                    # is trial in an "engaged" range
                    r1 = e_ind[:, 0] <= x
                    r2 = e_ind[:, 1] >= x
                    e_bool = sum(np.logical_and(r1, r2)) > 0

                    # is trial in an "dis-engaged" range
                    r1 = d_ind[:, 0] <= x
                    r2 = d_ind[:, 1] >= x
                    d_bool = sum(np.logical_and(r1, r2)) > 0

                    if e_bool and not d_bool:
                        # accept and indicate trial index as engaged
                        e_list[x] = True
                    elif not e_bool and d_bool:
                        # accept and indicate trial as dis-engaged
                        e_list[x] = False
                    elif e_bool and d_bool:
                        # mark trial to be skipped
                        print('trial index is between both the ENGAGED and DIS-ENGAGED range ...something went wrong setting to None, trial will be skipped')
                        e_list[x] = None

                self.trial_class['jb_engaged'] = e_list

            else:
                print('e_ind or d_ind not specified, both  must be provided, not doing anything')

    def get_trials2analyze(self, kind='run_boolean', engaged=None):
        """
        Return a boolean array indicating whether a trial should be analyzed.
        Creates lists with the number of good/bad/total trials for each
        stimulus condition.

        kind can be set to either 'run_boolean' (default), 'wsk_boolean',
        or 'jb_engaged'

        NOTE: THISE DOES NOT REDO THE ANALYSIS WITH THE NEW SPECIFICATIONS. TO
            HAVE THE NEW CRITERIA APPLIED RUN RATES() WITH THE DESIRED SPECS

        20200226 G.T. added engaged flag.
        If jb_behavior is True then use engaged flag to filter trials

        Note: can currently ID running vs not running trials during no behavior
        and during behavior can ID (running AND engaged) or (running AND disengaged)
        as good trials.

        output: will return a list indicating whether a trial is good or bad
        """

        print('\n-----get_trials2analyze-----\nkind = {}, engaged = {}'.format(kind, engaged))

        num_good_trials = list()
        num_bad_trials = list()
        num_all_trials  = list()

        trials2analyze = [False]*len(self.f)

        if engaged is None:
            use_engaged_value = False

        else:

            if self.jb_behavior:
                use_engaged_value = True
                print('using engaged value')

            else:
                use_engaged_value = False
                print('can not use engaged value, no behavior data found\nusing RUNNING to indicate trials2analyze')


        for stim_id in self.stim_ids:

            good_trials  = 0
            bad_trials = 0
            all_trials  = 0

            for k, seg in enumerate(self.f):

                if self.good_trials[k]:

                    # use only kind (e.g. running) to indicate good trials
                    if not use_engaged_value:

                        if self.stim_ids_all[k] == stim_id and self.trial_class[kind][k] == True:
                            trials2analyze[k] = True
                            good_trials += 1

                        elif self.stim_ids_all[k] == stim_id and self.trial_class[kind][k] == False:
                            trials2analyze[k] = False
                            bad_trials += 1

                        if self.stim_ids_all[k] == stim_id:
                            all_trials += 1

                    # if this is a behavior session AND engaged value was set
                    # then indicate good trials based on what engaged is set to
                    elif use_engaged_value:

                        # e.g. running and disengaged (run == True and jb_engaged == False)
                        if self.stim_ids_all[k] == stim_id and self.trial_class[kind][k] == True\
                                and self.trial_class['jb_engaged'][k] == engaged:
                            trials2analyze[k] = True
                            good_trials += 1

                        elif self.stim_ids_all[k] == stim_id and self.trial_class[kind][k] == False\
                                and self.trial_class['jb_engaged'][k] == engaged:
                            trials2analyze[k] = False
                            bad_trials += 1

                        if self.stim_ids_all[k] == stim_id:
                            all_trials += 1

            num_good_trials.append(good_trials)
            num_bad_trials.append(bad_trials)
            num_all_trials.append(all_trials)

        self.num_good_trials = num_good_trials
        self.num_bad_trials  = num_bad_trials
        self.num_all_trials  = num_all_trials

        return trials2analyze

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

    def rates(self, psth_t_start= -0.500, psth_t_stop=2.000, kind='run_boolean',\
            engaged=True, all_trials=False, t_window=None):
        """
        rates computes the absolute and evoked firing rate and counts for the
        specified stimulus period. The time to start analyzing after the stimulus
        start time is specified by neuro.t_after_stim. The baseline period to
        be analyzed is taken to be the same size as the length of the stimulus
        period.

        kind can be set to either 'wsk_boolean' or 'run_boolean' (default) or 'jb_engaged'

        For behavior you can have running AND engaged or running AND not engaged

        t_window: dictionary, with keys 'start_time', 'stop_time', 'base_start',
            'base_stop'. Use to have rates look at a different analysis window

        """

        print('\n-----computing rates----')
        absolute_rate   = list()
        evoked_rate     = list()
        absolute_counts = list()
        evoked_counts   = list()
        binned_spikes   = list()
        psth            = list()
        running         = list()
        licks           = list()
        bids            = list() # behavior IDs
        lick_bool       = list()
        trial_choice    = list()

        # make whisker tracking list wt
        if self.wt_bool:
            wt = list()

        if kind == 'wsk_boolean' and not self.wt_bool:
            warnings.warn('**** NO WHISKER TRACKING DATA AVAILABLE ****\n\
                    using run speed to select good trials')
            kind = 'run_boolean'
            trials2analyze = self.get_trials2analyze(kind='run_boolean')

        elif kind == 'wsk_boolean' and self.wt_bool:
            print('using whisking to find good trials')
            trials2analyze = self.get_trials2analyze(kind='wsk_boolean')

        else:
            trials2analyze = self.get_trials2analyze(kind=kind, engaged=engaged)

        num_trials = self.num_good_trials

        if all_trials == True:
            self.reclassify_run_trials(set_all_to_true=True)
            trials2analyze = self.get_trials2analyze(kind='run_boolean')
            num_trials = self.num_all_trials

        # make bins for rasters and PSTHs
#        bins = np.arange(-self.min_tbefore_stim, self.min_tafter_stim, 0.001)
        bins = np.arange(psth_t_start, psth_t_stop, 0.001)
        kernel = self.__make_kernel(kind='square', resolution=0.100)
#        kernel = self.__make_kernel(kind='alpha', resolution=0.025)
        self._bins = bins
        self.bins_t = bins[0:-1]


        # preallocation loop
        for k, trials_ran in enumerate(num_trials):
                running.append(np.zeros((self.run_t.shape[0], trials_ran)))

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
                    trial_choice.append([list() for x in range(trials_ran)])


        ### main loops (stimulus type (outer), all trials (inner))
        ### main loops (stimulus type (outer), all trials (inner))

        for stim_ind, stim_id in enumerate(self.stim_ids):
            good_trial_ind = 0

            # iterate through all segments in HDF5 file
            for k, seg in enumerate(self.f):

                # if running or whisking or jb_engaged trial add data to arrays
                #if  self.stim_ids_all[k] == stim_id and (self.trial_class[kind][k] == engaged or \
                #        all_trials == True):
                if self.stim_ids_all[k] == stim_id and trials2analyze[k]:

                    # add run data to list
                    running[stim_ind][:, good_trial_ind] = self.run_data[:, k]

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
                        trial_choice[stim_ind][good_trial_ind] = self.correct_list[k]

                    if self.spikes_bool:
                        # get baseline and stimulus period times for this trial
                        # this uses ABSOLUTE TIME not relative time
                        obj_stop = self.f[seg].attrs['stim_times'][0]

                        if t_window == None:
                            stim_start = obj_stop + self.t_after_stim
                            stim_stop= obj_stop + self.stim_duration
                            base_start = obj_stop - np.abs((stim_stop - stim_start))
                            base_stop  = obj_stop
                        else:
                            stim_start = obj_stop + t_window['start_time']
                            stim_stop  = obj_stop + t_window['stop_time']
                            base_start = obj_stop + t_window['base_start']
                            base_stop  = obj_stop + t_window['base_stop']

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

        self.run           = running

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
            self.trial_choice = trial_choice

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
                for k, trials_not_ran in enumerate(self.num_bad_trials):
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
            trial2analyze = self.get_trials2analyze(kind='wsk_boolean')
        elif kind == 'run_boolean':
            print('using running to find good trials')
            trial2analyze = self.get_trials2analyze(kind='run_boolean')

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
#                if self.stim_ids_all[k] == stim_id and self.trial_class[kind][k] == True:
                if trial2analyze[k]:

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
###### Eight-Position e-phys experiment specific functions #####
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

    def get_psychometric_curve(self, axis=None):
        """
        compute psychometric curve for entire behavioral experiment

        learned that confidence intervals for binary data can be computed using
        the binomial distribution!
        """
        if self.jb_behavior:
            num_cond = len(self.stim_ids)
            prob_lick = np.zeros((num_cond, ))
            lick_error = np.zeros((2, num_cond ))

            for cond in range(num_cond ):
                licks = float(np.sum(self.lick_bool[cond]))
                nobs  = self.lick_bool[cond].shape[0]
                if nobs != 0:
                    prob_lick[cond] = licks/nobs
                else:
                    prob_lick[cond] = 0
                # returns the ACTUAL values of the 95% CI instead of the
                # DIFFERENCE from the mean. This is why I have to subtract off
                # the mean to get errorbar to plot errors correctly!
                #err = statsmodels.stats.proportion.proportion_confint(licks, nobs, method='jeffreys')
                err = statsmodels.stats.proportion.proportion_confint(licks, nobs, method='wilson')
                lick_error[0, cond] = err[0] # lower 95% CI
                lick_error[1, cond] = err[1] # upper 95% CI
                print(cond, prob_lick[cond], np.round(lick_error[:, cond], decimals=3))

            # subtract off mean because errorbar will add/subtract these value
            # from the mean INSTEAD of plotting the error bar from [a, b]
            lick_error[0, :] = np.abs(prob_lick - lick_error[0, :])
            lick_error[1, :] = np.abs(prob_lick - lick_error[1, :])

            pos = range(1, self.control_pos)
            line_color = ['k','r','b']
            if axis is None:
                fig, ax = plt.subplots()
            else:
                ax = axis

            self.plot_mean_err(prob_lick, lick_error, axis=ax)
            ax.set_title(self.fid + ' psychometric curve')
            ax.set_ylim(0, 1.1)
            ax.set_ylabel('P(lick)')
            ax.set_xlabel('<--GO -- NOGO-->\npositions')

            return prob_lick, lick_error

    def plot_lick_rate(self, t_start=0, t_stop=1, axis=None):
        """
        compute lick rate for each condition
        """
        if self.jb_behavior:
            num_cond = len(self.stim_ids)
            lick_rate = [list() for x in range(num_cond)]

            # check if axis was provided
            if axis == None:
                fig, ax = plt.subplots()
            else:
                ax = axis


            for cond in range(num_cond):

                for licks in self.licks[cond]:

                    lick_rate_temp = np.sum(np.logical_and(licks > t_start, licks < t_stop))
                    if lick_rate_temp == 0 or np.isnan(lick_rate_temp):
                        lick_rate[cond].append(0)
                    elif  lick_rate_temp > 0:
                        lick_rate[cond].append(lick_rate_temp)

            mean_lick_rate = [np.mean(x) for x in lick_rate]
            sem_lick_rate = [sp.stats.sem(x) for x in lick_rate]

            ax = self.plot_mean_err(mean_lick_rate, sem_lick_rate, axis=ax)
            ax.set_title(self.fid + ' lick rate')
            ax.set_ylabel('Licks / sec')
            ax.set_xlabel('<--GO -- NOGO-->\npositions')

            return mean_lick_rate, sem_lick_rate

    def plot_lick_raster(self):

        if self.jb_behavior:
            num_manipulations = len(self.stim_ids)/self.control_pos # no light, light 1 region, light 2 regions
            line_color = ['k','r','b']
            fig, ax = plt.subplots(num_manipulations, self.control_pos, figsize=(12, 6), sharex=True, sharey=True)
            for manip in range(num_manipulations):
                for cond in range(self.control_pos):
                    trial = 0
                    for licks in self.licks[cond + (manip*self.control_pos)]:
                        if not np.isnan(np.sum(licks)):
                            ax[manip][cond].vlines(licks, trial, trial+1, color=line_color[manip], linewidth=1.0)
                            trial += 1

            max_ylim = ax[manip][cond].get_ylim()[1]
            for manip in range(num_manipulations):
                for cond in range(self.control_pos):
                    ax[manip][cond].axvspan(0, 1, alpha=0.3, color='green')
                    ax[manip][cond].set_ylim(0, max_ylim)
                    ax[manip][cond].set_title('n = {}'.format(len(self.licks[cond + (manip * self.control_pos)])))

            fig.suptitle(fid + ' lick raster')

    def plot_time2lick(self, t_start=0, axis=None):
        """
        compute time (mean +/- sem) to first lick for all angles
        """

        # check if axis was provided
        if axis == None:
            fig, ax = plt.subplots()
        else:
            ax = axis

        if self.jb_behavior:
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

            self.plot_mean_err(time2lick_mean, time2lick_sem, axis=ax)
            ax.set_title('time to lick')
            ax.set_ylabel('time (s)')
            ax.set_xlabel('<--GO -- NOGO-->\npositions')

            return time2lick_mean, time2lick_sem

    def performance_vs_time(self, bin=25, axis=None, split=False):
        """compute performance vs time from 0-100%"""

        # check if axis was provided
        if axis == None:
            fig, ax = plt.subplots()
        else:
            ax = axis

        win = np.ones(bin)/float(bin)
        if not split:
            ## overall performance
            pc = self.correct_list
            performance = np.convolve(self.correct_list, win, 'valid')

            # plot
            ax.plot(performance, 'tab:grey')
            ax.hlines(0.5, 0, pc.shape[0], linestyles='dashed')
            ax.set_ylim(0, 1.05); ax.set_ylabel('Mouse performance')
            ax.set_xlim(0, pc.shape[0]); ax.set_xlabel('Trial number')


        elif split:
            pc = self.correct_list
            ## nolight vs light performance
            # no light
            nol_inds = np.where(self.stim_ids_all <= self.control_pos)[0]
            nol_choice = pc[nol_inds]
            nol_performance = np.convolve(nol_choice, win, 'same')

            # light
            light_inds = np.where(self.stim_ids_all > self.control_pos)[0]
            light_choice = pc[light_inds]
            light_performance = np.convolve(light_choice, win, 'same')

            # plot
            ax.plot(nol_inds, nol_performance, 'dimgrey')
            ax.plot(light_inds, light_performance, 'tab:red')
            ax.hlines(0.5, 0, pc.shape[0], linestyles='dashed')
            ax.set_ylim(0, 1.05); ax.set_ylabel('Mouse performance')
            ax.set_xlim(0, pc.shape[0]); ax.set_xlabel('Trial number')
            ax.legend(['light off', 'light on'], loc='lower right')
            ax.set_title('Performance across experiment')

            performance = [nol_performance, light_performance]

        if hasattr(self, 'engaged_trials') is True:
            print('engaged trials found!')
            # python iterates row-wise
            for row in self.engaged_trials:
                print(row)
                ax.hlines(1.01, row[0], row[1], colors='tab:green', linewidth=2)
            for row in self.disengaged_trials:
                ax.hlines(1.01, row[0], row[1], colors='tab:red', linewidth=2)


        return performance

###########################
#### whisker analysis for jb_behavior experiments ####
###########################

### ONLY ANALYZE TRIALS WHERE THE MOUSE GOT IT CORRECT ??? ###


    def plot_mean_whisker(self, t_window=[-0.5, 0.5], kind='setpoint',\
            cond2plot=[0, 1, 2], correct=None, delta=False, axis=None):
        """
        correct variable: None, all trials filtered by "rates" will be used
            True, only correct trials
            False, only incorrect trials (i.e. when mouse made a mistake
        """

        ### is set-point different ???
        # compute mean set-point for each correct condition and plot
        ### is the slope of the set-point after contact different?
        start_ind = np.argmin(np.abs(self.wtt - t_window[0]))
        stop_ind  = np.argmin(np.abs(self.wtt - t_window[1]))
        mean_kin, sem_kin, _ = self.get_wt_kinematic(t_window=t_window, kind=kind, correct=correct)

        # plot
        num_manipulations = len(self.stim_ids)/self.control_pos # no light, light 1 region, light 2 regions
        line_color = ['k','r','b']

        # if an axis is given AND only one condition is given it will plot the
        # data in the given axis
        if axis is None and len(cond2plot) > 1:
            fig, ax = plt.subplots(1, len(cond2plot), sharey=True)
        elif axis is None and len(cond2plot) == 1:
            print('\nVERY HACKY WAY OF PLOTTING TO ONE AXIS')
            fig, ax = plt.subplots(1, len(cond2plot), sharey=True)
            ax = [ax, None]
        elif axis is not None and len(cond2plot) == 1:
            print('\nVERY HACKY WAY OF PLOTTING TO ONE AXIS')
            ax = [axis, None]
        else:
            raise Exception('To plot to a given axis only ONE condition must be selected!')

        for k, cond in enumerate(cond2plot):

            trial_type = self.stim_ids[cond]
            if trial_type < 5:
                ax[k].set_title('GO (position {})'.format(cond))
            if trial_type >= 5 and cond < 9:
                ax[k].set_title('NOGO (position {})'.format(cond))
            if trial_type == 9:
                ax[k].set_title('Catch (position control)')

            ax[k].set_xlabel('time (s)')
            ax[k].set_ylabel('{} (deg)'.format(kind))

            if not delta:
                for manip in range(num_manipulations):
                    if not np.where(np.isnan(mean_kin[cond + (self.control_pos*manip)]) == True)[0].shape[0] >= 1:
                        x = self.wtt[start_ind:stop_ind]
                        mean_vals = mean_kin[cond + (self.control_pos*manip)][start_ind:stop_ind]
                        err_vals = sem_kin[cond + (self.control_pos*manip)][start_ind:stop_ind]

                        self.plot_cont_mean_err(x, mean_vals, err_vals, axis=ax[k], line_color=line_color[manip])

            elif delta:
                for manip in range(num_manipulations - 1):
                    if not np.where(np.isnan(mean_kin[cond + (self.control_pos*manip)]) == True)[0].shape[0] >= 1:
                        x = self.wtt[start_ind:stop_ind]
                        mean_delta = mean_kin[cond + (self.control_pos*(manip + 1))][start_ind:stop_ind] - mean_kin[cond][start_ind:stop_ind]

                        sem_manip = sem_kin[cond + (self.control_pos*(manip + 1))][start_ind:stop_ind]
                        sem_nolight = sem_kin[cond][start_ind:stop_ind]
                        sem_delta = np.sqrt(sem_manip**2 + sem_nolight**2)

                        self.plot_cont_mean_err(x, mean_delta, err_delta, axis=ax[k], line_color=line_color[manip + 1])

                ax[k].plot(t_window, [0,0],'--k')

    def plot_mean_runspeed(self, t_window=[-0.5, 0.5], cond2plot=[0, 1, 2], correct=None, delta=False, axis=None):
        """
        plot the mean and sem runspeed for given trials

        correct variable: None, all trials filtered by "rates" will be used
            True, only correct trials
            False, only incorrect trials (i.e. when mouse made a mistake

        """

        # get window indices
        start_ind = np.argmin(np.abs(self.run_t - t_window[0]))
        stop_ind  = np.argmin(np.abs(self.run_t - t_window[1]))

        # get all the setpoints
        run = [list() for x in range(len(self.stim_ids))]
        mean_run = [list() for x in range(len(self.stim_ids))]
        sem_run = [list() for x in range(len(self.stim_ids))]

        for cond in range(len(self.stim_ids)):

            for trial in range(len(self.lick_bool[cond])):

                # if mouse made correct choice
                if  correct and self.trial_choice[cond][trial]:
                    # get set-point
                    run[cond].append(self.run[cond][:, trial])
                # if mouse made wrong choice
                elif not correct and not self.trial_choice[cond][trial]:
                    run[cond].append(self.run[cond][:, trial])
                # all trials correct and incorrect
                elif correct == None:
                    run[cond].append(self.run[cond][:, trial])

        # convert to arrays
        for index in range(len(run)):
            run[index] = np.asarray(run[index])
            mean_run[index]   = np.mean(run[index], axis=0)
            sem_run[index]    = sp.stats.sem(run[index], axis=0)
            print('cond {} has {} trials'.format(index, len(run[index])))

        # plot
        num_manipulations = len(self.stim_ids)/self.control_pos # no light, light 1 region, light 2 regions
        line_color = ['k','r','b']

        # if an axis is given AND only one condition is given it will plot the
        # data in the given axis
        if axis is None and len(cond2plot) > 1:
            fig, ax = plt.subplots(1, len(cond2plot), sharey=True)
        elif axis is None and len(cond2plot) == 1:
            print('\nVERY HACKY WAY OF PLOTTING TO ONE AXIS')
            fig, ax = plt.subplots(1, len(cond2plot), sharey=True)
            ax = [ax, None]
        elif axis is not None and len(cond2plot) == 1:
            print('\nVERY HACKY WAY OF PLOTTING TO ONE AXIS')
            ax = [axis, None]
        else:
            raise Exception('To plot to a given axis only ONE condition must be selected!')


        for k, cond in enumerate(cond2plot):

            if cond < 4:
                ax[k].set_title('GO (position {})'.format(cond))
            if cond >= 4 and cond < 8:
                ax[k].set_title('NOGO (position {})'.format(cond))
            if cond == 8:
                ax[k].set_title('Catch (position control)')

            ax[k].set_xlabel('time (s)')
            ax[k].set_ylabel('runspeed (cm/sec)')

            if not delta:
                for manip in range(num_manipulations):
                    if not np.where(np.isnan(mean_run[cond + (self.control_pos*manip)]) == True)[0].shape[0] >= 1:
                        x = self.run_t[start_ind:stop_ind]
                        mean_vals = mean_run[cond + (self.control_pos*manip)][start_ind:stop_ind]
                        err_vals = sem_run[cond + (self.control_pos*manip)][start_ind:stop_ind]

                        self.plot_cont_mean_err(x, mean_vals, err_vals, axis=ax[k], line_color=line_color[manip])

            elif delta:
                for manip in range(num_manipulations - 1):
                    if not np.where(np.isnan(mean_run[cond + (self.control_pos*manip)]) == True)[0].shape[0] >= 1:
                        x = self.wtt[start_ind:stop_ind]
                        mean_delta = mean_run[cond + (self.control_pos*(manip))][start_ind:stop_ind] - mean_run[cond][start_ind:stop_ind]

                        sem_manip = sem_run[cond + (self.control_pos*(manip))][start_ind:stop_ind]
                        sem_nolight = sem_run[cond][start_ind:stop_ind]
                        sem_delta = np.sqrt(sem_manip**2 + sem_nolight**2)

                        self.plot_cont_mean_err(x, mean_delta, err_delta, axis=ax[k], line_color=line_color[manip + 1])

                ax[k].plot(t_window, [0,0],'--k')

    def get_wt_kinematic(self, kind='setpoint', t_window=[-0.5, 0.5], cond=[0, 1, 2], correct=True):
        """
        get the mean and sem of either setpoint or amplitude

        correct variable: None, all trials filtered by "rates" will be used
            True, only correct trials
            False, only incorrect trials (i.e. when mouse made a mistake

        Returns: mean, sem, and num_trials per condition
        """

        if kind == 'setpoint':
            whisk_ind = 1
        elif kind == 'amplitude':
            whisk_ind = 2
        else:
            print('TYPO "kind" probably misspelled')


        # get window indices
        start_ind = np.argmin(np.abs(self.wtt - t_window[0]))
        stop_ind  = np.argmin(np.abs(self.wtt - t_window[1]))

        # make lists for data allocation
        whisk_kinematic = [list() for x in range(len(self.stim_ids))]
        mean_kin   = [list() for x in range(len(self.stim_ids))]
        sem_kin    = [list() for x in range(len(self.stim_ids))]
        num_trials = np.zeros((len(self.stim_ids), 1))

        for cond in range(len(self.stim_ids)):

            for trial in range(len(self.lick_bool[cond])):

                # gets trials, filtered by 'rates' and further filters by
                # correct, incorrect, or all trials
                # if mouse made correct choice
                if  correct and self.trial_choice[cond][trial]:
                    # get whisk kinematic values
                    whisk_kinematic[cond].append(self.wt[cond][:, whisk_ind, trial])
                # if mouse made wrong choice
                elif not correct and not self.trial_choice[cond][trial]:
                    # get whisk kinematic values
                    whisk_kinematic[cond].append(self.wt[cond][:, whisk_ind, trial])
                # all trials correct and incorrect
                elif correct == None:
                    # get whisk kinematic values
                    whisk_kinematic[cond].append(self.wt[cond][:, whisk_ind, trial])

        # convert to arrays
        for index in range(len(whisk_kinematic)):
            whisk_kinematic[index] = np.asarray(whisk_kinematic[index])
            mean_kin[index]   = np.mean(whisk_kinematic[index], axis=0)
            sem_kin[index]    = sp.stats.sem(whisk_kinematic[index], axis=0)
            print('cond {} has {} trials'.format(index, len(whisk_kinematic[index])))

        return mean_kin, sem_kin, num_trials

    def plot_wt_freq(self, t_window=[-0.5, 0.5], cond2plot=[0, 1, 2], all_correct_trials=False):

        print('\n #!#! CAUTION this uses nanmean/nansem. NOT sure why there are NANs #!#!')
        ### is whisk frequency different ???

        # get window indices
        start_ind = np.argmin(np.abs(self.wtt - t_window[0]))
        stop_ind  = np.argmin(np.abs(self.wtt - t_window[1]))

        # get all the angles
        wt_angle      = [list() for x in range(len(self.stim_ids))]
        mean_psd = [list() for x in range(len(self.stim_ids))]
        sem_psd  = [list() for x in range(len(self.stim_ids))]

        for cond in range(len(self.stim_ids)):

            for trial in range(len(self.lick_bool[cond])):

                if not all_correct_trials:
                    wt_angle[cond].append(self.wt[cond][start_ind:stop_ind, 0, trial])

                elif self.trial_choice[cond][trial]:
                    # get angle
                    wt_angle[cond].append(self.wt[cond][start_ind:stop_ind, 0, trial])

        # convert to arrays
        for index in range(len(wt_angle)):
            wt_angle[index] = np.asarray(wt_angle[index])
            print(wt_angle[index].shape)

            # compute PSD
            angle_temp = wt_angle[index]
            if angle_temp.shape[0] == 0:
                mean_psd[index] = np.nan
                sem_psd[index] = np.nan
            else:

                f, frq_mat_temp = self.get_psd(angle_temp.T, 500)
                #mean_psd[index] = np.mean(frq_mat_temp, axis=1)
                mean_psd[index] = np.nanmean(frq_mat_temp, axis=1)
                sem_psd[index]  = sp.stats.sem(frq_mat_temp, axis=1, nan_policy='omit')

        # plot
        f = np.linspace(0, 250, mean_psd[0].shape[0])
        num_manipulations = len(self.stim_ids)/self.control_pos # no light, light 1 region, light 2 regions
        line_color = ['k','r','b']
        fig, ax = plt.subplots(1, len(cond2plot), sharey=True)

        for k, cond in enumerate(cond2plot):

            if cond < 4:
                ax[k].set_title('GO (position {})'.format(cond))
            if cond >= 4 and cond < 8:
                ax[k].set_title('NOGO (position {})'.format(cond))
            if cond == 8:
                ax[k].set_title('Catch (position control)')

            ax[k].set_xlim(0, 30)
            ax[k].set_xlabel('frequency (Hz)')
            ax[k].set_ylabel('PSD power')

            for manip in range(num_manipulations):
                if not type(mean_psd[cond + (self.control_pos*manip)]) == float:
                    ax[k].plot(f, mean_psd[cond + (self.control_pos*manip)], color=line_color[manip])
                    ax[k].fill_between(f, mean_psd[cond + (self.control_pos*manip)] + sem_psd[cond + (self.control_pos*manip)],\
                            mean_psd[cond + (self.control_pos*manip)] - sem_psd[cond + (self.control_pos*manip)], facecolor=line_color[manip], alpha=0.3)
                    ax[k].set_yscale("log", nonposy='clip')

            fig.suptitle(fid + ' whisker angle frequency with t_window = {}'.format(t_window))

        return fig, ax


    def plot_wt_flip_book(self, t_window=[-1, 0], kind='angle'):

        ### TODO edit this to only plot trials where the mouse was correct! ###
        ### TODO make sure trials where animal was ENGAGED are being plotted ! ###
        start_ind = np.argmin(np.abs(self.wtt - t_window[0]))
        stop_ind  = np.argmin(np.abs(self.wtt - t_window[1]))
        line_color = ['k','r','b']

        if kind == 'angle':
            kind_ind = 0
        elif kind == 'setpoint':
            kind_ind = 1
        elif kind == 'amplitude':
            kind_ind = 2

        min_trials2plot = np.min(np.reshape(self.num_good_trials, [2,9]), axis=0)
        num_manipulations = len(self.stim_ids)/self.control_pos # no light, light 1 region, light 2 regions

        with PdfPages(os.path.expanduser('~/Desktop/' + fid + '_wt_{}.pdf'.format(kind))) as pdf:
            for manip in range(num_manipulations):
                for cond in range(self.control_pos-1):
                    # get number of correct and incorrect trials for this condition
                    # PLUS manipulation

                    num_correct   = sum(self.trial_choice[cond + (self.control_pos*manip)])
                    correct_inds = np.where(np.asarray(self.trial_choice[cond + (self.control_pos*manip)]) == True)[0]

                    num_incorrect = len(self.trial_choice[cond + (self.control_pos*manip)]) - num_correct
                    incorrect_inds = np.where(np.asarray(self.trial_choice[cond + (self.control_pos*manip)]) == False)[0]

                    min_trials2plot = min([num_correct, num_incorrect])

                    #for k in range(min_trials2plot[cond]):
                    # creates a new page
                    for k in range(min_trials2plot):
                        fig, ax = plt.subplots(1, 2)
                        fig.suptitle('Pos {}'.format(cond))

                        # plot correct trials
                        ax[0].plot(self.wtt[start_ind:stop_ind],\
                                self.wt[cond + (self.control_pos*manip)][start_ind:stop_ind, kind_ind, correct_inds[k]],\
                                color=line_color[manip])
                        # plot incorrect trials
                        ax[1].plot(self.wtt[start_ind:stop_ind],\
                                self.wt[cond + (self.control_pos*manip)][start_ind:stop_ind, kind_ind, incorrect_inds[k]],\
                                color=line_color[manip])

                        ax[0].set_xlabel('time (s)')
                        ax[0].set_ylabel('set-point (deg)')
                        ax[0].set_ylim(80, 160)
                        ax[0].set_title('Correct choice')
                        ax[1].set_xlabel('time (s)')
                        ax[1].set_ylabel('set-point (deg)')
                        ax[1].set_ylim(80, 160)
                        ax[1].set_title('Incorrect choice')

                        pdf.savefig()
                        fig.clear()
                        plt.close()


    def find_change_points(self, data, window_length=401, polyorder=2, deriv=2):
        """
        use savgol_filter to smooth and calculate derivative of input signal

        data: n x l array (samples x trials)

        returns: delta_inds (int)  l x 2 onset index, offset index
                delta(float array): change from onset to offset

        """

        l_trials = data.shape[1]
        delta_inds  = np.zeros((l_trials, 2), dtype=int)
        delta = np.zeros((l_trials, ))
        for k in range(l_trials):
            sg = sp.signal.savgol_filter(data[:, k], window_length=window_length,\
                    polyorder=polyorder, deriv=deriv)
            onset  = np.argmin(sg)
            offset = np.argmax(sg)
            delta[k]  = data[offset, k] - data[onset, k]

            delta_inds[k, :] = np.asarray((onset, offset))

        return delta_inds, delta


    def get_trial_delta(self, plot=True):
        """
        Computes onset, offset, and amount change for setpoint

        Returns: onset and offset time? Does not compute change in degrees?

        TODO: Get change point working
              compute change in set-point
              add runspeed capability

        """
        cond2compute = range(len(self.stim_ids))
        deltas = [list() for x in cond2compute]

        for k, cond in enumerate(cond2compute):
            data = self.wt[cond][:, 1, :]
            delta_inds, delta = find_change_points(self, data, window_length=401, polyorder=2, deriv=2)
            deltas[k] = np.asarray((self.wtt[delta_inds[:,0]], self.wtt[delta_inds[:, 1]], delta)).T

        mean_onset = [np.mean(x[:, 0]) for x in deltas]
        sem_onset  = [sp.stats.sem(x[:, 0]) for x in deltas]



        mean_offset = [np.mean(x[:, 1]) for x in deltas]
        sem_offset  = [sp.stats.sem(x[:, 1]) for x in deltas]

        num_manipulations = len(self.stim_ids)/self.control_pos # no light, light 1 region, light 2 regions
        bins = np.arange(-1.5, 1.5, 0.05)
        all_onset  = [np.zeros((len(bins)-1, ), dtype=int) for x in range(num_manipulations)]
        all_offset = [np.zeros((len(bins)-1, ), dtype=int) for x in range(num_manipulations)]

        for manip in range(num_manipulations):
            for cond in range(self.control_pos):
                cond_ind = cond + (manip *self.control_pos)
                # onset
                counts_onset = np.histogram(deltas[cond_ind][:, 0], bins=bins)[0]
                all_onset[manip-1] += counts_onset

                # offset
                counts_offset = np.histogram(deltas[cond_ind][:, 1], bins=bins)[0]
                all_offset[manip-1] += counts_offset

        if plot:
            fig, ax = plt.subplots(1, 4)
            self.plot_mean_err(mean_onset, sem_onset, axis=ax[0])
            self.plot_mean_err(mean_offset, sem_offset, axis=ax[1])


            ax[2].bar(bins[:-1], all_onset[0]/float(sum(all_onset[0])), width=0.05, color='tab:blue', alpha=0.35)
            ax[2].bar(bins[:-1], all_onset[1]/float(sum(all_onset[1])), width=0.05, color='tab:red', alpha=0.35)

            ax[3].bar(bins[:-1], all_offset[0]/float(sum(all_offset[0])), width=0.05, color='tab:blue', alpha=0.35)
            ax[3].bar(bins[:-1], all_offset[1]/float(sum(all_offset[1])), width=0.05, color='tab:red', alpha=0.35)

            del ax

        means_and_errs = [(mean_onset, sem_onset), (mean_offset, sem_offset) ]

        return deltas, means_and_errs





###############################################################################
##### Plotting Functions #####
###############################################################################

    def plot_tuning_curve(self, unit_ind=None, kind='abs_count', axis=None, labels=None, xlabel=None):
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

        if labels is None:
            labels = [str(i) for i in pos]; labels.append('NC')

        if xlabel is None:
            xlabel = 'Bar Position'
        #line_color = ['k','r','b']

        num_manipulations = len(self.stim_ids)/control_pos # no light, light 1 region, light 2 regions

        if num_manipulations <= 3:
            # use my default color choices
            line_color = ['k','r','b']
        else:
            line_color = list()
            color_idx = np.linspace(0, 1, num_manipulations)
            for manip in range(num_manipulations):
                line_color.append(plt.cm.Blues(color_idx[manip]))

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
                        color=line_color[control_pos_count], marker='o', markersize=6.0, linewidth=2)
                # plot control position separately from stimulus positions
                ax.errorbar(control_pos, meanr[(control_pos_count+1)*control_pos-1], yerr=stder[(control_pos_count+1)*control_pos-1],\
                        color=line_color[control_pos_count], marker='o', markersize=6.0, linewidth=2)

            ax.set_xlabel(xlabel, fontsize=8)
            ax.plot([0, control_pos+1],[0,0],'--k')

        # if all tuning curves from all units are to be plotted
        else:
            # determine number of rows and columns in the subplot
            unit_ind = range(self.num_units)
            num_rows, num_cols = 3, 3

            #num_manipulations = len(self.stim_ids)/control_pos # no light, light 1 region, light 2 regions

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
                            color=line_color[control_pos_count], marker='o', markersize=6.0, linewidth=2)
                    # plot control position separately from stimulus positions
                    ax[row][col].errorbar(control_pos, meanr[(control_pos_count+1)*control_pos-1], yerr=stder[(control_pos_count+1)*control_pos-1],\
                            color=line_color[control_pos_count], marker='o', markersize=6.0, linewidth=2)

                ax[row][col].set_title('shank: ' + self.shank_names[self.shank_ids[unit]] + \
                        ' depth: ' + str(self.depths[unit]) + \
                        '\ncell type: ' + str(self.cell_type[unit]) + ' unit index: ' + str(unit))
                        #' depth: ' + str(self.neo_obj.segments[0].spiketrains[unit].annotations['depth']) + \
                ax[row][col].plot([0, control_pos+1],[0,0],'--k')
                ax[row][col].set_xlim(0, control_pos+1)
                ax[row][col].set_ylim(ax[row][col].get_ylim()[0]-1, ax[row][col].get_ylim()[1]+1)
                #ax[row][col].set_xticks([])
                ax[row][col].set_xticks(x_vals)
                ax[row][col].set_xticklabels(labels, fontsize=8, rotation=45)
                ax[row][col].set_xlabel(xlabel, fontsize=10)
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

    def plot_raster(self, unit_ind=0, trial_type=0, axis=None, burst=False, stim_choice=True):
        """
        Makes a raster plot for the given unit index and trial type.
        If called alone it will plot a raster to the current axis. This function
        is called by plot_all_rasters and returns an axis handle to the current
        subplot. This allows plot_all_rasters to plot rasters in the appropriate
        subplots.

        stim_choice will highlight the stim region for 8-bar position experiment
            or the decision region for jb_behavior
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
        if stim_choice:
            ax.axvspan(self.t_after_stim, self.stim_duration, alpha=0.2, color='green')
        ax.set_xlim(self._bins[0], self._bins[-1])
        ax.set_ylim(0, trial+1)

        return ax

    def plot_raster_all_conditions(self, unit_ind=0, num_trials=None, offset=5, axis=None, burst=False, stim_choice=True):
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
        if stim_choice:
            ax.axvspan(self.t_after_stim, self.stim_duration, alpha=0.2, color='tab:blue')
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

    def plot_all_rasters(self, unit_ind=0, burst=False, stim_choice=True):
        """
        Plots all rasters for a given unit with subplots.
        Each positions is a row and each manipulation is a column.
        """
        num_manipulations = int(self.stim_ids.shape[0]/self.control_pos)
        subplt_indices    = np.arange(self.control_pos*num_manipulations).reshape(self.control_pos, num_manipulations)
        fig, ax = plt.subplots(self.control_pos, num_manipulations, figsize=(6*num_manipulations, 12))

        for manip in range(num_manipulations):
            for trial in range(self.control_pos):
                plt.subplot(self.control_pos, num_manipulations, subplt_indices[trial, manip]+1)
                self.plot_raster(unit_ind=unit_ind, trial_type=(trial + self.control_pos*manip), burst=burst, stim_choice=stim_choice)

        return ax

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

    def plot_mean_err(self, mean_vals, err_vals, axis=None):
        """
        plot a simple bar plot with error for all conditions
        Each manipulation will get a different color.

        input (list or array): mean_vals, and err_vals. must be the same length
            and one-dimensional. The code will find the "control position" and
            how many manipulations were present in the experiment. It will plot
            lines in different colors for each manipulation

            err_vals can be 1-dimensional if the error is symmetrical
                or it can be 2-dimensional if the error is asymmetrical
                in the ASYMMETRICAL case make sure that the array is 2xN
                where ROW1 is the value to be ADDED to the mean and ROW2
                is the value to be SUBTRACTED from the mean

        ax (optional): pass an axis so this plat can be easily added to an
            existing subplot

        TODO: if more than 3 manipulations this will break FIX THIS
        """

        # check err_vals is either 1dim or 2xN
        err_vals = np.asarray(err_vals)
        ndims = np.ndim(err_vals)

        if ndims == 1:
            asymmetrical = False

        elif ndims == 2:
            if err_vals.shape[0] != 2:
                raise Exception('the err_value array is NOT 2xN dimensions!')
            asymmetrical = True

        # check if axis was provided
        if axis == None:
            fig, ax = plt.subplots()
        else:
            ax = axis

        # make plots
        pos = range(1, self.control_pos)
        line_color = ['k','r','b']

        for control_pos_count, first_pos in enumerate(range(0, len(self.stim_ids), self.control_pos)):
            x  = pos[0:self.control_pos-1]
            xc = self.control_pos
            y  = mean_vals[(control_pos_count*self.control_pos):((control_pos_count+1)*self.control_pos-1)]
            yc = mean_vals[(control_pos_count+1)*self.control_pos-1]


            if asymmetrical:
                yerr  = err_vals[:, (control_pos_count*self.control_pos):((control_pos_count+1)*self.control_pos-1)]
                yerrc = err_vals[:, (control_pos_count+1)*self.control_pos-1].reshape(2, 1)

            else:
                yerr  = err_vals[(control_pos_count*self.control_pos):((control_pos_count+1)*self.control_pos-1)]
                yerrc = err_vals[(control_pos_count+1)*self.control_pos-1]

            # plot connected points
            _ , caps,  bars  = ax.errorbar(x, y, yerr, color=line_color[control_pos_count],\
                    marker='o', markersize=6.0, linewidth=2, capsize=5, markeredgewidth=1.5)
            # plot control position
            _ , capsc, barsc = ax.errorbar(xc, yc, yerrc, color=line_color[control_pos_count],\
                    marker='o', markersize=6.0, linewidth=2, capsize=5, markeredgewidth=1.5)

            ### CONTROL POSITION ERROR BARS ARE NOT CONNECTED ###

            # loop through bars and caps and set the alpha value
            [bar.set_alpha(0.25) for bar in bars]
            [cap.set_alpha(0.25) for cap in caps]
            [bar.set_alpha(0.25) for bar in barsc]
            [cap.set_alpha(0.25) for cap in capsc]

        return ax

    def plot_cont_mean_err(self, x, mean_vals, err_vals, axis=None, line_color='b', alpha=0.3):
        """ plot a continuous variable with shaded between error """
        # check if axis was provided
        if axis == None:
            fig, ax = plt.subplots()
        else:
            ax = axis

        # make plots
        ax.plot(x, mean_vals, color=line_color)
        ax.fill_between(x, mean_vals - err_vals, mean_vals + err_vals, facecolor=line_color, alpha=0.3)

    def plot_yyaxis(self, x, y1, y2):
        fig, ax1 = plt.subplots()
        ax1.plot(x, y1, color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.plot(x, y2, color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        fig.tight_layout()


########## MAIN CODE ##########
########## MAIN CODE ##########

if __name__ == "__main__":
    sns.set_style("whitegrid", {'axes.grid' : False})

    if os.path.isdir('/Users/Greg/Dropbox/A1Thesis/data/hdf5/'):
        data_dir = '/Users/Greg/Dropbox/A1Thesis/data/hdf5/'
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

# NOTE after specifying trials as ENGAGED/DIS-ENGAGED nothing immediately
# changes except the trial_class['jb_engaged'] labels. To see plots based on
# either the newly labeled ENGAGED/DIS-ENGAGED trials you must run "rates()"
# with engaged set to True/False




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


# NOTE after specifying trials as ENGAGED/DIS-ENGAGED nothing immediately
# changes except the trial_class['jb_engaged'] labels. To see plots based on
# either the newly labeled ENGAGED/DIS-ENGAGED trials you must run "rates()"
# with engaged set to True/False





#    ## for set-point
#    stp = neuro.wt[7][:, 1, 0]
##    msp, _, _ = neuro.get_wt_kinematic(t_window=[-2, 2], correct=None)
##    stp = msp[7+9]
#    sg = sp.signal.savgol_filter(stp, window_length=401, polyorder=2, deriv=2)
#
#
#
#    ## for running
#    #stp = neuro.run[0][:, 3]
#    #sg = sp.signal.savgol_filter(stp, window_length=12001, polyorder=2, deriv=2)
#    onset = np.argmin(sg)
#    offset = np.argmax(sg)
#
#    fig, ax = plt.subplots()
#    ax.plot(neuro.wtt, stp, color='tab:blue')
#    ax.tick_params(axis='y', labelcolor='tab:blue')
#    ymin, ymax = ax.get_ylim()
#    ax.vlines(neuro.wtt[onset], ymin, ymax, linestyles='dashed')
#    ax.vlines(neuro.wtt[offset], ymin, ymax, linestyles='dashed')
#    ax.set_ylabel('set-point (deg)')
#    ax.set_xlabel('time (s)')
#    ax.set_title('determining single trial set-point\n retraction onset & offset times')
#
#    ax2 = ax.twinx()
#    ax2.plot(neuro.wtt, sg, color='tab:red')
#    ax2.tick_params(axis='y', labelcolor='tab:red')
#    ax2.set_ylabel('savgol-filter second derivative')
#    fig.tight_layout()
#
#def find_change_points(self, data, window_length=401, polyorder=2, deriv=2):
#    """
#    use savgol_filter to smooth and calculate derivative of input signal
#
#    data: n x l array (samples x trials)
#
#    returns: delta_inds (int)  l x 2 onset index, offset index
#             delta(float array): change from onset to offset
#
#    """
#
#    l_trials = data.shape[1]
#    delta_inds  = np.zeros((l_trials, 2), dtype=int)
#    delta = np.zeros((l_trials, ))
#    for k in range(l_trials):
#        sg = sp.signal.savgol_filter(data[:, k], window_length=window_length,\
#                polyorder=polyorder, deriv=deriv)
#        onset  = np.argmin(sg)
#        offset = np.argmax(sg)
#        delta[k]  = data[offset, k] - data[onset, k]
#
#        delta_inds[k, :] = np.asarray((onset, offset))
#
#    return delta_inds, delta
#
#
#def plot_trial_delta(self, cond2compute=[0]):
#    """
#    blah
#
#    ADD SELF ONCE THIS WORKS
#
#    """
#
#    deltas = [list() for x in range(len(cond2compute))]
#
#    for k, cond in enumerate(cond2compute):
#        data = self.wt[cond][:, 1, :]
#        delta_inds, delta = find_change_points(self, data, window_length=401, polyorder=2, deriv=2)
#        deltas[k] = np.asarray((self.wtt[delta_inds[:,0]], self.wtt[delta_inds[:, 1]], delta)).T
#
#    return deltas
#
#
#
























