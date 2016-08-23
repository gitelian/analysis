#!/bin/bash

import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from neo.io import NeoHdf5IO


def get_stimulus_index(segments, trial_type, run_trials=True):
    '''
    Returns the neo segment indices for the desired trial type. When run_trials
    is set to True only trials that the animal was running are included. When
    False all trials are included.
    '''
    stim_index = np.where(stimids == trial_type)[0]

    if run_trials:
        stim_index  = np.array([run_stim_index for run_stim_index in stim_index \
                if segments[run_stim_index].annotations['run_boolean']])
    return stim_index

def single_trial_all_unit_raster(segments, trial_type, stim_times, stim_ids, run_trials=True):
    '''
    Plot single trial rasters with all recorded units

    segments: neo segments object.
    trial_typ: trial type to analyze. int
    stim_times: list containing onset and offset of the stimulus period (s). float
    run_trials: boolean. whether to analyze only running trials.

    OUTPUT
    multiple raster plots


    TODO: Sort units by depth
    '''

    num_units = len(segments[0].spiketrains)
    stim_index = get_stimulus_index(segments, trial_type, run_trials=True)

    for k, sind in enumerate(stim_index):

        print('on iteration ' + str(k))
        fig = plt.figure()
        for unit in range(num_units):
            plt.vlines(segments[sind].spiketrains[unit], unit, unit+1, 'k')
        plt.vlines(np.array(stim_times)*1000, 0, plt.ylim()[1], 'r')
        plt.show()

def make_raster_movie(segments, trial_type, stim_times, run_trials=True):

    stim_index = get_stimulus_index(segments, trial_type, run_trials)
    num_frames = len(stim_index)
    num_units = len(segments[0].spiketrains)

    def get_spike_xy_coord(segment, num_units):
        x = list()
        y = list()
        for unit in range(num_units):
            x.extend(segment.spiketrains[unit].tolist())
            y.extend(np.ones(len(segment.spiketrains[unit]), )*unit)
        return np.array(x), np.array(y)

    FFMpegWriter = animation.writers['ffmpeg']
    writer = FFMpegWriter(fps=3)
    fig = plt.figure()

    with writer.saving(fig, "writer_test_" + str(trial_type) + ".mp4", 70):
        for sind in stim_index:
            x, y = get_spike_xy_coord(segments[sind], num_units)
            plt.vlines(x, y-1, y, 'k')
            plt.vlines(np.array(stim_times)*1000, 0, num_units, 'r')
            plt.ylim(0, num_units)
            plt.xlabel('time (ms)')
            plt.ylabel('unit')
            plt.title('all unit raster')
            writer.grab_frame()
            plt.clf()
    plt.close()

class NeuroAnalyzer(object):
    """Analyzes data contained in a neo object"""

    def __init__(self, neo_obj):
        # segments can get out of the order they were created
        sorted_index_list = np.argsort([k.index for k in neo_obj.segments])
        neo_obj.segments  = [neo_obj.segments[k] for k in sorted_index_list]

        self.neo_obj        = neo_obj
        self.stim_ids       = np.sort(np.unique([k.annotations['trial_type'] for k in neo_obj.segments]))
        self.num_run_trials = self.__get_num_run_trials()
        self.num_units      = len(self.neo_obj.segments[0].spiketrains)
        self.shank_names    = np.sort(np.unique([k.annotations['shank'] for k in neo_obj.segments[0].spiketrains]))
        self.shank_ids      = self.__get_shank_ids()

    def __get_shank_ids(self):
        shank_ids = np.zeros((self.num_units, ))
        for k, shank_name in enumerate(self.shank_names):
            for j, unit in enumerate(self.neo_obj.segments[0].spiketrains):
                if unit.annotations['shank'] == shank_name:
                    shank_ids[j] = k
        return shank_ids

    def __get_num_run_trials(self):
        '''return a list with the number of running trials for each stimulus condition'''
        num_run_trials = list()
        for stim_id in self.stim_ids:
            run_count = 0
            for trial in self.neo_obj.segments:
                if trial.annotations['trial_type'] == stim_id and trial.annotations['run_boolean']:
                    run_count += 1
            num_run_trials.append(run_count)
        return num_run_trials

    def rates(self):

        t_after_stim = 0.500 # change this from being hardcoded!!!
        absolute_rate    = list()
        evoked_rate      = list()
        absolute_counts  = list()
        evoked_counts    = list()

        for k, trials_ran in enumerate(self.num_run_trials):
                absolute_rate.append(np.zeros((trials_ran, self.num_units)))
                evoked_rate.append(np.zeros((trials_ran, self.num_units)))
                absolute_counts.append(np.zeros((trials_ran, self.num_units)))
                evoked_counts.append(np.zeros((trials_ran, self.num_units)))

        for stim_ind, stim_id in enumerate(self.stim_ids):
            good_trial_ind = 0

            for trial in self.neo_obj.segments:
                if trial.annotations['trial_type'] == stim_id and trial.annotations['run_boolean']:

                    stim_start = trial.annotations['stim_times'][0] + t_after_stim
                    stim_stop  = trial.annotations['stim_times'][1]
                    base_start = trial.annotations['stim_times'][0] - (stim_stop - stim_start)
                    base_stop  = trial.annotations['stim_times'][0]

                    for unit, spike_train in enumerate(trial.spiketrains):
                        spk_times = np.asarray(spike_train.tolist())

                        # calculate absolute and evoked rate
                        abs_rate = (np.logical_and(spk_times > stim_start, spk_times < stim_stop).sum())/(stim_stop - stim_start)
                        evk_rate = (np.logical_and(spk_times > base_start, spk_times < base_stop).sum())/(stim_stop - stim_start)
                        absolute_rate[stim_ind][good_trial_ind, unit] = abs_rate
                        evoked_rate[stim_ind][good_trial_ind, unit]   = evk_rate

                        # calculate absolute and evoked counts
                        abs_count = np.logical_and(spk_times > stim_start, spk_times < stim_stop).sum()
                        evk_count   = (np.logical_and(spk_times > stim_start, spk_times < stim_stop).sum()) - \
                                (np.logical_and(spk_times > base_start, spk_times < base_stop).sum())
                        absolute_counts[stim_ind][good_trial_ind, unit] = abs_count
                        evoked_counts[stim_ind][good_trial_ind, unit]   = evk_count

                    good_trial_ind += 1

        self.abs_rate  = absolute_rate
        self.abs_count = absolute_counts
        self.evk_rate  = evoked_rate
        self.evk_count = evoked_counts

    def make_simple_tuning_curve(self, kind='abs_count'):
        if hasattr(self, 'abs_rate') is False:
            self.rates()
        kind_dict      = {'abs_rate': 0, 'abs_count': 1, 'evk_rate': 2, 'evk_count': 3}
        kind_of_tuning = [self.abs_rate, self.abs_count, self.evk_rate, self.evk_count]
        for unit in range(self.num_units):
            meanr = [k[unit].mean(axis=0) for k in kind_of_tuning[kind_dict[kind]]]
            stder = [k[unit].std(axis=0)/np.sqrt(k.shape[0]) for k in kind_of_tuning[kind_dict[kind]]]
            plt.figure()
            plt.errorbar(np.arange(len(meanr))+1, meanr, yerr=stder, c='k')
            plt.title('from shank: ' + self.shank_names[self.shank_ids[unit]])
            plt.xlim(0, len(meanr)+2)
            plt.show()


########## MAIN CODE ##########
########## MAIN CODE ##########

    if __name__ == "__main__":
    data_dir = '/Users/Greg/Documents/AdesnikLab/Data/'
    manager = NeoHdf5IO(os.path.join(data_dir + 'FID1293.h5'))
    block = manager.read()
    manager.close()
    exp1 = block[0]

    neuro = NeuroAnalyzer(exp1)
    num_units = len(fid1.segments[0].spiketrains)
    stimids = np.array([k.annotations['trial_type'] for k in block.segments])

    num_units  = len(fid1.segments[0].spiketrains)
    trial_types = np.unique(stimids).shape[0]
    num_trials = stimids.shape[0]/trial_types
    m1_spk_count = np.zeros((num_units, num_trials, trial_types))
    m1_spk_count[:] = np.nan
    s1_spk_count = m1_spk_count.copy()

    m1_block = block[0]
    s1_block = block[1]

    for unit in range(num_units):
        for trial in range(num_trials):
            for trial_type in trial_types:
                spk_times = np.array(m1_block.segments[seg_ind].spiketrains[unit].tolist())



    #how to get spike times: block.segments[0].spiketrains[0].tolist()


