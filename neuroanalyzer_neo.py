#!/bin/bash

import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from neo.io import NeoHdf5IO

#manager = NeoHdf5IO(os.path.join('/Users', 'Greg', 'Desktop', 'fid0871_0872_0873_vS1.h5'))
#manager = NeoHdf5IO(os.path.join('/Users', 'Greg', 'Desktop', 'fid0871_m1_s1.h5'))
manager = NeoHdf5IO(os.path.join('/Users', 'Greg', 'Desktop', 'fid1220_m1_s1.h5'))

block = manager.read()
manager.close()

print block


fid1 = block[0]

num_units = len(fid1.segments[0].spiketrains)
stimids = np.array([k.annotations['trial_type'] for k in fid1.segments])

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


#########


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


























