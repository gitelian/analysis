#!/bin/bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.io as sio
import h5py
import glob
from scipy.interpolate import interp1d
from scipy import signal as sig
import re
import os
import pandas as pd
#import itertools as it
from sklearn.cluster import KMeans
from sklearn import mixture
import multiprocessing as mp
import time
## NEW STUFF FOR NEO ##
import os.path
import neo
from neo.io import NeoHdf5IO
import quantities as pq
## NEW STUFF FOR WARNING
from warnings import warn

def load_spike_file(path):
    """
    Loads spikes file from specified path
    """
    mat  = h5py.File(path)
    spks = mat['spikes']
    assigns     = np.asarray(spks['assigns']).T        # ndarray shape(n) n: num of spike times for all units
    trials      = np.asarray(spks['trials']).T         # ndarray shape(n) n: num of spike times for all units
    spike_times = np.asarray(spks['spiketimes']).T     # ndarray shape(n) n: num of spike times for all units
    waves       = np.asarray(spks['waveforms']).T      # ndarray shape(n x m x p) n: num of spike times for all units m: num of range(m)les in waveform
    trial_times = np.asarray(spks['trial_times']).T    # p: num of recording channels
    labels      = np.asarray(spks['labels']).T
    nsamp       = waves.shape[1]
    nchan       = waves.shape[2]  # get number of channels used

    uniqassigns = np.unique(assigns)  # get unique unit numbers
    unit_type   = labels[:, 1]        # get unit type 0=noise, 1=multi-unit, 2=single-unit, 3=unsorted
    ids         = labels[:, 0]
    nunit       = len(ids)

    return labels, assigns, trials, spike_times, waves, nsamp, nchan, ids, nunit, unit_type, trial_times

def load_v73_mat_file(file_path, variable_name='spike_measures'):
    '''
    This will load in a vector, matrix, or simple cell array
    A simple cell array is considered a cell array that contains vectors,
    matrices, or other entries that are NOT additional cells.

    WARNING:
    Loading in MATLAB -v7.3 files is finiky. You may need to play around with
    the h5py.File('path/to/file') function to figure out how to access the data
    '''
    # if it is a simple cell array it will have h5 references in each cell
    # these references must be passed into the original file ('mat') to
    # retrieve the data
    # MAKES SURE IT IS THE CORRECT SHAPE! Loading in -v7.3 mat files is FINIKY!!!

    print('\n----- load_v73_mat_file -----')
    print('Loading data from: ' + file_path + '\nvariable: ' + variable_name)
    mat = h5py.File(file_path)

    if variable_name == 'run_cell':
        data = [mat[element][0][:].T for element in mat[variable_name][0]]

    elif variable_name == 'lfp':
        data = list()
        for k in range(mat['lfp'][0].shape[0]):
            data.append(mat[mat['lfp'][0][k]][:].T)

    elif variable_name == 'spike_measures':
        data = mat[variable_name][:].T

    else: # try this and hope it works!
        data = mat[variable_name][:].T

    return data

def load_mat_file(file_path, variable_name='spike_msr_mat'):
    '''Loads in MATLAB files that are not -v7.3'''
    print('\n----- load_mat_file -----')
    print('Loading data from: ' + file_path)
    mat = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    return mat[variable_name]

def calculate_runspeed(run_list,Fs=30000.0):
    '''
    Calculates runspeed from a list of numpy arrays containing distance
    traveled. Distance traveled is calculated by a MATLAB function that
    integrates the number of encoder pulses identified. Each entry in the list
    should be the distance traveled for the trial of the same index.

    This function convolves a differentiated Gaussian which simultaneously
    smooths and differentiates the distance traveled to velocity. It does this
    by creating a parallel pool of workers where each worker computes the
    convolution of a single trial.

    Returns a downsampled velocity list and a trial time list containing the
    corresponding times for each velocity entry.
    '''

    print('\n----- calculate_runspeed -----')
    processes = 4
    pool = mp.Pool(processes)
    t = time.time()

    ntrials = len(run_list)
    down_samp = int(5.0)
    down_samp_fs = round(Fs/down_samp)

    vel_list = list(run_list)
    gauss_window = make_gauss_window(int(down_samp_fs),255.0,down_samp_fs)
    window_len = len(gauss_window)

    x_t = [trial[0::down_samp] for trial in run_list]
    trtime_list = [np.linspace(0, trial.shape[0]/down_samp_fs - 1/down_samp_fs, trial.shape[0]) for trial in x_t]

    print('computing run speed with {0} processes'.format(processes))
    results = [pool.apply_async(calculate_runspeed_derivative, args=(x_t[i][:], gauss_window,window_len,down_samp_fs,i)) for i in range(len(x_t))]
    results = [p.get() for p in results]
    # ensure output is in correct order. 'apply_async' does not ensure order preservation
    order,data = zip(*[(entry[0],entry[1]) for entry in results])
    sort_ind = np.argsort(order)
    for ind in sort_ind:
            vel_list[ind] = data[ind]

    elapsed = time.time() - t
    pool.close()
    print('total time: ' + str(elapsed))

    return vel_list, trtime_list

def calculate_runspeed_derivative(x_t,gauss_window,window_len,down_samp_fs,order):
	# calculate slope of last 500 samples (last 83ms), use it to extrapolate x(t) signal. assumes mouse runs at
	# that constant velocity. Prevents convolution from producing a huge drop in velocity due to repeating last values
	xx_t = np.append(x_t,np.linspace(x_t[-1],(x_t[-1]-x_t[-500])/500.0*window_len+ x_t[-1],window_len))
	xx_t = np.append(np.linspace(x_t[0] - (x_t[500]-x_t[0])/500.0*window_len, x_t[1], window_len), xx_t)
	dx_dt_gauss_window = np.append(0,np.diff(gauss_window))/(1.0/down_samp_fs)
	dx_dt = np.convolve(xx_t,dx_dt_gauss_window,mode='same')
	dx_dt = dx_dt[window_len:-window_len]
        return order,dx_dt

def fwhm(x,y):
    '''
    function width = fwhm(x,y)
    Full-Width at Half-Maximum (FWHM) of the waveform y(x)
    and its polarity.
    The FWHM result in 'width' will be in units of 'x'
    Rev 1.2, April 2006 (Patrick Egan)
    Translated from MATLAB to Python December 2014 by Greg I. Telian
    '''

    y = y/max(y)
    N = len(y)
    lev50 = 0.5

    if y[0] < lev50:
        center_index = np.argmax(y)
        Pol = +1
    else:
        center_index = np.argmin(y)
        Pol = -1

    i = 1
    while np.sign(y[i] - lev50) == np.sign(y[i+1] - lev50):
        i += 1

    interp = (lev50 - y[i-1]) / (y[i] - y[i-1])
    tlead = x[i-1] + interp*(x[i] - x[i-1])

    i = center_index + 1
    while ((np.sign(y[i] - lev50) == np.sign(y[i-1] - lev50)) and (i <= N-1)):
        i = i+1

    if i != N:
        interp = (lev50 - y[i-1]) / (y[i] - y[i-1])
        ttrial = x[i-1] + interp*(x[i] - x[i-1])
        width  = ttrial - tlead

    return width

def make_gauss_window(length,std,Fs,make_plot=False):
    '''
    Takes input of window length and alpha and retruns a vector containing a
    smoothing kernel. Output the full width at half max to the display

    Input
    length: length of window, in number of samples
    alpha: alpha parameter for gaussian shape
    Fs: sampling rate of actual data
    Outputs
    smooth_win: vector of smoothing kernel points
    FWHM: full width at half maximum value, in seconds
    also outputs FWHM to display

    J. Schroeder
    Boston University, Ritt lab
    5/31/2012

    Translated from MATLAB to Python December 2014 by Greg I. Telian
    '''
    length = int(length)
    std = float(std)
    Fs = float(Fs)

    window = sig.gaussian(length,std)
    window = window/sum(window)

    if make_plot is True:
        fig = plt.subplots()
        plt.plot(np.arange(length)/Fs,window)

	FWHM = fwhm(np.arange(length)/Fs,window)
	print('Full width at half max for Gauss Kernel is ' + str(FWHM*1000) + ' ms')

    return window

def plot_running_subset(trtime_list, vel_list, stim_time_list, conversion=False):
    num_rows = 5
    num_cols = 6
    step = len(vel_list)/30

    if conversion:
        # circumfrence per revolution (with a mouse 6cm away from center) / number of pulses per revolution
        scale_factor = (2*np.pi*6)/360.0
        vel_list = [scale_factor* trial for trial in vel_list]

    # make heatmap of all running trials during the stimulus period
    num_trials = len(stim_time_list)
    stim_period_inds = np.where((trtime_list[0] >= stim_time_list[0][0])\
            & (trtime_list[0] <= stim_time_list[0][1]))[0]
    for k in range(num_trials):
        if k == 0:
            vel_mat = np.zeros((num_trials, len(stim_period_inds)))
            vel_mat[k, :] = vel_list[k][stim_period_inds]
        else:
            vel_mat[k, :] = vel_list[k][stim_period_inds]
    plt.figure()
    plt.imshow(vel_mat, interpolation='nearest', aspect='auto', cmap='hot', \
            extent=[trtime_list[0][stim_period_inds[0]], trtime_list[0][stim_period_inds[-1]],\
            0, num_trials])
    plt.xlabel('time (s)'); plt.ylabel('trial')
    plt.title('running speed during stimulus period across entire experiment')
    cbar = plt.colorbar().set_label('runspeed cm/s')

    # make subplots of a subset of trials
    i = 0
    subplot_count = 1
    fig, ax = plt.subplots(num_rows, num_cols, sharex=True, sharey=True)
    while subplot_count <= 30:
        if i == 0:
            ax1 = plt.subplot(num_rows, num_cols, subplot_count)
            plt.plot(trtime_list[i], vel_list[i])
            plt.xlim(0, trtime_list[i][-1])
            plt.title('trial: ' + str(i))

            if conversion:
                plt.ylim(0, 105)
                plt.ylabel('cm/s')
            else:
                plt.ylim(0, 1000)
                plt.ylabel('pulses/s')
        else:
            ax2 = plt.subplot(num_rows, num_cols, subplot_count, sharex=ax1, sharey=ax1)
            plt.plot(trtime_list[i], vel_list[i])
            plt.xlim(0, trtime_list[i][-1])
            plt.title('trial: ' + str(i))

            if conversion:
                plt.ylim(0, 105)
            else:
                plt.ylim(0, 1000)

        i += step
        subplot_count += 1

    plt.suptitle('birds eye view of runspeeds across experiment')
    plt.show()

def classify_run_trials(vel_list, trtime_list, stim_time_list, t_after_start=0.250,\
        t_after_stop=0.5, mean_thresh=250, sigma_thresh=150, low_thresh=200, display=False):
    num_trials = len(vel_list)
    mean_vel = []
    sigm_vel = []
    run_bool_list = [False]*num_trials

    for count, trial in enumerate(range(num_trials)):
        stim_period_inds = (trtime_list[trial] >= (stim_time_list[trial][0] + t_after_start))\
                & (trtime_list[trial] <= (stim_time_list[trial][1] + t_after_stop))
        vel = vel_list[trial][stim_period_inds]
        mean_vel.append(np.mean(vel))
        sigm_vel.append(np.std(vel))
        if np.mean(vel) >= mean_thresh and np.std(vel) <= sigma_thresh and (sum(vel <= low_thresh)/len(vel)) <= 0.1:
            run_bool_list[count] = True

    if display:
        bins = range(0, 1000, 5)
        fig,  ax = plt.subplots()
        plt.subplot(2, 1, 1)
        mean_counts,  _ = np.histogram(mean_vel, bins)
        plt.bar(bins[:-1], mean_counts, width=5.0)
        plt.subplot(2, 1, 2)
        sigm_counts,  _ = np.histogram(sigm_vel, bins)
        plt.bar(bins[:-1], sigm_counts, width=5.0)
        plt.show()

    return run_bool_list

def get_exp_details_info(data_dir_path, fid, key):
    ##### LOAD IN EXPERIMENT DETAILS CSV FILE #####
    print('\n----- get_exp_details_info -----')
    experiment_details_path = data_dir_path + 'experiment_details.csv'
    print('Loading experiment details file for FID: ' + str(fid) + '\nkey: ' + key)
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
        print('Loading experiment info for fid: ' + str(fid))
        return exp_info[key]

def make_neo_object(writer, data_dir, fid, lfp_files, spikes_files, \
        wtrack_files, vel_list, run_bool_list, stim, stim_time_list):
    print('\n----- make_neo_block -----')

    ## Make neo block and segment objects
    #  block = neo.Block(name='test_block', description="This is a simple Neo example")
    #  create block for experiment. make the same number of segments as number of trials
    for trial_ind in np.arange(stim.shape[0]):

        # add trial info to trial segment data
        seg_description = 'Trial number: ' + str(trial_ind) + ' Trial type: ' + str(stim[trial_ind][0])
        segment = neo.Segment(
                description  = seg_description,
                index        = trial_ind,                    # trial number as an index
                run_boolean  = run_bool_list[trial_ind],     # running=1, not-running=0
                trial_number = trial_ind,                    # trial number as an annotation
                trial_type   = stim[trial_ind][0],           # stimulus presented
                stim_times   = stim_time_list[trial_ind])    # start/stop times of the stimulus object
        block.segments.append(segment)

        # add velocity data to trial segment
        sig0 = neo.AnalogSignal(
                signal=vel_list[trial_ind][:],
                units=pq.deg/pq.S,
                sampling_rate=6*pq.kHz,
                name='run speed')
        block.segments[trial_ind].analogsignals.append(sig0)
        ##### How to access important metadata from analogsignals #####
        #     a = block.segments[0].analogsignals[::2][0]
        #     a.t_start, a.tstop, num_samples=len(a)
        #     to make a time vector you could do: t = np.linspace(a.t_start,
        #     a.t_stop, num_samples)

    ## Add LFPs to trial segment
    if lfp_files:
        for e, lfp_path in enumerate(lfp_files):
            lfp_fname       = os.path.split(lfp_path)[1]
            electrode_match = re.search(r'e\d{0,2}', lfp_fname)
            e_name          = electrode_match.group()
            e_num           = int(e_name[1::])
            print('loading LFPs from: ' + lfp_fname)
            lfp             = load_v73_mat_file(lfp_path, variable_name='lfp')

            # for each trial add LFPs for every channel on the electrode
            for trial_ind in np.arange(stim.shape[0]):
                sig0 = neo.AnalogSignalArray(
                        signal=lfp[trial_ind],
                        units=pq.uV,
                        sampling_rate=1.5*pq.kHz,
                        name='LFPs'+'-'+e_name,
                        shank_name=e_name,
                        shank_num=e_num)
                block.segments[trial_ind].analogsignals.append(sig0)

    if wtrack_files:
        for e, wt_path in enumerate(wtrack_files):
            wt_fname = os.path.split(wt_path)[1]
            print('loading whisker tracking data from: ' + wt_name)
            wt       = load_v73_mat_file(lfp_path, variable_name='lfp')

            for trial_ind in np.arange(stim.shape[0]):
                sig0 = neo.AnalogSignal(
                        signal=wt[trial_ind][:,0],
                        units=pq.deg,
                        sampling_rate=500*pq.Hz,
                        name='angle')
                sig1 = neo.AnalogSignal(
                        signal=wt[trial_ind][:,1],
                        units=pq.deg,
                        sampling_rate=500*pq.Hz,
                        name='set-point')
                sig2 = neo.AnalogSignal(
                        signal=wt[trial_ind][:,2],
                        units=pq.deg,
                        sampling_rate=500*pq.Hz,
                        name='amplitude')
                sig3 = neo.AnalogSignal(
                        signal=wt[trial_ind][:,3],
                        units=pq.deg/pq.S,
                        sampling_rate=500*pq.Hz,
                        name='velocity')
                # ADD PHASE???
                sig4 = neo.AnalogSignal(
                        signal=wt[trial_ind][:,3],
                        units=pq.rad,
                        sampling_rate=500*pq.Hz,
                        name='phase')
                block.segments[trial_ind].analogsignals.append(sig0)
                block.segments[trial_ind].analogsignals.append(sig1)
                block.segments[trial_ind].analogsignals.append(sig2)
                block.segments[trial_ind].analogsignals.append(sig3)
                block.segments[trial_ind].analogsignals.append(sig4)

    ## Load in spike measure mat file ##
    spike_measure_path = data_dir + 'spike_measures.mat'
    if os.path.exists(spike_measure_path):
        spk_msrs = load_mat_file(spike_measure_path, variable_name='spike_msr_mat')

    if spikes_files:
        for e, spike_path in enumerate(spikes_files):
            spike_fname       = os.path.split(spike_path)[1]
            electrode_match = re.search(r'e\d{0,2}', spike_fname)
            e_name          = electrode_match.group()
            e_num           = int(e_name[1::])
            fid_match       = re.search(r'FID\d{0,4}', spike_fname, re.IGNORECASE)
            fid_name        = fid_match.group()
            fid_num         = int(fid_name[3::])
            fid_inds        = np.where(spk_msrs[:, 0] == fid_num)[0]
            e_inds          = np.where(spk_msrs[:, 1] == e_num)[0]
            exp_inds        = np.intersect1d(fid_inds, e_inds)
            print('loading spikes from: ' + spike_fname)
            labels, assigns, trials, spiketimes, _, _,\
                    _, ids, nunit, unit_type, trial_times = load_spike_file(spike_path)

            for trial_ind in np.arange(stim.shape[0]):
                for k, unit in enumerate(ids): # iterate through ALL UNITS FROM ONE SPIKE FILE

                    ## GETS SPIKE TIMES FOR ONE UNIT FOR ONE TRIAL ##
                    spk_times_bool = sp.logical_and(trials == trial_ind + 1, assigns == unit) # trial_ind + 1 since trials are 1 based
                    if unit_type[k] > 0 and unit_type[k] < 3: # get multi-unit and single-units

                        # look for unit in spike_measures matrix
                        unit_inds = np.where(spk_msrs[:, 2] == unit)[0]
                        unit_ind = np.intersect1d(exp_inds, unit_inds)

                        # if unit is in spike_measures file add appropriate data
                        if unit_ind:
                            # round t_stop because sometimes a spiketime would
                            # be slightly longer than it (by about 0.0001 s)
                            block.segments[trial_ind].spiketrains.append(neo.SpikeTrain(spiketimes[spk_times_bool],
                                    t_start=trial_times[trial_ind, 0] * pq.s,
                                    t_stop=trial_times[trial_ind, 1] * pq.s,
                                    sampling_rate=30 * pq.kHz,
                                    units=pq.s,
                                    description="Spike train for: " + fid_name + '-' +  e_name + '-unit' +  str(int(unit)),
                                    depth=spk_msrs[unit_ind, 3]*pq.um,
                                    cell_type=spk_msrs[unit_ind, 7],
                                    fid=fid_name,
                                    shank=e_name,
                                    unit_id=unit))

                        # if unit isn't in spike_measures file add spike times
                        # and which experiment and shank it came from and label
                        # it as an unclassified cell type with an unknown depth
                        else:
                            block.segments[trial_ind].spiketrains.append(neo.SpikeTrain(spiketimes[spk_times_bool],
                                    t_start=trial_times[trial_ind, 0] * pq.s,
                                    t_stop=trial_times[trial_ind, 1] * pq.s,
                                    sampling_rate=30 * pq.kHz,
                                    units=pq.s,
                                    description="Spike train for: " + fid_name + '-' +  e_name + '-unit' +  str(int(unit)),
                                    depth=np.nan * pq.um,
                                    cell_type=3,
                                    fid=fid_name,
                                    shank=e_name,
                                    unit_id=unit))

        # close writer object to stop adding blocks to the file
    writer.write(block)
    return block

########## MAIN CODE ##########
########## MAIN CODE ##########

if __name__ == "__main__":
    # Select which experiments to analyze
    fids = ['FID1293']
    #data_dir = '/Users/Greg/Documents/AdesnikLab/Data/'
    data_dir = '/media/greg/Data/Neuro/'

    writer = NeoHdf5IO('/media/greg/Data/Neuro/neo/' + fids[0] + '_neo_object.h5')

    for fid in fids:
        # get paths to run, whiser tracking, lfp, and spikes files if they
        # exist.
        # REMEMBER glob.glob returns a LIST of path strings you must
        # index into the appropriate one for whatever experiment/electrode
        # you're trying to add to the neo object
        run_file = glob.glob(data_dir + fid + '*/' + fid + '*.run')
        lfp_files = glob.glob(data_dir + fid + '*/' + fid + '_e*/' + fid + '*LFP.mat')
        spikes_files = glob.glob(data_dir + fid + '*/' + fid + '_e*/' + fid + '*spikes.mat')
        wtrack_files = glob.glob(data_dir + fid + '*/' + fid + '*.wtr')

        # Calculate runspeed
        run_list = load_v73_mat_file(run_file[0], variable_name='run_cell')
        vel_list, trtime_list = calculate_runspeed(run_list)

        # get stimulus on/off times and stimulus ID for each trial
        stim_time_list = load_v73_mat_file(run_file[0], variable_name='stimulus_times')
        stim = load_v73_mat_file(run_file[0], variable_name='stimsequence')

        # Plot runspeed
        plot_running_subset(trtime_list, vel_list, stim_time_list, conversion=True)

        # # Create running trial dictionary
        run_bool_list = classify_run_trials(vel_list, trtime_list, stim_time_list, t_after_start=1.25,\
                t_after_stop=2.25, mean_thresh=250, sigma_thresh=150, low_thresh=200, display=True)

        ## get control position
        control_pos = get_exp_details_info(data_dir, int(fid[3::]), 'control_pos')

        # Put data into a neo object and save
        block = neo.Block(name=fid, description='This is a neo block for experiment ' + fid, \
                control_pos=control_pos) # create block for experiment
        block = make_neo_object(writer, data_dir, fid, lfp_files, spikes_files,\
                wtrack_files, vel_list, run_bool_list, stim, stim_time_list)
    writer.close()


#how to get spike times: block.segments[0].spiketrains[0].tolist()


