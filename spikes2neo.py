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

def load_spike_file(path):
    """
    Loads spikes file from specified path
    """
    mat  = h5py.File(path)
    spks = mat['spikes']

    labels      = np.asarray(spks['labels']).T     # ndarray shape(n x m) n: num elements, m0 = unit label, m1 = type of unit (i.e. single, multi, garbage)
    assigns     = np.asarray(spks['assigns']).T    # ndarray shape(n) n: num of spike times for all units
    trials      = np.asarray(spks['trials']).T     # ndarray shape(n) n: num of spike times for all units
    spike_times = np.asarray(spks['spiketimes']).T # ndarray shape(n) n: num of spike times for all units
    waves       = np.asarray(spks['waveforms']).T  # ndarray shape(n x m x p) n: num of spike times for all units m: num of range(m)les in waveform
                                    # p: num of recording channels
    nsamp       = waves.shape[1]
    nchan       = waves.shape[2]  # get number of channels used

    uniqassigns = np.unique(assigns)                   # get unique unit numbers
    unit_type   = labels[:, 1]                          # get unit type 0=noise, 1=singlunit, 3=multi-unit, 4=unsorted
    ids         = labels[:, 0]
    nunit       = len(ids)

    return labels, assigns, trials, spike_times, waves, nsamp, nchan, ids, nunit, unit_type

def load_v73_mat_file(file_path, variable_name='spike_measures'):
    '''
    This will load in a vector, matrix, or simple cell array
    A simple cell array is considered a cell array that contains vectors,
    matrices, or other entries that are NOT additional cells.
    '''
    print('\n----- load_v73_mat_file -----')
    print('Loading data from: ' + file_path + '\nvariable: ' + variable_name)
    mat = h5py.File(file_path)
    # if it is a simple cell array it will have h5 references in each cell
    # these references must be passed into the original file ('mat') to
    # retrieve the data
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
    print('MAKES SURE IT IS THE CORRECT SHAPE! Loading in -v7.3 mat files is FINIKY!!!')
    return data

def load_mat_file(file_path, variable_name='spike_msr_mat'):
    print('\n----- load_mat_file -----')
    print('Loading data from: ' + file_path)
    mat = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    return mat[variable_name]

def calculate_runspeed(run_list,Fs=30000.0):
    # original time for 350 trials: 19.8s
    # time with parallel pool using 'apply': 13.2s
    # time with parallel pool using 'apply_async' with 1-4 processes: 13.0s, 8.42, 6.14, 5.79s
    # no improvement when processes exceeds number of virtual cores

    print('\n----- calculate_runspeed -----')
    processes = 4
    pool = mp.Pool(processes)
    t = time.time()

    ntrials = len(run_list)
    #nsamples = run_mat.shape[1]
    down_samp = 5.0
    down_samp_fs = round(Fs/down_samp)

    vel_list = list(run_list)
    gauss_window = make_gauss_window(round(down_samp_fs),255.0,down_samp_fs)
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

def make_neo_object(writer, data_dir, lfp_files, spikes_files,\
                wtrack_files, vel_list, run_bool_list, stim, stim_time_list)
    print('\n----- make_neo_block -----')

    ##### REMOVE THIS AFTER DEBUGGING #####
    block = neo.Block(name=fid, description='This is a neo block for experiment' + fid) # create block for experiment
    ##### REMOVE THIS AFTER DEBUGGING #####

    ## Make neo block and segment objects
    #  block = neo.Block(name='test_block', description="This is a simple Neo example")
    #  create block for experiment. make the same number of segments as number of trials
    for trial_ind in np.arange(stim.shape[0]):

        # add trial info to trial segment data
        seg_description = 'Trial number: ' + str(trial_ind) + ' Trial type: ' + str(stim[trial_ind][0])
        segment = neo.Segment(
                description  = seg_description,
                index        = trial_ind,                    # trial number
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
            disp('loading LFPs from: ' + lfp_fname)
            lfp             = load_v73_mat_file(lfp_path, variable_name='lfp')

            for trial_ind in np.arange(stim.shape[0]):
                sig0 = neo.AnalogSignal(
                        signal=lfp[trial_ind],
                        units=pq.uV,
                        sampling_rate=1.5*pq.kHz,
                        name='LFPs',
                        shank_name=e_name,
                        shank_num=e_num)
                block.segments[trial_ind].analogsignals.append(sig0)

    if wtrack_files:
        for e, wt_path in enumerate(wtrack_files):
            wt_fname = os.path.split(wt_path)[1]
            disp('loading whisker tracking data from: ' + wt_name)
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

    if spk_files:
        for e, spk_path in enumerate(spk_files):
            spk_fname       = os.path.split(spk_path)[1]
            electrode_match = re.search(r'e\d{0,2}', spk_fname)
            e_name          = electrode_match.group()
            e_num           = int(e_name[1::])
            disp('loading spikes from: ' + spk_fname)
            labels, assigns, trials, spike_times, waves, nsamp,\
                    nchan, ids, nunit, unit_type = load_spikes_file(spk_path)

            for trial_ind in np.arange(stim.shape[0]):
                print('number of units')
                print(ids)

                for k, unit in enumerate(ids): # iterate through ALL UNITS FROM ONE SPIKE FILE

                    ## GETS SPIKE TIMES FOR ONE UNIT FOR ONE TRIAL ##
                    spk_times_bool = sp.logical_and(trials == trial_index + 1, assigns == unit) # trial_index + 1 since trials are 1 based (i.e. there is no trial 0)
                    if unit_type[k] < 3 and trial_index == 0:

                        unit_id = sort_file_basename + '-u' + str(unit).zfill(3)
                        unit_ids = df_spk_msr['unit_id'].tolist()
                        unit_index = unit_ids.index(unit_id)

                        block.segments[trial_index].spiketrains.append(neo.SpikeTrain(spike_times[spk_times_bool]*1000,
## replace with variable ----->>>>> t_start=0 * pq.ms
## replace with variable ----->>>>> t_stop=3200 * pq.ms,
                                sampling_rate=30 * pq.kHz,
                                units=pq.ms,
                                description="Spike train for unit: " + str(unit),
                                depth=df_spk_msr.loc[unit_index,'depth'],
                                cell_type=df_spk_msr.loc[unit_index,'cell_type'],
                                wave_duration=df_spk_msr.loc[unit_index,'wave_duration'],
                                wave_ratio=df_spk_msr.loc[unit_index,'wave_ratio'],
                                mean_waves=str2nparray(df_spk_msr.loc[unit_index,'mean_waves']),
                                std_waves=str2nparray(df_spk_msr.loc[unit_index,'std_waves']),
                                spike_file= spike_file_name,
                                unit_id=unit,
                                fid=fid))

                    elif unit_type[k] == 3 and trial_index == 0: #if multi-unit add to segment with no other information
                        block.segments[trial_index].spiketrains.append(neo.SpikeTrain(spike_times[spk_times_bool]*1000,
                                t_stop=3200 * pq.ms,
                                sampling_rate=30 * pq.kHz,
                                units=pq.ms,
                                description="Spike train for unit: " + str(unit),
                                depth=np.nan,
                                cell_type='MU',
                                wave_duration=np.nan,
                                wave_ratio=np.nan,
                                mean_waves=np.nan,
                                std_waves=np.nan,
                                spike_file= spike_file_name,
                                unit_id=unit,
                                fid=fid))
                    elif unit_type[k] < 3 and trial_index > 0: # add spiketrains for single unit
                        block.segments[trial_index].spiketrains.append(neo.SpikeTrain(spike_times[spk_times_bool]*1000,
                                t_stop=3200 * pq.ms,
                                sampling_rate=30 * pq.kHz,
                                units=pq.ms))
                    elif unit_type[k] == 3 and trial_index > 0: # add spiketrains for multi-unit
                        block.segments[trial_index].spiketrains.append(neo.SpikeTrain(spike_times[spk_times_bool]*1000,
                                t_stop=3200 * pq.ms,
                                sampling_rate=30 * pq.kHz,
                                units=pq.ms))
        # close writer object to stop adding blocks to the file
    writer.write(block)
    return block


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

def classify_run_trials(vel_mat, trtime_list, stim_time_list, t_after_start=0.250,\
        t_after_stop=0.5, mean_thresh=250, sigma_thresh=150, low_thresh=200, display=False):
    num_trials = len(vel_mat)
    mean_vel = []
    sigm_vel = []
    run_bool_list = [False]*num_trials

    for count, trial in enumerate(range(num_trials)):
        stim_period_inds = (trtime_list[trial] >= (stim_time_list[trial][0] + t_after_start))\
                & (trtime_list[trial] <= (stim_time_list[trial][1] + t_after_stop))
        vel = vel_mat[trial][stim_period_inds]
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

def str2nparray(string):
    '''
    convert string of numbers in CSV file to numpy array.

    Saving to a CSV saves everything as a string. Including the numpy brakets
    '[', ']', periods '.', negative signs '-', white space ' ', and 'e' for
    scientific notation. This function removes the mentioned strings (with the
    exception of white space...unless there is multiple white space strings)
    and converts the remaining strings to numbers.

    RETURNS: numpy array
    '''
    for i, x in enumerate(string):
        try:
            float(x)
        except ValueError:
            if x != ' ' and x != '.' and x != '-' and x != 'e' and x != ',':
                string = string.translate(None, x)
    string = string.replace('  ', ' ')
    string = string.replace('  ', ' ')
    k = 0
    while k < 10:
        if string[0] == ' ':
            string = string[1::]
        elif string[-1] == ' ':
            string = string[:-2]
        k += 1
    string = string.replace('  ', ' ')
    c = map(float, string.split(' '))
    return np.array(c)


########## MAIN CODE ##########
########## MAIN CODE ##########

if __name__ == "__main__":
    # Select which experiments to analyze
    fids = ['0871','0872','0873']
    fids = ['FID1293']
    writer = NeoHdf5IO(os.path.join('/Users', 'Greg', 'Desktop', 'fid0872_m1_s1.h5'))


    for fid in fids:
        # get paths to run, whiser tracking, lfp, and spikes files if they
        # exist.
        # REMEMBER glob.glob returns a LIST of path strings you must
        # index into the appropriate one for whatever experiment/electrode
        # you're trying to add to the neo object
        data_dir = '/Users/Greg/Documents/AdesnikLab/Data/'
        run_file = glob.glob(data_dir + fid + '*/' + fid + '*.run')
        lfp_files = glob.glob(data_dir + fid + '*/' + fid + '_e*/' + fid + '*LFP.mat')
        spikes_files = glob.glob(data_dir + fid + '*/' + fid + '_e*/' + fid + '*spikes.mat')
        wtrack_file = glob.glob(data_dir + fid + '*/' + fid + '*.wtr')

        # Calculate runspeed
        run_list = load_v73_mat_file(run_file[0], variable_name='run_cell')
        vel_list, trtime_list = calculate_runspeed(run_list)

        # get stimulus on/off times
        stim_time_list = load_v73_mat_file(run_file[0], variable_name='stimulus_times')
        stim = load_v73_mat_file(run_file[0], variable_name='stimsequence')

        # Plot runspeed
        plot_running_subset(trtime_list, vel_list, stim_time_list, conversion=True)

        # # Create running trial dictionary
        run_bool_list = classify_run_trials(vel_mat, trtime_list, stim_time_list, t_after_start=1.25,\
                t_after_stop=2.25, mean_thresh=250, sigma_thresh=150, low_thresh=200, display=True)

        # Put data into a neo object and save
        block = neo.Block(name=fid +, description='This is a neo block for experiment FID ' + fid) # create block for experiment
        block = make_neo_object(writer, data_dir, lfp_files, spikes_files,\
                wtrack_files, vel_list, run_bool_list, stim, stim_time_list)
    writer.close()


#how to get spike times: block.segments[0].spiketrains[0].tolist()

