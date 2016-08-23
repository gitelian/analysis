#!/bin/bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.io as sio
import h5py
import glob
from scipy import stats
from scipy.interpolate import interp1d
from scipy import signal as sig
import re
import os
import itertools as it
from sklearn.cluster import KMeans
from sklearn import mixture
import multiprocessing as mp
import time
import random as rd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import KFold

def load_spike_file(path):
    mat = sio.loadmat(path,struct_as_record=False, squeeze_me=True)
    spks = mat['spikes']

    labels      = spks.labels     # ndarray shape(n x m) n: num elements, m0 = unit label, m1 = type of unit (i.e. single, multi, garbage)
    assigns     = spks.assigns    # ndarray shape(n) n: num of spike times for all units
    trials      = spks.trials     # ndarray shape(n) n: num of spike times for all units
    spike_times = spks.spiketimes # ndarray shape(n) n: num of spike times for all units
    waves       = spks.waveforms  # ndarray shape(n x m x p) n: num of spike times for all units m: num of range(m)les in waveform
                                    # p: num of recording channels
    nsamp       = waves.shape[1]
    nchan       = waves.shape[2]  # get number of channels used

    uniqassigns = np.unique(assigns)                      # get unique unit numbers
    unit_type   = labels[:,1]                             # get unit type 1=in progress, 2=singleiuint, 3=multi-unit, 4=garbage
    spike_ids   = labels[unit_type < 4,0];                # get good spike ids
    ids_set     = [set(uniqassigns), set(spike_ids)]      # convert a list containing two sets
    ids         = np.sort(list(set.intersection(*ids_set))) # convert to set, find intersection, return list of good unit ids
    nunit       = len(ids)

    return labels, assigns, trials, spike_times, waves, nsamp, nchan, ids, nunit, unit_type

def load_stimsequence(file_path):
    print('\n----- load_stimsequence -----')
    print('Loading stimsequence from: ' + file_path)
    mat = h5py.File(file_path)
    stim = mat['stimsequence'][:]
    return stim

def load_run_file(file_path):
    print('\n----- load_run_file -----')
    print('Loading running data from: ' + file_path)
    mat = h5py.File(file_path)
    run_mat = mat['run_data']
    return run_mat

def load_mat_file(file_path):
    print('\n----- load_mat_file -----')
    print('Loading running data from: ' + file_path)
    mat = sio.loadmat(file_path,struct_as_record=False, squeeze_me=True)
    return mat

def calculate_runspeed(run_mat,Fs=30000.0):#,highthresh=250.0,lowthresh=25.0,stimstart=1.5,stimstop=2.5):
    # original time for 350 trials: 19.8s
    # time with parallel pool using 'apply': 13.2s
    # time with parallel pool using 'apply_async' with 1-4 processes: 13.0s, 8.42, 6.14, 5.79s
    # no improvement when processes exceeds number of virtual cores

    print('\n----- calculate_runspeed -----')
    processes = 4
    pool = mp.Pool(processes)
    t = time.time()

    ntrials = run_mat.shape[0]
    nsamples = run_mat.shape[1]
    down_samp = 5.0
    down_samp_fs = round(Fs/down_samp)

    vel_mat = np.zeros((ntrials,nsamples/down_samp))
    gauss_window = make_gauss_window(round(down_samp_fs),255.0,down_samp_fs)
    window_len = len(gauss_window)
    trial_time = np.linspace(0,vel_mat.shape[1]/down_samp_fs - 1/down_samp_fs,vel_mat.shape[1])

    x_t = run_mat[:,0::down_samp]

    print('computing run speed with {0} processes'.format(processes))
    results = [pool.apply_async(calculate_runspeed_derivative, args=(x_t[i,:], gauss_window,window_len,down_samp_fs,i)) for i in range(x_t.shape[0])]
    results = [p.get() for p in results]
    # ensure output is in correct order. 'apply_async' does not ensure order preservation
    order,data = zip(*[(entry[0],entry[1]) for entry in results])
    sort_ind = np.argsort(order)
    for ind in sort_ind:
            vel_mat[ind,:] = data[ind]

    elapsed = time.time() - t
    pool.close()
    print('total time: ' + str(elapsed))

    return vel_mat, trial_time

def calculate_runspeed_derivative(x_t,gauss_window,window_len,down_samp_fs,order):
	# calculate slope of last 500 samples (last 83ms), use it to extrapolate x(t) signal. assumes mouse runs at
	# that constant velocity. Prevents convolution from producing a huge drop in velocity due to repeating last values
	x_t = np.append(x_t,np.linspace(x_t[-1],(x_t[-1]-x_t[-500])/500.0 *window_len+ x_t[-1],window_len))
	dx_dt_gauss_window = np.append(0,np.diff(gauss_window))/(1.0/down_samp_fs)
	dx_dt = np.convolve(x_t,dx_dt_gauss_window,mode='same')
	dx_dt = dx_dt[:-window_len]
	return order,dx_dt

def make_df(spike_file_path_list, data_dir_path, region):

    print('\n----- make_df -----')
    ## Load in spike measure csv file ##
    spike_measure_path = data_dir_path + 'spike_measures' + '_' + region + '.csv'
    if os.path.exists(spike_measure_path):
        print('Loading spike measures csv file: ' + spike_measure_path)
        df_spk_msr = pd.read_csv(spike_measure_path, sep=',')
        if 'index' in df_spk_msr.keys():
            df_spk_msr = df_spk_msr.set_index('index')

    df = pd.DataFrame()
    count = 0
    stim_file_path = []

    for path in np.sort(spike_file_path_list):

        print(path)
        sort_file_basename = os.path.basename(path)
        ## Use path list to determine which .dat file path to open ##
        spike_file_name = os.path.basename(path)
        fid = spike_file_name[3:7]
        temp_path = glob.glob(data_dir_path + fid + '*.dat')[0]

        if stim_file_path != temp_path:
            stim_file_path = temp_path
            stim = load_stimsequence(stim_file_path)

        f_name = os.path.basename(path)
        print('Loading sorted spikes from: ' + f_name)
        labels, assigns, trials, spike_times, waves, nsamp, nchan, ids, nunit, unit_type = load_spike_file(path)

        for k, unit in enumerate(ids):
            print('unit: ' + str(unit))
            df_temp = pd.DataFrame()

            for con in np.unique(stim):
                spk_time_list = []

                for ind in np.where(stim == con)[0]:
                    spk_times_bool = sp.logical_and(trials == ind+1, assigns == unit)
                    spk_time_list.append(spike_times[spk_times_bool])
                if con < 10:
                    constr = '0' + str(int(con))
                else:
                    constr = str(int(con))

                df_temp['cond' + constr] = pd.Series([spk_time_list])

            df = df.append(df_temp,ignore_index=True)

            if unit_type[k] == 3 and count == 0 and 'depth' not in df.keys() and 'cell_type' not in df.keys():
                df.insert(0, 'depth', np.nan)
                df.insert(1, 'cell_type', 'MU')
                df.insert(2, 'wave_duration', np.nan)
                df.insert(3, 'wave_ratio', np.nan),
                df.insert(4, 'mean_waves', np.nan),
                df.insert(5, 'std_waves', np.nan),
                df['spike_file'] = spike_file_name
                df['unit_id'] = unit
            elif unit_type[k] == 3: #if multi-unit add to df with no other information
                df.loc[[count],'cell_type']     = 'MU'
                df.loc[[count],'depth']         = np.nan
                df.loc[[count],'wave_duration'] = np.nan
                df.loc[[count],'wave_ratio']    = np.nan
                df.loc[[count],'mean_waves']    = np.nan
                df.loc[[count],'std_waves']     = np.nan
                df.loc[[count],'spike_file']    = spike_file_name
                df.loc[[count],'unit_id']       = unit

            elif unit_type[k] < 3:
                unit_id = sort_file_basename + '-u' + str(unit).zfill(3)
                unit_ids = df_spk_msr['unit_id'].tolist()
                unit_index = unit_ids.index(unit_id)
                if count == 0 and 'depth' not in df.keys() and 'cell_type' not in df.keys():
                    df.insert(0,'depth',df_spk_msr.loc[unit_index,'depth'])
                    df.insert(1,'cell_type',df_spk_msr.loc[unit_index,'cell_type'])
                    df.insert(2,'wave_duration',df_spk_msr.loc[unit_index,'wave_duration'])
                    df.insert(3,'wave_ratio',df_spk_msr.loc[unit_index,'wave_ratio'])
                    df.insert(4,'mean_waves',df_spk_msr.loc[unit_index,'mean_waves'])
                    df.insert(5,'std_waves',df_spk_msr.loc[unit_index,'std_waves'])
                    df['spike_file'] = spike_file_name
                    df['unit_id'] = unit
                else:
                    df.loc[[count],'cell_type']     = df_spk_msr.loc[unit_index,'cell_type']
                    df.loc[[count],'depth']         = df_spk_msr.loc[unit_index,'depth']
                    df.loc[[count],'wave_duration'] = df_spk_msr.loc[unit_index,'wave_duration']
                    df.loc[[count],'wave_ratio']    = df_spk_msr.loc[unit_index,'wave_ratio']
                    df.loc[[count],'mean_waves']    = df_spk_msr.loc[unit_index,'mean_waves']
                    df.loc[[count],'std_waves']     = df_spk_msr.loc[unit_index,'std_waves']
                    df.loc[[count],'spike_file']    = spike_file_name
                    df.loc[[count],'unit_id']       = unit

            count += 1

        keys = df.keys().tolist()
        keys.remove('depth'),keys.insert(1,'depth')
        keys.remove('wave_duration'),keys.insert(2,'wave_duration')
        keys.remove('wave_ratio'),keys.insert(3,'wave_ratio')
        df = df[keys]

    return df

def get_depth(tt,bad_tt_chan,best_chan,exp_info,region):

    # tt is the tetrode number which is not zero based indexed. tt1 corresponds to the bottommost contact
    tt_ind = tt -1

    if region == 'vM1':
        tip_depth = exp_info['e1_depth']
        probe     = exp_info['e1_probe']
    elif region == 'vS1':
        tip_depth = exp_info['e2_depth']
        probe     = exp_info['e2_probe']

    if probe == 'a1x16':
        depth_mat = np.abs(np.arange(0,400,25)-tip_depth).reshape(4,4)
        depth_vec = depth_mat[tt_ind,:]
    elif probe == 'a1x32':
        depth_mat = np.abs(np.arange(0,800,25)-tip_depth).reshape(8,4)
        depth_vec = depth_mat[tt_ind,:]

    if bad_tt_chan is None:
        depth = depth_vec[best_chan]
    else:
        if best_chan >= bad_tt_chan:
            depth = depth_vec[best_chan + 1]
        else:
            depth = depth_vec[best_chan]

    return depth


def spike_measures2csv(data_dir_path,sorted_spikes_dir_path,region,overwrite=False):

    print('\n----- spike_measure2csv -----')
    ## Load in spike measure csv file ##
    spike_measure_path = data_dir_path + 'spike_measures' + '_' + region + '.csv'
    if os.path.exists(spike_measure_path):
        print('Loading spike measures csv file: ' + spike_measure_path)
        df_spk_msr = pd.read_csv(spike_measure_path, sep=',') #read csv file, index_col = 0 gets rid of the column of indices
        if 'index' in df_spk_msr.keys():
            df_spk_msr = df_spk_msr.set_index('index')
    else:
        raise Exception('File not found: ' + spike_measure_path + '\n Create empty csv file first with column names then continue')


    # Load in experiment details csv file
    experiment_details_path = data_dir_path + 'experiment_details.csv'
    print('Loading experiment details file: ' + experiment_details_path)
    df_exp_det_regular_index = pd.read_csv(experiment_details_path,sep=',')
    df_exp_det_fid_index = df_exp_det_regular_index.set_index('fid')
    print(df_exp_det_fid_index)

    sort_file_paths = np.sort(glob.glob(sorted_spikes_dir_path +'*' + region + '*.mat'))
    fid   = []
    count = 0

    for sort_file in sort_file_paths:
        sort_file_basename = os.path.basename(sort_file)

        fid_match = re.search(r'fid\d{0,4}',sort_file_basename)
        fid_temp = int(fid_match.group()[3::])

        if fid != fid_temp:
            fid = fid_temp

            if fid not in df_exp_det_fid_index.index:
                raise Exception('fid' + str(fid) + ' not found in experiment details csv file.\n\
                                Please update file and try again.')
            else:
                print('Loading experiment info for fid: ' + str(fid))
                exp_info = df_exp_det_fid_index.loc[fid]

        print('Loading sorted spike data file: ' + sort_file_basename)
        # loads in matlab structure as python objects
        # makes syntax for getting data more similar to matlab's structure syntax
        labels, assigns, trials, spike_times, waves, nsamp, nchan, ids, nunit, unit_type = load_spike_file(sort_file)

        # Get tetrode information--used for calculating unit depth
        tt_match = re.search(r'tt\d{0,1}',sort_file_basename)
        tt = int(tt_match.group()[-1])

        # Get location of bad contact if there was one--used for calculating unit depth
        match = re.search(r'\d{0,2}_\d{0,2}_\d{0,2}_\d{0,2}.mat',sort_file_basename)
        tt_str = os.path.splitext(match.group())[0]
        tt_list = tt_str.split('_')

        if '00' in tt_list:
            bad_tt_chan = tt_list.index('00')
        else:
            bad_tt_chan = None

            # Loop through units from sorted spike file
        print(ids)
        print('\n')
        for k, unit_id in enumerate(ids):

            if unit_type[k] < 3:

                # If the unit measure is not in the measure csv file OR overwrite is True
                # The sorted spike file will be loaded and measures will be calculated
                csv_index_name = sort_file_basename + '-u' + str(unit_id).zfill(3)
                unit_ids = df_spk_msr['unit_id'].tolist()
                if csv_index_name not in unit_ids or overwrite is True:

                    print('Working on: ' + csv_index_name)
                    # csv_index_list.append(csv_index_name)
                    spike_bool     = assigns == unit_id
                    wave_temp      = waves[spike_bool]
                    mean_wave_temp = np.mean(wave_temp,axis=0)
                    std_wave_temp  = np.std(wave_temp,axis=0)
                    wave_min       = np.min(mean_wave_temp,axis=0)
                    best_chan      = np.argmin(wave_min)

                    # Upsample via cubic spline interpolation and calculate waveform measures from
                    # upsampled waveforms. This gives a more precise, less obviously discretized measurements
                    newsamp = nsamp*4
                    xnew = np.linspace(0,nsamp-1,newsamp)
                    f = interp1d(range(nsamp),mean_wave_temp[:,best_chan],kind='cubic')
                    ynew = f(xnew)
                    min_index  = np.argmin(ynew)
                    max_index1 = np.argmax(ynew[(23*4+1):-1])+23*4+1
                    max_index0 = np.argmax(ynew[0:(23*4+1)])
                    min_value  = ynew[min_index]
                    max_value1 = ynew[max_index1]
                    max_value0 = ynew[max_index0]
                    duration   = (max_index1-min_index)/(30.0*4+1)
                    wave_ratio = (max_value1 - max_value0)/(max_value0 + max_value1)

                    # Append depth, wave duration, and wave ratio to respective lists
                    depth = get_depth(tt,bad_tt_chan,best_chan,exp_info,region)
                    dur   = duration
                    ratio = wave_ratio
                    # cell_type_list.append(np.nan)

                    # Add mean and standard deviation of best waveform to respective numpy arrays
                    if count == 0:
                        mean_waves = mean_wave_temp[:,best_chan].reshape(1,nsamp)
                        std_waves  = std_wave_temp[:,best_chan].reshape(1,nsamp)
                    else:
                        mean_waves = np.concatenate((mean_waves,mean_wave_temp[:,best_chan].reshape(1,nsamp)),axis=0)
                        std_waves  = np.concatenate((std_waves,std_wave_temp[:,best_chan].reshape(1,nsamp)),axis=0)

                    if csv_index_name not in unit_ids:
                        temp_df = pd.DataFrame({'unit_id': csv_index_name,
                                                'depth': depth,
                                                'cell_type': np.nan,
                                                'wave_duration': dur,
                                                'wave_ratio': ratio,
                                                'mean_waves': pd.Series([mean_wave_temp[:,best_chan]]),
                                                'std_waves': pd.Series([std_wave_temp[:,best_chan]])})
                        df_spk_msr = df_spk_msr.append(temp_df,ignore_index=True)
                    elif csv_index_name in unit_ids and overwrite is True:
                        #print('IS THIS WHATS WORKING')
                        df_spk_msr = df_spk_msr.drop(unit_ids.index(csv_index_name))
                        df_spk_msr.index = range(len(df_spk_msr))
                        temp_df = pd.DataFrame({'unit_id': csv_index_name,
                                                'depth': depth,
                                                'cell_type': np.nan,
                                            'wave_duration': dur,
                                            'wave_ratio': ratio,
                                            'mean_waves': pd.Series([mean_wave_temp[:,best_chan]]),
                                            'std_waves': pd.Series([std_wave_temp[:,best_chan]])})
                    cols = ['unit_id','depth','cell_type','wave_duration','wave_ratio','mean_waves','std_waves']
                    df_spk_msr = df_spk_msr.append(temp_df,ignore_index=True)

                count += 1

    print('\n ----- \n')

    cols = ['unit_id','depth','cell_type','wave_duration','wave_ratio','mean_waves','std_waves']
    df_spk_msr = df_spk_msr[cols]
    df_spk_msr = df_spk_msr.sort(columns='unit_id')

    print('saving csv measure file: ' + spike_measure_path)
    df_spk_msr.to_csv(spike_measure_path, sep=',')

def classify_units(data_dir_path,region):

    print('\n----- classify_units function -----')
    ## Load in spike measure csv file ##
    spike_measure_path = data_dir_path + 'spike_measures' + '_' + region + '.csv'
    if os.path.exists(spike_measure_path):
        print('Loading spike measures csv file: ' + spike_measure_path)
        df_spk_msr = pd.read_csv(spike_measure_path, sep=',')
        if 'index' in df_spk_msr.keys():
            df_spk_msr = df_spk_msr.set_index('index')

    dur_array   = df_spk_msr['wave_duration'].values
    ratio_array = df_spk_msr['wave_ratio'].values
    dur_array.resize(len(dur_array),1)
    ratio_array.resize(len(ratio_array),1)
    dur_ratio_array = np.concatenate((dur_array, ratio_array),axis=1)

    ## GMM Clustering
    clf = mixture.GMM(n_components=2, covariance_type='full')
    clf.fit(dur_ratio_array)
    pred_prob = clf.predict_proba(dur_ratio_array)
    gmm_means = clf.means_

    if gmm_means[0,0] < gmm_means[1,0] and gmm_means[0,1] > gmm_means[1,1]:
        pv_index = 0
        rs_index = 1
    else:
        pv_index = 1
        rs_index = 0

    ## Assign PV or RS label to a unit if it has a 0.90 probability of belonging
    ## to a group otherwise label it as UC for unclassified
    cell_type_list = []
    for val in pred_prob:
        if val[pv_index] >= 0.90:
            cell_type_list.append('PV')
        elif val[rs_index] >= 0.90:
            cell_type_list.append('RS')
        else:
            cell_type_list.append('UC')

    df_spk_msr['cell_type'] = cell_type_list

    print('saving csv measure file: ' + spike_measure_path)
    df_spk_msr.to_csv(spike_measure_path, sep=',')

    pv_bool = np.asarray([True if x is 'PV' else False for x in cell_type_list])
    rs_bool = np.asarray([True if x is 'RS' else False for x in cell_type_list])
    uc_bool = np.asarray([True if x is 'UC' else False for x in cell_type_list])

    fig = plt.subplots()
    plt.scatter(dur_ratio_array[pv_bool,0],dur_ratio_array[pv_bool,1],color='r',label='PV')
    plt.scatter(dur_ratio_array[rs_bool,0],dur_ratio_array[rs_bool,1],color='g',label='RS')
    plt.scatter(dur_ratio_array[uc_bool,0],dur_ratio_array[uc_bool,1],color='k',label='UC')
    plt.xlabel('duration (ms)')
    plt.ylabel('ratio')
    plt.legend(loc='upper right')
    plt.show()

def convert2array(string):
    '''
    Convert weird string produced by CSV file to a numpy array.
    It removes the brackets and other non-numerical or numerical
    related characters.
    '''
    return np.array([float(ss) for ss in string[1:-1].split()])

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

def calc_evoked_firing_rate(list_of_time_stamp_arrays,trials_ran_bool=None,base_start=0,
        base_stop=1.0,stim_start=1.25,stim_stop=2.25):
    '''
    array of nans used to collect spike counts. Trials in which the
    animal was not running will remain nan otherwise it will equal
    0 for no spikes or the number of spikes.
    '''
    num_trials = len(list_of_time_stamp_arrays)
    base_counts = np.empty((num_trials,1))
    stim_counts = np.empty((num_trials,1))
    base_counts.fill(np.nan)
    stim_counts.fill(np.nan)

    for i in range(num_trials):
        time_stamps = list_of_time_stamp_arrays[i]
        if trials_ran_bool is None:
            base_counts[i] = sum((time_stamps >= base_start) & (time_stamps < base_stop))
            stim_counts[i] = sum((time_stamps >= stim_start) & (time_stamps < stim_stop))
        elif trials_ran_bool[i] is True:
            base_counts[i] = sum((time_stamps >= base_start) & (time_stamps < base_stop))
            stim_counts[i] = sum((time_stamps >= stim_start) & (time_stamps < stim_stop))

    base_rate = base_counts/(base_stop - base_start)
    stim_rate = stim_counts/(stim_stop - stim_start)
    evok_rate = stim_rate - base_rate
    mean_evok_rate = np.nanmean(evok_rate)
    std_evok_rate  = np.nanstd(evok_rate)
    std_evok_rate  = std_evok_rate/np.sqrt(sum(trials_ran_bool))

    return mean_evok_rate, std_evok_rate

def calc_absolute_firing_rate(list_of_time_stamp_arrays,trials_ran_bool=None,base_start=0,
        base_stop=1.0,stim_start=1.25,stim_stop=2.25):
    '''
    array of nans used to collect spike counts. Trials in which the
    animal was not running will remain nan otherwise it will equal
    0 for no spikes or the number of spikes.
    '''
    num_trials = len(list_of_time_stamp_arrays)
    stim_counts = np.empty((num_trials,1))
    stim_counts.fill(np.nan)

    for i in range(num_trials):
        time_stamps = list_of_time_stamp_arrays[i]
        if trials_ran_bool is None:
            stim_counts[i] = sum((time_stamps >= stim_start) & (time_stamps < stim_stop))
        elif trials_ran_bool[i] is True:
            stim_counts[i] = sum((time_stamps >= stim_start) & (time_stamps < stim_stop))

    stim_rate = stim_counts/(stim_stop - stim_start)
    mean_stim_rate = np.nanmean(stim_rate)
    std_stim_rate  = np.nanstd(stim_rate)
    std_stim_rate  = std_stim_rate/np.sqrt(sum(trials_ran_bool))

    return mean_stim_rate, std_stim_rate

def remove_non_modulated_units(df, trials_ran_dict, base_start=0, base_stop=1.0, stim_start=1.50, stim_stop=2.50):
    cond_match = re.findall(r'cond\d{0,2}', str(df.keys().tolist()))
    num_cond = len(cond_match)
    units2drop = list()

    for unit in range(df.shape[0]):
        p_val_list = list()
        for cond in range(num_cond):
            list_of_time_stamp_arrays = df[cond_match[cond]][unit]
            num_trials = len(list_of_time_stamp_arrays)
            base_counts = np.empty((num_trials,1))
            stim_counts = np.empty((num_trials,1))
            if trials_ran_dict is None:
                for i in range(len(num_trials)):
                    time_stamps = list_of_time_stamp_arrays[i]
                    base_counts[i] = sum((time_stamps >= base_start) & (time_stamps < base_stop))
                    stim_counts[i] = sum((time_stamps >= stim_start) & (time_stamps < stim_stop))
            else:
                for i in range(len(list_of_time_stamp_arrays)):
                    time_stamps = list_of_time_stamp_arrays[i]
                    base_counts[i] = sum((time_stamps >= base_start) & (time_stamps < base_stop))
                    stim_counts[i] = sum((time_stamps >= stim_start) & (time_stamps < stim_stop))
            p_val = sp.stats.ranksums(base_counts, stim_counts)[1]
            p_val_list.append(p_val)
        if np.sum(np.asarray(p_val_list) < 0.01) < 1:
            units2drop.append(unit)

#    df_evoked = df.iloc[units2keep, :]
    df_evoked = df.drop(units2drop)
    df_evoked.index = range(len(df_evoked))

    return df_evoked

def make_evoke_rate_array(df,trials_ran_dict=None,base_start=0 ,base_stop=1.0,stim_start=1.25,stim_stop=2.25):

    cond_match = re.findall(r'cond\d{0,2}', str(df.keys().tolist()))
    num_cond = len(cond_match)
    evoke_array_mean = np.zeros((df.shape[0],num_cond))
    evoke_array_std  = np.zeros((df.shape[0],num_cond))

    for unit in range(df.shape[0]):
        for cond in range(num_cond):
            if trials_ran_dict is None:
                evoke_array_mean[unit,cond], evoke_array_std[unit,cond] = calc_evoked_firing_rate(df[cond_match[cond]][unit],
                        trials_ran_dict,base_start,base_stop,stim_start,stim_stop)
            else:
                evoke_array_mean[unit,cond], evoke_array_std[unit,cond] = calc_evoked_firing_rate(df[cond_match[cond]][unit],
                        trials_ran_dict[cond_match[cond]],base_start,base_stop,stim_start,stim_stop)

    return evoke_array_mean, evoke_array_std

def make_absolute_rate_array(df,trials_ran_dict=None,base_start=0 ,base_stop=1.0,stim_start=1.25,stim_stop=2.25):

    cond_match = re.findall(r'cond\d{0,2}', str(df.keys().tolist()))
    num_cond = len(cond_match)
    absolute_array_mean = np.zeros((df.shape[0],num_cond))
    absolute_array_std  = np.zeros((df.shape[0],num_cond))

    for unit in range(df.shape[0]):
        for cond in range(num_cond):
            if trials_ran_dict is None:
                absolute_array_mean[unit,cond], absolute_array_std[unit,cond] = calc_absolute_firing_rate(df[cond_match[cond]][unit],
                        trials_ran_dict,base_start,base_stop,stim_start,stim_stop)
            else:
                absolute_array_mean[unit,cond], absolute_array_std[unit,cond] = calc_absolute_firing_rate(df[cond_match[cond]][unit],
                        trials_ran_dict[cond_match[cond]],base_start,base_stop,stim_start,stim_stop)

    return absolute_array_mean, absolute_array_std

def make_tuning_curve(mean_rate_array, std_rate_array,control_pos=7,depth=None,cell_type_list=None,
                                            fig_title=None,share_yax=True):
    '''
    Add different stimulus functionality. That is, if I do whisker trimming or optogenetics for different
    positions wrap the different indices to the first positions
    '''
    pos = range(1,control_pos)
    x_vals = range(1, control_pos+1)
    labels = [str(i) for i in pos]; labels.append('No Contact')
    num_units = mean_rate_array.shape[0]
    num_manipulations = mean_rate_array.shape[1]/control_pos
    unit_count = 0
    for unit in range(num_units):
        line_color = ['k','r','g']
        y = np.zeros((num_manipulations,len(pos)))
        err = np.zeros((num_manipulations,len(pos)))
        yc = np.zeros((num_manipulations,1))
        errc = np.zeros((num_manipulations,1))

        control_pos_count = 1
        count = 0
        # iterate through positions based on what happened there (e.g. laser
        # on/off). Example: if control positions was at position 7 this will
        # iterate from 0, 7, 14, 21,...etc
        for ind in range(0,mean_rate_array.shape[1],control_pos):
            y[count,:] = mean_rate_array[unit,ind:control_pos*control_pos_count-1]
            err[count,:] = std_rate_array[unit,ind:control_pos*control_pos_count-1]
            yc[count,0] = mean_rate_array[unit,control_pos*control_pos_count-1]
            errc[count,0] =std_rate_array[unit,control_pos*control_pos_count-1]
            control_pos_count += 1
            count += 1
        fig = plt.figure()
        plt.tick_params(axis='x', which='both', bottom='off', top='off')
        for i in range(num_manipulations):
            plt.errorbar(pos,y[i,:], yerr=err[i,:], fmt=line_color[i], marker='o', markersize=8.0, linewidth=2)
            plt.errorbar(control_pos, yc[i,0], yerr=errc[i,0], fmt=line_color[i], marker='o', markersize=8.0, linewidth=2)

            if depth is not None and cell_type_list is not None:
                plt.title(str(depth[unit_count]) + 'um' + ' ' + str(cell_type_list[unit_count]),size=16)
        unit_count += 1
        plt.hlines(0, 0, control_pos+1, colors='k', linestyles='dashed')
        plt.xlim(0, control_pos+1)

#        plt.ylim(-2, 12)

        plt.xticks(x_vals, labels)
        plt.xlabel('Bar Position', fontsize=16)
        plt.ylabel('Evoked Spike Rate (Hz)')
        plt.show()

def make_tuning_curves(mean_rate_array, std_rate_array,control_pos=7,depth=None,cell_type_list=None,
                                            fig_title=None,share_yax=True):
    '''
    Makes tuning curve for all units in input array. If the data is from an experiment where
    each position had a different stiulus (e.g. whisker trim, optogenetics) the
    control_pos variable is used to split the data and wrap it using a different color
    for the different stimuli.
    '''
    print('\n----- make_tuning_curves -----')
    # matplotlib.rcParams.update({'font.size': 8})
    pos = range(1,control_pos)
    num_units = mean_rate_array.shape[0]
    if num_units % 5 == 0:
        num_rows = num_units/5
    else:
        num_rows = num_units/5 + 1
    # num_rows = num_units/5 + 1
    num_cols = 4
    num_rows  = 4
    num_manipulations = mean_rate_array.shape[1]/control_pos
    line_color = ['k','r','g']


    unit_count = 0
    plot_count = 0
    for unit in range(num_units):

        if share_yax and plot_count == 0:

            fig,ax = plt.subplots(num_rows,num_cols,sharex=True,sharey=True)
        elif plot_count == 0:
            fig,ax = plt.subplots(num_rows,num_cols,sharex=True,sharey=False)

        y = np.zeros((num_manipulations,len(pos)))
        err = np.zeros((num_manipulations,len(pos)))
        yc = np.zeros((num_manipulations,1))
        errc = np.zeros((num_manipulations,1))

        control_pos_count = 1
        count = 0
        for ind in range(0,mean_rate_array.shape[1],control_pos):
            y[count,:] = mean_rate_array[unit,ind:control_pos*control_pos_count-1]
            err[count,:] = std_rate_array[unit,ind:control_pos*control_pos_count-1]
            yc[count,0] = mean_rate_array[unit,control_pos*control_pos_count-1]
            errc[count,0] =std_rate_array[unit,control_pos*control_pos_count-1]
            control_pos_count += 1
            count += 1

            for i in range(num_manipulations):
                if unit == 0:
                    ax1 = plt.subplot(num_rows,5,unit+1)
                    ax1.set_xticks([])

                    plt.errorbar(pos,y[i,:],yerr=err[i,:],fmt=line_color[i],marker='o',markersize=2.0)
                    plt.errorbar(control_pos,yc[i,0],yerr=errc[i,0],fmt=line_color[i],marker='o',markersize=2.0)

                    plt.plot([0, control_pos+1],[0,0],'--k')
                    plt.xlim(0, control_pos+1)
                    if depth is not None and cell_type_list is not None:
                        plt.title(str(depth[unit_count]) + 'um' + ' ' + str(cell_type_list[unit_count]),size=8)
                        # plt.title('unit: ' + str(unit))
                else:
                    if share_yax:
                        ax2 = plt.subplot(num_rows, num_cols,plot_count+1,sharex=ax1,sharey=ax1)
                    else:
                        ax2 = plt.subplot(num_rows, num_cols,plot_count+1,sharex=ax1)
                        ax2.set_xticks([])

                        plt.errorbar(pos,y[i,:],yerr=err[i,:],fmt=line_color[i],marker='o',markersize=2.0)
                        plt.errorbar(control_pos,yc[i,0],yerr=errc[i,0],fmt=line_color[i],marker='o',markersize=2.0)

                        plt.plot([0, control_pos+1],[0,0],'--k')
                        plt.xlim(0, control_pos+1)
                        if depth is not None and cell_type_list is not None:
                            plt.title(str(depth[unit_count]) + 'um' + ' ' + str(cell_type_list[unit_count]),size=8)
                        # plt.title('unit: ' + str(unit))

        if plot_count == 0 and fig_title is not None:
            plt.suptitle(fig_title)

        unit_count += 1
        plot_count += 1
        if plot_count == (num_cols*num_rows):
            plot_count = 0
            plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.93, wspace=0.11, hspace=0.21)
            plt.show()

def plot_running_subset(trial_time,vel_mat,conversion=False):
    num_rows = 5
    num_cols = 6
    step = vel_mat.shape[0]/30

    if conversion:
        scale_factor = (2*np.pi*6)/360.0 # circumfrence per revolution (with a mouse 6cm away from center) / number of pulses per revolution
        vel_mat = vel_mat*scale_factor

    i = 0
    subplot_count = 1
    fig, ax = plt.subplots(num_rows,num_cols,sharex=True,sharey=True)
    while subplot_count <= 30:
        if i == 0:
            ax1 = plt.subplot(num_rows,num_cols,subplot_count)
            plt.plot(trial_time,vel_mat[i])
            plt.xlim(0, trial_time[-1])
            if conversion:
                plt.ylim(0, 105)
                plt.ylabel('cm/s')
            else:
                plt.ylim(0, 1000)
                plt.ylabel('pulses/s')
        else:
            ax2 = plt.subplot(num_rows,num_cols,subplot_count,sharex=ax1,sharey=ax1)
            plt.plot(trial_time,vel_mat[i])
            plt.xlim(0, trial_time[-1])
            if conversion:
                plt.ylim(0, 105)
            else:
                plt.ylim(0, 1000)

        i += step
        subplot_count += 1

    plt.suptitle('birds eye view of runspeeds across experiment')
    plt.show()

def classify_run_trials(stim,vel_mat,trial_time,stim_start=1.25,stim_stop=2.25,mean_thresh=250,sigma_thresh=150,low_thresh=200,display=False):
    unique_conditions = np.unique(stim)
    trials_per_condition = len(stim)/len(unique_conditions)
    stim_period = (trial_time >= stim_start) & (stim_stop <= 2.50)
    trials_ind_dict = {}
    trials_ran_dict = {}
    mean_vel = []
    sigm_vel = []

    for cond in unique_conditions:
        temp_trial_list = np.where(stim == cond)[0]
        #temp_bool_list = [False]*trials_per_condition
        temp_bool_list = [False]*len(temp_trial_list)
        count = 0
        for trial_ind in temp_trial_list:
            vel = vel_mat[trial_ind][stim_period]
            mean_vel.append(np.mean(vel))
            sigm_vel.append(np.std(vel))
            if np.mean(vel) >= mean_thresh and np.std(vel) <= sigma_thresh and (sum(vel <= low_thresh)/len(vel)) <= 0.1:
                temp_bool_list[count] = True
            count += 1

            if cond < 10:
                trials_ind_dict['cond0' + str(int(cond))] = temp_trial_list
                trials_ran_dict['cond0' + str(int(cond))] = temp_bool_list
            else:
                trials_ind_dict['cond' + str(int(cond))] = temp_trial_list
                trials_ran_dict['cond' + str(int(cond))] = temp_bool_list

    if display:
        bins = range(0,1000,5)
        fig, ax = plt.subplots()
        plt.subplot(2,1,1)
        mean_counts, _ = np.histogram(mean_vel,bins)
        plt.bar(bins[:-1],mean_counts,width=5.0)
        plt.subplot(2,1,2)
        sigm_counts, _ = np.histogram(sigm_vel,bins)
        plt.bar(bins[:-1],sigm_counts,width=5.0)
        plt.show()

    return trials_ind_dict,trials_ran_dict

def make_psth(df,trials_ran_dict, start_time=0.0, stop_time=2.5, binsize=0.001):
    cond_keys = np.sort(trials_ran_dict.keys())
    bins = np.arange(start_time, stop_time+binsize, binsize)
    unit_count_list = list()
    num_units = df.shape[0]
    num_bins = len(bins)-1
    t_bins = np.arange(start_time, stop_time, binsize)


    # Build alpha kernel with 25msec resolution
    alpha = 1.0/0.025
    tau   = np.arange(0,1/alpha*10, 0.001)
    alpha_kernel = alpha**2*tau*np.exp(-alpha*tau)
    num_points = num_bins - 1
    t = np.linspace(start_time, stop_time, num_points)

    psth_mean = np.zeros((num_units, num_points, len(cond_keys)))
    psth_ci_h = np.zeros((num_units, num_points, len(cond_keys)))

    for unit in range(num_units):
        # Make count_mat list. Each entry is a binary matrix where each row is a
        # trial and each column is a time point.
        count_mats = list()
        for cond_ind, cond in enumerate(cond_keys):
            trial_inds = np.where(trials_ran_dict[cond])[0]
            count_mat = np.zeros((len(trial_inds), num_bins))
            for i, trial in enumerate(trial_inds):
                spike_times = df[cond][unit][trial]
                counts = np.histogram(spike_times, bins=bins)[0]
                count_mat[i, :] = counts
            count_mats.append(count_mat)
        unit_count_list.append(count_mats)


        for cond_ind, cond in enumerate(cond_keys):
            ### Compute PSTH mean and 95% confidence interval ###
            ### Compute PSTH mean and 95% confidence interval ###
            num_trials = count_mats[cond_ind].shape[0]
            psth_matrix = np.zeros((num_trials, num_points))
            for k in range(num_trials):
                psth_matrix[k, :] = np.convolve(count_mats[cond_ind][k, :], alpha_kernel)[:-alpha_kernel.shape[0]]

            mean_psth = psth_matrix.mean(axis=0)
            se = sp.stats.sem(psth_matrix, axis=0)
            # inverse of the CDF is the percentile function. ppf is the percent point funciton of t.
            h = se*sp.stats.t.ppf((1+0.95)/2.0, num_trials-1) # (1+1.95)/2 = 0.975

            psth_mean[unit, :, cond_ind]   = mean_psth
            psth_ci_h[unit, :, cond_ind]   = h

    return unit_count_list, t_bins, psth_mean, psth_ci_h, t

def plot_psth(psth_mean, psth_ci_h, t, unit=0, trial=0):

    # Select position with greatest response
    f, ax = plt.subplots()
    ax.plot(t, psth_mean[unit, :, trial], 'k')
    ax.fill_between(t, psth_mean[unit, :, trial] - psth_ci_h[unit, :, trial], # blue over S1
            psth_mean[unit, :, trial] + psth_ci_h[unit, :, trial], facecolor='k', alpha=0.3)
    return ax
    ##### UNCOMMENT to plot second stimulus #####
#    plt.plot(t, psth_mean[unit, :, trial+9], 'b')
#    plt.fill_between(t, psth_mean[unit, :, trial+9] - psth_ci_h[unit, :,trial+9], # blue of S1 red over M1
#            psth_mean[unit, :, trial+9] + psth_ci_h[unit, :, trial+9], facecolor='b', alpha=0.3)
#    plt.show()

def make_raster(df,trials_ran_dict, unit=0, start_time=0, stop_time=2.3):
    cond_keys = np.sort(trials_ran_dict.keys())
    #bins = np.arange(start_time, stop_time+binsize, binsize)
    ax = plt.gca()
    num_good_trials = 0

    # delete this
    all_spikes = list()

    for i, cond in enumerate(cond_keys):
        trial_inds = np.where(trials_ran_dict[cond])[0]
        for k, trial in enumerate(trial_inds):
            if k < 10:
                spike_times = df[cond][unit][trial]
                for spike_time in spike_times:
                    #plt.vlines(spike_time, num_good_trials + 2.5, num_good_trials + 1.5, color='k')
                    plt.plot(spike_time, np.ones(1,)*num_good_trials,  '-k.')
                    #all_spikes.append(spike_time)
                num_good_trials += 1
            num_good_trials += 2
#        plt.hlines(num_good_trials, start_time, stop_time, color='k')
        plt.text(0, num_good_trials-10, cond)
    plt.xlim(start_time, stop_time)
    plt.ylim(0.5, num_good_trials + 0.5)

#    bins = np.arange(start_time, stop_time, 0.050)
#    plt.vlines(bins, 0, num_good_trials + 0.5, color='b')

    return ax

def raster_psth_2cond(df, trials_ran_dict, psth_mean, psth_ci_h, t, unit=0, conds=0):
    cond_keys = np.sort(trials_ran_dict.keys())
    pivot = len(cond_keys)/2
    cond_keys = [cond_keys[conds]]#, cond_keys[conds+pivot-1]]
    start_time=0.5
    stop_time=2.7
    #ax = plt.gca()
    f, axarr = plt.subplots(len(cond_keys)+1, 1)
    axarr[0].set_title('unit ' + str(unit))

    # delete this
    all_times_temp = list()

    for i,cond in enumerate(cond_keys):
        num_good_trials = 0
        trial_inds = np.where(trials_ran_dict[cond])[0]
        for trial in trial_inds:
            spike_times = df[cond][unit][trial]
            for spike_time in spike_times:
                #axarr[i].vlines(spike_time, num_good_trials + 0.5, num_good_trials + 1.5, color='k')
                axarr[i].plot(spike_time, np.ones(1,)*num_good_trials - 0.5,  '-k.')

                # delete this
                all_times_temp.append(spike_time)

            num_good_trials += 1
        axarr[i].set_xlim(start_time, stop_time)
        axarr[i].set_ylim(0, num_good_trials + 0.5)

        f.patch.set_visible(False)
        axarr[i].axis('off')

    # PSTH full pad condition
    #axarr[len(cond_keys)].plot(t, psth_mean[unit, :, conds], 'k')
    axarr[len(cond_keys)].hist(all_times_temp, bins=np.arange(0,3.0,0.050), align='left')

#    axarr[len(cond_keys)].fill_between(t, psth_mean[unit, :, conds] - psth_ci_h[unit, :, conds], # blue over S1
#        psth_mean[unit, :, conds] + psth_ci_h[unit, :, conds], facecolor='k', alpha=0.3)

    # PSTH trimmed condition
#    axarr[len(cond_keys)].plot(t, psth_mean[unit, :, conds+pivot-1], 'b')
#    axarr[len(cond_keys)].fill_between(t, psth_mean[unit, :, conds+pivot-1] - psth_ci_h[unit, :,conds+pivot-1], # blue of S1 red over M1
#            psth_mean[unit, :, conds+pivot-1] + psth_ci_h[unit, :, conds+pivot-1], facecolor='b', alpha=0.3)

    # PSTH axis adjustments
    axarr[len(cond_keys)].tick_params(axis=u'both', which=u'both',length=0) # remove ticks
    axarr[len(cond_keys)].set_xlim(start_time, stop_time)
    f.patch.set_visible(False)
    plt.show()


def get_spike_rates(fid, region, conds=(0,7)):

    # Load in data
    usr_dir = os.path.expanduser('~')
    sorted_spikes_dir_path = usr_dir + '/Documents/AdesnikLab/SortedSpikes/'
    fid_region = 'fid' + fid + '_' + region
    sort_file_paths = glob.glob(sorted_spikes_dir_path + fid_region + '*.mat')

    data_dir_path = usr_dir + '/Documents/AdesnikLab/Data/'
    data_dir_paths  = glob.glob(data_dir_path + fid + '*.dat')

    # #Calculate runspeed
    run_mat = load_run_file(data_dir_paths[0]).value
    vel_mat, trial_time = calculate_runspeed(run_mat)

    # # Get stimulus id list
    stim = load_stimsequence(data_dir_paths[0])

    # # Create running trial dictionary
    cond_ind_dict,trials_ran_dict = classify_run_trials(stim,vel_mat,trial_time,stim_start=1.25,stim_stop=2.50,
            mean_thresh=250,sigma_thresh=150,low_thresh=200,display=False)

    # Find the condition with the least number of trials
    min_run_trials  = min([sum(trials_ran_dict[x]) for x in trials_ran_dict.keys()])

    # Put data into a Pandas dataframe
    df = make_df(sort_file_paths,data_dir_path,region=region)

    # plot tuning curves
    depth = df['depth']
    cell_type_list = df['cell_type']
#    em, _ = make_evoke_rate_array(df, trials_ran_dict, base_start=0, base_stop=1.0, stim_start=1.50,
#            stim_stop=2.50)
    am, _ = make_absolute_rate_array(df, trials_ran_dict, base_start=0, base_stop=1.0, stim_start=1.50,
            stim_stop=2.50)
    am = [x[conds[0]:conds[1]] for x in am]

    return am

def get_wt_run_vals(wtmat, trials_ran_dict, cond_ind_dict, vel_mat, trial_time):
    num_trials_ran = [sum(trials_ran_dict[x]) for x in np.sort(trials_ran_dict.keys())]
    num_frames = wtmat['numFrames'][0]
    camTime = wtmat['camTime'][:, 0]
    set_point_list = list()
    amp_list = list()
    num_samples = vel_mat.shape[1]
    vel_list = list()
    ang_list = list()
    for cond_ind, cond in enumerate(np.sort(trials_ran_dict.keys())):
        print(cond)
        temp_setpoint = np.zeros((num_trials_ran[cond_ind], num_frames))
        temp_amp = np.zeros((num_trials_ran[cond_ind], num_frames))
        temp_ang = np.zeros((num_trials_ran[cond_ind], num_frames))
        temp_vel = np.zeros((num_trials_ran[cond_ind], num_samples))
        temp_inds = np.where(np.array(trials_ran_dict[cond]) == True)[0]
        run_inds  = cond_ind_dict[cond][temp_inds]
        print(run_inds)

        # Given a condition iterate through all of its running trials
        good_inds = list()
        for trial_ind,runind in enumerate(run_inds):

            temp_sp = wtmat['setPointMat'][runind]
            temp_a  = wtmat['angleMat'][runind]
            bad_set_point = np.sum(np.logical_or(temp_sp < 0, temp_sp > 200)) > 0
            bad_angle = np.sum(np.logical_or(temp_a < 0, temp_a > 200)) > 0
            if not bad_set_point and not bad_angle:
                good_inds.append(trial_ind)

            temp_setpoint[trial_ind, :] = wtmat['setPointMat'][runind]
            temp_amp[trial_ind, :] = wtmat['ampMat'][runind]
            temp_ang[trial_ind, :] = wtmat['angleMat'][runind]
            temp_vel[trial_ind, :] = vel_mat[runind, :]
        set_point_list.append(temp_setpoint[good_inds, :])
        amp_list.append(temp_amp[good_inds, :])
        vel_list.append(temp_vel[good_inds, :])
        ang_list.append(temp_ang[good_inds, :])

    return camTime, set_point_list, amp_list, vel_list, ang_list

def velocity_analysis_parameters(wtmat, trials_ran_dict, cond_ind_dict, vel_mat, trial_time):
    numframes2bin = 50
    num_trials_ran = [sum(trials_ran_dict[x]) for x in np.sort(trials_ran_dict.keys())]
    num_frames = wtmat['numFrames'][0]
    camTime = wtmat['camTime'][:, 0]
    run_time_ind = [np.argmin(np.abs(x - trial_time)) for x in camTime]
    set_point_list = list()
    amp_list = list()
    num_samples = vel_mat.shape[1]
    vel_list = list()
    for cond_ind, cond in enumerate(np.sort(trials_ran_dict.keys())):
        print(cond)
        temp_setpoint = np.empty(0)
        temp_amp = np.empty(0)
        temp_vel = np.empty(0)
        #temp_inds = np.where(np.array(trials_ran_dict[cond]) == True)[0]
        run_inds  = cond_ind_dict[cond]#[temp_inds]
        print(run_inds)

        # Given a condition iterate through all of its running trials
        for trial_ind,runind in enumerate(run_inds):
            temp_setpoint = np.concatenate((temp_setpoint,
                np.nanmean(wtmat['setPointMat'][runind].reshape(-1, numframes2bin), axis=1)))
            temp_amp = np.concatenate((temp_amp,
                np.nanmean(wtmat['ampMat'][runind].reshape(-1, numframes2bin), axis=1)))
            temp_vel = np.concatenate((temp_vel,
                np.nanmean(vel_mat[runind, run_time_ind].reshape(-1, numframes2bin), axis=1)))
        good_inds = np.where(np.logical_and(temp_setpoint >= 0, temp_setpoint <= 200))
        set_point_list.append(temp_setpoint[good_inds])
        amp_list.append(temp_amp[good_inds])
        vel_list.append(temp_vel[good_inds]) # removes columns

    return camTime, set_point_list, amp_list, vel_list



########## MAIN CODE ##########
########## MAIN CODE ##########

if __name__ == "__main__":
    # Select which experiments to analyze
    fids = ['0871','0872','0873']
    fids = ['1118', '1123']
    fids = ['0872']
    fids = ['1220']

    fids = ['1034','1044','1054','1058','1062'] # whisker traces and sorted spikes
    fids = ['1034'] #44, shit tuning shape, 58 shit, 62 unit 5 nice tuning and psth (5hz evoked).
    region = 'vS1'
    control_pos = 9
    unit_count_list = list()
    ang_dict = dict()

    for fid in fids:
        usr_dir = os.path.expanduser('~')
        sorted_spikes_dir_path = usr_dir + '/Documents/AdesnikLab/SortedSpikes/'
        fid_region = 'fid' + fid + '_' + region
        sort_file_paths = glob.glob(sorted_spikes_dir_path + fid_region + '*.mat')

        data_dir_path = usr_dir + '/Documents/AdesnikLab/Data/'
        data_dir_paths  = glob.glob(data_dir_path + fid + '*.dat')

        # #Calculate runspeed
        run_mat = load_run_file(data_dir_paths[0]).value
        vel_mat, trial_time = calculate_runspeed(run_mat)

        # #Plot runspeed
        # plot_running_subset(trial_time,vel_mat,conversion=True)

        # # Get stimulus id list
        stim = load_stimsequence(data_dir_paths[0])

        # # Create running trial dictionary
        cond_ind_dict,trials_ran_dict = classify_run_trials(stim,vel_mat,trial_time,stim_start=1.5,stim_stop=2.50,
                mean_thresh=175,sigma_thresh=150,low_thresh=200,display=True)
        # easy running thresholds
        #cond_ind_dict,trials_ran_dict = classify_run_trials(stim,vel_mat,trial_time,stim_start=1.25,stim_stop=2.50,
        #        mean_thresh=175,sigma_thresh=150,low_thresh=100,display=True)

        # Find the condition with the least number of trials
        min_run_trials  = min([sum(trials_ran_dict[x]) for x in trials_ran_dict.keys()])

        # Put data into a Pandas dataframe
        df = make_df(sort_file_paths,data_dir_path,region=region)
#        df = remove_non_modulated_units(df, trials_ran_dict, base_start=0, base_stop=1.0, stim_start=1.50, stim_stop=2.50)

        # plot tuning curves
        depth = df['depth']
        cell_type_list = df['cell_type']

        em, es = make_evoke_rate_array(df, trials_ran_dict, base_start=0, base_stop=1.0, stim_start=1.50,
                stim_stop=2.50)
        make_tuning_curves(em, es,depth=depth,cell_type_list=cell_type_list, control_pos=control_pos,
                fig_title='Evoked Firing Rate--fid' + fid + region + ' full pad',
                share_yax=False)
#
        # unit 5 goes down
        # unit 17 goes up

        ##### whisker tracking data #####
        ##### whisker tracking data #####
        vel_mat = vel_mat*(2*np.pi*6)/360.0

#        hsv_mat_path = glob.glob(usr_dir + '/Documents/AdesnikLab/Processed_HSV/FID' + fid + '-data*.mat')[0]
#        wtmat = h5py.File(hsv_mat_path)
#
#
#        [camTime, set_point_vals, amp_vals, vel_vals] = velocity_analysis_parameters(wtmat,
#                trials_ran_dict, cond_ind_dict, vel_mat, trial_time)
#
#        camTime, set_point_list, amp_list, vel_list, ang_list = get_wt_run_vals(wtmat, trials_ran_dict,
#                cond_ind_dict, vel_mat, trial_time)
#
#        ang_dict[fid] = ang_list
#
#        sp_samples_list  = [np.nanmean(x[:, 750:1000], axis=1) for x in set_point_list]
#        sp_samples_list  = [np.delete(x, np.where(np.isnan(x))) for x in sp_samples_list] # remove nans
#
#        amp_samples_list  = [np.nanmean(x[:, 750:1000], axis=1) for x in amp_list]
#        amp_samples_list  = [np.delete(x, np.where(np.isnan(x))) for x in amp_samples_list] # remove nans

        # make PSTHs with 95% CIs
        num_units = df.shape[0]
        unit_count_list_temp, t_bins, psth_mean, psth_ci_h, t = \
                make_psth(df, trials_ran_dict, start_time=0.5, stop_time=2.7, binsize=0.001)
        unit_count_list.extend(unit_count_list_temp)
#
    for unit in range(num_units):
        if unit in [8]:
#            make_tuning_curve(em[unit:unit+1,:], es[unit:unit+1,:],control_pos=9,depth=None,cell_type_list=None,
#                                                fig_title=None,share_yax=True)
#            raster_psth_2cond(df, trials_ran_dict, psth_mean, psth_ci_h,\
#                    t, unit=unit, conds=4)
            make_raster(df,trials_ran_dict, unit=unit, start_time=0.5, stop_time=2.7)

#    mean_full = np.nanmean(angles[0], axis=0)
#    mean_sing = np.nanmean(angles[0+9], axis=0)
#    sem_full  = np.nanstd(angles[0], axis=0)/np.sqrt(angles[0].shape[0])*1.8
#    sem_sing  = np.nanstd(angles[0+9], axis=0)/np.sqrt(angles[0+9].shape[0])*1.8
#
#    f, ax = subplots()
#    ax.plot(camTime, mean_full, 'k')
#    ax.fill_between(camTime, mean_full - sem_full, # blue over S1
#        mean_full + sem_full, facecolor='b', alpha=0.3)
#    ax.plot(camTime, mean_sing, 'k')
#    ax.fill_between(camTime, mean_sing - sem_sing, # blue over S1
#        mean_sing + sem_sing, facecolor='r', alpha=0.3)





