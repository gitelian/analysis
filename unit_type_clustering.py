
for unit_id in ids:

    # If the unit measure is not in the measure csv file OR overwrite is True
    # The sorted spike file will be loaded and measures will be calculated
    csv_index_name = sort_file_basename + '-u' + str(unit_id).zfill(3)
    unit_ids = df_spk_msr['unit_id'].tolist() # Same unit IDs from the UMS spike structure

    spike_bool     = assigns == unit_id # get index for unit_id
    wave_temp      = waves[spike_bool]  # use index to grab the waveforms for unit_id
    mean_wave_temp = np.mean(wave_temp,axis=0) # calculate the mean waveform for unit_id
    std_wave_temp  = np.std(wave_temp,axis=0)  # calculate the standard deviation for unit_id
    wave_min       = np.min(mean_wave_temp,axis=0) # find the "peak" of each of the 4 waveforms for unit_id
    best_chan      = np.argmin(wave_min)           # find the index of the largest waveform. All compuations
                                                   # will be performed on the largest waveform

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
    duration   = (max_index1-min_index)/(30.0*4+1) # Calculate the duration of the waveform
    wave_ratio = (max_value1 - max_value0)/(max_value0 + max_value1) # Calculate the ratio of the waveform
    # See Reyes-Puerta et al. (from Luhmann lab) Cerebral Cortex 2014 for
    # details

    # Append depth, wave duration, and wave ratio to respective lists
    ##### Save this data to a matrix or something that you can feed into the next function
    ##### The next function uses wave duration and ratio information to classify
    ##### each unit as regular spiking (RS), fast spiking (FS), or unclassified (UC)
    depth = get_depth(tt,bad_tt_chan,best_chan,exp_info,region)
    dur   = duration
    ratio = wave_ratio

def classify_units(data_dir_path,region):

    print('\n----- classify_units function -----')
    ## Load in spike measure csv file ##
    ##### LOAD IN DATA STRUCTURE MADE ABOVE WITH ALL OF YOUR UNITS DURATION AND
    ##### RATIO INFORMATION
    spike_measure_path = data_dir_path + 'spike_measures' + '_' + region + '.csv'
    if os.path.exists(spike_measure_path):
        print('Loading spike measures csv file: ' + spike_measure_path)
        df_spk_msr = pd.read_csv(spike_measure_path, sep=',')
        if 'index' in df_spk_msr.keys():
            df_spk_msr = df_spk_msr.set_index('index')

    dur_array   = df_spk_msr['wave_duration'].values # Put all duration values in a vector
    ratio_array = df_spk_msr['wave_ratio'].values    # Pul all ratio values in a vector
    dur_array.resize(len(dur_array),1)
    ratio_array.resize(len(ratio_array),1)
    dur_ratio_array = np.concatenate((dur_array, ratio_array),axis=1)

    ## GMM Clustering on duration and ratio data points
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
