from neuroanalyzer import *
from hdnet.stimulus import Stimulus
from hdnet.spikes import Spikes
from hdnet.spikes_model import SpikeModel

if __name__ == "__main__":
    # Select which experiments to analyze
    fid = '0873'
    region = 'vS1'

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
    cond_ind_dict,trials_ran_dict = classify_run_trials(stim,vel_mat,trial_time,stim_start=1.25,stim_stop=2.50,
            mean_thresh=400,sigma_thresh=150,low_thresh=200,display=True)
    # easy running thresholds
    #cond_ind_dict,trials_ran_dict = classify_run_trials(stim,vel_mat,trial_time,stim_start=1.25,stim_stop=2.50,
    #        mean_thresh=175,sigma_thresh=150,low_thresh=100,display=True)


    # Put data into a Pandas dataframe
    df = make_df(sort_file_paths,data_dir_path,region=region)
    binsize=0.001
    count_mats = bin_data(df,trials_ran_dict, start_time=0, stop_time=2.5, binsize=binsize)

    for k in range(len(count_mats)):
        temp = count_mats[k]
        temp[temp > 0] = 1
        count_mats[k] = temp


#    cond = 13
#    plt.figure()
#    plt.imshow(count_mats[cond], interpolation='nearest', aspect='auto')
#    plt.show()

    cond = 3-1
    plt.matshow(count_mats[cond],cmap='gray',aspect='auto')
    plt.title('matshow counts')

    plt.figure()
    time = np.arange(0, count_mats[cond].shape[1]*binsize, binsize)
    for i in range(count_mats[cond].shape[0]):
        print('now on unit' + str(i))
        for k in range(count_mats[cond].shape[1]):
            if count_mats[cond][i, k] > 0:
                plt.vlines(time[k], i + 0.5, i + 1.5, color='k')
    plt.xlim(0, time[-1])
    plt.xlabel('time (s)')
    plt.ylabel('unit')
    plt.show()

##### Train the model #####
    spikes = Spikes(spikes=count_mats[cond], bin_size=0.500)
    spikes_model = SpikeModel(spikes=spikes)#, window_size=1)
    spikes_model.fit()  # note: this fits a single network to all trials
    spikes_model.chomp()

    # What is this supposed to do???
    #converged_spikes = Spikes(spikes=spikes_model.hopfield_spikes)

    plt.matshow(spikes_model.hopfield_spikes.rasterize(), cmap='gray',aspect='auto')
    plt.title('Converge dynamics on Raw data')
    #plt.matshow(converged_spikes.rasterize(), cmap='gray')

#    plt.matshow(spikes_model.hopfield_spikes.covariance().reshape((2 * 10, 10)), cmap='gray')
#    plt.title('Covariance of converged memories')

