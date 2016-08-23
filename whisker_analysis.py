#!/bin/bash
from neuroanalyzer import *
import statsmodels.stats.multitest as smm

if __name__ == "__main__":
    # Select which experiments to analyze
    #fids = ['0871','0872','0873']
    fid = '1125'
    region = 'vM1'

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
            mean_thresh=175,sigma_thresh=150,low_thresh=200,display=True)
    # easy running thresholds
    #cond_ind_dict,trials_ran_dict = classify_run_trials(stim,vel_mat,trial_time,stim_start=1.25,stim_stop=2.50,
    #        mean_thresh=175,sigma_thresh=150,low_thresh=100,display=True)

    # Find the condition with the least number of trials
    min_run_trials  = min([sum(trials_ran_dict[x]) for x in trials_ran_dict.keys()])

    # Put data into a Pandas dataframe
    df = make_df(sort_file_paths,data_dir_path,region=region)
    df = remove_non_modulated_units(df, trials_ran_dict, base_start=0, base_stop=1.0, stim_start=1.50, stim_stop=2.50)
    count_mats = bin_data(df,trials_ran_dict, start_time=0, stop_time=2.5, binsize=0.001)
#    plt.figure()
#    plt.imshow(count_mats[4], interpolation='nearest', aspect='auto')

    # plot tuning curves
    depth = df['depth']
    cell_type_list = df['cell_type']

    em, es = make_evoke_rate_array(df, trials_ran_dict, base_start=0, base_stop=1.0, stim_start=1.20,
            stim_stop=2.00)
    make_tuning_curves(em, es,depth=depth,cell_type_list=cell_type_list, control_pos=7,
            fig_title='Evoked Firing Rate--fid' + fid + region + ' full pad',
            share_yax=False)
    make_blah()
###############################################################################
##################### Analyze Whisker Tracking Data ###########################
###############################################################################
    hsv_mat_path = glob.glob(usr_dir + '/Documents/AdesnikLab/Processed_HSV/FID' + fid + '-data*.mat')[0]
    wtmat = h5py.File(hsv_mat_path)
    # example: this returns the angle values for trial 7 wtmat['angleCell'][6]


    num_trials_ran = [sum(trials_ran_dict[x]) for x in np.sort(trials_ran_dict.keys())]
    num_frames = wtmat['numFrames'][0]
    camTime = wtmat['camTime'][:, 0]
    set_point_list = list()
    amp_list = list()
    ang_list = list()
    for cond_ind, cond in enumerate(np.sort(trials_ran_dict.keys())):
        print(cond)
        temp_setpoint = np.zeros((num_trials_ran[cond_ind], num_frames))
        temp_amp = np.zeros((num_trials_ran[cond_ind], num_frames))
        temp_ang = np.zeros((num_trials_ran[cond_ind], num_frames))
        temp_inds = np.where(np.array(trials_ran_dict[cond]) == True)[0]
        run_inds  = cond_ind_dict[cond][temp_inds]
        print(run_inds)

        # Given a condition iterate through all of its running trials
        for trial_ind,runind in enumerate(run_inds):
            temp_setpoint[trial_ind, :] = wtmat['setPointMat'][runind]
            temp_amp[trial_ind, :] = wtmat['ampMat'][runind]
            temp_ang[trial_ind, :] = wtmat['angleMat'][runind]
        set_point_list.append(temp_setpoint)
        amp_list.append(temp_amp)
        ang_list.append(temp_ang)

    positions = 9
    f, ax = plt.subplots()
    for x in range(positions):
        set_mean_pretrim = np.nanmean(set_point_list[x], axis=0)
        set_std_pretrim  = np.nanstd(set_point_list[x], axis=0)
        set_mean_posttrim = np.nanmean(set_point_list[x+9], axis=0)
        set_std_posttrim  = np.nanstd(set_point_list[x+9], axis=0)

        plt.subplot(2, 5, x+1)
        plt.plot(camTime, set_mean_pretrim, 'b', wtmat['camTime'], set_mean_posttrim, 'r')
        plt.fill_between(camTime, set_mean_pretrim + set_std_pretrim,
                set_mean_pretrim - set_std_pretrim, facecolor='b', alpha=0.3)
        plt.fill_between(camTime, set_mean_posttrim + set_std_posttrim,
                set_mean_posttrim - set_std_posttrim, facecolor='r', alpha=0.3)
        plt.plot(camTime,np.transpose(set_point_list[x]), 'b', alpha=0.7)
        plt.plot(camTime,np.transpose(set_point_list[x+9]), 'r', alpha=0.7)
        plt.ylim(80, 170)
#        plt.ylim(120, 160)

#        amp_mean_pretrim  = np.nanmean(amp_list[x], axis=0)
#        amp_std_pretrim   = np.nanstd(amp_list[x], axis=0)/np.sqrt(amp_list[x].shape[0])
#        amp_mean_posttrim = np.nanmean(amp_list[x+9], axis=0)
#        amp_std_posttrim  = np.nanstd(amp_list[x+9], axis=0)/np.sqrt(amp_list[x+9].shape[0])
#
#        plt.subplot(2, positions, x+1+positions)
#        plt.plot(camTime, amp_mean_pretrim, 'b', wtmat['camTime'][0], amp_mean_posttrim, 'r')
#        plt.fill_between(camTime, amp_mean_pretrim + amp_std_pretrim,
#                amp_mean_pretrim - amp_std_pretrim, facecolor='b', alpha=0.5)
#        plt.fill_between(camTime, amp_mean_posttrim + amp_std_posttrim,
#                amp_mean_posttrim - amp_std_posttrim, facecolor='r', alpha=0.5)
#        plt.ylim(0, 25)
    plt.show()

###############################################################################
##################### spike triggered whisker parameters ######################
###############################################################################
    start_time = 1.5;
    stop_time = 2.5;
    num_units = df.shape[0]
    conditions = np.sort(trials_ran_dict.keys())
    num_trials_ran = [sum(trials_ran_dict[x]) for x in conditions]

    set_point_sta = list()
    amp_sta = list()
    phase_sta = list()
    angle_sta = list()

    set_point_cum = list()
    amp_cum = list()
    phase_cum = list()
    angle_cum = list()
    total_trials = list()
    run_cum = list()

    camInds = np.logical_and(camTime >= start_time, camTime <= stop_time)
    camIndsControl = range(num_frames)
    velRunInds = np.logical_and(trial_time >= start_time, trial_time <= stop_time)
    velRunIndsControl = np.logical_and(trial_time >= wtmat['startTime'][0], trial_time <= wtmat['stopTime'][0])

    # Iterate through all units in data frame
    for unit_ind in range(num_units):
        print('working on unit ' + str(unit_ind))
        unit_set_point = list()
        unit_amp = list()
        unit_phase = list()
        unit_angle = list()

        # Iterate through all conditions for a given unit
        for i, cond in enumerate(conditions):
            print('condition ' + str(i) + ' of ' + str(len(conditions)))
            cond_set_point = list()
            cond_amp = list()
            cond_phase = list()
            cond_angle = list()

            if unit_ind == 0:
                cum_cond_set_point = np.empty((1,1))
                cum_cond_amp = np.empty((1,1))
                cum_cond_phase = np.empty((1,1))
                cum_cond_angle = np.empty((1,1))
                cum_cond_run = np.empty((1,1))


            # Get running indices for the given condition
            cond_inds = np.where(np.array(trials_ran_dict[cond]) == True)[0]
            # Get trial index for grabbing the right hsv data
            run_inds  = cond_ind_dict[cond][cond_inds]

            # Iterate through all running trials
            for trial_ind, runind in enumerate(run_inds):
                print('running trial ' + str(trial_ind) + ' of ' + str(len(run_inds)))

#                st = df[cond][unit_ind][cond_inds[trial_ind]]
#                if (i+1)%9 != 0:
#                    st = st[np.logical_and(st >= start_time, st <= stop_time)]
#                else:
#                    st = st[np.logical_and(st >= wtmat['startTime'][0], st <= wtmat['stopTime'][0])]

#                # Iterate through all spike times
#                for spike_time in st:
#                    # Get hsv index closest to spike time
#                    hsv_ind = np.argmin(np.abs(camTime - spike_time))
#                    # Add hsv data corresponding to that spike time
#                    cond_set_point.append(wtmat['setPointMat'][runind][hsv_ind])
#                    cond_amp.append(wtmat['ampMat'][runind][hsv_ind])
#                    cond_phase.append(wtmat['phaseMat'][runind][hsv_ind])
#                    cond_angle.append(wtmat['angleMat'][runind][hsv_ind])

                # Get whisker values during analysis time for histogram
                # normalization
                if (i+1)%9 != 0  and unit_ind == 0:
                    cum_cond_set_point = np.append(cum_cond_set_point, wtmat['setPointMat'][runind][camInds])
                    cum_cond_amp = np.append(cum_cond_amp, wtmat['ampMat'][runind][camInds])
                    cum_cond_phase = np.append(cum_cond_phase, wtmat['phaseMat'][runind][camInds])
                    cum_cond_angle = np.append(cum_cond_angle, wtmat['angleMat'][runind][camInds])

                    cum_cond_run = np.append(cum_cond_run, vel_mat[runind,velRunInds])
                elif (i+1)%9 == 0  and unit_ind == 0:
                    cum_cond_set_point = np.append(cum_cond_set_point, wtmat['setPointMat'][runind][camIndsControl])
                    cum_cond_amp = np.append(cum_cond_amp, wtmat['ampMat'][runind][camIndsControl])
                    cum_cond_phase = np.append(cum_cond_phase, wtmat['phaseMat'][runind][camIndsControl])
                    cum_cond_angle = np.append(cum_cond_angle, wtmat['angleMat'][runind][camIndsControl])

                    cum_cond_run = np.append(cum_cond_run, vel_mat[runind,velRunIndsControl])

            # Add condition data to unit lists
            unit_set_point.append(cond_set_point)
            unit_amp.append(cond_amp)
            unit_phase.append(cond_phase)
            unit_angle.append(cond_angle)

            # Add cummulative whisker info to list
            if unit_ind == 0:
                set_point_cum.append(cum_cond_set_point)
                amp_cum.append(cum_cond_amp)
                phase_cum.append(cum_cond_phase)
                angle_cum.append(cum_cond_angle)
                total_trials.append(trial_ind+1)
                print(cum_cond_run)
                run_cum.append(cum_cond_run)

#        # Add unit data to sta lists
#        set_point_sta.append(unit_set_point)
#        amp_sta.append(unit_amp)
#        phase_sta.append(unit_phase)
#        angle_sta.append(unit_angle)

##### run speed distribution per trial #####
    pos = 9
    binsize = 5
    bin_start = 0
    bin_stop  = 1250
    run_bins = np.arange(bin_start,bin_stop, binsize)
    run_pre_trim_hist = np.histogram(run_cum[pos-1], run_bins)[0]/total_trials[pos-1]
    run_post_trim_hist = np.histogram(run_cum[pos-1+9], run_bins)[0]/total_trials[pos-1+9]
    run_max_ylim = np.nanmax(np.append(run_pre_trim_hist, run_post_trim_hist))
    plt.figure()
    plt.subplot(2,1,1)
    plt.bar(run_bins[:-1], run_pre_trim_hist, binsize);plt.xlim(bin_start, bin_stop)
    plt.ylim(0, run_max_ylim)
    plt.subplot(2,1,2)
    plt.bar(run_bins[:-1], run_post_trim_hist, binsize);plt.xlim(bin_start, bin_stop)
    plt.ylim(0, run_max_ylim)
    plt.show()

##### set-point spike triggered averages  #####
    pos = 9
    binsize = 2
    bin_start = 120
    bin_stop  = 160

    for unit in range(len(set_point_sta)):
        bins = np.arange(bin_start, bin_stop, binsize)
        set_point_sta_hist = np.histogram(set_point_sta[unit][pos-1], bins)[0]
        set_point_cum_hist = np.histogram(set_point_cum[pos-1], bins)[0]*0.002
        set_point_norm = set_point_sta_hist/set_point_cum_hist

        set_point_sta_hist_trim = np.histogram(set_point_sta[unit][pos-1+9], bins)[0]
        set_point_cum_hist_trim = np.histogram(set_point_cum[pos-1+9], bins)[0]*0.002
        set_point_norm_trim = set_point_sta_hist_trim/set_point_cum_hist_trim

        sta_max = np.nanmax(np.append(set_point_sta_hist, set_point_sta_hist_trim))
        cum_max = np.nanmax(np.append(set_point_cum_hist, set_point_cum_hist_trim))
        nor_max = np.nanmax(np.append(set_point_norm, set_point_norm_trim))

        plt.figure()

        plt.subplot(3,2,1)
        plt.bar(bins[:-1], set_point_sta_hist, binsize);plt.xlim(bin_start, bin_stop)
        plt.ylim(0, sta_max)
        plt.title('Pre-trim')
        plt.ylabel('spikes/bin')
        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        plt.subplot(3,2,3)
        plt.bar(bins[:-1], set_point_cum_hist, binsize);plt.xlim(bin_start, bin_stop)
        plt.ylim(0, cum_max)
        plt.ylabel('seconds/bin')
        plt.subplot(3,2,5)
        plt.bar(bins[:-1], set_point_norm, binsize);plt.xlim(bin_start, bin_stop)
        plt.ylim(0, nor_max)
        plt.tick_params(axis='x', which='both', top='off')
        plt.ylabel('spike rate (Hz)')
        plt.xlabel('set-point (deg)')

        plt.subplot(3,2,2)
        plt.bar(bins[:-1], set_point_sta_hist_trim, binsize);plt.xlim(bin_start, bin_stop)
        plt.ylim(0, sta_max)
        plt.title('Post-trim')
        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        plt.subplot(3,2,4)
        plt.bar(bins[:-1], set_point_cum_hist_trim, binsize);plt.xlim(bin_start, bin_stop)
        plt.ylim(0, cum_max)
        plt.subplot(3,2,6)
        plt.bar(bins[:-1], set_point_norm_trim, binsize);plt.xlim(bin_start, bin_stop)
        plt.ylim(0, nor_max)
        plt.tick_params(axis='x', which='both', top='off')
        plt.xlabel('set-point (deg)')

        plt.show()

##### distribution of time spent in each set-point bin #####
    pos = 9
    for pos in range(1,10):
        binsize = 2
        bin_start = 100
        bin_stop  = 160
        bins = np.arange(bin_start, bin_stop, binsize)
        set_point_cum_hist = np.histogram(set_point_cum[pos-1], bins)[0]*0.002
        set_point_cum_hist_trim = np.histogram(set_point_cum[pos-1+9], bins)[0]*0.002
        set_point_max = np.nanmax(np.append(set_point_cum_hist/total_trials[pos-1],set_point_cum_hist_trim/total_trials[pos-1+9]))
#        plt.figure()
        plt.subplot(2,9,pos)
        plt.bar(bins[:-1], set_point_cum_hist/total_trials[pos-1], binsize);plt.xlim(bin_start, bin_stop)
        plt.ylim(0, set_point_max)
        plt.title('Pre-trim' + ' cond ' + str(pos))
        plt.ylabel('seconds/(bin*trial)')
        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        plt.subplot(2,9,pos+9)
        plt.bar(bins[:-1], set_point_cum_hist_trim/total_trials[pos-1+9], binsize);plt.xlim(bin_start, bin_stop)
        plt.ylim(0, set_point_max)
        plt.title('Post-trim')
        plt.xlabel('set-point (deg)')
        plt.ylabel('seconds/(bin*trial)')

    plt.show()


########################## set-point dist overlay #############################
########################## run speed dist overlay #############################
    binsize = 2
    bin_start = 90
    bin_stop  = 150
    runbinsize = 50
    run_bin_start = 0
    run_bin_stop  = 1250
    bins = np.arange(bin_start, bin_stop, binsize)
    run_bins = np.arange(run_bin_start,run_bin_stop, runbinsize)
    plt.figure()
    for pos in range(1,10):
        set_point_cum_hist = np.histogram(set_point_cum[pos-1], bins)[0]*0.002
        set_point_cum_hist_trim = np.histogram(set_point_cum[pos-1+9], bins)[0]*0.002
        set_point_max = np.nanmax(np.append(set_point_cum_hist/total_trials[pos-1],set_point_cum_hist_trim/total_trials[pos-1+9]))
        plt.subplot(2,9,pos)
        plt.plot(bins[:-1], set_point_cum_hist/total_trials[pos-1], 'b');plt.xlim(bin_start, bin_stop)
        plt.ylim(0, set_point_max)
        plt.ylabel('seconds/(bin*trial)')
        plt.plot(bins[:-1], set_point_cum_hist_trim/total_trials[pos-1+9],'r');plt.xlim(bin_start, bin_stop)
        plt.ylim(0, set_point_max + set_point_max*0.1)
        plt.xlabel('set-point (deg)')

        run_pre_trim_hist = np.histogram(run_cum[pos-1], run_bins)[0]/total_trials[pos-1]
        run_post_trim_hist = np.histogram(run_cum[pos-1+9], run_bins)[0]/total_trials[pos-1+9]
        run_max_ylim = np.nanmax(np.append(run_pre_trim_hist, run_post_trim_hist))
        plt.subplot(2,9,pos+9)
        plt.plot(run_bins[:-1], run_pre_trim_hist,'b');plt.xlim(run_bin_start, run_bin_stop)
        plt.plot(run_bins[:-1], run_post_trim_hist,'r');plt.xlim(run_bin_start, run_bin_stop)
        plt.ylim(0, run_max_ylim + run_max_ylim*0.1)
    plt.show()
########################## amplitude #################################
########################## amplitude #################################

##### amplitude spike triggered averages  #####
    pos = 9
    binsize = 1
    bin_start = 0
    bin_stop  = 25

    for unit in range(len(amp_sta)):
        bins = np.arange(bin_start, bin_stop, binsize)
        amp_sta_hist = np.histogram(amp_sta[unit][pos-1], bins)[0]
        amp_cum_hist = np.histogram(amp_cum[pos-1], bins)[0]*0.002
        amp_norm = amp_sta_hist/amp_cum_hist

        amp_sta_hist_trim = np.histogram(amp_sta[unit][pos-1+9], bins)[0]
        amp_cum_hist_trim = np.histogram(amp_cum[pos-1+9], bins)[0]*0.002
        amp_norm_trim = amp_sta_hist_trim/amp_cum_hist_trim

        sta_max = np.nanmax(np.append(amp_sta_hist, amp_sta_hist_trim))
        cum_max = np.nanmax(np.append(amp_cum_hist, amp_cum_hist_trim))
        nor_max = np.nanmax(np.append(amp_norm, amp_norm_trim))

        plt.figure()

        plt.subplot(3,2,1)
        plt.bar(bins[:-1], amp_sta_hist, binsize);plt.xlim(bin_start, bin_stop)
        plt.ylim(0, sta_max)
        plt.title('Pre-trim')
        plt.ylabel('spikes/bin')
        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        plt.subplot(3,2,3)
        plt.bar(bins[:-1], amp_cum_hist, binsize);plt.xlim(bin_start, bin_stop)
        plt.ylim(0, cum_max)
        plt.ylabel('seconds/bin')
        plt.subplot(3,2,5)
        plt.bar(bins[:-1], amp_norm, binsize);plt.xlim(bin_start, bin_stop)
        plt.ylim(0, nor_max)
        plt.tick_params(axis='x', which='both', top='off')
        plt.ylabel('spike rate (Hz)')
        plt.xlabel('amp (deg)')

        plt.subplot(3,2,2)
        plt.bar(bins[:-1], amp_sta_hist_trim, binsize);plt.xlim(bin_start, bin_stop)
        plt.ylim(0, sta_max)
        plt.title('Post-trim')
        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        plt.subplot(3,2,4)
        plt.bar(bins[:-1], amp_cum_hist_trim, binsize);plt.xlim(bin_start, bin_stop)
        plt.ylim(0, cum_max)
        plt.subplot(3,2,6)
        plt.bar(bins[:-1], amp_norm_trim, binsize);plt.xlim(bin_start, bin_stop)
        plt.ylim(0, nor_max)
        plt.tick_params(axis='x', which='both', top='off')
        plt.xlabel('amp (deg)')

        plt.show()

##### distribution of time spent in each amplitude bin #####
    pos = 9
    binsize = 1
    bin_start = 0
    bin_stop  = 25
    bins = np.arange(bin_start, bin_stop, binsize)
    for pos in range(1,10):
        amp_cum_hist = np.histogram(amp_cum[pos-1], bins)[0]*0.002
        amp_cum_hist_trim = np.histogram(amp_cum[pos-1+9], bins)[0]*0.002
        amp_max = np.nanmax(np.append(amp_cum_hist/total_trials[pos-1],amp_cum_hist_trim/total_trials[pos-1+9]))
#        plt.figure()
        plt.subplot(2,9,pos)
        plt.bar(bins[:-1], amp_cum_hist/total_trials[pos-1], binsize);plt.xlim(bin_start, bin_stop)
        plt.ylim(0, amp_max)
        plt.title('Pre-trim')
        plt.ylabel('seconds/(bin*trial)')
        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        plt.subplot(2,9,pos+9)
        plt.bar(bins[:-1], amp_cum_hist_trim/total_trials[pos-1+9], binsize);plt.xlim(bin_start, bin_stop)
        plt.ylim(0, amp_max)
        plt.title('Post-trim')
        plt.xlabel('amplitude (deg)')
        plt.ylabel('seconds/(bin*trial)')

    plt.show()

############################
    unit = 15
    pos = 9
    binsize = 0.05
    bin_start = -np.pi
    bin_stop  =  np.pi
    bins = np.arange(bin_start, bin_stop, binsize)
    phase_sta_hist = np.histogram(phase_sta[unit][pos-1], bins)[0]
    phase_cum_hist = np.histogram(phase_cum[pos-1], bins)[0]*0.002
    phase_norm = phase_sta_hist/phase_cum_hist

    plt.subplot(3,1,1)
    plt.bar(bins[:-1], phase_sta_hist, binsize)
    plt.xlim(bin_start, bin_stop)
    plt.subplot(3,1,2)
    plt.bar(bins[:-1], phase_cum_hist, binsize)
    plt.xlim(bin_start, bin_stop)
    plt.subplot(3,1,3)
    plt.bar(bins[:-1], phase_norm, binsize)
    plt.xlim(bin_start, bin_stop)

































