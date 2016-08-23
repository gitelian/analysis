#!/bin/bash
from neuroanalyzer import *

def get_spike_rates(fid, region):

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

    return am

########## MAIN CODE ##########
########## MAIN CODE ##########

if __name__ == "__main__":

#    em_m1 = get_evoked_spike_rates('0872', 'vM1')
#    em_s1 = get_evoked_spike_rates('0872', 'vS1')

    fids = ['0871','0872','0873']
    fids = ['1118','1123']
    m1_a = list(); #s1_a = list()
    srt=0
    stp=6
    num_stim=6

    for fid in fids:
        m1_a.append(get_spike_rates(fid, 'vM1'))
        #s1_a.append(get_spike_rates(fid, 'vS1'))


    pos_m1 = list()
    sel_m1  = list()
    for am_m1 in m1_a:
        temp = list()
        temp_sel = list()
        for tuning_curve in am_m1:
            pos = np.arange(1, num_stim+1)
            #compute preferred position
            temp.append(np.sum(pos*tuning_curve[srt:stp])/np.sum(tuning_curve[srt:stp]))

            #compute selectivity
            temp_sel.append(1 - (np.linalg.norm(tuning_curve[srt:stp]/np.max(tuning_curve[srt:stp])) - 1)/ \
                    (np.sqrt(num_stim) -1))

            # Old max() based preference method
            #temp.append(np.argmax(np.abs(tuning_curve[0:-1])))
            #temp_sel.append(np.max(np.abs(tuning_curve[0:-1]))/(np.mean(np.abs(tuning_curve[0:-1]))))
        temp_norm_weights = temp_sel/np.sum(temp_sel)
        weighted_mean = np.sum(temp_norm_weights*temp)
        pos_m1.extend(temp)# - np.median(temp))
        #pos_m1.extend(temp - weighted_mean)
        sel_m1.extend(temp_sel)

    pos_s1 = list()
    sel_s1  = list()
    for am_s1 in s1_a:
        temp = list()
        temp_sel = list()
        for tuning_curve in am_s1:
            pos = np.arange(1, num_stim+1)
            #compute preferred position
            temp.append(np.sum(pos*tuning_curve[str:stp])/np.sum(tuning_curve[str:stp]))

            #compute selectivity
            temp_sel.append(1 -
            (np.linalg.norm(tuning_curve[str:stp]/np.max(tuning_curve[str:stp])) - 1)/ \
                    (np.sqrt(num_stim) -1))

#            temp.append(np.argmax(np.abs(tuning_curve[0:-1])))
#            temp_sel.append(np.max(np.abs(tuning_curve[0:-1]))/(np.mean(np.abs(tuning_curve[0:-1]))))
        temp_norm_weights = temp_sel/np.sum(temp_sel)
        weighted_mean = np.sum(temp_norm_weights*temp)
        #pos_s1.extend(temp - np.median(temp))
        pos_s1.extend(temp - weighted_mean)
        sel_s1.extend(temp_sel)

    sns.set_style("white")
    sns.set_context("talk", font_scale=1.25)
    sns.set_palette("muted")
    bins = np.arange(-5,5,0.1)
    #fig = plt.subplots(2,1)
    plt.figure(figsize=(6,8))
    plt.subplot(2,1,2)
    m1_dist = plt.hist(pos_m1, bins, align='left', normed=True)[0]
    plt.xlim(-2, 2)
    plt.ylim(0,3)
    plt.title('vM1')
    plt.ylabel('PDF')

#    plt.subplot(2,1,2)
#    s1_dist = plt.hist(pos_s1, bins, align='left', normed=True)[0]
#    plt.xlim(-2, 2)
#    plt.ylim(0,3)
#    sns.despine()
#    plt.title('vS1')
#    plt.ylabel('PDF')
#    plt.xlabel('Position')
#    plt.suptitle('Preferred Position', size=24)
    plt.show()

    # Plot modulation index
    sel_bins = np.arange(0,1,0.025)
    #fig = plt.subplots(2,1)
    plt.figure(figsize=(6,8))
    plt.subplot(2,1,1)
    m1_sel_dist = plt.hist(sel_m1, sel_bins, normed=True)[0]
    plt.xlim(0,1)
    plt.ylim(0,5)
    plt.title('vM1')
    plt.ylabel('PDF')

#    plt.subplot(2,1,2)
#    s1_weights = np.ones_like(sel_s1)/len(sel_s1)
#    s1_sel_dist = plt.hist(sel_s1, sel_bins, normed=True)[0]
#    plt.xlim(0,1)
#    plt.ylim(0,5)
#    sns.despine()
#    plt.show()
#    plt.title('vS1')
#    plt.ylabel('PDF')
#    plt.xlabel('Selectivity Index')
    plt.suptitle('Selectivity Index', size=24)

    # sort by positions
    pos_m1 = np.asarray(pos_m1)
    sel_m1 = np.asarray(sel_m1)
    pos_s1 = np.asarray(pos_s1)
    sel_s1 = np.asarray(sel_s1)

    m1_sort_ind = np.argsort(pos_m1)
    s1_sort_ind = np.argsort(pos_s1)

    pos_m1_sort = pos_m1[m1_sort_ind]
    sel_m1_sort = sel_m1[m1_sort_ind]
    pos_s1_sort = pos_s1[s1_sort_ind]
    sel_s1_sort = sel_s1[s1_sort_ind]

    sel_m1_norm = sel_m1_sort/np.sum(sel_m1_sort)
    sel_s1_norm = sel_s1_sort/np.sum(sel_s1_sort)

    plt.figure()
    plt.scatter()


    cond1 = np.asarray([pos_nolight, sel_nolight]).T
    cond2 = np.asarray([pos_m1, sel_m1]).T
    def pos_preference_bootstrap(cond1, cond2, num_samples=15000):
        prb1 = cond1[:, 1]/np.sum(cond1[:, 1])
        prb2 = cond2[:, 1]/np.sum(cond2[:, 1])
        exp1 = np.sum(prb1*cond1[:, 0])
        exp2 = np.sum(prb2*cond2[:, 0])
        var1 = sum(prb1*(cond1[:, 0]**2)) - exp1**2
        var2 = sum(prb2*(cond2[:, 0]**2)) - exp2**2
        s_diff = var1 - var2

        n1 = cond1.shape[0]
        n2 = cond2.shape[0]
        inds_array = np.arange(n1+n2)
        null_dist = np.zeros((num_samples, 1))
        alldata = np.concatenate([cond1, cond2], axis=0)

        for k in np.arange(num_samples):

            rand_inds1 = np.random.choice(inds_array, size=n1, replace=True)
            samp1 = alldata[rand_inds1, :]

            rand_inds2 = np.random.choice(inds_array, size=n2, replace=True)
            samp2 = alldata[rand_inds2, :]

            prbsamp1 = samp1[:, 1]/np.sum(samp1[:, 1])
            prbsamp2 = samp2[:, 1]/np.sum(samp2[:, 1])
            expsamp1 = np.sum(prbsamp1*samp1[:, 0])
            expsamp2 = np.sum(prbsamp2*samp2[:, 0])
            varsamp1 = np.sum(prbsamp1*(samp1[:,0]**2)) - expsamp1**2
            varsamp2 = np.sum(prbsamp2*(samp2[:,0]**2)) - expsamp2**2
            null_dist[k, 0] = varsamp1 - varsamp2

        p = (np.sum(null_dist > np.abs(s_diff)) + np.sum(null_dist < - np.abs(s_diff)))/float(num_samples)
        return s_diff, p
