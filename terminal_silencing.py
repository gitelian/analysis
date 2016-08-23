#!/bin/bash
from neuroanalyzer import *
import statsmodels.stats.multitest as smm

def make_psth(m1df, start_time=0.0, stop_time=2.5, binsize=0.001):
    cond_keys = np.sort(m1df.keys())
    bins = np.arange(start_time, stop_time+binsize, binsize)
    unit_count_list = list()
    num_units = m1df.shape[0]
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
            count_mat = np.zeros((len(m1df[cond][unit]), num_bins))
            for i, trial in enumerate(m1df[cond][unit]):
                spike_times = trial
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
                ### CHANGE 0 IN COUNT_MATS TO SOME VARIABLE ###
                ### CHANGE 0 IN COUNT_MATS TO SOME VARIABLE ###
                psth_matrix[k, :] = np.convolve(count_mats[cond_ind][k, :], alpha_kernel)[:-alpha_kernel.shape[0]]

            mean_psth = psth_matrix.mean(axis=0)
            se = sp.stats.sem(psth_matrix, axis=0)
            # inverse of the CDF is the percentile function. ppf is the percent point funciton of t.
            h = se*sp.stats.t.ppf((1+0.95)/2.0, num_trials-1) # (1+1.95)/2 = 0.975

            psth_mean[unit, :, cond_ind]   = mean_psth
            psth_ci_h[unit, :, cond_ind]   = h

    return unit_count_list, t_bins, psth_mean, psth_ci_h, t

def plot_psth(psth_mean, psth_ci_h, unit=0):

#    ylower = np.percentile(psth_mean[unit,:,:] - psth_ci_h[unit,:,:], 1)
#    yupper = np.percentile(psth_mean[unit,:,:] + psth_ci_h[unit,:,:], 99)
    plt.figure()
    ax1 = plt.subplot(1,2,1)
    plt.plot(t, psth_mean[unit, :, 0], 'b')
    plt.fill_between(t, psth_mean[unit, :, 0] - psth_ci_h[unit, :, 0], # blue over S1
            psth_mean[unit, :, 0] + psth_ci_h[unit, :, 0], facecolor='b', alpha=0.3)
    plt.plot(t, psth_mean[unit, :, 1], 'r')
    plt.fill_between(t, psth_mean[unit, :, 1] - psth_ci_h[unit, :, 1], # blue of S1 red over M1
            psth_mean[unit, :, 1] + psth_ci_h[unit, :, 1], facecolor='r', alpha=0.3)
#    plt.ylim(ylower, yupper)


    ax2 = plt.subplot(122, sharey=ax1)
    plt.plot(t, psth_mean[unit, :, 2], 'k')
    plt.fill_between(t, psth_mean[unit, :, 2] - psth_ci_h[unit, :, 2], # nothing
            psth_mean[unit, :, 2] + psth_ci_h[unit, :, 2], facecolor='k', alpha=0.3)
    plt.plot(t, psth_mean[unit, :, 3], 'r')
    plt.fill_between(t, psth_mean[unit, :, 3] - psth_ci_h[unit, :, 3], # red over M1
            psth_mean[unit, :, 3] + psth_ci_h[unit, :, 3], facecolor='r', alpha=0.3)
#   plt.ylim(ylower, yupper)
    plt.show()

    ## HOW TO COMPUTE A BOOTSTRAP CONFIDENCE INTERVAL ##
    ## THE BOOTSTRAP CI IS NEARLY IDENTICAL TO THE PARAMETRIC CI#
#    boot_samples = np.zeros((num_samples, num_points))
#    for k in range(num_samples):
#        rand_inds = np.random.choice(range(num_trials), size=num_trials)
#        boot_samples[k, :] = psth_matrix[rand_inds, :].mean(axis=0) # grab random samples for each time point and calculate the mean
#    lower_ci, upper_ci = np.zeros(num_points,), np.zeros(num_points,)
#    for k in range(num_points):
#        lower_ci[k]  = np.percentile(boot_samples[:, k], 2.5)
#        upper_ci[k]  = np.percentile(boot_samples[:, k], 97.5)
#    plt.fill_between(t, lower_ci,
#            upper_ci, facecolor='k', alpha=0.3)
    return count_mats

def make_raster(df, unit=0, start_time=0, stop_time=2.3):
    cond_keys = np.sort(df.keys())
    ax = plt.gca()
    num_good_trials = 0

    for cond in cond_keys:
        trial_inds = range(len(df[cond][unit]))
        for trial in trial_inds:
            spike_times = df[cond][unit][trial]
            for spike_time in spike_times:
                plt.vlines(spike_time, num_good_trials + 0.5, num_good_trials + 1.5, color='k')
            num_good_trials += 1
        plt.hlines(num_good_trials, start_time, stop_time, color='k')
    plt.xlim(start_time, stop_time)
    plt.ylim(0.5, num_good_trials + 0.5)

    return ax

########## MAIN CODE ##########
########## MAIN CODE ##########

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
    # # Analyze ALL trials
    cond_ind_dict,trials_ran_dict = classify_run_trials(stim,vel_mat,trial_time,stim_start=1.25,stim_stop=2.50,
            mean_thresh=0,sigma_thresh=1000,low_thresh=-1000,display=True)

    # Find the condition with the least number of trials
    min_run_trials  = min([sum(trials_ran_dict[x]) for x in trials_ran_dict.keys()])

    # Put data into a Pandas dataframe
    df = make_df(sort_file_paths,data_dir_path,region=region)

    # Combine columns of data frame
    num_units = df.shape[0]
    m1df = pd.DataFrame()
    count = 0

    for k in range(num_units):
        if k == 0:
            m1df.insert(0, 'control', [df['cond01'][k] + df['cond02'][k]])
            m1df.insert(1, 'red_M1', [df['cond03'][k] + df['cond04'][k]])
            m1df.insert(2, 'blue_S1', [df['cond05'][k] + df['cond06'][k]])
            m1df.insert(3, 'both_lights', [df['cond07'][k] + df['cond08'][k]])
        else:
            m1df.loc[count, 'control'] =     df['cond01'][k] + df['cond02'][k]
            m1df.loc[count, 'red_M1']  =     df['cond03'][k] + df['cond04'][k]
            m1df.loc[count, 'blue_S1'] =     df['cond05'][k] + df['cond06'][k]
            m1df.loc[count, 'both_lights'] = df['cond07'][k] + df['cond08'][k]
        count += 1

    # make PSTHs with 95% CIs
    num_units = m1df.shape[0]
    unit_count_list, t_bins, psth_mean, psth_ci_h, t = \
            make_psth(m1df, start_time=0.0, stop_time=2.5, binsize=0.001)
    for unit in range(num_units):
        plot_psth(psth_mean, psth_ci_h, unit)

    # do statistics
    inds = np.where(np.logical_and(t > 1.25, t < 1.4))[0]
    pvals = np.ones((num_units,))
    omi = np.zeros((num_units,))
    for unit in range(num_units):
        cond01_counts = unit_count_list[unit][1][:, inds].sum(axis=1)
        cond00_counts = unit_count_list[unit][0][:, inds].sum(axis=1)
        pvals[unit] = sp.stats.mannwhitneyu(cond01_counts, cond00_counts)[1]

        omi[unit] = (cond01_counts.mean() - cond00_counts.mean())/\
                (cond01_counts.mean() + cond00_counts.mean())

    rej, pval_corr = smm.multipletests(pvals, alpha=0.05, method='simes-hochberg')[:2]
    print( 'num of units where terminal silencing had a significant effect: ' + str(sum(rej)/float(len(rej))))

    # plot OMI and test OMI
    plt.figure()
    counts, bins, _ = plt.hist(omi, bins=np.arange(-1, 1.05, 0.05), align='left')
    print(sp.stats.wilcoxon(omi))


#    ##### To Store Data #####
#    ##### To Store Data #####
#    store = pd.HDFStore('fid0871_m1_s1_data.h5')
#    store['m1'] = df
#    run_df = pd.DataFrame.from_dict(trials_ran_dict)
#    # To Read Data
#    test_df = pd.DataFrame['name_of_file.h5')
#    m1_df = test_df['m1']


















