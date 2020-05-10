import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import ruptures as rpt


# save wt data in a temp pickle file
# detect change points
# save and open in original analysis script

# import pickle

## using pickle across python2 and python3 (UGHHHH!!!!!)

# Python2 save pickle
    # pickle.dump(variable, open('name.p', 'wb')
# Python 2 load pickle
    # pickle.load( open('name.p', 'rb') ) # for python3

# Python3 load Python2 pickle
    # pickle.load( open('name.p', 'rb'), encoding='latin1') # for python3
# Python3 save pickle for Python2
    # pickle.dump(variable, open('name.p', 'wb'), 2)


########## MAIN CODE ##########
########## MAIN CODE ##########

if __name__ == "__main__":
    # input: wt matrix, and what to analyze (e.g. 1 for setpoint)
    # output: list of nx2 matrices. row corresponds to trial and col1 and col2
    # correspond to the time of the first and second changepoint

    # how to grap inputs
    #sys.argv[1]

    temp_dir = '/home/greg/code/analysis/temp/'

    # load data
    p = pickle.load( open(temp_dir + 'fid2147_wt_data.p', 'rb'), encoding='latin1')
    wt  = p[0]
    wtt = p[1]

    # good trial example
    data = wt[11][:, 0, 0]

    # Pelt search method
    model = "rbf"
    algo = rpt.Pelt(model=model).fit(data)
    result = algo.predict(pen=10)
    rpt.display(data, result)

    # Binary segmentation search method
    model = "l2"
    algo = rpt.Binseg(model=model).fit(data)
    my_bkps = algo.predict(n_bkps=3, pen=10)
    rpt.display(data, my_bkps)

    # Window method (width=285 SO FAR BEST WITH M1LIGHT DATA)
    model= "l2"
    #algo = rpt.Window(width=285, model=model).fit(data)
    algo = rpt.Window(width=100, model=model).fit(data)
    my_bkps = algo.predict(n_bkps=2)
    rpt.display(data, my_bkps)

    # Dynamic programming
    # so far the best for both nolight and m1light
    model = "l2"
    algo =  rpt.Dynp(model=model, min_size=100, jump=5).fit(data)
    my_bkps = algo.predict(n_bkps=2)
    rpt.display(data, my_bkps)


    ### run model to predict breakpoints on all data ###
    bkps = [list() for x in range(18)]

    for condi, trial_data in enumerate(wt):
        num_trials = trial_data.shape[2]
        temp_bkps = np.zeros((num_trials, 2))

        for k in range(num_trials):
            model = "l2"
            algo =  rpt.Dynp(model=model, min_size=100, jump=5).fit(trial_data[:, 0, k])
            my_bkps = algo.predict(n_bkps=2)

            temp_bkps[k, :] = np.asarray(my_bkps[0:2])

        bkps[condi] = temp_bkps
        del temp_bkps





###    ### manually mark change points to get ground truth ###
###    ### manually mark change points to get ground truth ###
###
###    ground_truth = [list() for x in range(18)]
###    fig, ax = plt.subplots(1,1, figsize=[18.4, 6.47])
###
###    #for condi, trial_data in zip(range(8,18), wt_temp):
###    for condi, trial_data in enumerate(wt):
###        num_trials = trial_data.shape[2]
###        temp_data = np.zeros((num_trials, 2))
###        for k in range(num_trials):
###            ax.set_title('condition {}, trial {}'.format(condi, k))
###            ax.plot(wtt, trial_data[:, 0, k])
###            ax.set_ylim(70, 150)
###            fig.canvas.draw_idle()
###            pts = fig.ginput(n=2, timeout=0, show_clicks=True, mouse_add=1, mouse_pop=3, mouse_stop=2)
###            ax.clear()
###
###            temp_data[k, :] = np.asarray([pts[0][0], pts[1][0]])
###
###        ground_truth[condi] = temp_data
###        del temp_data
###        # save after each iteration
###        pickle.dump(ground_truth, open(temp_dir + 'fid2147_ground_truth.p', 'wb'), 2)






    ### make summary plots of ground truth ###
    ### make summary plots of ground truth ###
    gt = pickle.load(open(temp_dir + 'fid2147_ground_truth.p', 'rb'))

    ### convert indices to timestamps
    temp = [list() for x in range(18)]
    for k in range(18):
        temp[k] = np.asarray([wtt[bkps[k][:, 0].astype(int)], wtt[bkps[k][:, 1].astype(int)]]).T

    gt = temp


    cmap_nolight = plt.cm.coolwarm(np.linspace(0, 1, 8))
    cmap_m1light = plt.cm.PiYG(np.linspace(0, 1, 8))
    bins = np.arange(-1.25, 1.0, 0.02)

    # get all data for summary histograms
    all_nolight = np.zeros((1,2))
    all_m1light = np.zeros((1,2))
    for x in range(8):
        all_nolight = np.concatenate((all_nolight, gt[x]))
        all_m1light = np.concatenate((all_m1light, gt[x+9]))
    all_nolight = all_nolight[1::, :]
    all_m1light = all_m1light[1::, :]

    fig, ax = plt.subplots(2,2)

    # make CDFs
    for k in range(8):

        ## CDFs
        # nolight onset
        ax[0][0].hist(gt[k][:, 0], bins=bins, edgecolor=cmap_nolight[k], fill=False, linewidth=2.2,\
                density=True, histtype='step', cumulative=True, alpha=0.75)

        # m1light onset
        ax[0][0].hist(gt[k+9][:, 0], bins=bins, edgecolor=cmap_m1light[k], fill=False, linewidth=2.2,\
                density=True, histtype='step', cumulative=True, alpha=0.75)

        ##
        # nolight offset
        ax[1][0].hist(gt[k][:, 1], bins=bins, edgecolor=cmap_nolight[k], fill=False, linewidth=2.2,\
                density=True, histtype='step', cumulative=True, alpha=0.75)

        # m1light offsett
        ax[1][0].hist(gt[k+9][:, 1], bins=bins, edgecolor=cmap_m1light[k], fill=False, linewidth=2.2,\
                density=True, histtype='step', cumulative=True, alpha=0.75)


    # all onsets/offsets for nolight and m1light
    ax[0][1].set_title('Distribution of all retraction onset times')
    ax[0][1].hist(all_nolight[:, 0], bins=bins, edgecolor=None, fill=True,\
            color='tab:blue', alpha=0.5, label='retraction onset, nolight')
    ax[0][1].hist(all_m1light[:, 0], bins=bins, edgecolor=None, fill=True,\
            color='tab:red', alpha=0.5, label='retraction onset, m1light')
    ax[0][1].legend(loc='top right')

    ax[1][1].set_title('Distribution of all retraction stop times')
    ax[1][1].hist(all_nolight[:, 1], bins=bins, edgecolor=None, fill=True,\
            color='tab:blue', alpha=0.5, label='retraction stop, nolight')
    ax[1][1].hist(all_m1light[:, 1], bins=bins, edgecolor=None, fill=True,\
            color='tab:red', alpha=0.5, label='retraction stop, m1light')
    ax[1][1].legend(loc='top left')


    # labels
    ax[0][0].set_title('CDF of whisking retraction onset times\nBlue - Red (no light), Purple - Green (m1 light)')
    ax[1][0].set_title('whisking retraction stop')
    ax[1][0].set_xlabel('time from object stop (s)')
    ax[1][1].set_xlabel('time from object stop (s)')

    fig.suptitle('FID2147 ground truth retraction start and stop times', size=16)




    ### correlations ###
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(all_nolight[:, 0], rp_nolight[:, 0])
    ax[0].axis('equal')
    ax[0].set(xlim=(-1, 1), ylim=(-1, 1))
    ax[0].set_aspect('equal', 'box')

    ax[1].scatter(all_nolight[:, 1], rp_nolight[:, 1])
    ax[1].axis('equal')
    ax[1].set(xlim=(-1, 1), ylim=(-1, 1))
    ax[1].set_aspect('equal', 'box')



#    sns.jointplot(x=all_nolight[:, 0], y=rp_nolight[:, 0])
#    sns.regplot(x=all_nolight[:, 0], y=rp_nolight[:, 0], fit_reg=True)















































