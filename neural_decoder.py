from neuroanalyzer import *
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA
from sklearn import cross_validation

def make_tuning_curve(mean_rate_array, std_rate_array,control_pos=7,depth=None,cell_type_list=None,
                                            fig_title=None,share_yax=True):
    '''
    Add different stimulus functionality. That is, if I do whisker trimming or optogenetics for different
    positions wrap the different indices to the first positions
    '''
    pos = range(1,control_pos)
    x_vals = range(1, control_pos+1)
    labels = [str(i) for i in pos]; labels.append('NC')
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
        #fig = plt.figure()
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
        #plt.xlabel('Bar Position', fontsize=16)
        plt.ylabel('Evoked Spike Rate (Hz)')
        #plt.show()

def bin_data(df,trials_ran_dict,min_run_trials,stim_start=1.0,stim_stop=2.5,bin_size=0.005):

    dict_keys = np.sort(trials_ran_dict.keys())
    num_units = df.shape[0]
    dt        = stim_stop - stim_start
    bins      = np.arange(stim_start,stim_stop+bin_size,bin_size)
    num_bins  = len(bins) - 1
    print('\n----- bin_data function -----')
    print('not using last {0}ms of data'.format(str((dt - num_bins*bin_size))))
    print('num units: {0}\nnum trials: {1}\nnum bins: {2}'.format(str(num_units),str(min_run_trials),str(num_bins)))

    mat_list = []

    for cond in dict_keys:
        bin_mat = np.zeros((num_units,num_bins,min_run_trials))
        cond_bool = np.array(trials_ran_dict[cond])
        good_trial_inds = np.where(cond_bool == True)[0]
        for unit_ind in range(num_units):
            for ind in range(min_run_trials):
                # bin spike times for each unit and add to bin_mat
                spike_times = df[cond][unit_ind][good_trial_inds[ind]]
                bin_mat[unit_ind,:,ind] = np.histogram(spike_times,bins)[0]
            ## end trial loop
        ## end unit loop
        mat_list.append(bin_mat)
    return mat_list

#def bin_data(df,trials_ran_dict,min_run_trials,stim_start=1.0,stim_stop=2.5,bin_size=0.005):
#
#    dict_keys = np.sort(trials_ran_dict.keys())
#    num_units = df.shape[0]
#    dt        = stim_stop - stim_start
#    bins      = np.arange(stim_start,stim_stop+bin_size,bin_size)
#    base_bins = np.arange(0, stim_stop-stim_start+bin_size, bin_size)
#    num_bins  = len(bins) - 1
#    print('\n----- bin_data function -----')
#    print('not using last {0}ms of data'.format(str((dt - num_bins*bin_size))))
#    print('num units: {0}\nnum trials: {1}\nnum bins: {2}'.format(str(num_units),str(min_run_trials),str(num_bins)))
#
#    mat_list = []
#
#    for cond in dict_keys:
#        bin_mat = np.zeros((num_units,num_bins,min_run_trials))
#        cond_bool = np.array(trials_ran_dict[cond])
#        good_trial_inds = np.where(cond_bool == True)[0]
#        for unit_ind in range(num_units):
#            for ind in range(min_run_trials):
#                # bin spike times for each unit and add to bin_mat
#                spike_times = df[cond][unit_ind][good_trial_inds[ind]]
#                stim_rate_spikes = np.histogram(spike_times,bins)[0]
#                base_rate_spikes = np.histogram(spike_times,base_bins)[0]
#                bin_mat[unit_ind,:,ind] = stim_rate_spikes - base_rate_spikes
#            ## end trial loop
#        ## end unit loop
#        mat_list.append(bin_mat)
#    return mat_list

def reorganize_bin_data(mat_list):
    '''
    Reorganizes binned spike data to the appropriate format for use in linear
    regression.

    Input: binned spike data list. The list contains 3-d numpy array where each
    entry in the list corresponds to a particular stimulus type. The row of the
    arrays correspond to an individual unit. Columns correspond to a bin with
    the number of spikes for that bin. The 3-d dimension corresponds to repeated
    trials.

    Output: A txn permuted array where t corresponds to a particular bin and n
    is an individual unit. A tx1 position array is returned. Each row in the
    target vector corresponds to the same row or time point in the data vector.
    The columns correspond to the basis values for the given stimulus. A tx1
    array that is the position vector converted to circuilar coordinates.
    '''
    num_trials = mat_list[0].shape[2]
    num_pos = len(mat_list)
    num_units = mat_list[0].shape[0]
    num_bins = mat_list[0].shape[1]

    num_rows = (num_bins)*num_trials*num_pos
    num_cols = num_units
    X = np.zeros((num_rows,num_cols))
    pos_array = np.zeros((num_rows,))
    count = 0

    for trial in range(num_trials):
        for pos in range(num_pos):
            X[count*num_bins:(count+1)*num_bins,:] = mat_list[pos][:,:,trial].T
            pos_array[count*num_bins:(count+1)*num_bins] = np.ones((num_bins,))*pos
            count += 1
        # end pos
    # end trial

    step_size = 2*np.pi/num_pos
    theta = pos_array*step_size

    permuted_inds = np.random.permutation(X.shape[0])
    perm_X = X[permuted_inds,:]
    perm_pos = pos_array[permuted_inds]
    perm_theta = theta[permuted_inds]

    return perm_X, perm_pos, perm_theta

def reorganize_single_unit_data(X, y):
    num_pos = len(np.unique(y))
    pos_array = y

    step_size = 2*np.pi/num_pos
    theta = pos_array*step_size

    permuted_inds = np.random.permutation(X.shape[0])
    perm_X = X[permuted_inds,:]
    perm_pos = pos_array[permuted_inds]
    perm_theta = theta[permuted_inds]

    return perm_X, perm_pos, perm_theta

def psth_decoder(X):
    num_trials = X.shape[0]
    num_categories = X.shape[2]
    labels = range(num_categories)
    predicted = np.zeros((num_categories*num_trials,))
    actual    = np.zeros((num_categories*num_trials,))
    loo = cross_validation.LeaveOneOut(num_trials)
    count = 0

    for train_index, test_index in loo:
        mean_psth = X[train_index, :, :].mean(axis=0)
#        print('-------------------------')
#        print(mean_psth)
        for i in range(num_categories):
            single_trial_data = X[test_index, :, i]
#            print('single trial: ' +str(single_trial_data))
            # get distance measures
            dist_temp = np.empty((num_categories,))
            for k in range(num_categories):
                dist_temp[k] = np.linalg.norm(mean_psth[:, k] - single_trial_data)
            # select category with smallest distance
            choice = np.random.choice(np.where(np.min(dist_temp) == dist_temp)[0])
            #choice = np.argmin(dist_temp)
#            print("choice: " + str(choice))
            predicted[count,] = choice
            actual[count,] = i
            count += 1

    #compute confusion matrix
    #This is Mike Schakter's way
    cmat = confusion_matrix(actual, predicted, labels)
    cmat = cmat.astype(float)
    cmat = (cmat.T/cmat.sum(axis=1)).T
    cmat = np.nan_to_num(cmat)
    #plt.figure()
    #plt.imshow(cmat,vmin=0,vmax=1,interpolation='none',cmap='hot')

    return cmat

def get_single_unit_matrix(bin_mat_list, unit=0):

    num_trials = bin_mat_list[0].shape[2]
    num_bins   = bin_mat_list[0].shape[1]
    num_pos    = len(bin_mat_list)
    data_mat   = np.zeros((num_trials, num_bins, num_pos))

    for i in range(num_pos):
        data = bin_mat_list[i][unit, :, :].T # unit, bins, trials (transpose to get trials x bins)
        data_mat[:,:,i] = data

    return data_mat

def reorganize_evan_data(X, y, min_trials):

    num_pos = len(np.unique(y))
    x_temp = np.zeros((min_trials*num_pos, X.shape[1]))
    pos_array = np.zeros((min_trials*num_pos, 1))
    for pos in range(num_pos):
        pos_inds = np.where(y == pos)[0]
        x_temp[pos*min_trials:(pos+1)*min_trials, :] = X[pos_inds[0:min_trials], :]
        pos_array[pos*min_trials:(pos+1)*min_trials,:] = y[pos_inds[0:min_trials], :]

    step_size = 2*np.pi/num_pos
    theta = pos_array*step_size

    permuted_inds = np.random.permutation(x_temp.shape[0])
    perm_X = x_temp[permuted_inds,:]
    perm_pos = pos_array[permuted_inds]
    perm_theta = theta[permuted_inds]

    return perm_X, perm_pos, perm_theta

def reshape_ca_design_matrix(X, y, min_trials):
    num_trials = min_trials
    num_units  = X.shape[1]
    num_pos    = len(np.unique(y))
    new_X      = np.zeros((num_trials, num_units, num_pos))
    for pos in range(num_pos):
        data_inds = np.where(y == pos)[0] #replace pos_pre with pos_vec
        new_X[:,:, pos] = X[data_inds, :]

    return new_X

def get_single_neuron_bin_data(mat_list,unit_ind=0):
    num_pos = len(mat_list)
    unit_data = list()
    pos_vect = list()
    for pos in range(num_pos):
        if pos == 0:
            unit_data = mat_list[pos][unit_ind,:,:].T
            pos_vect = np.ones((mat_list[pos].shape[2],1))*pos
        else:
            unit_data = np.concatenate((unit_data,mat_list[pos][unit_ind,:,:].T),axis=0)
            pos_vect = np.concatenate((pos_vect,np.ones((mat_list[pos].shape[2],1))*pos),axis=0)

    permuted_inds = np.random.permutation(unit_data.shape[0])
    perm_unit_data = unit_data[permuted_inds,:]
    perm_pos_vect  = pos_vect[permuted_inds,:]
    return perm_unit_data, perm_pos_vect

def gen_fake_data_multiunit(num_trials=100,num_pos=7,bin_size=0.100,trial_duration=1.0,control_pos=True):

    num_bins = int(trial_duration/bin_size)
    bin_mat_list = list()
    bins = np.arange(0,trial_duration+bin_size,bin_size)
    if control_pos is False:
        num_pos = 6
    num_units = num_pos

    for pos in range(num_pos):
        data_mat = np.zeros((num_units,num_bins,num_trials))

        for trial in range(num_trials):
            for unit in range(num_units):
                if unit == 6 or pos == 6:
                    fr = 5
                else:
                    fr = 70.0 - 10.0*(np.abs(pos - unit))
                # unit firing rate
                spk_times = list()
                while sum(spk_times) < trial_duration:
                    spk_times.append(rd.expovariate(fr))
                spk_times = np.cumsum(spk_times)
                data_mat[unit,:,trial] = np.histogram(spk_times,bins)[0]
        bin_mat_list.append(data_mat)

    return bin_mat_list


def gen_fake_data(num_trials=10,num_pos=7.0,bin_size=0.005,trial_duration=1.0,start_rate=5,stop_rate=10):

    num_bins  = int(trial_duration/bin_size)
    data_mat  = np.zeros((num_pos*num_trials,num_bins))
    trial_mat = np.zeros((num_pos*num_trials,1))
    spike_rate = np.arange(start_rate,stop_rate+0.1,(stop_rate - start_rate)/(num_pos -1.0))
    print('spike rates: ' + str(spike_rate))
    #spike_rate = np.arange(5,5*num_pos+1,5)
    bins      = np.arange(0,trial_duration+bin_size,bin_size)
    count = 0

    for pos in range(num_pos):
        lambda_rate = spike_rate[pos]

        for trial in range(num_trials):
            spk_times = list()

            while sum(spk_times) < trial_duration:
                spk_times.append(rd.expovariate(lambda_rate))
            spk_times = np.cumsum(spk_times)

            data_mat[count,:] = np.histogram(spk_times,bins)[0]
            trial_mat[count] = pos
            count += 1

    permuted_inds = np.random.permutation(count)
    perm_data_mat = data_mat[permuted_inds,:]
    perm_trial_mat = trial_mat[permuted_inds,:]

    return perm_data_mat, perm_trial_mat

def basis_vals(kappa,theta,theta_k):
    '''
    Returns a txk array of basis values.
    Input: theta is a tx1 array of stimulus position values. theta_k is a kx1
    array of position coefficients for the von mises basis functions.
    Output: a txk array of basis values where each row corresponds to a time
    point in the position and data arrays.
    '''

    B = np.zeros((theta.shape[0],theta_k.shape[0]))
    count = 0

    for pos in theta:
        B[count,:] = np.exp(kappa*np.cos(pos - theta_k)).reshape(1,theta_k.shape[0])
        count += 1

    return B

def predict_position(data,kappa,theta_k,W):

    # Make sure data is a 1xn vector (n: number of neurons)
    # W is the weight matrix of size nxk (k: number of positions/von_mises
    # functions)

    theta_val_mat = np.zeros((theta_k.shape[0],2))
    count = 0
    for theta in theta_k:
        theta = theta.reshape(1,)
        B = basis_vals(kappa,theta,theta_k)
        theta_val_mat[count,0] = theta
        theta_val_mat[count,1] = data.dot(W.dot(B.T))

        count += 1

    max_ind = np.argmax(theta_val_mat[:,1])
    theta_hat = theta_val_mat[max_ind,0]
    #print('predicted position: ' + str(max_ind))
    return max_ind

def fit_logistic_regression(unit_data,pos_data,C_to_try=[1e-3,1e-2,1e-1,1.0],k_folds=10):
    '''
    Use logistic regression to fit data from a single unit in order to try and predict
    the stimulus using k-fold cross validation.
    '''
    best_C = None
    best_pc = 0
    best_weights = None
    best_intercept = None
    labels = np.unique(pos_data)
    num_pos = float(len(labels))
    pos_data = np.ravel(pos_data)

    unit_data_original = unit_data
    pos_data_original  = pos_data

    for C in C_to_try:
        weights = list()
        intercepts = list()
        cmats = list()
        perc_correct = list()
        for a in range(100):

            # Randomize data
            permuted_indices = np.random.permutation(pos_data_original.shape[0])
            unit_data = unit_data_original[permuted_indices, :]
            pos_data  = pos_data_original[permuted_indices,]

            # Fit model and test with k-fold cross validation
            for train_inds, test_inds in KFold(len(pos_data),k_folds):
                assert len(np.intersect1d(train_inds,test_inds)) == 0
                #break the data matrix up into training and test sets
                Xtrain, Xtest, ytrain, ytest = unit_data[train_inds], unit_data[test_inds], pos_data[train_inds], pos_data[test_inds]

                # make logistic regression object
                lr = LogisticRegression(C=C)
                lr.fit(Xtrain, ytrain)

                # predict the stimulus with the test set
                ypred = lr.predict(Xtest)

                # compute confusion matrix
                cmat = confusion_matrix(ytest,ypred,labels)
                cmat = cmat.astype(float)

                #computed. This was Mike Schakter's way...I think it is wrong.
                cmat = (cmat.T/cmat.sum(axis=1)).T
                cmat = np.nan_to_num(cmat)
                # normalize each row of the confusion matrix so they represent
                # probabilities
            #    cmat = cmat.T/cmat.sum(axis=1).astype(float)
            #    cmat = np.nan_to_num(cmat)

                # compute the percent correct
                perc_correct.append(np.trace(cmat, offset=0)/num_pos)

                # record confusino matrix for this fold
                cmats.append(cmat)

                # record the weights and intercept
                weights.append(lr.coef_)
                intercepts.append(lr.intercept_)

        # Compute the mean confusion matrix
        # cmats and mean_pc get overwritten with ever iteration of the C parameter
        cmats = np.array(cmats)
        Cmean = cmats.mean(axis=0)

        # Compute the mean percent correct
        mean_pc = np.mean(perc_correct)
        std_pc  = np.std(perc_correct,ddof=1)

        # Compute the mean weights
        weights = np.array(weights)
        mean_weights = weights.mean(axis=0)
        mean_intercept = np.mean(intercepts)

        # Determine if we've found the best model thus far
        if mean_pc > best_pc:
            best_pc = mean_pc
            best_C = C
            best_Cmat = Cmean
            best_weights = mean_weights
            best_intercept = mean_intercept

    # Print best percent correct and plot confusion matrix
    print('Mean percent correct: ' + str(mean_pc*100) + str('%'))
#    plt.figure()
#    plt.imshow(best_Cmat,vmin=0,vmax=1,interpolation='none',cmap='hot')
#    plt.colorbar()
    #plt.show()
    print('sum: ' + str(best_Cmat.sum(axis=1)))

    return best_Cmat, best_C

def fit_ole_decoder(X, pos, theta, kappa_to_try, k_folds=10):
    '''
    Use optimal linear estimation (OLE) to model the instantaneous stimulus
    position using data from many single units using k-fold cross validation.
    '''

    best_kappa = None
    best_pc = 0
    best_weights = None
    best_intercept = None

    theta_k = np.unique(theta)
    labels  = np.unique(pos)
    num_pos = float(len(theta_k))

    for kappa in kappa_to_try:
        weights = list()
        intercepts = list()
        cmats = list()
        perc_correct = list()
        y = basis_vals(kappa,theta,theta_k)

        for train_inds, test_inds in KFold(X.shape[0],k_folds):
            assert len(np.intersect1d(train_inds,test_inds)) == 0
            #break the data matrix up into training and test sets
            Xtrain, Xtest, ytrain, _  = X[train_inds,:], X[test_inds,:], y[train_inds,:], y[test_inds,:]

            # make  regression object
            clf = Ridge(alpha=1.0,fit_intercept=False)
            clf.fit(Xtrain,ytrain)
            W = clf.coef_.T

            # predict the stimulus with the test set
            ypred = list()

            for test_ind in range(Xtest.shape[0]):
                ypred.append(predict_position(Xtest[test_ind,:],kappa,theta_k,W))

            ypred = np.array(ypred)
#           print('predicted: ' + str(ypred))
#           print('actual   : ' + str(pos[test_inds]))

            # compute confusion matrix
            cmat = confusion_matrix(pos[test_inds],ypred,labels)
            cmat = cmat.astype(float)

            # normalize each row of the confusion matrix so they represent
            # probabilities
            cmat = (cmat.T/cmat.sum(axis=1)).T
            cmat = np.nan_to_num(cmat)

            # compute the percent correct
            perc_correct.append(np.trace(cmat, offset=0)/num_pos)

            # record confusino matrix for this fold
            cmats.append(cmat)

            # record the weights and intercept
            weights.append(W)
            intercepts.append(clf.intercept_)

        # Compute the mean confusion matrix
        # cmats and mean_pc get overwritten with ever iteration of the C parameter
        cmats = np.array(cmats)
        Cmean = cmats.mean(axis=0)

        # Compute the mean percent correct
        mean_pc = np.mean(perc_correct)
        std_pc  = np.std(perc_correct,ddof=1)

        # Compute the mean weights
        weights = np.array(weights)
        mean_weights = weights.mean(axis=0)
        mean_intercept = np.mean(intercepts)

        # Determine if we've found the best model thus far
        if mean_pc > best_pc:
            best_pc = mean_pc
            best_kappa = kappa
            best_Cmat = Cmean
            best_weights = mean_weights
            best_intercept = mean_intercept

    # Print best percent correct and plot confusion matrix
    print('Mean percent correct: ' + str(mean_pc*100) + str('%'))
    print('Best kappa: ' + str(best_kappa))
#   print('Mean confusion matrix: ' + str(best_Cmat))
    plt.figure()
    plt.imshow(best_Cmat,vmin=0,vmax=1,interpolation='none',cmap='hot')
    plt.title('PCC: ' + "{:.2f}".format(mean_pc*100))
    plt.colorbar()
    plt.show()
    print('sum: ' + str(best_Cmat.sum(axis=1)))

    return best_weights, mean_pc, best_Cmat

def calc_global_selectivity(cmat):
    num_pos = float(cmat.shape[0])
    pcc = np.diagonal(cmat)*(1.0/np.trace(cmat))
#    pcc = np.diagonal(cmat)/num_pos

    h_obs = np.sum(np.nan_to_num(-pcc*np.log2(pcc)))
    h_max = -np.log2(1.0/num_pos)

    gs = 1.0 - (h_obs/h_max)

    return gs

def calc_max_selectivity(cmat):
    num_pos = float(cmat.shape[0])
    pcc = np.diagonal(cmat)

    pref_pos = np.argmax(pcc)
    other_positions = np.delete(np.arange(num_pos), pref_pos).astype(int)
    max_selectivity = np.log2( ((num_pos - 1)*pcc[pref_pos])/(np.sum(pcc[other_positions])))

    return max_selectivity

def calc_invariance(cmat):
    num_pos = float(cmat.shape[0])
    pcc = np.diagonal(cmat)/num_pos

    h_obs = np.sum(np.nan_to_num(-cmat*np.log2(cmat)))
    h_max = -np.log2(1/num_pos**2)

    h_min = 0;
    for row in range(cmat.shape[0]):
        for col in range(cmat.shape[1]):
            Pj_of_i = np.sum(cmat[row, :])
        h_min += np.sum(-Pj_of_i * np.log2(Pj_of_i))

    invariance = (h_obs - h_min)/(h_max - h_min)

    return invariance

def calc_mi(cmat):
    I = cmat.shape[0]
    J = cmat.shape[1]
    mi = 0
    for i in range(int(I)):
        for j in range(int(J)):
            P_ij = cmat[i, j]
            P_i  = 1.0/I # Assumes that equal number of trials for each
                         # for each category were used in the decoding process
            P_j  = cmat[:, j].sum()#/float(J)

            mi_temp = np.nan_to_num(P_ij*np.log2(P_ij/(P_i*P_j)))
            mi += mi_temp
    return mi

########## MAIN CODE ##########
########## MAIN CODE ##########

if __name__ == "__main__":
    # Select which experiments to analyze
    fids = ['0871']
    #fids = ['1034', '1044', '1054', '1058', '1062']
    figpath = '/home/greg/Documents/AdesnikLab/Figures/WhiskerTrim/neuro/AllTuningCurves_cmats/'
    cell_type = list()
    PCC_Pre = np.empty(0)
    PCC_Post = np.empty(0)
    GS_Pre = np.empty(0)
    GS_Post = np.empty(0)
    Inv_Pre = np.empty(0)
    Inv_Post = np.empty(0)
    Select_Pre = np.empty(0)
    Select_Post = np.empty(0)
    best_pcc_Pre = np.empty(0)
    best_pcc_Post = np.empty(0)
    pos_pred_pre = np.empty(0)
    pos_pred_post = np.empty(0)

    evoked_rate = list()
    exp_ind     = np.empty(0) # save indices for experiments so I can index into the cmat matrices
    pre_cmats   = np.empty((8,8))
    post_cmats  = np.empty((8,8))

    for eind, fid in enumerate(fids):

        sns.set_context("poster")
        sns.set_style("white")

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
        #cond_ind_dict,trials_ran_dict = classify_run_trials(stim,vel_mat,trial_time,stim_start=1.25,stim_stop=2.50,
        #        mean_thresh=250,sigma_thresh=150,low_thresh=200,display=False)
        # easy running thresholds
        cond_ind_dict,trials_ran_dict = classify_run_trials(stim,vel_mat,trial_time,stim_start=1.5,stim_stop=2.50,
                mean_thresh=150,sigma_thresh=250,low_thresh=100,display=False)

        # Find the condition with the least number of trials
        min_run_trials  = min([sum(trials_ran_dict[x]) for x in trials_ran_dict.keys()])

        # Put data into a Pandas dataframe
        df = make_df(sort_file_paths,data_dir_path,region=region)

        # plot tuning curves
        depth = df['depth']
        cell_type_list = df['cell_type']
        em, es = make_evoke_rate_array(df, trials_ran_dict, base_start=0, base_stop=1.0, stim_start=1.50,
                stim_stop=2.50)
        evoked_rate.append((em, es))


        make_tuning_curves(em, es,depth=depth,cell_type_list=cell_type_list, control_pos=7,
                fig_title='Evoked Firing Rate--fid' + fid + region + ' full pad',
                share_yax=False)

        # bin data and try to make a prediction about the stimuli
        bin_mat_list = bin_data(df, trials_ran_dict, min_run_trials, stim_start=1.250, stim_stop=2.50, bin_size=1.250)


#    ##### PULL OUT DATA FOR SINGLE UNIT AND RUN THROUGH OLE DECODER #####
#        unit_data,pos_data = get_single_neuron_bin_data(bin_mat_list,unit_ind=3)
#        pre_inds = np.where(pos_data < 9)[0]
#        pre_unit_data = unit_data[pre_inds]
#        pre_pos_data  = pos_data[pre_inds]
#        X, pos, theta = reorganize_single_unit_data(pre_unit_data, pre_pos_data)
#        kappa_to_try = np.arange(1,101,1)
#        print('runnng ole_decoder')
#        w, pcc = fit_ole_decoder(X, pos, theta, kappa_to_try, k_folds=10)

    ##    ##### MAKE FAKE SPIKE DATA FROM A HOMOGENOUS POISSON PROCESS #####
    ##    bin_mat_list = gen_fake_data_multiunit(num_trials=100, num_pos=7, bin_size=1, trial_duration=1.0, control_pos=True)
    ##    plt.figure()
    ##    plt.imshow(np.mean(bin_mat_list[0], axis=2), interpolation='none')
    ##    plt.show()
    ##
    ##    ##### END FAKE DATA BLOCK #####

    ##    #unit_data,pos_dat = gen_fake_data(num_trials=30,num_pos=7.0,bin_size=0.100,trial_duration=1.0,start_rate=1,stop_rate=100)
    ##    #fit_logistic_regression(unit_data,pos_data,C_to_try=[1e-3,1e-2,1e-1,1.0],k_folds=10)
    #
        #wtmat = h5py.File(hsv_mat_path)
    ##    bin_mat_list = bin_mat_list[9::]
#        bin_mat_list = bin_mat_list[0:7]
#        bin_mat_list = bin_mat_list[7:14]
#        bin_mat_list = bin_mat_list[14::]
        # reorganize data for regression
        X,pos,theta = reorganize_bin_data(bin_mat_list)
        kappa_to_try = np.arange(1,101,1)
        print('runnng ole_decoder')
        w, pcc, best_cmat = fit_ole_decoder(X, pos, theta, kappa_to_try, k_folds=10)
        fail()

##### ##### LOAD AND ANLYZE EVANS DATA ##### #####
##### ##### LOAD AND ANLYZE EVANS DATA ##### #####
#        evan_pre = h5py.File('/home/greg/Documents/AdesnikLab/Figures/WhiskerTrim/neuro/CaImaging/2479pre.mat')

        pre_cmat_all  = np.zeros((8, 8, 50000))
        post_cmat_all = np.zeros((8, 8, 50000))
        evan_post = h5py.File('/home/greg/Documents/AdesnikLab/Figures/WhiskerTrim/neuro/CaImaging/3152post.mat')
        evan_pre = h5py.File('/home/greg/Documents/AdesnikLab/Figures/WhiskerTrim/neuro/CaImaging/3152pre.mat')
        l4_pos = h5py.File('/home/greg/Documents/AdesnikLab/Figures/WhiskerTrim/neuro/CaImaging/3152_150914_pos.mat')
        for i in range(50000):
            if i%100 == 0:
                print('On trial: ' + str(i))
        #    evan_pre = h5py.File('/home/greg/Documents/AdesnikLab/Figures/WhiskerTrim/neuro/CaImaging/3152pre.mat')
        #    l4_pos = h5py.File('/home/greg/Documents/AdesnikLab/Figures/WhiskerTrim/neuro/CaImaging/3152_150914_pos.mat')
            l4_pwc = l4_pos['insidePWC'][:][0]
            pwc_inds = np.where(l4_pwc == 0)[0]
            X_pre = evan_pre['Data'][:]
            y_pre = evan_pre['StimID'][:]
            contact_ind = np.where(y_pre != 0)[0]
            X_pre = X_pre[contact_ind, :]
            X_pre = X_pre[:, pwc_inds]
            y_pre = y_pre[contact_ind]-1

    #        evan_post = h5py.File('/home/greg/Documents/AdesnikLab/Figures/WhiskerTrim/neuro/CaImaging/2479post.mat')
            #evan_post = h5py.File('/home/greg/Documents/AdesnikLab/Figures/WhiskerTrim/neuro/CaImaging/3152post.mat')
            X_post = evan_post['Data'][:]
            y_post = evan_post['StimID'][:]
            contact_ind = np.where(y_post != 0)[0]
            X_post = X_post[contact_ind, :]
            X_post = X_post[:, pwc_inds]
            y_post = y_post[contact_ind]-1
            min_trials = np.min([np.min(np.histogram(y_pre, bins=np.arange(0, 9))[0]),
                    np.min(np.histogram(y_post, bins=np.arange(0, 9))[0])])

            X_pre,pos_pre,theta_pre = reorganize_evan_data(X_pre, y_pre, min_trials)
            X_post,pos_post,theta_post = reorganize_evan_data(X_post, y_post, min_trials)

    #        kappa_to_try = np.arange(1,201,1)
    #        print('runnng ole_decoder')
    #        w, pcc, cmat_pre = fit_ole_decoder(X_pre, pos_pre, theta_pre, kappa_to_try, k_folds=10)


            X_pre = reshape_ca_design_matrix(X_pre, pos_pre, min_trials)
            cmat_pre = psth_decoder(X_pre[0:5, :, :])

            X_post = reshape_ca_design_matrix(X_post, pos_post, min_trials)
            cmat_post = psth_decoder(X_post[0:5, :, :])

            cmat_pre = cmat_pre*(1.0/cmat_pre.sum(axis=1)[:, None])
            cmat_post = cmat_post*(1.0/cmat_post.sum(axis=1)[:, None])
            pre_cmat_all[:,:,i] = cmat_pre
            post_cmat_all[:,:,i] = cmat_post
#        X_post,pos_post,theta = reorganize_evan_data(X_post, y_post)
#        kappa_to_try = np.arange(1,201,1)
#        print('runnng ole_decoder')
#        w, pcc, cmat_post = fit_ole_decoder(X_post, pos_post, theta_pre, kappa_to_try, k_folds=10)

        cmat_pre = pre_cmat_all.mean(axis=2)
        cmat_post = post_cmat_all.mean(axis=2)

        gs_pre = calc_global_selectivity(cmat_pre)
        inv_pre = calc_invariance(cmat_pre)
        max_selec_pre = calc_max_selectivity(cmat_pre)
        pcc_pre = np.diagonal(cmat_pre).sum()/cmat_pre.shape[0]*100
        bb_pre  = np.max(np.diagonal(cmat_pre))*100
        num_pred_pos_pre = np.sum(np.diagonal(cmat_pre) > 1.0/8.0)

        gs_post = calc_global_selectivity(cmat_post)
        inv_post = calc_invariance(cmat_post)
        max_selec_post = calc_max_selectivity(cmat_post)
        pcc_post = np.diagonal(cmat_post).sum()/cmat_post.shape[0]*100
        bb_post  = np.max(np.diagonal(cmat_post))*100
        num_pred_pos_post = np.sum(np.diagonal(cmat_post) > 1.0/8.0)

        mi_pre  = calc_mi(cmat_pre/8.0)
        mi_post = calc_mi(cmat_post/8.0)


        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(cmat_pre,vmin=0,vmax=1, aspect='equal', interpolation='none',cmap='hot')
        plt.colorbar()
        plt.title('Post-trim ' + 'PCC: ' + "{:.4f}".format(pcc_pre) + '\n'
                + 'Global Sel: ' + "{:.4f}".format(gs_pre) + '\n'
                + 'Mutual Info: ' + "{:.4f}".format(mi_pre))

        plt.subplot(1,2,2)
        plt.imshow(cmat_post,vmin=0,vmax=1, aspect='equal', interpolation='none',cmap='hot')
        plt.colorbar()
        plt.title('Post-trim ' + 'PCC: ' + "{:.4f}".format(pcc_post) + '\n'
                + 'Global Sel: ' + "{:.4f}".format(gs_post) + '\n'
                + 'Mutual Info: ' + "{:.4f}".format(mi_post))

        print(np.diagonal(cmat_pre)[2:6])
        print(np.diagonal(cmat_post)[2:6])

        evan_post = h5py.File('/home/greg/Documents/AdesnikLab/Figures/WhiskerTrim/neuro/CaImaging/3152post.mat')
        evan_pre = h5py.File('/home/greg/Documents/AdesnikLab/Figures/WhiskerTrim/neuro/CaImaging/3152pre.mat')
        l4_pos = h5py.File('/home/greg/Documents/AdesnikLab/Figures/WhiskerTrim/neuro/CaImaging/3152_150914_pos.mat')
        l4_pwc = l4_pos['insidePWC'][:][0]

        X_pre = evan_pre['Data'][:]
        y_pre = evan_pre['StimID'][:]
        contact_ind = np.where(y_pre != 0)[0]
        X_pre = X_pre[contact_ind, :]
        y_pre = y_pre[contact_ind]-1

        X_post = evan_post['Data'][:]
        y_post = evan_post['StimID'][:]
        contact_ind = np.where(y_post != 0)[0]
        X_post = X_post[contact_ind, :]
        y_post = y_post[contact_ind]-1

        x_coord = l4_pos['centroids'][0,:]
        y_coord = l4_pos['centroids'][1,:]
        for i in range(8):
            trial_inds_pre = np.where(y_pre == i)[0]
            deltaf_pre = X_pre[trial_inds_pre, :].mean(axis=0)*100
            trial_inds_post = np.where(y_post == i)[0]
            deltaf_post = X_post[trial_inds_post, :].mean(axis=0)*100

            plt.figure()
            plt.scatter(x_coord, y_coord, s=deltaf_pre, c='black',alpha=0.5)
            plt.scatter(x_coord, y_coord, s=deltaf_post, c='red',alpha=0.5)
            plt.ylim(600,0)
            plt.title('L4 Position: ' +str(i+1))
            plt.savefig('/home/greg/Documents/AdesnikLab/Figures/WhiskerTrim/neuro/CaImaging/L4pos' + str(i+1) + '.pdf')



##### ##### END EVANS DATA ##### #####
##### ##### END EVANS DATA ##### #####

##### ##### ANALYZE INDIVIDUAL UNITS ##### #####
##### ##### ANALYZE INDIVIDUAL UNITS ##### #####
#        X = np.zeros((min_run_trials, 1, 8))
        for unit2analyze in range(df.shape[0]):
            print('Unit: ' + str(unit2analyze))
            #unit2analyze = 31
            bin_mat_list = bin_data(df, trials_ran_dict, min_run_trials, stim_start=1.5, stim_stop=2.50, bin_size=1.0)
            if np.max(em[unit2analyze][0:8]) > 1.75:
                bin_mat_list_pre = bin_mat_list[0:8]
    #            unit_data_pre, pos_data_pre = get_single_neuron_bin_data(bin_mat_list_pre,unit_ind=unit2analyze)
    #            best_cmat_pre, c_pre  = fit_logistic_regression(unit_data_pre, pos_data_pre, C_to_try=[1e-3,1e-2,1e-1,1.0], k_folds=10)
                X = get_single_unit_matrix(bin_mat_list_pre, unit=unit2analyze)
                best_cmat_pre = psth_decoder(X)

                best_cmat_pre = best_cmat_pre*(1.0/best_cmat_pre.sum(axis=1)[:, None])
                gs_pre = calc_global_selectivity(best_cmat_pre)
                inv_pre = calc_invariance(best_cmat_pre)
                max_selec_pre = calc_max_selectivity(best_cmat_pre)
                pcc_pre = np.diagonal(best_cmat_pre).sum()/best_cmat_pre.shape[0]*100
                bb_pre  = np.max(np.diagonal(best_cmat_pre))*100
                num_pred_pos_pre = np.sum(np.diagonal(best_cmat_pre) > 1.0/8.0)

                bin_mat_list_post = bin_mat_list[9:-1]
                #unit_data_post, pos_data_post = get_single_neuron_bin_data(bin_mat_list_post,unit_ind=unit2analyze)
                #best_cmat_post, c_post = fit_logistic_regression(unit_data_post, pos_data_post, C_to_try=[1e-3,1e-2,1e-1,1.0], k_folds=10)
                X = get_single_unit_matrix(bin_mat_list_post, unit=unit2analyze)
                best_cmat_post = psth_decoder(X)

                best_cmat_post = best_cmat_post*(1.0/best_cmat_post.sum(axis=1)[:, None])
                gs_post = calc_global_selectivity(best_cmat_post)
                inv_post = calc_invariance(best_cmat_post)
                max_selec_post = calc_max_selectivity(best_cmat_post)
                pcc_post = np.diagonal(best_cmat_post).sum()/best_cmat_post.shape[0]*100
                bb_post  = np.max(np.diagonal(best_cmat_post))*100
                num_pred_pos_post = np.sum(np.diagonal(best_cmat_post) > 1.0/8.0)

                GS_Pre        = np.append(GS_Pre, gs_pre)
                GS_Post       = np.append(GS_Post, gs_post)
                Inv_Pre       = np.append(Inv_Pre, inv_pre)
                Inv_Post      = np.append(Inv_Post, inv_post)
                Select_Pre    = np.append(Select_Pre, max_selec_pre)
                Select_Post   = np.append(Select_Post, max_selec_post)
                PCC_Pre       = np.append(PCC_Pre, pcc_pre)
                PCC_Post      = np.append(PCC_Post, pcc_post)
                best_pcc_Pre  = np.append(best_pcc_Pre, bb_pre)
                best_pcc_Post = np.append(best_pcc_Post, bb_post)
                pos_pred_pre  = np.append(pos_pred_pre, num_pred_pos_pre)
                pos_pred_post = np.append(pos_pred_post, num_pred_pos_post)

                exp_ind   = np.append(exp_ind, eind)
                pre_cmats = np.dstack((pre_cmats, best_cmat_pre))
                post_cmats = np.dstack((post_cmats, best_cmat_post))

                cell_type.append(df['cell_type'][unit2analyze])

            else:
                print('DID NOT ANALYZE UNIT DUE TO LOW FIRING RATE')


        pre_cmats  = pre_cmats[:,:,1::]
        post_cmats = post_cmats[:,:,1::]

        ####### INDENT FOR MULTIPLE FID ANALYSIS ###########
        ####### INDENT FOR MULTIPLE FID ANALYSIS ###########

#        X_pre = np.empty((min_run_trials,1,8))
#        X_post = np.empty((min_run_trials,1,8))
#
#        for unit2analyze in range(df.shape[0]):
#            print('Unit: ' + str(unit2analyze))
#            #unit2analyze = 31
#            bin_mat_list = bin_data(df, trials_ran_dict, min_run_trials, stim_start=1.5, stim_stop=2.50, bin_size=1.0)
#
#            if np.max(em[unit2analyze][0:8]) > 1.75:
#                bin_mat_list_pre = bin_mat_list[0:8]
#                X_temp_pre = get_single_unit_matrix(bin_mat_list_pre, unit=unit2analyze)
#                X_pre = np.append(X_pre, X_temp_pre, axis=1)
#
#                bin_mat_list_post = bin_mat_list[9:-1]
#                X_temp_post= get_single_unit_matrix(bin_mat_list_post, unit=unit2analyze)
#                X_post = np.append(X_post, X_temp_post, axis=1)
#            else:
#                print('DID NOT ANALYZE UNIT DUE TO LOW FIRING RATE')
#
#        X_pre = X_pre[:,1::,:]
#        X_post = X_post[:,1::,:]
#
#        best_cmat_pre = psth_decoder(X_pre)
#        best_cmat_pre = best_cmat_pre*(1.0/best_cmat_pre.sum(axis=1)[:, None])
#        gs_pre = calc_global_selectivity(best_cmat_pre)
#        inv_pre = calc_invariance(best_cmat_pre)
#        max_selec_pre = calc_max_selectivity(best_cmat_pre)
#        pcc_pre = np.diagonal(best_cmat_pre).sum()/best_cmat_pre.shape[0]*100
#        bb_pre  = np.max(np.diagonal(best_cmat_pre))*100
#        num_pred_pos_pre = np.sum(np.diagonal(best_cmat_pre) > 1.0/8.0)
#
#        best_cmat_post = psth_decoder(X_post)
#        best_cmat_post = best_cmat_post*(1.0/best_cmat_post.sum(axis=1)[:, None])
#        gs_post = calc_global_selectivity(best_cmat_post)
#        inv_post = calc_invariance(best_cmat_post)
#        max_selec_post = calc_max_selectivity(best_cmat_post)
#        pcc_post = np.diagonal(best_cmat_post).sum()/best_cmat_post.shape[0]*100
#        bb_post  = np.max(np.diagonal(best_cmat_post))*100
#        num_pred_pos_post = np.sum(np.diagonal(best_cmat_post) > 1.0/8.0)
#
#        GS_Pre        = np.append(GS_Pre, gs_pre)
#        GS_Post       = np.append(GS_Post, gs_post)
#        Inv_Pre       = np.append(Inv_Pre, inv_pre)
#        Inv_Post      = np.append(Inv_Post, inv_post)
#        Select_Pre    = np.append(Select_Pre, max_selec_pre)
#        Select_Post   = np.append(Select_Post, max_selec_post)
#        PCC_Pre       = np.append(PCC_Pre, pcc_pre)
#        PCC_Post      = np.append(PCC_Post, pcc_post)
#        best_pcc_Pre  = np.append(best_pcc_Pre, bb_pre)
#        best_pcc_Post = np.append(best_pcc_Post, bb_post)
#        pos_pred_pre  = np.append(pos_pred_pre, num_pred_pos_pre)
#        pos_pred_post = np.append(pos_pred_post, num_pred_pos_post)
#
#        exp_ind   = np.append(exp_ind, eind)
#        pre_cmats = np.dstack((pre_cmats, best_cmat_pre))
#        post_cmats = np.dstack((post_cmats, best_cmat_post))
#
#        cell_type.append(df['cell_type'][unit2analyze])
#
#        pre_cmats  = pre_cmats[:,:,1::]
#        post_cmats = post_cmats[:,:,1::]
        ####### END MULTIPLE FID ANALYSIS ###########

    ##### ##### Plot Tuning Curves and PCCs as Tuning Curves
    for exp_id in np.unique(exp_ind).astype(int):

        exp_indices = np.where(exp_ind == exp_id)[0]

        # Grab and scale all evoked tuning curves
        em_pre = evoked_rate[exp_id][0][:, 0:8]
        em_pre = em_pre/np.max(np.abs(em_pre), axis=1)[:, None]

        em_post = evoked_rate[exp_id][0][:, 9:-1]
        em_post = em_post/np.max(np.abs(em_post), axis=1)[:, None]

        # Grab all Diagonals from cmats
        pcc_pre_diag = np.zeros((exp_indices.shape[0], 8))
        pcc_post_diag = np.zeros((exp_indices.shape[0], 8))

        for i, unit_ind in enumerate(exp_indices):
            pcc_pre_diag[i, :] = np.diagonal(pre_cmats[:, :, unit_ind])
            pcc_post_diag[i, :] = np.diagonal(post_cmats[:, :, unit_ind])

        plt.figure()
        plt.subplot(2,2,1)
        plt.plot(em_pre.T)
        plt.ylim(-1.1, 1.1)
        plt.title('Full Pad Normalized Tuning Curves')
        plt.subplot(2,2,3)
        plt.plot(em_post.T)
        plt.ylim(-1.1, 1.1)
        plt.title('Single Whisker Normalized Tuning Curves')

        plt.subplot(2,2,2)
        plt.plot(pcc_pre_diag.T)
        plt.ylim(0, 1)
        plt.title('Full Pad PCC Values')
        plt.subplot(2,2,4)
        plt.plot(pcc_post_diag.T)
        plt.ylim(0, 1)
        plt.title('Single Whisker PCC Values')

        plt.suptitle('FID: ' + fids[exp_id])

        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(pcc_pre_diag,vmin=0,vmax=1, aspect='equal', interpolation='none',cmap='hot')
        plt.subplot(1,2,2)
        plt.imshow(pcc_post_diag,vmin=0,vmax=1, aspect='equal', interpolation='none',cmap='hot')


    pearson_coef, pearson_p = sp.stats.pearsonr(Select_Pre, diff_selectivity)
    pv_inds = [i for i, x in enumerate(cell_type) if x == 'PV']
    rs_inds = [i for i, x in enumerate(cell_type) if x == 'RS']
    uc_inds = [i for i, x in enumerate(cell_type) if x == 'UC']

##### ##### Change in best PCCs vs Best PCC for full pad ##### #####
    plt.figure()
    diff_best_PCC = best_pcc_Post - best_pcc_Pre
    pearson_coef, pearson_p = sp.stats.pearsonr(best_pcc_Pre, diff_best_PCC)

    plt.scatter(best_pcc_Pre[pv_inds], diff_best_PCC[pv_inds], color='r')
    plt.scatter(best_pcc_Pre[rs_inds], diff_best_PCC[rs_inds], color='k')
    plt.scatter(best_pcc_Pre[uc_inds], diff_best_PCC[uc_inds], color='g')

    xlims = plt.xlim()
    ylims = plt.ylim()
    plt.hlines(0, xlims[0], xlims[1], colors='k', linestyles='dashed')
    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])
    plt.text(xlims[0]+xlims[0]*0.2, ylims[1]-ylims[1]*0.2, "Pearson's rho: " + "{:.3f}".format(pearson_coef)
            + '\np = ' + str(pearson_p), fontsize='20')
    plt.xlabel('Full Pad PCC')
    plt.ylabel('Single whisker - full pad PCC')
    plt.title('Trimming Best PCC Change')
    plt.show()

    # permutation test
    pearson_null = np.zeros((10000,))
    for i in range(10000):
        diff_best_pcc_perm = np.random.permutation(best_pcc_Post) - best_pcc_Pre
        prho_permute, _ = sp.stats.pearsonr(best_pcc_Pre, diff_best_pcc_perm)
        pearson_null[i,] = prho_permute
    permute_pval = float(np.sum(pearson_null < -np.abs(pearson_coef)) + np.sum(pearson_null > np.abs(pearson_coef)))/pearson_null.shape[0]

##### ##### Number of positions above chance ##### #####
    plt.figure()
    diff_num_pos_pred = pos_pred_post - pos_pred_pre

    plt.hist(diff_num_pos_pred, np.arange(-5, 5), align='left')
    plt.xlim(-4, 4)
    plt.xlabel('Number of positions above chance Full Pad')
    plt.ylabel('Change in number of positions above chance (Single whisker - full pad)')
    plt.title('Change in number of positions above chance')
    plt.show()

##### ##### Change in Global Selectivity vs change in PCC ##### #####
    select_change = GS_Post - GS_Pre
    pcc_change    = PCC_Post - PCC_Pre
    prho, pval = sp.stats.pearsonr(pcc_change, select_change)

    # permutation test
    pearson_null = np.zeros((10000,))
    for i in range(10000):
        prho_permute, _ = sp.stats.pearsonr(np.random.permutation(select_change),
            pcc_change)
        pearson_null[i,] = prho_permute
    permute_pval = float(np.sum(pearson_null < -np.abs(prho)) + np.sum(pearson_null > np.abs(prho)))/pearson_null.shape[0]

    plt.figure()
    plt.scatter(pcc_change, select_change, c='k')
    x_lim = plt.xlim()
    y_lim = plt.ylim()
    plt.xlim(x_lim[0], x_lim[1])
    plt.ylim(y_lim[0], y_lim[1])
    plt.hlines(0, x_lim[0], x_lim[1], linestyles='dashed')
    plt.vlines(0, y_lim[0], y_lim[1], linestyles='dashed')
    plt.xlabel('Change in PCC')
    plt.ylabel('Change in Global Sel')
    plt.show()
##### ##### Change in mutual information ##### #####
    mi_pre = np.zeros((pre_cmats.shape[2],))
    mi_post = np.zeros((post_cmats.shape[2],))
    for i in range(pre_cmats.shape[2]):
        mi_pre[i] = calc_mi(pre_cmats[:,:,i]/8.0)
        mi_post[i] = calc_mi(post_cmats[:,:,i]/8.0)

    mi_change = mi_post - mi_pre
    prho, pval = sp.stats.pearsonr(pcc_change, mi_change)

    plt.figure()
    plt.scatter(pcc_change, mi_change, c='k')
    x_lim = plt.xlim()
    y_lim = plt.ylim()
    plt.xlim(x_lim[0], x_lim[1])
    plt.ylim(y_lim[0], y_lim[1])
    plt.hlines(0, x_lim[0], x_lim[1], linestyles='dashed')
    plt.vlines(0, y_lim[0], y_lim[1], linestyles='dashed')
    plt.xlabel('Change in PCC')
    plt.ylabel('Change in Mutual Info')
    plt.show()

##### ##### Descriptive Statistical Plots ##### #####
##### ##### Scatter Plots ##### #####
    plt.figure()
    plt.scatter(GS_Pre, GS_Post, c='k')
    plt.xlim(0, 0.9)
    plt.ylim(0, 0.9)
    plt.plot([0, 0.9], [0, 0.9], '--k')
    plt.xlabel('GS Full Pad')
    plt.ylabel('GS Single Whisker')

    plt.figure()
    plt.scatter(PCC_Pre, PCC_Post, c='k')
    plt.xlim(0, 35)
    plt.ylim(0, 35)
    plt.plot([0, 35], [0, 35], '--k')
    plt.xlabel('PCC Full Pad')
    plt.ylabel('PCC Single Whisker')

    plt.figure()
    plt.scatter(mi_pre, mi_post, c='k')
    plt.xlim(0, 0.85)
    plt.ylim(0, 0.85)
    plt.plot([0, 0.85], [0, 0.85], '--k')
    plt.xlabel('MI Full Pad')
    plt.ylabel('MI Single Whisker')

##### ##### Histograms ##### #####
    gs_bins = np.arange(0, 1.1, 0.05)
    gs_pre_hist = np.histogram(GS_Pre, bins=gs_bins)[0]
    gs_post_hist = np.histogram(GS_Post, bins=gs_bins)[0]
    plt.figure()
    plt.plot(gs_bins[0:-1], gs_pre_hist, 'k', gs_bins[0:-1], gs_post_hist, 'r')
    plt.show()

    pcc_bins = np.arange(0, 101, 2)
    pcc_pre_hist = np.histogram(PCC_Pre, bins=pcc_bins)[0]
    pcc_post_hist = np.histogram(PCC_Post, bins=pcc_bins)[0]
    plt.figure()
    plt.plot(pcc_bins[0:-1], pcc_pre_hist, 'k', pcc_bins[0:-1], pcc_post_hist, 'r')
    plt.show()

    mi_bins = np.arange(0, 1.1, 0.05)
    mi_pre_hist = np.histogram(mi_pre, bins=mi_bins)[0]
    mi_post_hist = np.histogram(mi_post, bins=mi_bins)[0]
    plt.figure()
    plt.plot(mi_bins[0:-1], mi_pre_hist, 'k', mi_bins[0:-1], mi_post_hist, 'r')
    plt.show()
###########################
    unit2analyze=31
    print('Unit: ' + str(unit2analyze))
    #unit2analyze = 31
    bin_mat_list = bin_data(df, trials_ran_dict, min_run_trials, stim_start=1.50, stim_stop=2.50, bin_size=1.0)

    bin_mat_list_pre = bin_mat_list[0:8]
    #unit_data_pre, pos_data_pre = get_single_neuron_bin_data(bin_mat_list_pre,unit_ind=unit2analyze)
    #best_cmat_pre, c_pre  = fit_logistic_regression(unit_data_pre, pos_data_pre, C_to_try=[1e-3,1e-2,1e-1,1.0], k_folds=10)
    X = get_single_unit_matrix(bin_mat_list_pre, unit=unit2analyze)
    best_cmat_pre = psth_decoder(X)

    best_cmat_pre = best_cmat_pre*(1.0/best_cmat_pre.sum(axis=1)[:, None])
    gs_pre = calc_global_selectivity(best_cmat_pre)
    inv_pre = calc_invariance(best_cmat_pre)
    max_selec_pre = calc_max_selectivity(best_cmat_pre)
    pcc_pre = np.diagonal(best_cmat_pre).sum()/best_cmat_pre.shape[0]*100
    bb_pre  = np.max(np.diagonal(best_cmat_pre))*100
    num_pred_pos_pre = np.sum(np.diagonal(best_cmat_pre) > 1.0/8.0)

    bin_mat_list_post = bin_mat_list[9:-1]
    #unit_data_post, pos_data_post = get_single_neuron_bin_data(bin_mat_list_post,unit_ind=unit2analyze)
    #best_cmat_post, c_post = fit_logistic_regression(unit_data_post, pos_data_post, C_to_try=[1e-3,1e-2,1e-1,1.0], k_folds=10)
    X = get_single_unit_matrix(bin_mat_list_post, unit=unit2analyze)
    best_cmat_post = psth_decoder(X)

    best_cmat_post = best_cmat_post*(1.0/best_cmat_post.sum(axis=1)[:, None])
    gs_post = calc_global_selectivity(best_cmat_post)
    inv_post = calc_invariance(best_cmat_post)
    max_selec_post = calc_max_selectivity(best_cmat_post)
    pcc_post = np.diagonal(best_cmat_post).sum()/best_cmat_post.shape[0]*100
    bb_post  = np.max(np.diagonal(best_cmat_post))*100
    num_pred_pos_post = np.sum(np.diagonal(best_cmat_post) > 1.0/8.0)

    mi_pre  = calc_mi(best_cmat_pre/8.0)
    mi_post = calc_mi(best_cmat_post/8.0)
    tick_pos = np.arange(0, 8)
    tick_labels = np.arange(1, 10).astype(str)

    plt.figure(figsize=(26.5, 7.6))
    plt.subplot(1,3,1)
    make_tuning_curve(em[unit2analyze].reshape(1,18), es[unit2analyze].reshape(1,18), control_pos=9,depth=None,cell_type_list=None,
                                                fig_title=None,share_yax=True)
    plt.title('Unit type: ' + str(cell_type_list[unit2analyze]) + ' '
        'Depth: ' + str(depth[unit2analyze]))

    plt.subplot(1,3,2)
    plt.imshow(best_cmat_pre, vmin=0, vmax=1, interpolation='none', cmap='hot')
    plt.xticks(tick_pos, tick_labels)
    plt.yticks(tick_pos, tick_labels)
    plt.title('Pre-trim ' + 'PCC: ' + "{:.4f}".format(pcc_pre) + '\n'
            + 'Global Sel: ' + "{:.4f}".format(gs_pre) + '\n'
            + 'Mutual Info: ' + "{:.4f}".format(mi_pre))
    plt.colorbar()

    plt.subplot(1,3,3)
    plt.imshow(best_cmat_post, vmin=0, vmax=1, interpolation='none', cmap='hot')
    plt.xticks(tick_pos, tick_labels)
    plt.yticks(tick_pos, tick_labels)
    plt.title('Post-trim ' + 'PCC: ' + "{:.4f}".format(pcc_post) + '\n'
            + 'Global Sel: ' + "{:.4f}".format(gs_post) + '\n'
            + 'Mutual Info: ' + "{:.4f}".format(mi_post))
    plt.colorbar()

    plt.show()

















