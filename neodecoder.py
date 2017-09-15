import seaborn as sns
from sklearn.linear_model import Ridge
import sklearn.metrics as skmetrics
from sklearn.model_selection import KFold


#kappa_to_try = np.arange(1,101,1) # from old code file

class NeuroDecoder(object):
    """
    Decodes positions from data contained in a desing matrix
    This is a child of NeuroAnalyzer

    REMEMBER: any changes to the passed in neo object also occur to the original
    object. They point to the same place in memory. No copy is made.
    """

    def __init__(self, X, y):

        print('\n-----__init__-----')
        # Add design matrix and position data to the class
        self.X = X
        self.y = y - np.min(y) #must start with stimulus ID of zero
        self.num_cond = len(np.unique(y))
        self.num_trials = X.shape[0]


#        # sort by shank/region and then by depth
#        self.__sort_units(neo_obj)
#
#        # add neo object to class instance
#        self.neo_obj         = neo_obj

    def __permute_data(self):
        '''
        permutes data in the design matrix and stimulus ID arrays
        '''

        perm_inds = np.random.permutation(self.num_trials)

        if self.decoder_type == 'ole':
            # map linear positions to circular positions
            # That is, scale values to be between 0 and 2pi.
            step_size       = 2*np.pi/self.num_cond
            self.theta      = self.y*step_size
            self.perm_theta = self.theta[perm_inds]

        # else: do nothing for now

        self.perm_X    = self.X[perm_inds,:]
        self.perm_y    = self.y[perm_inds]

    def fit(self, kind='ole', nfolds=10, kappa_to_try=None):
        """
        fit the specified decoder to the data and specify number of folds

        This fits the specified neural decoder and performs k-fold
        cross-validation to the data. The mean confusion matrix and model
        weights will be added to the class, use dot notation to access them.

        Parameters
        __________
        kind: string
            specify whether to use an optimal linear estimator ('ole') or
            logistic regression ('lr')
        nfolds: int
            how many k-folds to use
        kappa_to_try: array like
            values of kappa to try when using OLE decoder
            Leave set to None when using logistic regression
        """

        # check inputs
        # OLE parameters
        if kind == 'ole' and kappa_to_try is not None:
            print('using optimal linear estimator')
            self.kappa_to_try = np.asarray(kappa_to_try)

        elif kind == 'ole' and kappa_to_try is None:
            print('optimal linear estimator selected but kappa_to_try is None\
                    \nusing default values for kappa: range(0, 100, 1)')
            self.kappa_to_try = np.arange(0, 100)

        # logistic regression parameters
        elif kind == 'lr' and kappa_to_try is None:
            print('using logistic regression')

        elif kind == 'lr' and kappa_to_try is not None:
            print('logistic regression selected but kappa_to_try is not None\
                    \nsetting kappa_to_try to None')
            kappa_to_try = None

        #general parameters
        self.nfolds = nfolds

        # prepare data by permuting data
        # create theta array for OLE decoder
        self.decoder_type = kind
        self.__permute_data()

        # use k-fold cross-validation to fit decoders and generate initial
        # confusion matrices

        if self.decoder_type == 'ole':
            print('fitting OLE decoder')
            self.fit_ole_decoder()

    def fit_ole_decoder(self):
    #(X, pos, theta, kappa_to_try, k_folds=10):
        '''
        Use optimal linear estimation (OLE) to model the instantaneous stimulus
        position using data from many single units using k-fold cross validation.
        '''

        # initialize values used to compare and save parameters from well
        # performing decoding runs
        best_pcc       = 0
        best_kappa     = None
        best_weights   = None
        best_intercept = None

        # theta will be a matrix is used for temporal decoding. theta_k is the mean
        # value of each unique von misses function
        labels  = np.unique(self.y)
        theta_k = np.unique(self.theta)

        kf = KFold(n_splits=self.nfolds)
        for kappa in self.kappa_to_try:
            weights    = list()
            intercepts = list()
            cmats      = list()
            pcc        = list()
            y          = self.__basis_vals(kappa, self.perm_theta, theta_k)

            # perform k-fold cross-validation for the current value of kappa
            # save the parameters for the best performing model
            for train_inds, test_inds in kf.split(self.perm_X):
                assert len(np.intersect1d(train_inds,test_inds)) == 0

                #break the data matrix up into training and test sets
                Xtrain, Xtest, ytrain, _  = self.perm_X[train_inds,:], self.perm_X[test_inds,:],\
                        y[train_inds,], y[test_inds,]

                # make regression object, fit, and get weights
                clf = Ridge(alpha=1.0, fit_intercept=False)
                clf.fit(Xtrain, ytrain)
                W = clf.coef_.T

                # predict the stimulus with the test set
                ypred = list()
                for test_ind in range(Xtest.shape[0]):
                    # iterate through all test trials and predict position
                    ypred.append( self.__predict_stimulus(Xtest[test_ind,:], kappa, theta_k, W) )

                ypred = np.asarray(ypred)
    #           print('predicted: ' + str(ypred))
    #           print('actual   : ' + str(pos[test_inds]))

                # compute confusion matrix
                cmat = skmetrics.confusion_matrix(self.perm_y[test_inds], ypred, labels)
                cmat = cmat.astype(float)

                # normalize each row of the confusion matrix so they represent
                # probabilities
                cmat = (cmat.T/cmat.sum(axis=1)).T
                cmat = np.nan_to_num(cmat)

                # compute the percent correct
                pcc.append(np.trace(cmat, offset=0)/self.num_cond)

                # record confusino matrix for this fold
                cmats.append(cmat)

                # record the weights and intercept
                weights.append(W)
                intercepts.append(clf.intercept_)

            # Compute the mean confusion matrix
            # cmats and mean_pcc get overwritten with ever iteration of the C parameter
            cmats = np.array(cmats)
            cmean = cmats.mean(axis=0)

            # Compute the mean percent correct
            mean_pcc = np.mean(pcc)
            std_pc   = np.std(pcc, ddof=1)

            # Compute the mean weights
            weights        = np.array(weights)
            mean_weights   = weights.mean(axis=0)
            mean_intercept = np.mean(intercepts)

            # Determine if we've found the best model thus far
            if mean_pcc > best_pcc:
                best_pc        = mean_pcc
                best_kappa     = kappa
                best_cmat      = cmean
                best_weights   = mean_weights
                best_intercept = mean_intercept

        # Print best percent correct and plot confusion matrix
        print('Mean percent correct: ' + str(mean_pcc*100) + str('%'))

#        plt.figure()
#        plt.imshow(best_cmat,vmin=0,vmax=1,interpolation='none',cmap='hot')
#        plt.title('PCC: ' + "{:.2f}".format(mean_pcc*100))
#        plt.colorbar()
#        plt.show()
#        print('sum: ' + str(best_cmat.sum(axis=1)))

        self.w        = best_weights
        self.mean_pcc = mean_pcc
        self.cmat     = best_cmat



    def __basis_vals(self, kappa, theta, theta_k):
        """
        Returns a txk array of basis values.

        Used by fit_ole_decoder and predict stimulus
        Von Mises function: e^(cos(x - mu))


        Parameters
            __________

        theta: array or single value
            Theta is a tx1 array of stimulus values. That is, theta is an array
            where the stimulus condition/trial type value has been mapped to be
            between 0 and 2pi.
        theta_k: array like
            theta_k is a kx1 array of coefficients for the von mises basis
            functions. Each stimulus condition has its own von mises function
            and therefor has a corresponding coefficient

        Output
        _____
        B: array
            B is a txk array of basis values where each row corresponds to a
            time point (row) in the condition and data arrays. That is, each
            row is an evaluated von mises function at the current value of
            kappa and the value of theta that corresponds to each stimulus condition
        """

        B = np.zeros((theta.shape[0], theta_k.shape[0]))

        for row, cond in enumerate(theta):
            # evaluate the von mises function at the current value of kappa and
            # for the value of theta that corresponds to the current trial/condition
            B[row, :] = np.exp(kappa*np.cos(cond - theta_k)).reshape(1, theta_k.shape[0])

        return B

    def __predict_stimulus(self, data, kappa, theta_k, W):

    #ypred.append( self.__predict_stimulus(Xtest[test_ind,:], kappa, theta_k, W) )
        # Make sure data is a 1xn vector (n: number of neurons)
        # W is the weight matrix of size nxk (k: number of conditions/von_mises
        # functions)

        # for every stimulus conditions evaluate theta hat (the data (matrix) x
        # weights (matrix) from regression x the values of the evaluated basis
        # functions (vector 1xk))

        theta_val_mat = np.zeros((theta_k.shape[0], 2))
        for row, theta in enumerate(theta_k):
            theta = theta.reshape(1,)
            B = self.__basis_vals(kappa,theta,theta_k)
            theta_val_mat[row, 0] = theta
            theta_val_mat[row, 1] = data.dot(W.dot(B.T))

        max_ind   = np.argmax(theta_val_mat[:, 1])
        theta_hat = theta_val_mat[max_ind, 0]
        #print('predicted condition: ' + str(max_ind))
        return max_ind

##### scratch space #####
##### scratch space #####

fig, ax = plt.subplots(2, 2)

# M1
pos_inds = np.arange(8)
X, y     = neuro.get_design_matrix(trode=0, cond_inds=pos_inds)
decoder  = NeuroDecoder(X, y)
decoder.fit(kind='ole')

im = ax[0][0].imshow(decoder.cmat,vmin=0,vmax=1,interpolation='none',cmap='hot')
ax[0][0].set_title('PCC: ' + "{:.2f}".format(decoder.mean_pcc*100))
fig.colorbar(im, ax=ax[0][0])

# S1
pos_inds = np.arange(8)
X, y     = neuro.get_design_matrix(trode=1, cond_inds=pos_inds)
decoder  = NeuroDecoder(X, y)
decoder.fit(kind='ole')

im = ax[1][0].imshow(decoder.cmat,vmin=0,vmax=1,interpolation='none',cmap='hot')
ax[1][0].set_title('PCC: ' + "{:.2f}".format(decoder.mean_pcc*100))
fig.colorbar(im, ax=ax[1][0])

# M1 + S1 light
pos_inds = np.arange(8)+9
X, y     = neuro.get_design_matrix(trode=0, cond_inds=pos_inds)
decoder  = NeuroDecoder(X, y)
decoder.fit(kind='ole')

im = ax[0][1].imshow(decoder.cmat,vmin=0,vmax=1,interpolation='none',cmap='hot')
ax[0][1].set_title('PCC: ' + "{:.2f}".format(decoder.mean_pcc*100))
fig.colorbar(im, ax=ax[0][1])

# S1 + M1 light
pos_inds = np.arange(8)+9+9
X, y     = neuro.get_design_matrix(trode=1, cond_inds=pos_inds)
decoder  = NeuroDecoder(X, y)
decoder.fit(kind='ole')

im = ax[1][1].imshow(decoder.cmat,vmin=0,vmax=1,interpolation='none',cmap='hot')
ax[1][1].set_title('PCC: ' + "{:.2f}".format(decoder.mean_pcc*100))
fig.colorbar(im, ax=ax[1][1])





#def predict(self):
#    """
#    ughhh
#    """







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

def fit_logistic_regression(unit_data,pos_data,C_to_try=[1e-3,1e-2,1e-1,1.0],k_folds=10):
    '''
    Use logistic regression to fit data from a single unit in order to try and predict
    the stimulus using k-fold cross validation.
    '''
    best_C = None
    best_pcc = 0
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
        pcc= list()
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
                pcc.append(np.trace(cmat, offset=0)/num_pos)

                # record confusino matrix for this fold
                cmats.append(cmat)

                # record the weights and intercept
                weights.append(lr.coef_)
                intercepts.append(lr.intercept_)

        # Compute the mean confusion matrix
        # cmats and mean_pcc get overwritten with ever iteration of the C parameter
        cmats = np.array(cmats)
        Cmean = cmats.mean(axis=0)

        # Compute the mean percent correct
        mean_pcc = np.mean(pcc)
        std_pc  = np.std(pcc,ddof=1)

        # Compute the mean weights
        weights = np.array(weights)
        mean_weights = weights.mean(axis=0)
        mean_intercept = np.mean(intercepts)

        # Determine if we've found the best model thus far
        if mean_pcc > best_pcc:
            best_pcc = mean_pcc
            best_C = C
            best_Cmat = Cmean
            best_weights = mean_weights
            best_intercept = mean_intercept

    # Print best percent correct and plot confusion matrix
    print('Mean percent correct: ' + str(mean_pcc*100) + str('%'))
#    plt.figure()
#    plt.imshow(best_Cmat,vmin=0,vmax=1,interpolation='none',cmap='hot')
#    plt.colorbar()
    #plt.show()
    print('sum: ' + str(best_Cmat.sum(axis=1)))

    return best_Cmat, best_C

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

