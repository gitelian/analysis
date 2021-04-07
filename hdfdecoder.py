import seaborn as sns
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
import os
import time
from sklearn.linear_model import Ridge
import sklearn.metrics as skmetrics
from sklearn.model_selection import KFold
import multiprocessing as mp
import sys
from progress.bar import Bar # doesn't exist?
from tqdm import tqdm

# so I can easily load experiment data without copy/paste
import hdfanalyzer as hdfa

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
        self.num_cond   = len(np.unique(y))
        self.num_trials = X.shape[0]
        self.num_units  = X.shape[1]
        self.uids        = None # will use all the data
        self.num_runs   = 5 # number times decoder finds the best solution by iterating through different kappas

    def __permute_data(self):
        '''
        permutes data in the design matrix and stimulus ID arrays
        '''

        perm_inds = np.random.permutation(self.num_trials)

        if self.decoder_type == 'ole':
            self.perm_theta = self.theta[perm_inds]

        # select a subset of units to decode
        if self.uids is not None:
            perm_X_temp = self.X[perm_inds, :]
            self.perm_X = perm_X_temp[:, self.uids].reshape(perm_inds.shape[0], self.uids.shape[0])
        # use all units
        else:
            self.perm_X    = self.X[perm_inds,:]

        # permute the stimulus IDs
        self.perm_y    = self.y[perm_inds]

    def fit_ole_decoder(self, num_runs=5, plot_cmat=False):
        '''
        Use optimal linear estimation (OLE) to model the instantaneous stimulus
        position using data from many single units using k-fold cross validation.
        '''

        #print('-----Decoding!-----')
        # initialize values used to compare and save parameters from well
        # performing decoding runs
        best_pcc       = 0
        best_kappa     = None
        best_weights   = None
        best_intercept = None

        all_kappas = list()
        all_pcc    = list()
        all_cmats  = list() # collect all the mean cmats and average them together to get a cmat for all runs

        # theta will be a matrix is used for temporal decoding. theta_k is the mean
        # value of each unique von misses function
        labels  = np.unique(self.y)
        theta_k = np.unique(self.theta)

        kf = KFold(n_splits=self.nfolds)
#        bar = Bar('Iterating through runs', max=num_runs)
        for run in range(num_runs):
            #bar.next()
            self.__permute_data()

            for kappa in self.kappa_to_try:
                #print('run {} kappa {}'.format(run, kappa))
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
                        stim_ind = self.__predict_stimulus(Xtest[test_ind, :], kappa, theta_k, W)

                        ypred.append(stim_ind)

                    ypred = np.asarray(ypred)

                    # compute confusion matrix
                    cmat = skmetrics.confusion_matrix(self.perm_y[test_inds], ypred, labels)
                    cmat = cmat.astype(float)

                    # normalize each row of the confusion matrix so they represent
                    # probabilities
                    cmat = (cmat.T/cmat.sum(axis=1)).T
                    cmat = np.nan_to_num(cmat)
                    # make sure the probabilities sum to 1
                    cmat = cmat*(1.0/cmat.sum(axis=1)[:, None])
                    cmat = np.nan_to_num(cmat)

                    # compute the percent correct
                    pcc.append(100* (np.trace(cmat, offset=0)/self.num_cond))

                    # record confusion matrix for this fold
                    cmats.append(cmat)

                    # record the weights and intercept
                    weights.append(W)
                    intercepts.append(clf.intercept_)

                # Compute the mean confusion matrix
                # cmats and mean_pcc get overwritten with ever iteration of the C parameter
                cmats = np.array(cmats)
                cmean = cmats.mean(axis=0)
                all_cmats.append(cmean) # collect all cmats for all runs

                # Compute the mean percent correct
                mean_pcc = np.mean(pcc)
                std_pc   = np.std(pcc, ddof=1)

                # Compute the mean weights
                weights        = np.array(weights)
                mean_weights   = weights.mean(axis=0)
                mean_intercept = np.mean(intercepts)

                # save kappa and mean_pcc for each iteration of kappa tried
                all_kappas.append(kappa)
                all_pcc.append(mean_pcc)

                # Determine if we've found the best model thus far
                if mean_pcc > best_pcc:
                    #print('best so far: {}'.format(mean_pcc))
                    best_pcc       = mean_pcc
                    best_kappa     = kappa
                    best_cmat      = cmean
                    best_weights   = mean_weights
                    best_intercept = mean_intercept
            # END kappa for loop
        # END number of runs loop

        # Print best percent correct and plot best confusion matrix
#        print('Mean percent correct: ' + str(best_pcc) + str('%'))

        if plot_cmat == True:
            plt.figure()
            plt.imshow(best_cmat, vmin=0, vmax=1, interpolation='none', cmap='afmhot')
            plt.title('PCC: ' + "{:.2f}".format(best_pcc))
            plt.colorbar()
            plt.show()
            print('sum: ' + str(best_cmat.sum(axis=1)))

        self.best_pcc   = best_pcc
        self.best_kappa = best_kappa
        self.best_cmat  = best_cmat
        self.w          = best_weights
        self.best_intercept = mean_intercept

        self.all_kappas = all_kappas
        self.all_pcc    = all_pcc
        self.cmat       = np.mean(np.asarray(all_cmats), axis=0)
        self.cmat_sem   = sp.stats.sem(np.asarray(all_cmats), axis=0)
        self.cmat_pcc   = 100 * (np.trace(self.cmat, offset=0)/self.num_cond)
        self.cmat_pcc_sem   = 100 * (np.trace(self.cmat_sem, offset=0)/self.num_cond)

        #bar.finish()


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
        """
        Predict the stimulus from single trial data

        Parameters
        __________

        data: array
            firing rates for each unit for a single trial
        kappa: float
            The kappa value for the von Mises functions
        theta_k: array
            theta_k is a kx1 array of coefficients for the von mises basis
            functions. Each stimulus condition has its own von mises function
            and therefor has a corresponding coefficient
        W: array
            weights for the fitted model

        General notes:
        Make sure data is a 1xn vector (n: number of neurons)
        W is the weight matrix of size nxk (k: number of conditions/von_mises
        functions)

        For every stimulus conditions evaluate theta hat (the data (matrix) x
        weights (matrix) from regression x the values of the evaluated basis
        functions (vector 1xk))
        """

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

    ##### Methods the user can call #####
    ##### Methods the user can call #####

    def fit(self, kind='ole', nfolds=5, num_runs=5, kappa_to_try=None, plot_cmat=False, run=True):
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
        plot_cmat: boolean
            whether to plot the mean confusion matrix after fitting
        run: boolean
            will actually fit the model. Other functions will use 'fit' to set
            parameters without having to run the model (???)
        """

        # check inputs
        # OLE parameters
        if kind == 'ole' and kappa_to_try is not None:
            print('using optimal linear estimator')
            self.kappa_to_try = np.asarray(kappa_to_try)

        elif kind == 'ole' and kappa_to_try is None:
            print('optimal linear estimator selected but kappa_to_try is None\
                    \nusing default values for kappa: range(0, 50, 0.25)')
            self.kappa_to_try = np.arange(0, 50, 0.25)

        # logistic regression parameters
        elif kind == 'lr' and kappa_to_try is None:
            print('using logistic regression')

        elif kind == 'lr' and kappa_to_try is not None:
            print('logistic regression selected but kappa_to_try is not None\
                    \nsetting kappa_to_try to None')
            kappa_to_try = None

        #general parameters
        self.nfolds       = nfolds
        self.decoder_type = kind

        # use k-fold cross-validation to fit decoders and generate initial
        # confusion matrices

        print('----- Model parameters set -----')
        if self.decoder_type == 'ole':

            # map linear positions to circular positions
            # That is, scale values to be between 0 and 2pi.
            step_size  = 2*np.pi/self.num_cond
            self.theta = self.y*step_size

            # fit the decoder (finds the kappa that produces the best decoding)
            # then decode 500 times to get a good average confusion matrix
            if run:
                print('----- fitting OLE decoder -----')
                self.fit_ole_decoder(plot_cmat=plot_cmat, num_runs=num_runs)
                #self.get_pcc_distribution()

    def get_pcc_distribution(self, num_runs=500):
        """
        produces a distribution of PCC for the specified number of runs

        After fitting the model this will use the best kappa to refit, decode,
        and measure the PCC. All the PCCs for each run will be added to
        NeuroDecoder.all_pcc
        """
        # this can only be run after the model is fit
        if hasattr(self, 'best_kappa'):
            self.kappa_to_try = np.array(self.best_kappa).reshape(1,)
            self.num_runs = num_runs
            print('num_runs has been updated to {}'.format(num_runs))
            self.fit_ole_decoder(num_runs=num_runs)
        else:
            print('You must fit the model before creating a PCC distribution')

    def decode_subset(self, niter=50, num_runs=500):
        """
        use only a random subset on units to decode

        Performance vs number of units in model. This will take data in the
        design matrix (trials X num_units) and pull out random subsets of units
        Eventually it will iterate through all subset sizes producing an array
        of iterations X subset size. This can be used to produce a plot to see
        how decoding improves with the size of the neural population


        Parameters
        ----------

        niter: int
            number of iterations per subset
        """

        if not hasattr(self, 'theta'):
            print('must fit model to set parameters first\nyou can use fit with "run=False" so it only sets parameters.')
        pcc_array = np.zeros((niter, self.num_units - 1))

        subset_size = np.arange(2, self.num_units+1) # arange doesn't include the stop range value in the created array
        for n, nsize in enumerate(tqdm(subset_size, ncols=80, desc='Subset size')):
#            print('\n##### Using subset size: {} #####'.format(nsize))

            # recode this to run in parallel?
            # niter is how many times it grabs random units
            for m in tqdm(range(niter), leave=False, ncols=80, desc='Iterations'):
                #print('iteration: {}/{}'.format(m, niter))
                # select subset of units, no repeats, and in order from least to greatest
                self.uids = np.sort(np.random.choice(self.num_units, nsize, replace=False))

                # fit model with new subset and find best kappa
                #self.num_runs = 5
                self.fit_ole_decoder(num_runs=5)
                # fit_ole_decoder sets best_kappa, we feed this in to
                # kappa_to_try to avoid searching again next time
                self.kappa_to_try = np.array(self.best_kappa).reshape(1,)

                # compute mean pcc for this subset and best kappa
                # this will do 500 fit & decoding cycles...giving us a better
                # estimate of pcc for each subset of units and iteration
                #self.num_runs = 500 # 100 was default
                self.fit_ole_decoder(num_runs=num_runs)

                pcc_array[m, n] = np.mean(self.all_pcc)


        return pcc_array











#def predict(self):
#    """
#    ughhh
#    """







##def psth_decoder(X):
##    num_trials = X.shape[0]
##    num_categories = X.shape[2]
##    labels = range(num_categories)
##    predicted = np.zeros((num_categories*num_trials,))
##    actual    = np.zeros((num_categories*num_trials,))
##    loo = cross_validation.LeaveOneOut(num_trials)
##    count = 0
##
##    for train_index, test_index in loo:
##        mean_psth = X[train_index, :, :].mean(axis=0)
###        print('-------------------------')
###        print(mean_psth)
##        for i in range(num_categories):
##            single_trial_data = X[test_index, :, i]
###            print('single trial: ' +str(single_trial_data))
##            # get distance measures
##            dist_temp = np.empty((num_categories,))
##            for k in range(num_categories):
##                dist_temp[k] = np.linalg.norm(mean_psth[:, k] - single_trial_data)
##            # select category with smallest distance
##            choice = np.random.choice(np.where(np.min(dist_temp) == dist_temp)[0])
##            #choice = np.argmin(dist_temp)
###            print("choice: " + str(choice))
##            predicted[count,] = choice
##            actual[count,] = i
##            count += 1
##
##    #compute confusion matrix
##    #This is Mike Schakter's way
##    cmat = confusion_matrix(actual, predicted, labels)
##    cmat = cmat.astype(float)
##    cmat = (cmat.T/cmat.sum(axis=1)).T
##    cmat = np.nan_to_num(cmat)
##    #plt.figure()
##    #plt.imshow(cmat,vmin=0,vmax=1,interpolation='none',cmap='hot')
##
##    return cmat
##
##
## NOTE UNCOMMENT STARTING HERE UNTIL MAIN FUNCTION...PUT THIS SOMEWHERE
## IN THE CLASS???
##def gen_fake_data_multiunit(num_trials=100,num_pos=7,bin_size=0.100,trial_duration=1.0,control_pos=True):
##
##    num_bins = int(trial_duration/bin_size)
##    bin_mat_list = list()
##    bins = np.arange(0,trial_duration+bin_size,bin_size)
##    if control_pos is False:
##        num_pos = 6
##    num_units = num_pos
##
##    for pos in range(num_pos):
##        data_mat = np.zeros((num_units,num_bins,num_trials))
##
##        for trial in range(num_trials):
##            for unit in range(num_units):
##                if unit == 6 or pos == 6:
##                    fr = 5
##                else:
##                    fr = 70.0 - 10.0*(np.abs(pos - unit))
##                # unit firing rate
##                spk_times = list()
##                while sum(spk_times) < trial_duration:
##                    spk_times.append(rd.expovariate(fr))
##                spk_times = np.cumsum(spk_times)
##                data_mat[unit,:,trial] = np.histogram(spk_times,bins)[0]
##        bin_mat_list.append(data_mat)
##
##    return bin_mat_list
##
##
##def gen_fake_data(num_trials=10,num_pos=7.0,bin_size=0.005,trial_duration=1.0,start_rate=5,stop_rate=10):
##
##    num_bins  = int(trial_duration/bin_size)
##    data_mat  = np.zeros((num_pos*num_trials,num_bins))
##    trial_mat = np.zeros((num_pos*num_trials,1))
##    spike_rate = np.arange(start_rate,stop_rate+0.1,(stop_rate - start_rate)/(num_pos -1.0))
##    print('spike rates: ' + str(spike_rate))
##    #spike_rate = np.arange(5,5*num_pos+1,5)
##    bins      = np.arange(0,trial_duration+bin_size,bin_size)
##    count = 0
##
##    for pos in range(num_pos):
##        lambda_rate = spike_rate[pos]
##
##        for trial in range(num_trials):
##            spk_times = list()
##
##            while sum(spk_times) < trial_duration:
##                spk_times.append(rd.expovariate(lambda_rate))
##            spk_times = np.cumsum(spk_times)
##
##            data_mat[count,:] = np.histogram(spk_times,bins)[0]
##            trial_mat[count] = pos
##            count += 1
##
##    permuted_inds = np.random.permutation(count)
##    perm_data_mat = data_mat[permuted_inds,:]
##    perm_trial_mat = trial_mat[permuted_inds,:]
##
##    return perm_data_mat, perm_trial_mat
##
##def fit_logistic_regression(unit_data,pos_data,C_to_try=[1e-3,1e-2,1e-1,1.0],k_folds=10):
##    '''
##    Use logistic regression to fit data from a single unit in order to try and predict
##    the stimulus using k-fold cross validation.
##    '''
##    best_C = None
##    best_pcc = 0
##    best_weights = None
##    best_intercept = None
##    labels = np.unique(pos_data)
##    num_pos = float(len(labels))
##    pos_data = np.ravel(pos_data)
##
##    unit_data_original = unit_data
##    pos_data_original  = pos_data
##
##    for C in C_to_try:
##        weights = list()
##        intercepts = list()
##        cmats = list()
##        pcc= list()
##        for a in range(100):
##
##            # Randomize data
##            permuted_indices = np.random.permutation(pos_data_original.shape[0])
##            unit_data = unit_data_original[permuted_indices, :]
##            pos_data  = pos_data_original[permuted_indices,]
##
##            # Fit model and test with k-fold cross validation
##            for train_inds, test_inds in KFold(len(pos_data),k_folds):
##                assert len(np.intersect1d(train_inds,test_inds)) == 0
##                #break the data matrix up into training and test sets
##                Xtrain, Xtest, ytrain, ytest = unit_data[train_inds], unit_data[test_inds], pos_data[train_inds], pos_data[test_inds]
##
##                # make logistic regression object
##                lr = LogisticRegression(C=C)
##                lr.fit(Xtrain, ytrain)
##
##                # predict the stimulus with the test set
##                ypred = lr.predict(Xtest)
##
##                # compute confusion matrix
##                cmat = confusion_matrix(ytest,ypred,labels)
##                cmat = cmat.astype(float)
##
##                #computed. This was Mike Schakter's way...I think it is wrong.
##                cmat = (cmat.T/cmat.sum(axis=1)).T
##                cmat = np.nan_to_num(cmat)
##                # normalize each row of the confusion matrix so they represent
##                # probabilities
##            #    cmat = cmat.T/cmat.sum(axis=1).astype(float)
##            #    cmat = np.nan_to_num(cmat)
##
##                # compute the percent correct
##                pcc.append(np.trace(cmat, offset=0)/num_pos)
##
##                # record confusino matrix for this fold
##                cmats.append(cmat)
##
##                # record the weights and intercept
##                weights.append(lr.coef_)
##                intercepts.append(lr.intercept_)
##
##        # Compute the mean confusion matrix
##        # cmats and mean_pcc get overwritten with ever iteration of the C parameter
##        cmats = np.array(cmats)
##        Cmean = cmats.mean(axis=0)
##
##        # Compute the mean percent correct
##        mean_pcc = np.mean(pcc)
##        std_pc  = np.std(pcc,ddof=1)
##
##        # Compute the mean weights
##        weights = np.array(weights)
##        mean_weights = weights.mean(axis=0)
##        mean_intercept = np.mean(intercepts)
##
##        # Determine if we've found the best model thus far
##        if mean_pcc > best_pcc:
##            best_pcc = mean_pcc
##            best_C = C
##            best_Cmat = Cmean
##            best_weights = mean_weights
##            best_intercept = mean_intercept
##
##    # Print best percent correct and plot confusion matrix
##    print('Mean percent correct: ' + str(mean_pcc*100) + str('%'))
###    plt.figure()
###    plt.imshow(best_Cmat,vmin=0,vmax=1,interpolation='none',cmap='hot')
###    plt.colorbar()
##    #plt.show()
##    print('sum: ' + str(best_Cmat.sum(axis=1)))
##
##    return best_Cmat, best_C
##
##def calc_global_selectivity(cmat):
##    num_pos = float(cmat.shape[0])
##    pcc = np.diagonal(cmat)*(1.0/np.trace(cmat))
###    pcc = np.diagonal(cmat)/num_pos
##
##    h_obs = np.sum(np.nan_to_num(-pcc*np.log2(pcc)))
##    h_max = -np.log2(1.0/num_pos)
##
##    gs = 1.0 - (h_obs/h_max)
##
##    return gs
##
##def calc_max_selectivity(cmat):
##    num_pos = float(cmat.shape[0])
##    pcc = np.diagonal(cmat)
##
##    pref_pos = np.argmax(pcc)
##    other_positions = np.delete(np.arange(num_pos), pref_pos).astype(int)
##    max_selectivity = np.log2( ((num_pos - 1)*pcc[pref_pos])/(np.sum(pcc[other_positions])))
##
##    return max_selectivity
##
##def calc_invariance(cmat):
##    num_pos = float(cmat.shape[0])
##    pcc = np.diagonal(cmat)/num_pos
##
##    h_obs = np.sum(np.nan_to_num(-cmat*np.log2(cmat)))
##    h_max = -np.log2(1/num_pos**2)
##
##    h_min = 0;
##    for row in range(cmat.shape[0]):
##        for col in range(cmat.shape[1]):
##            Pj_of_i = np.sum(cmat[row, :])
##        h_min += np.sum(-Pj_of_i * np.log2(Pj_of_i))
##
##    invariance = (h_obs - h_min)/(h_max - h_min)
##
##    return invariance
##
##def calc_mi(cmat):
##    I = cmat.shape[0]
##    J = cmat.shape[1]
##    mi = 0
##    for i in range(int(I)):
##        for j in range(int(J)):
##            P_ij = cmat[i, j]
##            P_i  = 1.0/I # Assumes that equal number of trials for each
##                         # for each category were used in the decoding process
##            P_j  = cmat[:, j].sum()#/float(J)
##
##            mi_temp = np.nan_to_num(P_ij*np.log2(P_ij/(P_i*P_j)))
##            mi += mi_temp
##    return mi
##
########## MAIN CODE ##########
########## MAIN CODE ##########

if __name__ == "__main__":

##### NOTES FID1340, S1 decoding improves with M1 silencing...lol
##### evk_rates dont do as well as absolute rates -- check with other datasets
#### TODO add a way to decode new data with old weights
####pcc_exp
####array([[38., 64.], #fid1336, m1 vs s1 pcc (no light)
####    [43., 53.],    #fid1338  "  "  "
####    [34., 56.],
####    [31., 42.],
####    [23., 36.],
####    [38., 57.]])
####
#### TODO plot change in PCC distributions (Light - NoLight)

#### TABLE OF CONTENTS
##### Cross-Experiment decoding: combine data across all experiments     #####
##### Regular decoding: plot confusion matrices for combinations of units#####
##### Single-Experiment: Computes PCC for random subsamples of units     #####
##### Single-Experiment:Use SAVED DATA to remake PCC vs Num Units figures#####
##### Single-Multi: decode each experiment independently & combine metrics#####
#####                    Parallel processing                             #####
#####    Scratch space: predict whisker angle from neural data??         #####


##############################################################################
##### Cross-Experiment decoding: combine data across all experiments     #####
##############################################################################
#####
#####

    def remove_correlations(design_matrix, stim_labels):
        """
        Removes correlations units have within trials
        The mean rates are unchanged. Each column of the design matrix has been
        shuffled within each unique stimulus position

        """
        # np.apply_along_axis( function1d, axis, array), will apply function1d to
        # every column independently
        # np.random.permutation() shuffles contents of columns BUT all columns are
        # permuted the same...so correlations between units are NOT REMOVED
        # This is equivalent to permuted trials

        Xcorr = np.zeros(design_matrix.shape)

        for stim_id in np.unique(stim_labels):
            stim_inds = np.where(stim_labels == stim_id)[0]
            current_vals = design_matrix[stim_inds, :]
            shuffled_vals = np.apply_along_axis(np.random.permutation, 0, current_vals)
            Xcorr[stim_inds, :] = shuffled_vals

        return Xcorr

    ############## LOAD DATA ##########
    #from hdfanalyzer import *
    fids = ['1336', '1338', '1339', '1340', '1345'] # skipping 1343 for now...low trial counts
#    fids = ['1336', '1338']#, '1339', '1340', '1345'] # skipping 1343 for now...low trial counts
    ### collect individual experiments and store in a list ###
    experiments = list()
    min_trials = 30
    for fid_name in fids:

        if hdfa.os.path.isdir('/Users/Greg/Dropbox/A1Thesis/data/hdf5/'):
            data_dir = '/Users/Greg/Dropbox/A1Thesis/data/hdf5/'
        elif hdfa.os.path.isdir('/media/greg/data/neuro/hdf5/'):
            data_dir = '/media/greg/data/neuro/hdf5/'

        #manager = NeoHdf5IO(os.path.join(data_dir + 'FID1295_neo_object.h5'))
        #print(sys.argv)
        try:
            fid = 'FID' + fid_name
        except:
            print('no argument provided!')

        f = hdfa.h5py.File(hdfa.os.path.join(data_dir + 'FID' + fid_name + '.hdf5'),'r+')

        neuro = hdfa.NeuroAnalyzer(f, fid)
        f.close()

        experiments.append(neuro)
    ############## LOAD DATA ##########


    ############## Combine data and create two models: vM1 and vS1 ##########
    npand = np.logical_and
    pos_inds = np.arange(8)
    min_trials = 30

    Xm = np.zeros((min_trials*8, 1))
    Xs = np.zeros((min_trials*8, 1))

    for n in experiments:
        uind_m1 = np.where( npand(n.driven_units == True, n.shank_ids == 0) )[0]
        Xm1, ym1, _ = n.get_design_matrix(min_trials=min_trials, unit_inds=uind_m1, cond_inds=pos_inds, rate_type='abs_count')
        Xm = np.concatenate((Xm, Xm1), axis=1)

        uind_s1 = np.where( npand(n.driven_units == True, n.shank_ids == 1) )[0]
        Xs1, ys1, _ = n.get_design_matrix(min_trials=min_trials, unit_inds=uind_s1, cond_inds=pos_inds, rate_type='abs_count')
        Xs = np.concatenate((Xs, Xs1), axis=1)
    Xm = Xm[:, 1::]
    Xs = Xs[:, 1::]

    save_dir= '/home/greg/Desktop/desktop2dropbox/decoder/'
    save_name = 'design_matrices_no1343_'
    fname = save_dir + save_name + time.strftime('%Y%m%d_%Hh%Mm') + '.mat'
    sp.io.savemat(fname, {'Xm':Xm, 'Xs':Xs, 'ym1':ym1, 'ys1':ys1, 'fids':fids})


    ##### Basic Decoding #####
    ### M1 combo decode ###
    print('----- vM1 basic decoding -----\n')
    m1c = NeuroDecoder(Xm, ym1)
    m1c.fit(kind='ole', plot_cmat=False, run=True)
    m1c.get_pcc_distribution(num_runs=1000)

    ### S1 combo decode ###
    print('----- vS1 basic decoding -----\n')
    s1c = NeuroDecoder(Xs, ys1)
    s1c.fit(kind='ole', plot_cmat=False, run=True)
    s1c.get_pcc_distribution(num_runs=1000)

    #### Plot confusion matrix and PCC distributions before calculating subsets
    fig, ax = plt.subplots(1,3, figsize=(15.0, 3.0))
    im_m1  = ax[0].imshow(m1c.cmat, vmin=0, vmax=1, interpolation='none', cmap='afmhot')
    im_s1  = ax[1].imshow(s1c.cmat, vmin=0, vmax=1, interpolation='none', cmap='afmhot')
    fig.colorbar(im_m1,  ax=ax[0]); ax[0].set_title('vM1 PCC: ' + "{:.1f}".format(m1c.cmat_pcc))
    fig.colorbar(im_s1,  ax=ax[1]); ax[1].set_title('vS1 PCC: ' + "{:.1f}".format(s1c.cmat_pcc))
    ax[2].hist(m1c.all_pcc, bins=np.arange(0, 100, 1), density=True, alpha=0.5,\
            color='tab:blue', align='left', cumulative=False)
    ax[2].hist(s1c.all_pcc, bins=np.arange(0, 100, 1), density=True, alpha=0.5,\
            color='tab:red', align='left', cumulative=False)
    ax[2].set_xlim(50,100)
    ax[2].legend(['vM1', 'vS1'], loc='upper left')

    save_name = 'm1_s1_figure_simple_decoding_nolight'
    fname = save_dir + save_name + time.strftime('%Y%m%d_%Hh%mm') + '.pdf'
    fig.savefig(fname)
    plt.close(fig)

    ### M1 and S1 combo save ###
    save_name = 'm1_s1_simple_decoding_values_'
    fname = save_dir + save_name + time.strftime('%Y%m%d_%Hh%Mm') + '.mat'
    sp.io.savemat(fname, {'Xm':Xm, 'Xs':Xs, 'ym1':ym1, 'ys1':ys1, 'fids':fids,\
            'm1c_pccs':m1c.all_pcc, 'm1c_cmat':m1c.cmat, 'm1c_cmat_pcc':m1c.cmat_pcc,\
            's1c_pccs':s1c.all_pcc, 's1c_cmat':s1c.cmat, 's1c_cmat_pcc':s1c.cmat_pcc})


    ##### M1 and S1 subset decoding #####
    print('----- vM1 subset decoding -----\n')
    m1c_pcc_array = m1c.decode_subset(niter=50, num_runs=500)

    save_name = 'm1_subsampling_units_decoding_'
    fname = save_dir + save_name + time.strftime('%Y%m%d_%Hh%Mm') + '.mat'
    sp.io.savemat(fname, {'Xm':Xm, 'ym1':ym1, 'fids':fids,\
            'm1c_pccs':m1c.all_pcc, 'm1c_cmat':m1c.cmat,\
            'm1c_cmat_pcc':m1c.cmat_pcc, 'm1c_pcc_array':m1c_pcc_array})

    print('----- vS1 subset decoding -----\n')
    s1c_pcc_array = s1c.decode_subset(niter=50, num_runs=500)

    save_name = 's1_subsampling_units_decoding_'
    fname = save_dir + save_name + time.strftime('%Y%m%d_%Hh%Mm') + '.mat'
    sp.io.savemat(fname, {'Xs':Xs, ,'ys1':ys1, 'fids':fids,\
            's1c_pccs':s1c.all_pcc, 's1c_cmat':s1c.cmat,\
            's1c_cmat_pcc':s1c.cmat_pcc, 's1c_pcc_array':s1c_pcc_array})


##### Add subsample figure and save it #####
##### Add subsample figure and save it #####
##    s1c_pcc_array = s1c.decode_subset(niter=5, num_runs=20)
##    s1c_mean_pcc  = s1c_pcc_array.mean(axis=0)
##    s1c_std_pcc   = s1c_pcc_array.std(axis=0)
##    plt.figure()
##    plt.errorbar(np.arange(2, s1c_mean_pcc.shape[0]+2), s1c_mean_pcc, yerr=s1c_std_pcc,\
##            marker='o', markersize=6.0, linewidth=2, color='k')



##    ###################### MANUAL DECODING SPACE ########################
##    npand = np.logical_and
##    pos_inds = np.arange(8)
##    min_trials = 30
##
##    #### M1 ####
##    uind_m1 = np.where( npand(n1.driven_units == True, n1.shank_ids == 0) )[0]
##    Xm1, ym1, _ = n1.get_design_matrix(min_trials=min_trials, unit_inds=uind_m1, cond_inds=pos_inds, rate_type='abs_count')
##
##    uind_m2 = np.where( npand(n2.driven_units == True, n2.shank_ids == 0) )[0]
##    Xm2, ym2, _ = n2.get_design_matrix(min_trials=min_trials, unit_inds=uind_m2, cond_inds=pos_inds, rate_type='abs_count')
##
##    # check ym1 == ym2 then proceede
##    Xm= np.concatenate((Xm1, Xm2), axis=1)
##    ym= ym1
##
##    ## M1 decode!
##    m1c = NeuroDecoder(Xm, ym)
##    m1c.fit(kind='ole', plot_cmat=True, run=True)
##    m1c.get_pcc_distribution(num_runs=500)
##
##
##    #### S1 ####
##    npand = np.logical_and
##    pos_inds = np.arange(8)
##    min_trials = 30
##
##    uind_s1 = np.where( npand(n1.driven_units == True, n1.shank_ids == 1) )[0]
##    Xs1, ys1, _ = n1.get_design_matrix(min_trials=min_trials, unit_inds=uind_s1, cond_inds=pos_inds, rate_type='abs_count')
##
##    uind_s2 = np.where( npand(n2.driven_units == True, n2.shank_ids == 1) )[0]
##    Xs2, ys2, _ = n2.get_design_matrix(min_trials=min_trials, unit_inds=uind_s2, cond_inds=pos_inds, rate_type='abs_count')
##
##    # check ys1 == ys2 then proceede
##    Xs = np.concatenate((Xs1, Xs2), axis=1)
##    ys = ys1
##
##    ## S1 decode!
##    s1c = NeuroDecoder(Xs, ys)
##    s1c.fit(kind='ole', plot_cmat=True, run=True)
##    s1c.get_pcc_distribution(num_runs=500)
##
##    #### Plot confusion matrix and PCC distributions before calculating subsets
##    fig, ax = plt.subplots(1,3, figsize=(15.0, 3.0))
##    im_m1  = ax[0].imshow(m1c.cmat, vmin=0, vmax=1, interpolation='none', cmap='afmhot')
##    im_s1  = ax[1].imshow(s1c.cmat, vmin=0, vmax=1, interpolation='none', cmap='afmhot')
##    fig.colorbar(im_m1,  ax=ax[0]); ax[0].set_title('vM1 PCC: ' + "{:.1f}".format(m1c.cmat_pcc))
##    fig.colorbar(im_s1,  ax=ax[1]); ax[1].set_title('vS1 PCC: ' + "{:.1f}".format(s1c.cmat_pcc))
##    ax[2].hist(m1c.all_pcc, bins=np.arange(0, 100, 1), density=True, alpha=0.5,\
##            color='tab:blue', align='left', cumulative=False)
##    ax[2].hist(s1c.all_pcc, bins=np.arange(0, 100, 1), density=True, alpha=0.5,\
##            color='tab:red', align='left', cumulative=False)
##    ax[2].set_xlim(50,100)
##    ax[2].legend(['vM1', 'vS1'], loc='upper left')
##
##
##    ## S1 subset: PCC per size of sample
##    ## Randomly take data from x unique units and decode, repeat niter time,
##    ## iterate through all sample sizes
##
##    s1c_pcc_array = s1c.decode_subset(niter=5, num_runs=20)
##    s1c_mean_pcc  = s1c_pcc_array.mean(axis=0)
##    s1c_std_pcc   = s1c_pcc_array.std(axis=0)
##    plt.figure()
##    plt.errorbar(np.arange(2, s1c_mean_pcc.shape[0]+2), s1c_mean_pcc, yerr=s1c_std_pcc,\
##            marker='o', markersize=6.0, linewidth=2, color='k')
##

##############################################################################
##### Regular decoding: plot confusion matrices for combinations of units#####
##############################################################################

##### regular decoding: find best kappa and plot confusion matrices
##### also plot performance vs kappas

## NOTE UNCOMMENT ## STARTING FROM HERE TO END OF FILE
##npand = np.logical_and
##pos_inds = np.arange(8)
##
#### M1 driven units
##unit_inds = np.where( npand(neuro.driven_units == True, neuro.shank_ids == 0) )[0]
##Xm, ym, uinds = neuro.get_design_matrix(unit_inds=unit_inds, cond_inds=pos_inds, rate_type='abs_count')
##
##m1d = NeuroDecoder(Xm, ym)
##m1d.fit(kind='ole', plot_cmat=True, run=True)
##m1d.get_pcc_distribution(num_runs=500)
##
##m1cor = NeuroDecoder(Xcorr, ym)
##m1cor.fit(kind='ole', plot_cmat=True, run=True)
##m1cor.get_pcc_distribution(num_runs=500)
##m1_cmat = m1d.cmat
##m1_pcc  = m1d.cmat_pcc
##
##
#### S1 driven units
##unit_inds = np.where( npand(neuro.driven_units == True, neuro.shank_ids == 1) )[0]
##Xs, ys, uinds = neuro.get_design_matrix(unit_inds=unit_inds, cond_inds=pos_inds, rate_type='abs_count')
##
###s1d = NeuroDecoder(Xcorr, ys)
##s1d = NeuroDecoder(Xs, ys)
##s1d.fit(kind='ole', plot_cmat=True, run=True)
##s1d.get_pcc_distribution(num_runs=500)
##s1pcc = s1d.all_pcc
##s1_cmat = s1d.cmat
##s1_pcc  = s1d.cmat_pcc
##
##
#### M1 and S1 driven units
##unit_inds = np.where(neuro.driven_units == True)[0]
##Xms, yms, uinds = neuro.get_design_matrix(unit_inds=unit_inds, cond_inds=pos_inds, rate_type='abs_count')
##
##ms1d = NeuroDecoder(Xms, yms)
##ms1d.fit(kind='ole', plot_cmat=True, run=True)
##ms1d.get_pcc_distribution(num_runs=500)
##ms1_cmat = ms1d.cmat
##ms1_pcc  = ms1d.cmat_pcc
##
###### Silencing condition and summary plot (violin/hist/???) ####
##
#### M1 driven units
##unit_inds = np.where( npand(neuro.driven_units == True, neuro.shank_ids == 0) )[0]
##XmL, ymL, uinds = neuro.get_design_matrix(unit_inds=unit_inds, cond_inds=pos_inds+9, rate_type='abs_count')
##
##m1Ld = NeuroDecoder(XmL, ymL)
##m1Ld.fit(kind='ole', plot_cmat=True, run=True)
##m1Ld.get_pcc_distribution(num_runs=500)
##m1L_cmat = m1Ld.cmat
##m1L_pcc  = m1Ld.cmat_pcc
##
##
#### S1 driven units
##unit_inds = np.where( npand(neuro.driven_units == True, neuro.shank_ids == 1) )[0]
##XsL, ysL, uinds = neuro.get_design_matrix(unit_inds=unit_inds, cond_inds=pos_inds+9+9, rate_type='abs_count')
##
##s1Ld = NeuroDecoder(XsL, ysL)
##s1Ld.fit(kind='ole', plot_cmat=True, run=True)
##s1Ld.get_pcc_distribution(num_runs=500)
##s1L_cmat = s1Ld.cmat
##s1L_pcc  = s1Ld.cmat_pcc
##
#### TODO histogram of PCC distributions for the five conditions above
##
##
##fig, ax = plt.subplots(2, 3)
##cmap ='afmhot' # 'inferno' 'afmhot', 'hot'
##vmax = 0.7
###vmax = np.max(np.asarray([m1_cmat.ravel(), s1_cmat.ravel(), m1L_cmat.ravel(), s1L_cmat.ravel()]))
##
##### Top of the figure ###
##im_m1  = ax[0][0].imshow(m1_cmat, vmin=0, vmax=vmax, interpolation='none', cmap=cmap)
##im_s1  = ax[0][1].imshow(s1_cmat, vmin=0, vmax=vmax, interpolation='none', cmap=cmap)
##im_ms1 = ax[0][2].imshow(ms1_cmat, vmin=0, vmax=vmax, interpolation='none', cmap=cmap)
##
##fig.colorbar(im_m1,  ax=ax[0][0]); ax[0][0].set_title('PCC: ' + "{:.1f}".format(m1_pcc))
##fig.colorbar(im_s1,  ax=ax[0][1]); ax[0][1].set_title('PCC: ' + "{:.1f}".format(s1_pcc))
##fig.colorbar(im_ms1, ax=ax[0][2]); ax[0][2].set_title('PCC: ' + "{:.1f}".format(ms1_pcc))
##
##### Bottom of the figure ###
##im_m1L  = ax[1][0].imshow(m1_cmat, vmin=0, vmax=vmax, interpolation='none', cmap=cmap)
##im_s1L  = ax[1][1].imshow(s1_cmat, vmin=0, vmax=vmax, interpolation='none', cmap=cmap)
###im_ms1 = ax[0][2].imshow(ms1_cmat, vmin=0, vmax=vmax, interpolation='none', cmap=cmap)
##
##fig.colorbar(im_m1L,  ax=ax[1][0]); ax[1][0].set_title('PCC: ' + "{:.1f}".format(m1L_pcc))
##fig.colorbar(im_s1L,  ax=ax[1][1]); ax[1][1].set_title('PCC: ' + "{:.1f}".format(s1L_pcc))
###fig.colorbar(im_ms1, ax=ax[0][2]); ax[0][2].set_title('PCC: ' + "{:.1f}".format(ms1_pcc))
##
##
#### plot scatter plot
###fig, ax = plt.subplots(1, 1)
###ax.scatter(all_kappas, all_pcc, s=1.0)
##
##
################################################################################
####### Single-Experiment: Computes PCC for random subsamples of units     #####
################################################################################
##
####### performance per unit #####
####### must have a neuro class loaded
##
#### M1
##print('M1 no light')
##pos_inds = np.arange(8)+9+9
##X, y, uinds     = neuro.get_design_matrix(trode=1, cond_inds=pos_inds, rate_type='abs_count')
##decoder  = NeuroDecoder(X, y)
##decoder.fit(kind='ole', plot_cmat=True, run=True)
##decoder.get_pcc_distribution(num_runs=500)
###decoder.fit_ole_decoder(num_runs=10,plot_cmat=True)
##m1_pcc_array = decoder.decode_subset(niter=5, num_runs=20)
##m1_mean_pcc  = m1_pcc_array.mean(axis=0)
##m1_std_pcc   = m1_pcc_array.std(axis=0)
##
#### M1 S1 light
##print('M1 S1 light')
##pos_inds = np.arange(8)+9
##X, y, uinds     = neuro.get_design_matrix(trode=0, cond_inds=pos_inds, rate_type='abs_count')
##decoder  = NeuroDecoder(X, y)
##decoder.fit(kind='ole', run=False)
##s1_light_pcc_array = decoder.decode_subset()
##s1_light_mean_pcc  = s1_light_pcc_array.mean(axis=0)
##s1_light_std_pcc   = s1_light_pcc_array.std(axis=0)
##
##plt.figure()
##plt.errorbar(np.arange(2, m1_mean_pcc.shape[0]+2), m1_mean_pcc, yerr=m1_std_pcc,\
##        marker='o', markersize=6.0, linewidth=2, color='k')
##plt.errorbar(np.arange(2, s1_light_mean_pcc.shape[0]+2), s1_light_mean_pcc, yerr=s1_light_std_pcc,\
##        marker='o', markersize=6.0, linewidth=2, color='r')
##plt.hlines(16, plt.xlim()[0], plt.xlim()[1], colors='k', linestyles='dashed')
##plt.xlabel('number of units in decoder')
##plt.ylabel('Decoding performance (PCC)')
##plt.title(neuro.fid + ' vM1 decoding performance')
##
#### S1
##print('S1 no light')
##pos_inds = np.arange(8)
##X, y, uinds     = neuro.get_design_matrix(trode=1, cond_inds=pos_inds, rate_type='abs_count')
##decoder  = NeuroDecoder(X, y)
##decoder.fit(kind='ole', run=False)
##s1_pcc_array = decoder.decode_subset()
##s1_mean_pcc  = s1_pcc_array.mean(axis=0)
##s1_std_pcc   = s1_pcc_array.std(axis=0)
##
### S1 M1 light
##print('S1 M1 light')
##pos_inds = np.arange(8)+9+9
##X, y, uinds     = neuro.get_design_matrix(trode=1, cond_inds=pos_inds, rate_type='abs_count')
##decoder  = NeuroDecoder(X, y)
##decoder.fit(kind='ole', run=False)
##m1_light_pcc_array = decoder.decode_subset()
##m1_light_mean_pcc  = m1_light_pcc_array.mean(axis=0)
##m1_light_std_pcc   = m1_light_pcc_array.std(axis=0)
##
##plt.figure()
##plt.errorbar(np.arange(2, s1_mean_pcc.shape[0]+2), s1_mean_pcc, yerr=s1_std_pcc,\
##        marker='o', markersize=6.0, linewidth=2, color='k')
##plt.errorbar(np.arange(2, m1_light_mean_pcc.shape[0]+2), m1_light_mean_pcc, yerr=m1_light_std_pcc,\
##        marker='o', markersize=6.0, linewidth=2, color='r')
##plt.hlines(16, plt.xlim()[0], plt.xlim()[1], colors='k', linestyles='dashed')
##plt.xlabel('number of units in decoder')
##plt.ylabel('Decoding performance (PCC)')
##plt.title(neuro.fid + ' vS1 decoding performance')
##
##
### save pcc arrays
##sp.io.savemat('/home/greg/Desktop/decoding_unit_performance/pcc_arrays/' + fid + '_pcc.mat',
##        {'m1_pcc_array':m1_pcc_array,
##            's1_light_pcc_array':s1_light_pcc_array,
##            's1_pcc_array':s1_pcc_array,
##            'm1_light_pcc_array':m1_light_pcc_array})
##
##
##
################################################################################
####### Single-Experiment:Use SAVED DATA to remake PCC vs Num Units figures#####
################################################################################
####### make pcc vs unit figures #####
####### make pcc vs unit figures #####
#######
####### Uses previously computed pcc arrays to immediately remake figures #####
##pcc_exp
##array([[38., 64.], #fid1336, m1 vs s1 pcc (no light)
##    [43., 53.],    #fid1338  "  "  "
##    [34., 56.],
##    [31., 42.],
##    [23., 36.],
##    [38., 57.]])
###dir_path = '/home/greg/Desktop/decoding_unit_performance/pcc_arrays/'
##dir_path = '/media/greg/data/neuro/figures/decoding_unit_performance/pcc_arrays'
##file_list = os.listdir(dir_path)
##
##xmin, xmax = 0, 30
##ymin, ymax = 0, 70
##
##for fname in file_list:
##    # open file
##    temp = sp.io.loadmat(dir_path + os.sep + fname)
##
##    # extract variables
##    m1_pcc_array = temp['m1_pcc_array']
##    m1_mean_pcc  = m1_pcc_array.mean(axis=0)
##    m1_std_pcc   = m1_pcc_array.std(axis=0)
##
##    s1_light_pcc_array = temp['s1_light_pcc_array']
##    s1_light_mean_pcc  = s1_light_pcc_array.mean(axis=0)
##    s1_light_std_pcc   = s1_light_pcc_array.std(axis=0)
##
##    s1_pcc_array = temp['s1_pcc_array']
##    s1_mean_pcc  = s1_pcc_array.mean(axis=0)
##    s1_std_pcc   = s1_pcc_array.std(axis=0)
##
##    m1_light_pcc_array = temp['m1_light_pcc_array']
##    m1_light_mean_pcc  = m1_light_pcc_array.mean(axis=0)
##    m1_light_std_pcc   = m1_light_pcc_array.std(axis=0)
##
##    # make figure
##    fig, ax = plt.subplots(1, 2, figsize=(10,5))
##
##    # plot M1 data
##    ax[0].errorbar(np.arange(2, m1_mean_pcc.shape[0]+2), m1_mean_pcc, yerr=m1_std_pcc,\
##            marker='o', markersize=6.0, linewidth=2, color='k')
##    ax[0].errorbar(np.arange(2, s1_light_mean_pcc.shape[0]+2), s1_light_mean_pcc, yerr=s1_light_std_pcc,\
##            marker='o', markersize=6.0, linewidth=2, color='r')
##    ax[0].hlines(16, xmin, xmax, colors='k', linestyles='dashed')
##    ax[0].set_xlabel('number of units in decoder')
##    ax[0].set_ylabel('Decoding performance (PCC)')
##    ax[0].set_title(fname + ' vM1 decoding performance')
##
##    # plot S1 data
##    ax[1].errorbar(np.arange(2, s1_mean_pcc.shape[0]+2), s1_mean_pcc, yerr=s1_std_pcc,\
##            marker='o', markersize=6.0, linewidth=2, color='k')
##    ax[1].errorbar(np.arange(2, m1_light_mean_pcc.shape[0]+2), m1_light_mean_pcc, yerr=m1_light_std_pcc,\
##            marker='o', markersize=6.0, linewidth=2, color='r')
##    ax[1].hlines(16, xmin, xmax, colors='k', linestyles='dashed')
##    ax[1].set_xlabel('number of units in decoder')
##    ax[1].set_ylabel('Decoding performance (PCC)')
##    ax[1].set_title(fname + ' vS1 decoding performance')
##
##    # set x and y limits
##    ax[0].set_xlim(xmin, xmax)
##    ax[0].set_ylim(ymin, ymax)
##    ax[1].set_xlim(xmin, xmax)
##    ax[1].set_ylim(ymin, ymax)
##
##    # save figure
##    #fig.savefig('/home/greg/Desktop/decoding_unit_performance/' +  fname[0:-8] + '.pdf')
##    fig.savefig('/home/greg/Desktop/desktop2dropbox/' +  fname[0:-8] + '.pdf')
##
##
################################################################################
####### Single-Multi: decode each experiment independently & combine metrics#####
################################################################################
##
####### make pcc vs selectivity figures/correlation analysis #####
####### make pcc vs selectivity figures/correlation analysis #####
##
##dir_path = '/media/greg/data/neuro/figures/decoding_unit_performance/pcc_arrays'
##npand   = np.logical_and
##file_list = np.sort(os.listdir(dir_path))
##exps = list()
##m1_sel_nolight = list()
##m1_pcc_nolight = list()
##
##m1_sel_s1light = list()
##m1_pcc_s1light = list()
##
##s1_sel_nolight = list()
##s1_pcc_nolight = list()
##
##s1_sel_m1light = list()
##s1_pcc_m1light = list()
##
##m1_depth       = list()
##s1_depth       = list()
##
##for fname in file_list:
##    # open file
##    temp = sp.io.loadmat(dir_path + os.sep + fname)
##
##    # extract variables
##    m1_pcc_array = temp['m1_pcc_array']
##    m1_mean_pcc  = m1_pcc_array.mean(axis=0)
##    m1_std_pcc   = m1_pcc_array.std(axis=0)
##
##    s1_light_pcc_array = temp['s1_light_pcc_array']
##    s1_light_mean_pcc  = s1_light_pcc_array.mean(axis=0)
##    s1_light_std_pcc   = s1_light_pcc_array.std(axis=0)
##
##    s1_pcc_array = temp['s1_pcc_array']
##    s1_mean_pcc  = s1_pcc_array.mean(axis=0)
##    s1_std_pcc   = s1_pcc_array.std(axis=0)
##
##    m1_light_pcc_array = temp['m1_light_pcc_array']
##    m1_light_mean_pcc  = m1_light_pcc_array.mean(axis=0)
##    m1_light_std_pcc   = m1_light_pcc_array.std(axis=0)
##
##    # add pcc values to lists
##    m1_pcc_nolight.append(m1_mean_pcc)
##    m1_pcc_s1light.append(s1_light_mean_pcc)
##
##    s1_pcc_nolight.append(s1_mean_pcc)
##    s1_pcc_m1light.append(m1_light_mean_pcc)
##
##    # run hdfanalyzer
##    get_ipython().magic(u"run hdfanalyzer.py {}".format(fname[3:7]))
##    exps.append(neuro)
##
##    # get m1 and s1 indices
##    #m1_inds = npand(npand(neuro.shank_ids==0, neuro.driven_units==True), neuro.cell_type=='RS')
##    #s1_inds = npand(npand(neuro.shank_ids==1, neuro.driven_units==True), neuro.cell_type=='RS')
##    #m1_inds = npand(neuro.shank_ids==0, neuro.driven_units==True)
##    #s1_inds = npand(neuro.shank_ids==1, neuro.driven_units==True)
##    m1_inds = neuro.shank_ids==0
##    s1_inds = neuro.shank_ids==1
##
##    # get mean selectivity for m1 and s1
##    m1_sel_nolight.append(np.mean(neuro.selectivity[m1_inds, 0]))
##    m1_sel_s1light.append(np.mean(neuro.selectivity[m1_inds, 1]))
##
##    s1_sel_nolight.append(np.mean(neuro.selectivity[s1_inds, 0]))
##    s1_sel_m1light.append(np.mean(neuro.selectivity[s1_inds, 2]))
##
##    # get mean depth for m1 and s1 units
##    neuro.depths = np.asarray(neuro.depths)
##    m1_depth.append(np.mean(neuro.depths[m1_inds]))
##    s1_depth.append(np.mean(neuro.depths[s1_inds]))
##
##m1_pcc_nolight = np.asarray(m1_pcc_nolight)
##m1_pcc_s1light = np.asarray(m1_pcc_s1light)
##
##s1_pcc_nolight = np.asarray(s1_pcc_nolight)
##s1_pcc_m1light = np.asarray(s1_pcc_m1light)
##
##m1_sel_nolight = np.asarray(m1_sel_nolight)
##m1_sel_s1light = np.asarray(m1_sel_s1light)
##
##s1_sel_nolight = np.asarray(s1_sel_nolight)
##s1_sel_m1light = np.asarray(s1_sel_m1light)
##
##m1_depth       = np.asarray(m1_depth)
##s1_depth       = np.asarray(s1_depth)
##
##
##for k in range(len(file_list)):
##    print('#####     #####\n#####     #####')
##    print('experiment: {}'.format(file_list[k]))
##    print('\nm1 selectivity nolight vs s1light {}, {}'.format(m1_sel_nolight[k], m1_sel_s1light[k]))
##    print('m1 depth: {}'.format(m1_depth[k]))
##    print('m1 pcc nolight vs s1light {}, {}'.format(m1_pcc_nolight[k][-1], m1_pcc_s1light[k][-1]))
##
##    print('\ns1 selectivity nolight vs m1light {}, {}'.format(s1_sel_nolight[k], s1_sel_m1light[k]))
##    print('s1 pcc nolight vs m1light {}, {}'.format(s1_pcc_nolight[k][-1], s1_pcc_m1light[k][-1]))
##    print('s1 depth: {}'.format(s1_depth[k]))
##
##m1_pcc_mean_nolight = np.asarray([np.mean(k) for k in m1_pcc_nolight])
##m1_pcc_mean_s1light = np.asarray([np.mean(k) for k in m1_pcc_s1light])
##
##m1_sel_diff = m1_sel_s1light - m1_sel_nolight
##m1_pcc_diff = m1_pcc_mean_s1light - m1_pcc_mean_nolight
##plt.scatter(m1_sel_diff, m1_pcc_diff)
##
##s1_pcc_mean_nolight = np.asarray([np.mean(k) for k in s1_pcc_nolight])
##s1_pcc_mean_m1light = np.asarray([np.mean(k) for k in s1_pcc_m1light])
##
##s1_sel_diff = s1_sel_m1light - s1_sel_nolight
##s1_pcc_diff = s1_pcc_mean_m1light - s1_pcc_mean_nolight
##plt.scatter(s1_sel_diff, s1_pcc_diff)
##hlines(0, -.2, .2, linestyle='dashed', linewidth=1)
##vlines(0, -25, 25, linestyle='dashed', linewidth=1)
##xlabel('change in selectivity')
##ylabel('change in PCC')
##title('PCC vs selevtivity changes')
##
##
##sel_combo_diff = np.concatenate( (m1_sel_diff, s1_sel_diff))
##pcc_combo_diff = np.concatenate( (m1_pcc_diff, s1_pcc_diff))
##
##sp.stats.pearsonr(sel_combo_diff, pcc_combo_diff)
##
##
##plt.scatter(m1_depth, m1_pcc_diff)
##plt.scatter(s1_depth, s1_pcc_diff)
##
##
##### compute PCC per unit THIS SHOULD BE THE BAR CHART (PCC / Unit +/- sem)
##m1 = list()
##s1 = list()
##for k in range(len(file_list)):
##    m1.append(np.mean(m1_pcc_nolight[k][12])/12.0)
##    s1.append(np.mean(s1_pcc_nolight[k][12])/12.0)
##
##m1 = np.asarray(m1)
##s1 = np.asarray(s1)
##
##fig, ax = plt.subplots()
##ax.bar(0, np.mean(m1),
##        0.5,
##        alpha=0.5,
##        color='b',
##        yerr=sp.stats.sem(m1))
##
##ax.bar(0.75, np.mean(s1),
##        0.5,
##        alpha=0.5,
##        color='r',
##        yerr=sp.stats.sem(s1))


##############################################################################
#####                    Parallel processing                             #####
##############################################################################


# parallel processing sandbox


##def run_decoder_in_parallel(k, X, y):
###    rand_inds = np.sort(np.random.choice(X.shape[1], num_samples, replace=False))
##    decoder = NeuroDecoder(X, y)
##    decoder.fit(plot_cmat=True, run=True)
##    print(mp.active_children())
##    return k, decoder
###    decoder.fit(kind='ole')
###    decoder.get_pcc_distribution()
###    all_pcc = decoder.all_pcc
###    return np.mean(all_pcc)
##
##
##import multiprocessing as mp
##
##X = Xs1
##y = ys1
##
##pool = mp.Pool(processes=8)
##results = [pool.apply(run_decoder_in_parallel, args=(k, X, y, )) for k in range(10)]
##
##
##def myfun_test(k, my_input=10):
##    num = np.random.randint(0, 100)
##    print(k, my_input)
##    return k, num
##
##pool = mp.Pool(processes=2)
##results = [pool.apply(myfun_test, args=(x,) ) for x in np.arange(10000)]
##
#############
##npand = np.logical_and
##pos_inds = np.arange(8)
##
#### M1 driven units
##unit_inds = np.where( npand(neuro.driven_units == True, neuro.shank_ids == 0) )[0]
##X, y, uinds = neuro.get_design_matrix(unit_inds=unit_inds, cond_inds=pos_inds, rate_type='abs_count')
##
##m1d = NeuroDecoder(X, y)
##m1d.fit(kind='ole', plot_cmat=True, run=True)
###m1d.get_pcc_distribution(num_runs=500)
##pool = mp.Pool(processes=2)
##num_runs=100
##results = [pool.apply(m1d.fit_ole_decoder) for x in np.arange(10)]
##
##input_list = np.arange(100)
### could try this too
###result = pool.map(func=fun, iterable=input_list, chunksize=n)
##result = pool.map(func=myfun_test, iterable=input_list)
##


##############################################################################
#####    Scratch space: predict whisker angle from neural data??         #####
##############################################################################

##### compute distributions of PCC for single units
##### use logistic regression decoder


##### try to predict whisker angle from neural data #####
##### try to predict whisker angle from neural data #####











## from tqdm.auto import tqdm  # notebook compatible
#import time
#for k, i1 in enumerate(tqdm(range(5), ncols=80)):
#    for i2 in tqdm(range(300), leave=False, ncols=80):
#        # do something, e.g. sleep
#        time.sleep(0.01)











