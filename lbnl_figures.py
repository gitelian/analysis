import os
import h5py
from hdfanalyzer import *
import matplotlib.pyplot as plt
import statsmodels.stats.multitest as smm
import numpy as np

# create plotting functions
def fr_vs_light(neuro, unit=1, stims=[1, 2, 3]):
    fig, ax = plt.subplots()
    neuro.plot_psth(unit_ind=unit, trial_type=stims[0], color='k', error='sem')
    neuro.plot_psth(unit_ind=unit, trial_type=stims[1], color='g', error='sem')
    neuro.plot_psth(unit_ind=unit, trial_type=stims[2], color='b', error='sem')
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_xlabel('time (s)')
    ax.legend(['0.85V', '0.65V', '0.45V'])

    return fig, ax

def compare_units(neuro, units=[0, 1], stims=[0, 1]):
    fig, ax = plt.subplots(1, 2)

    # plot first unit in the two conditions
    neuro.plot_psth(axis=ax[0], unit_ind=units[0], trial_type=stims[0],\
            error='sem', color='k')
    neuro.plot_psth(axis=ax[1], unit_ind=units[0], trial_type=stims[1],\
            error='sem', color='k')

    # plot second unit in the two conditions
    neuro.plot_psth(axis=ax[0], unit_ind=units[1], trial_type=stims[0],\
            error='sem', color='g')
    neuro.plot_psth(axis=ax[1], unit_ind=units[1], trial_type=stims[1],\
            error='sem', color='g')

    fig.suptitle('Differential activation via optical gratings', size=14)
    ax[0].set_ylabel('Firing rate (Hz)')
    ax[0].set_xlabel('time (s)')
    ax[1].set_xlabel('time (s)')

    return fig, ax

if __name__ == "__main__":

    ## Load data from h5py file ##
    f = h5py.File(os.getcwd() + os.sep + 'FID2142.hdf5', 'r+')
    #neuro = hdfanalyzer.NeuroAnalyzer(f, '2142')
    neuro = NeuroAnalyzer(f, '2142')

    ##### LBNL figures for Nature Communications
    ##### data from FID2142

    ### single unit FR vs time at three light levels (unit 5, stim 1, 2, 3)

    fr_vs_light(neuro, unit=5, stims=[1, 2, 3])

    ## compute statistical tests on firing rate (stim period vs baseline)
    ## i.e. which light condition modulated the firing rate from baseline?

    base_inds = np.where(np.logical_and(neuro.bins_t > -1, neuro.bins_t < 0) == True)[0]
    stim_inds = np.where(np.logical_and(neuro.bins_t > 0, neuro.bins_t < 1) == True)[0]

    # stim 1 (0.85V)
    s1_base = neuro.binned_spikes[1][base_inds, :, 5].sum(axis=0)
    s1_stim = neuro.binned_spikes[1][stim_inds, :, 5].sum(axis=0)

    # stim 2 (0.65V)
    s2_base = neuro.binned_spikes[2][base_inds, :, 5].sum(axis=0)
    s2_stim = neuro.binned_spikes[2][stim_inds, :, 5].sum(axis=0)

    # stim 3 (0.45V)
    s3_base = neuro.binned_spikes[3][base_inds, :, 5].sum(axis=0)
    s3_stim = neuro.binned_spikes[3][stim_inds, :, 5].sum(axis=0)


    # Calculate the Wilcoxon signed-rank test.
    #
    # The Wilcoxon signed-rank test tests the null hypothesis that two
    # related paired samples come from the same distribution. In particular,
    # it tests whether the distribution of the differences x - y is symmetric
    # about zero. It is a non-parametric version of the paired T-test.

    s1_stat, s1_pval = sp.stats.wilcoxon(s1_base, s1_stim)
    s2_stat, s2_pval = sp.stats.wilcoxon(s2_base, s2_stim)
    s3_stat, s3_pval = sp.stats.wilcoxon(s3_base, s3_stim)

    test_statistics = [s1_stat, s2_stat, s3_stat]
    pvals = [s1_pval, s2_pval, s3_pval]
    rej, pval_corr, _, alphacBonf = smm.multipletests(pvals, alpha=0.05, method='bonferroni')

    # test: Wilcoxon signed-rank test with bonferroni correction for multiple comparisons
    # stim_conditions [0.85V, 0.65V, 0.45V]
    # test_statistics [0.0, 0.0, 1770.5]
    # pvals_corrected [1.15545805e-17, 1.14778619e-17, 1.00000000e+00]


    # Kruskal-Wallis 1-way ANOVA with Dunn's multiple comparison test
    # (i.e. a non-parametric way to compare firing rates at different light levels
    # with the proper multiple comparison test). Tests differences in medians.

    # from dunn.py
    #    H: float
    #        Kruskal-Wallis H-statistic
    #
    #    p_omnibus: float
    #        p-value corresponding to the global null hypothesis that the medians of
    #        the groups are all equal
    #
    #    Z_pairs: float array
    #        Z-scores computed for the absolute difference in mean ranks for each
    #        pairwise comparison
    #
    #    p_corrected: float array
    #        corrected p-values for each pairwise comparison, corresponding to the
    #        null hypothesis that the pair of groups has equal medians. note that
    #        these are only meaningful if the global null hypothesis is rejected.
    #
    #    reject: bool array
    #        True for pairs where the null hypothesis can be rejected for the given
    #        alpha
    #
    #    Reference:
    #    ---------------
    #    Gibbons, J. D., & Chakraborti, S. (2011). Nonparametric Statistical
    #    Inference (5th ed., pp. 353-357). Boca Raton, FL: Chapman & Hall.

    groups = [s1_stim, s2_stim, s3_stim]
    to_compare = [(0, 1), (0, 2), (1, 2)]
    H, p_omnibus, Z_pairs, p_corrected, reject = dunn.kw_dunn(groups, to_compare=to_compare, alpha=0.05, method='bonf') # or 'bonf' for bonferoni OR 'simes-hochberg'

    ## overall
    # Kruskal-Wallis H-statistic 265.8961902144867
    # p-value corresponding to the global null hypothesis that the medians of the
    # groups are all equal: 1.825475108195245e-58

    ## within groups
    # comparisons:       0.85V vs 0.65V , 0.85V vs 0.45V , 0.65V vs 0.45V
    # pvalues corrected: [1.07928148e-15, 2.82896919e-59, 1.07928148e-15]
    # reject null-hypothesis: [True, True, True]



    ### FID2142 ###
    ### FID2142 ###
    ### two single units firing rates during illumination of two spatially distinct light gratingsu

    # matching units/stims Black lines: (2, 1), (2, 2), Green lines: (6, 1), (6, 2)
    # stims 1 and 2 are 0.85V and 0.65V respectively
    compare_units(neuro, units=[2,6], stims=[1,2])

    # unit 2, stim 1 (0.85V)
    u2s1 = neuro.binned_spikes[1][stim_inds, :, 2].sum(axis=0)
    u2s2 = neuro.binned_spikes[2][stim_inds, :, 2].sum(axis=0)

    u6s1 = neuro.binned_spikes[1][stim_inds, :, 6].sum(axis=0)
    u6s2 = neuro.binned_spikes[2][stim_inds, :, 6].sum(axis=0)

    groups = [u2s1, u6s1, u2s2, u6s2]
    to_compare = [(0, 1), (0, 2), (1, 3), (2, 3)]
    H, p_omnibus, Z_pairs, p_corrected, reject = dunn.kw_dunn(groups, to_compare=to_compare, alpha=0.05, method='bonf') # or 'bonf' for bonferoni OR 'simes-hochberg'

    ## overall
    # Kruskal-Wallis H-statistic: 299.0670019531058
    # p-value corresponding to the global null hypothesis that the medians of the
    # groups are all equal: 1.5837802632107045e-64

    ## within groups
    # comparisons: grating #1: black FR vs green FR, black FR: grating #1 vs #2,
    #             green FR: grating #1 vs #1, grating #2: black FR vs green FR
    # pvalues corrected: [3.55814762e-05, 2.66849855e-31, 8.43766834e-25, 2.95551809e-08]
    # reject null-hypothesis: [ True,  True,  True,  True]


    ### FID2143 ###
    ### FID2143 ###
    ### two single units firing rates during illumination of two spatially distinct light gratingsu

    # matching units/stims Black lines: (2, 1), (2, 2), Green lines: (6, 1), (6, 2)
    # stims 1 and 2 are 0.85V and 0.65V respectively
    compare_units(neuro, units=[2,6], stims=[1,2])

    # unit 2, stim 1 (0.85V)
    u2s1 = neuro.binned_spikes[1][stim_inds, :, 2].sum(axis=0)
    u2s2 = neuro.binned_spikes[2][stim_inds, :, 2].sum(axis=0)

    u6s1 = neuro.binned_spikes[1][stim_inds, :, 6].sum(axis=0)
    u6s2 = neuro.binned_spikes[2][stim_inds, :, 6].sum(axis=0)

    groups = [u2s1, u6s1, u2s2, u6s2]
    to_compare = [(0, 1), (0, 2), (1, 3), (2, 3)]
    H, p_omnibus, Z_pairs, p_corrected, reject = dunn.kw_dunn(groups, to_compare=to_compare, alpha=0.05, method='bonf') # or 'bonf' for bonferoni OR 'simes-hochberg'

    ## overall
    # Kruskal-Wallis H-statistic: 299.0670019531058
    # p-value corresponding to the global null hypothesis that the medians of the
    # groups are all equal: 1.5837802632107045e-64

    ## within groups
    # comparisons: grating #1: black FR vs green FR, black FR: grating #1 vs #2,
    #             green FR: grating #1 vs #1, grating #2: black FR vs green FR
    # pvalues corrected: [3.55814762e-05, 2.66849855e-31, 8.43766834e-25, 2.95551809e-08]
    # reject null-hypothesis: [ True,  True,  True,  True]

















