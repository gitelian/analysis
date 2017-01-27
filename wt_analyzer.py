import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal
import statsmodels.stats.multitest as smm
from scipy.optimize import curve_fit

#fids = ['1289', '1290', '1295', '1302']
fids = ['1325']
exps = list()
for fid in fids:
    #get_ipython().magic(u"run neoanalyzer.py {'1290'}")
    get_ipython().magic(u"run neoanalyzer.py {}".format(fid))
    neuro.rates(kind='wsk_boolean')
    exps.append(neuro)
# neuro.plot_tuning_curve(kind='evk_count')

neuro = exps[3]

neuro.plot_tuning_curve(kind='evk_count')

# LDA analysis
trode = 1

plt.figure()
lda = LinearDiscriminantAnalysis(n_components=2)
X, y = neuro.make_design_matrix('evk_count', trode=trode)
X_r0 = X[y<8, :]
y_r0 = y[y<8]
X_r0 = lda.fit(X_r0, y_r0).transform(X_r0)
plt.subplot(1,2,1)
color=iter(cm.rainbow(np.linspace(0,1,len(np.unique(y_r0)))))
for k in range(len(np.unique(y_r0))):
   c = next(color)
   plt.plot(X_r0[y_r0==k, 0], X_r0[y_r0==k, 1], 'o', c=c, label=str(k))
plt.legend(loc='best')


X, y = neuro.make_design_matrix('evk_count', trode=trode)
trial_inds = np.logical_and(y>=9, y<17) # no control position
#trial_inds = np.logical_and(y>=18, y<26) # no control position
X_r0 = X[trial_inds, :]
y_r0 = y[trial_inds]
X_r0 = lda.fit(X_r0, y_r0).transform(X_r0)
color=iter(cm.rainbow(np.linspace(0,1,len(np.unique(y_r0)))))
plt.subplot(1,2,2)
for k in range(len(np.unique(y_r0))):
    c = next(color)
    plt.plot(X_r0[y_r0==k+9, 0], X_r0[y_r0==k+9, 1], 'o', c=c, label=str(k))
    plt.xlim(-5,3)
    plt.ylim(-6, 4)
#    plt.plot(X_r0[y_r0==k+9+9, 0], X_r0[y_r0==k+9+9, 1], 'o', c=c, label=str(k))
plt.legend(loc='best')
plt.show()

        ##### WHISKER TRACKING ANALYSIS #####
        ##### WHISKER TRACKING ANALYSIS #####

def plot_setpoint(neuro, cond=0, color='k', error='sem'):
    ax = plt.gca()
    sp_temp = neuro.wt[cond][:, 1, :]
    mean_sp = np.mean(sp_temp, axis=1)
    se      = sp.stats.sem(sp_temp, axis=1)

    # inverse of the CDF is the percentile function. ppf is the percent point funciton of t.
    if error == 'ci':
        err = se*sp.stats.t.ppf((1+0.95)/2.0, sp_temp.shape[1]-1) # (1+1.95)/2 = 0.975
    elif error == 'sem':
        err = se

    plt.plot(neuro.wtt, mean_sp, color)
    plt.fill_between(neuro.wtt, mean_sp - err, mean_sp + err, facecolor=color, alpha=0.3)
    return ax

for k in range(neuro.control_pos):
    plt.figure()
    plot_setpoint(neuro, cond=k, color='k')
#    plot_setpoint(neuro, cond=k+9, color='r')
#     plot_setpoint(neuro, cond=k+9+9, color='b')
    plt.xlim(-0.5, 2.0)
    plt.xlabel('time (s)')
    plt.ylabel('set-point (deg)')
    plt.title('condition {}'.format(str(k)))
    plt.show()

######################################################
sp_diff = list()
sp_t = list()
sp_p = list()
vel_diff = list()
vel_t = list()
vel_p = list()
frq_diff = list()
frq_t = list()
frq_p = list()
#run_diff = list()
#run_t = list()
#run_p = list()

frq_nolight_mean  = np.zeros(9,)
frq_light_mean = np.zeros(9,)
frq_nolight_err   = np.zeros(9,)
frq_light_err  = np.zeros(9,)

frq_nolight = list()
frq_light   = list()

for neuro in exps:
    neuro = exps[0]
    for k in range(9):
        base_inds = np.logical_and(neuro.wtt > 0, neuro.wtt < 1.0)
        stim_inds = np.logical_and(neuro.wtt > 0.5, neuro.wtt < 1.5)
        ##### set-point #####
        sp_nolight = np.nanmean(neuro.wt[k][stim_inds, 2, :], axis=1)
        sp_light   = np.nanmean(neuro.wt[k+9][stim_inds, 2, :], axis=1) # s1 silencin
        sp_diff.append(np.nanmean(sp_light) - np.nanmean(sp_nolight))
        t, p = sp.stats.ttest_ind(sp_light, sp_nolight)
        sp_t.append(t); sp_p.append(p)

        #### velocity #####
        vel_nolight = np.nanmean(neuro.wt[k][stim_inds, 4, :], axis=1)
        vel_light   = np.nanmean(neuro.wt[k+9][stim_inds, 4, :], axis=1) # s1 silencing
        vel_diff.append(np.nanmean(vel_light) - np.nanmean(vel_nolight))
        t, p = sp.stats.ttest_ind(vel_light, vel_nolight)
        vel_t.append(t); vel_p.append(p)

#        #### run speed #####
#        run_nolight  = remove_nans(np.nanmean(vel_dict[mouse][k][:, -15000:], axis=1))
#        run_light = remove_nans(np.nanmean(vel_dict[mouse][k+9][:, -15000:], axis=1))
#        run_diff.append(np.nanmean(run_light) - np.nanmean(run_nolight))
#        t, p = sp.stats.ttest_ind(run_light, run_nolight)
#        run_t.append(t); run_p.append(p)

        ##### frequeny #####
        ang_nolight = np.nanmean(neuro.wt[k][stim_inds, 0, :], axis=1)
        ang_light   = np.nanmean(neuro.wt[k+9][stim_inds, 0, :], axis=1) # s1 silencing

        ang_nolight_temp = neuro.wt[k][stim_inds, 0, :]
        ang_light_temp   = neuro.wt[k+9][stim_inds, 0, :]
        f_nolight_temp = list()
        f_light_temp = list()

        num_trials = ang_nolight_temp.shape[1]
        frq_mat_temp = np.zeros((500/2, num_trials))
        for trial in range(num_trials):
            f, Pxx_den = sp.signal.periodogram(ang_nolight_temp[:, trial], 500)
            f_inds = np.where(f >= 2)[0]
            f_nolight_temp.append(f[f_inds[np.argmax(Pxx_den[f_inds])]])

            frq_mat_temp[:, trial] = Pxx_den
        frq_nolight.append(frq_mat_temp)

        num_trials = ang_light_temp.shape[1]
        frq_mat_temp = np.zeros((500/2, num_trials))
        for trial in range(num_trials):
            f, Pxx_den = sp.signal.periodogram(ang_light_temp[:, trial], 500)
            f_inds = np.where(f >= 2)[0]
            f_light_temp.append(f[f_inds[np.argmax(Pxx_den[f_inds])]])

            frq_mat_temp[:, trial] = Pxx_den
        frq_light.append(frq_mat_temp)

        t, p = sp.stats.ttest_ind(f_light_temp, f_nolight_temp)
        frq_t.append(t); frq_p.append(p)
        frq_diff.append(np.nanmean(f_light_temp) - np.nanmean(f_nolight_temp))

        frq_nolight_mean[k] = np.nanmean(f_nolight_temp)
        frq_nolight_err[k]  = np.nanstd(f_nolight_temp)/np.sqrt(len(f_nolight_temp))*2
        frq_light_mean[k-9] = np.nanmean(f_light_temp)
        frq_light_err[k-9] = np.nanstd(f_light_temp)/np.sqrt(len(f_light_temp))*2

    plt.errorbar(np.arange(1,10), frq_nolight_mean, yerr=frq_nolight_err, fmt='-o')
    plt.errorbar(np.arange(1,10), frq_light_mean, yerr=frq_light_err, fmt='-o')

##### make distribution plots and do statistical test corrections #####
plt.subplots(1, 3)

# set-point
plt.subplot(1, 3, 1)
plt.hist(sp_diff, bins=np.arange(-20,20,2), align='left')
rej, pval_corr = smm.multipletests(sp_p, alpha=0.05, method='sh')[:2]
plt.title('set-point; num-sig: ' + str(np.sum(rej)))

# velocity
plt.subplot(1, 3, 2)
plt.hist(vel_diff, bins=np.arange(-400,400,50), align='left')
rej, pval_corr = smm.multipletests(vel_p, alpha=0.05, method='sh')[:2]
plt.title('vel; num-sig: ' + str(np.sum(rej)))

# frequency
plt.subplot(1, 3, 3)
plt.hist(frq_diff, bins=np.arange(-4,4,0.5), align='left')
rej, pval_corr = smm.multipletests(frq_p, alpha=0.05, method='sh')[:2]
plt.title('frq; num-sig: ' + str(np.sum(rej)))

#    # run speed
#    plt.subplot(1, 5, 4)
#    plt.hist(run_diff, bins=np.arange(-200,200,10), align='left')
#    rej, pval_corr = smm.multipletests(run_p, alpha=0.05, method='sh')[:2]
#    plt.title('run; num-sig: ' + str(np.sum(rej)))
#

def plot_freq(neuro, cond=0, color='k', error='sem'):
    base_inds = np.logical_and(neuro.wtt > 0, neuro.wtt < 1.0)
    stim_inds = np.logical_and(neuro.wtt > 0.5, neuro.wtt < 1.5)

    ##### frequeny #####
    ang_temp = neuro.wt[cond][stim_inds, 0, :]
    f_temp = list()

    num_trials = ang_temp.shape[1]
    frq_mat_temp = np.zeros((500/2, num_trials))
    for trial in range(num_trials):
        f, Pxx_den = sp.signal.periodogram(ang_temp[:, trial], 500)
        frq_mat_temp[:, trial] = Pxx_den

    ax = plt.gca()
    mean_frq = np.mean(frq_mat_temp, axis=1)
    se       = sp.stats.sem(frq_mat_temp, axis=1)

    # inverse of the CDF is the percentile function. ppf is the percent point funciton of t.
    if error == 'ci':
        err = se*sp.stats.t.ppf((1+0.95)/2.0, frq_mat_temp.shape[1]-1) # (1+1.95)/2 = 0.975
    elif error == 'sem':
        err = se

    plt.plot(f, mean_frq, color)
    plt.fill_between(f, mean_frq - err, mean_frq + err, facecolor=color, alpha=0.3)
    return ax









