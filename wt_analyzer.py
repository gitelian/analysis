import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal
import statsmodels.stats.multitest as smm
from scipy.optimize import curve_fit
import sys
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

sns.set_style("whitegrid", {'axes.grid' : False})

#fids = ['1289', '1290', '1295', '1302']
#fids = ['1295', '1302', '1328']
fids = ['1330']
# fid with good whisker tracking
fids = ['1330', '1336', '1338', '1339', '1340']
exps = list()
for fid in fids:
    #get_ipython().magic(u"run neoanalyzer.py {'1290'}")
    get_ipython().magic(u"run neoanalyzer.py {}".format(fid))
    #neuro.rates(kind='wsk_boolean')
    exps.append(neuro)
# neuro.plot_tuning_curve(kind='evk_count')

# plot all set-point traces
plot(neuro.wtt, neuro.wt[6+9][:,1,:])

#### LDA analysis #####
#### LDA analysis #####
trode = 2

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
#trial_inds = np.logical_and(y>=9, y<17) # no control position
trial_inds = np.logical_and(y>=18, y<26) # no control position
X_r0 = X[trial_inds, :] y_r0 = y[trial_inds] X_r0 = lda.fit(X_r0, y_r0).transform(X_r0)
color=iter(cm.rainbow(np.linspace(0,1,len(np.unique(y_r0)))))
plt.subplot(1,2,2)
for k in range(len(np.unique(y_r0))):
    c = next(color)
#    plt.plot(X_r0[y_r0==k+9, 0], X_r0[y_r0==k+9, 1], 'o', c=c, label=str(k))
    plt.xlim(-5,3)
    plt.ylim(-6, 4)
    plt.plot(X_r0[y_r0==k+9+9, 0], X_r0[y_r0==k+9+9, 1], 'o', c=c, label=str(k))
plt.legend(loc='best')
plt.show()

        ##### WHISKER TRACKING ANALYSIS #####
        ##### WHISKER TRACKING ANALYSIS #####

def plot_setpoint(neuro, axis=axis, cond=0, color='k', error='sem'):
    #ax = plt.gca()
    sp_temp = neuro.wt[cond][:, 1, :]
    mean_sp = np.mean(sp_temp, axis=1)
    se      = sp.stats.sem(sp_temp, axis=1)

    # inverse of the CDF is the percentile function. ppf is the percent point funciton of t.
    if error == 'ci':
        err = se*sp.stats.t.ppf((1+0.95)/2.0, sp_temp.shape[1]-1) # (1+1.95)/2 = 0.975
    elif error == 'sem':
        err = se

    axis.plot(neuro.wtt, mean_sp, color)
    axis.fill_between(neuro.wtt, mean_sp - err, mean_sp + err, facecolor=color, alpha=0.3)

for neuro in exps:
    fig, ax = plt.subplots(neuro.control_pos, 2)
    for i in range(2):
        for k in range(neuro.control_pos):
            axis = ax[k][i]
            plot_setpoint(neuro, axis=axis, cond=k, color='k')
            if i == 0:
                plot_setpoint(neuro, axis=axis, cond=k+9, color='r')
            else:
                plot_setpoint(neuro, axis=axis, cond=k+9+9, color='b')
            ax[k][i].set_xlim(-0.5, 2.0)
            ax[k][i].set_xlabel('time (s)')
            ax[k][i].set_ylabel('set-point (deg)')
            ax[k][i].set_title('condition {}'.format(str(k)))

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
#    neuro = exps[0]
    for k in range(9):
        base_inds = np.logical_and(neuro.wtt > -1.0, neuro.wtt < 0)
        stim_inds = np.logical_and(neuro.wtt > 0.5, neuro.wtt < 1.5)
        ##### set-point #####
        sp_nolight = np.nanmean(neuro.wt[k][stim_inds, 1, :], axis=1)
        sp_light   = np.nanmean(neuro.wt[k+9][stim_inds, 1, :], axis=1) # s1 silencin
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
plt.hist(sp_diff, bins=np.arange(-20,20,1), align='left')
rej, pval_corr = smm.multipletests(sp_p, alpha=0.05, method='sh')[:2]
plt.title('set-point; num-sig: ' + str(np.sum(rej)))

# velocity
plt.subplot(1, 3, 2)
plt.hist(vel_diff, bins=np.arange(-100,100,1), align='left')
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


######################################################
##### single experiment whisking analysis #####
######################################################

##### plot PSD of whisking frequency for s1 and m1 silencing #####
##### plot PSD of whisking frequency for s1 and m1 silencing #####

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

plt.figure()
plot_freq(neuro, cond=8, color='k')
plot_freq(neuro, cond=8+9, color='r')
#plot_freq(neuro, cond=8+9+9, color='b')
plt.xlim(0,40); plt.xlabel('frequency (Hz)')

##### plot spike triggered averages #####
##### plot spike triggered averages #####

###### Plot unit protraction summaries #####
###### Plot unit protraction summaries #####

## Fit sinusoid function ##
## Fit sinusoid function ##

from scipy.optimize import leastsq
def residuals(p, y, x):
    A, f, theta, D = p
    err = y - ( A*sin(2*np.pi*f*x + theta) + D)
    return err

def peval(x, p):
    return p[0]*sin(2*np.pi*p[1]*x + p[2]) + p[3]

def fit_sinusoid(bins, y_meas):
    p0 = [2*np.std(y_meas), 20, 0, np.mean(y_meas)]
    x = bins[:-1]
    plsq = leastsq(residuals, p0, args=(y_meas, x))
    return plsq

# remove gride lines
sns.set_style("whitegrid", {'axes.grid' : False})
npand   = np.logical_and

neuro.get_pta_depth()
dt      = 0.005 # seconds
window  = [-0.06, 0.06]
#window  = [-0.75, 0.755]

with PdfPages(fid + '_unit_protraction_summaries.pdf') as pdf:
    for uid in range(neuro.num_units):
        fig, ax = subplots(3, 3, figsize=(12,8))
        fig.suptitle('Region: {}, depth: {}, unit type: {}, mouse: {}, driven: {}'.format(\
                neuro.region_dict[neuro.shank_ids[uid]], \
                neuro.depths[uid], \
                neuro.cell_type[uid], \
                fid, \
                neuro.driven_units[uid]))

        best_pos = int(neuro.best_contact[uid])
#        best_pos = 3

        # best position
        spks_per_bin, sem, bins = neuro.pta(cond=best_pos, unit_ind=uid, window=window, dt=dt)
        ax[0][0].bar(bins[:-1], spks_per_bin, width=dt, edgecolor='none')
        ax[0][0].set_title('Best Position\nMI: {}'.format(neuro.mod_index[uid, best_pos]))
        ax[0][0].set_ylabel('Firing Rate (Hz)')
        #ax[0][0].errorbar(bins[:-1]+dt/2, spks_per_bin, yerr=sem*2.2, fmt='.', color='k')

        ### fit sine wave ###
        plsq = fit_sinusoid(bins, spks_per_bin)
        ax[0][0].plot(bins[:-1], peval(bins[:-1], plsq[0]), 'r')
        ax[0][0].set_title('Amp: {0:.3f}, std: {1:.3f}'.format(plsq[0][0], np.std(spks_per_bin)))

        # best position S0 silencing
        spks_per_bin, sem, bins = neuro.pta(cond=best_pos+9, unit_ind=uid, window=window, dt=dt)
        ax[1][0].bar(bins[:-1], spks_per_bin, width=dt, edgecolor='none')
        ax[1][0].set_title('S1 silencing')
        ax[1][0].set_ylabel('Firing Rate (Hz)')
        #ax[1][0].errorbar(bins[:-1]+dt/2, spks_per_bin, yerr=sem*2.2, fmt='.', color='k')

        ### fit sine wave ###
        plsq = fit_sinusoid(bins, spks_per_bin)
        ax[1][0].plot(bins[:-1], peval(bins[:-1], plsq[0]), 'r')
        ax[1][0].set_title('Amp: {0:.3f}, std: {1:.3f}'.format(plsq[0][0], np.std(spks_per_bin)))

        # best position M1 silencing
        spks_per_bin, sem, bins = neuro.pta(cond=best_pos+9+9, unit_ind=uid, window=window, dt=dt)
        ax[2][0].bar(bins[:-1], spks_per_bin, width=dt, edgecolor='none')
        ax[2][0].set_title('M1 silencing')
        ax[2][0].set_xlabel('Time from protraction (s)')
        ax[2][0].set_ylabel('Firing Rate (Hz)')
        #ax[2][0].errorbar(bins[:-1]+dt/2, spks_per_bin, yerr=sem*2.2, fmt='.', color='k')

        ### fit sine wave ###
        plsq = fit_sinusoid(bins, spks_per_bin)
        ax[2][0].plot(bins[:-1], peval(bins[:-1], plsq[0]), 'r')
        ax[2][0].set_title('Amp: {0:.3f}, std: {1:.3f}'.format(plsq[0][0], np.std(spks_per_bin)))

        # no contact position
        spks_per_bin, sem, bins = neuro.pta(cond=neuro.control_pos-1, unit_ind=uid, window=window, dt=dt)
        ax[0][1].bar(bins[:-1], spks_per_bin, width=dt, edgecolor='none')
        ax[0][1].set_title('No Contact Position')
        #ax[0][1].errorbar(bins[:-1]+dt/2, spks_per_bin, yerr=sem*2.2, fmt='.', color='k')

        ### fit sine wave ###
        plsq = fit_sinusoid(bins, spks_per_bin)
        ax[0][1].plot(bins[:-1], peval(bins[:-1], plsq[0]), 'r')
        ax[0][1].set_title('Amp: {0:.3f}, std: {1:.3f}'.format(plsq[0][0], np.std(spks_per_bin)))

        # no contact position S1 silencing
        spks_per_bin, sem, bins = neuro.pta(cond=neuro.control_pos-1+9, unit_ind=uid, window=window, dt=dt)
        ax[1][1].bar(bins[:-1], spks_per_bin, width=dt, edgecolor='none')
        ax[1][1].set_title('S1 silencing')
        #ax[1][1].errorbar(bins[:-1]+dt/2, spks_per_bin, yerr=sem*2.2, fmt='.', color='k')

        ### fit sine wave ###
        plsq = fit_sinusoid(bins, spks_per_bin)
        ax[1][1].plot(bins[:-1], peval(bins[:-1], plsq[0]), 'r')
        ax[1][1].set_title('Amp: {0:.3f}, std: {1:.3f}'.format(plsq[0][0], np.std(spks_per_bin)))

        # no contact position M1 silencing
        spks_per_bin, sem, bins = neuro.pta(cond=neuro.control_pos-1+9+9, unit_ind=uid, window=window, dt=dt)
        ax[2][1].bar(bins[:-1], spks_per_bin, width=dt, edgecolor='none')
        ax[2][1].set_title('M1 silencing')
        ax[2][1].set_xlabel('Time from protraction (s)')
        #ax[2][1].errorbar(bins[:-1]+dt/2, spks_per_bin, yerr=sem*2.2, fmt='.', color='k')

        ### fit sine wave ###
        plsq = fit_sinusoid(bins, spks_per_bin)
        ax[2][1].plot(bins[:-1], peval(bins[:-1], plsq[0]), 'r')
        #ax[2][1].set_title('Amp: {0:.3f}, std: {0:.3f}'.format(plsq[0][0], np.std(spks_per_bin)))
        ax[2][1].set_title('Amp: {0:.3f}, std: {1:.3f}'.format(34, 5))

        ## set ylim to the max ylim of all PTA histogram subplots
        ylim_max = 0
        for row in ax:
            for coli, col in enumerate(row):
                if coli < 2:
                    ylim_temp = col.get_ylim()[1]
                    if ylim_temp > ylim_max:
                        ylim_max = ylim_temp
        for row in ax:
            for coli, col in enumerate(row):
                if coli < 2:
                    col.set_ylim(0, ylim_max)

        # top right: evoked tuning curves
        neuro.plot_tuning_curve(unit_ind=uid, kind='abs_rate', axis=ax[0][2])
        ax[0][2].set_xlim(0, 10)
        ax[0][2].hlines(0, 0, 10, colors='k', linestyles='dashed')
#        ax[0][2].set_xlabel('bar position')
        ax[0][2].set_title('absolute tc')

        # middle right: evoked tuning curves
        neuro.plot_tuning_curve(unit_ind=uid, kind='evk_rate', axis=ax[1][2])
        ax[1][2].set_xlim(0, 10)
        ax[1][2].hlines(0, 0, 10, colors='k', linestyles='dashed')
#        ax[1][2].set_xlabel('bar position')
        ax[1][2].set_title('evoked tc')

        # bottom right: pta modulation depth tuning curve
        print('CHECK THE PTA MODULATION DEPTH CODE')
        ax[2][2].plot(np.arange(1,10), neuro.mod_index[uid, 0:9], '-ko',\
                np.arange(1,10), neuro.mod_index[uid, 9:18], '-ro',\
                np.arange(1,10), neuro.mod_index[uid, 18:27], '-bo')
        ax[2][2].set_xlim(0, 10)
        ax[2][2].hlines(0, 0, 10, colors='k', linestyles='dashed')
#        ax[2][2].set_xlabel('bar position')
        ax[2][2].set_title('protraction modulation depth (fano factor)')

        pdf.savefig()
        fig.clear()
        plt.close()



##### FID1337 GPR26 AAV-ChR2 Analyzer #####
##### FID1337 GPR26 AAV-ChR2 Analyzer #####

neuro.reclassify_run_trials(mean_thresh=250)
stim_times = np.arange(0.5, 1.5, 1.0/20.0)
npand   = np.logical_and
stim_time_inds = npand(neuro.wtt >= 0.5, neuro.wtt <= 1.5)

ylow, yhigh = 20, 180 # degrees (yaxis lims)
xlow, xhigh = -0.5, 2.0 # time (xaxis lims)

fig, ax = subplots(2, 2, figsize=(5,8), sharex=True, sharey=True)
#plt.subplots_adjust(left=0.10, bottom=0.10, right=0.90, top=0.90, wspace=0.20, hspace=0.45)
# add annotations
ax[0][0].set_title('POM (optrode) 20Hz stimulation')
ax[0][1].set_title('M1 (400um fiber) 20Hz stimulation')

# running: POM stim, control position
neuro.rates(running=True)
ax[0][0].plot(neuro.wtt, neuro.wt[neuro.control_pos-1+9][:, 0, :], linewidth=0.25, color='grey')
ax[0][0].plot(neuro.wtt, neuro.wt[neuro.control_pos-1+9][:, 0, :].mean(axis=1), linewidth=2.0, color='black')
ax[0][0].set_ylim([ylow, yhigh])
ax[0][0].set_xlim([xlow, xhigh])
ax[0][0].vlines(stim_times, ylow, yhigh, color='red', linewidth=0.5)

# running: M1 stim, control position
ax[0][1].plot(neuro.wtt, neuro.wt[neuro.control_pos-1+9+9][:, 0, :], linewidth=0.25, color='grey')
ax[0][1].plot(neuro.wtt, neuro.wt[neuro.control_pos-1+9+9][:, 0, :].mean(axis=1), linewidth=2.0, color='black')
ax[0][1].set_ylim([ylow, yhigh])
ax[0][1].set_xlim([xlow, xhigh])
ax[0][1].vlines(stim_times, ylow, yhigh, color='red', linewidth=0.5)

# non-running: POM stim, control position
neuro.rates(running=False)
ax[1][0].plot(neuro.wtt, neuro.wt[neuro.control_pos-1+9][:, 0, :], linewidth=0.25, color='grey')
ax[1][0].plot(neuro.wtt, neuro.wt[neuro.control_pos-1+9][:, 0, :].mean(axis=1), linewidth=2.0, color='black')
ax[1][0].set_ylim([ylow, yhigh])
ax[1][0].set_xlim([xlow, xhigh])
ax[1][0].vlines(stim_times, ylow, yhigh, color='red', linewidth=0.5)

# non-running: M1 stim, control position
ax[1][1].plot(neuro.wtt, neuro.wt[neuro.control_pos-1+9+9][:, 0, :], linewidth=0.25, color='grey')
ax[1][1].plot(neuro.wtt, neuro.wt[neuro.control_pos-1+9+9][:, 0, :].mean(axis=1), linewidth=2.0, color='black')
ax[1][1].set_ylim([ylow, yhigh])
ax[1][1].set_xlim([xlow, xhigh])
ax[1][1].vlines(stim_times, ylow, yhigh, color='red', linewidth=0.5)


##### PSD running vs non-running #####
##### PSD running vs non-running #####
ylow, yhigh = 0, 200 # arbitrary power (yaxis lims)
xlow, xhigh = 0, 30  # frequency (xaxis lims)

fig, ax = subplots(1, 2, figsize=(12,8), sharex=True, sharey=True)

# running: POM and M1
neuro.rates(running=True)
f, frq_mat_temp = neuro.get_psd(neuro.wt[neuro.control_pos-1+9][stim_time_inds, 0, :], 500.0)
neuro.plot_freq(f, frq_mat_temp, axis=ax[0], color='red')

f, frq_mat_temp = neuro.get_psd(neuro.wt[neuro.control_pos-1+9+9][stim_time_inds, 0, :], 500.0)
neuro.plot_freq(f, frq_mat_temp, axis=ax[0], color='black')

# non-running: POM and M1
neuro.rates(running=False)
f, frq_mat_temp = neuro.get_psd(neuro.wt[neuro.control_pos-1+9][stim_time_inds, 0, :], 500.0)
neuro.plot_freq(f, frq_mat_temp, axis=ax[1], color='orange')

f, frq_mat_temp = neuro.get_psd(neuro.wt[neuro.control_pos-1+9+9][stim_time_inds, 0, :], 500.0)
neuro.plot_freq(f, frq_mat_temp, axis=ax[1], color='grey')


ax[0].set_title('Running')
ax[0].set_xlim([xlow, xhigh])
ax[0].legend(['POM', 'M1'])
ax[0].set_ylabel('Power')
ax[0].set_xlabel('frequency (Hz)')

ax[1].set_title('Non-Running')
ax[1].set_xlim([xlow, xhigh])
ax[1].legend(['POM', 'M1'])
ax[1].set_ylabel('Power')
ax[1].set_xlabel('frequency (Hz)')

cmap = mpl.cm.viridis
cmap = mpl.cm.hot
fig, ax = subplots(1, 2, figsize=(8,4), sharex=True, sharey=True)

# add annotations
ax[0].set_title('S1 stimulation')
ax[1].set_title('M1 stimulation')

for stim in range(num_stims):
    # compute and plot PSD for S1
    f, frq_mat_temp = whisk.get_psd(whisk.wt[s1_stim_inds[stim]][stim_time_inds, 0, :], 500.0)
    whisk.plot_freq(f, frq_mat_temp, axis=ax[0], color=cmap(stim / float(num_stims)))
    ax[0].set_ylim([ylow, yhigh])


###### Plot unit summaries for stimulation triggered averages #####
###### Plot unit summaries for stimulation triggered averages #####

# remove gride lines
sns.set_style("whitegrid", {'axes.grid' : False})
npand   = np.logical_and
dt      = 0.005 # seconds
window  = [-0.06, 0.06]
stim_times = np.arange(0.5, 1.5, 0.050)

with PdfPages(fid + '_unit_stimulation_locked_summaries.pdf') as pdf:
    for uid in range(neuro.num_units):
        fig, ax = subplots(4, 2, figsize=(10,8))
        fig.suptitle('Region: {}, depth: {}, unit type: {}, mouse: {}, driven: {}'.format(\
                neuro.region_dict[neuro.shank_ids[uid]], \
                neuro.depths[uid], \
                neuro.cell_type[uid], \
                fid, \
                neuro.driven_units[uid]))

        best_pos = int(neuro.best_contact[uid])
#        best_pos = 3

        # best position
        spks_per_bin, sem, bins = neuro.eta(stim_times, cond=best_pos, unit_ind=uid, window=window, dt=dt)
        ax[0][0].bar(bins[:-1], spks_per_bin, width=dt, edgecolor='none')
        ax[0][0].set_ylabel('Firing Rate (Hz)')
        #ax[0][0].errorbar(bins[:-1]+dt/2, spks_per_bin, yerr=sem*2.2, fmt='.', color='k')

        # mean angle trace

        # best position S0 silencing
        spks_per_bin, sem, bins = neuro.eta(stim_times, cond=best_pos+9, unit_ind=uid, window=window, dt=dt)
        ax[1][0].bar(bins[:-1], spks_per_bin, width=dt, edgecolor='none')
        ax[1][0].set_title('S1 silencing')
        ax[1][0].set_ylabel('Firing Rate (Hz)')
        #ax[1][0].errorbar(bins[:-1]+dt/2, spks_per_bin, yerr=sem*2.2, fmt='.', color='k')

        # best position M1 silencing
        spks_per_bin, sem, bins = neuro.eta(stim_times, cond=best_pos+9+9, unit_ind=uid, window=window, dt=dt)
        ax[2][0].bar(bins[:-1], spks_per_bin, width=dt, edgecolor='none')
        ax[2][0].set_title('M1 silencing')
        ax[2][0].set_xlabel('Time from protraction (s)')
        ax[2][0].set_ylabel('Firing Rate (Hz)')
        ax[2][0].vlines(0, ax[2][0].get_ylim()[0], ax[2][0].get_ylim()[1], color='r')
        #ax[2][0].errorbar(bins[:-1]+dt/2, spks_per_bin, yerr=sem*2.2, fmt='.', color='k')

        # no contact position
        spks_per_bin, sem, bins = neuro.eta(stim_times, cond=neuro.control_pos-1, unit_ind=uid, window=window, dt=dt)
        ax[0][1].bar(bins[:-1], spks_per_bin, width=dt, edgecolor='none')
        ax[0][1].set_title('No Contact Position')
        #ax[0][1].errorbar(bins[:-1]+dt/2, spks_per_bin, yerr=sem*2.2, fmt='.', color='k')

        # no contact position S1 silencing
        spks_per_bin, sem, bins = neuro.eta(stim_times, cond=neuro.control_pos-1+9, unit_ind=uid, window=window, dt=dt)
        ax[1][1].bar(bins[:-1], spks_per_bin, width=dt, edgecolor='none')
        ax[1][1].set_title('S1 silencing')
        #ax[1][1].errorbar(bins[:-1]+dt/2, spks_per_bin, yerr=sem*2.2, fmt='.', color='k')

        # no contact position M1 silencing
        spks_per_bin, sem, bins = neuro.eta(stim_times, cond=neuro.control_pos-1+9+9, unit_ind=uid, window=window, dt=dt)
        ax[2][1].bar(bins[:-1], spks_per_bin, width=dt, edgecolor='none')
        ax[2][1].set_title('M1 silencing')
        ax[2][1].set_xlabel('Time from protraction (s)')
        #ax[2][1].errorbar(bins[:-1]+dt/2, spks_per_bin, yerr=sem*2.2, fmt='.', color='k')

        ## set ylim to the max ylim of all ETA histogram subplots
        ylim_max = 0
        for rowi, row in enumerate(ax):
            if rowi < 3:
                for coli, col in enumerate(row):
                    if coli < 2:
                        ylim_temp = col.get_ylim()[1]
                        if ylim_temp > ylim_max:
                            ylim_max = ylim_temp
        for rowi, row in enumerate(ax):
            if rowi < 3:
                for coli, col in enumerate(row):
                    if coli < 2:
                        col.set_ylim(0, ylim_max)
                        col.vlines(0, 0, ylim_max, color='r')

        # best position
        mean_trace, err, _ , trace_time = neuro.eta_wt(stim_times, cond=best_pos, kind='angle')
        ax[3][0].plot(trace_time, mean_trace, 'k')
        ax[3][0].fill_between(trace_time, mean_trace - err, mean_trace + err, facecolor='k', alpha=0.3)

        mean_trace, err, _ , trace_time = neuro.eta_wt(stim_times, cond=best_pos+9, kind='angle')
        ax[3][0].plot(trace_time, mean_trace, 'r')
        ax[3][0].fill_between(trace_time, mean_trace - err, mean_trace + err, facecolor='r', alpha=0.3)

        mean_trace, err, _ , trace_time = neuro.eta_wt(stim_times, cond=best_pos+9+9, kind='angle')
        ax[3][0].plot(trace_time, mean_trace, 'b')
        ax[3][0].fill_between(trace_time, mean_trace - err, mean_trace + err, facecolor='b', alpha=0.3)
        ax[3][0].set_ylabel('angle (deg)')
        ax[3][0].set_xlabel('time (s)')
        ax[3][0].vlines([0, 0.010], ax[3][0].get_ylim()[0], ax[3][0].get_ylim()[1], 'b', linestyles='dashed')

        # no contact position
        mean_trace, err, _ , trace_time = neuro.eta_wt(stim_times, cond=neuro.control_pos-1, kind='angle')
        ax[3][1].plot(trace_time, mean_trace, 'k')
        ax[3][1].fill_between(trace_time, mean_trace - err, mean_trace + err, facecolor='k', alpha=0.3)

        mean_trace, err, _ , trace_time = neuro.eta_wt(stim_times, cond=neuro.control_pos-1+9, kind='angle')
        ax[3][1].plot(trace_time, mean_trace, 'r')
        ax[3][1].fill_between(trace_time, mean_trace - err, mean_trace + err, facecolor='r', alpha=0.3)

        mean_trace, err, _ , trace_time = neuro.eta_wt(stim_times, cond=neuro.control_pos-1+9+9, kind='angle')
        ax[3][1].plot(trace_time, mean_trace, 'b')
        ax[3][1].fill_between(trace_time, mean_trace - err, mean_trace + err, facecolor='b', alpha=0.3)
        ax[3][1].set_ylabel('angle (deg)')
        ax[3][1].set_xlabel('time (s)')
        ax[3][1].vlines([0, 0.010], ax[3][1].get_ylim()[0], ax[3][1].get_ylim()[1], 'b', linestyle='dashed')

        pdf.savefig()
        fig.clear()
        plt.close()

# plot average whisker traces (angle) aligned to light onset
fig, ax = subplots(9, 1, sharex=True, sharey=True)
for cond in range(9):
    mean_trace, err, _ , trace_time = neuro.eta_wt(stim_times, cond=cond, kind='angle')
    ax[cond].plot(trace_time, mean_trace, 'k')
    ax[cond].fill_between(trace_time, mean_trace - err, mean_trace + err, facecolor='k', alpha=0.3)

    mean_trace, err, _ , trace_time = neuro.eta_wt(stim_times, cond=cond+9, kind='angle')
    ax[cond].plot(trace_time, mean_trace, 'r')
    ax[cond].fill_between(trace_time, mean_trace - err, mean_trace + err, facecolor='r', alpha=0.3)

    mean_trace, err, _ , trace_time = neuro.eta_wt(stim_times, cond=cond+9+9, kind='angle')
    ax[cond].plot(trace_time, mean_trace, 'b')
    ax[cond].fill_between(trace_time, mean_trace - err, mean_trace + err, facecolor='b', alpha=0.3)

    ax[cond].vlines([0, 0.010], ax[cond].get_ylim()[0], ax[cond].get_ylim()[1], 'b', linestyle='dashed')
    ax[cond].set_title('Position: {}'.format(cond))
    ax[cond].set_ylabel('firing rate (Hz)')

    if cond == 8:
        ax[cond].set_xlabel('time from stimulation (s)')






















##### plot spike triggered averages #####
##### plot spike triggered averages #####

# 1 x 3 subplots (control, S1 silencing, M1 silencing)
sns.set_style("whitegrid", {'axes.grid' : False})
npand   = np.logical_and
dt      = 5 # seconds, phase, degrees
window  = [90, 180] # seconds, phase, degree, etc
bins    = np.arange(window[0], window[1], dt)
wt_type = 0 # {0: 'angle',1: 'set-point',2: 'amplitude',3: 'phase',4: 'velocity'}

# ADD THIS TO THE NEURO CLASS
def st_norm(st_vals, all_vals, wt_type, bins, dt):
    st_count = np.histogram(st_vals[:, wt_type], bins=bins)[0].astype(float)
    all_count = np.histogram(all_vals[:, wt_type], bins=bins)[0].astype(float)
    count_norm = np.nan_to_num(st_count/all_count)/dt
    return count_norm

#with PdfPages(fid + '_unit_protraction_summaries.pdf') as pdf:
# >>> INDENT ME >>>
for uid in range(neuro.num_units):
    fig, ax = subplots(1, 3, figsize=(12,8))
    fig.suptitle('Region: {}, depth: {}, unit type: {}, mouse: {}, driven: {}'.format(\
            neuro.region_dict[neuro.shank_ids[uid]], \
            neuro.depths[uid], \
            neuro.cell_type[uid], \
            fid, \
            neuro.driven_units[uid]))

    # control position no light
    st_vals, all_vals   = neuro.sta_wt(cond=neuro.control_pos-1, unit_ind=uid) # analysis_window=[0.5, 1.5]
    count_norm = st_norm(st_vals, all_vals, wt_type, bins, dt)
    ax[0].bar(bins[:-1], count_norm, width=dt, edgecolor='none')

    # control position S1 silencing
    st_vals, all_vals   = neuro.sta_wt(cond=neuro.control_pos-1+9, unit_ind=uid) # analysis_window=[0.5, 1.5]
    count_norm = st_norm(st_vals, all_vals, wt_type, bins, dt)
    ax[1].bar(bins[:-1], count_norm, width=dt, edgecolor='none')

    # control position M1 silencing
    st_vals, all_vals   = neuro.sta_wt(cond=neuro.control_pos-1+9+9, unit_ind=uid) # analysis_window=[0.5, 1.5]
    count_norm = st_norm(st_vals, all_vals, wt_type, bins, dt)
    ax[2].bar(bins[:-1], count_norm, width=dt, edgecolor='none')

##### quantify modulation depth #####
mod_index = np.zeros((neuro.num_units, 2))
for uid in range(neuro.num_units):
    st_vals, all_vals = neuro.sta_wt(cond=neuro.control_pos-1, unit_ind=uid) # analysis_window=[0.5, 1.5]
    st_count, st_bins = np.histogram(st_vals, bins=bins)
    mod_index[uid, :] = None # calculate modulation index here




















