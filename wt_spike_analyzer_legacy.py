import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal
import statsmodels.stats.multitest as smm
from scipy.optimize import curve_fit
import sys
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# change default figure type to PDF
mpl.rcParams['savefig.format'] = 'pdf'
plt.rc('font',family='Arial')
sns.set_style("whitegrid", {'axes.grid' : False})

#fids = ['1289', '1290', '1295', '1302']
#fids = ['1295', '1302', '1328']
fids = ['1330']
# fid with good whisker tracking
fids = ['1336', '1338', '1339', '1340', '1343', '1345']
exps = list()
for fid in fids:
    #get_ipython().magic(u"run neoanalyzer.py {'1290'}")
    get_ipython().magic(u"run hdfanalyzer.py {}".format(fid))
    #neuro.rates(kind='wsk_boolean')
    exps.append(neuro)
# neuro.plot_tuning_curve(kind='evk_count')

# plot all set-point traces
plot(neuro.wtt, neuro.wt[6+9][:,1,:])

#### LDA analysis #####
#### LDA analysis #####
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

###### plot example whisker traces #####
#pos = neuro.control_pos - 1
#npand   = np.logical_and
#stim_time_inds = npand(neuro.wtt >= 0.5, neuro.wtt <= 1.5)
#trial = 2
#
#fig = plt.figure(figsize=(10, 6))
#ax1 = plt.subplot2grid((2,2), (0,0), colspan=2, rowspan=1)
#ax2 = plt.subplot2grid((2,2), (1,0), colspan=1, rowspan=1)
#ax3 = plt.subplot2grid((2,2), (1,1), colspan=1, rowspan=1)
#
#ax1.plot(neuro.wtt, neuro.wt[pos][:, 0, trial], 'k', linewidth=2)
#ax1.plot(neuro.wtt, neuro.wt[pos][:, 1, trial], 'r', linewidth=2)
#ax1.set_ylim([90, 160])
#ax1.vlines([0.5, 1.5], 90, 160, 'b', linestyles='dashed')
#ax1.set_xlabel('time (s)')
#ax1.set_ylabel('angle (deg)')
#
#ax2.plot(neuro.wtt, neuro.wt[pos][:, 0, trial], 'k', linewidth=2)
#ax2.plot(neuro.wtt, neuro.wt[pos][:, 1, trial], 'r', linewidth=2)
#ax2.set_xlim(0.5, 1.5)
#ax2.set_xlabel('time (s)')
#ax2.set_ylabel('angle (deg)')
#
#f, frq_mat_temp = neuro.get_psd(neuro.wt[pos][stim_time_inds, 0, :], 500.0)
#ax3.plot(f, frq_mat_temp[:, trial], 'k', linewidth=2)
#ax3.set_xlim(0, 35)
#ax3.set_xlabel('frequency (Hz)')
#ax3.set_ylabel('power (arb units)')

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

def plot_amplitude(neuro, axis=axis, cond=0, color='k', error='sem'):
    #ax = plt.gca()
    amp_temp = neuro.wt[cond][:, 2, :]
    mean_amp = np.mean(amp_temp, axis=1)
    se      = sp.stats.sem(amp_temp, axis=1)

    # inverse of the CDF is the percentile function. ppf is the percent point funciton of t.
    if error == 'ci':
        err = se*sp.stats.t.ppf((1+0.95)/2.0, amp_temp.shape[1]-1) # (1+1.95)/2 = 0.975
    elif error == 'sem':
        err = se

    axis.plot(neuro.wtt, mean_amp, color)
    axis.fill_between(neuro.wtt, mean_amp - err, mean_amp + err, facecolor=color, alpha=0.3)

ylow, yhigh = 90, 160
ylow, yhigh = 0, 30
for neuro in exps:
    fig, ax = plt.subplots(neuro.control_pos, 2, sharex=True, sharey=True)
    for i in range(2):
        for k in range(neuro.control_pos):
            axis = ax[k][i]
            #plot_setpoint(neuro, axis=axis, cond=k, color='k')
            plot_amplitude(neuro, axis=axis, cond=k, color='k')
            if i == 0:
                #plot_setpoint(neuro, axis=axis, cond=k+9, color='r')
                plot_amplitude(neuro, axis=axis, cond=k+9, color='r')
            else:
                #plot_setpoint(neuro, axis=axis, cond=k+9+9, color='b')
                plot_amplitude(neuro, axis=axis, cond=k+9+9, color='b')
            ax[k][i].vlines([0.5, 1.5], ylow, yhigh, color='b')
            ax[k][i].set_xlim(-0.5, 2.0)
            ax[k][i].set_ylim(ylow, yhigh)
            ax[k][i].set_xlabel('time (s)')
            ax[k][i].set_ylabel('set-point (deg)')
            ax[k][i].set_title('condition {}'.format(str(k)))

##### mean set-points all experiments (control position) #####
fig, ax = plt.subplots(len(exps), 1)
for k, neuro in enumerate(exps):
    plot_setpoint(neuro, axis=ax[k], cond=neuro.control_pos-1, color='k')
    plot_setpoint(neuro, axis=ax[k], cond=neuro.control_pos-1+9, color='r')
    plot_setpoint(neuro, axis=ax[k], cond=neuro.control_pos-1+9+9, color='b')
    ax[k].set_title(neuro.fid)
    ax[k].vlines([0.5, 1.5], 110, 140, color='blue')
    ax[k].set_ylim(115, 135)
    ax[k].set_xlim(0, 2.0)

##### PSDs of angle all experiments (control position) #####
fig, ax = plt.subplots(len(exps), 1)
for k, neuro in enumerate(exps):
    ax[k].set_ylim(0, 15)
    ax[k].set_xlim(0, 35)
    stim_inds = np.logical_and(neuro.wtt > 0.5, neuro.wtt < 1.5)

    # no-light PSD
    f, frq_mat_temp = neuro.get_psd(neuro.wt[neuro.control_pos-1][stim_inds, 0, :], 500)
    neuro.plot_freq(f, frq_mat_temp, axis=ax[k], color='black')

    # s1-light psd
    f, frq_mat_temp = neuro.get_psd(neuro.wt[neuro.control_pos-1+9][stim_inds, 0, :], 500)
    neuro.plot_freq(f, frq_mat_temp, axis=ax[k], color='red')

    # m1-light psd
    f, frq_mat_temp = neuro.get_psd(neuro.wt[neuro.control_pos-1+9+9][stim_inds, 0, :], 500)
    neuro.plot_freq(f, frq_mat_temp, axis=ax[k], color='blue')

fig, ax = plt.subplots(9, 1, sharex=True, sharey=True)
for k in range(9):
    f, frq_mat_temp = neuro.get_psd(neuro.wt[k][stim_inds, 0, :], 500)
    neuro.plot_freq(f, frq_mat_temp, axis=ax[k], color='black')

    f, frq_mat_temp = neuro.get_psd(neuro.wt[k+9][stim_inds, 0, :], 500)
    neuro.plot_freq(f, frq_mat_temp, axis=ax[k], color='red')

    f, frq_mat_temp = neuro.get_psd(neuro.wt[k+9+9][stim_inds, 0, :], 500)
    neuro.plot_freq(f, frq_mat_temp, axis=ax[k], color='blue')

    ax[k].set_xlim(0, 35)

######################################################
sp_s1diff = list()
sp_m1diff = list()
sp_diff  = list()
sp_t_s1  = list()
sp_p_s1  = list()
sp_t_m1  = list()
sp_p_m1  = list()

frq_nolight   = list()
frq_s1light   = list()
frq_m1light   = list()
frq_t_s1  = list()
frq_p_s1  = list()
frq_t_m1  = list()
frq_p_m1  = list()

for neuro in exps:
    print(neuro.fid)
#    neuro = exps[0]
    for k in range(9):
        base_inds = np.logical_and(neuro.wtt > -1.0, neuro.wtt < 0)
        stim_inds = np.logical_and(neuro.wtt > 0.5, neuro.wtt < 1.5)

        ##### set-point #####
        sp_nolight = np.nanmean(neuro.wt[k][stim_inds, 1, :], axis=0)
        sp_s1light = np.nanmean(neuro.wt[k+9][stim_inds, 1, :], axis=0) # s1 silencing
        sp_m1light = np.nanmean(neuro.wt[k+9+9][stim_inds, 1, :], axis=0) # m1 silencing

        sp_s1diff.append(np.nanmean(sp_s1light) - np.nanmean(sp_nolight))
        sp_m1diff.append(np.nanmean(sp_m1light) - np.nanmean(sp_nolight))

        t, p = sp.stats.ttest_ind(sp_s1light, sp_nolight)
        sp_t_s1.append(t); sp_p_s1.append(p)

        t, p = sp.stats.ttest_ind(sp_m1light, sp_nolight)
        sp_t_m1.append(t); sp_p_m1.append(p)

        ##### frequeny #####
        f, frq_mat_temp_nolight = neuro.get_psd(neuro.wt[k][stim_inds, 0, :], 500)
        f, frq_mat_temp_s1light = neuro.get_psd(neuro.wt[k+9][stim_inds, 0, :], 500)
        f, frq_mat_temp_m1light = neuro.get_psd(neuro.wt[k+9+9][stim_inds, 0, :], 500)
        f_inds = np.where(f >= 2.5)[0]

        # for each trial get the peak PSD value
        nolight_temp = f[f_inds[np.argmax(frq_mat_temp_nolight[f_inds, :], axis=0)]]
        s1light_temp = f[f_inds[np.argmax(frq_mat_temp_s1light[f_inds, :], axis=0)]]
        m1light_temp = f[f_inds[np.argmax(frq_mat_temp_m1light[f_inds, :], axis=0)]]

        frq_nolight.append(np.mean(nolight_temp))
        frq_s1light.append(np.mean(s1light_temp))
        frq_m1light.append(np.mean(m1light_temp))

        t, p = sp.stats.ttest_ind(frq_s1light, frq_nolight)
        frq_t_s1.append(t); frq_p_s1.append(p)

        t, p = sp.stats.ttest_ind(frq_m1light, frq_nolight)
        frq_t_m1.append(t); frq_p_m1.append(p)

##### make distribution plots and do statistical test corrections #####
fig, ax = plt.subplots(1, 2)
# set-point
ax[0].hist(sp_s1diff, bins=np.arange(-20,20,1), align='left', color='red', edgecolor='none', alpha=0.5)
ax[0].hist(sp_m1diff, bins=np.arange(-20,20,1), align='left', color='blue', edgecolor='none', alpha=0.5)
rej_s1, pval_corr = smm.multipletests(sp_p_s1, alpha=0.05, method='sh')[:2]
rej_m1, pval_corr = smm.multipletests(sp_p_m1, alpha=0.05, method='sh')[:2]
ax[0].set_title('set-point diff num significant. S1: {}, M1: {}'.format(np.sum(rej_s1), np.sum(rej_m1)))

# frequency
ax[1].hist(frq_nolight, bins=np.arange(-50,50,1), align='left', color='black', edgecolor='none', alpha=0.5)
ax[1].hist(frq_s1light, bins=np.arange(-50,50,1), align='left', color='red', edgecolor='none', alpha=0.5)
ax[1].hist(frq_m1light, bins=np.arange(-50,50,1), align='left', color='blue', edgecolor='none', alpha=0.5)
#rej, pval_corr = smm.multipletests(frq_p, alpha=0.05, method='sh')[:2]
rej_s1, pval_corr = smm.multipletests(frq_p_s1, alpha=0.05, method='sh')[:2]
rej_m1, pval_corr = smm.multipletests(frq_p_m1, alpha=0.05, method='sh')[:2]
ax[1].set_title('frequency diff num significant. S1: {}, M1: {}'.format(np.sum(rej_s1), np.sum(rej_m1)))

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



######################################################
##### Plot unit protraction summaries #####
######################################################

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



######################################################
##### FID1337 GPR26 AAV-ChR2 Analyzer #####
######################################################

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
ax[0][0].vlines(stim_times, ylow, yhigh, color='blue', linewidth=0.5)

# running: M1 stim, control position
ax[0][1].plot(neuro.wtt, neuro.wt[neuro.control_pos-1+9+9][:, 0, :], linewidth=0.25, color='grey')
ax[0][1].plot(neuro.wtt, neuro.wt[neuro.control_pos-1+9+9][:, 0, :].mean(axis=1), linewidth=2.0, color='black')
ax[0][1].set_ylim([ylow, yhigh])
ax[0][1].set_xlim([xlow, xhigh])
ax[0][1].vlines(stim_times, ylow, yhigh, color='blue', linewidth=0.5)

# non-running: POM stim, control position
neuro.rates(running=False)
ax[1][0].plot(neuro.wtt, neuro.wt[neuro.control_pos-1+9][:, 0, :], linewidth=0.25, color='grey')
ax[1][0].plot(neuro.wtt, neuro.wt[neuro.control_pos-1+9][:, 0, :].mean(axis=1), linewidth=2.0, color='black')
ax[1][0].set_ylim([ylow, yhigh])
ax[1][0].set_xlim([xlow, xhigh])
ax[1][0].vlines(stim_times, ylow, yhigh, color='blue', linewidth=0.5)

# non-running: M1 stim, control position
ax[1][1].plot(neuro.wtt, neuro.wt[neuro.control_pos-1+9+9][:, 0, :], linewidth=0.25, color='grey')
ax[1][1].plot(neuro.wtt, neuro.wt[neuro.control_pos-1+9+9][:, 0, :].mean(axis=1), linewidth=2.0, color='black')
ax[1][1].set_ylim([ylow, yhigh])
ax[1][1].set_xlim([xlow, xhigh])
ax[1][1].vlines(stim_times, ylow, yhigh, color='blue', linewidth=0.5)

# temp
inds = list()
for k, s in enumerate(neuro.neo_obj.segments):
    a = s.annotations
    if a['run_boolean'] and a['trial_type'] == 27:
        inds.append(k)

plt.plot(neuro.wtt, neuro.neo_obj.segments[inds[5]].analogsignals[2])
plt.vlines(np.arange(0, 1, 0.050), 100, 165, 'b')


run = s.analogsignals[0]
rt  = s.analogsignals[1]
ang = s.analogsignals[2]
spt = s.analogsignals[3]
amp = s.analogsignals[4]
phs = s.analogsignals[5]
wtt = neuro.wtt

sp.io.savemat('gpr26_stim.mat',\
        {'run':run,\
        'rt':rt,\
        'ang':ang,\
        'spt':spt,\
        'amp':amp,\
        'phs':phs,\
        'wtt':wtt})

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


#############################################################################
###### Plot unit summaries for stimulation triggered averages #####
#############################################################################

# given a sequence of stim times this bins spikes around each time within a
# specified window. It is an event triggered average.

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
fig, ax = subplots(2, 1, sharex=True, sharey=True)
#for cond in range(9):
cond=1
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



############################################################
##### plot spike triggered averages                     #####
##### i.e. spike-phase, spike-setpoint, spike-amplitude #####
#############################################################


# 1 x 3 subplots (control, S1 silencing, M1 silencing)
sns.set_style("whitegrid", {'axes.grid' : False})
npand   = np.logical_and

# for ANGLE
dt      = 5 # degrees, radians (for phase)
window  = [90, 160] # seconds, phase, degree, etc
bins    = np.arange(window[0], window[1], dt)
wt_type = 0 # {0:'angle', 1:'set-point', 2:'amplitude', 3:'phase', 4:'velocity'}

# for set-point
dt      = 1 # degrees, radians (for phase)
window  = [90, 180] # seconds, phase, degree, etc
bins    = np.arange(window[0], window[1], dt)
wt_type = 1 # {0:'angle', 1:'set-point', 2:'amplitude', 3:'phase', 4:'velocity'}

# for PHASE
neuro.get_phase_modulation_depth()
bins = np.linspace(-np.pi, np.pi, 40)
dt   = bins[1] - bins[0]
wt_type = 3 # {0:'angle', 1:'set-point', 2:'amplitude', 3:'phase', 4:'velocity'}

#sample_period is how often a frame occurrs in seconds
# deg*spike / deg*sample*2msec

win_len=5 # for phase poly=3
poly=3
num_boot_samples=1000
md  = neuro.mod_index
mdp = neuro.mod_pval

def sta_bootstrap(data, data_all, wt_type, bins, dt, num_boot_samples):
    num_bins          = count_norm.shape[0]
    boot_dist         = np.zeros((num_boot_samples, num_bins))
    ci_mat            = np.zeros((num_bins, 2))


    if data.shape[0] != 0:
        for k in range(num_boot_samples):
            # collect random samples with replacement
            data_samp         = np.random.choice(data, size=data.shape[0], replace=True)
            data_all_samp        = np.random.choice(data_all, size=data_all.shape[0], replace=True)
            count_norm_samp = neuro.st_norm(data_samp, data_all_samp, 0, bins, dt)
            boot_dist[k, :] = count_norm_samp

        for bin_ind in range(num_bins):
            bin_temp = boot_dist[:, bin_ind]
            ci_mat[bin_ind, 0] = np.percentile(bin_temp, 2.5)
            ci_mat[bin_ind, 1] = np.percentile(bin_temp, 97.5)

    elif data.shape[0] == 0:
        # return ci_mat with zeros
        print('data array is empty...must be good silencing')

    return ci_mat

with PdfPages('/media/greg/data/neuro/' + fid + '_spike_phase_histogram_CIs.pdf') as pdf:
    print('Working on unit:')
    for uid in range(neuro.num_units):
        print(uid)
        fig, ax = subplots(2, 3, figsize=(12,8))
        fig.suptitle('Region: {}, depth: {}, unit type: {}, mouse: {}, driven: {}'.format(\
                neuro.region_dict[neuro.shank_ids[uid]], \
                neuro.depths[uid], \
                neuro.cell_type[uid], \
                fid, \
                neuro.driven_units[uid]), fontsize=18)
        if neuro.shank_ids[uid] == 0:
            c = '#5e819d'
        elif neuro.shank_ids[uid] == 1:
#            c = '#a83c09' # rust
            c = '#a03623' # brick

        # control position no light
        st_vals, all_vals   = neuro.sta_wt(cond=neuro.control_pos-1, unit_ind=uid) # analysis_window=[0.5, 1.5]
        count_norm = neuro.st_norm(st_vals, all_vals, wt_type, bins, dt)
#        ax[0][0].bar(bins[:-1], count_norm, width=dt, edgecolor=c, color=c)
        smooth_data = neuro.sg_smooth(count_norm, win_len, poly, neg_vals=False)
#        ax[0][0].bar(bins[:-1], smooth_data, width=dt, edgecolor=c, color=c)
        # new plotting
        ci_mat = sta_bootstrap(st_vals[:, wt_type], all_vals[:, wt_type], wt_type, bins, dt, num_boot_samples)
        ax[0][0].plot(bins[:-1], smooth_data, 'k', alpha=1)
        ax[0][0].fill_between(bins[:-1], ci_mat[:, 0], ci_mat[:, 1], facecolor=c, alpha=0.5)

        # control position S1 silencing
        st_vals, all_vals   = neuro.sta_wt(cond=neuro.control_pos-1+9, unit_ind=uid) # analysis_window=[0.5, 1.5]
        count_norm = neuro.st_norm(st_vals, all_vals, wt_type, bins, dt)
#        ax[0][1].bar(bins[:-1], count_norm, width=dt, edgecolor=c, color=c)
        smooth_data = neuro.sg_smooth(count_norm, win_len, poly, neg_vals=False)
#        ax[0][1].bar(bins[:-1], smooth_data, width=dt, edgecolor=c, color=c)
        # new plotting

        ## Code it like this!
#        ci_mat = neuro.sta_wt_bootstrap(st_vals[:, wt_type], all_vals[:, wt_type], wt_type, bins, dt, num_boot_samples)
        ci_mat = sta_bootstrap(st_vals[:, wt_type], all_vals[:, wt_type], wt_type, bins, dt, num_boot_samples)
        ax[0][1].plot(bins[:-1], smooth_data, 'k', alpha=1)
        ax[0][1].fill_between(bins[:-1], ci_mat[:, 0], ci_mat[:, 1], facecolor=c, alpha=0.5)


        # control position M1 silencing
        st_vals, all_vals   = neuro.sta_wt(cond=neuro.control_pos-1+9+9, unit_ind=uid) # analysis_window=[0.5, 1.5]
        count_norm = neuro.st_norm(st_vals, all_vals, wt_type, bins, dt)
#        ax[0][2].bar(bins[:-1], count_norm, width=dt, edgecolor=c, color=c)
        smooth_data = neuro.sg_smooth(count_norm, win_len, poly, neg_vals=False)
#        ax[0][2].bar(bins[:-1], smooth_data, width=dt, edgecolor=c, color=c)
        # new plotting
        ci_mat = sta_bootstrap(st_vals[:, wt_type], all_vals[:, wt_type], wt_type, bins, dt, num_boot_samples)
        ax[0][2].plot(bins[:-1], smooth_data, 'k', alpha=1)
        ax[0][2].fill_between(bins[:-1], ci_mat[:, 0], ci_mat[:, 1], facecolor=c, alpha=0.5)

        # best position no light
        st_vals, all_vals   = neuro.sta_wt(cond=neuro.best_contact[uid], unit_ind=uid) # analysis_window=[0.5, 1.5]
        count_norm = neuro.st_norm(st_vals, all_vals, wt_type, bins, dt)
#        ax[1][0].bar(bins[:-1], count_norm, width=dt, edgecolor=c, color=c)
        smooth_data = neuro.sg_smooth(count_norm, win_len, poly, neg_vals=False)
#        ax[1][0].bar(bins[:-1], smooth_data, width=dt, edgecolor=c, color=c)
        # new plotting
        ci_mat = sta_bootstrap(st_vals[:, wt_type], all_vals[:, wt_type], wt_type, bins, dt, num_boot_samples)
        ax[1][0].plot(bins[:-1], smooth_data, 'k', alpha=1)
        ax[1][0].fill_between(bins[:-1], ci_mat[:, 0], ci_mat[:, 1], facecolor=c, alpha=0.5)


        # best position S1 silencing
        st_vals, all_vals   = neuro.sta_wt(cond=neuro.best_contact[uid]+9, unit_ind=uid) # analysis_window=[0.5, 1.5]
        count_norm = neuro.st_norm(st_vals, all_vals, wt_type, bins, dt)
#        ax[1][1].bar(bins[:-1], count_norm, width=dt, edgecolor=c, color=c)
        smooth_data = neuro.sg_smooth(count_norm, win_len, poly, neg_vals=False)
#        ax[1][1].bar(bins[:-1], smooth_data, width=dt, edgecolor=c, color=c)
        # new plotting
        ci_mat = sta_bootstrap(st_vals[:, wt_type], all_vals[:, wt_type], wt_type, bins, dt, num_boot_samples)
        ax[1][1].plot(bins[:-1], smooth_data, 'k', alpha=1)
        ax[1][1].fill_between(bins[:-1], ci_mat[:, 0], ci_mat[:, 1], facecolor=c, alpha=0.5)


        # best position M1 silencing
        st_vals, all_vals   = neuro.sta_wt(cond=neuro.best_contact[uid]+9+9, unit_ind=uid) # analysis_window=[0.5, 1.5]
        count_norm = neuro.st_norm(st_vals, all_vals, wt_type, bins, dt)
#        ax[1][2].bar(bins[:-1], count_norm, width=dt, edgecolor=c, color=c)
        smooth_data = neuro.sg_smooth(count_norm, win_len, poly, neg_vals=False)
#        ax[1][2].bar(bins[:-1], smooth_data, width=dt, edgecolor=c, color=c)
        # new plotting
        ci_mat = sta_bootstrap(st_vals[:, wt_type], all_vals[:, wt_type], wt_type, bins, dt, num_boot_samples)
        ax[1][2].plot(bins[:-1], smooth_data, 'k', alpha=1)
        ax[1][2].fill_between(bins[:-1], ci_mat[:, 0], ci_mat[:, 1], facecolor=c, alpha=0.5)


#        ylim_max = 0
#        for rowi, row in enumerate(ax):
#            for coli, col in enumerate(row):
#                ylim_temp = col.get_ylim()[1]
#                if ylim_temp > ylim_max:
#                    ylim_max = ylim_temp
#        for rowi, row in enumerate(ax):
#            for coli, col in enumerate(row):
#                col.set_ylim(0, ylim_max)
#                col.set_xlim(bins[0], bins[-1])
#                #col.set_xlim(bins[0], 160)
#                col.spines['top'].set_visible(False)
#                col.spines['right'].set_visible(False)
#                col.tick_params(axis='both', direction='out')
#                col.get_xaxis().tick_bottom()
#                col.get_yaxis().tick_left()

        # top left
        fig.subplots_adjust(left=0.12, bottom=0.10, right=0.90, top=0.90, wspace=0.20, hspace=0.30)
        ax[0][0].set_ylabel('Firing rate (Hz)')
        ax[0][0].set_title('Free whisking\nvstrength: {:05.4f},\nvangle: {:05.4f}\np-val: {:05.5f}'\
                .format(md[uid, neuro.control_pos-1, 0], md[uid, neuro.control_pos-1, 1],\
                mdp[uid, neuro.control_pos-1]))

        # top middle
        ax[0][1].set_title('Free whisking\nS1 silencing\nvstrength: {:05.4f},\nvangle: {:05.4f}\np-val: {:05.5f}'\
                .format(md[uid, neuro.control_pos-1+9, 0], md[uid, neuro.control_pos-1+9, 1],\
                mdp[uid, neuro.control_pos-1+9]))

        # top right
        ax[0][2].set_title('Free whisking\nM1 silencing\nvstrength: {:05.4f},\nvangle: {:05.4f}\np-val: {:05.5f}'\
                .format(md[uid, neuro.control_pos-1+9+9, 0], md[uid, neuro.control_pos-1+9+9, 1],\
                mdp[uid, neuro.control_pos-1+9+9]))

        # bottom left
        ax[1][0].set_ylabel('Firing rate (Hz)')
        ax[1][0].set_xlabel('Phase (rad)')
        ax[1][0].set_title('Best contact\nvstrength: {:05.4f},\nvangle: {:05.4f}\np-val: {:05.5f}'\
                .format(md[uid, neuro.best_contact[uid], 0], md[uid, neuro.best_contact[uid], 1],\
                mdp[uid, neuro.best_contact[uid]]))

        # bottom middle
        ax[1][1].set_xlabel('Phase (rad)')
        ax[1][1].set_title('Best contact\nS1 silencing\nvstrength: {:05.4f},\nvangle: {:05.4f}\np-val: {:05.5f}'\
                .format(md[uid, neuro.best_contact[uid]+9, 0], md[uid, neuro.best_contact[uid]+9, 1],\
                mdp[uid, neuro.best_contact[uid]+9]))

        # bottom right
        ax[1][2].set_xlabel('Phase (rad)')
        ax[1][2].set_title('Best contact\nM1 silencing\nvstrength: {:05.4f},\nvangle: {:05.4f}\np-val: {:05.5f}'\
                .format(md[uid, neuro.best_contact[uid]+9+9, 0], md[uid, neuro.best_contact[uid]+9+9, 1],\
                mdp[uid, neuro.best_contact[uid]+9+9]))

        fig.subplots_adjust(left=0.10, bottom=0.10, right=0.90, top=0.80, wspace=0.20, hspace=0.50)

        pdf.savefig()
        fig.clear()
        plt.close()



###################################################
##### plot spike spike-phase polar plots #####
###################################################

# polar scatter plot can handle negative angles

npand   = np.logical_and
neuro.get_phase_modulation_index()

m1_rs = npand(neuro.shank_ids == 0, neuro.cell_type == 'RS')
s1_rs = npand(neuro.shank_ids == 1, neuro.cell_type == 'RS')
m1_fs = npand(neuro.shank_ids == 0, neuro.cell_type == 'FS')
s1_fs = npand(neuro.shank_ids == 1, neuro.cell_type == 'FS')

fig, ax = plt.subplots(2, 2, figsize=(10,6),subplot_kw=dict(polar=True)) #fig.suptitle('Region: {}, depth: {}, unit type: {}, mouse: {}, driven: {}'.format(\

# top left: control position # M1 RS units with and without light
ax[0][0].scatter(neuro.mod_index[m1_rs, neuro.control_pos-1, 1],\
        neuro.mod_index[m1_rs, neuro.control_pos-1, 0], c='k')

ax[0][0].scatter(neuro.mod_index[m1_rs, neuro.control_pos-1+9, 1],\
        neuro.mod_index[m1_rs, neuro.control_pos-1+9, 0], c='b')

# bottom left: control position # S1 RS units with and without light
ax[1][0].scatter(neuro.mod_index[s1_rs, neuro.control_pos-1, 1],\
        neuro.mod_index[s1_rs, neuro.control_pos-1, 0], c='k')

ax[1][0].scatter(neuro.mod_index[s1_rs, neuro.control_pos-1+9+9, 1],\
        neuro.mod_index[s1_rs, neuro.control_pos-1+9+9, 0], c='b')

# top right: best contact position # M1 FS units with and without light
ax[0][1].scatter(neuro.mod_index[m1_fs, neuro.best_contact[m1_fs], 1],\
        neuro.mod_index[m1_fs, neuro.best_contact[m1_fs], 0], c='k')

ax[0][1].scatter(neuro.mod_index[m1_fs, neuro.best_contact[m1_fs]+9, 1],\
        neuro.mod_index[m1_fs, neuro.best_contact[m1_fs]+9, 0], c='r')

# top right: best contact position # S1 FS units with and without light
ax[1][1].scatter(neuro.mod_index[s1_fs, neuro.best_contact[s1_fs], 1],\
        neuro.mod_index[s1_fs, neuro.best_contact[s1_fs], 0], c='k')

ax[1][1].scatter(neuro.mod_index[s1_fs, neuro.best_contact[s1_fs]+9+9, 1],\
        neuro.mod_index[s1_fs, neuro.best_contact[s1_fs]+9+9, 0], c='r')

# set plot labels
# top left
ax[0][0].set_title('M1 RS units\nNo contact')
# bottom left
ax[1][0].set_title('S1 RS units\nNo contact')
#ax[1][0].set_ylim(0,0.3)
# top right
ax[0][1].set_title('M1 FS units\nNo contact')
# bottom right
ax[1][1].set_title('S1 FS units\nNo contact')


fig.subplots_adjust(top=0.90, bottom=0.05, left=0.10, right=0.90, hspace=0.35, wspace=0.20)

## polar plot toy example
#theta = np.arange(-np.pi, np.pi, 5*np.pi/180)
#r = np.linspace(0, 3, theta.shape[0])
#colors = theta
#ax = subplot(111, polar=True)
#ax.set_ylim(0,5)
##c = scatter(theta, r) # one color for all the dots
#c = scatter(theta, r, c=colors) # unique colors for all the dots


################################################################
##### plot spike spike-phase vector strength scatter plot  #####
################################################################

neuro.get_phase_modulation_index()

m1_rs = npand(npand(neuro.shank_ids==0, neuro.driven_units==True), neuro.cell_type=='RS')
s1_rs = npand(npand(neuro.shank_ids==1, neuro.driven_units==True), neuro.cell_type=='RS')
m1_fs = npand(npand(neuro.shank_ids==0, neuro.driven_units==True), neuro.cell_type=='FS')
s1_fs = npand(npand(neuro.shank_ids==1, neuro.driven_units==True), neuro.cell_type=='FS')

fig, ax = plt.subplots(2, 2, figsize=(10,9), sharex=True, sharey=True)
fig.suptitle('spike-phase paired vector strength')

# top left: M1 control position
ax[0][0].plot(neuro.mod_index[m1_rs, neuro.control_pos-1, 0],\
        neuro.mod_index[m1_rs, neuro.control_pos-1+9, 0], 'ok', label='RS')
ax[0][0].plot(neuro.mod_index[m1_fs, neuro.control_pos-1, 0],\
        neuro.mod_index[m1_fs, neuro.control_pos-1+9, 0], 'ob', label='FS')
ax[0][0].set_title('M1 No contact')
ax[0][0].set_xlabel('No Light')
ax[0][0].set_ylabel('S1 Silencing')
ax[0][0].legend(loc='upper left')

# bottom left: M1 best positions
ax[1][0].plot(neuro.mod_index[m1_rs, neuro.best_contact[m1_rs], 0],\
        neuro.mod_index[m1_rs, neuro.best_contact[m1_rs]+9, 0], 'ok', label='RS')
ax[1][0].plot(neuro.mod_index[m1_fs, neuro.best_contact[m1_fs], 0],\
        neuro.mod_index[m1_fs, neuro.best_contact[m1_fs]+9, 0], 'ob', label='FS')
ax[1][0].set_title('M1 contact')
ax[1][0].set_xlabel('No Light')
ax[1][0].set_ylabel('S1 Silencing')
ax[1][0].legend()

# top right: S1 best positions
ax[0][1].plot(neuro.mod_index[s1_rs, neuro.control_pos-1, 0],\
        neuro.mod_index[s1_rs, neuro.control_pos-1+9+9, 0], 'ok', label='RS')
ax[0][1].plot(neuro.mod_index[s1_fs, neuro.control_pos-1, 0],\
        neuro.mod_index[s1_fs, neuro.control_pos-1+9+9, 0], 'ob', label='FS')
ax[0][1].set_title('S1 No contact')
ax[0][1].set_xlabel('No Light')
ax[0][1].set_ylabel('S1 Silencing')
ax[0][1].legend(loc='upper left')

# top right: S1 best positions
ax[1][1].plot(neuro.mod_index[s1_rs, neuro.best_contact[s1_rs]-1, 0],\
        neuro.mod_index[s1_rs, neuro.best_contact[s1_rs]-1+9+9, 0], 'ok', label='RS')
ax[1][1].plot(neuro.mod_index[s1_fs, neuro.best_contact[s1_fs]-1, 0],\
        neuro.mod_index[s1_fs, neuro.best_contact[s1_fs]-1+9+9, 0], 'ob', label='FS')
ax[1][1].set_title('S1 Best contact')
ax[1][1].set_xlabel('No Light')
ax[1][1].set_ylabel('S1 Silencing')
ax[1][1].legend(loc='upper left')

# set ylim to the max ylim of all subplots and plot line of unity
ylim_max = 0
xlim_max = 0
for row in ax:
    for col in row:
        ylim_temp = np.max(np.abs(col.get_ylim()))
        xlim_temp = np.max(np.abs(col.get_xlim()))
        if ylim_temp > ylim_max:
            ylim_max = ylim_temp
        if xlim_temp > xlim_max:
            xlim_max = xlim_temp
for row in ax:
    for col in row:
#        col.set_ylim(-ylim_max, ylim_max)
#        col.set_xlim(-xlim_max, xlim_max)
        col.plot([-xlim_max, xlim_max], [-ylim_max, ylim_max], 'k')



################################################################
##### plot spike spike-phase preferred phase scatter plot  #####
################################################################

neuro.get_phase_modulation_index()

m1_rs = npand(npand(neuro.shank_ids==0, neuro.driven_units==True), neuro.cell_type=='RS')
s1_rs = npand(npand(neuro.shank_ids==1, neuro.driven_units==True), neuro.cell_type=='RS')
m1_fs = npand(npand(neuro.shank_ids==0, neuro.driven_units==True), neuro.cell_type=='FS')
s1_fs = npand(npand(neuro.shank_ids==1, neuro.driven_units==True), neuro.cell_type=='FS')

fig, ax = plt.subplots(2, 2, figsize=(10,9), sharex=True, sharey=True)
fig.suptitle('spike-phase paired preferred phase')

# top left: M1 control position
ax[0][0].plot(neuro.mod_index[m1_rs, neuro.control_pos-1, 1],\
        neuro.mod_index[m1_rs, neuro.control_pos-1+9, 1], 'ok', label='RS')
ax[0][0].plot(neuro.mod_index[m1_fs, neuro.control_pos-1, 1],\
        neuro.mod_index[m1_fs, neuro.control_pos-1+9, 1], 'ob', label='FS')
ax[0][0].set_title('M1 No contact')
ax[0][0].set_xlabel('No Light')
ax[0][0].set_ylabel('S1 Silencing')
ax[0][0].legend(loc='upper left')

# bottom left: M1 best positions
ax[1][0].plot(neuro.mod_index[m1_rs, neuro.best_contact[m1_rs], 1],\
        neuro.mod_index[m1_rs, neuro.best_contact[m1_rs]+9, 1], 'ok', label='RS')
ax[1][0].plot(neuro.mod_index[m1_fs, neuro.best_contact[m1_fs], 1],\
        neuro.mod_index[m1_fs, neuro.best_contact[m1_fs]+9, 1], 'ob', label='FS')
ax[1][0].set_title('M1 Best contact')
ax[1][0].set_xlabel('No Light')
ax[1][0].set_ylabel('S1 Silencing')
ax[1][0].legend()

# top right: S1 best positions
ax[0][1].plot(neuro.mod_index[s1_rs, neuro.control_pos-1, 1],\
        neuro.mod_index[s1_rs, neuro.control_pos-1+9+9, 1], 'ok', label='RS')
ax[0][1].plot(neuro.mod_index[s1_fs, neuro.control_pos-1, 1],\
        neuro.mod_index[s1_fs, neuro.control_pos-1+9+9, 1], 'ob', label='FS')
ax[0][1].set_title('S1 No contact')
ax[0][1].set_xlabel('No Light')
ax[0][1].set_ylabel('S1 Silencing')
ax[0][1].legend(loc='upper left')

# top right: S1 best positions
ax[1][1].plot(neuro.mod_index[s1_rs, neuro.best_contact[s1_rs]-1, 1],\
        neuro.mod_index[s1_rs, neuro.best_contact[s1_rs]-1+9+9, 1], 'ok', label='RS')
ax[1][1].plot(neuro.mod_index[s1_fs, neuro.best_contact[s1_fs]-1, 1],\
        neuro.mod_index[s1_fs, neuro.best_contact[s1_fs]-1+9+9, 1], 'ob', label='FS')
ax[1][1].set_title('S1 Best contact')
ax[1][1].set_xlabel('No Light')
ax[1][1].set_ylabel('S1 Silencing')
ax[1][1].legend(loc='upper left')

# set ylim to the max ylim of all subplots and plot line of unity
ylim_max = 0
xlim_max = 0
for row in ax:
    for col in row:
        ylim_temp = np.max(np.abs(col.get_ylim()))
        xlim_temp = np.max(np.abs(col.get_xlim()))
        if ylim_temp > ylim_max:
            ylim_max = ylim_temp
        if xlim_temp > xlim_max:
            xlim_max = xlim_temp
for row in ax:
    for col in row:
#        col.set_ylim(-ylim_max, ylim_max)
#        col.set_xlim(-xlim_max, xlim_max)
        col.plot([-xlim_max, xlim_max], [-ylim_max, ylim_max], 'k')


######################################################
##### multiple experiment whisking analysis #####
######################################################
######################################################
##### multiple experiment whisking analysis #####
######################################################

# poly2 trode experiments with whisker tracking
fids = ['1336','1338', '1339', '1340', '1343', '1345']

##### Load experiments #####
mxperiments = list()
for fid in fids:
    get_ipython().magic(u"run neoanalyzer.py {}".format(fid))
    # del neo objects to save memory
    del neuro.neo_obj
    del block
    del exp1
    del manager
    experiments.append(neuro)

##### create arrays and lists for concatenating specified data from all experiments
cell_type    = list()
shank_ids    = np.empty((1, ))
depths       = np.empty((1, ))
driven       = np.empty((1, ))
best_contact = np.empty((1, ))
mod_index    =  np.empty((1, 27, 3))
mod_pval     =  np.empty((1, 27))
abs_rate     = np.empty((1, 27, 2))
evk_rate     = np.empty((1, 27, 2))

for neuro in experiments:
    # calculate measures that weren't calculated at init
    neuro.get_best_contact()
    neuro.get_phase_modulation_depth()

    # concatenate measures
    cell_type.extend(neuro.cell_type)
    shank_ids    = np.append(shank_ids, neuro.shank_ids)
    depths       = np.append(depths, np.asarray(neuro.depths))
    driven       = np.append(driven, neuro.driven_units, axis=0)
    best_contact = np.append(best_contact, neuro.best_contact)
    mod_index    = np.concatenate( (mod_index, neuro.mod_index), axis=0)

    for unit_index in range(neuro.num_units):

        # compute absolute rate (mean and sem)
        temp = np.zeros((1, 27, 2))
        temp[0, :, 0] = np.array([np.mean(k[:, unit_index]) for k in neuro.abs_rate])[:]
        temp[0, :, 1] = np.array([sp.stats.sem(k[:, unit_index]) for k in neuro.abs_rate])
        abs_rate = np.append(abs_rate, temp, axis=0)

        # compute evoked rate (mean and sem)
        temp = np.zeros((1, 27, 2))
        temp[0, :, 0] = np.array([np.mean(k[:, unit_index]) for k in neuro.evk_rate])[:]
        temp[0, :, 1] = np.array([sp.stats.sem(k[:, unit_index]) for k in neuro.evk_rate])
        evk_rate = np.append(evk_rate, temp, axis=0)

cell_type = np.asarray(cell_type)

shank_ids = shank_ids[1:,]
shank_ids = shank_ids.astype(int)

depths = depths[1:,]

driven = driven[1:,]
driven = driven.astype(int)

best_contact = best_contact[1:,]
best_contact = best_contact.astype(int)

mod_index = np.nan_to_num(mod_index[1:, :, :])

abs_rate    = abs_rate[1:, :]
evk_rate    = evk_rate[1:, :]

########################################
##### multiple experiment analysis #####
########################################


##################################################################
##### multiple experiment plot spike spike-phase polar plots #####
##################################################################

control_pos = 9
npand = np.logical_and

rs_fs = 'FS'
if rs_fs == 'RS':
    c = 'ok'
elif rs_fs == 'FS':
    c = 'ob'

m1_inds = npand(npand(shank_ids==0, driven==True), cell_type==rs_fs)
s1_inds = npand(npand(shank_ids==1, driven==True), cell_type==rs_fs)
#m1_inds = npand(shank_ids==0, cell_type==rs_fs)
#s1_inds = npand(shank_ids==1, cell_type==rs_fs)

## bins for density plot
#dt = 2*np.pi/10
#theta_bins    = np.linspace(-np.pi, np.pi+dt, 15)
#vstrength_bins = np.arange(0, 1, 0.1)

### M1 plot
### M1 plot
fig, ax = plt.subplots(2, 2, figsize=(10,6),subplot_kw=dict(polar=True)) #fig.suptitle('Region: {}, depth: {}, unit type: {}, mouse: {}, driven: {}'.format(\

# top left: control position # M1 RS no light
## density plot
# https://stackoverflow.com/questions/40327794/contour-density-plot-in-matplotlib-using-polar-coordinates
#H, theta_edges, r_edges = np.histogram2d(mod_index[m1_inds, control_pos-1, 1],\
#        mod_index[m1_inds, control_pos-1, 0], bins=(theta_bins, vstrength_bins))
##r_mid     = .5 * (r_edges[:-1] + r_edges[1:])
#theta_mid = .5 * (theta_edges[:-1] + theta_edges[1:])
#ax[0][0].contourf(theta_mid, r_edges[:-1], H.T, 20, vmin=0, cmap=plt.cm.Spectral)
# scatter plot
ax[0][0].scatter(mod_index[m1_inds, control_pos-1, 1],\
        mod_index[m1_inds, control_pos-1, 0], c='k', s=15)


# top right: best contact position # M1 RS no light
#scatter
ax[0][1].scatter(mod_index[m1_inds, 6, 1],\
        mod_index[m1_inds, 6, 0], c='k', s=15)


# bottom left: control position # M1 RS units with S1 silencing
# scatter
ax[1][0].scatter(mod_index[m1_inds, control_pos-1+9, 1],\
        mod_index[m1_inds, control_pos-1+9, 0], c='k', s=15)


# bottom right: best contact position # M1 RS units with S1 silencing
#scatter
ax[1][1].scatter(mod_index[m1_inds, 6+9, 1],\
        mod_index[m1_inds, 6+9, 0], c='k', s=15)


# set plot labels
fig.suptitle('M1 {} units'.format(rs_fs))
# top left
ax[0][0].set_title('No contact')
ax[0][0].set_ylim(0,.3)
# bottom left
ax[1][0].set_title('No contact + S1 silencing')
ax[1][0].set_ylim(0,0.3)
# top right
ax[0][1].set_title('Best contact')
ax[0][1].set_ylim(0,.3)
# bottom right
ax[1][1].set_title('Best contact + S1 silencing')
ax[1][1].set_ylim(0,.3)

fig.subplots_adjust(top=0.90, bottom=0.05, left=0.10, right=0.90, hspace=0.35, wspace=0.20)



### S1 plot
### S1 plot
fig, ax = plt.subplots(2, 2, figsize=(10,6),subplot_kw=dict(polar=True)) #fig.suptitle('Region: {}, depth: {}, unit type: {}, mouse: {}, driven: {}'.format(\


# top left: control position # S1 RS no light
## density plot
#H, theta_edges, r_edges = np.histogram2d(mod_index[s1_inds, control_pos-1, 1],\
#        mod_index[s1_inds, control_pos-1, 0], bins=(theta_bins, vstrength_bins))
##r_mid     = .5 * (r_edges[:-1] + r_edges[1:])
#theta_mid = .5 * (theta_edges[:-1] + theta_edges[1:])
#ax[0][0].contourf(theta_mid, r_edges[:-1], H.T, 20, vmin=0, cmap=plt.cm.)
# scatter plot
ax[0][0].scatter(mod_index[s1_inds, control_pos-1, 1],\
        mod_index[s1_inds, control_pos-1, 0], c='k', s=15)


# top right: best contact position # S1 RS no light
#scatter
ax[0][1].scatter(mod_index[s1_inds, best_contact[s1_inds], 1],\
        mod_index[s1_inds, best_contact[s1_inds], 0], c='k', s=15)


# bottom left: control position # S1 RS units with S1 silencing
# scatter
ax[1][0].scatter(mod_index[s1_inds, control_pos-1+9+9, 1],\
        mod_index[s1_inds, control_pos-1+9+9, 0], c='k', s=15)


# top right: best contact position # S1 RS units with and without light
#scatter ax[1][1].scatter(mod_index[s1_inds, best_contact[s1_inds]+9+9, 1],\
        mod_index[s1_inds, best_contact[s1_inds]+9+9, 0], c='k', s=15)


# set plot labels
fig.suptitle('S1 {} units'.format(rs_fs))
# top left
ax[0][0].set_title('No contact')
ax[0][0].set_ylim(0,0.3)
# bottom left
ax[1][0].set_title('No contact + M1 silencing')
ax[1][0].set_ylim(0,0.3)
# top right
ax[0][1].set_title('Best contact')
ax[0][1].set_ylim(0,0.3)
# bottom right
ax[1][1].set_title('Best contact + M1 silencing')
ax[1][1].set_ylim(0,.3)

fig.subplots_adjust(top=0.90, bottom=0.05, left=0.10, right=0.90, hspace=0.35, wspace=0.20)




####################################################################################
##### multiple experiment plot spike spike-phase vector strength scatter plots #####
####################################################################################

control_pos = 9
npand = np.logical_and

rs_fs = 'RS'
if rs_fs == 'RS':
    c = 'ok'
elif rs_fs == 'FS':
    c = 'ob'

m1_inds = npand(npand(shank_ids==0, driven==True), cell_type==rs_fs)
s1_inds = npand(npand(shank_ids==1, driven==True), cell_type==rs_fs)
m1_inds = npand(shank_ids==0, cell_type==rs_fs)
s1_inds = npand(shank_ids==1, cell_type==rs_fs)

fig, ax = plt.subplots(2, 2, figsize=(10,9), sharex=True, sharey=True)
fig.suptitle('spike-phase paired vector strength {} cells'.format(rs_fs))

# top left: M1 control position
ax[0][0].plot(mod_index[m1_inds, control_pos-1, 0],\
        mod_index[m1_inds, control_pos-1+9, 0], c, label=rs_fs)
ax[0][0].set_title('M1 No contact')
ax[0][0].set_xlabel('No Light')
ax[0][0].set_ylabel('S1 Silencing')

# top right: M1 best positions
ax[0][1].plot(mod_index[m1_inds, best_contact[m1_inds], 0],\
        mod_index[m1_inds, best_contact[m1_inds]+9, 0], c, label=rs_fs)
ax[0][1].set_title('M1 contact')
ax[0][1].set_xlabel('No Light')
ax[0][1].set_ylabel('S1 Silencing')

# bottom left: S1 control positions
ax[1][0].plot(mod_index[s1_inds, control_pos-1, 0],\
        mod_index[s1_inds, control_pos-1+9+9, 0], c, label=rs_fs)
#ax[1][0].plot(mod_index[s1_fs, control_pos-1, 0],\
#        mod_index[s1_fs, control_pos-1+9+9, 0], 'ob', label='FS')
ax[1][0].set_title('S1 No contact')
ax[1][0].set_xlabel('No Light')
ax[1][0].set_ylabel('S1 Silencing')

# top right: S1 best positions
ax[1][1].plot(mod_index[s1_inds, best_contact[s1_inds], 0],\
        mod_index[s1_inds, best_contact[s1_inds]+9+9, 0], c, label=rs_fs)
ax[1][1].set_title('S1 Best contact')
ax[1][1].set_xlabel('No Light')
ax[1][1].set_ylabel('S1 Silencing')

xlim_max, ylim_max = 0.5, 0.5
xlim_min, ylim_min = 0, 0
for row in ax:
    for col in row:
        col.set_ylim(ylim_min, ylim_max)
        col.set_xlim(xlim_min, xlim_max)
        col.plot([xlim_min, xlim_max], [ylim_min, ylim_max], 'k')

##### Stats tests #####

# M1 Units
#no contact
r, p = sp.stats.wilcoxon(mod_index[m1_inds, control_pos-1, 0], mod_index[m1_inds, control_pos-1+9, 0])
# best contact
r, p = sp.stats.wilcoxon(mod_index[m1_inds, best_contact[m1_inds], 0], mod_index[m1_inds, best_contact[m1_inds]+9, 0])

# S1 Units
#no contact
r, p = sp.stats.wilcoxon(mod_index[s1_inds, control_pos-1, 0], mod_index[s1_inds, control_pos-1+9+9, 0])
# best contact
r, p = sp.stats.wilcoxon(mod_index[s1_inds, best_contact[s1_inds], 0], mod_index[s1_inds, best_contact[s1_inds]+9+9, 0])



##############################################################################
##### multiple experiment plot spike-phase preferred phase scatter plot  #####
##############################################################################

control_pos = 9
npand = np.logical_and

rs_fs = 'FS'
if rs_fs == 'RS':
    c = 'ok'
elif rs_fs == 'FS':
    c = 'ob'

m1_inds = npand(npand(shank_ids==0, driven==True), cell_type==rs_fs)
s1_inds = npand(npand(shank_ids==1, driven==True), cell_type==rs_fs)
#m1_inds = npand(shank_ids==0, cell_type==rs_fs)
#s1_inds = npand(shank_ids==1, cell_type==rs_fs)

fig, ax = plt.subplots(2, 2, figsize=(10,9), sharex=True, sharey=True)
fig.suptitle('spike-phase paired preferred phase')

# top left: M1 control position
ax[0][0].plot(mod_index[m1_inds, control_pos-1, 1],\
        mod_index[m1_inds, control_pos-1+9, 1], c, label=rs_fs)
ax[0][0].set_title('M1 No contact')
ax[0][0].set_xlabel('No Light')
ax[0][0].set_ylabel('S1 Silencing')

# top right: M1 best positions
ax[0][1].plot(mod_index[m1_inds, best_contact[m1_inds], 1],\
        mod_index[m1_inds, best_contact[m1_inds]+9, 1], c, label=rs_fs)
ax[0][1].set_title('M1 Best contact')
ax[0][1].set_xlabel('No Light')
ax[0][1].set_ylabel('S1 Silencing')

# bottom left: S1 best positions
ax[1][0].plot(mod_index[s1_inds, control_pos-1, 1],\
        mod_index[s1_inds, control_pos-1+9+9, 1], c, label=rs_fs)
ax[1][0].set_title('S1 No contact')
ax[1][0].set_xlabel('No Light')
ax[1][0].set_ylabel('S1 Silencing')

# top right: S1 best positions
ax[1][1].plot(mod_index[s1_inds, best_contact[s1_inds]-1, 1],\
        mod_index[s1_inds, best_contact[s1_inds]-1+9+9, 1], c, label=rs_fs)
ax[1][1].set_title('S1 Best contact')
ax[1][1].set_xlabel('No Light')
ax[1][1].set_ylabel('S1 Silencing')

# set ylim to the max ylim of all subplots and plot line of unity
for row in ax:
    for col in row:
        col.set_ylim(-4, 4)
        col.set_xlim(-4, 4)
        col.plot([-4, 4], [-4, 4], 'k')



#################################################################################
##### multiple experiment plot spike-phase vector strength region histogram #####
#################################################################################

npand = np.logical_and
# driven units
m1_rs = npand(npand(shank_ids==0, driven==True), cell_type=='RS')
s1_rs = npand(npand(shank_ids==1, driven==True), cell_type=='RS')
m1_fs = npand(npand(shank_ids==0, driven==True), cell_type=='FS')
s1_fs = npand(npand(shank_ids==1, driven==True), cell_type=='FS')

# all units
m1_rs = npand(shank_ids==0, cell_type=='RS')
s1_rs = npand(shank_ids==1, cell_type=='RS')
m1_fs = npand(shank_ids==0, cell_type=='FS')
s1_fs = npand(shank_ids==1, cell_type=='FS')
bins = np.arange(0, 0.35, 0.025)


##### Changed plot setting to normed and cumulative to True
#####  DEBUG THIS!
fig, ax = plt.subplots(1, 2, figsize=(10,4), sharex=True, sharey=True)
fig.suptitle('spike-phase paired vector strength')
ax[0].hist(mod_index[m1_rs, control_pos, 0], bins=bins, alpha=0.3, label='M1 RS', color='k', normed=True)
ax[0].hist(mod_index[s1_rs, control_pos, 0], bins=bins, alpha=0.3, label='S1 RS', color='r', normed=True)
ax[0].hist(mod_index[m1_rs, control_pos, 0], bins=bins, alpha=0.3, label='M1 RS', color='k', normed=True, cumulative=True)
ax[0].hist(mod_index[s1_rs, control_pos, 0], bins=bins, alpha=0.3, label='S1 RS', color='r', normed=True, cumulative=True)
ax[0].legend()
ax[0].set_xlabel('vector strength')

ax[1].hist(mod_index[m1_fs, control_pos, 0], bins=bins, alpha=0.3, label='M1 FS', color='k', normed=True)
ax[1].hist(mod_index[s1_fs, control_pos, 0], bins=bins, alpha=0.3, label='S1 FS', color='r', normed=True)
ax[1].hist(mod_index[m1_fs, control_pos, 0], bins=bins, alpha=0.3, label='M1 FS', color='k', normed=True, cumulative=True)
ax[1].hist(mod_index[s1_fs, control_pos, 0], bins=bins, alpha=0.3, label='S1 FS', color='r', normed=True, cumulative=True)
ax[1].legend()
ax[1].set_xlabel('vector strength')














##### simulate vector strength distribution

import pycircstat as pycirc

num_samples = 5000
vec_dist    = np.zeros((num_samples,))
dt          = 0.001

for k in range(num_samples):

    # randomly select angles
    rand_angles = 2*np.pi*np.random.uniform(size=500)

    # calculate vector strength
    vec_dist[k] = pycirc.descriptive.resultant_vector_length(rand_angles) # angles in radian, weightings (counts)

fig, ax = plt.subplots(1, 1)
ax.hist(vec_dist, bins=np.arange(0, 1, dt), width=dt, normed=True)
ax.hist(vec_dist, bins=np.arange(0, 1, dt), normed=True, cumulative=True, histtype='step')




# example whisker traces

trial=7
fig, ax = plt.subplots(2, 1)
ax[0].plot(neuro.wtt, neuro.wt[8][:, 0, trial], 'k')
ax[0].plot(neuro.wtt, neuro.wt[8+9][:, 0, trial], 'r')
ax[0].plot(neuro.wtt, neuro.wt[8+9+9][:, 0, trial], 'b')
ax[0].set_xlim(0, 1.5)
ax[0].set_ylabel('angle (deg)')
ax[0].set_title('angle')

ax[1].plot(neuro.wtt, neuro.wt[8][:, 1, trial], 'k')
ax[1].plot(neuro.wtt, neuro.wt[8+9][:, 1, trial], 'r')
ax[1].plot(neuro.wtt, neuro.wt[8+9+9][:, 1, trial], 'b')
ax[1].set_xlim(0, 1.5)
ax[1].set_ylabel('angle (deg)')
ax[1].set_xlabel('time (s)')
ax[1].set_title('set-point')















