import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from lmfit import Model
from scipy.interpolate import UnivariateSpline


##### !!! used to compare my selectivity metric to full-width half-max !!! ###
##### !!! used to compare my selectivity metric to full-width half-max !!! ###

##### good code, keep for possible inclusion in paper methods section #####


def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (sqrt(2*pi) * wid)) * exp(-(x-cen)**2 / (2*wid**2))

def RMSE(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred)**2))

def fwhm(result):
    """calculates full width at half max of fitted Gaussian"""
    amp   = result.best_values['amp']
    mu    = result.best_values['cen']
    sigma = result.best_values['wid']
    x = np.arange(-sigma*5, sigma*5, 0.1)
    gauss = gaussian(x, amp, mu, sigma)
    gauss = gauss - np.max(gauss)/2
    gauss_thresh = np.where(gauss > 0)[0]
    width = x[gauss_thresh[-1]] - x[gauss_thresh[0]]

    return width

x = np.arange(1, 9)

m1_rs = list()
m1_fs = list()
s1_rs = list()
s1_fs = list()

m1_rs_sel = list()
m1_fs_sel = list()
s1_rs_sel = list()
s1_fs_sel = list()

fids = ['1336', '1338', '1339', '1340', '1343', '1345']
experiments = list()
for fid_name in fids:
    get_ipython().magic(u"run hdfanalyzer.py {}".format(fid_name))
    experiments.append(neuro)


#### SCRATCH SPACE ####
#### SCRATCH SPACE ####
cvals_all = np.zeros((neuro.num_units, 4))
mvals_all = np.zeros((neuro.num_units, 4))
sel_all= np.zeros((neuro.num_units, ))
for unit_index in range(neuro.num_units):
    sel_all[unit_index] = neuro.selectivity[unit_index][0]
    #unit_index = 18
    meanr_abs = np.array([np.mean(k[:, unit_index]) for k in neuro.abs_rate])

    counts = np.zeros((1, ))
    x      = np.zeros((1, ))
    xr     = np.arange(0, neuro.control_pos-1)
    #x_test = np.ones((1, ))
    for cond in xr:
        num_trials = neuro.num_good_trials[cond]
        counts = np.concatenate((counts, neuro.abs_rate[cond][:, unit_index]))
        x = np.concatenate((x, np.repeat(cond, num_trials)))
    #    x_test = np.concatenate((x_test, L
    # "normalize" from 0-1
    max_val = np.max(counts)
    meanr_abs = meanr_abs[0:neuro.control_pos-1]
    meanr_abs = meanr_abs / max_val
    counts = counts / max_val

    gmodel0 = Model(gaussian)
    mean_amp = np.mean(counts)
    count_fit = gmodel0.fit(counts, x=x, amp=np.mean(counts), cen=np.mean(x), wid=np.std(x))
    cvals = count_fit.best_values #dictionary of estimated parameters
    cfit  = gaussian(x, cvals['amp'], cvals['cen'], cvals['wid'])
    #cfit_mean = np.unique(np.round(cfit, decimals=4))
    cRMSE = RMSE(counts, cfit)
    print('Counts RMSE: {}'.format(cRMSE))
    cvals_all[unit_index, :] = cvals['amp'], cvals['cen'], cvals['wid'], cRMSE

    gmodel1 = Model(gaussian)
    meanr_fit = gmodel1.fit(meanr_abs, x=xr, amp=np.mean(meanr_abs), cen=neuro.best_contact[unit_index], wid=1)
    mvals = meanr_fit.best_values #dictionary of estimated parameters
    mfit  = gaussian(xr, mvals['amp'], mvals['cen'], mvals['wid'])
    #mfit  = gaussian(x, mvals['amp'], mvals['cen'], mvals['wid'])
    mfit = np.round(mfit, decimals=4)
    #mRMSE = RMSE(counts, mfit)
    mRMSE = RMSE(meanr_abs, mfit)
    print('MeanR RMSE: {}\n'.format(mRMSE))
    mvals_all[unit_index, :] = mvals['amp'], mvals['cen'], mvals['wid'], mRMSE

plt.figure()
plt.scatter(x, counts, c='tab:blue')
plt.plot(x, cfit, 'tab:cyan', xr, meanr_abs, 'tab:red', xr, mfit, 'tab:pink')


#### SCRATCH END ####

for neuro in experiments:
    for unit_index in range(neuro.num_units):

        if neuro.driven_units[unit_index]:
            meanr_abs = np.array([np.mean(k[:, unit_index]) for k in neuro.abs_rate])
            best_contact = np.argmax(meanr_abs[0:8])

#            if best_contact > 0 and best_contact < 7:
            y = meanr_abs[0:8]
            y = y/np.max(y)

            gmodel = Model(gaussian)
            result = gmodel.fit(y, x=x, amp=1, cen=best_contact, wid=1)

#            print(result.fit_report())

            #if result.chisqr < 0.13:
            plt.figure()
            plt.plot(x, y, 'bo--', linewidth=0.5)
        #    plt.plot(x, result.init_fit, 'k--')
            plt.plot(x, result.best_fit, 'r-')
            plt.title('fwhm: {} | chisqr: {} | region: {}'.format(fwhm(result), result.chisqr, neuro.shank_ids[unit_index]))
            plt.show()
            accept = raw_input('accept fit?')
            plt.close()

            if accept == 'y':

                if neuro.shank_ids[unit_index] == 0:

                    if neuro.cell_type[unit_index] == 'RS':
                        m1_rs.append(fwhm(result))
                        m1_rs_sel.append(neuro.selectivity[unit_index, 0])

                    elif neuro.cell_type[unit_index] == 'FS':
                        m1_fs.append(fwhm(result))
                        m1_fs_sel.append(neuro.selectivity[unit_index, 0])
                elif neuro.shank_ids[unit_index] == 1:

                    if neuro.cell_type[unit_index] == 'RS':
                        s1_rs.append(fwhm(result))
                        s1_rs_sel.append(neuro.selectivity[unit_index, 0])

                    elif neuro.cell_type[unit_index] == 'FS':
                        s1_fs.append(fwhm(result))
                        s1_fs_sel.append(neuro.selectivity[unit_index, 0])

import scipy.io as sio
sio.savemat('/home/greg/Desktop/tuning_width.mat',
        {'m1_rs':m1_rs,
         'm1_fs':m1_fs,
         's1_rs':s1_rs,
         's1_fs':s1_fs,
         'm1_rs_sel':m1_rs_sel,
         'm1_fs_sel':m1_fs_sel,
         's1_rs_sel':s1_rs_sel,
         's1_fs_sel':s1_fs_sel,
         })


bins = np.arange(0, 25)
bins_sel = np.arange(0, 1, 0.05)

fig, ax = plt.subplots(2,2)
ax[0][0].hist(m1_rs, bins=bins, color='k', alpha=0.5)
ax[0][0].hist(s1_rs, bins=bins, color='r', alpha=0.5)
ax[0][0].legend(['M1', 'S1'])

ax[1][0].hist(m1_rs_sel, bins=bins_sel, color='k', alpha=0.5)
ax[1][0].hist(s1_rs_sel, bins=bins_sel, color='r', alpha=0.5)

ax[0][1].hist(m1_fs, bins=bins, color='k', alpha=0.5)
ax[0][1].hist(s1_fs, bins=bins, color='r', alpha=0.5)

ax[1][1].hist(m1_fs_sel, bins=bins_sel, color='k', alpha=0.5)
ax[1][1].hist(s1_fs_sel, bins=bins_sel, color='r', alpha=0.5)

ax[0][0].set_title('FWHM RS units')
ax[0][1].set_title('FWHM FS units')
ax[1][0].set_title('Selectivity RS units')
ax[1][1].set_title('Selectivity FS units')


all_fwhm = m1_rs + s1_rs + m1_fs + s1_fs
all_sel  = m1_rs_sel + s1_rs_sel + m1_fs_sel + s1_fs_sel
rho, p = sp.stats.pearsonr(all_sel, all_fwhm)

fig, ax = plt.subplots()
ax.scatter(all_sel, all_fwhm, color='k')
ax.set_xlim(0, 1)
ax.set_ylim(0, 20)

ax.set_xlabel('Selectivity')
ax.set_ylabel('FWHM')
ax.set_title('Correlation coef: {0:0.2f}, p-value: {0:.2e}'.format(rho, p))


























#                plt.figure()
#                plt.plot(x, y, 'bo--', linewidth=0.5)
#            #    plt.plot(x, result.init_fit, 'k--')
#                plt.plot(x, result.best_fit, 'r-')
#                plt.title('chisqr: {} | region: {}'.format(result.chisqr, neuro.shank_ids[unit_index]))
#                plt.show()
