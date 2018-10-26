import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from lmfit import Model
from scipy.interpolate import UnivariateSpline


def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (sqrt(2*pi) * wid)) * exp(-(x-cen)**2 / (2*wid**2))

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
