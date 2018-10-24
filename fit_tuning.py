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

            if best_contact > 0 and best_contact < 7:
                y = meanr_abs[0:8]
                y = y/np.max(y)

                gmodel = Model(gaussian)
                result = gmodel.fit(y, x=x, amp=1, cen=best_contact, wid=1)

    #            print(result.fit_report())

                if result.chisqr < 0.13:

                    if neuro.shank_ids[unit_index] == 0:

                        if neuro.cell_type[unit_index] == 'RS':
                            m1_rs.append(fwhm(result))

                        elif neuro.cell_type[unit_index] == 'FS':
                            m1_fs.append(fwhm(result))
                    elif neuro.shank_ids[unit_index] == 1:

                        if neuro.cell_type[unit_index] == 'RS':
                            s1_rs.append(fwhm(result))

                        elif neuro.cell_type[unit_index] == 'FS':
                            s1_fs.append(fwhm(result))


bins = np.arange(0, 100)

hist(m1_rs, bins=bins, color='k', alpha=0.5)
hist(s1_rs, bins=bins, color='r', alpha=0.5)































#                plt.figure()
#                plt.plot(x, y, 'bo--', linewidth=0.5)
#            #    plt.plot(x, result.init_fit, 'k--')
#                plt.plot(x, result.best_fit, 'r-')
#                plt.title('chisqr: {} | region: {}'.format(result.chisqr, neuro.shank_ids[unit_index]))
#                plt.show()
