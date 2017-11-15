import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import icsd
import quantities as pq

def iCSD(lfp_data):
    #patch quantities with the SI unit Siemens if it does not exist
    for symbol, prefix, definition, u_symbol in zip(
        ['siemens', 'S', 'mS', 'uS', 'nS', 'pS'],
        ['', '', 'milli', 'micro', 'nano', 'pico'],
        [pq.A/pq.V, pq.A/pq.V, 'S', 'mS', 'uS', 'nS'],
        [None, None, None, None, u'uS', None]):
        if type(definition) is str:
            definition = lastdefinition / 1000
        if not hasattr(pq, symbol):
            setattr(pq, symbol, pq.UnitQuantity(
                prefix + 'siemens',
                definition,
                symbol=symbol,
                u_symbol=u_symbol))
        lastdefinition = definition

    #prepare lfp data for use, by changing the units to SI and append quantities,
    #along with electrode geometry, conductivities and assumed source geometry

    lfp_data = lfp_data * 1E-6 * pq.V        # [uV] -> [V]
    #z_data = np.linspace(100E-6, 2300E-6, 23) * pq.m  # [m]
    z_data = np.linspace(100E-6, 1000E-6, 32) * pq.m  # [m]
    #diam = 500E-6 * pq.m                              # [m]
    diam = 250E-6 * pq.m                              # [m] bigger vals make smaller sources/sinks
    h = 100E-6 * pq.m                                 # [m]  (makes no difference with spline iCSD method)
    sigma = 0.3 * pq.S / pq.m                         # [S/m] or [1/(ohm*m)] (makes no difference with spline iCSD method)
    sigma_top = 0.3 * pq.S / pq.m                     # [S/m] or [1/(ohm*m)]

    # Input dictionaries for each method
    spline_input = {
        'lfp' : lfp_data,
        'coord_electrode' : z_data,
        'diam' : diam,
        'sigma' : sigma,
        'sigma_top' : sigma,
        'num_steps' : 201,      # Spatial CSD upsampling to N steps
        'tol' : 1E-12,
        'f_type' : 'gaussian',
        'f_order' : (20, 5),}

    # how to call!
    csd_obj = icsd.SplineiCSD(**spline_input)
    csd = csd_obj.get_csd()
    csd = csd_obj.filter_csd(csd)
    return csd

##### start analysis #####
##### start analysis #####

sns.set_style("whitegrid", {'axes.grid' : False})
get_ipython().magic(u"run neoanalyzer.py {}".format(sys.argv[1]))

##### LFP analysis #####
##### LFP analysis #####

neuro.get_lfps()
lfps = neuro.lfps
shank = 1
contact = 10
pos = 3

stim_inds = np.logical_and(neuro.lfp_t > 0.6, neuro.lfp_t < 1.4)
lfp_nolight = lfps[shank][pos][stim_inds,     contact, :]
lfp_s1light = lfps[shank][pos+9][stim_inds,   contact, :]
lfp_m1light = lfps[shank][pos+9+9][stim_inds, contact, :]

f, frq_nolight = neuro.get_psd(lfp_nolight, 1500.)
f, frq_s1light = neuro.get_psd(lfp_s1light, 1500.)
f, frq_m1light = neuro.get_psd(lfp_m1light, 1500.)
plt.figure()
neuro.plot_freq(f, frq_nolight, color='k')
neuro.plot_freq(f, frq_s1light, color='r')
neuro.plot_freq(f, frq_m1light, color='b')
plt.xlim(0, 200)
plt.legend(('no light', 's1 light', 'm1 light'))
plt.title('{} PSD'.format(neuro.region_dict[shank]))

plt.show()

# mean LFP during control position
m1_nolight = np.mean(lfps[0][8][:, 15, :], axis=1)
m1_s1light = np.mean(lfps[0][8+9][:, 15, :], axis=1)
plt.figure()
plt.plot(neuro.lfp_t, m1_nolight, 'k', neuro.lfp_t, m1_s1light, 'r')

# mean LFP during control position
s1_nolight = np.mean(lfps[1][4][:, 4, :], axis=1)
s1_m1light = np.mean(lfps[1][4+9+9][:, 4, :], axis=1)
plt.figure()
plt.plot(neuro.lfp_t, s1_nolight, 'k', neuro.lfp_t, s1_m1light, 'b')


##### iCSD analysis #####
##### iCSD analysis #####

shank = 0
lfps_mat = lfps[shank][3]
num_chan = neuro.chan_per_shank[shank]
edist = 25.0 # microns
chan_depth = np.arange(np.asarray(neuro.shank_depths[shank]) - num_chan * edist, np.asarray(neuro.shank_depths[shank]), edist)

# compute iCSD for first condition
shank = 0
pos = 5
lfps_mat = lfps[shank][pos]
num_chan = neuro.chan_per_shank[shank]
edist = 25.0 # microns
chan_depth = np.arange(np.asarray(neuro.shank_depths[shank]) - num_chan * edist, np.asarray(neuro.shank_depths[shank]), edist)
for k in range(lfps_mat.shape[2]):
    csd_temp = iCSD(lfps_mat[:, :, k].T)
    if k == 0:
        csd0 = np.zeros((csd_temp.shape[0], csd_temp.shape[1], lfps_mat.shape[2]))
    csd0[:, :, k] = csd_temp

# compute iCSD for second condition
lfps_mat = lfps[shank][pos+9]
num_chan = neuro.chan_per_shank[shank]
edist = 25.0 # microns
chan_depth = np.arange(np.asarray(neuro.shank_depths[shank]) - num_chan * edist, np.asarray(neuro.shank_depths[shank]), edist)
for k in range(lfps_mat.shape[2]):
    csd_temp = iCSD(lfps_mat[:, :, k].T)
    if k == 0:
        csd1 = np.zeros((csd_temp.shape[0], csd_temp.shape[1], lfps_mat.shape[2]))
    csd1[:, :, k] = csd_temp

#plot iCSD signal smoothed
scale = 0.90
fig, axes = plt.subplots(2,1, figsize=(8,8))
ax = axes[0]
im = ax.imshow(np.array(csd0.mean(axis=2)), origin='lower', vmin=-abs(csd0.mean(axis=2)).max()*scale, \
        vmax=abs(csd0.mean(axis=2)).max()*scale, cmap='jet_r', interpolation='nearest', \
        extent=(neuro.lfp_t[0], neuro.lfp_t[-1], chan_depth[-1], chan_depth[0]))
ax.axis(ax.axis('tight'))
cb = plt.colorbar(im, ax=ax)
ax.set_ylabel('theoretical depth')
ax.set_title('region: {0}, position {1}\nno light'.format(neuro.region_dict[shank], pos))
ax.set_xlim(-0.1, 1.0)
#ax.set_xlim(-0.1, 0.250)

#plot iCSD signal smoothed
ax = axes[1]
im = ax.imshow(np.array(csd1.mean(axis=2)), origin='lower', vmin=-abs(csd1.mean(axis=2)).max()*scale, \
        vmax=abs(csd1.mean(axis=2)).max()*scale, cmap='jet_r', interpolation='nearest', \
        extent=(neuro.lfp_t[0], neuro.lfp_t[-1], chan_depth[-1], chan_depth[0]))
ax.axis(ax.axis('tight'))
ax.axis(sharex=axes[0])
ax.set_title('light')
ax.axis(sharex=True)
cb = plt.colorbar(im, ax=ax)
ax.set_xlabel('time (s)')
ax.set_ylabel('theoretical depth')
ax.set_xlim(-0.1, 1.0)
#ax.set_xlim(-0.1, 0.250)


##### calculate all positions
shank = 1
num_chan = neuro.chan_per_shank[shank]
edist = 25.0 # microns
chan_depth = np.arange(np.asarray(neuro.shank_depths[shank]) - num_chan * edist, np.asarray(neuro.shank_depths[shank]), edist)
csd = list()
for pos in range(9):
    lfps_mat = lfps[shank][pos+9+9]
    print('\n\n##### WORKING ON CONDITION: {} #####'.format(pos))
    for k in range(lfps_mat.shape[2]):
        csd_temp = iCSD(lfps_mat[:, :, k].T)
        if k == 0:
            csd0 = np.zeros((csd_temp.shape[0], csd_temp.shape[1], lfps_mat.shape[2]))
        csd0[:, :, k] = csd_temp
    csd.append(csd0)


##### plot all positions
scale = 0.90
fig, axes = plt.subplots(9,1, figsize=(8,8))
for pos in range(9):
    ax = axes[pos]
#    im = ax.imshow(np.array(csd[pos].mean(axis=2)), origin='lower', vmin=-abs(csd[pos].mean(axis=2)).max()*scale, \
#            vmax=abs(csd[pos].mean(axis=2)).max()*scale, cmap='jet_r', interpolation='nearest', \
#            extent=(neuro.lfp_t[0], neuro.lfp_t[-1], chan_depth[-1], chan_depth[0]))
    im = ax.imshow(np.array(csd[pos].mean(axis=2)), origin='lower', vmin=-35000, \
            vmax=35000, cmap='jet_r', interpolation='nearest', \
            extent=(neuro.lfp_t[0], neuro.lfp_t[-1], chan_depth[-1], chan_depth[0]))
    ax.axis(ax.axis('tight'))
    cb = plt.colorbar(im, ax=ax)
    ax.set_ylabel('theoretical depth')
    ax.set_title('position {}'.format(pos))
    ax.set_xlim(-0.1, 1.0)







