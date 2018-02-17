import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import icsd
import quantities as pq

# import and start MATLAB
import matlab.engine
eng = matlab.engine.start_matlab()

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
shank = 0
contact = 12
pos = 8

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

##### compute PSD using Chronux #####
##### compute PSD using Chronux #####

# 1302 (no wt), 1326 (no hdf file), 1328, 1330 (linear probe, great experiments)
fids = ['1336', '1338', '1339', '1340', '1343', '1345']

neuro.get_lfps()
lfps = neuro.lfps
shank = 0
contact = 24
pos = 5
stim_inds = np.logical_and(neuro.lfp_t > 0.6, neuro.lfp_t < 1.4)

# get data
lfp_nolight = lfps[shank][pos][stim_inds,     contact, :]
lfp_s1light = lfps[shank][pos+9][stim_inds,   contact, :]
lfp_m1light = lfps[shank][pos+9+9][stim_inds, contact, :]

# format data to MATLAB friendly form
lfp_nolight = matlab.double(lfp_nolight.tolist())
lfp_s1light = matlab.double(lfp_s1light.tolist())
lfp_m1light = matlab.double(lfp_m1light.tolist())

# calculate PSD
S_nolight, f_nolight, Serr_nolight = eng.lfp_psd(lfp_nolight, nargout=3)
S_s1light, f_s1light, Serr_s1light = eng.lfp_psd(lfp_s1light, nargout=3)
S_m1light, f_m1light, Serr_m1light = eng.lfp_psd(lfp_m1light, nargout=3)

# get data back into numpy friendly form
S_nolight = np.squeeze(np.array(S_nolight))
S_s1light = np.squeeze(np.array(S_s1light))
S_m1light = np.squeeze(np.array(S_m1light))

f    = np.squeeze(np.array(f_nolight))

Serr_nolight = np.squeeze(np.array(Serr_nolight))
Serr_s1light = np.squeeze(np.array(Serr_s1light))
Serr_m1light = np.squeeze(np.array(Serr_m1light))

plt.figure()
plt.semilogy(f, S_nolight, 'k', f, S_s1light, 'r', linewidth=1)
plt.fill_between(f, Serr_nolight[0, :], Serr_nolight[1, :], facecolor='k', alpha=0.3)
plt.fill_between(f, Serr_s1light[0, :], Serr_s1light[1, :], facecolor='r', alpha=0.3)
plt.xlim(0, 125)
title('contact: {}, pos: {}'.format(contact, pos))

plt.figure()
plt.semilogy(f, S_nolight, 'k', f, S_m1light, 'b', linewidth=1)
plt.xlim(0, 125)


##### plot power at specified frequency for all positions and contacts #####
##### plot power at specified frequency for all positions and contacts #####

### make a heat map where the difference between light and nolight power is plotted
### each row is a contact and each column is a position

# 1302 (no wt), 1326 (no hdf file), 1328, 1330 (linear probe, great experiments)
fids = ['1336', '1338', '1339', '1340', '1343', '1345']

neuro.get_lfps()
lfps = neuro.lfps
stim_inds = np.logical_and(neuro.lfp_t > 0.6, neuro.lfp_t < 1.4)
shank = 0

all_S_nolight = np.zeros((32, 8))
all_S_s1light = np.zeros((32, 8))
power_diff = np.zeros((32, 8))


for contact_i in range(32): # for linear probe
#for contact_i in range(16): # for poly2 probe
    print('working on contact {}'.format(contact_i))
    for pos_k in range(8):

        # get data
        lfp_nolight = lfps[shank][pos_k][stim_inds,     contact_i, :]
        lfp_s1light = lfps[shank][pos_k+9][stim_inds,   contact_i, :]

        # format data to MATLAB friendly form
        lfp_nolight = matlab.double(lfp_nolight.tolist())
        lfp_s1light = matlab.double(lfp_s1light.tolist())

        # calculate PSD
        S_nolight, f_nolight, Serr_nolight = eng.lfp_psd(lfp_nolight, nargout=3)
        S_s1light, f_s1light, Serr_s1light = eng.lfp_psd(lfp_s1light, nargout=3)

        # get data back into numpy friendly form
        S_nolight = np.squeeze(np.array(S_nolight))
        S_s1light = np.squeeze(np.array(S_s1light))

        f = np.squeeze(np.array(f_nolight))

        Serr_nolight = np.squeeze(np.array(Serr_nolight))
        Serr_s1light = np.squeeze(np.array(Serr_s1light))

        # find index closest to 35Hz
        # TODO average the values between a specified frequency range
        f_ind = np.argmin(np.abs(f - 35.0))

        # compute difference in power at f_ind
        diff_temp = S_s1light[f_ind] - S_nolight[f_ind]

        # save values
        power_diff[contact_i, pos_k] = diff_temp
        all_S_nolight[contact_i, pos_k] = S_nolight[f_ind]
        all_S_s1light[contact_i, pos_k] = S_s1light[f_ind]

# plot power_diff array as a heat map
min_temp = np.min(power_diff)
max_temp = np.max(power_diff)
max_val_diff  = np.max(np.abs(np.asarray([min_temp, max_temp])))

max_abs = np.max(np.asarray([all_S_nolight, all_S_s1light]))

num_chan = neuro.chan_per_shank[shank]
edist = 25.0 # microns
chan_depth = np.arange(np.asarray(neuro.shank_depths[shank]) - num_chan * edist, np.asarray(neuro.shank_depths[shank]), edist)

fig, ax = plt.subplots(1, 3)
im = ax[0].imshow(all_S_nolight, vmin=0, vmax=max_abs, interpolation='none',\
        origin='lower', aspect='auto',\
        extent=(0, 8, chan_depth[-1], chan_depth[0]))
cb = plt.colorbar(im, ax=ax[0])
im = ax[1].imshow(all_S_s1light, vmin=0, vmax=max_abs, interpolation='none',\
        origin='lower', aspect='auto',\
        extent=(0, 8, chan_depth[-1], chan_depth[0]))
cb = plt.colorbar(im, ax=ax[1])
im = ax[2].imshow(power_diff, vmin=-max_val_diff, vmax=max_val_diff, interpolation='none',\
        origin='lower', cmap='coolwarm',  aspect='auto',\
        extent=(0, 8, chan_depth[-1], chan_depth[0]))
cb = plt.colorbar(im, ax=ax[2])

ax[0].set_title('no light')
ax[1].set_title('s1 light')
ax[2].set_title('difference')
ax[0].set_ylabel('depth (um)')
ax[0].set_xlabel('position')
ax[1].set_xlabel('position')
ax[2].set_xlabel('position')
fig.suptitle(fid)

##### compute coherence between S1 and M1 using SCIPY #####
##### compute coherence between S1 and M1 using SCIPY #####

neuro.get_lfps()
lfps = neuro.lfps
m1contact = 11
s1contact = 8
pos = 7
stim_inds = np.logical_and(neuro.lfp_t > 0.6, neuro.lfp_t < 1.4)
t = neuro.lfp_t[stim_inds]

# get LFPs
s1_contact = lfps[0][pos][stim_inds, s1contact, :]
m1_contact = lfps[1][pos][stim_inds, m1contact, :]

num_trials = s1_contact.shape[1]

for k in range(num_trials):
    if k == 0:
        f, Cxy_temp = sp.signal.coherence(s1_contact[:, k], m1_contact[:, k], fs=1500)
        num_samples = Cxy_temp.shape[0]
        Cxy_mean = np.zeros((num_samples, num_trials))
        Cxy_mean[:, k] = Cxy_temp
    else:
        f, Cxy_temp = sp.signal.coherence(s1_contact[:, k], m1_contact[:, k], fs=1500)
        Cxy_mean[:, k] = Cxy_temp
figure()
plot(f, Cxy_mean.mean(axis=1))
title('pos {}'.format(pos))
plt.xlim(0, 150)
plt.ylim(0, 0.5)


# create arrays and save them as mat files in order to measure coherence in
# matlab

# position 3
s1_pos3 = lfps[0][3][stim_inds, s1contact, :]
m1_pos3 = lfps[1][3][stim_inds, m1contact, :]

# position 5
s1_pos5 = lfps[0][5][stim_inds, s1contact, :]
m1_pos5 = lfps[1][5][stim_inds, m1contact, :]

# position 7
s1_pos7 = lfps[0][7][stim_inds, s1contact, :]
m1_pos7 = lfps[1][7][stim_inds, m1contact, :]

# position 8
s1_pos8 = lfps[0][8][stim_inds, s1contact, :]
m1_pos8 = lfps[1][8][stim_inds, m1contact, :]

# save variables

sp.io.savemat('/home/greg/Desktop/' + fid + '_lfps.mat', {'s1_pos3':s1_pos3,\
        'm1_pos3':m1_pos3,
        's1_pos5':s1_pos5,
        'm1_pos5':m1_pos5,
        's1_pos7':s1_pos7,
        'm1_pos7':m1_pos7,
        's1_pos8':s1_pos8,
        'm1_pos8':m1_pos8})

#### MATLAB scratch code #####
#### MATLAB scratch code #####

# MATLAB engine expects doubles. You can only get double by converting a numpy
# array to a list and then converting to double
x_np = np.arange(10)
y_np = x_np**2
x = matlab.double(x_np.tolist())
y = matlab.double(y_np.tolist())
eng.plot(x,y)


##### compute coherence between S1 and M1 using MATLAB and Chronux #####
##### compute coherence between S1 and M1 using MATLAB and Chronux #####

neuro.get_lfps()
lfps = neuro.lfps
m1contact = 12
s1contact = 10
pos = 8+9
stim_inds = np.logical_and(neuro.lfp_t > 0.6, neuro.lfp_t < 1.4)
t = neuro.lfp_t[stim_inds]

# get LFPs
s1_contact = lfps[0][pos][stim_inds, s1contact, :]
m1_contact = lfps[1][pos][stim_inds, m1contact, :]

num_trials = s1_contact.shape[1]

# convert to doubles
x = matlab.double(s1_contact.tolist())
y = matlab.double(m1_contact.tolist())

# calculate coherence
Cxy, f, Cerr = eng.lfp_coherence(x, y, nargout=3)

# convert back to numpy array
Cxy  = np.squeeze(np.array(Cxy))
f    = np.squeeze(np.array(f))
Cerr = np.squeeze(np.array(Cerr))

# plot coherence plot
plt.plot(f, Cxy, linewidth=1)
plt.xlim(0, 125)
plt.plot(f, Cerr[0, :], 'r', f, Cerr[1, :],'r', linewidth=0.5)

# plot coherence plot
plt.plot(f, Cxy, linewidth=1)













##### LFP + Whisker tracking #####
##### LFP + Whisker tracking #####
region   = 1
stim_ind = 8
chan     = 4
scale    = 50
trial    = 1

fig, ax = plt.subplots(2, 1)
ax[0].plot(neuro.lfp_t, lfps[region][stim_ind][:, trial, chan],\
        neuro.wtt, (neuro.wt[stim_ind][:, 0, trial]-neuro.wt[stim_ind][:, 1, trial])*scale)

ax[1].plot(neuro.lfp_t, lfps[region][stim_ind+9][:, trial, chan],\
        neuro.wtt, (neuro.wt[stim_ind+9][:, 0, trial]-neuro.wt[stim_ind+9][:, 1, trial])*scale)


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
shank = 0
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







