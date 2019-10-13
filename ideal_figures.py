import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2 # default: 1.5
mpl.rcParams['axes.titlesize'] = '24'
mpl.rcParams['axes.labelsize'] = '20'
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['axes.edgecolor'] = '.1'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


        # 8-bar-position experiments
        # good neural data and good whisker tracking
        # fids = ['1336', '1338', '1339', '1340', '1343', '1345']


#### plot example whisker plots where there is no change due to silencing ####
#### plot example whisker plots where there is no change due to silencing ####

# fid1345
# control pos
# 2, (9, 8)
# s1 silencing
# 1, (4)
# m1 silencing
# 9, (5, 6,...0-10, 15, 22, 36)

# control + NoLight
fig, ax = plt.subplots()
###
ax.plot(neuro.wtt, neuro.wt[8][:, 0, 2], color='dimgray')
#ax.plot(neuro.wtt, neuro.wt[8+9][:, 0, 1], color='tab:red')
#ax.plot(neuro.wtt, neuro.wt[8+9+9][:, 0, 9], color='tab:blue')
###
ax.set_ylim(90, 150)
ax.set_xlim(0.4, 1.6)
###
ax.set_ylabel('Whisker angle (deg)')
#ax.set_ylabel('Set-point (deg)')
##
ax.set_xlabel('Time (s)')
ax.set_xticks([0.5, 1.0, 1.5])

## make quick plots for initial quick looking ##
#cond = 8+9+9
#for trial in range(neuro.num_good_trials[cond]):
#    fig = plt.figure()
#    plt.title(trial)
#    plt.plot(neuro.wtt, neuro.wt[cond][:, 0, trial])

#####--------------------------------------------------------------------#####



        # JB_behavior Notes
        ##### GT015_LT #####
        # GT015_LT vM1 silencing with GOOD behavior but NO WHISKER TRACKING

        # THERE was tracking recorded...but I couldn't get it to work...maybe there is
        # something I can do now to recover it??
        # fids = ['1855', '1874', '1882', '1891', '1892']
        #experiments that may be good: 1861

        ##### Gt015_LT vS1 silencing + tracking #####
        # fids = ['1895', '1898', '1904']

        ##### GT017_NT vM1 silencing + tracking #####
        # fids = ['1911', '1912', '1913', '1923', '1924', '1929']
        # experiments that may be good: [1896, 1916, [1919 and 1920 are of same day...dropped frames]


#### JB_BEHAVIOR PLOTTING SECTION ####
#### JB_BEHAVIOR PLOTTING SECTION ####


#### whisker angle and setpoint plots ####
## vM1 silencing + whisking
## FID1911 vM1 silencing, FID1895 vS1 silencing
# control + NoLight
# vM1 light
## FID1911 vM1 silencing
# trial index in control position (no contact) used to make figures
# 3, 9 huge decrease with STEADY quick running, (5,7,8, set-point decreases while runspeed increases!!!)

# vS1 light
## FID1895 and 1898 with vS1 silencing
# trial index in control position (no contact) used to make figures
# (5, 2), (9, 9) (setpoint light on, vs light off) very good examples
# fid1895(0, 0), fid1898(0, 0), fid1898(3 , 2) (vS1 silencing plots)

cond=0
fig, ax = plt.subplots()
###
ax.plot(neuro.wtt, neuro.wt[8][:, 1, 0], color='dimgray')
ax.plot(neuro.wtt, neuro.wt[8+9][:, 1, cond], color='tab:red')
#ax.plot(neuro.wtt, neuro.wt[8+9][:, 0, cond], color='tab:blue')
#ax.plot(neuro.wtt, mean(neuro.wt[8+9][:, 1, :], axis=1), color='tab:gray')
###
#ax.set_ylim(90, 160) # angle limits
ax.set_ylim(100, 135) # setpoint limits
ax.set_xlim(-1.2, 0.2)
#ax.hlines(160, -1, 0, color='tab:blue', linewidth=5)
ax.hlines(160, -1, 0, color='tab:red', linewidth=5)
###
#ax.set_ylabel('Whisker angle (deg)')
ax.set_ylabel('Set-point (deg)')
##
ax.set_xlabel('Time (s)')
#ax.set_xticks([0.5, 1.0, 1.5])


#### vS1 silencing psychometric curve ####
#### FID1895 ####
neuro.get_psychometric_curve()


a1 = neuro.wt[0][:, 0, :]
a2 = neuro.wt[1][:, 0, :]
a3 = neuro.wt[8][:, 0, :]
a = np.concatenate( (a1, a2, a3), axis=1)
af, afmat = neuro.get_psd(a, 500)

b1 = neuro.wt[0+9][:, 0, :]
b2 = neuro.wt[1+9][:, 0, :]
b3 = neuro.wt[8+9][:, 0, :]
b = np.concatenate( (b1, b2, b3), axis=1)
bf, bfmat = neuro.get_psd(b, 500)

neuro.plot_freq(af, afmat, color='dimgray', error='ci')
neuro.plot_freq(bf, bfmat, color='tab:blue', error='ci')







