import matplotlib.pyplot as plt
def cu_test():
    fig, ax = plt.subplots()

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
