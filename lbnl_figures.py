



def compare_units(neuro, units=[0, 1], stims=[0, 1]):
    fig, ax = plt.subplots(2, 2)

    # plot first unit in the two conditions
    neuro.plot_psth(axis=ax[0], unit_ind=units[0], trial_type=stims[0],...
            error='sem', color='k')
    neuro.plot_psth(axis=ax[1], unit_ind=units[0], trial_type=stims[1],...
            error='sem', color='k')

    # plot second unit in the two conditions
    neuro.plot_psth(axis=ax[0], unit_ind=units[1], trial_type=stims[0],...
            error='sem', color='g')
    neuro.plot_psth(axis=ax[1], unit_ind=units[1], trial_type=stims[1],...
            error='sem', color='g')

    return fig, ax
