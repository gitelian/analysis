from matplotlib.backends.backend_pdf import PdfPages

# change default figure type to PDF
mpl.rcParams['savefig.format'] = 'pdf'
plt.rc('font',family='Arial')
sns.set_style("whitegrid", {'axes.grid' : False})


#get_ipython().magic(u"run hdfanalyzer.py {}".format('1937'))
get_ipython().magic(u"run hdfanalyzer.py {}".format('1966'))
neuro.rates(all_trials=True)
neuro.get_num_good_trials()


def plot_tuning_curve(unit_ind):
    fig, ax = plt.subplots()
    neuro.plot_tuning_curve(unit_ind=unit_ind, axis=ax)
    xlabel_pos = np.arange(1, neuro.control_pos+1)
    xpos = np.arange(0, 4, 0.5)
    #led_legend = ['0.78mW', '1.69mW', '2.51mW', '3.27mW', '3.99mW', '4.66mW']
    led_power = [0.100000000000000,
            0.168586277878877,
            0.284213330890540,
            0.479144675783939,
            0.807772174558958,
            1.36179304283021,
            2.29579620332096,
            3.87039736686339,
            6.52495885991707,
            11.0001852750622]
    led_power_label = np.round(led_power, decimals=3)
    ax.set_xticks(xlabel_pos)
    ax.set_xticklabels(xpos)
    ax.set_xlabel('distance from recording site (mm)')
    ax.set_ylabel('firing rate (Hz)')

    # get x and y data and set to correct x data?

    # just update the x tick labels???

    # just plot tuning curves from sratch here???


    ## This is stupid an will likely break ##
    lines = ax.lines[2:11:2]
    ax.legend(lines, led_legend)

    return fig


with PdfPages(fid + '_RS_dose_response.pdf') as pdf:

    for unit_ind, cell_type in enumerate(neuro.cell_type):

        if cell_type == 'RS':
            plot_tuning_curve(unit_ind)

            pdf.savefig()
            fig.clear()
            plt.close()


uind = 12
fig, ax = plt.subplots()
neuro.plot_tuning_curve(unit_ind=uind, axis=ax)
xlabel_pos = np.arange(1, neuro.control_pos+1)
xlabel = np.arange(0, 4, 0.5)

### write out the equation that will let me calculate the actual output power ###
led_power = [0.100000000000000,
        0.168586277878877,
        0.284213330890540,
        0.479144675783939,
        0.807772174558958,
        1.36179304283021,
        2.29579620332096,
        3.87039736686339,
        6.52495885991707,
        11.0001852750622]
led_power_label = np.round(led_power, decimals=3)
ax.set_xticks(xlabel_pos)
ax.set_xticklabels(xlabel)
ax.set_xlabel('Distance from electrode (mm)')
ax.set_ylabel('Firing rate (Hz)')

## This is stupid an will likely break ##
lines = ax.lines[0::2]
ax.legend(lines, led_power_label)
ax.set_title('LED dose response curve for {} unit: {}'.format(neuro.cell_type[uind], uind))

