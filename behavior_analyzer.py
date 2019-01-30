#!/bin/bash
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
import h5py

#import 3rd party code found on github
#import icsd
import ranksurprise
import dunn
from IPython.core.debugger import Tracer





# plot mean set-point, indicate light on
# TODO: overlay object x-position

fig, ax = neuro.plot_mean_setpoint(t_window=[-1.5, 1.5], cond2plot=[0, 3, 4, 7, 8], all_trials=True)

fig.suptitle(fid + ' mean set-point ALL TRIALS')
ylim = ax[0].get_ylim()
for a in ax:
    a.vlines([-1, 1], ylim[0], ylim[1], colors='c')
    a.set_ylim(ylim)
    a.axvspan(-1, 0, alpha=0.2, color='green')

