#!/bin/bash
from neuroanalyzer import *

if __name__ == "__main__":
    regions = ['vM1', 'vS1']
#    regions = [ 'vM1']
    for region in regions:
        usr_dir = os.path.expanduser('~')
        data_dir_path = usr_dir + '/Documents/AdesnikLab/Data/'
        sorted_spikes_dir_path = usr_dir + '/Documents/AdesnikLab/SortedSpikes/'
        # # Update spike measure csv files
        spike_measures2csv(data_dir_path,sorted_spikes_dir_path,region,overwrite=False)
        classify_units(data_dir_path,region)
