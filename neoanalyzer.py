#!/bin/bash
import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from neo.io import NeoHdf5IO

class NeuroAnalyzer(object):
    """
    Analyzes data contained in a neo object

    REMEMBER: any changes to the passed in neo object also occur to the original
    object. They point to the same place in memory. No copy is made.
    """

    def __init__(self, neo_obj):

        # segments can get out of the order they were created
        sorted_index_list = np.argsort([k.index for k in neo_obj.segments])
        neo_obj.segments  = [neo_obj.segments[k] for k in sorted_index_list]

        # sort by shank/region and then by depth
        sorted_index_list = self.__sort_units(neo_obj)

        self.neo_obj        = neo_obj
        self.stim_ids       = np.sort(np.unique([k.annotations['trial_type'] for k in neo_obj.segments]))
        self.num_run_trials = self.__get_num_run_trials()
        self.num_units      = len(self.neo_obj.segments[0].spiketrains)
        self.shank_names    = np.sort(np.unique([k.annotations['shank'] for k in neo_obj.segments[0].spiketrains]))
        self.shank_ids      = self.__get_shank_ids()


    def __sort_units(self, neo_obj):
        '''
        Units are saved out of order in the neo object.
        This reorders units by experiment FID, shank, and depth across all segments.
        '''

        for i, seg in enumerate(neo_obj.segments):
            fid, shank, depth = list(), list(), list()

            for spike in seg.spiketrains:
                fid.append(spike.annotations['fid'])
                shank.append(spike.annotations['shank'])
                # would freak out if you tried to index into an array with an
                # nan because it was not an array
                if np.isnan(spike.annotations['depth']):
                    depth.append(np.asarray(-1))
                else:
                    depth.append(spike.annotations['depth'][0])

            spike_msr_sort_inds = np.lexsort((tuple(depth), tuple(shank), tuple(fid)))
            sorted_spiketrains = [seg.spiketrains[j] for j in spike_msr_sort_inds]
            for k, ordered_spiketrain in enumerate(sorted_spiketrains):
                neo_obj.segments[i].spiketrains[k] = ordered_spiketrain

    def __get_shank_ids(self):
        '''
        Return shank IDs for each unit
        '''
        shank_ids = np.zeros((self.num_units, ))
        for k, shank_name in enumerate(self.shank_names):
            for j, unit in enumerate(self.neo_obj.segments[0].spiketrains):
                if unit.annotations['shank'] == shank_name:
                    shank_ids[j] = k
        return shank_ids

    def __get_num_run_trials(self):
        '''
        Return a list with the number of running trials for each stimulus condition
        '''
        num_run_trials = list()
        for stim_id in self.stim_ids:
            run_count = 0
            for trial in self.neo_obj.segments:
                if trial.annotations['trial_type'] == stim_id and trial.annotations['run_boolean']:
                    run_count += 1
            num_run_trials.append(run_count)
        return num_run_trials

    def get_annotations_index(self, key, value):
        '''Returns trial index for the given key value pair'''
        stim_index = [ index for index, segment in enumerate(self.neo_obj.segments) \
                if segment.annotations[key] == value]
        return stim_index

    def rates(self):

        t_after_stim = 0.500 # change this from being hardcoded!!!
        absolute_rate    = list()
        evoked_rate      = list()
        absolute_counts  = list()
        evoked_counts    = list()
        binned_spikes    = list()
        psth             = list()
        t_before = 0.500
        bins = np.arange(-t_before, 2.5, 0.001)
        alpha_kernel = self.__make_alpha_kernel()
        self._bins = bins

        for k, trials_ran in enumerate(self.num_run_trials):
                absolute_rate.append(np.zeros((trials_ran, self.num_units)))
                evoked_rate.append(np.zeros((trials_ran, self.num_units)))
                absolute_counts.append(np.zeros((trials_ran, self.num_units)))
                evoked_counts.append(np.zeros((trials_ran, self.num_units)))
                binned_spikes.append(np.zeros((bins.shape[0]-1, self.num_units, trials_ran)))
                psth.append(np.zeros((bins.shape[0]-1, self.num_units, trials_ran)))

        for stim_ind, stim_id in enumerate(self.stim_ids):
            good_trial_ind = 0

            for trial in self.neo_obj.segments:
                if trial.annotations['trial_type'] == stim_id and trial.annotations['run_boolean']:

                    stim_start = trial.annotations['stim_times'][0] + t_after_stim
                    stim_stop  = trial.annotations['stim_times'][1]
                    base_start = trial.annotations['stim_times'][0] - (stim_stop - stim_start)
                    base_stop  = trial.annotations['stim_times'][0]

                    for unit, spike_train in enumerate(trial.spiketrains):
                        spk_times = np.asarray(spike_train.tolist())

                        # bin spikes for rasters
                        spk_times_relative = spk_times - trial.annotations['stim_times'][0]
                        counts = np.histogram(spk_times_relative, bins=bins)[0]
                        binned_spikes[stim_ind][:, unit, good_trial_ind] = counts

                        # convolve binned spikes to make PSTH
                        psth[stim_ind][:, unit, good_trial_ind] =\
                                np.convolve(counts, alpha_kernel)[:-alpha_kernel.shape[0]+1]

                        # calculate absolute and evoked rate
                        abs_rate = (np.logical_and(spk_times > stim_start, spk_times < stim_stop).sum())/(stim_stop - stim_start)
                        evk_rate = (np.logical_and(spk_times > base_start, spk_times < base_stop).sum())/(stim_stop - stim_start)
                        absolute_rate[stim_ind][good_trial_ind, unit] = abs_rate
                        evoked_rate[stim_ind][good_trial_ind, unit]   = evk_rate

                        # calculate absolute and evoked counts
                        abs_count = np.logical_and(spk_times > stim_start, spk_times < stim_stop).sum()
                        evk_count   = (np.logical_and(spk_times > stim_start, spk_times < stim_stop).sum()) - \
                                (np.logical_and(spk_times > base_start, spk_times < base_stop).sum())
                        absolute_counts[stim_ind][good_trial_ind, unit] = abs_count
                        evoked_counts[stim_ind][good_trial_ind, unit]   = evk_count

                    good_trial_ind += 1

        self.abs_rate      = absolute_rate
        self.abs_count     = absolute_counts
        self.evk_rate      = evoked_rate
        self.evk_count     = evoked_counts
        self.binned_spikes = binned_spikes
        self.psth          = psth

    def __make_alpha_kernel(self, resolution=0.025):
        # Build alpha kernel with 25msec resolution
        alpha = 1.0/resolution
        tau   = np.arange(0,1/alpha*10, 0.001)
        alpha_kernel = alpha**2*tau*np.exp(-alpha*tau)
        return alpha_kernel

    def get_burst_isi(self):
        '''
        Compute the interspike interval for spikes during the stimulus period.
        get_burst_isi creates a list that has n_stimulus_types entries. Each
        stimulus type has a list which contains a numpy array for each unit.

        These values can be used to identify bursting activity.
        '''

        t_after_stim = 0.500 # change this from being hardcoded!!!
        bisi_list = [list() for stim in self.stim_ids]

        # iterate through all stimulus IDs
        for stim_ind, stim_id in enumerate(self.stim_ids):
            good_trial_ind = 0
            # this is a list of numpy arrays. There is an array for each unit
            bisi_np_list = [np.empty((1, 4)) for k in range(self.num_units)]

            # iterate through all neo trial segments
            for trial in self.neo_obj.segments:

                # if a trial segment is from the current stim_id and it is a
                # running trial get the spike ISIs.
                if trial.annotations['trial_type'] == stim_id and trial.annotations['run_boolean']:

                    # get the stimulus start and stop times for this trial
                    stim_start = trial.annotations['stim_times'][0]
                    stim_stop  = trial.annotations['stim_times'][1]

                    # iterate through all unit's spike trains. find times units fired
                    # during the stimulus period and calculate the inter-spike-
                    # interval between them. This will be used to determine
                    # bursting
                    for unit, spike_train in enumerate(trial.spiketrains):

                        # get spike tiimes during stimulus period
                        spk_times_all = np.asarray(spike_train.tolist()) # spike times for current spike train
                        spk_times     = spk_times_all[np.logical_and(spk_times_all > stim_start, spk_times_all < stim_stop)] # spike times during stimulus period
                        num_spk_times = len(spk_times)
                        if num_spk_times > 3:
                            # create the burst isi array (bisi) if there are  more
                            # than 3 spikes all spikes except the first and last
                            # will be analyze. this way we don't measure ISIs
                            # outside of the stimulus period

                            bisi_temp = np.zeros((num_spk_times-2, 4))
                            for k in range(1, num_spk_times - 1):
                                t_before = spk_times[k] - spk_times[k-1]
                                t_after  = spk_times[k+1] - spk_times[k]
                                bisi_temp[k-1, 0] = t_before
                                bisi_temp[k-1, 1] = t_after
                                bisi_temp[k-1, 2] = stim_id
                                bisi_temp[k-1, 3] = good_trial_ind # identifies spikes from the same trial
                        bisi_np_list[unit] = np.concatenate((bisi_np_list[unit], bisi_temp), axis=0)


                    good_trial_ind += 1

            # remove first row which is junk
            for k, nparray in enumerate(bisi_np_list):
                bisi_np_list[k] = nparray[1::, :]
            bisi_list[stim_ind] = bisi_np_list

        self.bisi_list = bisi_list

    def make_simple_tuning_curve(self, unit_ind=[], kind='abs_count'):
        '''
        make_simple_tuning_curve allows one to specify what type of tuning
        curve to plot as well as which unit for a single tuning curve or all
        units for a subplot of tuning curves

        Kinds of tuning curves:
        Absolute rate:   'abs_rate'
        Absolute counts: 'abs_count'
        Evoked rate:     'evk_rate'
        Evoked counts:   'evk_count'
        '''

        # if the rates have not been calculated do that now
        if hasattr(self, 'abs_rate') is False:
            self.rates()
        # dictionary of types of plots. allows user to specify what type of
        # tuning curve should be plotted.
        kind_dict      = {'abs_rate': 0, 'abs_count': 1, 'evk_rate': 2, 'evk_count': 3}
        kind_of_tuning = [self.abs_rate, self.abs_count, self.evk_rate, self.evk_count]

        control_pos = 9 ######## GET THIS FROM FILE ########### self.neo_obj.annotations['control_pos']
        # setup x-axis with correct labels and range
        pos = range(1,control_pos)
        x_vals = range(1, control_pos+1)
        labels = [str(i) for i in pos]; labels.append('NC')
        line_color = ['k','r','g']

        # determine number of rows and columns in the subplot
        if unit_ind:
            num_rows, num_cols = 1, 1
        else:
            unit_ind = range(self.num_units)
            num_rows, num_cols = 3, 3

        num_manipulations = len(self.stim_ids)/control_pos # no light, light 1 region, light 2 regions

        unit_count, plot_count = 0, 0
        fig = plt.subplots(num_rows, num_cols)
        for unit in unit_ind:
            meanr = [np.mean(k[:, unit]) for k in kind_of_tuning[kind_dict[kind]]]
            stder = [np.std(k[:, unit]) / np.sqrt(k[:, unit].shape[0]) for k in kind_of_tuning[kind_dict[kind]]]
            for control_pos_count, first_pos in enumerate(range(0, len(self.stim_ids), control_pos )):
                # compute the means and standard errors

                ax = plt.subplot(num_rows, num_cols, plot_count+1)
                # plot stimulus positions separately from control so the
                # control position is not connected with the others
                plt.errorbar(pos[0:control_pos-1],\
                        meanr[(control_pos_count*control_pos):((control_pos_count+1)*control_pos-1)],\
                        yerr=stder[(control_pos_count*control_pos):((control_pos_count+1)*control_pos-1)],\
                        fmt=line_color[control_pos_count], marker='o', markersize=8.0, linewidth=2)
                # plot control position separately from stimulus positions
                plt.errorbar(control_pos, meanr[(control_pos_count+1)*control_pos-1], yerr=stder[(control_pos_count+1)*control_pos-1],\
                        fmt=line_color[control_pos_count], marker='o', markersize=8.0, linewidth=2)

            plt.title('shank: ' + self.shank_names[self.shank_ids[unit]] + \
                    ' depth: ' + str(self.neo_obj.segments[0].spiketrains[unit].annotations['depth']))
            plt.plot([0, control_pos+1],[0,0],'--k')
            plt.xlim(0, control_pos+1)
            plt.ylim(plt.ylim()[0]-1, plt.ylim()[1]+1)
            ax.set_xticks([])
            plt.xticks(x_vals, labels)
            plt.xlabel('Bar Position', fontsize=8)
            plt.show()
            # count after each plot is made
            plot_count += 1
            unit_count += 1

            if unit_count == len(unit_ind) and plot_count == (num_rows*num_cols):
                print('finished')
                plt.subplots_adjust(left=0.04, bottom=0.07, right=0.99, top=0.92, wspace=0.17, hspace=0.35)
                plt.suptitle(kind)
                plt.show()
                return ax
            elif unit_count == len(unit_ind):
                print('finished')
                plt.subplots_adjust(left=0.04, bottom=0.07, right=0.99, top=0.92, wspace=0.17, hspace=0.35)
                plt.suptitle(kind)
                plt.show()
            elif plot_count == (num_rows*num_cols):
                print('made a new figure')
                plt.subplots_adjust(left=0.04, bottom=0.07, right=0.99, top=0.92, wspace=0.17, hspace=0.35)
                plt.suptitle(kind)
                plt.show()
                # create a new plot
                if plot_count != len(unit_ind):
                    fig = plt.subplots(num_rows, num_cols)
                plot_count = 0

    def make_raster(self, unit_ind=0, trial_type=0):
        '''
        Makes a raster plot for the given unit index and trial type.
        If called alone it will plot a raster to the current axis. This function
        is called by plot_all_rasters and returns an axis handle to the current
        subplot. This allows plot_all_rasters to plot rasters in the appropriate
        subplots.
        '''
        # if the rates have not been calculated do that now
        if hasattr(self, 'binned_spikes') is False:
            print('Spikes have not been binned! Binning data now.')
            self.rates()

        print('Making raster for unit {} and trial_type {}'.format(unit_ind, trial_type))
        ax = plt.gca()
        count_mat = self.binned_spikes[trial_type][:, unit_ind, :] # returns bins x num_trials array

        for trial in range(count_mat.shape[1]):
            trial_inds = np.where(count_mat[:, trial] > 0)[0]
            spike_times = self._bins[trial_inds]
            plt.vlines(spike_times, trial, trial+1, color='k')
        plt.hlines(trial+1, 0, 1.5, color='k')
        plt.xlim(self._bins[0], self._bins[-1])
        plt.ylim(0, trial+1)

        return ax

    def make_psth(self, unit_ind=0, trial_type=0):
        '''
        Makes a PSTH plot for the given unit index and trial type.
        If called alone it will plot a PSTH to the current axis. This function
        is called by plot_all_PSTHs and returns an axis handle to the current
        subplot. This allows plot_all_PSTHs to plot PSTHs in the appropriate
        subplots.
        '''
        # if the rates have not been calculated do that now
        if hasattr(self, 'psth') is False:
            print('Spikes have not been binned! Binning data now.')
            self.rates()

        print('Making PSTH for unit {} and trial_type {}'.format(unit_ind, trial_type))
        ax = plt.gca()

        psth_temp = self.psth[trial_type][:, unit_ind, :]
        mean_psth = np.mean(psth_temp, axis=1) # mean across all trials
        se = sp.stats.sem(psth_temp, axis=1)
        # inverse of the CDF is the percentile function. ppf is the percent point funciton of t.
        h = se*sp.stats.t.ppf((1+0.95)/2.0, psth_temp.shape[1]-1) # (1+1.95)/2 = 0.975

        plt.plot(self._bins[0:-1], mean_psth, 'k')
        plt.fill_between(self._bins[0:-1], mean_psth - h, mean_psth + h, facecolor='k', alpha=0.3)

        return ax

    def plot_all_rasters(self, unit_ind=0):
        '''
        Plots all rasters for a given unit with subplots.
        Each positions is a row and each manipulation is a column.

        TODO:
        - generalize code to see how many manipulations occurred and set the
          number of subplots occordingly.
        '''
        fig = subplots(9, 1)
        for trial in range(9):
            subplot(9,1,trial+1)
            self.make_raster(unit_ind=unit_ind, trial_type=trial)

    def plot_all_psths(self, unit_ind=0):
        '''
        Plots all PSTHs for a given unit with subplots.
        Each positions is a row and each manipulation is a column.

        TODO:
        - generalize code to see how many manipulations occurred and set the
          number of subplots occordingly.
        '''
        fig = subplots(9, 1)
        for trial in range(9):
            subplot(9,1,trial+1)
            self.make_psth(unit_ind=unit_ind, trial_type=trial)

###############################################################################
######## Doesn't work in Ubuntu...figure out why ##############################
###############################################################################
    def make_raster_movie(self, trial_type=1, run_trials=True):
        # stim_times is the time of the stimulus onset and offset

        stim_inds = self.get_annotations_index(key='trial_type', value=trial_type)
        run_inds  = self.get_annotations_index(key='run_boolean', value=True)
        stim_index = stim_inds and run_inds
        num_frames = len(stim_index)
        shank_borders = np.where(np.diff(self.shank_ids) > 0)[0]

        def __get_spike_xy_coord(segment):
            x = list()
            y = list()
            for unit in range(self.num_units):
                x.extend(segment.spiketrains[unit].tolist())
                y.extend(np.ones(len(segment.spiketrains[unit]), )*unit)
            return np.array(x), np.array(y)

        FFMpegWriter = animation.writers['ffmpeg']
        writer = FFMpegWriter(fps=3)
        fig = plt.figure()

        with writer.saving(fig, self.neo_obj.name + '_allunitraster_trial_'  + str(trial_type) + ".mp4", 70):
            for sind in stim_index:
                x, y = __get_spike_xy_coord(self.neo_obj.segments[sind])
                stim_times = self.neo_obj.segments[sind].annotations['stim_times']
                t_start = self.neo_obj.segments[sind].spiketrains[0].t_start
                t_stop  = self.neo_obj.segments[sind].spiketrains[0].t_stop

                plt.vlines(x, y-1, y, 'k')
                plt.vlines(np.array(stim_times), 0, self.num_units, 'r')
                plt.hlines(shank_borders, t_start, t_stop, 'g')
                plt.ylim(0, self.num_units)
                plt.xlim(t_start, t_stop)
                plt.gca().invert_yaxis()
                plt.xlabel('time (s)')
                plt.ylabel('unit')
                plt.title('all unit raster')
                writer.grab_frame()
                plt.clf()
        plt.close()
#
#def single_trial_all_unit_raster(segments, trial_type, stim_times, stim_ids, run_trials=True):
#    '''
#    Plot single trial rasters with all recorded units
#
#    segments: neo segments object.
#    trial_typ: trial type to analyze. int
#    stim_times: list containing onset and offset of the stimulus period (s). float
#    run_trials: boolean. whether to analyze only running trials.
#
#    OUTPUT
#    multiple raster plots
#
#
#
#    '''
#
#    num_units = len(segments[0].spiketrains)
#    stim_index = get_stimulus_index(segments, trial_type, run_trials=True)
#
#    for k, sind in enumerate(stim_index):
#
#        print('on iteration ' + str(k))
#        fig = plt.figure()
#        for unit in range(num_units):
#            plt.vlines(segments[sind].spiketrains[unit], unit, unit+1, 'k')
#        plt.vlines(np.array(stim_times)*1000, 0, plt.ylim()[1], 'r')
#        plt.show()
#
#


########## MAIN CODE ##########
########## MAIN CODE ##########

#data_dir = '/Users/Greg/Documents/AdesnikLab/Data/'
data_dir = '/media/greg/Data/Neuro/neo/'
manager = NeoHdf5IO(os.path.join(data_dir + 'FID1293_neo_object.h5'))
print('Loading...')
block = manager.read()
print('...Loading Complete!')
manager.close()
exp1 = block[0]
neuro = NeuroAnalyzer(exp1)
neuro.make_simple_tuning_curve()



    #how to get spike times: block.segments[0].spiketrains[0].tolist()


