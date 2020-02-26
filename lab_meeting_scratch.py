













neuro.reclassify_run_trials(mean_thresh=150, sigma_thresh=200, low_thresh=100)


## Running trials ##
neuro.rates(engaged=True)
fig, ax_temp = neuro.plot_mean_whisker(t_window=[-1.5, 1.5], cond2plot=[0,1], all_correct_trials=False)
ax_temp[0].set_ylim(80, 135)
ax_temp[0].set_title('Running\nAll choices')

fig, ax_temp = neuro.plot_mean_whisker(t_window=[-1.5, 1.5], cond2plot=[0,1], all_correct_trials=True)
ax_temp[0].set_ylim(80, 135)
ax_temp[0].set_title('Running\nCorrect choices')


## Slow running trials ##
neuro.rates(engaged=False)
fig, ax_temp = neuro.plot_mean_whisker(t_window=[-1.5, 1.5], cond2plot=[0,1], all_correct_trials=False)
ax_temp[0].set_ylim(80, 135)
ax_temp[0].set_title('Walking\nAll choices')

fig, ax_temp = neuro.plot_mean_whisker(t_window=[-1.5, 1.5], cond2plot=[0,1], all_correct_trials=True)
ax_temp[0].set_ylim(80, 135)
ax_temp[0].set_title('Walking\nCorrect choices')



































