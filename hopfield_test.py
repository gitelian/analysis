import numpy as np
import matplotlib.pyplot as plt
from hdnet.stimulus import Stimulus
from hdnet.spikes import Spikes
from hdnet.spikes_model import SpikeModel, BernoulliHomogeneous, DichotomizedGaussian

# Let's first make up some simuilated spikes: 2 trials
spikes = (np.random.random((2, 10, 200)) < .05).astype(int)
spikes[0, [1, 5], ::5] = 1  # insert correlations
spikes[1, [2, 3, 6], ::11] = 1  # insert correlations

spikes = Spikes(spikes=spikes)

# let's look at the raw spikes and their covariance
plt.matshow(spikes.rasterize(), cmap='gray')
plt.title('Raw spikes')
plt.show()

plt.matshow(spikes.covariance().reshape((2 * 10, 10)), cmap='gray')
plt.title('Raw spikes covariance')
plt.show()

# let's examine the structure in spikes using a spike modeler
spikes_model = BernoulliHomogeneous(spikes=spikes)
BH_sample_spikes = spikes_model.sample_from_model()

plt.matshow(BH_sample_spikes.rasterize(), cmap='gray')
plt.title('BernoulliHomogeneous sample')
print "%1.4f means" % BH_sample_spikes.spikes.mean()

plt.matshow(BH_sample_spikes.covariance().reshape((2 * 10, 10)), cmap='gray')
plt.title('BernoulliHomogeneous covariance')
plt.show()

# let's model them as DichotomizedGaussian:
# from the paper: Generating spike-trains with specified correlations, Macke et
# al.
spikes_model = DichotomizedGaussian(spikes=spikes)
DG_sample_spikes = spikes_model.sample_from_model()

plt.title('DichotomizedGaussian sample')
plt.matshow(DG_sample_spikes.rasterize(), cmap='gray')
plt.show()

plt.matshow(DG_sample_spikes.covariance().reshape((2 * 10, 10)), cmap='gray')
plt.title('DichotomizedGaussian covariance')
plt.show()


# the basic modeler trains a Hopfield network using MPF on the raw spikes
spikes = (np.random.random((2, 10, 200)) < .05).astype(int)
spikes[0, [1, 5], ::5] = 1  # insert correlations
spikes[1, [2, 3, 6], ::11] = 1  # insert correlations
spikes = Spikes(spikes=spikes)

plt.matshow(spikes.rasterize(), cmap='gray')
plt.title('Raw spikes')
plt.show()

spikes_model = SpikeModel(spikes=spikes)
spikes_model.fit()  # note: this fits a single network to all trials
spikes_model.chomp()

# What is this supposed to do???
#converged_spikes = Spikes(spikes=spikes_model.hopfield_spikes)

plt.matshow(spikes_model.hopfield_spikes.rasterize(), cmap='gray')
plt.title('Converge dynamics on Raw data')
plt.show()

plt.matshow(converged_spikes.covariance().reshape((2 * 10, 10)), cmap='gray')
plt.title('Covariance of converged memories')
plt.show()
