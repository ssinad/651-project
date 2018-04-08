import os
from matplotlib import pyplot as plt
import numpy as np

# filename = 'Value.npy'
files = ['dp.npy', 'non_dp.npy']
for filename in files:
    if os.path.exists(filename):
        data = np.load(filename)
        lmda = 0.90
        # d = np.mean(data,axis=0)
        plt.xticks(np.arange(len(data)))
        plt.plot(np.arange(0, data.shape[0]), data, label=filename)
        plt.xlabel('States')
        plt.ylabel('Difference in Policy Value Estimates')
        # plt.ylim([100,500])
        # plt.xlabel('Episode')
        # plt.ylabel('Steps per episode \naveraged over {} runs'.format(data.shape[0]))
        plt.legend()
plt.show()
