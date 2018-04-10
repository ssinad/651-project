import os
from matplotlib import pyplot as plt
import numpy as np

# filename = 'Value.npy'
files = ['dp.npy', 'non_dp.npy']
for filename in files:
    if os.path.exists(filename):
        data = np.load(filename)
        # lmda = 0.90
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
util = np.load('util.npy')
plt.xticks(np.arange(util.shape[0]), ('1.5', '1', '0.5'))
plt.plot(np.arange(0, util.shape[0]), util, label='util.npy')
plt.xlabel('epsilons')
plt.ylabel('utility measure')
# plt.ylim([100,500])
# plt.xlabel('episode')
# plt.ylabel('steps per episode \naveraged over {} runs'.format(data.shape[0]))
plt.legend()
plt.show()
