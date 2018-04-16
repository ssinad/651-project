import os
from matplotlib import pyplot as plt
import numpy as np

# filename = 'Value.npy'
files = ['idp.npy', 'non_dp.npy', 'odp.npy']
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
in_util, out_util = util.T
# plt.xticks(np.arange(out_util.shape[0]), ('{:.2f}'.format(x) for x in np.logspace(np.log10(0.5), np.log10(1.5),
#                                                                                   num=out_util.shape[0])))
locs = np.logspace(np.log10(0.01), np.log10(1.0), num=out_util.shape[0])
plt.xticks(locs, ("{:.2f}".format(x) for x in locs))
# ('0.5', '1', '1.5'))
plt.semilogx(locs, in_util, label='Input perturbation')
plt.semilogx(locs, out_util, label='Output perturbation')
plt.xlabel('$\epsilon$')
plt.ylabel('utility measure')
# plt.ylim([100,500])
# plt.xlabel('episode')
# plt.ylabel('steps per episode \naveraged over {} runs'.format(data.shape[0]))
plt.legend()
plt.show()
