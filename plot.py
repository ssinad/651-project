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
plt.xticks(np.arange(in_util.shape[0]), np.logspace(np.log10(0.01), np.log10(1), num=3))
#     (np.arange(in_util.shape[0]),('0.5', '1', '1.5'))
# plt.locator_params(axis='x', nticks=6)


plt.plot(np.arange(0, in_util.shape[0]), in_util, label='in_util.npy')
plt.plot(np.arange(0, out_util.shape[0]), out_util, label='out_util.npy')
plt.xlabel('epsilons')
plt.ylabel('utility measure')
# plt.ylim([100,500])
# plt.xlabel('episode')
# plt.ylabel('steps per episode \naveraged over {} runs'.format(data.shape[0]))
plt.legend()
plt.show()
in_dp_value = np.load("Value-0.8-inp_per.npy")
out_dp_value = np.load("Value-0.8-out_per.npy")
ndp_value = np.load("Value-0.8-.npy")
plt.plot(np.arange(0,in_dp_value.shape[0]), in_dp_value, label='input_per')
plt.plot(np.arange(0, out_dp_value.shape[0]), out_dp_value, label='output_per')
plt.plot(np.arange(0, ndp_value.shape[0]), ndp_value, label='non_dp')
plt.xlabel('States')
plt.ylabel('State Value Estimates')
# plt.ylim([100,500])
# plt.xlabel('Episode')
# plt.ylabel('Steps per episode \naveraged over {} runs'.format(data.shape[0]))
plt.legend()
plt.show()
plt.plot(np.arange(0, in_dp_value.shape[0]), abs(ndp_value-in_dp_value), label='input_per')
plt.plot(np.arange(0, out_dp_value.shape[0]), abs(ndp_value-out_dp_value), label='output_per')
plt.xlabel('States')
plt.ylabel('State Value Difference Estimates')
plt.legend()
plt.show()
