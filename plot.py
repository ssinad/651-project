import os
from matplotlib import pyplot as plt
import numpy as np

filename = 'Value.npy'

if os.path.exists(filename):
    data = np.load(filename)
    lmda = 0.90
    # d = np.mean(data,axis=0)

    plt.plot(np.arange(0, data.shape[0]), data, label='Sarsa_lambda={}'.format(lmda))
    # plt.ylim([100,500])
    # plt.xlabel('Episode')
    # plt.ylabel('Steps per episode \naveraged over {} runs'.format(data.shape[0]))
    # plt.legend()
    plt.show()
