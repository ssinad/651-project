#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
  Last Modified by: Mohammad M. Ajallooeian, Sina Ghiassian
  Last Modified on: 21/11/2017

"""
import numpy as np
from env import number_of_states
from rl_glue import *  # Required for RL-Glue
RLGlue("env", "agent")

if __name__ == "__main__":
    num_episodes = 200
    num_runs = 5
    N = number_of_states()
    policies = [0.1,0.8]
    dps = [0, 1]
    for p in policies:
        for dp in dps:
            runs = np.zeros((num_runs, N))
            steps = np.zeros([num_runs, num_episodes])
            RL_agent_message("p:"+str(p))
            RL_env_message("dp:" + str(dp))
            for r in range(num_runs):
                print("run number : ", r + 1)
                RL_init()
                for e in range(num_episodes):
                    # print '\tepisode {}'.format(e+1)
                    RL_episode(0)
                    steps[r, e] = RL_num_steps()
                runs[r, :] = RL_agent_message("ValueFunction")
            name = 'Value,'+str(p)+','+str(dp)
            np.save(name, np.average(runs, axis=0))
    files = ['Value,0.1,0.npy', 'Value,0.1,1.npy', 'Value,0.8,0.npy', 'Value,0.8,1.npy']
    dpv1 = np.load(files[1])
    dpv2 = np.load(files[3])
    # v1_diff = np.zeros(N)
    v1_diff = np.linalg.norm(dpv1-dpv2)
    ndpv1 = np.load(files[0])
    ndpv2 = np.load(files[2])
    # v2_diff = np.zeros(N)
    v2_diff = np.linalg.norm(ndpv1 - ndpv2)
    x = v1_diff-v2_diff
    print(x)
    np.save('dp',abs(dpv1-dpv2))
    np.save('non_dp', abs(ndpv1 - ndpv2))