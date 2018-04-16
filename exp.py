#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
  Last Modified by: Mohammad M. Ajallooeian, Sina Ghiassian
  Last Modified on: 21/11/2017

"""
import numpy as np
# from env import number_of_states

from rl_glue import *  # Required for RL-Glue
import agent
RLGlue("env", "agent")

def true_value_function(gamma, p, number_of_states):
    tmp = (1 - p * gamma) / gamma / (1 - p)
    return tmp ** (np.arange(number_of_states) - number_of_states) / gamma

if __name__ == "__main__":
    num_episodes = 200
    num_runs = 5
    N = 10
    policies = [0, 0.8]  # lower is better
    dps = ["", "inp_per", "out_per"]
    # epsilons = [1.5, 1, 0.5]
    epsilons = list(np.logspace(np.log10(0.01), np.log10(1), num=25))
    utility = np.zeros((len(epsilons), 2))


    for epsilon in epsilons:
        for p in policies:
            for dp in dps:
                runs = np.zeros((num_runs, N))
                steps = np.zeros([num_runs, num_episodes])
                # RL_env_message("dp:" + str(dp))
                for r in range(num_runs):
                    # print("run number : ", r + 1)
                    RL_init()
                    if dp == "inp_per":
                        RL_env_message("dp")
                    elif dp == "out_per":
                        RL_agent_message("dp")
                    RL_env_message("epsilon:" + str(epsilon))
                    RL_agent_message("epsilon:" + str(epsilon))
                    RL_env_message("N:" + str(N))
                    RL_agent_message("N:" + str(N))
                    RL_agent_message("p:" + str(p))
                    RL_env_message("p:" + str(p))
                    for e in range(num_episodes):
                        # print '\tepisode {}'.format(e+1)
                        RL_episode(0)
                        steps[r, e] = RL_num_steps()
                    runs[r, :] = RL_agent_message("ValueFunction")
                name = 'Value-'+str(p)+'-'+str(dp)
                avg = np.average(runs, axis=0)
                # if dp == "":
                    # print("Real difference is", abs(avg - true_value_function(agent.gamma, p, N)))
                np.save(name, avg)

        files = ['Value-0.1-.npy', 'Value-0.1-inp_per.npy', 'Value-0.1-out_per.npy', 'Value-0.8-.npy',
                 'Value-0.8-inp_per.npy', 'Value-0.8-out_per.npy']
        idpv1 = np.load(files[1])
        idpv2 = np.load(files[4])

        odpv1 = np.load(files[2])
        odpv2 = np.load(files[5])

        ndpv1 = np.load(files[0])
        ndpv2 = np.load(files[3])

        ix = abs(np.linalg.norm(idpv1-idpv2)-np.linalg.norm(ndpv1 - ndpv2))
        utility[epsilons.index(epsilon), 0] = ix
        print("Input perturbation error", ix)

        ox = abs(np.linalg.norm(odpv1 - odpv2) - np.linalg.norm(ndpv1 - ndpv2))
        utility[epsilons.index(epsilon), 1] = ox
        print("Output perturbation error", ox)

        np.save('idp', abs(idpv1-idpv2))
        np.save('odp', abs(odpv1 - odpv2))
        np.save('non_dp', abs(ndpv1 - ndpv2))
    np.save('util', utility)
